"""
src/models/promote.py — Explicit challenger promotion for CoralSense.

PROMOTION RULES
---------------
1. Promotion NEVER occurs as a side effect of retraining.
2. --approve flag is mandatory.
3. --approver (non-empty string) is mandatory.
4. --reason is mandatory.
5. A comparison report path is required; outcome must be
   "eligible_for_promotion" unless overridden with --force.
6. Quality gates are re-validated immediately before promotion.
7. The previous champion version is recorded.
8. A promotion receipt is written with full audit metadata.
9. No unauthenticated API endpoint is provided for promotion.

USAGE
-----
  python -m src.models.promote \\
    --model coralsense_reef_health \\
    --version 5 \\
    --comparison-report reports/compare_health.json \\
    --approve \\
    --approver "Divya" \\
    --reason "Reviewed challenger quality gates — July 2026"

  python -m src.models.promote \\
    --model coralsense_reef_health \\
    --version 5 \\
    --comparison-report reports/compare_health.json \\
    --approve \\
    --approver "Divya" \\
    --reason "Post-review approval" \\
    --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import Config, get_config, setup_logging
from src.models.registry import check_quality_gates, get_champion_version

logger = logging.getLogger(__name__)

_VALID_MODELS = {
    "coralsense_reef_health": "health",
    "coralsense_restoration_suitability": "restoration",
}

_SYNTHETIC_DISCLAIMER = (
    "Metrics reflect performance on synthetic data only. "
    "Do not use to infer real-world conservation accuracy."
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class PromotionReceipt:
    """Full audit record for one promotion action."""

    receipt_id: str
    timestamp: str
    model_name: str
    task: str
    previous_champion_version: str
    new_champion_version: str
    approver: str
    reason: str
    comparison_report_path: str
    comparison_report_sha256: str
    comparison_outcome: str
    gate_revalidation_passed: bool
    gate_failures: list[str]
    dry_run: bool
    forced: bool
    alias_set: bool
    synthetic_data_disclaimer: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_comparison_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Comparison report not found: {path}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _get_version_tags(model_name: str, version: str, mlflow_uri: str) -> dict[str, str]:
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=mlflow_uri)
    mv = client.get_model_version(name=model_name, version=version)
    raw_tags = mv.tags or {}
    return raw_tags if isinstance(raw_tags, dict) else {t.key: t.value for t in raw_tags}


def _task_for_model(model_name: str, cfg: Config) -> str:
    if model_name == cfg.mlflow_registered_health:
        return "health"
    if model_name == cfg.mlflow_registered_restoration:
        return "restoration"
    raise ValueError(
        f"Unknown model name '{model_name}'. "
        f"Expected one of: {cfg.mlflow_registered_health!r}, "
        f"{cfg.mlflow_registered_restoration!r}"
    )


# ---------------------------------------------------------------------------
# Core promotion logic
# ---------------------------------------------------------------------------


def promote_challenger(
    model_name: str,
    version: str,
    comparison_report_path: Path,
    approver: str,
    reason: str,
    mlflow_uri: str | None = None,
    cfg: Config | None = None,
    dry_run: bool = False,
    force: bool = False,
    output_dir: Path | None = None,
) -> PromotionReceipt:
    """
    Explicitly promote a challenger version to the champion alias.

    Parameters
    ----------
    model_name:
        Registered model name (e.g. "coralsense_reef_health").
    version:
        Version number string of the challenger to promote.
    comparison_report_path:
        Path to the JSON comparison report produced by compare.py.
    approver:
        Name or identifier of the approving person.
    reason:
        Human-readable promotion reason.
    mlflow_uri:
        SQLite tracking URI (defaults to config).
    cfg:
        Config instance.
    dry_run:
        Check everything but do NOT move the champion alias.
    force:
        Allow promotion even if comparison outcome is "review_required"
        (NOT "reject" — reject always blocks).
    output_dir:
        Directory for receipt JSON.

    Returns
    -------
    PromotionReceipt with full audit record.

    Raises
    ------
    ValueError: approver or reason missing, or outcome blocks promotion.
    RuntimeError: quality gate fails at revalidation.
    PermissionError: comparison outcome is "reject".
    """
    if not approver or not approver.strip():
        raise ValueError("--approver must be a non-empty identifier.")
    if not reason or not reason.strip():
        raise ValueError("--reason must be a non-empty string.")

    cfg = cfg or get_config()
    mlflow_uri = mlflow_uri or cfg.mlflow_tracking_uri
    output_dir = output_dir or cfg.paths.reports_dir

    task = _task_for_model(model_name, cfg)
    receipt_id = f"promote_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"

    # ── 1. Load comparison report ────────────────────────────────────────────
    report_sha256 = _hash_file(comparison_report_path)
    report = _load_comparison_report(comparison_report_path)
    comparison_outcome: str = report.get("outcome", "unknown")

    if comparison_outcome == "reject":
        raise PermissionError(
            f"Cannot promote: comparison outcome is 'reject'. "
            f"Failures: {report.get('failures', [])}"
        )
    if comparison_outcome == "review_required" and not force:
        raise PermissionError(
            "Cannot promote: comparison outcome is 'review_required'. "
            "Pass --force to override after manual review, or re-run comparison."
        )

    # ── 2. Record previous champion ──────────────────────────────────────────
    try:
        prev_info = get_champion_version(task, mlflow_uri=mlflow_uri, cfg=cfg)
        previous_champion_version = prev_info["version"]
    except Exception:
        previous_champion_version = "none"

    # ── 3. Re-validate quality gates ─────────────────────────────────────────
    tags = _get_version_tags(model_name, version, mlflow_uri)
    cv_f1 = float(tags.get("cv_macro_f1", 0.0))
    cv_bal = float(tags.get("cv_balanced_accuracy", 0.0))
    gate = check_quality_gates(task, cv_f1, cv_bal, cfg)

    if not gate.passed:
        raise RuntimeError(
            f"Quality gate re-validation FAILED for '{model_name}' v{version}: "
            f"{gate.failures}. Cannot promote."
        )
    logger.info(
        "[%s] Quality gate re-validation PASSED: cv_f1=%.4f cv_bal=%.4f",
        task,
        cv_f1,
        cv_bal,
    )

    # ── 4. Set champion alias (unless dry-run) ───────────────────────────────
    alias_set = False
    if not dry_run:
        from mlflow import MlflowClient

        client = MlflowClient(tracking_uri=mlflow_uri)
        client.set_registered_model_alias(
            name=model_name,
            alias=cfg.mlflow_champion_alias,
            version=version,
        )
        alias_set = True
        logger.info(
            "[%s] Champion alias '%s' set to version %s (was %s).",
            task,
            cfg.mlflow_champion_alias,
            version,
            previous_champion_version,
        )
    else:
        logger.info(
            "[%s] DRY-RUN: would set alias '%s' → v%s (current champion: v%s).",
            task,
            cfg.mlflow_champion_alias,
            version,
            previous_champion_version,
        )

    # ── 5. Write receipt ─────────────────────────────────────────────────────
    data_source = tags.get("data_source", "unknown")
    disclaimer = _SYNTHETIC_DISCLAIMER if data_source == "synthetic" else ""

    receipt = PromotionReceipt(
        receipt_id=receipt_id,
        timestamp=datetime.now(UTC).isoformat(),
        model_name=model_name,
        task=task,
        previous_champion_version=previous_champion_version,
        new_champion_version=version if not dry_run else "N/A (dry-run)",
        approver=approver.strip(),
        reason=reason.strip(),
        comparison_report_path=str(comparison_report_path),
        comparison_report_sha256=report_sha256,
        comparison_outcome=comparison_outcome,
        gate_revalidation_passed=gate.passed,
        gate_failures=gate.failures,
        dry_run=dry_run,
        forced=force,
        alias_set=alias_set,
        synthetic_data_disclaimer=disclaimer,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / f"{receipt_id}.json"
    with receipt_path.open("w", encoding="utf-8") as fh:
        json.dump(asdict(receipt), fh, indent=2)
    logger.info("Promotion receipt saved to %s", receipt_path)

    return receipt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point for explicit champion promotion.

    Exit codes
    ----------
    0   Success.
    1   Runtime error.
    2   Invalid arguments.
    3   Permission denied (comparison outcome blocks promotion).
    """
    setup_logging(__name__)
    parser = argparse.ArgumentParser(
        description="Explicitly promote a CoralSense challenger to champion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        metavar="MODEL_NAME",
        help="Registered model name (e.g. coralsense_reef_health).",
    )
    parser.add_argument(
        "--version",
        required=True,
        metavar="VERSION",
        help="Model version to promote to champion.",
    )
    parser.add_argument(
        "--comparison-report",
        required=True,
        type=Path,
        dest="comparison_report",
        metavar="JSON",
        help="Path to comparison report from compare.py.",
    )
    parser.add_argument(
        "--approve",
        action="store_true",
        default=False,
        help="Explicit approval flag (required).",
    )
    parser.add_argument(
        "--approver",
        required=True,
        metavar="NAME",
        help="Name or identifier of the approving person.",
    )
    parser.add_argument(
        "--reason",
        required=True,
        metavar="TEXT",
        help="Documented reason for promotion.",
    )
    parser.add_argument(
        "--mlflow-uri",
        default=None,
        dest="mlflow_uri",
        metavar="URI",
        help="MLflow tracking URI (default: from config).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        dest="output_dir",
        metavar="DIR",
        help="Directory for receipt JSON (default: reports/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        dest="dry_run",
        help="Check gates and report only — do not set champion alias.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Allow promotion when outcome is 'review_required' (not 'reject').",
    )
    args = parser.parse_args(argv)

    if not args.approve:
        parser.error("--approve is required to confirm the promotion decision.")

    try:
        cfg = get_config()
        receipt = promote_challenger(
            model_name=args.model,
            version=args.version,
            comparison_report_path=args.comparison_report,
            approver=args.approver,
            reason=args.reason,
            mlflow_uri=args.mlflow_uri,
            cfg=cfg,
            dry_run=args.dry_run,
            force=args.force,
            output_dir=args.output_dir,
        )
        action = "DRY-RUN: would promote" if receipt.dry_run else "Promoted"
        print(f"{action}: {receipt.model_name} v{receipt.new_champion_version}")
        print(f"  Previous champion: v{receipt.previous_champion_version}")
        print(f"  Approver: {receipt.approver}")
        print(f"  Comparison outcome: {receipt.comparison_outcome}")
        print(f"  Gate revalidation: {'PASS' if receipt.gate_revalidation_passed else 'FAIL'}")
        print(f"  Receipt: {receipt.receipt_id}")
        return 0
    except PermissionError as exc:
        logger.error("Promotion blocked: %s", exc)
        return 3
    except Exception as exc:
        logger.error("Promotion failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
