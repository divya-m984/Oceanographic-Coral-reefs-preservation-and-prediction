"""
src/models/rollback.py — Explicit champion rollback for CoralSense.

ROLLBACK RULES
--------------
1. --approve flag is mandatory.
2. --approver (non-empty string) is mandatory.
3. --reason is mandatory.
4. The target version must exist and be loadable (version exists check).
5. Dry-run mode is supported: checks are performed, alias is NOT moved.
6. Model versions are NEVER deleted.
7. A rollback receipt is written with full audit metadata.

USAGE
-----
  # Dry-run to verify the target version exists
  python -m src.models.rollback \\
    --model coralsense_reef_health \\
    --version 1 \\
    --approve \\
    --approver "Divya" \\
    --reason "Rollback verification — challenger underperforms in prod" \\
    --dry-run

  # Actual rollback
  python -m src.models.rollback \\
    --model coralsense_reef_health \\
    --version 1 \\
    --approve \\
    --approver "Divya" \\
    --reason "Rolling back after monitoring alert"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.config import Config, get_config, setup_logging
from src.models.registry import get_champion_version

logger = logging.getLogger(__name__)

_SYNTHETIC_DISCLAIMER = (
    "Metrics reflect performance on synthetic data only. "
    "Do not use to infer real-world conservation accuracy."
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class RollbackReceipt:
    """Full audit record for one rollback action."""

    receipt_id: str
    timestamp: str
    model_name: str
    task: str
    previous_champion_version: str
    rollback_to_version: str
    approver: str
    reason: str
    target_version_verified: bool
    dry_run: bool
    alias_set: bool
    synthetic_data_disclaimer: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _verify_version_exists(model_name: str, version: str, mlflow_uri: str) -> dict:
    """Verify a model version exists and return its metadata. Raises if not found."""
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=mlflow_uri)
    mv = client.get_model_version(name=model_name, version=version)
    raw_tags = mv.tags or {}
    tags = raw_tags if isinstance(raw_tags, dict) else {t.key: t.value for t in raw_tags}
    return {
        "version": str(mv.version),
        "run_id": mv.run_id,
        "algo_name": tags.get("algo_name", "unknown"),
        "data_source": tags.get("data_source", tags.get("synthetic_data_disclaimer", "")),
        "tags": tags,
    }


# ---------------------------------------------------------------------------
# Core rollback logic
# ---------------------------------------------------------------------------


def rollback_to_version(
    model_name: str,
    version: str,
    approver: str,
    reason: str,
    mlflow_uri: str | None = None,
    cfg: Config | None = None,
    dry_run: bool = False,
    output_dir: Path | None = None,
) -> RollbackReceipt:
    """
    Roll the champion alias back to *version*.

    Parameters
    ----------
    model_name:
        Registered model name.
    version:
        Version number to roll back to.
    approver:
        Name or identifier of the approving person.
    reason:
        Documented rollback reason.
    mlflow_uri:
        SQLite tracking URI.
    cfg:
        Config instance.
    dry_run:
        Verify but do NOT move the champion alias.
    output_dir:
        Directory for receipt JSON.

    Returns
    -------
    RollbackReceipt with full audit record.

    Raises
    ------
    ValueError: approver or reason missing.
    RuntimeError: target version does not exist.
    """
    if not approver or not approver.strip():
        raise ValueError("--approver must be a non-empty identifier.")
    if not reason or not reason.strip():
        raise ValueError("--reason must be a non-empty string.")

    cfg = cfg or get_config()
    mlflow_uri = mlflow_uri or cfg.mlflow_tracking_uri
    output_dir = output_dir or cfg.paths.reports_dir

    task = _task_for_model(model_name, cfg)
    receipt_id = f"rollback_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"

    # ── 1. Record current champion ───────────────────────────────────────────
    try:
        prev_info = get_champion_version(task, mlflow_uri=mlflow_uri, cfg=cfg)
        previous_champion_version = prev_info["version"]
    except Exception:
        previous_champion_version = "unknown"

    # ── 2. Verify target version exists ──────────────────────────────────────
    try:
        version_info = _verify_version_exists(model_name, version, mlflow_uri)
        target_verified = True
        logger.info(
            "[%s] Target version %s verified: algo=%s",
            task,
            version,
            version_info["algo_name"],
        )
    except Exception as exc:
        raise RuntimeError(
            f"Target version {version} of '{model_name}' does not exist or cannot be loaded: {exc}"
        ) from exc

    # ── 3. Set champion alias (unless dry-run) ───────────────────────────────
    alias_set = False
    if not dry_run:
        if version == previous_champion_version:
            logger.warning(
                "[%s] Version %s is already the champion — no change needed.",
                task,
                version,
            )
        else:
            from mlflow import MlflowClient

            client = MlflowClient(tracking_uri=mlflow_uri)
            client.set_registered_model_alias(
                name=model_name,
                alias=cfg.mlflow_champion_alias,
                version=version,
            )
            alias_set = True
            logger.info(
                "[%s] Champion alias '%s' rolled back to version %s (was %s).",
                task,
                cfg.mlflow_champion_alias,
                version,
                previous_champion_version,
            )
    else:
        logger.info(
            "[%s] DRY-RUN: would roll champion alias '%s' → v%s (current: v%s).",
            task,
            cfg.mlflow_champion_alias,
            version,
            previous_champion_version,
        )

    # ── 4. Write receipt ─────────────────────────────────────────────────────
    tags = version_info.get("tags", {})
    data_source = tags.get("data_source", "")
    disclaimer = _SYNTHETIC_DISCLAIMER if "synthetic" in data_source else ""

    receipt = RollbackReceipt(
        receipt_id=receipt_id,
        timestamp=datetime.now(UTC).isoformat(),
        model_name=model_name,
        task=task,
        previous_champion_version=previous_champion_version,
        rollback_to_version=version,
        approver=approver.strip(),
        reason=reason.strip(),
        target_version_verified=target_verified,
        dry_run=dry_run,
        alias_set=alias_set,
        synthetic_data_disclaimer=disclaimer,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / f"{receipt_id}.json"
    with receipt_path.open("w", encoding="utf-8") as fh:
        json.dump(asdict(receipt), fh, indent=2)
    logger.info("Rollback receipt saved to %s", receipt_path)

    return receipt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point for champion rollback.

    Exit codes
    ----------
    0   Success.
    1   Runtime error.
    2   Invalid arguments.
    """
    setup_logging(__name__)
    parser = argparse.ArgumentParser(
        description="Roll back the CoralSense champion alias to a previous version.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        metavar="MODEL_NAME",
        help="Registered model name.",
    )
    parser.add_argument(
        "--version",
        required=True,
        metavar="VERSION",
        help="Version number to roll back to.",
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
        help="Documented reason for rollback.",
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
        help="Verify target version only — do not move champion alias.",
    )
    args = parser.parse_args(argv)

    if not args.approve:
        parser.error("--approve is required to confirm the rollback decision.")

    try:
        cfg = get_config()
        receipt = rollback_to_version(
            model_name=args.model,
            version=args.version,
            approver=args.approver,
            reason=args.reason,
            mlflow_uri=args.mlflow_uri,
            cfg=cfg,
            dry_run=args.dry_run,
            output_dir=args.output_dir,
        )
        action = "DRY-RUN: would roll back" if receipt.dry_run else "Rolled back"
        print(
            f"{action}: {receipt.model_name} "
            f"v{receipt.previous_champion_version} → v{receipt.rollback_to_version}"
        )
        print(f"  Approver : {receipt.approver}")
        print(f"  Alias set: {receipt.alias_set}")
        print(f"  Receipt  : {receipt.receipt_id}")
        return 0
    except Exception as exc:
        logger.error("Rollback failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
