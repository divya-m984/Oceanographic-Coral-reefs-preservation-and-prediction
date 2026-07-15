"""
src/models/compare.py — Champion–challenger comparison for CoralSense.

Comparison rules (configurable via params.yaml retraining.comparison)
----------------------------------------------------------------------
- min_abs_macro_f1            : challenger must meet minimum absolute threshold
- max_macro_f1_regression     : challenger CV macro-F1 may not fall more than N
                                 points below champion's registered CV macro-F1
- min_balanced_accuracy       : challenger must meet minimum absolute threshold
- max_per_class_recall_regression : no single class recall may regress by more
                                    than N points vs champion (requires holdout
                                    per-class metrics from challenger artifact)

Outcomes
--------
  reject              — challenger fails one or more hard gates
  review_required     — challenger is within tolerances but no clear improvement;
                        human review recommended before promotion
  eligible_for_promotion — challenger passes all gates; eligible for promotion
                           (does NOT promote automatically)

The champion alias is NEVER changed by this module.

USAGE
-----
  python -m src.models.compare \\
    --task health \\
    --challenger-version VERSION

  python -m src.models.compare \\
    --task restoration \\
    --challenger-run-id RUN_ID \\
    --output reports/compare_restoration.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import Config, get_config, setup_logging
from src.models.registry import get_champion_version

logger = logging.getLogger(__name__)

_VALID_TASKS = ("health", "restoration")
_OUTCOME_REJECT = "reject"
_OUTCOME_REVIEW = "review_required"
_OUTCOME_ELIGIBLE = "eligible_for_promotion"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class MetricSnapshot:
    """CV and holdout metrics for one model version."""

    source: str  # "champion" or "challenger"
    model_name: str
    version: str
    run_id: str
    algo_name: str
    cv_macro_f1: float
    cv_balanced_accuracy: float
    holdout_macro_f1: float | None = None
    holdout_balanced_accuracy: float | None = None
    holdout_per_class_recall: dict[str, float] = field(default_factory=dict)


@dataclass
class ComparisonReport:
    """Full champion-challenger comparison report."""

    report_id: str
    timestamp: str
    task: str
    outcome: str  # reject | review_required | eligible_for_promotion
    champion: dict[str, Any]
    challenger: dict[str, Any]
    gate_results: list[dict[str, Any]]
    failures: list[str]
    warnings: list[str]
    comparison_rules: dict[str, Any]
    champion_alias_changed: bool = False  # always False


# ---------------------------------------------------------------------------
# Metric loaders
# ---------------------------------------------------------------------------


def _load_champion_metrics(task: str, mlflow_uri: str, cfg: Config) -> MetricSnapshot:
    """Load current champion metrics from registered model tags."""
    info = get_champion_version(task, mlflow_uri=mlflow_uri, cfg=cfg)
    tags = info.get("tags", {})
    return MetricSnapshot(
        source="champion",
        model_name=info["registered_model_name"],
        version=info["version"],
        run_id=info["run_id"],
        algo_name=info["algo_name"],
        cv_macro_f1=float(tags.get("cv_macro_f1", info.get("cv_macro_f1", 0.0))),
        cv_balanced_accuracy=float(tags.get("cv_balanced_accuracy", 0.0)),
        holdout_macro_f1=None,
        holdout_balanced_accuracy=None,
        holdout_per_class_recall={},
    )


def _load_challenger_by_version(
    task: str,
    version: str,
    mlflow_uri: str,
    cfg: Config,
) -> MetricSnapshot:
    """Load challenger metrics from registered model version tags."""
    from mlflow import MlflowClient

    model_name = (
        cfg.mlflow_registered_health if task == "health" else cfg.mlflow_registered_restoration
    )
    client = MlflowClient(tracking_uri=mlflow_uri)
    mv = client.get_model_version(name=model_name, version=version)
    raw_tags = mv.tags or {}
    tags = raw_tags if isinstance(raw_tags, dict) else {t.key: t.value for t in raw_tags}

    # Try loading holdout per-class recall from MLflow artifact
    per_class_recall: dict[str, float] = {}
    run_id = mv.run_id or tags.get("challenger_run_id", "")
    if run_id:
        per_class_recall = _load_per_class_recall_from_run(run_id, mlflow_uri)

    return MetricSnapshot(
        source="challenger",
        model_name=model_name,
        version=version,
        run_id=run_id,
        algo_name=tags.get("algo_name", "unknown"),
        cv_macro_f1=float(tags.get("cv_macro_f1", 0.0)),
        cv_balanced_accuracy=float(tags.get("cv_balanced_accuracy", 0.0)),
        holdout_macro_f1=float(tags.get("holdout_macro_f1", 0.0)) or None,
        holdout_balanced_accuracy=(float(tags.get("holdout_balanced_accuracy", 0.0)) or None),
        holdout_per_class_recall=per_class_recall,
    )


def _load_challenger_by_run_id(
    task: str,
    run_id: str,
    mlflow_uri: str,
    cfg: Config,
) -> MetricSnapshot:
    """Load challenger metrics directly from an MLflow run."""
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient(tracking_uri=mlflow_uri)
    run = client.get_run(run_id)
    metrics = run.data.metrics
    params = run.data.params
    tags = run.data.tags

    model_name = (
        cfg.mlflow_registered_health if task == "health" else cfg.mlflow_registered_restoration
    )

    per_class_recall = _load_per_class_recall_from_run(run_id, mlflow_uri)

    return MetricSnapshot(
        source="challenger",
        model_name=model_name,
        version="unregistered",
        run_id=run_id,
        algo_name=params.get("algorithm", tags.get("algorithm", "unknown")),
        cv_macro_f1=float(metrics.get("cv_macro_f1_mean", 0.0)),
        cv_balanced_accuracy=float(metrics.get("cv_balanced_accuracy_mean", 0.0)),
        holdout_macro_f1=float(metrics.get("holdout_macro_f1", 0.0)) or None,
        holdout_balanced_accuracy=(float(metrics.get("holdout_balanced_accuracy", 0.0)) or None),
        holdout_per_class_recall=per_class_recall,
    )


def _load_per_class_recall_from_run(run_id: str, mlflow_uri: str) -> dict[str, float]:
    """Try to load per-class recall from the challenger run artifact."""
    try:
        import mlflow

        mlflow.set_tracking_uri(mlflow_uri)
        client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_uri)
        artifact_path = "holdout_per_class_metrics.json"
        local_path = client.download_artifacts(run_id, artifact_path)
        with open(local_path, encoding="utf-8") as fh:
            per_class: dict = json.load(fh)
        return {
            cls: float(metrics.get("recall", 0.0))
            for cls, metrics in per_class.items()
            if isinstance(metrics, dict)
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Comparison engine
# ---------------------------------------------------------------------------


def _get_comparison_rules(task: str, cfg: Config) -> dict[str, float]:
    """Extract comparison rules for the given task from params.yaml."""
    rt = cfg.retraining
    comp = rt.get("comparison", {}).get(task, {})
    return {
        "min_abs_macro_f1": float(comp.get("min_abs_macro_f1", 0.65)),
        "max_macro_f1_regression": float(comp.get("max_macro_f1_regression", 0.05)),
        "min_balanced_accuracy": float(comp.get("min_balanced_accuracy", 0.65)),
        "max_per_class_recall_regression": float(comp.get("max_per_class_recall_regression", 0.10)),
    }


def compare_metrics(
    task: str,
    champion: MetricSnapshot,
    challenger: MetricSnapshot,
    cfg: Config,
) -> ComparisonReport:
    """
    Apply all comparison rules and produce a ComparisonReport.

    Does NOT change the champion alias.
    """
    rules = _get_comparison_rules(task, cfg)
    failures: list[str] = []
    warnings: list[str] = []
    gate_results: list[dict[str, Any]] = []

    ch_f1 = challenger.cv_macro_f1
    ch_bal = challenger.cv_balanced_accuracy
    cp_f1 = champion.cv_macro_f1

    # Gate 1: absolute minimum CV macro-F1
    g1_pass = ch_f1 >= rules["min_abs_macro_f1"]
    gate_results.append(
        {
            "gate": "min_abs_macro_f1",
            "passed": g1_pass,
            "challenger": ch_f1,
            "threshold": rules["min_abs_macro_f1"],
        }
    )
    if not g1_pass:
        failures.append(
            f"Challenger CV macro-F1 {ch_f1:.4f} < required minimum "
            f"{rules['min_abs_macro_f1']:.4f}."
        )

    # Gate 2: macro-F1 regression vs champion
    f1_diff = ch_f1 - cp_f1  # positive = improvement
    regression = max(0.0, -f1_diff)  # how much worse the challenger is
    g2_pass = regression <= rules["max_macro_f1_regression"]
    gate_results.append(
        {
            "gate": "max_macro_f1_regression",
            "passed": g2_pass,
            "challenger_cv_f1": ch_f1,
            "champion_cv_f1": cp_f1,
            "regression": regression,
            "max_allowed": rules["max_macro_f1_regression"],
        }
    )
    if not g2_pass:
        failures.append(
            f"Challenger CV macro-F1 regresses by {regression:.4f} vs champion "
            f"(max allowed {rules['max_macro_f1_regression']:.4f})."
        )

    # Gate 3: absolute minimum balanced accuracy
    g3_pass = ch_bal >= rules["min_balanced_accuracy"]
    gate_results.append(
        {
            "gate": "min_balanced_accuracy",
            "passed": g3_pass,
            "challenger": ch_bal,
            "threshold": rules["min_balanced_accuracy"],
        }
    )
    if not g3_pass:
        failures.append(
            f"Challenger CV balanced accuracy {ch_bal:.4f} < required "
            f"{rules['min_balanced_accuracy']:.4f}."
        )

    # Gate 4: per-class recall regression (optional — only if both have data)
    per_class_gate_results: list[dict] = []
    champ_per_class: dict[str, float] = champion.holdout_per_class_recall
    chall_per_class: dict[str, float] = challenger.holdout_per_class_recall
    if champ_per_class and chall_per_class:
        for cls in champ_per_class:
            if cls in chall_per_class:
                cls_reg = max(0.0, champ_per_class[cls] - chall_per_class[cls])
                cls_pass = cls_reg <= rules["max_per_class_recall_regression"]
                per_class_gate_results.append(
                    {
                        "class": cls,
                        "passed": cls_pass,
                        "champion_recall": champ_per_class[cls],
                        "challenger_recall": chall_per_class[cls],
                        "regression": cls_reg,
                        "max_allowed": rules["max_per_class_recall_regression"],
                    }
                )
                if not cls_pass:
                    failures.append(
                        f"Per-class recall for '{cls}' regresses by "
                        f"{cls_reg:.4f} vs champion "
                        f"(max allowed {rules['max_per_class_recall_regression']:.4f})."
                    )
    else:
        warnings.append(
            "Per-class recall comparison skipped: holdout per-class metrics "
            "not available for both champion and challenger."
        )
    if per_class_gate_results:
        gate_results.append(
            {"gate": "per_class_recall_regression", "results": per_class_gate_results}
        )

    # Gate 5: schema and class compatibility (informational check)
    if challenger.algo_name and champion.algo_name:
        warnings.append(
            f"Algorithm changed: champion={champion.algo_name}, "
            f"challenger={challenger.algo_name}. "
            "Verify inference API compatibility before promotion."
        )

    # Determine outcome
    if failures:
        outcome = _OUTCOME_REJECT
    elif f1_diff <= 0.0:
        outcome = _OUTCOME_REVIEW
        warnings.append(
            f"Challenger CV macro-F1 ({ch_f1:.4f}) does not improve on champion "
            f"({cp_f1:.4f}). Manual review recommended before promotion."
        )
    else:
        outcome = _OUTCOME_ELIGIBLE

    report_id = f"compare_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    logger.info(
        "[%s] Comparison outcome: %s (challenger=%.4f, champion=%.4f, diff=%+.4f)",
        task,
        outcome,
        ch_f1,
        cp_f1,
        f1_diff,
    )

    return ComparisonReport(
        report_id=report_id,
        timestamp=datetime.now(UTC).isoformat(),
        task=task,
        outcome=outcome,
        champion=asdict(champion),
        challenger=asdict(challenger),
        gate_results=gate_results,
        failures=failures,
        warnings=warnings,
        comparison_rules=rules,
        champion_alias_changed=False,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compare_challenger(
    task: str,
    challenger_version: str | None = None,
    challenger_run_id: str | None = None,
    mlflow_uri: str | None = None,
    output_path: Path | None = None,
    cfg: Config | None = None,
) -> ComparisonReport:
    """
    Compare a challenger against the current champion for *task*.

    Provide either ``challenger_version`` (registered version number) or
    ``challenger_run_id`` (MLflow run ID from retraining).

    The report is written as JSON to *output_path* (if provided).
    """
    if challenger_version is None and challenger_run_id is None:
        raise ValueError("Provide either challenger_version or challenger_run_id.")

    cfg = cfg or get_config()
    mlflow_uri = mlflow_uri or cfg.mlflow_tracking_uri

    champion = _load_champion_metrics(task, mlflow_uri, cfg)

    if challenger_version is not None:
        challenger = _load_challenger_by_version(task, challenger_version, mlflow_uri, cfg)
    else:
        assert challenger_run_id is not None
        challenger = _load_challenger_by_run_id(task, challenger_run_id, mlflow_uri, cfg)

    report = compare_metrics(task, champion, challenger, cfg)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(asdict(report), fh, indent=2)
        logger.info("Comparison report saved to %s", output_path)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point for champion–challenger comparison.

    Exit codes
    ----------
    0   Comparison complete (any outcome).
    1   Runtime error.
    2   Invalid arguments.
    """
    setup_logging(__name__)
    parser = argparse.ArgumentParser(
        description="Compare a CoralSense challenger against the current champion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=list(_VALID_TASKS),
        help="Task to compare.",
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--challenger-version",
        default=None,
        dest="challenger_version",
        metavar="VERSION",
        help="Registered model version number of the challenger.",
    )
    grp.add_argument(
        "--challenger-run-id",
        default=None,
        dest="challenger_run_id",
        metavar="RUN_ID",
        help="MLflow run ID of the challenger training run.",
    )
    parser.add_argument(
        "--mlflow-uri",
        default=None,
        dest="mlflow_uri",
        metavar="URI",
        help="MLflow tracking URI (default: from config).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="JSON",
        help="Path to write comparison report JSON.",
    )
    args = parser.parse_args(argv)

    try:
        cfg = get_config()
        report = compare_challenger(
            task=args.task,
            challenger_version=args.challenger_version,
            challenger_run_id=args.challenger_run_id,
            mlflow_uri=args.mlflow_uri,
            output_path=args.output,
            cfg=cfg,
        )
        print(f"Outcome : {report.outcome}")
        print(f"Task    : {report.task}")
        print(
            f"Champion: {report.champion['model_name']} v{report.champion['version']} "
            f"cv_f1={report.champion['cv_macro_f1']:.4f}"
        )
        print(
            f"Challenger: {report.challenger['model_name']} v{report.challenger['version']} "
            f"cv_f1={report.challenger['cv_macro_f1']:.4f}"
        )
        if report.failures:
            print("Failures:")
            for f in report.failures:
                print(f"  - {f}")
        if report.warnings:
            print("Warnings:")
            for w in report.warnings:
                print(f"  ! {w}")
        return 0
    except Exception as exc:
        logger.error("Comparison failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
