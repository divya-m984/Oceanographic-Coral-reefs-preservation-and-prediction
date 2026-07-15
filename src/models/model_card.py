"""
src/models/model_card.py — Model card generation for CoralSense.

Generates a Markdown model card for a registered model version, including:
- Model metadata (name, version, algorithm, task)
- Training provenance (data source, hash, timestamp, Git commit)
- Performance metrics (CV and holdout)
- Quality gate result
- Drift context (if available)
- Promotion/rollback history
- Synthetic data disclaimer

Model cards are saved as Markdown files under reports/model_cards/.

USAGE
-----
  python -m src.models.model_card \\
    --model coralsense_reef_health \\
    --version 1

  python -m src.models.model_card \\
    --model coralsense_reef_health \\
    --version 5 \\
    --output reports/model_cards/health_v5.md
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import Config, get_config, setup_logging

logger = logging.getLogger(__name__)

_SYNTHETIC_DISCLAIMER = (
    "**SYNTHETIC DATA DISCLAIMER**: All metrics in this model card reflect "
    "performance on the CoralSense *synthetic* dataset, generated for "
    "college project purposes. They must NOT be interpreted as evidence of "
    "real-world coral reef conservation accuracy."
)


# ---------------------------------------------------------------------------
# Metadata loader
# ---------------------------------------------------------------------------


def _load_version_metadata(
    model_name: str,
    version: str,
    mlflow_uri: str,
    cfg: Config,
) -> dict[str, Any]:
    """Load all available metadata for a registered model version."""
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=mlflow_uri)
    mv = client.get_model_version(name=model_name, version=version)
    raw_tags = mv.tags or {}
    tags = raw_tags if isinstance(raw_tags, dict) else {t.key: t.value for t in raw_tags}

    run_id = mv.run_id
    run_metrics: dict[str, float] = {}
    run_params: dict[str, str] = {}

    if run_id:
        try:
            run = client.get_run(run_id)
            run_metrics = dict(run.data.metrics)
            run_params = dict(run.data.params)
        except Exception:
            pass

    return {
        "model_name": model_name,
        "version": str(mv.version),
        "run_id": run_id,
        "description": mv.description or "",
        "creation_timestamp": mv.creation_timestamp,
        "tags": tags,
        "run_metrics": run_metrics,
        "run_params": run_params,
    }


# ---------------------------------------------------------------------------
# Model card renderer
# ---------------------------------------------------------------------------


def generate_model_card(
    model_name: str,
    version: str,
    mlflow_uri: str | None = None,
    output_path: Path | None = None,
    cfg: Config | None = None,
) -> str:
    """
    Generate a Markdown model card for a registered model version.

    Parameters
    ----------
    model_name:
        Registered model name.
    version:
        Version number string.
    mlflow_uri:
        SQLite tracking URI.
    output_path:
        If provided, write the Markdown to this path.
    cfg:
        Config instance.

    Returns
    -------
    Markdown string.
    """
    cfg = cfg or get_config()
    mlflow_uri = mlflow_uri or cfg.mlflow_tracking_uri

    meta = _load_version_metadata(model_name, version, mlflow_uri, cfg)
    tags = meta["tags"]
    metrics = meta["run_metrics"]
    params = meta["run_params"]

    task = tags.get("task", "unknown")
    algo = tags.get("algo_name", tags.get("algorithm", params.get("algorithm", "unknown")))
    data_source = tags.get("data_source", tags.get("synthetic_data_status", "unknown"))
    role = tags.get("role", "champion")
    cv_f1 = tags.get("cv_macro_f1", metrics.get("cv_macro_f1_mean", "N/A"))
    cv_bal = tags.get("cv_balanced_accuracy", metrics.get("cv_balanced_accuracy_mean", "N/A"))
    holdout_f1 = tags.get("holdout_macro_f1", metrics.get("holdout_macro_f1", "N/A"))
    holdout_bal = tags.get(
        "holdout_balanced_accuracy", metrics.get("holdout_balanced_accuracy", "N/A")
    )
    gate_passed = tags.get("quality_gate_passed", "unknown")
    input_sha = tags.get("input_sha256", "N/A")
    drift_sha = tags.get("drift_report_sha256", "")
    rt_reason = tags.get("retraining_reason", "initial training")
    git_commit = tags.get("git_commit", "unknown")
    label_names = tags.get("label_names", params.get("label_names", ""))
    n_features = tags.get("n_features", params.get("n_features", "N/A"))
    train_rows = tags.get("train_rows", params.get("train_rows", "N/A"))
    reg_ts = tags.get("registration_timestamp", "N/A")

    def _fmt(v: Any) -> str:
        try:
            return f"{float(v):.4f}"
        except (TypeError, ValueError):
            return str(v)

    lines: list[str] = [
        f"# Model Card — {model_name} v{version}",
        "",
        f"> Generated: {datetime.now(UTC).isoformat()}",
        "",
        "---",
        "",
        _SYNTHETIC_DISCLAIMER,
        "",
        "---",
        "",
        "## Overview",
        "",
        "| Field            | Value |",
        "|:-----------------|:------|",
        f"| Model name       | `{model_name}` |",
        f"| Version          | {version} |",
        f"| Role             | {role} |",
        f"| Task             | {task} |",
        f"| Algorithm        | {algo} |",
        f"| Data source      | {data_source} |",
        f"| Label classes    | `{label_names}` |",
        f"| Features         | {n_features} |",
        f"| Training rows    | {train_rows} |",
        f"| Quality gate     | {gate_passed} |",
        f"| Git commit       | `{git_commit}` |",
        f"| Registered       | {reg_ts} |",
        f"| MLflow version   | {version} |",
        f"| MLflow run ID    | `{meta['run_id'] or 'N/A'}` |",
        "",
        "## Performance",
        "",
        "All metrics are on **synthetic data** and do not indicate real-world accuracy.",
        "",
        "| Metric                    | Value |",
        "|:--------------------------|:------|",
        f"| CV macro-F1 (mean)        | {_fmt(cv_f1)} |",
        f"| CV balanced accuracy (mean)| {_fmt(cv_bal)} |",
        f"| Holdout macro-F1          | {_fmt(holdout_f1)} |",
        f"| Holdout balanced accuracy | {_fmt(holdout_bal)} |",
        "",
        "## Provenance",
        "",
        "| Field                | Value |",
        "|:---------------------|:------|",
        f"| Input SHA-256        | `{input_sha[:24]}...` |",
        f"| Drift report SHA-256 | `{drift_sha[:24] + '...' if drift_sha else 'N/A'}` |",
        f"| Retraining reason    | {rt_reason} |",
        "",
        "## Intended Use",
        "",
        "- **Purpose**: Prototype demonstration for an MLOps college project.",
        "- **Primary users**: Course instructors and students.",
        "- **Out-of-scope**: Real marine conservation decisions, deployment without",
        "  real labelled field data, regulatory or policy use.",
        "",
        "## Limitations",
        "",
        "- All training data is synthetic (computationally generated).",
        "- Label generation is partially deterministic; reported metrics overestimate",
        "  real-world predictive power.",
        "- No real sonar hardware has been used.",
        "- Class imbalance handling uses `class_weight=balanced` (sklearn) or",
        "  `compute_sample_weight` (XGBoost); real-world performance may differ.",
        "",
        "## Description",
        "",
        meta["description"] or "_No description provided._",
        "",
    ]

    card = "\n".join(lines)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            fh.write(card)
        logger.info("Model card saved to %s", output_path)

    return card


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point for model card generation.

    Exit codes
    ----------
    0   Success.
    1   Runtime error.
    2   Invalid arguments.
    """
    setup_logging(__name__)
    parser = argparse.ArgumentParser(
        description="Generate a Markdown model card for a CoralSense registered model version.",
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
        help="Model version number.",
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
        metavar="MD",
        help="Output Markdown path (default: reports/model_cards/<model>_v<version>.md).",
    )
    args = parser.parse_args(argv)

    try:
        cfg = get_config()
        output = args.output
        if output is None:
            output = cfg.paths.reports_dir / "model_cards" / f"{args.model}_v{args.version}.md"
        card = generate_model_card(
            model_name=args.model,
            version=args.version,
            mlflow_uri=args.mlflow_uri,
            output_path=output,
            cfg=cfg,
        )
        print(f"Model card written to: {output}")
        print(f"  Lines: {len(card.splitlines())}")
        return 0
    except Exception as exc:
        logger.error("Model card generation failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
