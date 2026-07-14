"""
src/monitoring/run_drift.py — CLI entry point for CoralSense drift monitoring.

Usage
-----
    python -m src.monitoring.run_drift
    python -m src.monitoring.run_drift --no-html
    python -m src.monitoring.run_drift --shift-scale 2.0
    python -m src.monitoring.run_drift --shift-scale 0 --no-html   # zero-drift baseline

What it does
------------
1. Loads the raw observations CSV and samples a reference window.
2. Generates a synthetic production window with configurable distribution shift.
3. Loads the champion InferencePipeline for both tasks.
4. Runs predictions on both windows.
5. Computes feature, prediction and confidence drift via DriftDetector.
6. Saves a JSON summary to reports/drift_summary.json.
7. Optionally saves Evidently HTML reports to reports/.

SYNTHETIC-DATA DISCLAIMER
--------------------------
All windows are based on SYNTHETIC data.  Results must NOT be used to guide
actual marine conservation decisions.

Exit codes
----------
0  Success.
1  Runtime error (missing file, model load failure).
2  Invalid arguments.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import get_config, setup_logging
from src.monitoring.drift import DriftDetector
from src.monitoring.generate_production import (
    generate_production_window,
    generate_reference_window,
)

logger = logging.getLogger(__name__)

_SYNTHETIC_DISCLAIMER = (
    "All windows used in CoralSense drift monitoring are based on SYNTHETIC data. "
    "Results must NOT be used to guide actual marine conservation decisions."
)


# ---------------------------------------------------------------------------
# Model prediction helper
# ---------------------------------------------------------------------------


def _get_predictions(
    pipeline_factory: Any,
    df: pd.DataFrame,
    task: str,
) -> tuple[list[str], list[float]]:
    """
    Run *df* through *pipeline_factory(task)* and return (labels, confidences).

    Handles pipelines that return dicts with ``predicted_class`` and
    ``confidence`` keys (same as InferencePipeline.predict_batch output).
    """
    pipeline = pipeline_factory(task)
    results = pipeline.predict_batch(df)
    labels = [r["predicted_class"] for r in results]
    confidences = [float(r["confidence"]) for r in results]
    return labels, confidences


# ---------------------------------------------------------------------------
# Main drift run function (testable without CLI)
# ---------------------------------------------------------------------------


def run_drift(
    cfg=None,
    shift_scale: float | None = None,
    generate_html: bool = True,
    pipeline_factory=None,
    raw_csv: Path | None = None,
) -> dict[str, Any]:
    """
    Execute the full drift-monitoring pipeline and return the summary dict.

    Parameters
    ----------
    cfg:
        Config instance (created from defaults if None).
    shift_scale:
        Overrides ``monitoring.shift_scale`` from params.
    generate_html:
        If True, write Evidently HTML reports to ``reports/``.
    pipeline_factory:
        Callable ``(task: str) -> InferencePipeline``.  If None, creates
        real InferencePipeline instances using the champion registry.
        Inject a mock in tests.
    raw_csv:
        Path to raw observations CSV.  Defaults to ``cfg.raw_data_path``.

    Returns
    -------
    The drift summary dict (same structure as written to drift_summary.json).
    """
    cfg = cfg or get_config()
    p = cfg.monitoring
    _shift_scale = shift_scale if shift_scale is not None else p["shift_scale"]
    _raw_csv = raw_csv or cfg.raw_data_path

    rng = np.random.default_rng(cfg.random_seed)

    # 1. Generate windows
    logger.info("Generating reference window (n=%d)", p["reference_n"])
    ref_df = generate_reference_window(_raw_csv, p["reference_n"], rng)

    logger.info(
        "Generating production window (n=%d, shift_scale=%.2f)",
        p["production_n"],
        _shift_scale,
    )
    prod_df = generate_production_window(_raw_csv, p["production_n"], _shift_scale, rng)

    # Save windows
    ref_dir = cfg.paths.reference_data_dir
    prod_dir = cfg.paths.production_data_dir
    ref_dir.mkdir(parents=True, exist_ok=True)
    prod_dir.mkdir(parents=True, exist_ok=True)
    ref_path = ref_dir / p["reference_filename"]
    prod_path = prod_dir / p["production_filename"]
    ref_df.to_csv(ref_path, index=False)
    prod_df.to_csv(prod_path, index=False)
    logger.info("Reference saved to %s", ref_path)
    logger.info("Production saved to %s", prod_path)

    # 2. Load inference pipelines (real or injected)
    if pipeline_factory is None:
        from src.models.predict import InferencePipeline

        def pipeline_factory(task: str) -> Any:  # noqa: F811
            return InferencePipeline(task=task, cfg=cfg)

    # Numeric feature columns for drift analysis
    numeric_cols = cfg.numeric_features

    # 3. Run predictions on both windows
    logger.info("Running predictions on reference window")
    ref_health_labels, ref_health_conf = _get_predictions(pipeline_factory, ref_df, "health")
    ref_rest_labels, ref_rest_conf = _get_predictions(pipeline_factory, ref_df, "restoration")

    logger.info("Running predictions on production window")
    prod_health_labels, prod_health_conf = _get_predictions(pipeline_factory, prod_df, "health")
    prod_rest_labels, prod_rest_conf = _get_predictions(pipeline_factory, prod_df, "restoration")

    # 4. Compute drift
    detector = DriftDetector(drift_threshold=cfg.drift_threshold)

    logger.info("Computing feature drift")
    feature_drift = detector.compute_feature_drift(ref_df, prod_df, columns=numeric_cols)

    logger.info("Computing prediction drift (health)")
    pred_health_drift = detector.compute_prediction_drift(
        ref_health_labels, prod_health_labels, cfg.health_classes
    )

    logger.info("Computing prediction drift (restoration)")
    pred_rest_drift = detector.compute_prediction_drift(
        ref_rest_labels, prod_rest_labels, cfg.restoration_classes
    )

    logger.info("Computing confidence drift (health)")
    conf_health_drift = detector.compute_confidence_drift(ref_health_conf, prod_health_conf)

    logger.info("Computing confidence drift (restoration)")
    conf_rest_drift = detector.compute_confidence_drift(ref_rest_conf, prod_rest_conf)

    # 5. Recommendation
    recommendation = detector.make_recommendation(
        feature_drifted=feature_drift["drifted_count"] > 0,
        pred_health_drifted=pred_health_drift["drifted"],
        pred_restoration_drifted=pred_rest_drift["drifted"],
    )

    # 6. Build summary
    summary: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "shift_scale": _shift_scale,
        "reference_n": len(ref_df),
        "production_n": len(prod_df),
        "drift_threshold": cfg.drift_threshold,
        "feature_drift": feature_drift,
        "prediction_drift": {
            "health": pred_health_drift,
            "restoration": pred_rest_drift,
        },
        "confidence_drift": {
            "health": conf_health_drift,
            "restoration": conf_rest_drift,
        },
        "recommendation": recommendation,
        "synthetic_data_disclaimer": _SYNTHETIC_DISCLAIMER,
    }

    # 7. Save JSON summary
    reports_dir = cfg.paths.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / p["summary_filename"]
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Drift summary saved to %s", summary_path)

    # 8. Optional HTML reports
    if generate_html:
        _write_html_reports(detector, ref_df, prod_df, numeric_cols, cfg, p)

    return summary


def _write_html_reports(
    detector: DriftDetector,
    ref_df: pd.DataFrame,
    prod_df: pd.DataFrame,
    numeric_cols: list[str],
    cfg: Any,
    p: dict[str, Any],
) -> None:
    """Write per-task Evidently HTML reports."""
    for task, report_key in [
        ("health", "report_filename_health"),
        ("restoration", "report_filename_restoration"),
    ]:
        report_path = cfg.paths.reports_dir / p[report_key]
        try:
            detector.generate_html_report(ref_df, prod_df, report_path, columns=numeric_cols)
            logger.info("[%s] HTML report saved to %s", task, report_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] HTML report generation failed: %s", task, exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    setup_logging(__name__)
    parser = argparse.ArgumentParser(
        description="CoralSense drift monitoring — compare reference vs production windows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--shift-scale",
        type=float,
        default=None,
        help=(
            "Magnitude of the synthetic distribution shift applied to the "
            "production window (0 = no shift, 1 = standard shift). "
            "Defaults to monitoring.shift_scale in params.yaml."
        ),
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip Evidently HTML report generation (faster).",
    )

    args = parser.parse_args(argv)

    try:
        summary = run_drift(
            shift_scale=args.shift_scale,
            generate_html=not args.no_html,
        )
        feature_count = summary["feature_drift"]["drifted_count"]
        pred_h = summary["prediction_drift"]["health"]["drifted"]
        pred_r = summary["prediction_drift"]["restoration"]["drifted"]
        print(
            f"\nDrift summary:\n"
            f"  Feature drift    : {feature_count} column(s) drifted\n"
            f"  Prediction drift : health={pred_h}, restoration={pred_r}\n"
            f"  Recommendation   : {summary['recommendation'][:80]}...\n"
        )
        return 0
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.error("Drift monitoring failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
