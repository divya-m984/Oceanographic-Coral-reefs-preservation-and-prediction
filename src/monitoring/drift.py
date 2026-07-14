"""
src/monitoring/drift.py — Core drift-detection logic for CoralSense.

Computes three types of drift between a reference window and a current
(production) window:

1. **Feature drift** — Whether the distribution of individual sensor features
   has shifted.  Uses Evidently's DataDriftPreset (Kolmogorov-Smirnov for
   continuous columns, chi-squared for categorical).

2. **Prediction drift** — Whether the distribution of predicted classes has
   changed.  Uses a chi-squared test on the label-count vectors.

3. **Confidence drift** — Whether the model's average confidence has changed.
   Uses a KS test on the per-sample max-probability scores.

All methods are pure-function: they accept DataFrames / lists and return
plain dicts.  No MLflow interaction, no model loading, no file I/O.

SYNTHETIC-DATA DISCLAIMER
--------------------------
All windows used in CoralSense drift monitoring are based on SYNTHETIC data.
Results must not be used to guide actual marine conservation decisions.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from scipy import stats

logger = logging.getLogger(__name__)

_SYNTHETIC_DISCLAIMER = (
    "Drift analysis performed on SYNTHETIC data. Results do not represent "
    "real ocean conditions and must not be used to guide conservation decisions."
)

# Recommendation templates
_REC_RETRAIN = (
    "RETRAIN RECOMMENDED: Significant feature drift and prediction-distribution "
    "drift detected. The current data distribution has moved away from the "
    "training distribution. Schedule model retraining on a representative "
    "dataset that includes the new distribution."
)
_REC_INVESTIGATE_FEATURE = (
    "INVESTIGATE: Feature drift detected but predictions appear stable. "
    "Monitor for continued drift over the next window. Verify that current "
    "sensor data remains within plausible physical ranges."
)
_REC_INVESTIGATE_PRED = (
    "INVESTIGATE: Prediction distribution has shifted but no significant feature "
    "drift was detected. Review data-quality pipelines and confirm that the "
    "production feature distribution is consistent with the reference window."
)
_REC_OK = (
    "OK: No significant drift detected across features, predictions or "
    "confidence scores. Continue routine monitoring."
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_feature_cols(df: pd.DataFrame, columns: list[str] | None) -> list[str]:
    """Return the list of columns to analyse (default: all non-object-typed numeric)."""
    if columns is not None:
        return [c for c in columns if c in df.columns]
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------


class DriftDetector:
    """
    Stateless helper that computes drift metrics between two data windows.

    Parameters
    ----------
    drift_threshold:
        p-value threshold below which a column is declared drifted (default 0.10).
    """

    def __init__(self, drift_threshold: float = 0.10) -> None:
        self.drift_threshold = drift_threshold

    # ------------------------------------------------------------------
    # Feature drift
    # ------------------------------------------------------------------

    def compute_feature_drift(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Run Evidently DataDriftPreset on *columns* and return a summary dict.

        Parameters
        ----------
        reference:
            Reference-window DataFrame.
        current:
            Current (production) window DataFrame.
        columns:
            Columns to analyse.  If None, all numeric columns are used.

        Returns
        -------
        Dict with keys:
          ``drifted_count``, ``total_columns``, ``drifted_share``,
          ``per_column`` (mapping column → ``{drifted, p_value, method}``).
        """
        cols = _to_feature_cols(reference, columns)
        ref_sub = reference[cols].copy()
        cur_sub = current[[c for c in cols if c in current.columns]].copy()

        report = Report([DataDriftPreset(drift_share=self.drift_threshold)])
        snap = report.run(reference_data=ref_sub, current_data=cur_sub)
        metrics = snap.dict()["metrics"]

        per_column: dict[str, dict[str, Any]] = {}
        drifted_count = 0
        drifted_share = 0.0

        for m in metrics:
            name: str = m["metric_name"]
            value = m["value"]

            if name.startswith("DriftedColumnsCount"):
                if isinstance(value, dict):
                    drifted_count = int(value.get("count", 0))
                    drifted_share = float(value.get("share", 0.0))
            elif name.startswith("ValueDrift("):
                # Parse column name from metric_name string
                # Format: "ValueDrift(column=X,method=Y,threshold=Z)"
                col_name = _parse_column_from_metric_name(name)
                p_value = float(value) if isinstance(value, (int, float)) else float("nan")
                method = _parse_method_from_metric_name(name)
                drifted = p_value < self.drift_threshold if not np.isnan(p_value) else False
                if col_name:
                    per_column[col_name] = {
                        "drifted": drifted,
                        "p_value": round(p_value, 6),
                        "method": method,
                    }

        return {
            "drifted_count": drifted_count,
            "total_columns": len(cols),
            "drifted_share": round(drifted_share, 4),
            "per_column": per_column,
        }

    # ------------------------------------------------------------------
    # Prediction drift
    # ------------------------------------------------------------------

    def compute_prediction_drift(
        self,
        ref_labels: list[str],
        cur_labels: list[str],
        label_names: list[str],
    ) -> dict[str, Any]:
        """
        Chi-squared test on label-count vectors.

        Parameters
        ----------
        ref_labels:
            Predicted class labels for the reference window.
        cur_labels:
            Predicted class labels for the current window.
        label_names:
            All valid label names (determines count-vector ordering).

        Returns
        -------
        Dict with keys:
          ``drifted``, ``p_value``, ``statistic``, ``method``,
          ``reference_distribution``, ``current_distribution``.
        """
        ref_counts = np.array([Counter(ref_labels).get(lbl, 0) for lbl in label_names], dtype=float)
        cur_counts = np.array([Counter(cur_labels).get(lbl, 0) for lbl in label_names], dtype=float)

        # Normalize to avoid zero-sum arrays
        ref_n = ref_counts.sum()
        cur_n = cur_counts.sum()

        if ref_n == 0 or cur_n == 0:
            return {
                "drifted": False,
                "p_value": 1.0,
                "statistic": 0.0,
                "method": "chi2",
                "reference_distribution": {},
                "current_distribution": {},
            }

        # Use chi2_contingency on a 2×k contingency table to avoid the
        # sum-matching precision requirement of scipy.stats.chisquare.
        contingency = np.array([ref_counts, cur_counts])
        # Avoid zero columns (would make chi2_contingency ill-defined)
        mask = (ref_counts + cur_counts) > 0
        if mask.sum() < 2:
            # Not enough classes with observations
            return {
                "drifted": False,
                "p_value": 1.0,
                "statistic": 0.0,
                "method": "chi2",
                "reference_distribution": {
                    lbl: round(float(ref_counts[i] / ref_n), 4) for i, lbl in enumerate(label_names)
                },
                "current_distribution": {
                    lbl: round(float(cur_counts[i] / cur_n), 4) for i, lbl in enumerate(label_names)
                },
            }
        chi2_stat, p_value, _, _ = stats.chi2_contingency(contingency[:, mask])

        ref_dist = {
            lbl: round(float(ref_counts[i] / ref_n), 4) for i, lbl in enumerate(label_names)
        }
        cur_dist = {
            lbl: round(float(cur_counts[i] / cur_n), 4) for i, lbl in enumerate(label_names)
        }

        return {
            "drifted": bool(p_value < self.drift_threshold),
            "p_value": round(float(p_value), 6),
            "statistic": round(float(chi2_stat), 4),
            "method": "chi2",
            "reference_distribution": ref_dist,
            "current_distribution": cur_dist,
        }

    # ------------------------------------------------------------------
    # Confidence drift
    # ------------------------------------------------------------------

    def compute_confidence_drift(
        self,
        ref_confidence: list[float],
        cur_confidence: list[float],
    ) -> dict[str, Any]:
        """
        KS test on per-sample confidence (max-probability) distributions.

        Parameters
        ----------
        ref_confidence:
            Confidence scores for the reference window.
        cur_confidence:
            Confidence scores for the current window.

        Returns
        -------
        Dict with keys:
          ``drifted``, ``p_value``, ``statistic``, ``method``,
          ``mean_reference``, ``mean_current``, ``delta``.
        """
        ref_arr = np.asarray(ref_confidence, dtype=float)
        cur_arr = np.asarray(cur_confidence, dtype=float)

        if len(ref_arr) == 0 or len(cur_arr) == 0:
            return {
                "drifted": False,
                "p_value": 1.0,
                "statistic": 0.0,
                "method": "ks",
                "mean_reference": 0.0,
                "mean_current": 0.0,
                "delta": 0.0,
            }

        ks_stat, p_value = stats.ks_2samp(ref_arr, cur_arr)
        mean_ref = float(np.mean(ref_arr))
        mean_cur = float(np.mean(cur_arr))

        return {
            "drifted": bool(p_value < self.drift_threshold),
            "p_value": round(float(p_value), 6),
            "statistic": round(float(ks_stat), 4),
            "method": "ks",
            "mean_reference": round(mean_ref, 4),
            "mean_current": round(mean_cur, 4),
            "delta": round(mean_cur - mean_ref, 4),
        }

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def generate_html_report(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        output_path: Path,
        columns: list[str] | None = None,
    ) -> None:
        """
        Generate an Evidently HTML report combining DataSummaryPreset and
        DataDriftPreset for the given *columns*.

        Parameters
        ----------
        reference:
            Reference-window DataFrame.
        current:
            Current (production) window DataFrame.
        output_path:
            Path to write the HTML file.
        columns:
            Columns to include.  If None, all numeric columns are used.
        """
        cols = _to_feature_cols(reference, columns)
        ref_sub = reference[cols].copy()
        cur_sub = current[[c for c in cols if c in current.columns]].copy()

        report = Report([DataSummaryPreset(), DataDriftPreset()])
        snap = report.run(reference_data=ref_sub, current_data=cur_sub)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        snap.save_html(str(output_path))
        logger.info("Evidently HTML report saved to %s", output_path)

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------

    def make_recommendation(
        self,
        feature_drifted: bool,
        pred_health_drifted: bool,
        pred_restoration_drifted: bool,
    ) -> str:
        """Return a plain-text retraining recommendation."""
        pred_drifted = pred_health_drifted or pred_restoration_drifted
        if feature_drifted and pred_drifted:
            return _REC_RETRAIN
        if feature_drifted:
            return _REC_INVESTIGATE_FEATURE
        if pred_drifted:
            return _REC_INVESTIGATE_PRED
        return _REC_OK


# ---------------------------------------------------------------------------
# Parsing helpers for Evidently 0.7 metric names
# ---------------------------------------------------------------------------


def _parse_column_from_metric_name(metric_name: str) -> str | None:
    """Extract column name from 'ValueDrift(column=X,method=Y,...)'."""
    try:
        inner = metric_name[metric_name.index("(") + 1 : metric_name.rindex(")")]
        for part in inner.split(","):
            part = part.strip()
            if part.startswith("column="):
                return part[len("column=") :]
    except (ValueError, IndexError):
        pass
    return None


def _parse_method_from_metric_name(metric_name: str) -> str:
    """Extract method name from 'ValueDrift(column=X,method=Y,...)'."""
    try:
        inner = metric_name[metric_name.index("(") + 1 : metric_name.rindex(")")]
        for part in inner.split(","):
            part = part.strip()
            if part.startswith("method="):
                return part[len("method=") :]
    except (ValueError, IndexError):
        pass
    return "unknown"
