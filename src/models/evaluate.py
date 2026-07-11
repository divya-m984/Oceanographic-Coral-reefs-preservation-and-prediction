"""
src/models/evaluate.py — Evaluation utilities for CoralSense classifiers.

Responsibilities
----------------
- Compute the full metrics dictionary from predictions.
- Format per-class and aggregate metrics for logging and display.
- Print a side-by-side model comparison table.

SYNTHETIC-DATA DISCLAIMER
--------------------------
All metrics reported by this module reflect performance on the CoralSense
*synthetic* dataset.  They must NOT be interpreted as evidence of real-world
conservation accuracy.  Correlation between features and labels is partially
deterministic by design (see docs/data_dictionary.md — Label generation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    y_true,
    y_pred,
    y_proba: np.ndarray | None,
    label_names: list[str],
) -> dict:
    """
    Compute the full evaluation metrics dictionary.

    Parameters
    ----------
    y_true:
        Ground-truth labels (string or integer array-like).
    y_pred:
        Predicted labels (same dtype as y_true).
    y_proba:
        Predicted probability matrix, shape (n_samples, n_classes).
        Pass ``None`` if the model does not support probability output.
    label_names:
        Ordered list of class names matching columns of y_proba.

    Returns
    -------
    Dict containing scalar metrics, per-class breakdowns, confusion matrix
    (as nested list), and a formatted classification report string.
    """
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    report_str = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=label_names).tolist()

    per_class: dict[str, dict] = {}
    for lbl in label_names:
        row = report_dict.get(lbl, {})
        per_class[lbl] = {
            "precision": round(float(row.get("precision", 0.0)), 4),
            "recall": round(float(row.get("recall", 0.0)), 4),
            "f1": round(float(row.get("f1-score", 0.0)), 4),
            "support": int(row.get("support", 0)),
        }

    metrics = {
        # Aggregate scalars
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "macro_precision": round(
            float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
        "macro_recall": round(
            float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(
            float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4
        ),
        # Per-class breakdown
        "per_class": per_class,
        # Confusion matrix and text report
        "confusion_matrix": cm,
        "classification_report": report_str,
    }
    return metrics


def format_comparison_table(task: str, model_results: dict) -> str:
    """
    Return an ASCII comparison table of all trained models for *task*.

    Parameters
    ----------
    task:
        ``"health"`` or ``"restoration"``.
    model_results:
        Dict mapping algorithm name → metrics dict (as returned by train_task).

    Returns
    -------
    Multi-line formatted string, ready to print.
    """
    rows = []
    for algo, r in model_results.items():
        rows.append(
            {
                "Model": algo,
                "CV macro-F1": f"{r.get('cv_macro_f1_mean', 0):.4f}"
                f" ±{r.get('cv_macro_f1_std', 0):.4f}",
                "CV bal-acc": f"{r.get('cv_balanced_accuracy_mean', 0):.4f}",
                "Test acc": f"{r.get('test_accuracy', 0):.4f}",
                "Test bal-acc": f"{r.get('test_balanced_accuracy', 0):.4f}",
                "Test macro-F1": f"{r.get('test_macro_f1', 0):.4f}",
                "Test wtd-F1": f"{r.get('test_weighted_f1', 0):.4f}",
            }
        )
    df = pd.DataFrame(rows).set_index("Model")
    sep = "-" * 78
    lines = [
        "",
        sep,
        f"  CoralSense — Model Comparison [{task}]",
        sep,
        df.to_string(),
        sep,
        "  [SYNTHETIC DATA — metrics do not indicate real-world performance]",
        sep,
        "",
    ]
    return "\n".join(lines)
