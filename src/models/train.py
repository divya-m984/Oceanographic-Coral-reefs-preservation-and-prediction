"""
src/models/train.py — Model training pipeline for CoralSense classifiers.

Overview
--------
Trains and evaluates three algorithm families for each classification task:

  1. LogisticRegression   — class_weight="balanced" (sklearn)
  2. RandomForestClassifier — class_weight="balanced" (sklearn)
  3. XGBClassifier         — compute_sample_weight per fold (xgboost 3.3+)

XGBoost note (xgboost 3.3.0)
-----------------------------
XGBoost 3.3.0 requires **integer-encoded** labels when sample weights are
provided.  This module handles encoding transparently via LabelEncoder in
``_run_xgb_cv()`` and ``_fit_xgb()``.  The external interface always uses
the original string label names.

sklearn note (1.9.0)
--------------------
``cross_validate`` in sklearn 1.9 uses the ``params`` keyword (replacing the
deprecated ``fit_params``).  Sample weights are passed via ``params``.

LEAKAGE PREVENTION
------------------
The ColumnTransformer saved by M4 (already fitted on X_train only) is loaded
from disk and used exclusively with ``.transform()`` — never ``.fit()`` or
``.fit_transform()``.  The function ``_assert_preprocessor_not_refitted()``
verifies this at runtime.

SYNTHETIC-DATA DISCLAIMER
--------------------------
Performance metrics on this synthetic dataset must not be used as evidence of
real-world coral reef conservation accuracy.

Usage
-----
  python -m src.models.train --task health
  python -m src.models.train --task restoration
  python -m src.models.train --task all
  python -m src.models.train --task all --quick   # fast mode (for CI / tests)
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier

from src.config import Config, get_config, setup_logging
from src.models.evaluate import compute_metrics, format_comparison_table

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------

_TASK_CFG: dict[str, dict] = {
    "health": {
        "target_col": "reef_health",
        "label_names": ["healthy", "stressed", "bleached", "severely_degraded"],
    },
    "restoration": {
        "target_col": "restoration_suitability",
        "label_names": ["suitable", "moderately_suitable", "unsuitable"],
    },
}

# ---------------------------------------------------------------------------
# Quick-mode overrides for fast testing / CI
# ---------------------------------------------------------------------------

_QUICK_OVERRIDES: dict[str, dict] = {
    "logistic_regression": {"max_iter": 50},
    "random_forest": {"n_estimators": 10, "max_depth": 4, "min_samples_leaf": 1},
    "xgboost": {"n_estimators": 10, "max_depth": 3},
}
_QUICK_CV_FOLDS = 3

# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class TaskData:
    """Loaded, transformed and split data for one task."""

    task: str
    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    X_train_t: np.ndarray  # after preprocessor.transform()
    X_test_t: np.ndarray
    y_train: pd.Series
    y_test: pd.Series
    label_names: list[str]
    target_col: str
    feature_names: list[str]  # from preprocessor.get_feature_names_out()
    preprocessor: object  # fitted ColumnTransformer (never refitted here)


# ---------------------------------------------------------------------------
# Helper: git hash
# ---------------------------------------------------------------------------


def _get_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    except Exception:
        return "unavailable"


# ---------------------------------------------------------------------------
# Helper: verify preprocessor was NOT refitted
# ---------------------------------------------------------------------------


def _assert_preprocessor_fitted_and_unchanged(preprocessor, X_train_raw: pd.DataFrame) -> None:
    """
    Assert that the preprocessor is already fitted (from M4) and that its
    StandardScaler mean matches X_train_raw — confirming it was fitted on
    X_train only.

    Raises RuntimeError if the preprocessor appears unfitted or inconsistent.
    """
    try:
        check_is_fitted(preprocessor)
    except Exception as exc:
        raise RuntimeError(
            "Preprocessor loaded from disk is not fitted. Re-run src.data.preprocess to regenerate."
        ) from exc

    # Spot-check: scaler mean should equal training-data column mean
    from src.features.build_features import NUMERIC_FEATURE_COLUMNS

    scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
    numeric_cols = [c for c in NUMERIC_FEATURE_COLUMNS if c in X_train_raw.columns]
    expected_mean = X_train_raw[numeric_cols].mean().values
    if not np.allclose(scaler.mean_, expected_mean, rtol=1e-4):
        raise RuntimeError(
            "Preprocessor scaler mean does not match X_train mean. "
            "This may indicate the preprocessor was fitted on a different dataset."
        )
    logger.debug("Preprocessor integrity check PASSED")


# ---------------------------------------------------------------------------
# Algorithm factory
# ---------------------------------------------------------------------------


def _get_algorithms(task: str, cfg: Config, quick: bool = False) -> dict[str, object]:
    """
    Build unfitted estimators from params.yaml hyperparameters.

    Parameters
    ----------
    task:
        ``"health"`` or ``"restoration"``.
    cfg:
        Config object (provides model hyperparameters).
    quick:
        If True, override with lightweight settings for CI / testing.

    Returns
    -------
    Dict mapping algorithm name to unfitted sklearn-compatible estimator.
    """
    task_params: dict = cfg.model_params.get(task, {})

    def _p(name: str) -> dict:
        base = dict(task_params.get(name, {}))
        if quick:
            base.update(_QUICK_OVERRIDES.get(name, {}))
        return base

    lr_p = _p("logistic_regression")
    lr_p.setdefault("random_state", cfg.random_seed)
    lr_p.setdefault("class_weight", "balanced")

    rf_p = _p("random_forest")
    rf_p.setdefault("random_state", cfg.random_seed)
    rf_p.setdefault("class_weight", "balanced")
    rf_p.setdefault("n_jobs", 1)

    xgb_p = _p("xgboost")
    xgb_p.setdefault("random_state", cfg.random_seed)
    xgb_p.setdefault("eval_metric", "mlogloss")
    xgb_p.setdefault("verbosity", 0)
    xgb_p.setdefault("n_jobs", 1)

    return {
        "logistic_regression": LogisticRegression(**lr_p),
        "random_forest": RandomForestClassifier(**rf_p),
        "xgboost": XGBClassifier(**xgb_p),
    }


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------


def load_task_data(task: str, processed_dir: Path) -> TaskData:
    """
    Load the M4 processed splits and apply the already-fitted preprocessor.

    The preprocessor is loaded from disk and used with ``.transform()`` only
    — it is NEVER refitted here.

    Parameters
    ----------
    task:
        ``"health"`` or ``"restoration"``.
    processed_dir:
        Directory containing the M4 processed CSVs and joblib preprocessors.

    Returns
    -------
    TaskData with raw and transformed splits.
    """
    tcfg = _TASK_CFG[task]
    target_col = tcfg["target_col"]
    label_names = tcfg["label_names"]

    X_train_raw = pd.read_csv(processed_dir / f"X_train_{task}.csv")
    X_test_raw = pd.read_csv(processed_dir / f"X_test_{task}.csv")
    y_train = pd.read_csv(processed_dir / f"y_train_{task}.csv").squeeze()
    y_test = pd.read_csv(processed_dir / f"y_test_{task}.csv").squeeze()

    preprocessor = joblib.load(processed_dir / f"preprocessor_{task}.joblib")

    _assert_preprocessor_fitted_and_unchanged(preprocessor, X_train_raw)

    # Transform — NEVER fit or fit_transform
    X_train_t: np.ndarray = preprocessor.transform(X_train_raw)
    X_test_t: np.ndarray = preprocessor.transform(X_test_raw)

    feature_names: list[str] = list(preprocessor.get_feature_names_out())

    logger.info(
        "[%s] Data loaded: X_train %s, X_test %s, classes=%s",
        task,
        X_train_t.shape,
        X_test_t.shape,
        label_names,
    )

    return TaskData(
        task=task,
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
        X_train_t=X_train_t,
        X_test_t=X_test_t,
        y_train=y_train,
        y_test=y_test,
        label_names=label_names,
        target_col=target_col,
        feature_names=feature_names,
        preprocessor=preprocessor,
    )


# ---------------------------------------------------------------------------
# Cross-validation helpers
# ---------------------------------------------------------------------------


def _run_cv_sklearn(
    estimator,
    X: np.ndarray,
    y: pd.Series,
    cv: StratifiedKFold,
    n_jobs: int,
) -> dict:
    """Run cross_validate for sklearn models (LR, RF) with class_weight='balanced'."""
    from sklearn.model_selection import cross_validate

    results = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        scoring={"macro_f1": "f1_macro", "balanced_accuracy": "balanced_accuracy"},
        n_jobs=n_jobs,
        return_train_score=False,
    )
    return {
        "cv_macro_f1_mean": float(results["test_macro_f1"].mean()),
        "cv_macro_f1_std": float(results["test_macro_f1"].std()),
        "cv_balanced_accuracy_mean": float(results["test_balanced_accuracy"].mean()),
        "cv_balanced_accuracy_std": float(results["test_balanced_accuracy"].std()),
    }


def _run_cv_xgb(
    estimator: XGBClassifier,
    X: np.ndarray,
    y: pd.Series,
    cv: StratifiedKFold,
) -> dict:
    """
    Manual CV for XGBClassifier.

    XGBoost 3.3.0 requires integer-encoded labels when sample_weight is
    supplied.  We encode labels and compute balanced sample weights
    independently in each fold.
    """
    macro_f1s: list[float] = []
    bal_accs: list[float] = []
    y_arr = np.asarray(y)

    for train_idx, val_idx in cv.split(X, y_arr):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

        le = LabelEncoder()
        y_tr_int = le.fit_transform(y_tr)
        y_val_int = le.transform(y_val)
        sw = compute_sample_weight("balanced", y_tr_int)

        fold_clf = clone(estimator)
        fold_clf.fit(X_tr, y_tr_int, sample_weight=sw)

        y_pred = fold_clf.predict(X_val)
        macro_f1s.append(f1_score(y_val_int, y_pred, average="macro", zero_division=0))
        bal_accs.append(balanced_accuracy_score(y_val_int, y_pred))

    return {
        "cv_macro_f1_mean": float(np.mean(macro_f1s)),
        "cv_macro_f1_std": float(np.std(macro_f1s)),
        "cv_balanced_accuracy_mean": float(np.mean(bal_accs)),
        "cv_balanced_accuracy_std": float(np.std(bal_accs)),
    }


# ---------------------------------------------------------------------------
# Final-fit helpers
# ---------------------------------------------------------------------------


def _fit_sklearn(estimator, X_train_t: np.ndarray, y_train: pd.Series):
    """Fit a sklearn estimator with class_weight='balanced'."""
    return estimator.fit(X_train_t, y_train)


def _fit_xgb(
    estimator: XGBClassifier,
    X_train_t: np.ndarray,
    y_train: pd.Series,
) -> tuple:
    """
    Fit XGBClassifier with integer-encoded labels and balanced sample weights.

    Returns
    -------
    (fitted_estimator, label_encoder)
    """
    le = LabelEncoder()
    y_int = le.fit_transform(np.asarray(y_train))
    sw = compute_sample_weight("balanced", y_int)
    estimator.fit(X_train_t, y_int, sample_weight=sw)
    return estimator, le


# ---------------------------------------------------------------------------
# Feature importance extractor
# ---------------------------------------------------------------------------


def _get_feature_importance(algo_name: str, estimator, le, feature_names: list[str]) -> dict | None:
    """
    Extract a {feature_name: importance} dict, or None if not supported.

    For multi-class LogisticRegression, importance = mean(|coef|) across classes.
    """
    try:
        if algo_name == "logistic_regression":
            coef = estimator.coef_  # shape (n_classes, n_features) for multiclass
            importance = np.mean(np.abs(coef), axis=0)
        elif algo_name in ("random_forest", "xgboost"):
            importance = estimator.feature_importances_
        else:
            return None

        # Align with feature_names (trim to shortest, in case of shape mismatch)
        n = min(len(importance), len(feature_names))
        return {feature_names[i]: round(float(importance[i]), 6) for i in range(n)}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------


def _log_mlflow_run(
    task: str,
    algo_name: str,
    estimator,
    le,
    cv_metrics: dict,
    test_metrics: dict,
    feature_importance: dict | None,
    params: dict,
    data: TaskData,
    git_hash: str,
    training_duration_s: float,
    experiment_name: str,
    mlflow_uri: str,
    output_dir: Path,
) -> str:
    """Log a single algorithm's results to MLflow. Returns the run_id."""
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=algo_name) as run:
        # Tags
        mlflow.set_tags(
            {
                "task": task,
                "algorithm": algo_name,
                "git_commit": git_hash,
                "synthetic_data_disclaimer": (
                    "SYNTHETIC — metrics do not reflect real-world accuracy"
                ),
            }
        )

        # Parameters
        mlflow.log_params(
            {
                "task": task,
                "algorithm": algo_name,
                "train_rows": int(data.X_train_t.shape[0]),
                "test_rows": int(data.X_test_t.shape[0]),
                "n_features": int(data.X_train_t.shape[1]),
                "label_names": ",".join(data.label_names),
                "random_seed": params.get("random_state", ""),
                **{f"param_{k}": str(v) for k, v in params.items()},
            }
        )

        # CV metrics
        mlflow.log_metrics(
            {
                "cv_macro_f1_mean": cv_metrics["cv_macro_f1_mean"],
                "cv_macro_f1_std": cv_metrics["cv_macro_f1_std"],
                "cv_balanced_accuracy_mean": cv_metrics["cv_balanced_accuracy_mean"],
                "cv_balanced_accuracy_std": cv_metrics["cv_balanced_accuracy_std"],
            }
        )

        # Test metrics (scalars)
        scalar_test = {k: v for k, v in test_metrics.items() if isinstance(v, (int, float))}
        mlflow.log_metrics({f"test_{k}": v for k, v in scalar_test.items()})

        # Training duration
        mlflow.log_metric("training_duration_s", training_duration_s)

        # Classification report (text artifact)
        mlflow.log_text(
            test_metrics["classification_report"],
            "classification_report.txt",
        )

        # Confusion matrix (JSON artifact)
        mlflow.log_dict(
            {
                "labels": data.label_names,
                "matrix": test_metrics["confusion_matrix"],
            },
            "confusion_matrix.json",
        )

        # Per-class metrics (JSON artifact)
        mlflow.log_dict(test_metrics["per_class"], "per_class_metrics.json")

        # Feature importance (JSON artifact, if available)
        if feature_importance:
            mlflow.log_dict(feature_importance, "feature_importance.json")

        # Log model
        if algo_name == "xgboost":
            mlflow.xgboost.log_model(estimator, "model")
        else:
            mlflow.sklearn.log_model(estimator, "model")

        run_id = run.info.run_id

    return run_id


# ---------------------------------------------------------------------------
# Main training orchestrator per task
# ---------------------------------------------------------------------------


def train_task(
    task: str,
    processed_dir: Path,
    output_dir: Path,
    cfg: Config,
    mlflow_uri: str,
    quick: bool = False,
    n_jobs: int = 1,
) -> dict:
    """
    Train all algorithms for *task*, select the best, save artefacts.

    Parameters
    ----------
    task:
        ``"health"`` or ``"restoration"``.
    processed_dir:
        Directory with M4 outputs (CSVs + joblib preprocessors).
    output_dir:
        Directory for saved models and evaluation JSON.
    cfg:
        Config object.
    mlflow_uri:
        SQLite MLflow tracking URI, e.g. ``"sqlite:///artifacts/mlruns.db"``.
    quick:
        If True, use reduced hyperparameters and fewer CV folds for fast testing.
    n_jobs:
        Parallelism for sklearn cross_validate.

    Returns
    -------
    Dict with keys: task, best_model_name, best_cv_macro_f1, label_names,
    training_duration_s, best_model_path, evaluation_path, models (per-algo dicts).
    """
    if task not in _TASK_CFG:
        raise ValueError(f"Unknown task {task!r}; expected 'health' or 'restoration'")

    output_dir.mkdir(parents=True, exist_ok=True)
    git_hash = _get_git_hash()
    t_start = time.time()

    logger.info("=" * 60)
    logger.info("[%s] Starting training (quick=%s)", task, quick)

    # ── Load data ────────────────────────────────────────────────────────────
    data = load_task_data(task, processed_dir)

    # ── Build algorithms ─────────────────────────────────────────────────────
    algorithms = _get_algorithms(task, cfg, quick=quick)
    cv_folds = _QUICK_CV_FOLDS if quick else cfg.cv_folds
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=cfg.random_seed)

    experiment_name = (
        cfg.mlflow_experiment_health if task == "health" else cfg.mlflow_experiment_restoration
    )

    # ── Train and evaluate each algorithm ────────────────────────────────────
    model_results: dict[str, dict] = {}

    for algo_name, estimator in algorithms.items():
        logger.info("[%s] Training: %s", task, algo_name)
        algo_t_start = time.time()

        # Cross-validation
        if algo_name == "xgboost":
            cv_metrics = _run_cv_xgb(estimator, data.X_train_t, data.y_train, cv)
        else:
            cv_metrics = _run_cv_sklearn(estimator, data.X_train_t, data.y_train, cv, n_jobs)

        logger.info(
            "[%s][%s] CV macro-F1 = %.4f ± %.4f",
            task,
            algo_name,
            cv_metrics["cv_macro_f1_mean"],
            cv_metrics["cv_macro_f1_std"],
        )

        # Final fit on full training set
        if algo_name == "xgboost":
            estimator, le = _fit_xgb(clone(estimator), data.X_train_t, data.y_train)
            y_pred = le.inverse_transform(estimator.predict(data.X_test_t))
            y_proba = estimator.predict_proba(data.X_test_t)
        else:
            estimator = _fit_sklearn(clone(estimator), data.X_train_t, data.y_train)
            le = None
            y_pred = estimator.predict(data.X_test_t)
            y_proba = (
                estimator.predict_proba(data.X_test_t)
                if hasattr(estimator, "predict_proba")
                else None
            )

        # Evaluate
        test_metrics = compute_metrics(data.y_test.values, y_pred, y_proba, data.label_names)

        algo_duration = time.time() - algo_t_start
        logger.info(
            "[%s][%s] Test macro-F1=%.4f bal-acc=%.4f  (%.1fs)",
            task,
            algo_name,
            test_metrics["macro_f1"],
            test_metrics["balanced_accuracy"],
            algo_duration,
        )

        # Feature importance
        params = estimator.get_params() if hasattr(estimator, "get_params") else {}
        feat_imp = _get_feature_importance(algo_name, estimator, le, data.feature_names)

        # MLflow logging
        run_id = _log_mlflow_run(
            task=task,
            algo_name=algo_name,
            estimator=estimator,
            le=le,
            cv_metrics=cv_metrics,
            test_metrics=test_metrics,
            feature_importance=feat_imp,
            params=params,
            data=data,
            git_hash=git_hash,
            training_duration_s=algo_duration,
            experiment_name=experiment_name,
            mlflow_uri=mlflow_uri,
            output_dir=output_dir,
        )

        # Aggregate per-algorithm result
        model_results[algo_name] = {
            **cv_metrics,
            **{f"test_{k}": v for k, v in test_metrics.items() if isinstance(v, (int, float))},
            "test_macro_f1": test_metrics["macro_f1"],
            "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_precision": test_metrics["macro_precision"],
            "test_macro_recall": test_metrics["macro_recall"],
            "test_weighted_f1": test_metrics["weighted_f1"],
            "test_per_class": test_metrics["per_class"],
            "test_confusion_matrix": test_metrics["confusion_matrix"],
            "test_classification_report": test_metrics["classification_report"],
            "mlflow_run_id": run_id,
            "feature_importance": feat_imp,
            # Store fitted model object for saving best
            "_estimator": estimator,
            "_le": le,
        }

    # ── Model selection: primary = CV macro-F1, secondary = CV balanced accuracy ──
    best_name = max(
        model_results,
        key=lambda n: (
            model_results[n]["cv_macro_f1_mean"],
            model_results[n]["cv_balanced_accuracy_mean"],
        ),
    )
    best = model_results[best_name]
    logger.info("[%s] Best model: %s (CV macro-F1=%.4f)", task, best_name, best["cv_macro_f1_mean"])

    # ── Save best model ───────────────────────────────────────────────────────
    best_model_path = output_dir / f"best_model_{task}.joblib"
    save_payload = {
        "estimator": best["_estimator"],
        "label_encoder": best.get("_le"),
        "algo_name": best_name,
        "task": task,
        "label_names": data.label_names,
        "feature_names": data.feature_names,
    }
    joblib.dump(save_payload, best_model_path)
    logger.info("[%s] Best model saved to %s", task, best_model_path)

    # ── Save evaluation JSON ──────────────────────────────────────────────────
    # Strip non-serialisable objects before saving
    serialisable_results = {
        algo: {k: v for k, v in r.items() if not k.startswith("_")}
        for algo, r in model_results.items()
    }
    total_duration = time.time() - t_start
    eval_summary = {
        "task": task,
        "best_model_name": best_name,
        "best_cv_macro_f1": best["cv_macro_f1_mean"],
        "label_names": data.label_names,
        "training_duration_s": round(total_duration, 2),
        "models": serialisable_results,
        "synthetic_data_disclaimer": (
            "SYNTHETIC DATASET — metrics do not indicate real-world accuracy."
        ),
    }
    eval_path = output_dir / f"evaluation_{task}.json"
    with eval_path.open("w", encoding="utf-8") as fh:
        json.dump(eval_summary, fh, indent=2)
    logger.info("[%s] Evaluation summary saved to %s", task, eval_path)

    # Print comparison table
    print(
        format_comparison_table(
            task,
            {
                n: {
                    **r,
                    "test_accuracy": r["test_accuracy"],
                    "test_balanced_accuracy": r["test_balanced_accuracy"],
                    "test_macro_f1": r["test_macro_f1"],
                    "test_weighted_f1": r["test_weighted_f1"],
                }
                for n, r in model_results.items()
            },
        )
    )

    return {
        "task": task,
        "best_model_name": best_name,
        "best_cv_macro_f1": best["cv_macro_f1_mean"],
        "label_names": data.label_names,
        "training_duration_s": round(total_duration, 2),
        "best_model_path": str(best_model_path),
        "evaluation_path": str(eval_path),
        "models": serialisable_results,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """
    CLI entry point for the model training pipeline.

    Exit codes
    ----------
    0   Training completed successfully.
    1   Runtime error.
    2   Invalid arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train CoralSense classifiers. Results are logged to MLflow (SQLite backend)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task",
        choices=["health", "restoration", "all"],
        default="all",
        help="Which task to train.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Processed data directory (default: data/processed/ from config).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Model output directory (default: models/ from config).",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default=None,
        metavar="URI",
        help="MLflow tracking URI (default: sqlite:///artifacts/mlruns.db).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        default=False,
        help="Quick mode: reduced estimators and CV folds (for CI / testing).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        metavar="N",
        help="Parallelism for sklearn cross_validate (-1 = all CPUs).",
    )
    args = parser.parse_args()
    setup_logging("coralsense.train")

    cfg = get_config()
    processed_dir = args.processed_dir or cfg.paths.processed_data_dir
    output_dir = args.output_dir or cfg.paths.models_dir
    mlflow_uri = args.mlflow_uri or f"sqlite:///{cfg.paths.artifacts_dir / 'mlruns.db'}"

    tasks = ["health", "restoration"] if args.task == "all" else [args.task]

    for task in tasks:
        try:
            train_task(
                task=task,
                processed_dir=processed_dir,
                output_dir=output_dir,
                cfg=cfg,
                mlflow_uri=mlflow_uri,
                quick=args.quick,
                n_jobs=args.n_jobs,
            )
        except Exception as exc:
            logger.error("[%s] Training failed: %s", task, exc, exc_info=True)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
