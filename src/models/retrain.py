"""
src/models/retrain.py — Controlled retraining workflow for CoralSense.

SCIENTIFIC SAFETY RULE
-----------------------
The M11 shifted production window is unlabelled.  It MUST NOT be used for
supervised retraining.  A RETRAIN recommendation from the drift monitor means:
"Collect or provide validated labelled observations and begin a controlled
challenger-training process" — NOT "train immediately on the unlabelled window."

This module enforces that rule by:
  1. Requiring explicit target columns (reef_health / restoration_suitability).
  2. Validating the full Pandera schema before any training begins.
  3. Requiring an explicit --data-source declaration.
  4. Requiring either a RETRAIN drift recommendation or an explicit manual reason.

LABELLED INPUT CONTRACT
-----------------------
- Input CSV must pass the CoralObservationSchema (all columns, valid ranges).
- Both target columns must be present when task="all".
- All canonical class labels must appear in the input.
- Each class must have at least min_class_count samples.
- Non-finite feature values are rejected.
- Exact-duplicate rows (all columns) exceeding 50% of the dataset are rejected.
- A SHA-256 hash of the CSV content is recorded in the receipt.
- The source must be declared as "synthetic" or "field_labelled".
  Synthetic data must NOT be described as field-labelled.

TRAINING ISOLATION
------------------
Every challenger run uses a dedicated MLflow experiment (base name + suffix).
Champion aliases are NEVER touched here.  Promotion is a separate manual step
via src.models.promote.

USAGE
-----
  python -m src.models.retrain \\
    --input labelled_data.csv \\
    --task all \\
    --data-source synthetic \\
    --drift-summary reports/drift_summary.json \\
    --quick

  python -m src.models.retrain \\
    --input labelled_data.csv \\
    --task health \\
    --data-source field_labelled \\
    --reason "Post-bleaching-event field survey — July 2026" \\
    --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import Config, get_config, setup_logging
from src.data.validate import validate_dataframe
from src.features.build_features import (
    CATEGORICAL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    add_derived_features,
)
from src.models.evaluate import compute_metrics
from src.models.train import (
    _QUICK_CV_FOLDS,
    _QUICK_OVERRIDES,
    _fit_sklearn,
    _fit_xgb,
    _get_feature_importance,
    _run_cv_sklearn,
    _run_cv_xgb,
)

logger = logging.getLogger(__name__)

_VALID_TASKS = ("health", "restoration")
_VALID_SOURCES = ("synthetic", "field_labelled")

_SYNTHETIC_DISCLAIMER = (
    "Metrics reflect performance on synthetic data only. "
    "Do not use to infer real-world conservation accuracy."
)

_RETRAINING_EXPERIMENT_TAG = "retraining_run"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class InputValidationResult:
    """Outcome of validating a labelled input CSV."""

    valid: bool
    path: str
    sha256: str
    n_rows: int
    tasks_validated: list[str]
    class_distributions: dict[str, dict[str, int]]
    failures: list[str]
    timestamp: str
    data_source: str


@dataclass
class DriftContext:
    """Drift summary context attached to a retraining run."""

    path: str
    sha256: str
    recommendation: str
    drifted_features: list[str]
    monitoring_timestamp: str | None


@dataclass
class ChallengerRegistration:
    """Details of a challenger model registered after retraining."""

    task: str
    registered_model_name: str
    version: str
    run_id: str
    algo_name: str
    cv_macro_f1: float
    cv_balanced_accuracy: float
    comparison_result: str  # "pending" until compare.py runs
    champion_alias_changed: bool = False  # always False from retrain


@dataclass
class RetrainingReceipt:
    """Complete audit record for one retraining run."""

    receipt_id: str
    timestamp: str
    input_path: str
    input_sha256: str
    n_rows: int
    data_source: str
    tasks: list[str]
    class_distributions: dict[str, dict[str, int]]
    drift_context: dict[str, Any] | None
    manual_reason: str | None
    git_commit: str
    quick_mode: bool
    dry_run: bool
    challengers: list[dict[str, Any]]
    synthetic_data_disclaimer: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_hash() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return r.stdout.strip()
    except Exception:
        return "unknown"


def hash_file(path: Path) -> str:
    """Return hex SHA-256 of a file's byte content."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_dataframe(df: pd.DataFrame) -> str:
    """Return hex SHA-256 of a DataFrame serialised as canonical CSV bytes."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _target_col(task: str) -> str:
    return "reef_health" if task == "health" else "restoration_suitability"


def _required_classes(task: str, cfg: Config) -> list[str]:
    return cfg.health_classes if task == "health" else cfg.restoration_classes


def validate_input_dataframe(
    df: pd.DataFrame,
    tasks: list[str],
    data_source: str,
    cfg: Config,
) -> InputValidationResult:
    """
    Validate a labelled DataFrame against all retraining input requirements.

    Checks
    ------
    1. Pandera schema passes.
    2. Required target column(s) present and non-null.
    3. All canonical class labels present per task.
    4. Minimum per-class sample count.
    5. No non-finite values in numeric feature columns.
    6. Not more than 50% exact-duplicate rows.
    7. Minimum total row count.
    8. data_source is a recognised value.

    Returns
    -------
    InputValidationResult with valid=True/False and failure list.
    """
    rt = cfg.retraining
    min_class_count = int(rt.get("min_class_count", 5))
    min_train_rows = int(rt.get("min_train_rows", 200))
    holdout_size = float(rt.get("holdout_size", 0.20))
    min_total = int(min_train_rows / (1.0 - holdout_size))

    sha256 = hash_dataframe(df)
    timestamp = datetime.now(UTC).isoformat()
    failures: list[str] = []

    # Source declaration
    if data_source not in _VALID_SOURCES:
        failures.append(f"Unknown data_source '{data_source}'. Must be one of {_VALID_SOURCES}.")

    # Pandera schema
    is_valid, schema_failures = validate_dataframe(df, lazy=True)
    if not is_valid:
        n = len(schema_failures) if schema_failures is not None else "?"
        failures.append(f"Pandera schema validation failed ({n} failure(s)).")

    # Minimum rows
    if len(df) < min_total:
        failures.append(
            f"Dataset has only {len(df)} rows; minimum required is {min_total} "
            f"(to yield {min_train_rows} training rows after {holdout_size:.0%} holdout)."
        )

    # Duplicate check — reject if >50% rows are exact duplicates
    n_dup = df.duplicated().sum()
    if n_dup > len(df) * 0.50:
        failures.append(
            f"{n_dup}/{len(df)} rows are exact duplicates (>{50}%); "
            "dataset appears to be near-duplicate only."
        )

    # Non-finite values in numeric columns
    num_cols = [c for c in NUMERIC_FEATURE_COLUMNS if c in df.columns]
    if num_cols:
        inf_mask = ~np.isfinite(df[num_cols].values)
        n_inf = int(inf_mask.sum())
        if n_inf > 0:
            failures.append(
                f"{n_inf} non-finite value(s) detected in numeric feature columns. "
                "Replace or remove before retraining."
            )

    # Per-task label validation
    class_distributions: dict[str, dict[str, int]] = {}
    for task in tasks:
        col = _target_col(task)
        if col not in df.columns:
            failures.append(
                f"Target column '{col}' is missing. "
                "Unlabelled production windows cannot be used for retraining."
            )
            continue
        if df[col].isna().any():
            failures.append(f"Target column '{col}' has null values.")
            continue
        present = set(df[col].unique())
        required = set(_required_classes(task, cfg))
        missing_cls = required - present
        if missing_cls:
            failures.append(
                f"[{task}] Missing classes: {sorted(missing_cls)}. "
                "All canonical classes must appear in the retraining dataset."
            )
        counts = df[col].value_counts().to_dict()
        class_distributions[task] = {str(k): int(v) for k, v in counts.items()}
        for cls, cnt in counts.items():
            if cls in required and cnt < min_class_count:
                failures.append(
                    f"[{task}] Class '{cls}' has only {cnt} samples (minimum {min_class_count})."
                )

    return InputValidationResult(
        valid=len(failures) == 0,
        path="<dataframe>",
        sha256=sha256,
        n_rows=len(df),
        tasks_validated=list(tasks),
        class_distributions=class_distributions,
        failures=failures,
        timestamp=timestamp,
        data_source=data_source,
    )


# ---------------------------------------------------------------------------
# Drift summary loader
# ---------------------------------------------------------------------------


def load_drift_summary(drift_summary_path: Path) -> DriftContext:
    """Load and hash a drift summary JSON, return a DriftContext."""
    if not drift_summary_path.exists():
        raise FileNotFoundError(f"Drift summary not found: {drift_summary_path}")
    sha256 = hash_file(drift_summary_path)
    with drift_summary_path.open(encoding="utf-8") as fh:
        d = json.load(fh)
    return DriftContext(
        path=str(drift_summary_path),
        sha256=sha256,
        recommendation=d.get("recommendation", "UNKNOWN"),
        drifted_features=d.get("drifted_features", []),
        monitoring_timestamp=d.get("timestamp"),
    )


def check_retraining_permission(
    drift_context: DriftContext | None,
    manual_reason: str | None,
) -> tuple[bool, str]:
    """
    Determine whether controlled retraining is permitted.

    Permit when:
      - drift_context is provided and recommendation starts with "RETRAIN", OR
      - manual_reason is a non-empty string.

    Returns (permitted: bool, rationale: str).
    """
    if drift_context is not None:
        rec = drift_context.recommendation.upper()
        if rec.startswith("RETRAIN"):
            return True, f"Drift report recommends retraining: {drift_context.recommendation}"
    if manual_reason and manual_reason.strip():
        return True, f"Manual reason provided: {manual_reason.strip()}"
    return False, (
        "Retraining not permitted: drift recommendation is not RETRAIN "
        "and no manual reason was supplied. "
        "Provide --drift-summary with a RETRAIN recommendation or --reason."
    )


# ---------------------------------------------------------------------------
# Preprocessing for retraining
# ---------------------------------------------------------------------------


@dataclass
class RetrainingTaskData:
    """Splits and transformed data for one retraining task."""

    task: str
    X_train_raw: pd.DataFrame
    X_holdout_raw: pd.DataFrame
    X_train_t: np.ndarray
    X_holdout_t: np.ndarray
    y_train: pd.Series
    y_holdout: pd.Series
    label_names: list[str]
    feature_names: list[str]
    preprocessor: ColumnTransformer


def _build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ohe",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", num_pipeline, numeric_cols),
            ("cat", cat_pipeline, categorical_cols),
        ]
    )


def split_and_preprocess(
    df: pd.DataFrame,
    task: str,
    cfg: Config,
    seed: int,
) -> RetrainingTaskData:
    """
    Add derived features, stratified-split into train/holdout, fit preprocessor
    on training portion only (never on holdout).

    The holdout is a frozen evaluation set — it is NEVER used in fitting.
    """
    rt = cfg.retraining
    holdout_size = float(rt.get("holdout_size", 0.20))

    target_col = _target_col(task)
    label_names = _required_classes(task, cfg)

    # Derive features
    df_feat = add_derived_features(df.copy())

    # Feature columns — numeric includes derived; categorical is ["region"]
    num_cols = [c for c in NUMERIC_FEATURE_COLUMNS if c in df_feat.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURE_COLUMNS if c in df_feat.columns]
    all_feat_cols = num_cols + cat_cols

    X = df_feat[all_feat_cols].copy()
    y = df_feat[target_col].copy()

    X_train_raw, X_holdout_raw, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=holdout_size,
        random_state=seed,
        stratify=y,
    )

    # Fit preprocessor on training split ONLY
    preprocessor = _build_preprocessor(num_cols, cat_cols)
    preprocessor.fit(X_train_raw)

    X_train_t = preprocessor.transform(X_train_raw)
    X_holdout_t = preprocessor.transform(X_holdout_raw)
    feature_names = list(preprocessor.get_feature_names_out())

    logger.info(
        "[%s] Split: train=%d holdout=%d features=%d",
        task,
        len(y_train),
        len(y_holdout),
        X_train_t.shape[1],
    )

    return RetrainingTaskData(
        task=task,
        X_train_raw=X_train_raw,
        X_holdout_raw=X_holdout_raw,
        X_train_t=X_train_t,
        X_holdout_t=X_holdout_t,
        y_train=y_train,
        y_holdout=y_holdout,
        label_names=label_names,
        feature_names=feature_names,
        preprocessor=preprocessor,
    )


# ---------------------------------------------------------------------------
# Challenger training
# ---------------------------------------------------------------------------


def _get_challenger_algorithms(
    task: str,
    cfg: Config,
    quick: bool,
) -> dict[str, object]:
    """Build unfitted estimators from params.yaml, with optional quick overrides."""
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

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


def train_challenger(
    task_data: RetrainingTaskData,
    cfg: Config,
    mlflow_uri: str,
    quick: bool,
    run_tags: dict[str, str],
) -> dict:
    """
    Train all algorithms on the retraining split, select best by CV macro-F1,
    log to a dedicated MLflow experiment, save best model to a temp path.

    Champion aliases are NEVER modified here.

    Returns
    -------
    Dict with best algo name, metrics, run_id, and saved estimator objects.
    """
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost

    rt = cfg.retraining
    exp_suffix = rt.get("experiment_suffix", "_retraining")
    base_exp = (
        cfg.mlflow_experiment_health
        if task_data.task == "health"
        else cfg.mlflow_experiment_restoration
    )
    experiment_name = base_exp + exp_suffix

    algorithms = _get_challenger_algorithms(task_data.task, cfg, quick=quick)
    cv_folds = _QUICK_CV_FOLDS if quick else cfg.cv_folds
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=cfg.random_seed)

    model_results: dict[str, dict] = {}

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    for algo_name, estimator in algorithms.items():
        t0 = time.time()
        logger.info("[%s] Challenger training: %s", task_data.task, algo_name)

        # Cross-validation
        from xgboost import XGBClassifier

        if isinstance(estimator, XGBClassifier):
            cv_metrics = _run_cv_xgb(estimator, task_data.X_train_t, task_data.y_train, cv)
        else:
            cv_metrics = _run_cv_sklearn(
                estimator, task_data.X_train_t, task_data.y_train, cv, n_jobs=1
            )

        # Final fit on full training split
        if isinstance(estimator, XGBClassifier):
            fitted_est, le = _fit_xgb(clone(estimator), task_data.X_train_t, task_data.y_train)
            y_pred_holdout = le.inverse_transform(fitted_est.predict(task_data.X_holdout_t))
            y_proba_holdout = fitted_est.predict_proba(task_data.X_holdout_t)
        else:
            fitted_est = _fit_sklearn(clone(estimator), task_data.X_train_t, task_data.y_train)
            le = None
            y_pred_holdout = fitted_est.predict(task_data.X_holdout_t)
            y_proba_holdout = (
                fitted_est.predict_proba(task_data.X_holdout_t)
                if hasattr(fitted_est, "predict_proba")
                else None
            )

        holdout_metrics = compute_metrics(
            task_data.y_holdout.values,
            y_pred_holdout,
            y_proba_holdout,
            task_data.label_names,
        )
        duration = time.time() - t0
        feat_imp = _get_feature_importance(algo_name, fitted_est, le, task_data.feature_names)
        params = fitted_est.get_params() if hasattr(fitted_est, "get_params") else {}

        # MLflow logging
        with mlflow.start_run(run_name=f"challenger_{algo_name}") as run:
            mlflow.set_tags(
                {
                    "task": task_data.task,
                    "algorithm": algo_name,
                    _RETRAINING_EXPERIMENT_TAG: "true",
                    **run_tags,
                }
            )
            mlflow.log_params(
                {
                    "task": task_data.task,
                    "algorithm": algo_name,
                    "train_rows": int(task_data.X_train_t.shape[0]),
                    "holdout_rows": int(task_data.X_holdout_t.shape[0]),
                    "n_features": int(task_data.X_train_t.shape[1]),
                    "label_names": ",".join(task_data.label_names),
                    "feature_names": ",".join(task_data.feature_names),
                    **{f"param_{k}": str(v) for k, v in params.items()},
                }
            )
            mlflow.log_metrics(
                {
                    "cv_macro_f1_mean": cv_metrics["cv_macro_f1_mean"],
                    "cv_macro_f1_std": cv_metrics["cv_macro_f1_std"],
                    "cv_balanced_accuracy_mean": cv_metrics["cv_balanced_accuracy_mean"],
                    "cv_balanced_accuracy_std": cv_metrics["cv_balanced_accuracy_std"],
                    "holdout_macro_f1": holdout_metrics["macro_f1"],
                    "holdout_balanced_accuracy": holdout_metrics["balanced_accuracy"],
                    "holdout_accuracy": holdout_metrics["accuracy"],
                    "holdout_macro_recall": holdout_metrics["macro_recall"],
                    "holdout_weighted_f1": holdout_metrics["weighted_f1"],
                    "training_duration_s": duration,
                }
            )
            mlflow.log_text(
                holdout_metrics["classification_report"], "holdout_classification_report.txt"
            )
            mlflow.log_dict(holdout_metrics["per_class"], "holdout_per_class_metrics.json")
            if feat_imp:
                mlflow.log_dict(feat_imp, "feature_importance.json")
            if isinstance(fitted_est, XGBClassifier):
                mlflow.xgboost.log_model(fitted_est, "model")
            else:
                mlflow.sklearn.log_model(fitted_est, "model")
            run_id = run.info.run_id

        logger.info(
            "[%s][%s] CV macro-F1=%.4f holdout macro-F1=%.4f (%.1fs)",
            task_data.task,
            algo_name,
            cv_metrics["cv_macro_f1_mean"],
            holdout_metrics["macro_f1"],
            duration,
        )

        model_results[algo_name] = {
            **cv_metrics,
            "holdout_macro_f1": holdout_metrics["macro_f1"],
            "holdout_balanced_accuracy": holdout_metrics["balanced_accuracy"],
            "holdout_per_class": holdout_metrics["per_class"],
            "mlflow_run_id": run_id,
            "_estimator": fitted_est,
            "_le": le,
            "_feat_imp": feat_imp,
        }

    # Select best by CV macro-F1
    best_name = max(
        model_results,
        key=lambda n: (
            model_results[n]["cv_macro_f1_mean"],
            model_results[n]["cv_balanced_accuracy_mean"],
        ),
    )
    best = model_results[best_name]
    logger.info(
        "[%s] Best challenger: %s (CV macro-F1=%.4f)",
        task_data.task,
        best_name,
        best["cv_macro_f1_mean"],
    )
    return {
        "task": task_data.task,
        "best_name": best_name,
        "best": best,
        "all_results": model_results,
        "experiment_name": experiment_name,
    }


# ---------------------------------------------------------------------------
# Challenger registration
# ---------------------------------------------------------------------------


def register_challenger(
    task: str,
    train_result: dict,
    input_sha256: str,
    data_source: str,
    drift_context: DriftContext | None,
    manual_reason: str | None,
    cfg: Config,
    mlflow_uri: str,
) -> ChallengerRegistration:
    """
    Register the best challenger as a new model version.

    - Tags version with challenger, data hash, drift hash, reason, disclaimer.
    - Does NOT assign the champion alias.
    - Prevents duplicate registration: raises if same run_id already registered.
    """
    from mlflow import MlflowClient

    best_name = train_result["best_name"]
    best = train_result["best"]
    run_id: str = best["mlflow_run_id"]
    experiment_name: str = train_result["experiment_name"]
    model_name = (
        cfg.mlflow_registered_health if task == "health" else cfg.mlflow_registered_restoration
    )

    client = MlflowClient(tracking_uri=mlflow_uri)

    # Duplicate-registration guard: check if this run_id is already registered
    try:
        existing = client.search_model_versions(f"name='{model_name}'")
        for mv in existing:
            raw_tags = mv.tags or {}
            tags_dict = (
                raw_tags if isinstance(raw_tags, dict) else {t.key: t.value for t in raw_tags}
            )
            if tags_dict.get("challenger_run_id") == run_id:
                raise RuntimeError(
                    f"Challenger run {run_id} is already registered as "
                    f"'{model_name}' version {mv.version}. "
                    "Duplicate registration prevented."
                )
    except RuntimeError:
        raise
    except Exception:
        pass  # search failed — proceed

    # Locate logged model artifact
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(
            f"MLflow experiment '{experiment_name}' not found. "
            "Ensure run_retraining was called first."
        )
    artifact_location: str | None = None
    try:
        logged_models = client.search_logged_models(experiment_ids=[exp.experiment_id])
        for lm in logged_models:
            if lm.source_run_id == run_id:
                artifact_location = lm.artifact_location
                break
    except Exception as exc:
        logger.warning("Could not search logged models: %s", exc)

    if artifact_location is None:
        raise RuntimeError(f"Could not find LoggedModel artifact for run {run_id}.")

    # Ensure registered model exists
    try:
        client.get_registered_model(model_name)
    except Exception:
        client.create_registered_model(name=model_name)

    cv_f1 = best["cv_macro_f1_mean"]
    cv_bal = best["cv_balanced_accuracy_mean"]
    gate_passed = _gate_passed(task, cv_f1, cv_bal, cfg)

    tags: dict[str, str] = {
        "role": "challenger",
        "task": task,
        "algo_name": best_name,
        "challenger_run_id": run_id,
        "input_sha256": input_sha256,
        "data_source": data_source,
        "cv_macro_f1": f"{cv_f1:.6f}",
        "cv_balanced_accuracy": f"{cv_bal:.6f}",
        "holdout_macro_f1": f"{best['holdout_macro_f1']:.6f}",
        "holdout_balanced_accuracy": f"{best['holdout_balanced_accuracy']:.6f}",
        "quality_gate_passed": str(gate_passed),
        "comparison_result": "pending",
        "synthetic_data_status": data_source,
        "synthetic_data_disclaimer": _SYNTHETIC_DISCLAIMER if data_source == "synthetic" else "",
        "drift_report_sha256": drift_context.sha256 if drift_context else "",
        "retraining_reason": (
            drift_context.recommendation if drift_context else (manual_reason or "manual")
        ),
        "registration_timestamp": datetime.now(UTC).isoformat(),
    }

    mv = client.create_model_version(
        name=model_name,
        source=artifact_location,
        run_id=run_id,
        tags=tags,
        description=(
            f"CHALLENGER | algo={best_name} | CV macro-F1={cv_f1:.4f} | "
            f"source={data_source} | gate={'PASS' if gate_passed else 'FAIL'}"
        ),
    )
    version_str = str(mv.version)
    logger.info(
        "[%s] Challenger registered as '%s' version %s (run=%s, cv_f1=%.4f, gate=%s)",
        task,
        model_name,
        version_str,
        run_id[:8],
        cv_f1,
        "PASS" if gate_passed else "FAIL",
    )

    return ChallengerRegistration(
        task=task,
        registered_model_name=model_name,
        version=version_str,
        run_id=run_id,
        algo_name=best_name,
        cv_macro_f1=cv_f1,
        cv_balanced_accuracy=cv_bal,
        comparison_result="pending",
        champion_alias_changed=False,
    )


def _gate_passed(task: str, cv_f1: float, cv_bal: float, cfg: Config) -> bool:
    gates = cfg.quality_gates.get(task, {})
    min_f1 = float(gates.get("min_cv_macro_f1", 0.0))
    min_bal = float(gates.get("min_cv_balanced_accuracy", 0.0))
    return cv_f1 >= min_f1 and cv_bal >= min_bal


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_retraining(
    input_path: Path,
    tasks: list[str],
    data_source: str,
    drift_summary_path: Path | None = None,
    manual_reason: str | None = None,
    mlflow_uri: str | None = None,
    output_dir: Path | None = None,
    cfg: Config | None = None,
    quick: bool = False,
    dry_run: bool = False,
) -> RetrainingReceipt:
    """
    Full retraining orchestration for one or more tasks.

    Parameters
    ----------
    input_path:
        Path to a validated labelled CSV (must include target columns).
    tasks:
        List of tasks to retrain: ["health"], ["restoration"], or ["health","restoration"].
    data_source:
        Either "synthetic" or "field_labelled".
    drift_summary_path:
        Optional path to reports/drift_summary.json to attach drift context.
    manual_reason:
        If no drift summary is provided, supply a non-empty documented reason.
    mlflow_uri:
        SQLite tracking URI (defaults to config value).
    output_dir:
        Directory to save receipts.
    cfg:
        Config instance (defaults to get_config()).
    quick:
        Use reduced hyperparameters and CV folds (for CI / tests).
    dry_run:
        Validate + preprocess but do not train, register, or write receipts.
        Does NOT mutate the MLflow database.

    Returns
    -------
    RetrainingReceipt with full audit record.
    """
    cfg = cfg or get_config()
    mlflow_uri = mlflow_uri or cfg.mlflow_tracking_uri
    output_dir = output_dir or cfg.paths.reports_dir

    for task in tasks:
        if task not in _VALID_TASKS:
            raise ValueError(f"Invalid task '{task}'. Must be one of {_VALID_TASKS}.")

    receipt_id = f"retrain_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    git_commit = _git_hash()

    # ── 1. Load input ────────────────────────────────────────────────────────
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    input_sha256 = hash_file(input_path)
    df = pd.read_csv(input_path)
    logger.info("Loaded %d rows from %s (sha256=%s)", len(df), input_path, input_sha256[:16])

    # ── 2. Validate input ────────────────────────────────────────────────────
    validation = validate_input_dataframe(df, tasks, data_source, cfg)
    if not validation.valid:
        raise ValueError(
            "Input validation failed:\n" + "\n".join(f"  - {f}" for f in validation.failures)
        )
    logger.info("Input validation PASSED for tasks=%s", tasks)

    # ── 3. Load drift context (optional) ────────────────────────────────────
    drift_context: DriftContext | None = None
    if drift_summary_path is not None:
        drift_context = load_drift_summary(drift_summary_path)
        logger.info(
            "Drift summary loaded: recommendation=%s, drifted=%s",
            drift_context.recommendation,
            drift_context.drifted_features,
        )

    # ── 4. Check permission ──────────────────────────────────────────────────
    permitted, rationale = check_retraining_permission(drift_context, manual_reason)
    if not permitted:
        raise PermissionError(rationale)
    logger.info("Retraining permitted: %s", rationale)

    if dry_run:
        logger.info("DRY-RUN mode: validation passed, stopping before training.")
        return RetrainingReceipt(
            receipt_id=receipt_id,
            timestamp=datetime.now(UTC).isoformat(),
            input_path=str(input_path),
            input_sha256=input_sha256,
            n_rows=len(df),
            data_source=data_source,
            tasks=tasks,
            class_distributions=validation.class_distributions,
            drift_context=asdict(drift_context) if drift_context else None,
            manual_reason=manual_reason,
            git_commit=git_commit,
            quick_mode=quick,
            dry_run=True,
            challengers=[],
            synthetic_data_disclaimer=_SYNTHETIC_DISCLAIMER if data_source == "synthetic" else "",
        )

    run_tags = {
        "retraining_run": "true",
        "input_sha256": input_sha256,
        "data_source": data_source,
        "git_commit": git_commit,
        "retraining_reason": rationale[:250],
        **({"drift_sha256": drift_context.sha256} if drift_context else {}),
    }

    challengers: list[dict[str, Any]] = []

    for task in tasks:
        logger.info("=" * 60)
        logger.info("[%s] Starting challenger training", task)

        # ── 5. Preprocess (fit on train split only) ──────────────────────────
        task_data = split_and_preprocess(df, task, cfg, cfg.random_seed)

        # ── 6. Train challenger ──────────────────────────────────────────────
        train_result = train_challenger(task_data, cfg, mlflow_uri, quick, run_tags)

        # ── 7. Register challenger (no champion alias) ───────────────────────
        reg = register_challenger(
            task=task,
            train_result=train_result,
            input_sha256=input_sha256,
            data_source=data_source,
            drift_context=drift_context,
            manual_reason=manual_reason,
            cfg=cfg,
            mlflow_uri=mlflow_uri,
        )
        challengers.append(
            {
                "task": task,
                "registered_model_name": reg.registered_model_name,
                "version": reg.version,
                "run_id": reg.run_id,
                "algo_name": reg.algo_name,
                "cv_macro_f1": reg.cv_macro_f1,
                "cv_balanced_accuracy": reg.cv_balanced_accuracy,
                "comparison_result": reg.comparison_result,
                "champion_alias_changed": reg.champion_alias_changed,
            }
        )

    # ── 8. Write receipt ─────────────────────────────────────────────────────
    receipt = RetrainingReceipt(
        receipt_id=receipt_id,
        timestamp=datetime.now(UTC).isoformat(),
        input_path=str(input_path),
        input_sha256=input_sha256,
        n_rows=len(df),
        data_source=data_source,
        tasks=tasks,
        class_distributions=validation.class_distributions,
        drift_context=asdict(drift_context) if drift_context else None,
        manual_reason=manual_reason,
        git_commit=git_commit,
        quick_mode=quick,
        dry_run=False,
        challengers=challengers,
        synthetic_data_disclaimer=_SYNTHETIC_DISCLAIMER if data_source == "synthetic" else "",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / f"{receipt_id}.json"
    with receipt_path.open("w", encoding="utf-8") as fh:
        json.dump(asdict(receipt), fh, indent=2)
    logger.info("Retraining receipt saved to %s", receipt_path)

    return receipt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point for controlled challenger retraining.

    Exit codes
    ----------
    0   Success.
    1   Runtime error (training failed, validation failed).
    2   Invalid arguments.
    3   Permission denied (no drift recommendation and no manual reason).
    """
    setup_logging(__name__)
    parser = argparse.ArgumentParser(
        description="CoralSense controlled challenger retraining.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        metavar="CSV",
        help="Path to labelled observations CSV (must include target columns).",
    )
    parser.add_argument(
        "--task",
        choices=["health", "restoration", "all"],
        default="all",
        help="Which task(s) to retrain.",
    )
    parser.add_argument(
        "--data-source",
        required=True,
        choices=_VALID_SOURCES,
        dest="data_source",
        help="Declare whether the data is 'synthetic' or 'field_labelled'.",
    )
    parser.add_argument(
        "--drift-summary",
        type=Path,
        default=None,
        dest="drift_summary",
        metavar="JSON",
        help="Path to drift_summary.json (provides RETRAIN recommendation context).",
    )
    parser.add_argument(
        "--reason",
        default=None,
        metavar="TEXT",
        help="Documented manual reason for retraining (required if no drift summary).",
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
        help="Directory for receipt JSON output (default: reports/).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        default=False,
        help="Quick mode: reduced estimators/CV folds (for CI / tests).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        dest="dry_run",
        help="Validate input and check permission only — do not train or register.",
    )
    args = parser.parse_args(argv)

    tasks = ["health", "restoration"] if args.task == "all" else [args.task]

    try:
        cfg = get_config()
        receipt = run_retraining(
            input_path=args.input,
            tasks=tasks,
            data_source=args.data_source,
            drift_summary_path=args.drift_summary,
            manual_reason=args.reason,
            mlflow_uri=args.mlflow_uri,
            output_dir=args.output_dir,
            cfg=cfg,
            quick=args.quick,
            dry_run=args.dry_run,
        )
        if receipt.dry_run:
            print(f"DRY-RUN: validation PASSED for tasks={tasks}, data_source={args.data_source}")
            print(f"Input rows: {receipt.n_rows}  SHA-256: {receipt.input_sha256[:16]}...")
        else:
            print(f"Retraining complete. Receipt ID: {receipt.receipt_id}")
            for ch in receipt.challengers:
                print(
                    f"  [{ch['task']}] {ch['registered_model_name']} v{ch['version']} "
                    f"algo={ch['algo_name']} cv_f1={ch['cv_macro_f1']:.4f}"
                )
        return 0
    except PermissionError as exc:
        logger.error("Permission denied: %s", exc)
        return 3
    except ValueError as exc:
        logger.error("Validation error: %s", exc)
        return 1
    except Exception as exc:
        logger.error("Retraining failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
