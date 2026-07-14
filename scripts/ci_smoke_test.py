"""
scripts/ci_smoke_test.py — CI smoke test for CoralSense MLOps.

Exercises the core ML pipeline end-to-end using small isolated temporary data.
Safe to run in CI: uses only temporary directories and a throwaway MLflow DB;
never touches project data, real models, or the canonical registry.

Verifies
--------
- Synthetic data generation produces the expected number of rows.
- Preprocessing creates the required train/test splits.
- ``train_task`` completes in quick mode for both tasks.
- Required keys are present in the training result.
- ``best_cv_macro_f1`` is a finite float in [0, 1].
- Loaded model produces predictions that are valid class names.
- Probability rows sum to approximately 1.

Exit codes
----------
0  All checks passed.
1  One or more checks failed.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def _check_task(
    task: str,
    expected_classes: set[str],
    processed_dir: Path,
    models_dir: Path,
    mlflow_uri: str,
    cfg: object,
) -> list[str]:
    """Train one task in quick mode and return a list of error strings."""
    from src.models.train import train_task

    errors: list[str] = []

    print(f"  Training {task!r} (quick mode, n_jobs=1) ...")
    result = train_task(
        task=task,
        processed_dir=processed_dir,
        output_dir=models_dir,
        cfg=cfg,
        mlflow_uri=mlflow_uri,
        quick=True,
        n_jobs=1,
    )

    # Verify result structure
    required_keys = ("best_model_name", "best_cv_macro_f1", "label_names", "best_model_path")
    for key in required_keys:
        if key not in result:
            errors.append(f"{task}: missing key {key!r} in train_task result")

    if errors:
        return errors

    # Verify CV metric
    cv_f1: float = result["best_cv_macro_f1"]
    if not (0.0 <= cv_f1 <= 1.0):
        errors.append(f"{task}: best_cv_macro_f1={cv_f1!r} is outside [0, 1]")

    # Verify label names
    label_names: list[str] = result["label_names"]
    if not label_names:
        errors.append(f"{task}: label_names is empty")

    # Load saved model artifact
    model_path = Path(result["best_model_path"])
    if not model_path.exists():
        errors.append(f"{task}: model file not found at {model_path}")
        return errors

    payload = joblib.load(model_path)
    estimator = payload["estimator"]
    label_encoder = payload.get("label_encoder")

    # Load preprocessor (saved separately by preprocess stage)
    preprocessor_path = processed_dir / f"preprocessor_{task}.joblib"
    if not preprocessor_path.exists():
        errors.append(f"{task}: preprocessor not found at {preprocessor_path}")
        return errors
    preprocessor = joblib.load(preprocessor_path)

    # Load test features (already split by preprocess stage)
    x_test_path = processed_dir / f"X_test_{task}.csv"
    if not x_test_path.exists():
        errors.append(f"{task}: {x_test_path.name} not found in processed_dir")
        return errors

    X_test = pd.read_csv(x_test_path)
    X_t = preprocessor.transform(X_test)

    # Predictions
    y_pred_enc = estimator.predict(X_t)
    y_proba: np.ndarray = estimator.predict_proba(X_t)

    if label_encoder is not None:
        y_pred = label_encoder.inverse_transform(y_pred_enc)
    else:
        y_pred = y_pred_enc

    # All predicted classes must be valid
    invalid = set(y_pred) - expected_classes
    if invalid:
        errors.append(f"{task}: invalid predicted classes {invalid}")

    # Probability rows must sum to ~1
    row_sums = y_proba.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        bad = np.where(~np.isclose(row_sums, 1.0, atol=1e-5))[0]
        errors.append(f"{task}: {len(bad)} probability rows do not sum to 1")

    if not errors:
        print(
            f"    OK — best={result['best_model_name']!r}"
            f"  cv_macro_f1={cv_f1:.4f}"
            f"  predictions={len(y_pred)}"
            f"  classes={sorted(set(y_pred))}"
        )

    return errors


def main() -> int:
    from src.config import get_config, reset_config
    from src.data.generate_data import generate_observations
    from src.data.preprocess import run_preprocessing

    print("=" * 60)
    print("CoralSense CI Smoke Test")
    print("=" * 60)

    reset_config()
    cfg = get_config()

    all_errors: list[str] = []

    with tempfile.TemporaryDirectory(prefix="cs_ci_smoke_") as tmp_str:
        tmp = Path(tmp_str)
        raw_csv = tmp / "observations.csv"
        processed_dir = tmp / "processed"
        models_dir = tmp / "models"
        models_dir.mkdir()
        mlflow_uri = f"sqlite:///{tmp}/mlruns.db"

        # Step 1: generate
        print("\n[1/3] Generating 500 synthetic observations ...")
        df = generate_observations(n_samples=500, seed=42, cfg=cfg)
        df.to_csv(raw_csv, index=False)
        if len(df) != 500:
            all_errors.append(f"Expected 500 rows, got {len(df)}")
            print(f"  ERROR: {all_errors[-1]}")
        else:
            print(f"  OK — {len(df)} rows, {len(df.columns)} columns")

        # Step 2: preprocess
        print("\n[2/3] Preprocessing ...")
        try:
            run_preprocessing(raw_csv, processed_dir, cfg)
            print(f"  OK — processed dir: {processed_dir}")
        except Exception as exc:
            all_errors.append(f"Preprocessing failed: {exc}")
            print(f"  ERROR: {all_errors[-1]}")
            _report(all_errors)
            return 1

        # Step 3: train + verify both tasks
        print("\n[3/3] Training models ...")
        tasks = [
            ("health", {"healthy", "stressed", "bleached", "severely_degraded"}),
            ("restoration", {"suitable", "moderately_suitable", "unsuitable"}),
        ]
        for task, expected_classes in tasks:
            errs = _check_task(task, expected_classes, processed_dir, models_dir, mlflow_uri, cfg)
            all_errors.extend(errs)

    _report(all_errors)
    return 1 if all_errors else 0


def _report(errors: list[str]) -> None:
    print("\n" + "=" * 60)
    if errors:
        print("SMOKE TEST FAILED")
        for e in errors:
            print(f"  ERROR: {e}")
    else:
        print("SMOKE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
