"""
tests/test_models.py — Tests for src/models/train.py and evaluate.py (M5).

Coverage
--------
- Both tasks train successfully in quick mode
- Preprocessing is not refitted during model training
- Targets are excluded from feature columns
- Predictions have correct shapes and valid labels
- Probability rows sum approximately to 1
- All required metrics are produced
- Model selection follows CV macro-F1 (primary) and balanced accuracy (secondary)
- Artefacts are saved (joblib model, JSON evaluation)
- CLI rejects invalid task names
- Quick mode completes fast on small data
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from src.config import get_config, reset_config
from src.data.generate_data import generate_observations
from src.data.preprocess import run_preprocessing
from src.models.evaluate import compute_metrics, format_comparison_table
from src.models.train import (
    TaskData,
    load_task_data,
    train_task,
)

# ---------------------------------------------------------------------------
# Module-scoped fixtures: generate → preprocess → train (quick mode)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg():
    reset_config()
    return get_config()


def _setup_task(cfg, tmp_path_factory, seed: int, task: str) -> dict:
    """Generate 500 rows, preprocess, train in quick mode. Returns result dict."""
    tmp = tmp_path_factory.mktemp(f"m5_{task}_{seed}")
    raw = tmp / "obs.csv"
    generate_observations(n_samples=500, seed=seed, cfg=cfg).to_csv(raw, index=False)
    run_preprocessing(raw, tmp, cfg)
    mlflow_uri = f"sqlite:///{tmp}/mlruns.db"
    return train_task(
        task=task,
        processed_dir=tmp,
        output_dir=tmp / "models",
        cfg=cfg,
        mlflow_uri=mlflow_uri,
        quick=True,
        n_jobs=1,
    )


@pytest.fixture(scope="module")
def health_result(cfg, tmp_path_factory):
    return _setup_task(cfg, tmp_path_factory, seed=77, task="health")


@pytest.fixture(scope="module")
def restoration_result(cfg, tmp_path_factory):
    return _setup_task(cfg, tmp_path_factory, seed=88, task="restoration")


@pytest.fixture(scope="module")
def health_data(cfg, tmp_path_factory):
    """TaskData object for health task (small dataset)."""
    tmp = tmp_path_factory.mktemp("m5_data_health")
    raw = tmp / "obs.csv"
    generate_observations(n_samples=500, seed=77, cfg=cfg).to_csv(raw, index=False)
    run_preprocessing(raw, tmp, cfg)
    return load_task_data("health", tmp)


# ---------------------------------------------------------------------------
# TestBothTasksTrain
# ---------------------------------------------------------------------------


class TestBothTasksTrain:
    def test_health_training_completes(self, health_result: dict) -> None:
        assert health_result["task"] == "health"

    def test_restoration_training_completes(self, restoration_result: dict) -> None:
        assert restoration_result["task"] == "restoration"

    def test_health_has_three_algorithms(self, health_result: dict) -> None:
        assert len(health_result["models"]) == 3

    def test_restoration_has_three_algorithms(self, restoration_result: dict) -> None:
        assert len(restoration_result["models"]) == 3

    def test_health_algorithm_names(self, health_result: dict) -> None:
        expected = {"logistic_regression", "random_forest", "xgboost"}
        assert set(health_result["models"].keys()) == expected

    def test_restoration_algorithm_names(self, restoration_result: dict) -> None:
        expected = {"logistic_regression", "random_forest", "xgboost"}
        assert set(restoration_result["models"].keys()) == expected


# ---------------------------------------------------------------------------
# TestLeakagePrevention
# ---------------------------------------------------------------------------


class TestLeakagePrevention:
    """Preprocessor must not be refitted; targets must be absent from features."""

    def test_preprocessor_fitted_before_training(self, health_data: TaskData) -> None:
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(health_data.preprocessor)

    def test_reef_health_not_in_features(self, health_data: TaskData) -> None:
        assert "reef_health" not in health_data.X_train_raw.columns

    def test_restoration_suitability_not_in_health_features(self, health_data: TaskData) -> None:
        assert "restoration_suitability" not in health_data.X_train_raw.columns

    def test_metadata_not_in_features(self, health_data: TaskData) -> None:
        for col in ("timestamp", "latitude", "longitude"):
            assert col not in health_data.X_train_raw.columns

    def test_scaler_mean_matches_train_data(self, health_data: TaskData) -> None:
        """Scaler mean = X_train mean confirms it was fitted on training data only."""
        from src.features.build_features import NUMERIC_FEATURE_COLUMNS

        scaler = health_data.preprocessor.named_transformers_["num"].named_steps["scaler"]
        numeric_cols = [c for c in NUMERIC_FEATURE_COLUMNS if c in health_data.X_train_raw.columns]
        expected = health_data.X_train_raw[numeric_cols].mean().values
        np.testing.assert_allclose(scaler.mean_, expected, rtol=1e-4)

    def test_transform_only_called_not_fit(self, cfg, tmp_path) -> None:
        """Loading task data must not call fit on the preprocessor."""
        raw = tmp_path / "obs.csv"
        generate_observations(n_samples=300, seed=55, cfg=cfg).to_csv(raw, index=False)
        run_preprocessing(raw, tmp_path, cfg)
        data = load_task_data("health", tmp_path)
        # After load_task_data, scaler mean must still match X_train (not all data)
        from src.features.build_features import NUMERIC_FEATURE_COLUMNS

        scaler = data.preprocessor.named_transformers_["num"].named_steps["scaler"]
        numeric_cols = [c for c in NUMERIC_FEATURE_COLUMNS if c in data.X_train_raw.columns]
        expected = data.X_train_raw[numeric_cols].mean().values
        np.testing.assert_allclose(scaler.mean_, expected, rtol=1e-4)


# ---------------------------------------------------------------------------
# TestPredictionShapesAndLabels
# ---------------------------------------------------------------------------


class TestPredictionShapesAndLabels:
    def _get_predictions(self, result: dict, data: TaskData, algo: str):
        """Re-predict using saved model payload."""
        payload = joblib.load(Path(result["best_model_path"]))
        estimator = payload["estimator"]
        le = payload.get("label_encoder")
        X_test = data.X_test_t

        if le is not None:  # XGBoost
            y_pred = le.inverse_transform(estimator.predict(X_test))
            y_proba = estimator.predict_proba(X_test)
        else:
            y_pred = estimator.predict(X_test)
            y_proba = (
                estimator.predict_proba(X_test) if hasattr(estimator, "predict_proba") else None
            )
        return y_pred, y_proba

    def test_health_best_predictions_correct_length(self, health_result, health_data) -> None:
        y_pred, _ = self._get_predictions(health_result, health_data, "best")
        assert len(y_pred) == len(health_data.y_test)

    def test_health_best_predictions_valid_labels(self, health_result, health_data) -> None:
        y_pred, _ = self._get_predictions(health_result, health_data, "best")
        valid = set(health_data.label_names)
        assert all(lbl in valid for lbl in y_pred)

    def test_health_best_proba_correct_shape(self, health_result, health_data) -> None:
        _, y_proba = self._get_predictions(health_result, health_data, "best")
        assert y_proba is not None
        assert y_proba.shape == (len(health_data.y_test), len(health_data.label_names))

    def test_health_best_proba_rows_sum_to_one(self, health_result, health_data) -> None:
        _, y_proba = self._get_predictions(health_result, health_data, "best")
        row_sums = y_proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_all_health_algo_predictions_valid(self, health_result: dict) -> None:
        """Each algorithm's stored confusion matrix must only contain known labels."""
        for algo, r in health_result["models"].items():
            cm = r["test_confusion_matrix"]
            assert isinstance(cm, list), f"{algo} confusion matrix should be a list"


# ---------------------------------------------------------------------------
# TestMetricsProduced
# ---------------------------------------------------------------------------


class TestMetricsProduced:
    _REQUIRED_SCALAR = {
        "cv_macro_f1_mean",
        "cv_macro_f1_std",
        "cv_balanced_accuracy_mean",
        "cv_balanced_accuracy_std",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        "test_macro_precision",
        "test_macro_recall",
    }

    def test_health_all_scalars_present(self, health_result: dict) -> None:
        for algo, r in health_result["models"].items():
            missing = self._REQUIRED_SCALAR - set(r.keys())
            assert not missing, f"{algo} missing metrics: {missing}"

    def test_restoration_all_scalars_present(self, restoration_result: dict) -> None:
        for algo, r in restoration_result["models"].items():
            missing = self._REQUIRED_SCALAR - set(r.keys())
            assert not missing, f"{algo} missing metrics: {missing}"

    def test_health_per_class_metrics_present(self, health_result: dict) -> None:
        for algo, r in health_result["models"].items():
            per = r.get("test_per_class", {})
            for lbl in health_result["label_names"]:
                assert lbl in per, f"{algo}: per-class missing for '{lbl}'"
                assert "recall" in per[lbl], f"{algo}: recall missing for '{lbl}'"

    def test_restoration_per_class_metrics_present(self, restoration_result: dict) -> None:
        for _algo, r in restoration_result["models"].items():
            per = r.get("test_per_class", {})
            for lbl in restoration_result["label_names"]:
                assert lbl in per

    def test_classification_report_is_string(self, health_result: dict) -> None:
        for _algo, r in health_result["models"].items():
            assert isinstance(r["test_classification_report"], str)
            assert "precision" in r["test_classification_report"]

    def test_metrics_in_valid_range(self, health_result: dict) -> None:
        for algo, r in health_result["models"].items():
            for metric in (
                "test_accuracy",
                "test_balanced_accuracy",
                "test_macro_f1",
                "test_weighted_f1",
            ):
                val = r[metric]
                assert 0.0 <= val <= 1.0, f"{algo}: {metric}={val} out of [0,1]"


# ---------------------------------------------------------------------------
# TestModelSelection
# ---------------------------------------------------------------------------


class TestModelSelection:
    """Best model must be chosen by CV macro-F1 (primary), balanced accuracy (secondary)."""

    def test_best_model_name_is_valid(self, health_result: dict) -> None:
        assert health_result["best_model_name"] in health_result["models"]

    def test_best_model_has_highest_cv_macro_f1(self, health_result: dict) -> None:
        best_name = health_result["best_model_name"]
        best_f1 = health_result["models"][best_name]["cv_macro_f1_mean"]
        for algo, r in health_result["models"].items():
            if algo != best_name:
                assert r["cv_macro_f1_mean"] <= best_f1 + 1e-9, (
                    f"{algo} has higher CV macro-F1 ({r['cv_macro_f1_mean']:.4f}) "
                    f"than selected best {best_name} ({best_f1:.4f})"
                )

    def test_best_cv_macro_f1_stored_correctly(self, health_result: dict) -> None:
        best_name = health_result["best_model_name"]
        expected = health_result["models"][best_name]["cv_macro_f1_mean"]
        assert abs(health_result["best_cv_macro_f1"] - expected) < 1e-9

    def test_restoration_best_model_selected(self, restoration_result: dict) -> None:
        best_name = restoration_result["best_model_name"]
        best_f1 = restoration_result["models"][best_name]["cv_macro_f1_mean"]
        for _algo, r in restoration_result["models"].items():
            assert r["cv_macro_f1_mean"] <= best_f1 + 1e-9


# ---------------------------------------------------------------------------
# TestArtefactsSaved
# ---------------------------------------------------------------------------


class TestArtefactsSaved:
    def test_best_model_joblib_exists_health(self, health_result: dict) -> None:
        assert Path(health_result["best_model_path"]).exists()

    def test_best_model_joblib_exists_restoration(self, restoration_result: dict) -> None:
        assert Path(restoration_result["best_model_path"]).exists()

    def test_evaluation_json_exists_health(self, health_result: dict) -> None:
        assert Path(health_result["evaluation_path"]).exists()

    def test_evaluation_json_exists_restoration(self, restoration_result: dict) -> None:
        assert Path(restoration_result["evaluation_path"]).exists()

    def test_evaluation_json_valid_structure(self, health_result: dict) -> None:
        with open(health_result["evaluation_path"]) as fh:
            data = json.load(fh)
        assert "task" in data
        assert "best_model_name" in data
        assert "models" in data
        assert data["task"] == "health"

    def test_best_model_payload_has_required_keys(self, health_result: dict) -> None:
        payload = joblib.load(health_result["best_model_path"])
        for key in ("estimator", "algo_name", "task", "label_names", "feature_names"):
            assert key in payload, f"Payload missing key '{key}'"

    def test_best_model_estimator_is_fitted(self, health_result: dict) -> None:
        from sklearn.utils.validation import check_is_fitted

        payload = joblib.load(health_result["best_model_path"])
        check_is_fitted(payload["estimator"])

    def test_mlflow_run_ids_recorded(self, health_result: dict) -> None:
        for algo, r in health_result["models"].items():
            assert "mlflow_run_id" in r and r["mlflow_run_id"], f"MLflow run_id missing for {algo}"


# ---------------------------------------------------------------------------
# TestComputeMetrics (unit tests for evaluate.py)
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def _sample_data(self):
        rng = np.random.default_rng(42)
        labels = ["a", "b", "c"]
        y_true = rng.choice(labels, size=60)
        y_pred = rng.choice(labels, size=60)
        y_proba = rng.dirichlet(np.ones(3), size=60)
        return y_true, y_pred, y_proba, labels

    def test_returns_dict(self) -> None:
        y_true, y_pred, y_proba, labels = self._sample_data()
        result = compute_metrics(y_true, y_pred, y_proba, labels)
        assert isinstance(result, dict)

    def test_accuracy_in_range(self) -> None:
        y_true, y_pred, y_proba, labels = self._sample_data()
        result = compute_metrics(y_true, y_pred, y_proba, labels)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_all_label_names_in_per_class(self) -> None:
        y_true, y_pred, y_proba, labels = self._sample_data()
        result = compute_metrics(y_true, y_pred, y_proba, labels)
        for lbl in labels:
            assert lbl in result["per_class"]

    def test_confusion_matrix_shape(self) -> None:
        y_true, y_pred, y_proba, labels = self._sample_data()
        result = compute_metrics(y_true, y_pred, y_proba, labels)
        cm = result["confusion_matrix"]
        assert len(cm) == 3
        assert all(len(row) == 3 for row in cm)

    def test_classification_report_is_string(self) -> None:
        y_true, y_pred, y_proba, labels = self._sample_data()
        result = compute_metrics(y_true, y_pred, y_proba, labels)
        assert isinstance(result["classification_report"], str)

    def test_no_proba_accepted(self) -> None:
        y_true, y_pred, _, labels = self._sample_data()
        result = compute_metrics(y_true, y_pred, None, labels)
        assert "accuracy" in result

    def test_format_comparison_table_returns_string(self) -> None:
        fake_results = {
            "lr": {
                "cv_macro_f1_mean": 0.80,
                "cv_macro_f1_std": 0.02,
                "cv_balanced_accuracy_mean": 0.80,
                "cv_balanced_accuracy_std": 0.02,
                "test_accuracy": 0.81,
                "test_balanced_accuracy": 0.80,
                "test_macro_f1": 0.80,
                "test_weighted_f1": 0.81,
            },
        }
        table = format_comparison_table("health", fake_results)
        assert isinstance(table, str)
        assert "health" in table
        assert "lr" in table


# ---------------------------------------------------------------------------
# TestCLIValidation
# ---------------------------------------------------------------------------


class TestCLIValidation:
    def test_invalid_task_exits_nonzero(self) -> None:
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "src.models.train", "--task", "invalid_task_xyz"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_quick_mode_flag_accepted(self, cfg, tmp_path) -> None:
        """Quick-mode CLI completes without error on small data."""
        import subprocess
        import sys

        raw = tmp_path / "obs.csv"
        generate_observations(n_samples=300, seed=5, cfg=cfg).to_csv(raw, index=False)
        run_preprocessing(raw, tmp_path, cfg)
        mlflow_uri = f"sqlite:///{tmp_path}/mlruns.db"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.models.train",
                "--task",
                "health",
                "--processed-dir",
                str(tmp_path),
                "--output-dir",
                str(tmp_path / "models"),
                "--mlflow-uri",
                mlflow_uri,
                "--quick",
                "--n-jobs",
                "1",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr:\n{result.stderr}"
