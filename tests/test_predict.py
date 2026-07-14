"""
tests/test_predict.py — Tests for src/models/predict.py.

All tests use isolated temporary MLflow databases and artifact directories.
No test touches the real registry, production models, or project datasets.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from src.config import get_config, reset_config
from src.data.generate_data import generate_observations
from src.data.preprocess import run_preprocessing
from src.models.predict import (
    InferencePipeline,
    _add_derived_features,
    _to_dataframe,
)
from src.models.registry import run_register_and_promote
from src.models.train import train_task

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg():
    reset_config()
    return get_config()


def _build_predict_env(tmp_path_factory, cfg, task: str) -> dict[str, Any]:
    """Build an isolated environment with trained + registered champion."""
    root = tmp_path_factory.mktemp(f"predict_{task}")
    raw_csv = root / "observations.csv"
    processed_dir = root / "processed"
    models_dir = root / "models"
    models_dir.mkdir()
    mlflow_uri = f"sqlite:///{root}/mlruns.db"

    generate_observations(n_samples=400, seed=77, cfg=cfg).to_csv(raw_csv, index=False)
    run_preprocessing(raw_csv, processed_dir, cfg)
    train_task(
        task=task,
        processed_dir=processed_dir,
        output_dir=models_dir,
        cfg=cfg,
        mlflow_uri=mlflow_uri,
        quick=True,
        n_jobs=1,
    )
    from mlflow import MlflowClient
    from src.models.registry import promote_champion

    reg_results = run_register_and_promote(
        task=task,
        mlflow_uri=mlflow_uri,
        output_dir=models_dir,
        cfg=cfg,
        promote=True,
    )
    if not reg_results[task]["champion_set"]:
        # Quality gate may not pass on tiny 400-sample datasets.
        # Force-promote so that InferencePipeline tests have a valid champion.
        model_name = reg_results[task]["registered_model_name"]
        version = str(reg_results[task]["version"])
        client = MlflowClient(tracking_uri=mlflow_uri)
        client.set_model_version_tag(
            name=model_name, version=version, key="quality_gate_passed", value="True"
        )
        promote_champion(task, version, mlflow_uri=mlflow_uri, cfg=cfg)

    return {
        "raw_csv": raw_csv,
        "processed_dir": processed_dir,
        "models_dir": models_dir,
        "mlflow_uri": mlflow_uri,
    }


@pytest.fixture(scope="module")
def health_env(tmp_path_factory, cfg):
    return _build_predict_env(tmp_path_factory, cfg, "health")


@pytest.fixture(scope="module")
def restoration_env(tmp_path_factory, cfg):
    return _build_predict_env(tmp_path_factory, cfg, "restoration")


@pytest.fixture(scope="module")
def health_pipeline(health_env, cfg):
    return InferencePipeline(
        task="health",
        mlflow_uri=health_env["mlflow_uri"],
        processed_dir=health_env["processed_dir"],
        models_dir=health_env["models_dir"],
        cfg=cfg,
    )


@pytest.fixture(scope="module")
def restoration_pipeline(restoration_env, cfg):
    return InferencePipeline(
        task="restoration",
        mlflow_uri=restoration_env["mlflow_uri"],
        processed_dir=restoration_env["processed_dir"],
        models_dir=restoration_env["models_dir"],
        cfg=cfg,
    )


@pytest.fixture(scope="module")
def sample_record(cfg, health_env) -> dict[str, Any]:
    """A single valid sensor record drawn from the generated dataset."""
    import pandas as pd

    df = pd.read_csv(health_env["raw_csv"])
    row = df.iloc[0]
    return {
        "depth_m": float(row["depth_m"]),
        "water_temperature_c": float(row["water_temperature_c"]),
        "ph": float(row["ph"]),
        "salinity_ppt": float(row["salinity_ppt"]),
        "dissolved_oxygen_mg_l": float(row["dissolved_oxygen_mg_l"]),
        "turbidity_ntu": float(row["turbidity_ntu"]),
        "light_intensity": float(row["light_intensity"]),
        "current_speed_m_s": float(row["current_speed_m_s"]),
        "sonar_backscatter": float(row["sonar_backscatter"]),
        "rugosity_index": float(row["rugosity_index"]),
        "hard_substrate_percentage": float(row["hard_substrate_percentage"]),
        "acoustic_complexity_index": float(row["acoustic_complexity_index"]),
        "coral_cover_percentage": float(row["coral_cover_percentage"]),
        "bleaching_percentage": float(row["bleaching_percentage"]),
        "disease_percentage": float(row["disease_percentage"]),
        "region": str(row["region"]),
    }


# ---------------------------------------------------------------------------
# TestInferencePipelineLoad
# ---------------------------------------------------------------------------


class TestInferencePipelineLoad:
    def test_health_pipeline_loads(self, health_pipeline) -> None:
        assert health_pipeline is not None

    def test_health_pipeline_has_label_names(self, health_pipeline) -> None:
        assert len(health_pipeline.label_names) >= 2

    def test_health_pipeline_has_algo_name(self, health_pipeline) -> None:
        assert health_pipeline.algo_name in ("logistic_regression", "random_forest", "xgboost")

    def test_health_pipeline_has_model_version(self, health_pipeline) -> None:
        assert health_pipeline.model_version is not None

    def test_health_pipeline_has_registered_model_name(self, health_pipeline, cfg) -> None:
        assert health_pipeline.registered_model_name == cfg.mlflow_registered_health

    def test_preprocessor_is_not_refitted(self, health_pipeline) -> None:
        """Verify the loaded preprocessor is already fitted (not a bare instance)."""
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(health_pipeline._preprocessor)  # raises if not fitted

    def test_invalid_task_raises(self, health_env, cfg) -> None:
        with pytest.raises(ValueError, match="task must be one of"):
            InferencePipeline(
                "bad_task",
                mlflow_uri=health_env["mlflow_uri"],
                processed_dir=health_env["processed_dir"],
                models_dir=health_env["models_dir"],
                cfg=cfg,
            )

    def test_missing_champion_raises(self, tmp_path, cfg) -> None:
        from mlflow.exceptions import MlflowException

        with pytest.raises((MlflowException, RuntimeError, Exception)):
            InferencePipeline(
                "health",
                mlflow_uri=f"sqlite:///{tmp_path}/empty.db",
                processed_dir=tmp_path,
                models_dir=tmp_path,
                cfg=cfg,
            )


# ---------------------------------------------------------------------------
# TestSinglePrediction
# ---------------------------------------------------------------------------


class TestSinglePrediction:
    def test_predict_single_returns_dict(self, health_pipeline, sample_record) -> None:
        result = health_pipeline.predict_single(sample_record)
        assert isinstance(result, dict)

    def test_predict_single_has_predicted_class(self, health_pipeline, sample_record) -> None:
        result = health_pipeline.predict_single(sample_record)
        assert "predicted_class" in result
        assert result["predicted_class"] in health_pipeline.label_names

    def test_predict_single_has_probabilities(self, health_pipeline, sample_record) -> None:
        result = health_pipeline.predict_single(sample_record)
        assert "probabilities" in result
        proba = result["probabilities"]
        assert isinstance(proba, dict)
        assert len(proba) == len(health_pipeline.label_names)

    def test_probabilities_sum_to_one(self, health_pipeline, sample_record) -> None:
        result = health_pipeline.predict_single(sample_record)
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 1e-4

    def test_predict_single_has_confidence(self, health_pipeline, sample_record) -> None:
        result = health_pipeline.predict_single(sample_record)
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0

    def test_confidence_equals_max_proba(self, health_pipeline, sample_record) -> None:
        result = health_pipeline.predict_single(sample_record)
        max_p = max(result["probabilities"].values())
        assert abs(result["confidence"] - max_p) < 1e-6

    def test_predict_single_has_task(self, health_pipeline, sample_record) -> None:
        result = health_pipeline.predict_single(sample_record)
        assert result["task"] == "health"

    def test_predict_single_has_model_version(self, health_pipeline, sample_record) -> None:
        result = health_pipeline.predict_single(sample_record)
        assert "model_version" in result

    def test_predict_single_has_run_id(self, health_pipeline, sample_record) -> None:
        result = health_pipeline.predict_single(sample_record)
        assert "run_id" in result

    def test_predict_single_has_timestamp(self, health_pipeline, sample_record) -> None:
        result = health_pipeline.predict_single(sample_record)
        assert "prediction_timestamp" in result

    def test_predict_single_has_disclaimer(self, health_pipeline, sample_record) -> None:
        result = health_pipeline.predict_single(sample_record)
        assert "synthetic_data_disclaimer" in result

    def test_predict_single_has_registered_model_name(self, health_pipeline, sample_record) -> None:
        result = health_pipeline.predict_single(sample_record)
        assert "registered_model_name" in result

    def test_predict_single_has_model_alias(self, health_pipeline, sample_record) -> None:
        result = health_pipeline.predict_single(sample_record)
        assert "model_alias" in result
        assert result["model_alias"] == "champion"


# ---------------------------------------------------------------------------
# TestBatchPrediction
# ---------------------------------------------------------------------------


class TestBatchPrediction:
    @pytest.fixture(scope="class")
    def batch_df(self, cfg, health_env):
        import pandas as pd

        df = pd.read_csv(health_env["raw_csv"]).head(10)
        return df[
            [
                "depth_m",
                "water_temperature_c",
                "ph",
                "salinity_ppt",
                "dissolved_oxygen_mg_l",
                "turbidity_ntu",
                "light_intensity",
                "current_speed_m_s",
                "sonar_backscatter",
                "rugosity_index",
                "hard_substrate_percentage",
                "acoustic_complexity_index",
                "coral_cover_percentage",
                "bleaching_percentage",
                "disease_percentage",
                "region",
            ]
        ]

    def test_predict_batch_returns_list(self, health_pipeline, batch_df) -> None:
        results = health_pipeline.predict_batch(batch_df)
        assert isinstance(results, list)

    def test_predict_batch_length_matches_input(self, health_pipeline, batch_df) -> None:
        results = health_pipeline.predict_batch(batch_df)
        assert len(results) == len(batch_df)

    def test_batch_all_predictions_valid_labels(self, health_pipeline, batch_df) -> None:
        results = health_pipeline.predict_batch(batch_df)
        for r in results:
            assert r["predicted_class"] in health_pipeline.label_names

    def test_batch_all_probabilities_sum_to_one(self, health_pipeline, batch_df) -> None:
        results = health_pipeline.predict_batch(batch_df)
        for i, r in enumerate(results):
            total = sum(r["probabilities"].values())
            assert abs(total - 1.0) < 1e-4, f"Row {i}: prob sum={total}"

    def test_batch_from_csv(self, health_pipeline, health_env) -> None:
        results = health_pipeline.predict_batch(health_env["raw_csv"])
        assert len(results) > 0
        assert all(r["predicted_class"] in health_pipeline.label_names for r in results)

    def test_restoration_batch(self, restoration_pipeline, restoration_env) -> None:
        import pandas as pd

        df = pd.read_csv(restoration_env["raw_csv"]).head(5)
        results = restoration_pipeline.predict_batch(
            df[
                [
                    "depth_m",
                    "water_temperature_c",
                    "ph",
                    "salinity_ppt",
                    "dissolved_oxygen_mg_l",
                    "turbidity_ntu",
                    "light_intensity",
                    "current_speed_m_s",
                    "sonar_backscatter",
                    "rugosity_index",
                    "hard_substrate_percentage",
                    "acoustic_complexity_index",
                    "coral_cover_percentage",
                    "bleaching_percentage",
                    "disease_percentage",
                    "region",
                ]
            ]
        )
        assert len(results) == 5
        for r in results:
            assert r["predicted_class"] in restoration_pipeline.label_names


# ---------------------------------------------------------------------------
# TestDeterminism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_repeated_single_prediction_is_deterministic(
        self, health_pipeline, sample_record
    ) -> None:
        r1 = health_pipeline.predict_single(sample_record)
        r2 = health_pipeline.predict_single(sample_record)
        assert r1["predicted_class"] == r2["predicted_class"]
        for lbl in r1["probabilities"]:
            assert abs(r1["probabilities"][lbl] - r2["probabilities"][lbl]) < 1e-9

    def test_repeated_batch_prediction_is_deterministic(self, health_pipeline, health_env) -> None:
        import pandas as pd

        df = pd.read_csv(health_env["raw_csv"]).head(5)
        raw_cols = [
            "depth_m",
            "water_temperature_c",
            "ph",
            "salinity_ppt",
            "dissolved_oxygen_mg_l",
            "turbidity_ntu",
            "light_intensity",
            "current_speed_m_s",
            "sonar_backscatter",
            "rugosity_index",
            "hard_substrate_percentage",
            "acoustic_complexity_index",
            "coral_cover_percentage",
            "bleaching_percentage",
            "disease_percentage",
            "region",
        ]
        r1 = health_pipeline.predict_batch(df[raw_cols])
        r2 = health_pipeline.predict_batch(df[raw_cols])
        for i in range(len(r1)):
            assert r1[i]["predicted_class"] == r2[i]["predicted_class"]


# ---------------------------------------------------------------------------
# TestInputValidation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_missing_required_field_raises(self, health_pipeline, sample_record) -> None:
        bad = {k: v for k, v in sample_record.items() if k != "ph"}
        with pytest.raises(ValueError, match="Missing required feature"):
            health_pipeline.predict_single(bad)

    def test_missing_region_raises(self, health_pipeline, sample_record) -> None:
        bad = {k: v for k, v in sample_record.items() if k != "region"}
        with pytest.raises(ValueError, match="Missing required feature"):
            health_pipeline.predict_single(bad)

    def test_extra_fields_ignored(self, health_pipeline, sample_record) -> None:
        """Unknown extra columns should not cause errors."""
        with_extra = dict(sample_record, reef_health="healthy", extra_col=99.9)
        result = health_pipeline.predict_single(with_extra)
        assert result["predicted_class"] in health_pipeline.label_names

    def test_empty_dataframe_raises(self, health_pipeline) -> None:
        import pandas as pd

        with pytest.raises((ValueError, KeyError, RuntimeError)):
            health_pipeline.predict_batch(pd.DataFrame())

    def test_missing_csv_raises(self, health_pipeline, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            health_pipeline.predict_batch(tmp_path / "nonexistent.csv")


# ---------------------------------------------------------------------------
# TestFeatureOrder
# ---------------------------------------------------------------------------


class TestFeatureOrder:
    def test_shuffled_column_order_gives_same_prediction(
        self, health_pipeline, sample_record
    ) -> None:
        """Column reordering must not change the prediction (enforced by pipeline)."""
        import pandas as pd

        df_orig = pd.DataFrame([sample_record])
        # Shuffle columns randomly
        cols = list(df_orig.columns)
        np.random.default_rng(0).shuffle(cols)
        df_shuffled = df_orig[cols]
        r_orig = health_pipeline.predict_batch(df_orig)
        r_shuf = health_pipeline.predict_batch(df_shuffled)
        assert r_orig[0]["predicted_class"] == r_shuf[0]["predicted_class"]
        for lbl in r_orig[0]["probabilities"]:
            assert abs(r_orig[0]["probabilities"][lbl] - r_shuf[0]["probabilities"][lbl]) < 1e-9


# ---------------------------------------------------------------------------
# TestPreprocessorNoRefit
# ---------------------------------------------------------------------------


class TestPreprocessorNoRefit:
    def test_preprocessor_mean_unchanged_after_prediction(
        self, health_pipeline, sample_record
    ) -> None:
        """
        Verify the preprocessor's scaler mean is unchanged before and after
        a prediction call, confirming transform-only usage.
        """
        scaler = health_pipeline._preprocessor.named_transformers_["num"].named_steps["scaler"]
        mean_before = scaler.mean_.copy()
        health_pipeline.predict_single(sample_record)
        mean_after = scaler.mean_.copy()
        np.testing.assert_array_equal(mean_before, mean_after)

    def test_preprocessor_has_no_partial_fit_called(self, health_pipeline, sample_record) -> None:
        """Predict multiple times and verify var_ doesn't change (no partial_fit)."""
        scaler = health_pipeline._preprocessor.named_transformers_["num"].named_steps["scaler"]
        var_before = scaler.var_.copy()
        for _ in range(3):
            health_pipeline.predict_single(sample_record)
        var_after = scaler.var_.copy()
        np.testing.assert_array_equal(var_before, var_after)


# ---------------------------------------------------------------------------
# TestDerivedFeatures
# ---------------------------------------------------------------------------


class TestDerivedFeatures:
    def test_add_derived_features_returns_dataframe(self, sample_record) -> None:
        import pandas as pd

        df = pd.DataFrame([sample_record])
        out = _add_derived_features(df)
        assert isinstance(out, pd.DataFrame)

    def test_all_six_derived_features_added(self, sample_record) -> None:
        import pandas as pd

        df = pd.DataFrame([sample_record])
        out = _add_derived_features(df)
        for feat in [
            "thermal_stress_index",
            "oxygen_stress_index",
            "acidity_deviation",
            "water_quality_index",
            "substrate_stability_score",
            "structural_complexity_score",
        ]:
            assert feat in out.columns

    def test_thermal_stress_index_clipped(self, sample_record) -> None:
        import pandas as pd

        df = pd.DataFrame([sample_record])
        out = _add_derived_features(df)
        val = out["thermal_stress_index"].iloc[0]
        assert 0.0 <= val <= 1.0

    def test_water_quality_index_clipped(self, sample_record) -> None:
        import pandas as pd

        df = pd.DataFrame([sample_record])
        out = _add_derived_features(df)
        val = out["water_quality_index"].iloc[0]
        assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# TestToDataframe
# ---------------------------------------------------------------------------


class TestToDataframe:
    def test_dict_becomes_single_row_df(self, sample_record) -> None:
        import pandas as pd

        df = _to_dataframe(sample_record)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_dataframe_passthrough(self, sample_record) -> None:
        import pandas as pd

        df_in = pd.DataFrame([sample_record])
        df_out = _to_dataframe(df_in)
        assert isinstance(df_out, pd.DataFrame)
        assert len(df_out) == len(df_in)

    def test_csv_path_loaded(self, health_env) -> None:
        import pandas as pd

        df = _to_dataframe(health_env["raw_csv"])
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_missing_path_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            _to_dataframe(tmp_path / "nope.csv")


# ---------------------------------------------------------------------------
# TestCLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_invalid_task_exits_nonzero(self) -> None:
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.models.predict",
                "--task",
                "bad_task",
                "--input",
                "nowhere.json",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_missing_input_exits_nonzero(self, health_env, cfg) -> None:
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.models.predict",
                "--task",
                "health",
                "--input",
                "/tmp/definitely_missing_file.csv",
                "--mlflow-uri",
                health_env["mlflow_uri"],
                "--processed-dir",
                str(health_env["processed_dir"]),
                "--models-dir",
                str(health_env["models_dir"]),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_predict_cli_json_input(self, tmp_path, health_env, sample_record, cfg) -> None:
        import subprocess
        import sys

        json_path = tmp_path / "input.json"
        json_path.write_text(json.dumps(sample_record))

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.models.predict",
                "--task",
                "health",
                "--input",
                str(json_path),
                "--mlflow-uri",
                health_env["mlflow_uri"],
                "--processed-dir",
                str(health_env["processed_dir"]),
                "--models-dir",
                str(health_env["models_dir"]),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr:\n{result.stderr}"
        assert "predicted_class" in result.stdout

    def test_predict_cli_csv_output(self, tmp_path, health_env, sample_record, cfg) -> None:
        import subprocess
        import sys

        import pandas as pd

        csv_in = tmp_path / "input.csv"
        pd.DataFrame([sample_record]).to_csv(csv_in, index=False)
        csv_out = tmp_path / "predictions.csv"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.models.predict",
                "--task",
                "health",
                "--input",
                str(csv_in),
                "--output",
                str(csv_out),
                "--mlflow-uri",
                health_env["mlflow_uri"],
                "--processed-dir",
                str(health_env["processed_dir"]),
                "--models-dir",
                str(health_env["models_dir"]),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr:\n{result.stderr}"
        assert csv_out.exists()
        out_df = pd.read_csv(csv_out)
        assert "predicted_class" in out_df.columns
        assert len(out_df) == 1
