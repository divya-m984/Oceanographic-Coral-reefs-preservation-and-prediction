"""
tests/test_registry.py — Tests for src/models/registry.py.

All tests use isolated temporary MLflow SQLite databases and temporary
artifact directories.  No test touches the real project registry,
production models, or datasets.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.config import get_config, reset_config
from src.data.generate_data import generate_observations
from src.data.preprocess import run_preprocessing
from src.models.registry import (
    QualityGateResult,
    RegistrationResult,
    check_quality_gates,
    get_champion_version,
    promote_champion,
    register_candidate,
    run_register_and_promote,
)
from src.models.train import train_task

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg():
    reset_config()
    return get_config()


def _build_registry_env(tmp_path_factory, cfg, task: str) -> dict[str, Any]:
    """
    Generate data, preprocess, train, and prepare an isolated MLflow env
    for the given task.  Returns a dict with all relevant paths.
    """
    root = tmp_path_factory.mktemp(f"registry_{task}")
    raw_csv = root / "observations.csv"
    processed_dir = root / "processed"
    models_dir = root / "models"
    models_dir.mkdir()
    mlflow_uri = f"sqlite:///{root}/mlruns.db"

    # Generate small dataset
    generate_observations(n_samples=400, seed=99, cfg=cfg).to_csv(raw_csv, index=False)
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
    return {
        "root": root,
        "raw_csv": raw_csv,
        "processed_dir": processed_dir,
        "models_dir": models_dir,
        "mlflow_uri": mlflow_uri,
    }


@pytest.fixture(scope="module")
def health_env(tmp_path_factory, cfg):
    return _build_registry_env(tmp_path_factory, cfg, "health")


@pytest.fixture(scope="module")
def restoration_env(tmp_path_factory, cfg):
    return _build_registry_env(tmp_path_factory, cfg, "restoration")


# ---------------------------------------------------------------------------
# TestQualityGates
# ---------------------------------------------------------------------------


class TestQualityGates:
    def test_pass_above_thresholds(self, cfg) -> None:
        result = check_quality_gates(
            "health",
            cv_macro_f1=0.80,
            cv_balanced_accuracy=0.80,
            cfg=cfg,
        )
        assert isinstance(result, QualityGateResult)
        assert result.passed is True
        assert result.failures == []

    def test_fail_below_f1_threshold(self, cfg) -> None:
        result = check_quality_gates(
            "health",
            cv_macro_f1=0.50,  # below min_cv_macro_f1=0.70
            cv_balanced_accuracy=0.80,
            cfg=cfg,
        )
        assert result.passed is False
        assert len(result.failures) >= 1
        assert any("cv_macro_f1" in f for f in result.failures)

    def test_fail_below_balanced_accuracy_threshold(self, cfg) -> None:
        result = check_quality_gates(
            "restoration",
            cv_macro_f1=0.80,
            cv_balanced_accuracy=0.50,  # below min_cv_balanced_accuracy=0.73
            cfg=cfg,
        )
        assert result.passed is False
        assert any("cv_balanced_accuracy" in f for f in result.failures)

    def test_fail_both_below_threshold(self, cfg) -> None:
        result = check_quality_gates(
            "health",
            cv_macro_f1=0.40,
            cv_balanced_accuracy=0.40,
            cfg=cfg,
        )
        assert result.passed is False
        assert len(result.failures) == 2

    def test_result_stores_thresholds(self, cfg) -> None:
        result = check_quality_gates("health", 0.75, 0.75, cfg)
        assert result.min_cv_macro_f1 > 0
        assert result.min_cv_balanced_accuracy > 0

    def test_gate_str_representation(self, cfg) -> None:
        result = check_quality_gates("health", 0.75, 0.75, cfg)
        s = str(result)
        assert "health" in s
        assert "cv_macro_f1" in s

    def test_unknown_task_uses_zero_thresholds(self, cfg) -> None:
        # Unknown task has no gates in config → thresholds default to 0.0
        result = check_quality_gates("bad_task", 0.0, 0.0, cfg)
        assert result.min_cv_macro_f1 == 0.0
        assert result.min_cv_balanced_accuracy == 0.0


# ---------------------------------------------------------------------------
# TestRegistration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register_health_returns_result(self, health_env, cfg) -> None:
        result = register_candidate(
            "health",
            mlflow_uri=health_env["mlflow_uri"],
            output_dir=health_env["models_dir"],
            cfg=cfg,
        )
        assert isinstance(result, RegistrationResult)

    def test_registration_result_has_model_name(self, health_env, cfg) -> None:
        result = register_candidate(
            "health",
            mlflow_uri=health_env["mlflow_uri"],
            output_dir=health_env["models_dir"],
            cfg=cfg,
        )
        assert result.registered_model_name == cfg.mlflow_registered_health

    def test_registration_result_has_version(self, health_env, cfg) -> None:
        result = register_candidate(
            "health",
            mlflow_uri=health_env["mlflow_uri"],
            output_dir=health_env["models_dir"],
            cfg=cfg,
        )
        assert result.version is not None
        assert isinstance(result.version, str)
        assert int(result.version) >= 1

    def test_registration_result_has_run_id(self, health_env, cfg) -> None:
        result = register_candidate(
            "health",
            mlflow_uri=health_env["mlflow_uri"],
            output_dir=health_env["models_dir"],
            cfg=cfg,
        )
        assert len(result.run_id) == 32  # MLflow UUID hex

    def test_registration_result_has_algo_name(self, health_env, cfg) -> None:
        result = register_candidate(
            "health",
            mlflow_uri=health_env["mlflow_uri"],
            output_dir=health_env["models_dir"],
            cfg=cfg,
        )
        assert result.algo_name in ("logistic_regression", "random_forest", "xgboost")

    def test_registration_result_has_cv_macro_f1(self, health_env, cfg) -> None:
        result = register_candidate(
            "health",
            mlflow_uri=health_env["mlflow_uri"],
            output_dir=health_env["models_dir"],
            cfg=cfg,
        )
        assert 0.0 < result.cv_macro_f1 <= 1.0

    def test_registration_result_has_gate(self, health_env, cfg) -> None:
        result = register_candidate(
            "health",
            mlflow_uri=health_env["mlflow_uri"],
            output_dir=health_env["models_dir"],
            cfg=cfg,
        )
        assert isinstance(result.gate, QualityGateResult)

    def test_register_restoration(self, restoration_env, cfg) -> None:
        result = register_candidate(
            "restoration",
            mlflow_uri=restoration_env["mlflow_uri"],
            output_dir=restoration_env["models_dir"],
            cfg=cfg,
        )
        assert result.registered_model_name == cfg.mlflow_registered_restoration

    def test_register_missing_eval_json_raises(self, tmp_path, cfg) -> None:
        with pytest.raises(FileNotFoundError, match="evaluation"):
            register_candidate(
                "health",
                mlflow_uri=f"sqlite:///{tmp_path}/mlruns.db",
                output_dir=tmp_path / "nodir",
                cfg=cfg,
            )

    def test_invalid_task_raises(self, tmp_path, cfg) -> None:
        with pytest.raises(ValueError, match="task must be one of"):
            register_candidate("bad_task", output_dir=tmp_path, cfg=cfg)

    def test_duplicate_registration_increments_version(self, health_env, cfg) -> None:
        r1 = register_candidate(
            "health",
            mlflow_uri=health_env["mlflow_uri"],
            output_dir=health_env["models_dir"],
            cfg=cfg,
        )
        r2 = register_candidate(
            "health",
            mlflow_uri=health_env["mlflow_uri"],
            output_dir=health_env["models_dir"],
            cfg=cfg,
        )
        assert int(r2.version) > int(r1.version)


# ---------------------------------------------------------------------------
# TestMetadataTags
# ---------------------------------------------------------------------------


class TestMetadataTags:
    """Verify rich metadata is stored in model version tags."""

    @pytest.fixture(scope="class")
    def registered_version(self, health_env, cfg):
        from mlflow import MlflowClient

        result = register_candidate(
            "health",
            mlflow_uri=health_env["mlflow_uri"],
            output_dir=health_env["models_dir"],
            cfg=cfg,
        )
        client = MlflowClient(tracking_uri=health_env["mlflow_uri"])
        mv = client.get_model_version(name=result.registered_model_name, version=result.version)
        # MLflow 3.x tags are a plain dict; older versions were a list of tag objects.
        raw = mv.tags or {}
        return raw if isinstance(raw, dict) else {t.key: t.value for t in raw}

    def test_tags_contain_task(self, registered_version) -> None:
        assert registered_version.get("task") == "health"

    def test_tags_contain_algo_name(self, registered_version) -> None:
        assert "algo_name" in registered_version
        assert registered_version["algo_name"] in (
            "logistic_regression",
            "random_forest",
            "xgboost",
        )

    def test_tags_contain_run_id(self, registered_version) -> None:
        assert "run_id" in registered_version
        assert len(registered_version["run_id"]) == 32

    def test_tags_contain_cv_macro_f1(self, registered_version) -> None:
        assert "cv_macro_f1" in registered_version
        val = float(registered_version["cv_macro_f1"])
        assert 0.0 < val <= 1.0

    def test_tags_contain_label_names(self, registered_version) -> None:
        assert "label_names" in registered_version
        labels = registered_version["label_names"].split(",")
        assert len(labels) >= 2

    def test_tags_contain_training_timestamp(self, registered_version) -> None:
        assert "training_timestamp" in registered_version

    def test_tags_contain_git_commit(self, registered_version) -> None:
        assert "git_commit" in registered_version

    def test_tags_contain_joblib_path(self, registered_version) -> None:
        assert "joblib_path" in registered_version
        assert registered_version["joblib_path"].endswith(".joblib")

    def test_tags_contain_quality_gate_status(self, registered_version) -> None:
        assert "quality_gate_passed" in registered_version
        assert registered_version["quality_gate_passed"].lower() in ("true", "false")

    def test_tags_contain_disclaimer(self, registered_version) -> None:
        assert "synthetic_data_disclaimer" in registered_version


# ---------------------------------------------------------------------------
# TestChampionPromotion
# ---------------------------------------------------------------------------


class TestChampionPromotion:
    @pytest.fixture(scope="class")
    def promoted_health(self, health_env, cfg):
        """Register and promote a health model."""
        result = run_register_and_promote(
            task="health",
            mlflow_uri=health_env["mlflow_uri"],
            output_dir=health_env["models_dir"],
            cfg=cfg,
            promote=True,
        )
        return result

    def test_run_register_and_promote_health_succeeds(self, promoted_health) -> None:
        assert "health" in promoted_health

    def test_champion_set_true_when_gate_passes(self, promoted_health) -> None:
        r = promoted_health["health"]
        if r["gate_passed"]:
            assert r["champion_set"] is True

    def test_champion_alias_resolves(self, health_env, cfg, promoted_health) -> None:
        if not promoted_health["health"]["gate_passed"]:
            pytest.skip("Gate did not pass — champion not set")
        info = get_champion_version(
            "health",
            mlflow_uri=health_env["mlflow_uri"],
            cfg=cfg,
        )
        assert info["version"] == promoted_health["health"]["version"]

    def test_champion_info_has_required_keys(self, health_env, cfg, promoted_health) -> None:
        if not promoted_health["health"]["gate_passed"]:
            pytest.skip("Gate did not pass")
        info = get_champion_version(
            "health",
            mlflow_uri=health_env["mlflow_uri"],
            cfg=cfg,
        )
        for key in (
            "registered_model_name",
            "version",
            "run_id",
            "algo_name",
            "task",
            "label_names",
            "cv_macro_f1",
            "joblib_path",
            "alias",
        ):
            assert key in info, f"Missing key: {key}"

    def test_champion_label_names_nonempty(self, health_env, cfg, promoted_health) -> None:
        if not promoted_health["health"]["gate_passed"]:
            pytest.skip("Gate did not pass")
        info = get_champion_version("health", mlflow_uri=health_env["mlflow_uri"], cfg=cfg)
        assert len(info["label_names"]) >= 2

    def test_promote_failing_candidate_raises(self, health_env, cfg) -> None:
        """Manually tag a version as gate-failed, then try to promote it."""
        from mlflow import MlflowClient

        # Register a fresh version
        result = register_candidate(
            "health",
            mlflow_uri=health_env["mlflow_uri"],
            output_dir=health_env["models_dir"],
            cfg=cfg,
        )
        client = MlflowClient(tracking_uri=health_env["mlflow_uri"])
        # Overwrite the quality gate tag to simulate failure
        client.set_model_version_tag(
            name=result.registered_model_name,
            version=result.version,
            key="quality_gate_passed",
            value="False",
        )
        with pytest.raises(RuntimeError, match="quality gate not passed"):
            promote_champion(
                "health",
                result.version,
                mlflow_uri=health_env["mlflow_uri"],
                cfg=cfg,
            )

    def test_champion_for_no_alias_raises(self, tmp_path, cfg) -> None:
        """Requesting champion when none is set should raise."""
        from mlflow.exceptions import MlflowException

        with pytest.raises((MlflowException, RuntimeError)):
            get_champion_version(
                "health",
                mlflow_uri=f"sqlite:///{tmp_path}/empty.db",
                cfg=cfg,
            )

    def test_run_register_and_promote_restoration(self, restoration_env, cfg) -> None:
        result = run_register_and_promote(
            task="restoration",
            mlflow_uri=restoration_env["mlflow_uri"],
            output_dir=restoration_env["models_dir"],
            cfg=cfg,
            promote=True,
        )
        assert "restoration" in result


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
                "src.models.registry",
                "--task",
                "invalid_xyz",
                "--register",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_no_action_flag_exits_nonzero(self) -> None:
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "src.models.registry", "--task", "health"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_register_and_promote_cli(self, health_env, cfg) -> None:
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.models.registry",
                "--task",
                "health",
                "--register",
                "--promote",
                "--mlflow-uri",
                health_env["mlflow_uri"],
                "--output-dir",
                str(health_env["models_dir"]),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr:\n{result.stderr}"
