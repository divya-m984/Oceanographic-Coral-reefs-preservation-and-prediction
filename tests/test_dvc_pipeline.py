"""
tests/test_dvc_pipeline.py — Unit tests for the M7 DVC pipeline scripts.

These tests exercise:
  - src.models.run_evaluate  (evaluate stage script)
  - src.models.run_register_candidate  (register_candidate stage script)
  - dvc.yaml syntactic validity
  - params.yaml DVC param key presence

Tests use temporary directories and small fixture data; they do NOT touch
real project data files or run the full DVC pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_evaluation_health(tmp_path: Path) -> Path:
    """Minimal evaluation_health.json matching train.py's schema."""
    data = {
        "task": "health",
        "best_model_name": "logistic_regression",
        "label_names": ["bleached", "healthy", "severely_degraded", "stressed"],
        "models": {
            "logistic_regression": {
                "cv_macro_f1_mean": 0.7612,
                "cv_balanced_accuracy_mean": 0.7800,
                "test_macro_f1": 0.7871,
                "test_balanced_accuracy": 0.8012,
                "test_accuracy": 0.7850,
                "mlflow_run_id": "abc123def456abc1",
            }
        },
    }
    out = tmp_path / "evaluation_health.json"
    out.write_text(json.dumps(data))
    return out


@pytest.fixture()
def fake_evaluation_restoration(tmp_path: Path) -> Path:
    """Minimal evaluation_restoration.json matching train.py's schema."""
    data = {
        "task": "restoration",
        "best_model_name": "xgboost",
        "label_names": ["moderately_suitable", "suitable", "unsuitable"],
        "models": {
            "xgboost": {
                "cv_macro_f1_mean": 0.7913,
                "cv_balanced_accuracy_mean": 0.8050,
                "test_macro_f1": 0.8029,
                "test_balanced_accuracy": 0.8121,
                "test_accuracy": 0.8010,
                "mlflow_run_id": "def456abc123def4",
            }
        },
    }
    out = tmp_path / "evaluation_restoration.json"
    out.write_text(json.dumps(data))
    return out


# ---------------------------------------------------------------------------
# run_evaluate tests
# ---------------------------------------------------------------------------


class TestRunEvaluate:
    def test_extract_metrics_returns_correct_keys(
        self,
        tmp_path: Path,
        fake_evaluation_health: Path,
        fake_evaluation_restoration: Path,
    ) -> None:
        from src.models.run_evaluate import extract_metrics

        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        metrics = extract_metrics(tmp_path, reports_dir)

        assert set(metrics.keys()) == {"health", "restoration"}
        for task in ("health", "restoration"):
            assert "best_algorithm" in metrics[task]
            assert "cv_macro_f1" in metrics[task]
            assert "test_macro_f1" in metrics[task]

    def test_extract_metrics_health_best_algo(
        self,
        tmp_path: Path,
        fake_evaluation_health: Path,
        fake_evaluation_restoration: Path,
    ) -> None:
        from src.models.run_evaluate import extract_metrics

        metrics = extract_metrics(tmp_path, tmp_path)
        assert metrics["health"]["best_algorithm"] == "logistic_regression"
        assert metrics["restoration"]["best_algorithm"] == "xgboost"

    def test_extract_metrics_values_rounded(
        self,
        tmp_path: Path,
        fake_evaluation_health: Path,
        fake_evaluation_restoration: Path,
    ) -> None:
        from src.models.run_evaluate import extract_metrics

        metrics = extract_metrics(tmp_path, tmp_path)
        # Values should be finite floats between 0 and 1
        assert 0.0 < metrics["health"]["cv_macro_f1"] <= 1.0
        assert 0.0 < metrics["restoration"]["test_macro_f1"] <= 1.0

    def test_extract_metrics_missing_file_raises(self, tmp_path: Path) -> None:
        from src.models.run_evaluate import extract_metrics

        with pytest.raises(FileNotFoundError, match="evaluation_health.json"):
            extract_metrics(tmp_path, tmp_path)

    def test_main_writes_metrics_json(
        self,
        tmp_path: Path,
        fake_evaluation_health: Path,
        fake_evaluation_restoration: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """main() should create reports/metrics.json and return 0."""
        from unittest.mock import MagicMock

        from src.models import run_evaluate

        mock_cfg = MagicMock()
        mock_cfg.paths.models_dir = tmp_path
        mock_cfg.paths.reports_dir = tmp_path / "reports"

        monkeypatch.setattr(run_evaluate, "get_config", lambda: mock_cfg)

        rc = run_evaluate.main()
        assert rc == 0
        out = tmp_path / "reports" / "metrics.json"
        assert out.exists()
        data = json.loads(out.read_text())
        assert "health" in data and "restoration" in data

    def test_main_returns_1_on_missing_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from src.models import run_evaluate

        mock_cfg = MagicMock()
        mock_cfg.paths.models_dir = tmp_path
        mock_cfg.paths.reports_dir = tmp_path / "reports"

        monkeypatch.setattr(run_evaluate, "get_config", lambda: mock_cfg)

        rc = run_evaluate.main()
        assert rc == 1


# ---------------------------------------------------------------------------
# run_register_candidate tests
# ---------------------------------------------------------------------------


class TestRunRegisterCandidate:
    def test_main_calls_run_register_and_promote_with_promote_false(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verifies that promote=False is always passed — champion must not change."""
        from src.models import run_register_candidate

        captured: dict = {}

        def fake_run(task, promote, cfg):
            captured["promote"] = promote
            captured["task"] = task
            return {
                "health": {
                    "registered_model_name": "coralsense_reef_health",
                    "version": "2",
                    "algo_name": "logistic_regression",
                    "cv_macro_f1": 0.7612,
                    "gate_passed": True,
                    "gate_failures": [],
                    "champion_set": False,
                },
                "restoration": {
                    "registered_model_name": "coralsense_restoration_suitability",
                    "version": "2",
                    "algo_name": "xgboost",
                    "cv_macro_f1": 0.7913,
                    "gate_passed": True,
                    "gate_failures": [],
                    "champion_set": False,
                },
            }

        mock_cfg = MagicMock()
        mock_cfg.paths.reports_dir = tmp_path / "reports"

        monkeypatch.setattr(run_register_candidate, "run_register_and_promote", fake_run)
        monkeypatch.setattr(run_register_candidate, "get_config", lambda: mock_cfg)

        rc = run_register_candidate.main()

        assert rc == 0
        assert captured["promote"] is False, "promote must be False — champion must not change"
        assert captured["task"] == "all"

    def test_main_writes_receipt_json(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from src.models import run_register_candidate

        def fake_run(task, promote, cfg):
            return {
                "health": {
                    "registered_model_name": "coralsense_reef_health",
                    "version": "2",
                    "algo_name": "logistic_regression",
                    "cv_macro_f1": 0.76,
                    "gate_passed": True,
                    "gate_failures": [],
                    "champion_set": False,
                },
                "restoration": {
                    "registered_model_name": "coralsense_restoration_suitability",
                    "version": "2",
                    "algo_name": "xgboost",
                    "cv_macro_f1": 0.79,
                    "gate_passed": True,
                    "gate_failures": [],
                    "champion_set": False,
                },
            }

        mock_cfg = MagicMock()
        mock_cfg.paths.reports_dir = tmp_path / "reports"

        monkeypatch.setattr(run_register_candidate, "run_register_and_promote", fake_run)
        monkeypatch.setattr(run_register_candidate, "get_config", lambda: mock_cfg)

        run_register_candidate.main()

        receipt_path = tmp_path / "reports" / "candidate_registration.json"
        assert receipt_path.exists()
        receipt = json.loads(receipt_path.read_text())
        assert "health" in receipt and "restoration" in receipt
        assert receipt["health"]["champion_set"] is False
        assert receipt["restoration"]["champion_set"] is False

    def test_main_returns_1_on_exception(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from src.models import run_register_candidate

        def fail_run(task, promote, cfg):
            raise RuntimeError("MLflow connection failed")

        mock_cfg = MagicMock()
        mock_cfg.paths.reports_dir = tmp_path / "reports"

        monkeypatch.setattr(run_register_candidate, "run_register_and_promote", fail_run)
        monkeypatch.setattr(run_register_candidate, "get_config", lambda: mock_cfg)

        rc = run_register_candidate.main()
        assert rc == 1

    def test_receipt_champion_set_is_false(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Even if the mock erroneously sets champion_set=True, the pipeline
        contract must hold: we verify run_register_and_promote is called with
        promote=False so the registry code never promotes."""
        from src.models import run_register_candidate

        promote_values: list[bool] = []

        def tracking_run(task, promote, cfg):
            promote_values.append(promote)
            return {
                "health": {
                    "registered_model_name": "m",
                    "version": "3",
                    "algo_name": "lr",
                    "cv_macro_f1": 0.5,
                    "gate_passed": False,
                    "gate_failures": ["min_cv_macro_f1"],
                    "champion_set": False,
                },
                "restoration": {
                    "registered_model_name": "m2",
                    "version": "3",
                    "algo_name": "xgb",
                    "cv_macro_f1": 0.5,
                    "gate_passed": False,
                    "gate_failures": ["min_cv_macro_f1"],
                    "champion_set": False,
                },
            }

        mock_cfg = MagicMock()
        mock_cfg.paths.reports_dir = tmp_path / "reports"

        monkeypatch.setattr(run_register_candidate, "run_register_and_promote", tracking_run)
        monkeypatch.setattr(run_register_candidate, "get_config", lambda: mock_cfg)

        run_register_candidate.main()
        assert all(p is False for p in promote_values)


# ---------------------------------------------------------------------------
# dvc.yaml structural tests
# ---------------------------------------------------------------------------


class TestDvcYaml:
    """Verify that dvc.yaml is well-formed and contains the required stages."""

    REQUIRED_STAGES = {
        "generate",
        "validate",
        "preprocess",
        "train",
        "evaluate",
        "register_candidate",
        "run_drift",
    }

    @pytest.fixture(autouse=True)
    def load_dvc_yaml(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        dvc_path = project_root / "dvc.yaml"
        with dvc_path.open(encoding="utf-8") as fh:
            self.dvc = yaml.safe_load(fh)

    def test_stages_present(self) -> None:
        assert "stages" in self.dvc
        assert self.REQUIRED_STAGES == set(self.dvc["stages"].keys())

    def test_generate_has_required_keys(self) -> None:
        stage = self.dvc["stages"]["generate"]
        assert "cmd" in stage
        assert "deps" in stage
        assert "params" in stage
        assert "outs" in stage

    def test_validate_depends_on_observations_csv(self) -> None:
        stage = self.dvc["stages"]["validate"]
        deps_flat = [d if isinstance(d, str) else list(d.keys())[0] for d in stage["deps"]]
        assert any("observations.csv" in d for d in deps_flat)

    def test_preprocess_depends_on_validated_csv(self) -> None:
        stage = self.dvc["stages"]["preprocess"]
        deps_flat = [d if isinstance(d, str) else list(d.keys())[0] for d in stage["deps"]]
        assert any("observations_validated.csv" in d for d in deps_flat)

    def test_train_outputs_model_files(self) -> None:
        stage = self.dvc["stages"]["train"]
        outs_flat = [o if isinstance(o, str) else list(o.keys())[0] for o in stage["outs"]]
        assert any("best_model_health.joblib" in o for o in outs_flat)
        assert any("best_model_restoration.joblib" in o for o in outs_flat)

    def test_evaluate_has_metrics(self) -> None:
        stage = self.dvc["stages"]["evaluate"]
        assert "metrics" in stage

    def test_register_candidate_cmd_does_not_contain_promote(self) -> None:
        """The pipeline stage must not promote champions."""
        cmd = self.dvc["stages"]["register_candidate"]["cmd"]
        assert "--promote" not in cmd

    def test_dag_order_validate_after_generate(self) -> None:
        """validate must depend on the generate output."""
        validate_deps = self.dvc["stages"]["validate"]["deps"]
        deps_flat = [d if isinstance(d, str) else list(d.keys())[0] for d in validate_deps]
        assert any("observations.csv" in d for d in deps_flat)

    def test_no_absolute_paths_in_cmds(self) -> None:
        """Stage commands must use a portable PATH-resolved interpreter."""
        for name, stage in self.dvc["stages"].items():
            cmd = stage.get("cmd", "")
            assert cmd.startswith("python -m "), (
                f"Stage '{name}' cmd must use PATH-resolved python: {cmd!r}"
            )
            assert ".venv/bin/python" not in cmd


# ---------------------------------------------------------------------------
# params.yaml DVC key tests
# ---------------------------------------------------------------------------


class TestParamsYaml:
    """Verify that params.yaml contains the keys referenced in dvc.yaml."""

    @pytest.fixture(autouse=True)
    def load_params(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        params_path = project_root / "params.yaml"
        with params_path.open(encoding="utf-8") as fh:
            self.params = yaml.safe_load(fh)

    def test_base_section_present(self) -> None:
        assert "base" in self.params
        assert "random_seed" in self.params["base"]

    def test_data_section_present(self) -> None:
        assert "data" in self.params
        for key in ("n_samples", "regions", "noise_scale"):
            assert key in self.params["data"], f"Missing data.{key}"

    def test_split_section_present(self) -> None:
        assert "split" in self.params
        for key in ("test_size", "val_size", "stratify"):
            assert key in self.params["split"], f"Missing split.{key}"

    def test_features_section_present(self) -> None:
        assert "features" in self.params
        for key in ("numeric", "categorical", "target_health", "target_restoration"):
            assert key in self.params["features"], f"Missing features.{key}"

    def test_models_section_present(self) -> None:
        assert "models" in self.params
        assert "cv_folds" in self.params["models"]
        assert "health" in self.params["models"]
        assert "restoration" in self.params["models"]

    def test_quality_gates_present(self) -> None:
        assert "quality_gates" in self.params
        assert "health" in self.params["quality_gates"]
        assert "restoration" in self.params["quality_gates"]
        assert "min_cv_macro_f1" in self.params["quality_gates"]["health"]

    def test_n_samples_is_15000(self) -> None:
        """Dataset size must not be accidentally changed during M7."""
        assert self.params["data"]["n_samples"] == 15000
