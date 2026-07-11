"""
tests/test_config.py — Unit tests for src/config.py (Milestone 1).

These tests require only: pytest, pyyaml, (optionally python-dotenv).
No ML or heavy dependencies are needed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Paths tests
# ---------------------------------------------------------------------------
class TestPaths:
    """Verify Paths dataclass builds correct paths from project root."""

    def test_from_root_returns_paths_instance(self) -> None:
        from src.config import Paths

        paths = Paths.from_root(PROJECT_ROOT)
        assert isinstance(paths, Paths)

    def test_project_root_matches(self) -> None:
        from src.config import Paths

        paths = Paths.from_root(PROJECT_ROOT)
        assert paths.project_root == PROJECT_ROOT

    def test_data_dir_is_inside_project_root(self) -> None:
        from src.config import Paths

        paths = Paths.from_root(PROJECT_ROOT)
        assert paths.data_dir == PROJECT_ROOT / "data"

    def test_all_subdirs_are_under_data(self) -> None:
        from src.config import Paths

        paths = Paths.from_root(PROJECT_ROOT)
        for attr in (
            "raw_data_dir",
            "processed_data_dir",
            "reference_data_dir",
            "production_data_dir",
        ):
            p: Path = getattr(paths, attr)
            assert str(p).startswith(str(paths.data_dir)), f"{attr} is not under data_dir"

    def test_paths_are_absolute(self) -> None:
        from src.config import Paths

        paths = Paths.from_root(PROJECT_ROOT)
        for fld in paths.__dataclass_fields__:
            p: Path = getattr(paths, fld)
            assert p.is_absolute(), f"{fld} is not absolute: {p}"

    def test_ensure_dirs_creates_directories(self, tmp_path: Path) -> None:
        from src.config import Paths

        paths = Paths.from_root(tmp_path)
        paths.ensure_dirs()
        for fld in paths.__dataclass_fields__:
            p: Path = getattr(paths, fld)
            assert p.exists(), f"Directory was not created: {fld}"


# ---------------------------------------------------------------------------
# Config.load() tests
# ---------------------------------------------------------------------------
class TestConfigLoad:
    """Verify Config loads params.yaml correctly."""

    def test_config_loads_without_error(self) -> None:
        from src.config import Config

        cfg = Config.load(PROJECT_ROOT)
        assert cfg is not None

    def test_random_seed_is_integer(self) -> None:
        from src.config import Config

        cfg = Config.load(PROJECT_ROOT)
        assert isinstance(cfg.random_seed, int)
        assert cfg.random_seed == 42

    def test_n_samples_is_positive_integer(self) -> None:
        from src.config import Config

        cfg = Config.load(PROJECT_ROOT)
        assert isinstance(cfg.n_samples, int)
        assert cfg.n_samples >= 10_000

    def test_regions_is_list_of_strings(self) -> None:
        from src.config import Config

        cfg = Config.load(PROJECT_ROOT)
        assert isinstance(cfg.regions, list)
        assert len(cfg.regions) >= 1
        for r in cfg.regions:
            assert isinstance(r, str)

    def test_expected_regions_present(self) -> None:
        from src.config import Config

        cfg = Config.load(PROJECT_ROOT)
        expected = {
            "Lakshadweep",
            "Gulf of Mannar",
            "Gulf of Kutch",
            "Andaman and Nicobar Islands",
        }
        assert expected.issubset(set(cfg.regions))

    def test_test_size_is_fraction(self) -> None:
        from src.config import Config

        cfg = Config.load(PROJECT_ROOT)
        assert 0.0 < cfg.test_size < 1.0

    def test_health_classes_complete(self) -> None:
        from src.config import Config

        cfg = Config.load(PROJECT_ROOT)
        expected = {"healthy", "stressed", "bleached", "severely_degraded"}
        assert expected == set(cfg.health_classes)

    def test_restoration_classes_complete(self) -> None:
        from src.config import Config

        cfg = Config.load(PROJECT_ROOT)
        expected = {"suitable", "moderately_suitable", "unsuitable"}
        assert expected == set(cfg.restoration_classes)

    def test_numeric_features_non_empty(self) -> None:
        from src.config import Config

        cfg = Config.load(PROJECT_ROOT)
        assert len(cfg.numeric_features) >= 15

    def test_paths_attribute_is_paths_instance(self) -> None:
        from src.config import Config, Paths

        cfg = Config.load(PROJECT_ROOT)
        assert isinstance(cfg.paths, Paths)

    def test_raw_data_path_is_csv(self) -> None:
        from src.config import Config

        cfg = Config.load(PROJECT_ROOT)
        assert cfg.raw_data_path.suffix == ".csv"

    def test_mlflow_experiment_names_non_empty(self) -> None:
        from src.config import Config

        cfg = Config.load(PROJECT_ROOT)
        assert cfg.mlflow_experiment_health
        assert cfg.mlflow_experiment_restoration

    def test_drift_threshold_is_fraction(self) -> None:
        from src.config import Config

        cfg = Config.load(PROJECT_ROOT)
        assert 0.0 < cfg.drift_threshold < 1.0

    def test_config_raises_on_missing_params_yaml(self, tmp_path: Path) -> None:
        from src.config import Config

        with pytest.raises(FileNotFoundError, match="params.yaml"):
            Config.load(tmp_path)


# ---------------------------------------------------------------------------
# Singleton tests
# ---------------------------------------------------------------------------
class TestConfigSingleton:
    """Verify get_config / reset_config behave correctly."""

    def test_get_config_returns_config(self) -> None:
        from src.config import get_config, reset_config

        reset_config()
        cfg = get_config()
        from src.config import Config

        assert isinstance(cfg, Config)

    def test_get_config_returns_same_instance(self) -> None:
        from src.config import get_config, reset_config

        reset_config()
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_reset_config_forces_reload(self) -> None:
        from src.config import get_config, reset_config

        reset_config()
        cfg1 = get_config()
        reset_config()
        cfg2 = get_config()
        # Different instances but equal content
        assert cfg1 is not cfg2
        assert cfg1.random_seed == cfg2.random_seed


# ---------------------------------------------------------------------------
# setup_logging tests
# ---------------------------------------------------------------------------
class TestSetupLogging:
    """Verify logging configuration."""

    def test_returns_logger(self) -> None:
        from src.config import setup_logging

        logger = setup_logging("test.coralsense")
        assert isinstance(logger, logging.Logger)

    def test_logger_name_matches(self) -> None:
        from src.config import setup_logging

        logger = setup_logging("coralsense.test_unit")
        assert logger.name == "coralsense.test_unit"

    def test_debug_level_accepted(self) -> None:
        from src.config import setup_logging

        logger = setup_logging("coralsense.debug_test", level="DEBUG")
        assert logger.isEnabledFor(logging.DEBUG)

    def test_invalid_level_falls_back_to_info(self) -> None:
        from src.config import setup_logging

        # Should not raise; invalid level silently falls back to INFO
        logger = setup_logging("coralsense.fallback", level="NOTAVALIDLEVEL")
        assert isinstance(logger, logging.Logger)
