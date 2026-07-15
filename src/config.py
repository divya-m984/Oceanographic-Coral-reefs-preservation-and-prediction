"""
config.py — Central configuration for CoralSense MLOps.

Responsibilities:
  - Locate the project root via __file__ (no hardcoded paths).
  - Load params.yaml into a typed Config dataclass.
  - Expose Path constants for every data / model / report directory.
  - Provide a setup_logging() helper used throughout the project.
  - Read runtime overrides from the .env file (via python-dotenv).

Usage:
    from src.config import Config, Paths, setup_logging
    logger = setup_logging(__name__)
    cfg = Config.load()
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Project root — two levels up from this file:  src/config.py → src/ → root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(root: Path) -> None:
    """Load .env if python-dotenv is available (optional dependency in M1)."""
    env_path = root / ".env"
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]

        load_dotenv(env_path, override=False)
    except ModuleNotFoundError:
        pass  # dotenv is not installed yet; environment variables still work


_load_dotenv(_PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Raw YAML loader (used before Config dataclass is available)
# ---------------------------------------------------------------------------
def _read_params(root: Path) -> dict[str, Any]:
    """Load params.yaml from the project root."""
    params_path = root / "params.yaml"
    if not params_path.exists():
        raise FileNotFoundError(f"params.yaml not found at {params_path}")
    with params_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError("params.yaml must contain a YAML mapping at the top level.")
    return data


# ---------------------------------------------------------------------------
# Paths — all paths derived from the single project root constant
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Paths:
    """Filesystem paths used throughout the project."""

    project_root: Path
    data_dir: Path
    raw_data_dir: Path
    processed_data_dir: Path
    reference_data_dir: Path
    production_data_dir: Path
    models_dir: Path
    reports_dir: Path
    artifacts_dir: Path
    scripts_dir: Path
    notebooks_dir: Path

    @classmethod
    def from_root(cls, root: Path) -> Paths:
        data = root / "data"
        return cls(
            project_root=root,
            data_dir=data,
            raw_data_dir=data / "raw",
            processed_data_dir=data / "processed",
            reference_data_dir=data / "reference",
            production_data_dir=data / "production",
            models_dir=root / "models",
            reports_dir=root / "reports",
            artifacts_dir=root / "artifacts",
            scripts_dir=root / "scripts",
            notebooks_dir=root / "notebooks",
        )

    def ensure_dirs(self) -> None:
        """Create all directories if they do not already exist."""
        for fld in self.__dataclass_fields__:  # type: ignore[attr-defined]
            path: Path = getattr(self, fld)
            path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Config — structured access to every section of params.yaml
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Typed wrapper around params.yaml + environment variables."""

    # ── params.yaml sections ────────────────────────────────────────────────
    random_seed: int
    project_name: str
    version: str

    n_samples: int
    raw_filename: str
    regions: list[str]
    region_bounds: dict[str, list[float]]
    noise_scale: float

    test_size: float
    val_size: float
    stratify: bool

    numeric_features: list[str]
    categorical_features: list[str]
    target_health: str
    target_restoration: str
    health_classes: list[str]
    restoration_classes: list[str]

    model_params: dict[str, Any]
    cv_folds: int

    mlflow_experiment_health: str
    mlflow_experiment_restoration: str
    mlflow_registered_health: str
    mlflow_registered_restoration: str
    mlflow_champion_alias: str
    quality_gates: dict[str, Any]

    drift_threshold: float
    reference_filename: str
    production_filename: str
    drift_report_filename: str
    production_shift_fraction: float
    drift_summary_filename: str
    drift_report_filename_health: str
    drift_report_filename_restoration: str
    reference_n: int
    production_n: int
    shift_scale: float

    shap_background_samples: int
    retraining: dict[str, Any]

    # ── Paths ────────────────────────────────────────────────────────────────
    paths: Paths = field(default_factory=lambda: Paths.from_root(_PROJECT_ROOT))

    # ── Runtime / environment ────────────────────────────────────────────────
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    mlflow_tracking_uri: str = field(
        default_factory=lambda: os.getenv(
            "MLFLOW_TRACKING_URI",
            f"sqlite:///{_PROJECT_ROOT / 'artifacts' / 'mlruns.db'}",
        )
    )
    mlflow_artifact_root: str = field(
        default_factory=lambda: os.getenv("MLFLOW_ARTIFACT_ROOT", "./artifacts/mlartifacts")
    )
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))

    @classmethod
    def load(cls, project_root: Path = _PROJECT_ROOT) -> Config:
        """Load config from params.yaml, resolving all paths relative to root."""
        p = _read_params(project_root)
        paths = Paths.from_root(project_root)

        return cls(
            # base
            random_seed=p["base"]["random_seed"],
            project_name=p["base"]["project_name"],
            version=p["base"]["version"],
            # data
            n_samples=p["data"]["n_samples"],
            raw_filename=p["data"]["raw_filename"],
            regions=p["data"]["regions"],
            region_bounds=p["data"]["region_bounds"],
            noise_scale=p["data"]["noise_scale"],
            # split
            test_size=p["split"]["test_size"],
            val_size=p["split"]["val_size"],
            stratify=p["split"]["stratify"],
            # features
            numeric_features=p["features"]["numeric"],
            categorical_features=p["features"]["categorical"],
            target_health=p["features"]["target_health"],
            target_restoration=p["features"]["target_restoration"],
            health_classes=p["features"]["health_classes"],
            restoration_classes=p["features"]["restoration_classes"],
            # models
            model_params=p["models"],
            cv_folds=p["models"]["cv_folds"],
            # mlflow
            mlflow_experiment_health=p["mlflow"]["experiment_health"],
            mlflow_experiment_restoration=p["mlflow"]["experiment_restoration"],
            mlflow_registered_health=p["mlflow"]["registered_model_health"],
            mlflow_registered_restoration=p["mlflow"]["registered_model_restoration"],
            mlflow_champion_alias=p["mlflow"]["champion_alias"],
            quality_gates=p["quality_gates"],
            # monitoring
            drift_threshold=p["monitoring"]["drift_threshold"],
            reference_filename=p["monitoring"]["reference_filename"],
            production_filename=p["monitoring"]["production_filename"],
            drift_report_filename=p["monitoring"]["report_filename_health"],
            production_shift_fraction=p["monitoring"]["production_shift_fraction"],
            drift_summary_filename=p["monitoring"]["summary_filename"],
            drift_report_filename_health=p["monitoring"]["report_filename_health"],
            drift_report_filename_restoration=p["monitoring"]["report_filename_restoration"],
            reference_n=p["monitoring"]["reference_n"],
            production_n=p["monitoring"]["production_n"],
            shift_scale=p["monitoring"]["shift_scale"],
            # shap
            shap_background_samples=p["shap"]["background_samples"],
            # retraining
            retraining=p.get("retraining", {}),
            # paths
            paths=paths,
        )

    @property
    def raw_data_path(self) -> Path:
        return self.paths.raw_data_dir / self.raw_filename

    @property
    def reference_data_path(self) -> Path:
        return self.paths.reference_data_dir / self.reference_filename

    @property
    def production_data_path(self) -> Path:
        return self.paths.production_data_dir / self.production_filename

    @property
    def drift_report_path(self) -> Path:
        return self.paths.reports_dir / self.drift_report_filename_health

    @property
    def drift_summary_path(self) -> Path:
        return self.paths.reports_dir / self.drift_summary_filename

    @property
    def monitoring(self) -> dict:
        """Return monitoring params as a dict (for generate_production.py and drift.py)."""
        return {
            "drift_threshold": self.drift_threshold,
            "reference_filename": self.reference_filename,
            "production_filename": self.production_filename,
            "summary_filename": self.drift_summary_filename,
            "report_filename_health": self.drift_report_filename_health,
            "report_filename_restoration": self.drift_report_filename_restoration,
            "reference_n": self.reference_n,
            "production_n": self.production_n,
            "shift_scale": self.shift_scale,
            "production_shift_fraction": self.production_shift_fraction,
        }


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(
    name: str = "coralsense",
    level: str | None = None,
) -> logging.Logger:
    """
    Configure root logging and return a named logger.

    Parameters
    ----------
    name:
        Logger name (use __name__ from the calling module).
    level:
        Override the log level; defaults to LOG_LEVEL env var, then INFO.
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,  # reconfigure even if already set (useful in tests)
    )
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Module-level convenience singleton — import and use directly
# ---------------------------------------------------------------------------
# Instantiated lazily so that importing config.py never fails even when
# params.yaml is temporarily absent during test isolation.
_config_singleton: Config | None = None


def get_config() -> Config:
    """Return the cached Config singleton, loading it on first call."""
    global _config_singleton
    if _config_singleton is None:
        _config_singleton = Config.load()
    return _config_singleton


def reset_config() -> None:
    """Clear the cached singleton (used in tests that mutate params)."""
    global _config_singleton
    _config_singleton = None
