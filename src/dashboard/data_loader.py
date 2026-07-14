"""
src/dashboard/data_loader.py — Read-only dataset and evaluation loading.

All functions are:
- Read-only: no file is ever written or mutated.
- Cached: Streamlit's @st.cache_data decorator is used so that large files
  are loaded once per session.
- Path-safe: all paths are derived from the project root; no absolute
  user-home paths appear in error messages.
- Column-validated: a missing-column error is raised with a clear message.

The original DataFrame returned by the cache is never mutated; callers that
need to modify data must call .copy() explicitly or use sample_for_display().
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path constants (relative to project root, never exposed as absolute paths)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RAW_CSV = _PROJECT_ROOT / "data" / "raw" / "observations.csv"
_EVAL_DIR = _PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# Expected dataset columns
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "timestamp",
        "latitude",
        "longitude",
        "region",
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
        "reef_health",
        "restoration_suitability",
    }
)

INFERENCE_FEATURES: tuple[str, ...] = (
    "region",
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
)

TARGET_COLUMNS: frozenset[str] = frozenset({"reef_health", "restoration_suitability"})


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_data
def load_observations() -> pd.DataFrame:
    """
    Load and return the synthetic observation dataset.

    The result is cached for the lifetime of the Streamlit session.
    Do not mutate the returned DataFrame; call .copy() if needed.

    Raises
    ------
    FileNotFoundError
        If the CSV is absent (dataset not yet generated).
    ValueError
        If required columns are missing from the file.
    """
    try:
        rel: Path | str = _RAW_CSV.relative_to(_PROJECT_ROOT)
    except ValueError:
        rel = _RAW_CSV
    if not _RAW_CSV.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{rel}'. Run 'python -m src.data.generate_data' to generate it."
        )
    df = pd.read_csv(_RAW_CSV)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset at '{rel}' is missing required columns: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


@st.cache_data
def load_evaluation(task: str) -> dict[str, Any] | None:
    """
    Load the evaluation JSON for *task* ('health' or 'restoration').

    Returns None (rather than raising) when the file is absent, so pages
    can degrade gracefully without crashing the dashboard.
    """
    path = _EVAL_DIR / f"evaluation_{task}.json"
    if not path.exists():
        return None
    with path.open() as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Sampling helper
# ---------------------------------------------------------------------------


def sample_for_display(
    df: pd.DataFrame,
    max_n: int = 2_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Return a bounded copy of *df* suitable for rendering.

    If ``len(df) <= max_n`` the full DataFrame (copy) is returned.
    Never mutates the original.
    """
    if len(df) <= max_n:
        return df.copy()
    return df.sample(n=max_n, random_state=seed).copy()
