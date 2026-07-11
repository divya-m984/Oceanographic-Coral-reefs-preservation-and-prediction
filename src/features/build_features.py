"""
src/features/build_features.py — Derived feature engineering for CoralSense.

PROTOTYPE DISCLAIMER
--------------------
All feature engineering formulas below are prototype assumptions derived from
published coral-reef science literature.  They have NOT been validated against
real calibrated sensor data and must be treated as engineering approximations
only.  Constants (thresholds, weights, normalisation denominators) should be
reviewed and updated when real sensor data becomes available.

Derived features
----------------
Six new columns are computed from existing sensor readings.  Each formula is
documented below and in DERIVED_FEATURE_FORMULAS.

Leakage notes
-------------
Because CoralSense uses *synthetic* labels derived by an explicit scoring
function (see docs/data_dictionary.md), several sensor columns are
mechanistically linked to a target.  These columns are RETAINED as features
(they represent legitimate sensor readings in a real deployment) but are
flagged in SYNTHETIC_LEAKAGE_NOTES so downstream model-evaluation code can
account for the artificially high predictive power.

Cross-target leakage
--------------------
reef_health and restoration_suitability are both targets.  Neither is used as
a feature for the other task.  This is enforced by get_feature_columns().
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

# ---------------------------------------------------------------------------
# Domain constants (prototype values — see PROTOTYPE DISCLAIMER above)
# ---------------------------------------------------------------------------

BLEACHING_THRESHOLD_C: float = 28.5
"""Indian Ocean bleaching onset temperature (Hughes et al. 2017)."""

BLEACHING_NORMALISER_C: float = 4.5
"""°C range above threshold that maps thermal excess to [0, 1]."""

HEALTHY_DO_MG_L: float = 6.0
"""Dissolved oxygen lower bound for un-stressed reef water (mg/L)."""

DO_STRESS_NORMALISER: float = 3.5
"""DO deficit range that maps oxygen stress to [0, 1] (mg/L)."""

REFERENCE_PH: float = 8.1
"""Open-ocean seawater pH reference (total scale)."""

PH_STRESS_NORMALISER: float = 0.55
"""pH deviation that maps acidity stress to [0, 1]."""

TURBIDITY_STRESS_NORMALISER: float = 20.0
"""NTU that maps turbidity to [0, 1] for WQI calculation."""

RUGOSITY_MIN: float = 1.0
"""Minimum rugosity index (flat surface)."""

RUGOSITY_MAX: float = 10.0
"""Maximum declared rugosity index (Field bound in schema)."""

# Weights for water_quality_index components (must sum to 1.0)
_WQI_W_THERMAL: float = 0.40
_WQI_W_DO: float = 0.30
_WQI_W_PH: float = 0.20
_WQI_W_TURB: float = 0.10

# Weights for substrate_stability_score components (must sum to 1.0)
_SSS_W_HARD: float = 0.60
_SSS_W_RUG: float = 0.40

# ---------------------------------------------------------------------------
# Derived feature catalogue (name → formula description)
# ---------------------------------------------------------------------------

DERIVED_FEATURE_FORMULAS: dict[str, str] = {
    "thermal_stress_index": (
        "clip((water_temperature_c - 28.5) / 4.5, 0, 1)  "
        "→ fractional thermal excess above Indian Ocean bleaching onset (28.5 °C); "
        "0 = no stress, 1 = fully stressed at ≥33 °C."
    ),
    "oxygen_stress_index": (
        "clip((6.0 - dissolved_oxygen_mg_l) / 3.5, 0, 1)  "
        "→ fractional DO deficit below healthy reef value (6.0 mg/L); "
        "0 = no hypoxia, 1 = severely hypoxic (≤2.5 mg/L)."
    ),
    "acidity_deviation": (
        "abs(ph - 8.1)  "
        "→ absolute pH deviation from open-ocean reference (8.1); "
        "0 = optimal, higher = more acidic or alkaline stress."
    ),
    "water_quality_index": (
        "1 - clip(0.40*thermal_stress_index + 0.30*oxygen_stress_index "
        "+ 0.20*clip(acidity_deviation/0.55,0,1) + 0.10*clip(turbidity_ntu/20,0,1), 0, 1)  "
        "→ composite water quality score in [0,1]; higher = better quality."
    ),
    "substrate_stability_score": (
        "0.60*(hard_substrate_percentage/100) + 0.40*((rugosity_index-1)/9)  "
        "→ composite substrate quality in [0,1]; higher = more stable hard substrate."
    ),
    "structural_complexity_score": (
        "0.50*acoustic_complexity_index + 0.50*((rugosity_index-1)/9)  "
        "→ composite structural complexity in [0,1]; higher = richer 3-D reef architecture."
    ),
}

DERIVED_FEATURE_NAMES: list[str] = list(DERIVED_FEATURE_FORMULAS.keys())

# ---------------------------------------------------------------------------
# Leakage documentation
# ---------------------------------------------------------------------------

SYNTHETIC_LEAKAGE_NOTES: dict[str, dict] = {
    "reef_health": {
        "description": (
            "The synthetic reef_health label is a weighted composite of sensor "
            "readings (see docs/data_dictionary.md).  The following sensor columns "
            "are mechanistically linked to this label and will appear artificially "
            "predictive on this synthetic dataset.  They are retained because they "
            "represent legitimate sensor inputs in a real deployment."
        ),
        "high_leakage_columns": {
            "bleaching_percentage": "weight 0.25 in health score — strongest direct signal",
            "disease_percentage": "weight 0.12 in health score",
            "coral_cover_percentage": "weight 0.13 in health score",
            "water_temperature_c": "weight 0.20 in health score",
            "ph": "weight 0.12 in health score",
            "dissolved_oxygen_mg_l": "weight 0.08 in health score",
            "turbidity_ntu": "weight 0.10 in health score",
            "thermal_stress_index": "derived from exact health-score thermal formula",
            "oxygen_stress_index": "derived from exact health-score DO formula",
            "water_quality_index": "composite of health-score components",
        },
        "excluded_for_leakage": ["restoration_suitability"],
    },
    "restoration_suitability": {
        "description": (
            "The synthetic restoration_suitability label is a weighted composite of "
            "sensor readings.  The following columns are mechanistically linked to "
            "this label.  They are retained for the same reasons as above."
        ),
        "high_leakage_columns": {
            "hard_substrate_percentage": "weight 0.22 in restoration score — strongest direct signal",
            "coral_cover_percentage": "weight 0.10 in restoration score",
            "light_intensity": "weight 0.10 in restoration score",
            "water_temperature_c": "weight 0.15 in restoration score",
            "ph": "weight 0.12 in restoration score",
            "turbidity_ntu": "weight 0.12 in restoration score",
            "depth_m": "weight 0.09 in restoration score",
            "current_speed_m_s": "weight 0.10 in restoration score",
            "substrate_stability_score": "derived from hard_substrate_percentage + rugosity",
        },
        "excluded_for_leakage": ["reef_health"],
    },
}

# ---------------------------------------------------------------------------
# Column group definitions
# ---------------------------------------------------------------------------

METADATA_COLUMNS: list[str] = ["timestamp", "latitude", "longitude"]
"""Columns excluded from all feature sets (spatial / temporal identifiers)."""

TARGET_COLUMNS: list[str] = ["reef_health", "restoration_suitability"]
"""Both supervised targets — never used as input features."""

_RAW_NUMERIC_COLS: list[str] = [
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
]

NUMERIC_FEATURE_COLUMNS: list[str] = _RAW_NUMERIC_COLS + DERIVED_FEATURE_NAMES
"""All numeric feature columns (raw + derived) after build_features()."""

CATEGORICAL_FEATURE_COLUMNS: list[str] = ["region"]
"""Categorical feature columns."""

ALL_FEATURE_COLUMNS: list[str] = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS
"""Complete ordered feature column list (numeric then categorical)."""

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and append the six derived features to *df*.

    Returns a copy — the original DataFrame is not modified.

    All arithmetic uses vectorised pandas / NumPy operations and is safe for
    DataFrames that contain NaN values (intermediate NaNs propagate correctly).

    Parameters
    ----------
    df:
        DataFrame containing at minimum the raw CoralSense sensor columns.

    Returns
    -------
    DataFrame with six additional columns appended.
    """
    out = df.copy()

    # 1. thermal_stress_index ------------------------------------------------
    out["thermal_stress_index"] = (
        (out["water_temperature_c"] - BLEACHING_THRESHOLD_C) / BLEACHING_NORMALISER_C
    ).clip(0.0, 1.0)

    # 2. oxygen_stress_index -------------------------------------------------
    out["oxygen_stress_index"] = (
        (HEALTHY_DO_MG_L - out["dissolved_oxygen_mg_l"]) / DO_STRESS_NORMALISER
    ).clip(0.0, 1.0)

    # 3. acidity_deviation ---------------------------------------------------
    out["acidity_deviation"] = (out["ph"] - REFERENCE_PH).abs()

    # 4. water_quality_index -------------------------------------------------
    _turb_stress = (out["turbidity_ntu"] / TURBIDITY_STRESS_NORMALISER).clip(0.0, 1.0)
    _ph_stress = (out["acidity_deviation"] / PH_STRESS_NORMALISER).clip(0.0, 1.0)
    _raw_wqi = (
        _WQI_W_THERMAL * out["thermal_stress_index"]
        + _WQI_W_DO * out["oxygen_stress_index"]
        + _WQI_W_PH * _ph_stress
        + _WQI_W_TURB * _turb_stress
    )
    out["water_quality_index"] = (1.0 - _raw_wqi).clip(0.0, 1.0)

    # 5. substrate_stability_score -------------------------------------------
    _hard_norm = out["hard_substrate_percentage"] / 100.0
    _rug_norm = (out["rugosity_index"] - RUGOSITY_MIN) / (RUGOSITY_MAX - RUGOSITY_MIN)
    out["substrate_stability_score"] = (_SSS_W_HARD * _hard_norm + _SSS_W_RUG * _rug_norm).clip(
        0.0, 1.0
    )

    # 6. structural_complexity_score -----------------------------------------
    out["structural_complexity_score"] = (
        0.5 * out["acoustic_complexity_index"] + 0.5 * _rug_norm
    ).clip(0.0, 1.0)

    return out


# ---------------------------------------------------------------------------
# Column-list helpers
# ---------------------------------------------------------------------------


def get_feature_columns(
    task: Literal["health", "restoration"],
) -> list[str]:
    """
    Return the ordered feature column list for *task*.

    Cross-target leakage is prevented by this function:
    - health features exclude ``restoration_suitability``
    - restoration features exclude ``reef_health``

    Parameters
    ----------
    task:
        ``"health"`` or ``"restoration"``.

    Returns
    -------
    Ordered list of column names to use as model inputs.
    """
    if task not in ("health", "restoration"):
        raise ValueError(f"task must be 'health' or 'restoration'; got {task!r}")
    # Both tasks share the same ALL_FEATURE_COLUMNS list; targets and metadata
    # are never included in that list, so no further filtering needed.
    return list(ALL_FEATURE_COLUMNS)


def get_target_column(task: Literal["health", "restoration"]) -> str:
    """Return the target column name for *task*."""
    if task == "health":
        return "reef_health"
    if task == "restoration":
        return "restoration_suitability"
    raise ValueError(f"task must be 'health' or 'restoration'; got {task!r}")


def build_feature_metadata() -> dict:
    """
    Build a serialisable dict describing all feature engineering decisions.

    Intended to be saved as ``data/processed/feature_metadata.json`` so that
    later model-training and API stages can reconstruct column lists without
    reimporting this module.
    """
    return {
        "metadata_columns": METADATA_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "raw_numeric_columns": _RAW_NUMERIC_COLS,
        "derived_feature_names": DERIVED_FEATURE_NAMES,
        "numeric_feature_columns": NUMERIC_FEATURE_COLUMNS,
        "categorical_feature_columns": CATEGORICAL_FEATURE_COLUMNS,
        "all_feature_columns": ALL_FEATURE_COLUMNS,
        "derived_feature_formulas": DERIVED_FEATURE_FORMULAS,
        "synthetic_leakage_notes": SYNTHETIC_LEAKAGE_NOTES,
        "domain_constants": {
            "BLEACHING_THRESHOLD_C": BLEACHING_THRESHOLD_C,
            "BLEACHING_NORMALISER_C": BLEACHING_NORMALISER_C,
            "HEALTHY_DO_MG_L": HEALTHY_DO_MG_L,
            "DO_STRESS_NORMALISER": DO_STRESS_NORMALISER,
            "REFERENCE_PH": REFERENCE_PH,
            "PH_STRESS_NORMALISER": PH_STRESS_NORMALISER,
            "TURBIDITY_STRESS_NORMALISER": TURBIDITY_STRESS_NORMALISER,
            "RUGOSITY_MIN": RUGOSITY_MIN,
            "RUGOSITY_MAX": RUGOSITY_MAX,
        },
    }
