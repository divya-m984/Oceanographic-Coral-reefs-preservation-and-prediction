"""
src/data/generate_data.py — Synthetic marine sensor data generator for CoralSense.

DISCLAIMER
----------
ALL DATA PRODUCED BY THIS MODULE IS SYNTHETIC. It is generated using
documented statistical rules and scientific approximations for the sole purpose
of prototyping an MLOps pipeline. These observations do not represent real
sensor readings and MUST NOT be used to guide actual coral reef conservation,
marine management, or environmental policy decisions.

Scientific basis
----------------
Parameter ranges and correlation directions are based on published literature
for Indian reef systems:
  - Venkataraman et al. (2011) "Coral Reefs of India" (ENVIS Report)
  - NCSCM (National Centre for Sustainable Coastal Management) reef surveys
  - Sheppard et al. (2010) "Coral Reefs of the Indian Ocean"
  - Hughes et al. (2017) "Global warming and recurrent mass bleaching of corals"
    Nature 543, 373-377.
  - Claar & Baum (2019) "Timing matters: survey timing during bleaching events
    can influence reef assessment." PLOS ONE.

All thresholds, distributions and correlation strengths are approximations
carrying large uncertainty. The architecture is designed so this module can be
replaced by a real sensor-ingestion pipeline without changing downstream code.

Label generation strategy
-------------------------
A composite *stress score* (reef health) and *suitability score* (restoration)
are computed from the generated feature values according to documented, weighted
rules. Gaussian noise is added to every score before thresholding to ensure:
  1. Overlapping class boundaries — no class is perfectly separable.
  2. Biological variability — identical environmental readings can yield
     different reef states.
  3. No single input feature perfectly predicts either target.

Usage
-----
  # From project root (venv active):
  python -m src.data.generate_data
  python -m src.data.generate_data --samples 5000 --seed 7 --output /tmp/test.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config, get_config, reset_config, setup_logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Region-level latent variable distributions
# ---------------------------------------------------------------------------
# Each region has a *stress index* (0 = pristine, 1 = severely degraded) and a
# *structure index* (0 = degraded soft substrate, 1 = excellent reef structure)
# modelled as Beta distributions.  mean = alpha / (alpha + beta).
#
# Andaman: least stressed, best structure (remote, large marine park)
# Lakshadweep: moderately healthy (atoll, some anthropogenic pressure)
# Gulf of Mannar: mixed (high fishing pressure, turbidity, some bleaching)
# Gulf of Kutch: most stressed (high salinity, turbidity, industry)

_REGION_STRESS: dict[str, dict[str, float]] = {
    "Lakshadweep": {"alpha": 1.8, "beta": 3.5},  # mean ≈ 0.34
    "Gulf of Mannar": {"alpha": 2.5, "beta": 2.5},  # mean ≈ 0.50
    "Gulf of Kutch": {"alpha": 4.0, "beta": 2.0},  # mean ≈ 0.67
    "Andaman and Nicobar Islands": {"alpha": 1.5, "beta": 4.0},  # mean ≈ 0.27
}

_REGION_STRUCTURE: dict[str, dict[str, float]] = {
    "Lakshadweep": {"alpha": 3.5, "beta": 1.8},  # mean ≈ 0.66
    "Gulf of Mannar": {"alpha": 2.5, "beta": 2.5},  # mean ≈ 0.50
    "Gulf of Kutch": {"alpha": 1.5, "beta": 3.5},  # mean ≈ 0.30
    "Andaman and Nicobar Islands": {"alpha": 4.0, "beta": 1.8},  # mean ≈ 0.69
}

# ---------------------------------------------------------------------------
# Per-region environmental baseline ranges  (min, max)
# ---------------------------------------------------------------------------
# All values represent observed field conditions before stress/structure
# modification.  Sources as listed in the module docstring.
_REGION_ENV: dict[str, dict[str, tuple[float, float]]] = {
    "Lakshadweep": {
        "depth_m": (0.5, 25.0),
        "base_temp_c": (27.5, 30.5),
        "base_ph": (8.00, 8.25),
        "base_salinity": (34.5, 36.0),
        "base_do": (5.5, 8.5),
        "base_turbidity": (0.2, 4.0),
        "base_current": (0.02, 0.50),
    },
    "Gulf of Mannar": {
        "depth_m": (0.5, 20.0),
        "base_temp_c": (27.0, 32.0),
        "base_ph": (7.90, 8.20),
        "base_salinity": (33.0, 36.0),
        "base_do": (5.0, 8.0),
        "base_turbidity": (1.0, 15.0),
        "base_current": (0.02, 0.60),
    },
    "Gulf of Kutch": {
        "depth_m": (0.5, 15.0),
        "base_temp_c": (20.0, 36.0),
        "base_ph": (7.75, 8.15),
        "base_salinity": (36.0, 43.0),
        "base_do": (3.5, 7.5),
        "base_turbidity": (3.0, 28.0),
        "base_current": (0.05, 0.90),
    },
    "Andaman and Nicobar Islands": {
        "depth_m": (0.5, 38.0),
        "base_temp_c": (27.0, 30.0),
        "base_ph": (8.05, 8.35),
        "base_salinity": (32.5, 35.0),
        "base_do": (6.0, 9.0),
        "base_turbidity": (0.1, 5.0),
        "base_current": (0.01, 0.50),
    },
}

# ---------------------------------------------------------------------------
# Temporal range
# ---------------------------------------------------------------------------
_TS_START = pd.Timestamp("2018-01-01")
_TS_END = pd.Timestamp("2024-12-31 23:59:59")
_TOTAL_SECONDS = int((_TS_END - _TS_START).total_seconds())

# ---------------------------------------------------------------------------
# Class labels and scoring thresholds
# ---------------------------------------------------------------------------
# Health score ∈ [0, 1]: 0 = pristine, 1 = worst degraded.
# The score is a weighted sum of normalised stress indicators; see
# _compute_health_score() for full documentation.
_HEALTH_THRESHOLDS: tuple[float, float, float] = (0.28, 0.47, 0.66)
# score < 0.28                     → healthy
# 0.28 ≤ score < 0.47              → stressed
# 0.47 ≤ score < 0.66              → bleached
# score ≥ 0.66                     → severely_degraded

# Restoration score ∈ [0, 1]: 0 = unsuitable, 1 = ideal.
# See _compute_restoration_score() for full documentation.
_RESTORATION_THRESHOLDS: tuple[float, float] = (0.40, 0.63)
# score < 0.40                     → unsuitable
# 0.40 ≤ score < 0.63              → moderately_suitable
# score ≥ 0.63                     → suitable

# ---------------------------------------------------------------------------
# Sample distribution across regions (approximating reef area proportions)
# Andaman > Lakshadweep ≈ Gulf of Mannar > Gulf of Kutch
# Reference: Wildlife Institute of India reef area estimates (2015).
# ---------------------------------------------------------------------------
_REGION_WEIGHTS: dict[str, float] = {
    "Lakshadweep": 0.23,
    "Gulf of Mannar": 0.20,
    "Gulf of Kutch": 0.15,
    "Andaman and Nicobar Islands": 0.42,
}


# ---------------------------------------------------------------------------
# Internal: per-region feature generation
# ---------------------------------------------------------------------------


def _generate_region_block(
    rng: np.random.Generator,
    n: int,
    region: str,
    region_bounds: dict[str, list[float]],
    noise_scale: float,
) -> dict[str, np.ndarray]:
    """
    Generate ``n`` synthetic sensor observations for a single reef region.

    Returns a dict mapping column names to numpy arrays.  Internal latent
    variables are prefixed with ``_`` and are stripped before building the
    final DataFrame.

    Parameters
    ----------
    rng:
        Seeded numpy random generator.
    n:
        Number of observations to generate.
    region:
        One of the four supported region names.
    region_bounds:
        Dict mapping region names to [lat_min, lat_max, lon_min, lon_max].
    noise_scale:
        Fractional noise coefficient (from params.yaml).  Controls how much
        independent Gaussian noise is added to each feature.
    """
    env = _REGION_ENV[region]
    lat_min, lat_max, lon_min, lon_max = region_bounds[region]

    # ── Latent variables ────────────────────────────────────────────────────
    # stress ∈ [0, 1]: drives temperature excess, bleaching, disease, pH drop
    stress = rng.beta(
        _REGION_STRESS[region]["alpha"],
        _REGION_STRESS[region]["beta"],
        n,
    )

    # structure ∈ [0, 1]: drives substrate hardness, rugosity, sonar return
    structure = rng.beta(
        _REGION_STRUCTURE[region]["alpha"],
        _REGION_STRUCTURE[region]["beta"],
        n,
    )

    # clarity ∈ [0, 1]: derived from stress and structure
    # Low-stress, well-structured reefs tend to have clearer water.
    clarity = np.clip(
        1.0 - 0.55 * stress + 0.25 * structure + rng.normal(0.0, 0.10, n),
        0.0,
        1.0,
    )

    # ── Spatial ─────────────────────────────────────────────────────────────
    latitude = rng.uniform(lat_min, lat_max, n)
    longitude = rng.uniform(lon_min, lon_max, n)

    # ── Timestamps (uniform over 2018-01-01 to 2024-12-31) ─────────────────
    rand_secs = rng.integers(0, _TOTAL_SECONDS, n).astype("int64")
    timestamps = _TS_START + pd.to_timedelta(rand_secs, unit="s")

    # ── Depth ───────────────────────────────────────────────────────────────
    d_min, d_max = env["depth_m"]
    depth_m = rng.uniform(d_min, d_max, n)

    # ── Water temperature ───────────────────────────────────────────────────
    # Higher stress pushes temperature toward the upper end of the regional
    # range, simulating thermal anomaly events.
    t_min, t_max = env["base_temp_c"]
    t_range = t_max - t_min
    base_temp = rng.uniform(t_min, t_max, n)
    thermal_push = stress * 0.40 * t_range
    water_temperature_c = np.clip(
        base_temp + thermal_push + rng.normal(0.0, 0.35, n),
        t_min - 1.5,
        t_max + 1.5,
    )

    # ── pH ──────────────────────────────────────────────────────────────────
    # Higher stress correlates with lower pH (local acidification/eutrophication).
    ph_min, ph_max = env["base_ph"]
    ph = np.clip(
        rng.uniform(ph_min, ph_max, n) - 0.22 * stress + rng.normal(0.0, 0.04, n),
        7.60,
        8.50,
    )

    # ── Salinity ────────────────────────────────────────────────────────────
    s_min, s_max = env["base_salinity"]
    salinity_ppt = np.clip(
        rng.uniform(s_min, s_max, n) + rng.normal(0.0, 0.35, n),
        s_min - 1.5,
        s_max + 1.5,
    )

    # ── Dissolved oxygen ────────────────────────────────────────────────────
    # Higher stress and lower clarity both reduce DO.
    do_min, do_max = env["base_do"]
    dissolved_oxygen_mg_l = np.clip(
        rng.uniform(do_min, do_max, n) - 1.6 * stress + 0.5 * clarity + rng.normal(0.0, 0.30, n),
        2.0,
        11.0,
    )

    # ── Turbidity ───────────────────────────────────────────────────────────
    # Generated on a log scale to avoid negative values and match the typical
    # right-skewed distribution of NTU measurements in coastal waters.
    turb_min, turb_max = env["base_turbidity"]
    log_turb_base = rng.uniform(np.log(turb_min + 0.05), np.log(turb_max + 0.05), n)
    turbidity_push = (1.0 - clarity) * 6.0
    turbidity_ntu = np.clip(
        np.exp(log_turb_base) + turbidity_push + rng.normal(0.0, 0.4, n),
        0.05,
        32.0,
    )

    # ── Light intensity ─────────────────────────────────────────────────────
    # PAR reaching the substrate (µmol photons m⁻² s⁻¹).
    # Attenuated by depth via Beer-Lambert and by turbidity via the extinction
    # coefficient k_ext.
    surface_par = rng.uniform(1200.0, 2100.0, n)
    # Empirical diffuse attenuation coefficient (m⁻¹):
    #   k_d ≈ a_w + 0.02 × NTU  (Kirk 1994; Devlin et al. 2008)
    # At 15 NTU: k ≈ 0.34 m⁻¹ → 109 µmol at 8 m (old formula gave 962 µmol — too bright).
    k_ext = 0.04 + 0.02 * turbidity_ntu  # m⁻¹; turbidity-dependent
    light_intensity = np.clip(
        surface_par * np.exp(-k_ext * depth_m) + rng.normal(0.0, 12.0, n),
        5.0,
        2200.0,
    )

    # ── Current speed ───────────────────────────────────────────────────────
    c_min, c_max = env["base_current"]
    current_speed_m_s = np.clip(
        rng.uniform(c_min, c_max, n) + rng.normal(0.0, 0.04, n),
        0.005,
        0.95,
    )

    # ── Hard substrate percentage ───────────────────────────────────────────
    # Driven by the structure latent variable.  Hard substrate (rock, dead
    # coral skeleton, live coral base) supports reef structure.
    hard_substrate_percentage = np.clip(
        structure * 82.0 + rng.normal(0.0, 7.0, n),
        0.0,
        100.0,
    )

    # ── Rugosity index ──────────────────────────────────────────────────────
    # Structural complexity of the reef surface measured from bathymetric
    # surveys.  Flat sand = 1.0; highly complex three-dimensional reef ≈ 4-5.
    # Correlates with structure, NOT directly with bleaching or disease.
    rugosity_index = np.clip(
        1.0 + structure * 3.3 + rng.normal(0.0, 0.28, n),
        1.0,
        4.8,
    )

    # ── Sonar backscatter ───────────────────────────────────────────────────
    # Calibrated target strength (dB).  Hard, complex reef returns higher
    # (less negative) values.  Soft sediment / mud returns lower values.
    # Used as a structural indicator only — not a proxy for bleaching.
    sonar_backscatter = np.clip(
        -35.0 + structure * 30.0 + rng.normal(0.0, 2.0, n),
        -36.0,
        -2.0,
    )

    # ── Acoustic complexity index ───────────────────────────────────────────
    # Normalised ACI (0-1).  High values indicate a diverse, biologically
    # active soundscape typical of healthy reefs.  Correlates with coral cover
    # and fish abundance; NOT a direct measure of bleaching.
    aci_base = 0.10 + rng.normal(0.0, 0.05, n)  # baseline ambient noise

    # ── Coral cover ─────────────────────────────────────────────────────────
    # Live coral coverage (%).  Inversely correlated with stress, positively
    # with structure.  A heavily stressed reef still retains some corals
    # (there is scatter).
    coral_cover_base = (1.0 - 0.72 * stress) * 78.0 * (0.45 + 0.55 * structure)
    coral_cover_percentage = np.clip(
        coral_cover_base + rng.normal(0.0, 7.5, n),
        0.0,
        90.0,
    )

    # ACI depends on coral cover (more coral → more fish → richer soundscape)
    acoustic_complexity_index = np.clip(
        aci_base + (coral_cover_percentage / 90.0) * 0.62 + (1.0 - stress) * 0.18,
        0.05,
        1.0,
    )

    # ── Bleaching percentage ─────────────────────────────────────────────────
    # Fraction of visible coral tissue showing bleaching signs.
    # Driven primarily by thermal excess above the ~28.5°C bleaching threshold
    # documented for Indian Ocean reefs (Hughes et al. 2017).
    thermal_excess = np.maximum(0.0, water_temperature_c - 28.5)  # °C above threshold
    bleaching_base = np.clip(thermal_excess * 20.0 + stress * 32.0, 0.0, 100.0)
    bleaching_percentage = np.clip(
        bleaching_base + rng.normal(0.0, 8.0, n),
        0.0,
        100.0,
    )

    # ── Disease percentage ───────────────────────────────────────────────────
    # Fraction of coral colonies showing disease signs.
    # Driven by pH stress (acidification impairs immune response) and low DO.
    ph_stress_comp = np.maximum(0.0, 8.05 - ph) * 38.0
    do_stress_comp = np.maximum(0.0, 5.5 - dissolved_oxygen_mg_l) * 7.0
    disease_base = stress * 28.0 + ph_stress_comp + do_stress_comp
    disease_percentage = np.clip(
        disease_base + rng.normal(0.0, 5.0, n),
        0.0,
        60.0,
    )

    return {
        # ── Identifiers / spatial ──────────────────────────────────────────
        "timestamp": timestamps,
        "latitude": np.round(latitude, 6),
        "longitude": np.round(longitude, 6),
        "region": np.full(n, region, dtype=object),
        # ── Physical ──────────────────────────────────────────────────────
        "depth_m": np.round(depth_m, 2),
        "water_temperature_c": np.round(water_temperature_c, 2),
        "ph": np.round(ph, 3),
        "salinity_ppt": np.round(salinity_ppt, 2),
        "dissolved_oxygen_mg_l": np.round(dissolved_oxygen_mg_l, 2),
        "turbidity_ntu": np.round(turbidity_ntu, 2),
        "light_intensity": np.round(light_intensity, 1),
        "current_speed_m_s": np.round(current_speed_m_s, 3),
        # ── Acoustic / structural ─────────────────────────────────────────
        "sonar_backscatter": np.round(sonar_backscatter, 2),
        "rugosity_index": np.round(rugosity_index, 3),
        "hard_substrate_percentage": np.round(hard_substrate_percentage, 1),
        "acoustic_complexity_index": np.round(acoustic_complexity_index, 4),
        # ── Biological ────────────────────────────────────────────────────
        "coral_cover_percentage": np.round(coral_cover_percentage, 1),
        "bleaching_percentage": np.round(bleaching_percentage, 1),
        "disease_percentage": np.round(disease_percentage, 1),
        # ── Internal latent variables (stripped before returning DataFrame) ─
        "_stress": stress,
        "_structure": structure,
    }


# ---------------------------------------------------------------------------
# Label scoring
# ---------------------------------------------------------------------------


def _compute_health_score(
    features: dict[str, np.ndarray],
    rng: np.random.Generator,
    noise_scale: float,
) -> np.ndarray:
    """
    Compute a composite reef-health stress score in [0, 1].

    Components and weights
    ----------------------
    thermal_stress   (0.20): temperature above Indian-Ocean bleaching threshold
                             (28.5 °C; Hughes et al. 2017).
    ph_stress        (0.12): ocean acidification / local pH depression below 8.10.
    do_stress        (0.08): hypoxia below 6.0 mg/L dissolved oxygen.
    turbidity_stress (0.10): light-limiting turbidity above 20 NTU.
    bleach_factor    (0.25): directly observed bleaching percentage.
    disease_factor   (0.12): disease prevalence.
    low_cover_factor (0.13): inverse of coral cover (degraded reef = low cover).

    Gaussian noise (std = noise_scale × 0.50) is added *after* the weighted
    sum.  This is the primary mechanism that prevents any single feature from
    being a perfect predictor of the health class.
    """
    temp = features["water_temperature_c"]
    ph = features["ph"]
    do = features["dissolved_oxygen_mg_l"]
    turb = features["turbidity_ntu"]
    bl = features["bleaching_percentage"]
    dis = features["disease_percentage"]
    cov = features["coral_cover_percentage"]

    thermal_stress = np.clip((temp - 28.5) / 4.5, 0.0, 1.0)
    ph_stress = np.clip((8.10 - ph) / 0.55, 0.0, 1.0)
    do_stress = np.clip((6.00 - do) / 3.50, 0.0, 1.0)
    turbidity_stress = np.clip(turb / 20.0, 0.0, 1.0)
    bleach_factor = bl / 100.0
    disease_factor = np.clip(dis / 45.0, 0.0, 1.0)
    low_cover_factor = np.clip(1.0 - cov / 72.0, 0.0, 1.0)

    score = (
        0.20 * thermal_stress
        + 0.12 * ph_stress
        + 0.08 * do_stress
        + 0.10 * turbidity_stress
        + 0.25 * bleach_factor
        + 0.12 * disease_factor
        + 0.13 * low_cover_factor
    )
    score = score + rng.normal(0.0, noise_scale * 0.50, len(temp))
    return np.clip(score, 0.0, 1.0)


def _assign_health_class(scores: np.ndarray) -> np.ndarray:
    """
    Map continuous health scores to ordinal class labels.

    Thresholds (see _HEALTH_THRESHOLDS):
      score < 0.28             → healthy
      0.28 ≤ score < 0.47     → stressed
      0.47 ≤ score < 0.66     → bleached
      score ≥ 0.66             → severely_degraded
    """
    t1, t2, t3 = _HEALTH_THRESHOLDS
    labels: np.ndarray = np.empty(len(scores), dtype=object)
    labels[scores < t1] = "healthy"
    labels[(scores >= t1) & (scores < t2)] = "stressed"
    labels[(scores >= t2) & (scores < t3)] = "bleached"
    labels[scores >= t3] = "severely_degraded"
    return labels


def _compute_restoration_score(
    features: dict[str, np.ndarray],
    rng: np.random.Generator,
    noise_scale: float,
) -> np.ndarray:
    """
    Compute a coral-restoration suitability score in [0, 1].

    Higher values indicate a more suitable site for active coral restoration
    (e.g., coral gardening, substrate transplantation).

    Components and weights
    ----------------------
    temp_stability      (0.15): proximity to optimal thermal range (26–30 °C).
    ph_acceptability    (0.12): pH above 7.75 for calcification.
    light_adequacy      (0.10): PAR reaching substrate (compensation > ~50 µmol).
    substrate_quality   (0.22): hard substrate enables coral attachment.
    current_quality     (0.10): moderate 0.05–0.35 m/s flushes larvae and food.
    turbidity_clarity   (0.12): low turbidity improves light & larval settlement.
    depth_suitability   (0.09): <15 m preferred; deeper sites penalised.
    coral_proximity     (0.10): nearby living coral = local larval supply.

    As with health scores, Gaussian noise is added before thresholding.
    """
    temp = features["water_temperature_c"]
    ph = features["ph"]
    light = features["light_intensity"]
    hard = features["hard_substrate_percentage"]
    current = features["current_speed_m_s"]
    turb = features["turbidity_ntu"]
    depth = features["depth_m"]
    coral = features["coral_cover_percentage"]

    temp_stability = np.clip(1.0 - np.abs(temp - 28.0) / 6.5, 0.0, 1.0)
    ph_acceptability = np.clip((ph - 7.75) / 0.55, 0.0, 1.0)
    light_adequacy = np.clip(light / 500.0, 0.0, 1.0)
    substrate_quality = np.clip(hard / 100.0, 0.0, 1.0)
    # Optimal currents: 0.18 m/s; penalty grows on both sides
    current_quality = np.clip(1.0 - np.abs(current - 0.18) / 0.45, 0.0, 1.0)
    turbidity_clarity = np.clip(1.0 - turb / 18.0, 0.0, 1.0)
    depth_suitability = np.clip(1.0 - np.maximum(0.0, depth - 15.0) / 25.0, 0.0, 1.0)
    coral_proximity = np.clip(coral / 65.0, 0.0, 1.0)

    score = (
        0.15 * temp_stability
        + 0.12 * ph_acceptability
        + 0.10 * light_adequacy
        + 0.22 * substrate_quality
        + 0.10 * current_quality
        + 0.12 * turbidity_clarity
        + 0.09 * depth_suitability
        + 0.10 * coral_proximity
    )
    score = score + rng.normal(0.0, noise_scale * 0.45, len(temp))
    return np.clip(score, 0.0, 1.0)


def _assign_restoration_class(scores: np.ndarray) -> np.ndarray:
    """
    Map continuous restoration scores to ordinal class labels.

    Thresholds (see _RESTORATION_THRESHOLDS):
      score < 0.40             → unsuitable
      0.40 ≤ score < 0.63     → moderately_suitable
      score ≥ 0.63             → suitable
    """
    t1, t2 = _RESTORATION_THRESHOLDS
    labels: np.ndarray = np.empty(len(scores), dtype=object)
    labels[scores < t1] = "unsuitable"
    labels[(scores >= t1) & (scores < t2)] = "moderately_suitable"
    labels[scores >= t2] = "suitable"
    return labels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_observations(
    n_samples: int,
    seed: int,
    cfg: Config,
) -> pd.DataFrame:
    """
    Generate a synthetic marine sensor observation dataset.

    Parameters
    ----------
    n_samples:
        Total number of rows to generate.
    seed:
        Random seed (fixes all stochastic elements for reproducibility).
    cfg:
        Loaded ``Config`` object from ``src.config``.

    Returns
    -------
    pd.DataFrame
        DataFrame with 21 columns and ``n_samples`` rows, shuffled.
        The column ``reef_health`` contains the health class label;
        ``restoration_suitability`` contains the restoration class label.
    """
    rng = np.random.default_rng(seed)
    logger.info("Generating %s synthetic observations (seed=%d)", f"{n_samples:,}", seed)

    # Distribute rows across regions proportionally to reef area
    region_ns: dict[str, int] = {}
    remaining = n_samples
    regions = cfg.regions
    for i, region in enumerate(regions):
        if i == len(regions) - 1:
            region_ns[region] = remaining  # absorb rounding remainder
        else:
            n = max(1, round(n_samples * _REGION_WEIGHTS.get(region, 0.25)))
            region_ns[region] = n
            remaining -= n

    logger.info("Region allocation: %s", region_ns)

    # Generate per-region blocks
    blocks: list[dict[str, np.ndarray]] = []
    for region, n in region_ns.items():
        logger.debug("  %s  →  %d rows", region, n)
        block = _generate_region_block(rng, n, region, cfg.region_bounds, cfg.noise_scale)
        blocks.append(block)

    # Concatenate all blocks
    combined: dict[str, np.ndarray] = {
        key: np.concatenate(
            [b[key] for b in blocks],
            axis=0,
        )
        for key in blocks[0]
    }

    # Compute target labels from the full feature array
    health_scores = _compute_health_score(combined, rng, cfg.noise_scale)
    restoration_scores = _compute_restoration_score(combined, rng, cfg.noise_scale)

    health_labels = _assign_health_class(health_scores)
    restoration_labels = _assign_restoration_class(restoration_scores)

    # Build DataFrame — drop internal latent variables
    public_keys = [k for k in combined if not k.startswith("_")]
    df = pd.DataFrame({k: combined[k] for k in public_keys})
    df[cfg.target_health] = health_labels
    df[cfg.target_restoration] = restoration_labels

    # Shuffle rows so region blocks are interleaved
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    logger.info(
        "Done. Health distribution: %s",
        df[cfg.target_health].value_counts().to_dict(),
    )
    logger.info(
        "     Restoration distribution: %s",
        df[cfg.target_restoration].value_counts().to_dict(),
    )
    return df


def print_summary(df: pd.DataFrame, output_path: Path) -> None:
    """Print a human-readable generation summary to stdout."""
    sep = "-" * 62
    health_col = "reef_health"
    resto_col = "restoration_suitability"

    print(f"\n{sep}")
    print("  CoralSense -- Synthetic Data Generation Summary")
    print(sep)
    print("  WARNING: SYNTHETIC DATA -- NOT FOR REAL CONSERVATION USE")
    print(sep)
    print(f"  Dataset shape : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Output path   : {output_path}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print()
    print("  Region distribution:")
    for region, count in df["region"].value_counts().sort_index().items():
        pct = 100.0 * count / len(df)
        print(f"    {str(region):<38} {count:>5,}  ({pct:.1f}%)")
    print()
    print("  Reef health classes:")
    order_h = ["healthy", "stressed", "bleached", "severely_degraded"]
    for cls in order_h:
        count = int((df[health_col] == cls).sum())
        pct = 100.0 * count / len(df)
        print(f"    {cls:<28} {count:>5,}  ({pct:.1f}%)")
    print()
    print("  Restoration suitability:")
    order_r = ["suitable", "moderately_suitable", "unsuitable"]
    for cls in order_r:
        count = int((df[resto_col] == cls).sum())
        pct = 100.0 * count / len(df)
        print(f"    {cls:<28} {count:>5,}  ({pct:.1f}%)")
    print(sep)
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """
    Command-line entry point.

    Examples
    --------
    # Default (reads n_samples and seed from params.yaml):
    python -m src.data.generate_data

    # Custom parameters:
    python -m src.data.generate_data --samples 5000 --seed 7 --output /tmp/test.csv
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic CoralSense marine sensor observations. "
            "WARNING: output is SYNTHETIC and not suitable for real conservation use."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        metavar="N",
        help="Number of observations (default: n_samples from params.yaml)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="S",
        help="Random seed (default: random_seed from params.yaml)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output CSV path (default: data/raw/observations.csv)",
    )

    args = parser.parse_args()
    setup_logging("coralsense.generate_data")
    reset_config()
    cfg = get_config()

    n_samples = args.samples if args.samples is not None else cfg.n_samples
    seed = args.seed if args.seed is not None else cfg.random_seed
    output_path = args.output if args.output is not None else cfg.raw_data_path

    logger.info("CoralSense data generator v%s", cfg.version)
    logger.info("n_samples=%d  seed=%d  output=%s", n_samples, seed, output_path)

    df = generate_observations(n_samples, seed, cfg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved to %s", output_path)

    print_summary(df, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
