#!/usr/bin/env python
"""
Paired coral-cover change across 26 repeated Maldivian Seaview transects, and
its association with NOAA CRW thermal exposure over the same intervals.

The question, stated once and not widened:

    Across the 26 Maldivian transects that Seaview surveyed twice, is greater
    thermal exposure between the two visits ASSOCIATED with a larger decline in
    the image-derived hard-coral-cover estimate?

This is an **observational association analysis** on n = 26. It is not causal
bleaching attribution, not an Indian validation analysis, not training data for
``reef_health`` or ``restoration_suitability``, and it does not touch the
registered synthetic champion models.

Design decisions that are not negotiable downstream, recorded here so they
travel with the code:

* **The row spine is the transect, not the image or the quadrat.** Seaview's
  Maldives component holds thousands of photo-quadrats, but they are
  pseudo-replicates within a transect. The inferential sample size is 26.
* **The response is the publisher's survey-level ``pr_hard_coral``.** It is NOT
  reconstructed from the quadrat label columns, because
  ``seaviewsurvey_reefcover_indianocean.csv`` is missing the ``MASE_MEA_L``
  (*Lobophyllia*, Hard Coral) column and under-reports hard coral for 1.83 % of
  quadrats. See ``docs/external_data.md`` §8.8.
* **``pr_hard_coral`` is a classifier output.** 98.33 % of Indian Ocean cover
  values come from a VGG-D 16 CNN. Call the response an *image-derived
  hard-coral-cover estimate*, never measured cover and never ground truth.
* **Exposure is defined before it is looked at.** The interval is first survey
  to second survey, per pair, and the two exposure metrics are fixed in advance.
  No post-hoc event window is selected to fit the observed decline.
* **No CRW observation after a pair's second survey enters that pair's
  exposure.** No future thermal information leaks backwards.

Usage
-----
    python scripts/analyze_seaview_maldives_pairs.py
    python scripts/analyze_seaview_maldives_pairs.py --figures
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import netcdf_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SURVEYS_CSV = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "raw"
    / "seaview"
    / "tabular-data"
    / "seaviewsurvey_surveys.csv"
)
CRW_DIR = PROJECT_ROOT / "data" / "external" / "raw" / "noaa_crw_5km_v3_1" / "maldives_seaview"
CRW_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "metadata"
    / "noaa_crw_5km_v3_1_maldives_seaview.manifest.json"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "external"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
PAIRS_CSV = REPORT_DIR / "seaview_maldives_pairs.csv"
PAIRS_SUMMARY_JSON = REPORT_DIR / "seaview_maldives_pairs_summary.json"
ASSOCIATION_JSON = REPORT_DIR / "seaview_maldives_crw_association.json"

ANALYSIS_SCOPE = "SEAVIEW_MALDIVES_REPEAT_TRANSECTS"

#: The response. One column, chosen for the reason in the module docstring.
RESPONSE_COLUMN = "pr_hard_coral"

#: Pre-specified thermal-exposure predictors, in reporting order. Two, and only
#: two: a larger family would invite reporting whichever correlates best.
PRIMARY_PREDICTORS: tuple[str, ...] = ("max_dhw", "max_hotspot")

#: CRW products, keyed as in the acquisition manifest.
CRW_VARIABLES: dict[str, str] = {
    "hotspot": "hotspot",
    "dhw": "degree_heating_week",
}

#: Maximum great-circle distance, in km, that a transect may be matched across
#: to reach a valid CRW ocean cell. One native cell width (0.05 deg is ~5.55 km
#: at this latitude, so ~5 km is a conservative reading of "one cell"). The
#: nearest cell CENTRE to an arbitrary point inside a valid grid is at most half
#: a cell diagonal away, ~3.9 km, so this threshold is only ever exercised when
#: the nominal cell is masked. Raising it would mean importing thermal history
#: from a different body of water and requires justification in the manifest
#: before, not after, the association is computed.
MAX_MATCH_DISTANCE_KM = 5.0

#: Fixed so the bootstrap intervals are reproducible.
RANDOM_SEED = 20260830
N_BOOTSTRAP = 10_000

EARTH_RADIUS_KM = 6371.0088


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------


def build_pairs(surveys_csv: Path = SURVEYS_CSV) -> pd.DataFrame:
    """
    Return one row per Maldivian transect surveyed exactly twice.

    Selection is deliberately strict rather than forgiving. A transect enters
    only if ``ocean == 'IND'``, ``country == 'MDV'`` and it has exactly two
    survey rows; a transect with one, three or more visits is excluded rather
    than reduced to a pair, because choosing *which* two visits to compare would
    be an analyst decision the data does not support.

    Chagos is not filtered out by a special case — it simply has no repeated
    transect, so no Chagos row can survive the "exactly two" rule.
    """
    frame = pd.read_csv(surveys_csv)
    maldives = frame[(frame["ocean"] == "IND") & (frame["country"] == "MDV")].copy()
    maldives["survey_date"] = pd.to_datetime(maldives["surveydate"].astype(str), format="%Y%m%d")

    counts = maldives["transectid"].value_counts()
    repeated = sorted(counts[counts == 2].index)
    ordered = maldives[maldives["transectid"].isin(repeated)].sort_values(
        ["transectid", "survey_date", "surveyid"]
    )

    grouped = ordered.groupby("transectid", sort=True)
    first = grouped.nth(0).set_index("transectid")
    second = grouped.nth(1).set_index("transectid")

    pairs = pd.DataFrame(
        {
            "transect_id": first.index,
            "first_survey_id": first["surveyid"].to_numpy(),
            "second_survey_id": second["surveyid"].to_numpy(),
            "first_survey_date": first["survey_date"].to_numpy(),
            "second_survey_date": second["survey_date"].to_numpy(),
            "first_folder_name": first["folder_name"].to_numpy(),
            "second_folder_name": second["folder_name"].to_numpy(),
            "first_lat_start": first["lat_start"].to_numpy(),
            "first_lng_start": first["lng_start"].to_numpy(),
            "first_lat_end": first["lat_end"].to_numpy(),
            "first_lng_end": first["lng_end"].to_numpy(),
            "second_lat_start": second["lat_start"].to_numpy(),
            "second_lng_start": second["lng_start"].to_numpy(),
            "second_lat_end": second["lat_end"].to_numpy(),
            "second_lng_end": second["lng_end"].to_numpy(),
            "first_pr_hard_coral": first[RESPONSE_COLUMN].to_numpy(),
            "second_pr_hard_coral": second[RESPONSE_COLUMN].to_numpy(),
        }
    )

    pairs["interval_days"] = (
        pairs["second_survey_date"] - pairs["first_survey_date"]
    ).dt.days.astype(int)

    # Representative transect coordinate: the mean of all four endpoints across
    # both visits. A transect is a swum line, not a point, and the two visits do
    # not re-navigate it identically; averaging gives ONE location per transect
    # so a single CRW cell serves both visits, which is what "the same site,
    # twice" means. ``coord_spread_km`` below reports how far the furthest
    # endpoint sits from it, so the approximation is measured, not assumed.
    lat_columns = ["first_lat_start", "first_lat_end", "second_lat_start", "second_lat_end"]
    lon_columns = ["first_lng_start", "first_lng_end", "second_lng_start", "second_lng_end"]
    pairs["survey_latitude"] = pairs[lat_columns].mean(axis=1)
    pairs["survey_longitude"] = pairs[lon_columns].mean(axis=1)
    pairs["coord_spread_km"] = [
        max(
            haversine_km(row[lat], row[lon], row["survey_latitude"], row["survey_longitude"])
            for lat, lon in zip(lat_columns, lon_columns, strict=True)
        )
        for _, row in pairs.iterrows()
    ]

    # ── The response ────────────────────────────────────────────────────────
    pairs["delta_hard_coral"] = pairs["second_pr_hard_coral"] - pairs["first_pr_hard_coral"]
    pairs["absolute_change"] = pairs["delta_hard_coral"].abs()
    # Relative change is descriptive only. It divides by a small first value for
    # the lowest-cover transects, which inflates it; the summary reports the
    # minimum denominator so a reader can see how far to trust it. Nothing is
    # tested on this column.
    pairs["relative_change"] = pairs["delta_hard_coral"] / pairs["first_pr_hard_coral"]

    return pairs.reset_index(drop=True)


def verify_pairs(pairs: pd.DataFrame, surveys_csv: Path = SURVEYS_CSV) -> dict:
    """
    Re-derive every claim the pair table makes, from the source table.

    Nothing here is repaired: a failed check raises, because an analysis built
    on a spine that does not match its source is worse than no analysis.
    """
    frame = pd.read_csv(surveys_csv)
    indian_ocean = frame[frame["ocean"] == "IND"]
    chagos = indian_ocean[indian_ocean["country"] == "CHA"]
    chagos_counts = chagos["transectid"].value_counts()

    checks = {
        "n_pairs": int(len(pairs)),
        "n_pairs_is_26": len(pairs) == 26,
        "each_transect_once": bool(pairs["transect_id"].is_unique),
        "all_survey_ids_distinct": bool(
            pd.concat([pairs["first_survey_id"], pairs["second_survey_id"]]).is_unique
        ),
        "maldives_only": bool(
            frame[frame["surveyid"].isin(pairs["first_survey_id"])]["country"].eq("MDV").all()
            and frame[frame["surveyid"].isin(pairs["second_survey_id"])]["country"].eq("MDV").all()
        ),
        "chagos_repeat_transects": int((chagos_counts >= 2).sum()),
        "chagos_surveys": int(len(chagos)),
        "first_before_second": bool(
            (pairs["first_survey_date"] < pairs["second_survey_date"]).all()
        ),
        "response_column": RESPONSE_COLUMN,
        "response_complete": bool(
            pairs[["first_pr_hard_coral", "second_pr_hard_coral"]].notna().all().all()
        ),
        "interval_days_min": int(pairs["interval_days"].min()),
        "interval_days_max": int(pairs["interval_days"].max()),
        "max_coord_spread_km": float(pairs["coord_spread_km"].max()),
        "earliest_first_survey": pairs["first_survey_date"].min().strftime("%Y-%m-%d"),
        "latest_second_survey": pairs["second_survey_date"].max().strftime("%Y-%m-%d"),
        "excluded_maldives_single_visit_transects": int(
            indian_ocean[indian_ocean["country"] == "MDV"]["transectid"].nunique() - len(pairs)
        ),
        "transects_visited_three_or_more_times": int(
            (indian_ocean["transectid"].value_counts() > 2).sum()
        ),
    }

    failures = [
        name
        for name in (
            "n_pairs_is_26",
            "each_transect_once",
            "all_survey_ids_distinct",
            "maldives_only",
            "first_before_second",
            "response_complete",
        )
        if not checks[name]
    ]
    if failures or checks["chagos_repeat_transects"] != 0:
        raise SystemExit(f"pair verification failed: {failures or 'chagos has repeats'}")
    return checks


# ---------------------------------------------------------------------------
# Paired descriptive analysis (response only, before any exposure is loaded)
# ---------------------------------------------------------------------------


def _distribution(values: pd.Series) -> dict:
    """Summarise one numeric column."""
    array = values.to_numpy(dtype=float)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "sd": float(array.std(ddof=1)),
        "min": float(array.min()),
        "q1": float(np.percentile(array, 25)),
        "q3": float(np.percentile(array, 75)),
        "iqr": float(np.percentile(array, 75) - np.percentile(array, 25)),
        "max": float(array.max()),
    }


def paired_change_summary(pairs: pd.DataFrame) -> dict:
    """
    Characterise the paired change on its own, before thermal exposure exists.

    Doing this first is not sequencing for its own sake: it fixes what the
    biological signal looks like independently of any predictor, so the
    association section cannot quietly redefine the response to suit it.
    """
    delta = pairs["delta_hard_coral"]

    # Normality is CHECKED and REPORTED rather than assumed, and BOTH paired
    # tests are reported. A significant Shapiro-Wilk result is not on its own a
    # licence to switch tests: Wilcoxon signed-rank carries its own assumptions
    # (symmetry of the differences about their median) and is not automatically
    # "the valid one" when normality fails. It is reported as the primary
    # rank-based paired analysis, with the paired t-test kept as a complementary
    # sensitivity analysis so the reader can see both.
    shapiro = stats.shapiro(delta.to_numpy())
    wilcoxon = stats.wilcoxon(
        pairs["second_pr_hard_coral"].to_numpy(),
        pairs["first_pr_hard_coral"].to_numpy(),
    )
    ttest = stats.ttest_rel(
        pairs["second_pr_hard_coral"].to_numpy(),
        pairs["first_pr_hard_coral"].to_numpy(),
    )

    n = len(delta)
    n_positive = int((delta > 0).sum())
    n_negative = int((delta < 0).sum())
    n_zero = int((delta == 0).sum())

    # Matched-pairs rank-biserial correlation: the signed-rank effect size,
    # bounded -1..1, reported because a p-value on n=26 says little about size.
    ranks = stats.rankdata(delta.abs().to_numpy())
    total_rank = ranks.sum()
    rank_biserial = float(
        (ranks[(delta > 0).to_numpy()].sum() - ranks[(delta < 0).to_numpy()].sum()) / total_rank
    )

    rng = np.random.default_rng(RANDOM_SEED)
    draws = rng.integers(0, n, size=(N_BOOTSTRAP, n))
    boot_median = np.median(delta.to_numpy()[draws], axis=1)
    boot_mean = delta.to_numpy()[draws].mean(axis=1)

    return {
        "n_pairs": n,
        "response": {
            "column": RESPONSE_COLUMN,
            "source_table": "seaviewsurvey_surveys.csv",
            "level": "survey (transect visit)",
            "nature": (
                "image-derived hard-coral-cover estimate; 98.33 % of Indian Ocean cover values "
                "are VGG-D 16 CNN output, not diver observation"
            ),
            "units": "proportion of classified points, 0-1",
            "reconstructed_from_quadrats": False,
            "why_not_reconstructed": (
                "seaviewsurvey_reefcover_indianocean.csv omits the MASE_MEA_L (Lobophyllia, Hard "
                "Coral) column and under-reports hard coral for 2 513 of 137 698 quadrats "
                "(1.83 %). See docs/external_data.md section 8.8."
            ),
        },
        "pre_distribution": _distribution(pairs["first_pr_hard_coral"]),
        "post_distribution": _distribution(pairs["second_pr_hard_coral"]),
        "delta_distribution": _distribution(delta),
        "absolute_change_distribution": _distribution(pairs["absolute_change"]),
        "relative_change_distribution": _distribution(pairs["relative_change"]),
        "relative_change_caveat": {
            "min_first_pr_hard_coral": float(pairs["first_pr_hard_coral"].min()),
            "note": (
                "Relative change divides by the first visit's cover. The smallest denominator in "
                "these 26 pairs is reported above; relative values for the lowest-cover "
                "transects are correspondingly unstable. Descriptive only — no test uses this "
                "column."
            ),
        },
        "direction": {
            "declining": n_negative,
            "increasing": n_positive,
            "unchanged": n_zero,
        },
        "outliers_removed": 0,
        "outlier_policy": (
            "None removed. Extreme changes are retained; section 13's leave-one-pair-out check "
            "reports whether any single transect drives the association instead."
        ),
        "interval_days": _distribution(pairs["interval_days"].astype(float)),
        "normality_check": {
            "test": "Shapiro-Wilk on the 26 paired differences",
            "statistic": float(shapiro.statistic),
            "p_value": float(shapiro.pvalue),
            "normal_at_0_05": bool(shapiro.pvalue >= 0.05),
            "interpretation": (
                "The paired-difference distribution was strongly non-normal (Shapiro-Wilk "
                f"p = {shapiro.pvalue:.5f}). Wilcoxon signed-rank is therefore reported as the "
                "primary rank-based paired analysis, while the paired t-test is retained as a "
                "complementary sensitivity analysis. This is a reporting choice, not a claim "
                "that a significant Shapiro-Wilk result makes the signed-rank test valid: "
                "Wilcoxon has its own assumptions, including symmetry of the differences about "
                "their median, which are not established by the normality test."
            ),
        },
        "paired_test": {
            "primary": (
                "Wilcoxon signed-rank (two-sided), second visit vs first visit — the primary "
                "rank-based paired analysis"
            ),
            "secondary": (
                "Paired t-test (two-sided), second visit vs first visit — retained as a "
                "complementary sensitivity analysis, not as a replacement for the rank-based "
                "test and not as the primary result"
            ),
            "wilcoxon_statistic": float(wilcoxon.statistic),
            "wilcoxon_p_value": float(wilcoxon.pvalue),
            "secondary_paired_t_statistic": float(ttest.statistic),
            "secondary_paired_t_p_value": float(ttest.pvalue),
            "effect_size_rank_biserial": rank_biserial,
            "median_change_bootstrap_ci_95": [
                float(np.percentile(boot_median, 2.5)),
                float(np.percentile(boot_median, 97.5)),
            ],
            "mean_change_bootstrap_ci_95": [
                float(np.percentile(boot_mean, 2.5)),
                float(np.percentile(boot_mean, 97.5)),
            ],
            "bootstrap_resamples": N_BOOTSTRAP,
            "random_seed": RANDOM_SEED,
            "interpretation_limit": (
                "A paired difference in an image-derived cover estimate between two visits two "
                "years apart. It does not establish a cause. Storm damage, disease, predation, "
                "transect re-navigation and classifier variation are not excluded by this design."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Spatial matching
# ---------------------------------------------------------------------------


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two WGS 84 points."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)))


def load_crw(key: str) -> dict:
    """
    Load one acquired CRW product into memory.

    Returns the value cube plus its axes and a per-cell validity mask. "Valid"
    means the cell has at least one finite value somewhere in the acquired
    window: a cell that is NaN on every single day is land or permanently
    masked, and matching a transect to it would fabricate exposure.
    """
    path = CRW_DIR / f"noaa_crw_5km_v3_1_{key}_maldives_seaview.nc"
    if not path.is_file():
        raise SystemExit(
            f"Missing {path.relative_to(PROJECT_ROOT)}. "
            f"Run scripts/fetch_noaa_crw_maldives.py first."
        )

    with netcdf_file(str(path), "r", mmap=False) as handle:
        variable = handle.variables[CRW_VARIABLES[key]]
        values = np.asarray(variable.data, dtype="float64")
        lat = np.asarray(handle.variables["latitude"].data, dtype="float64")
        lon = np.asarray(handle.variables["longitude"].data, dtype="float64")
        raw_time = np.asarray(handle.variables["time"].data, dtype="int64")
        units = variable._attributes.get("units", b"")
        units = units.decode() if isinstance(units, bytes) else str(units)
        fill = variable._attributes.get("_FillValue")

    if fill is not None:
        fill = float(np.asarray(fill, dtype="float64"))
        if np.isfinite(fill):
            values = np.where(values == fill, np.nan, values)

    dates = pd.to_datetime(raw_time.astype("datetime64[s]")).normalize()
    return {
        "key": key,
        "path": path,
        "values": values,
        "lat": lat,
        "lon": lon,
        "dates": dates,
        "units": units,
        "cell_valid": np.isfinite(values).any(axis=0),
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    """Return the SHA-256 of *path* as lowercase hex."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def match_cell(latitude: float, longitude: float, valid: np.ndarray, cube: dict) -> dict:
    """
    Map one survey location to a CRW grid cell.

    The rule, in full: take the nearest grid-cell centre by great-circle
    distance among cells that carry at least one finite value in the acquired
    window, and reject the match if that distance exceeds
    :data:`MAX_MATCH_DISTANCE_KM`.

    Two things this deliberately does not do. It does not interpolate, so no
    value is ever synthesised from a neighbourhood that may straddle a coast.
    And it does not silently walk to a farther cell — the distance is recorded
    on every row, and whether the chosen cell was also the nearest cell
    regardless of validity is recorded too, so a coast-driven displacement is
    visible rather than absorbed.
    """
    grid_lat, grid_lon = np.meshgrid(cube["lat"], cube["lon"], indexing="ij")
    distances = np.array(
        [
            haversine_km(latitude, longitude, float(a), float(b))
            for a, b in zip(grid_lat.ravel(), grid_lon.ravel(), strict=True)
        ]
    ).reshape(grid_lat.shape)

    nominal_index = np.unravel_index(int(np.argmin(distances)), distances.shape)

    masked = np.where(valid, distances, np.inf)
    if not np.isfinite(masked).any():
        return {"matched": False, "reason": "no valid CRW ocean cell in the acquired window"}

    index = np.unravel_index(int(np.argmin(masked)), masked.shape)
    distance_km = float(masked[index])
    if distance_km > MAX_MATCH_DISTANCE_KM:
        return {
            "matched": False,
            "reason": (
                f"nearest valid CRW ocean cell is {distance_km:.2f} km away, beyond the "
                f"{MAX_MATCH_DISTANCE_KM} km threshold"
            ),
            "nearest_valid_distance_km": distance_km,
        }

    return {
        "matched": True,
        "lat_index": int(index[0]),
        "lon_index": int(index[1]),
        "crw_latitude": float(cube["lat"][index[0]]),
        "crw_longitude": float(cube["lon"][index[1]]),
        "match_distance_km": distance_km,
        "nominal_cell_was_valid": bool(valid[nominal_index]),
        "displaced_by_masking": bool(index != nominal_index),
    }


# ---------------------------------------------------------------------------
# Temporal exposure
# ---------------------------------------------------------------------------


def exposure_for_pair(cube: dict, lat_index: int, lon_index: int, start, end) -> dict:
    """
    Summarise one product's exposure over one pair's interval.

    The interval is closed on both ends: ``start <= date <= end``, where *start*
    is the first survey date and *end* the second. Nothing dated after the
    second survey can enter — that bound is what keeps future thermal
    information out of a pair's own predictor.
    """
    window = (cube["dates"] >= start) & (cube["dates"] <= end)
    series = cube["values"][window, lat_index, lon_index]
    dates = cube["dates"][window]

    finite = np.isfinite(series)
    n_expected = int((end - start).days) + 1
    result = {
        "n_days_in_interval": n_expected,
        "n_crw_days_present": int(window.sum()),
        "n_valid_observations": int(finite.sum()),
        "missing_day_fraction": float(1.0 - finite.sum() / n_expected) if n_expected else None,
        "exposure_start": start.strftime("%Y-%m-%d"),
        "exposure_end": end.strftime("%Y-%m-%d"),
    }
    if not finite.any():
        result.update({"max": None, "date_of_max": None})
        return result

    valid_values = series[finite]
    valid_dates = dates[finite]
    peak = int(np.argmax(valid_values))
    result.update(
        {
            "max": float(valid_values[peak]),
            "date_of_max": valid_dates[peak].strftime("%Y-%m-%d"),
        }
    )
    return result


def attach_exposure(pairs: pd.DataFrame, cubes: dict[str, dict]) -> tuple[pd.DataFrame, list[dict]]:
    """Attach spatial match and thermal exposure to every pair."""
    # A cell must be usable in BOTH products; matching each product separately
    # could otherwise pin HotSpot and DHW to different cells for one transect.
    valid = cubes["hotspot"]["cell_valid"] & cubes["dhw"]["cell_valid"]

    # Every row carries every exposure column, matched or not. An unmatched pair
    # must still appear in the table as an explicit blank rather than vanish, and
    # building the frame from rows with differing keys would drop columns
    # entirely if no pair happened to match.
    blank_exposure = {"exposure_start": None, "exposure_end": None} | {
        f"{prefix}_{key}{suffix}": None
        for key in ("hotspot", "dhw")
        for prefix, suffix in (
            ("max", ""),
            ("date_of_max", ""),
            ("n_valid", "_days"),
            ("missing", "_day_fraction"),
        )
    }

    rows: list[dict] = []
    unmatched: list[dict] = []
    for _, pair in pairs.iterrows():
        match = match_cell(
            float(pair["survey_latitude"]),
            float(pair["survey_longitude"]),
            valid,
            cubes["hotspot"],
        )
        row: dict = {"transect_id": int(pair["transect_id"]), **blank_exposure}

        if not match["matched"]:
            unmatched.append({"transect_id": int(pair["transect_id"]), **match})
            row.update(
                {
                    "crw_matched": False,
                    "crw_unmatched_reason": match["reason"],
                    "crw_latitude": None,
                    "crw_longitude": None,
                    "match_distance_km": match.get("nearest_valid_distance_km"),
                    "nominal_cell_was_valid": None,
                    "displaced_by_masking": None,
                }
            )
            rows.append(row)
            continue

        row.update(
            {
                "crw_matched": True,
                "crw_unmatched_reason": "",
                "crw_latitude": match["crw_latitude"],
                "crw_longitude": match["crw_longitude"],
                "match_distance_km": match["match_distance_km"],
                "nominal_cell_was_valid": match["nominal_cell_was_valid"],
                "displaced_by_masking": match["displaced_by_masking"],
            }
        )

        for key in ("hotspot", "dhw"):
            summary = exposure_for_pair(
                cubes[key],
                match["lat_index"],
                match["lon_index"],
                pair["first_survey_date"],
                pair["second_survey_date"],
            )
            row[f"max_{key}"] = summary["max"]
            row[f"date_of_max_{key}"] = summary["date_of_max"]
            row[f"n_valid_{key}_days"] = summary["n_valid_observations"]
            row[f"missing_{key}_day_fraction"] = summary["missing_day_fraction"]
            row["exposure_start"] = summary["exposure_start"]
            row["exposure_end"] = summary["exposure_end"]
        rows.append(row)

    return pairs.merge(pd.DataFrame(rows), on="transect_id", how="left"), unmatched


# ---------------------------------------------------------------------------
# Association analysis
# ---------------------------------------------------------------------------


def spearman_with_ci(x: np.ndarray, y: np.ndarray) -> dict:
    """Spearman rho with a bootstrap percentile interval on a fixed seed."""
    result = stats.spearmanr(x, y)
    rho, p_value = float(result.statistic), float(result.pvalue)

    n = x.size
    rng = np.random.default_rng(RANDOM_SEED)
    draws = rng.integers(0, n, size=(N_BOOTSTRAP, n))
    boot = np.array(
        [
            stats.spearmanr(x[idx], y[idx]).statistic
            for idx in draws
            if np.unique(x[idx]).size > 1 and np.unique(y[idx]).size > 1
        ]
    )
    boot = boot[np.isfinite(boot)]

    # Fisher z on the rank correlation, reported next to the bootstrap because
    # the two disagreeing would itself be informative at this sample size.
    fisher = None
    if n > 3 and abs(rho) < 1.0:
        z = np.arctanh(rho)
        se = 1.03 / np.sqrt(n - 3)
        fisher = [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]

    return {
        "n": int(n),
        "rho": rho,
        "p_value": p_value,
        "bootstrap_ci_95": (
            [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
            if boot.size
            else None
        ),
        "bootstrap_resamples_used": int(boot.size),
        "fisher_z_ci_95": fisher,
        "random_seed": RANDOM_SEED,
    }


def leave_one_out(x: np.ndarray, y: np.ndarray, transect_ids: np.ndarray) -> dict:
    """Recompute Spearman rho with each pair dropped in turn."""
    rhos = []
    for i in range(x.size):
        keep = np.ones(x.size, dtype=bool)
        keep[i] = False
        rhos.append(float(stats.spearmanr(x[keep], y[keep]).statistic))
    rhos_array = np.array(rhos)
    full = float(stats.spearmanr(x, y).statistic)
    most_influential = int(np.argmax(np.abs(rhos_array - full)))
    return {
        "rho_full": full,
        "rho_min": float(rhos_array.min()),
        "rho_max": float(rhos_array.max()),
        "rho_range": float(rhos_array.max() - rhos_array.min()),
        "most_influential_transect_id": int(transect_ids[most_influential]),
        "rho_without_most_influential": float(rhos_array[most_influential]),
        "shift_from_most_influential": float(rhos_array[most_influential] - full),
        "sign_stable": bool(np.all(np.sign(rhos_array) == np.sign(full)) and full != 0),
        "dominated_by_one_pair": bool(
            np.any(np.sign(rhos_array) != np.sign(full)) if full != 0 else False
        ),
    }


def exposure_contrast(values: np.ndarray, *, quantum: float) -> dict:
    """
    Describe how much the predictor actually varies across the 26 transects.

    This is reported for both predictors regardless of what the correlation
    turned out to be, and it is the first thing to read before interpreting a
    null. All 26 transects sit inside one small Maldivian box and were exposed
    to the same 2016 thermal event, so the design may simply not contain enough
    exposure contrast to detect an association even if one exists. A rank test
    cannot rank what barely differs, and *quantum* — the resolution the values
    are stored at — bounds how finely it can try.
    """
    spread = float(values.max() - values.min())
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "range": spread,
        "sd": float(values.std(ddof=1)),
        "coefficient_of_variation": (
            float(values.std(ddof=1) / values.mean()) if values.mean() else None
        ),
        "n_distinct_values": int(np.unique(values).size),
        "n_transects": int(values.size),
        "storage_quantum": quantum,
        "range_in_quanta": round(spread / quantum) if quantum else None,
        "n_tied_ranks": int(values.size - np.unique(values).size),
    }


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjustment across the pre-specified primary tests."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p_value))
        adjusted[name] = running
    return adjusted


def associate(matched: pd.DataFrame) -> dict:
    """Run the pre-specified association analysis on the matched pairs."""
    response = matched["delta_hard_coral"].to_numpy(dtype=float)
    predictors = {name: matched[name].to_numpy(dtype=float) for name in PRIMARY_PREDICTORS}

    transect_ids = matched["transect_id"].to_numpy()
    tests = {name: spearman_with_ci(values, response) for name, values in predictors.items()}
    sensitivity = {
        name: leave_one_out(values, response, transect_ids) for name, values in predictors.items()
    }
    adjusted = holm({name: test["p_value"] for name, test in tests.items()})
    contrast = {
        "max_dhw": exposure_contrast(predictors["max_dhw"], quantum=0.01),
        "max_hotspot": exposure_contrast(predictors["max_hotspot"], quantum=0.01),
        "why_this_matters": (
            "All 26 transects lie within one Maldivian box roughly 0.9 deg across and were "
            "exposed to the same 2016 thermal event, so the design carries little exposure "
            "contrast. Read the ranges above before reading any null result below: a weak "
            "correlation across a narrow predictor range is evidence about THIS design, not "
            "evidence that thermal exposure and coral-cover change are unrelated in general."
        ),
    }

    collinear = stats.spearmanr(predictors["max_dhw"], predictors["max_hotspot"])
    return {
        "question": (
            "Is greater thermal exposure between the two visits ASSOCIATED with a larger decline "
            "in the image-derived hard-coral-cover estimate across the 26 repeated Maldivian "
            "transects?"
        ),
        "design": "observational, paired, cross-sectional in the change",
        "response": {
            "name": "delta_hard_coral",
            "definition": "second_pr_hard_coral - first_pr_hard_coral",
            "sign": "negative = decline in image-derived hard-coral cover",
        },
        "primary_predictors": list(PRIMARY_PREDICTORS),
        "primary_test": "Spearman rank correlation, two-sided",
        "n_analysed": int(len(matched)),
        "exposure_contrast": contrast,
        "tests": tests,
        "holm_adjusted_p_values": adjusted,
        "multiple_testing": {
            "n_pre_specified_tests": len(PRIMARY_PREDICTORS),
            "method": "Holm-Bonferroni across the two pre-specified primary tests only",
            "note": (
                "Raw p-values are reported above and are the values of record. The adjustment is "
                "offered for completeness across the two pre-specified tests; no additional test "
                "was run and discarded."
            ),
        },
        "predictor_collinearity": {
            "spearman_rho_dhw_vs_hotspot": float(collinear.statistic),
            "p_value": float(collinear.pvalue),
            "note": (
                "DHW is by construction the running 12-week accumulation of HotSpot values at or "
                "above 1 degC, so the two are not independent tests of separate hypotheses. A "
                "high correlation here means the two results are close to one result reported "
                "twice, and the Holm adjustment above is correspondingly conservative."
            ),
        },
        "sensitivity_leave_one_pair_out": sensitivity,
        "interpretation": {
            "any_test_significant_at_0_05": any(t["p_value"] < 0.05 for t in tests.values()),
            "per_predictor": {
                name: {
                    "rho_sign": "negative" if test["rho"] < 0 else "positive",
                    "direction_relative_to_hypothesis": (
                        "same direction as the hypothesis (more exposure, more decline)"
                        if test["rho"] < 0
                        else "opposite direction to the hypothesis"
                    ),
                    "ci_excludes_zero": (
                        bool(
                            test["bootstrap_ci_95"]
                            and (test["bootstrap_ci_95"][0] > 0 or test["bootstrap_ci_95"][1] < 0)
                        )
                    ),
                }
                for name, test in tests.items()
            },
            "reading": (
                "Read the effect sizes and intervals, not the p-values. A confidence interval "
                "spanning zero on n = 26 across a narrow exposure range means the data do not "
                "resolve the association in either direction. That is 'not detected here', which "
                "is not the same claim as 'not present'."
            ),
            "spatial_dependence": (
                "The 26 transects are spatially clustered and share the same regional 2016 "
                "heat-stress event. Consequently, the correlation p-values and bootstrap "
                "intervals above are interpreted descriptively and should not be treated as "
                "inference from 26 independent climatic replicates."
            ),
            "what_a_null_here_does_not_mean": (
                "It does not mean thermal stress is unrelated to coral-cover change. It means "
                "this design — 26 transects in one small area, all exposed to the same 2016 "
                "event, with an image-derived response — does not resolve a difference between "
                "them. See exposure_contrast above."
            ),
        },
        "not_performed": {
            "multivariable_model": "n = 26 does not support one",
            "train_test_split": "not an ML benchmark",
            "cross_validation": "not an ML benchmark",
            "causal_attribution": "the design does not support it",
            "outlier_removal": "none; influence is reported instead",
            "spatial_dependence_adjustment": (
                "none; the spatial clustering of the 26 transects is disclosed and the "
                "correlation inference read descriptively, not corrected for"
            ),
        },
        "scatter_data": [
            {
                "transect_id": int(row["transect_id"]),
                "delta_hard_coral": float(row["delta_hard_coral"]),
                "max_dhw": float(row["max_dhw"]),
                "max_hotspot": float(row["max_hotspot"]),
            }
            for _, row in matched.iterrows()
        ],
    }


# ---------------------------------------------------------------------------
# Figures (optional — matplotlib is not a declared project dependency)
# ---------------------------------------------------------------------------


def write_figures(joined: pd.DataFrame, matched: pd.DataFrame) -> list[Path]:
    """
    Render the three diagnostic plots.

    Off by default and imported lazily: matplotlib is not in ``requirements``,
    and the JSON artifacts already carry the full scatter data, so the analysis
    must not depend on it being installed.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    figure, axes = plt.subplots(figsize=(7, 5))
    for _, row in joined.iterrows():
        axes.plot(
            [0, 1],
            [row["first_pr_hard_coral"], row["second_pr_hard_coral"]],
            marker="o",
            color="#2a6f8e" if row["delta_hard_coral"] < 0 else "#b7791f",
            alpha=0.75,
            linewidth=1.2,
        )
    axes.set_xticks([0, 1])
    axes.set_xticklabels(["first visit\n(2015)", "second visit\n(2017)"])
    axes.set_ylabel("pr_hard_coral (image-derived estimate)")
    axes.set_title("Paired change, 26 repeated Maldivian transects")
    figure.tight_layout()
    path = FIGURE_DIR / "seaview_maldives_paired_change.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written.append(path)

    for predictor, filename in (
        ("max_dhw", "seaview_maldives_dhw_association.png"),
        ("max_hotspot", "seaview_maldives_hotspot_association.png"),
    ):
        figure, axes = plt.subplots(figsize=(7, 5))
        axes.axhline(0.0, color="#999999", linewidth=0.8)
        axes.scatter(matched[predictor], matched["delta_hard_coral"], color="#2a6f8e", s=42)
        axes.set_xlabel(predictor)
        axes.set_ylabel("delta_hard_coral (negative = decline)")
        result = stats.spearmanr(matched[predictor], matched["delta_hard_coral"])
        axes.set_title(
            f"{predictor} vs cover change — Spearman rho = {result.statistic:.3f}, "
            f"p = {result.pvalue:.3f}, n = {len(matched)}"
        )
        figure.tight_layout()
        path = FIGURE_DIR / filename
        figure.savefig(path, dpi=150)
        plt.close(figure)
        written.append(path)

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

PAIR_CSV_COLUMNS = [
    "transect_id",
    "first_survey_id",
    "second_survey_id",
    "first_survey_date",
    "second_survey_date",
    "interval_days",
    "survey_latitude",
    "survey_longitude",
    "coord_spread_km",
    "first_lat_start",
    "first_lng_start",
    "first_lat_end",
    "first_lng_end",
    "second_lat_start",
    "second_lng_start",
    "second_lat_end",
    "second_lng_end",
    "first_pr_hard_coral",
    "second_pr_hard_coral",
    "delta_hard_coral",
    "absolute_change",
    "relative_change",
    "crw_matched",
    "crw_unmatched_reason",
    "crw_latitude",
    "crw_longitude",
    "match_distance_km",
    "nominal_cell_was_valid",
    "displaced_by_masking",
    "exposure_start",
    "exposure_end",
    "max_dhw",
    "date_of_max_dhw",
    "n_valid_dhw_days",
    "missing_dhw_day_fraction",
    "max_hotspot",
    "date_of_max_hotspot",
    "n_valid_hotspot_days",
    "missing_hotspot_day_fraction",
]


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1] if __doc__ else "")
    parser.add_argument(
        "--figures",
        action="store_true",
        help="Also render diagnostic PNGs (requires matplotlib, not a project dependency).",
    )
    args = parser.parse_args()

    if not SURVEYS_CSV.is_file():
        print(f"Missing {SURVEYS_CSV.relative_to(PROJECT_ROOT)}; run scripts/fetch_seaview.py")
        return 1

    generated_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    pairs = build_pairs()
    checks = verify_pairs(pairs)
    print(
        f"Pairs: {checks['n_pairs']} Maldivian transects, Chagos repeats "
        f"{checks['chagos_repeat_transects']}, "
        f"{checks['earliest_first_survey']} .. {checks['latest_second_survey']}"
    )

    change = paired_change_summary(pairs)
    print(
        f"Change: mean {change['delta_distribution']['mean']:+.4f}, "
        f"median {change['delta_distribution']['median']:+.4f}, "
        f"{change['direction']['declining']} declining / "
        f"{change['direction']['increasing']} increasing; "
        f"Wilcoxon p = {change['paired_test']['wilcoxon_p_value']:.5f}"
    )

    cubes = {key: load_crw(key) for key in ("hotspot", "dhw")}
    joined, unmatched = attach_exposure(pairs, cubes)
    matched = joined[joined["crw_matched"]].copy()
    print(
        f"CRW: {len(matched)}/{len(joined)} matched, "
        f"match distance {matched['match_distance_km'].min():.2f}"
        f"..{matched['match_distance_km'].max():.2f} km"
    )

    association = associate(matched) if len(matched) >= 3 else None
    if association:
        for name in PRIMARY_PREDICTORS:
            test = association["tests"][name]
            print(
                f"  {name} vs delta_hard_coral: rho = {test['rho']:+.3f}, "
                f"p = {test['p_value']:.4f}, n = {test['n']}"
            )

    joined[PAIR_CSV_COLUMNS].to_csv(PAIRS_CSV, index=False)

    crw_manifest_sha = _sha256(CRW_MANIFEST) if CRW_MANIFEST.is_file() else None
    provenance = {
        "analysis_scope": ANALYSIS_SCOPE,
        "geography": "MALDIVES_NOT_INDIA",
        "generated_at_utc": generated_at,
        "generated_by": "scripts/analyze_seaview_maldives_pairs.py",
        "biological_source": {
            "dataset_id": "seaview_survey",
            "table": "seaviewsurvey_surveys.csv",
            "manifest": "data/external/metadata/seaview_survey.manifest.json",
            "sha256": _sha256(SURVEYS_CSV),
        },
        "thermal_source": {
            "dataset_id": "noaa_crw_5km_v3_1_maldives_seaview",
            "manifest": CRW_MANIFEST.relative_to(PROJECT_ROOT).as_posix(),
            "manifest_sha256": crw_manifest_sha,
            "files": {
                key: {
                    "path": cube["path"].relative_to(PROJECT_ROOT).as_posix(),
                    "sha256": cube["sha256"],
                    "units": cube["units"],
                    "n_days": int(cube["dates"].size),
                    "date_range": [
                        cube["dates"].min().strftime("%Y-%m-%d"),
                        cube["dates"].max().strftime("%Y-%m-%d"),
                    ],
                }
                for key, cube in cubes.items()
            },
            "note": (
                "Separate from the 2018-2024 India-region CRW acquisition "
                "(data/external/metadata/noaa_crw_5km_v3_1.manifest.json), which this analysis "
                "does not read and does not modify."
            ),
        },
    }

    spatial = {
        "rule": (
            "Nearest valid CRW ocean grid-cell centre by great-circle distance, where a cell is "
            "valid if it carries at least one finite value in BOTH acquired products across the "
            "window. No interpolation. No land crossing."
        ),
        "max_match_distance_km": MAX_MATCH_DISTANCE_KM,
        "max_match_distance_rationale": (
            "One native cell width. 0.05 deg is ~5.55 km at this latitude, and the nearest cell "
            "centre to an arbitrary point inside a valid grid is at most half a cell diagonal "
            "(~3.9 km) away, so the threshold binds only where masking displaces the match."
        ),
        "survey_coordinate_rule": (
            "Mean of all four transect endpoints across both visits, giving one location per "
            "transect so both visits share a CRW cell. coord_spread_km on each row is the "
            "distance from that mean to its furthest endpoint."
        ),
        "grid_resolution_deg": 0.05,
        "n_matched": int(len(matched)),
        "n_unmatched": int(len(joined) - len(matched)),
        "unmatched": unmatched,
        "n_displaced_by_masking": int(joined["displaced_by_masking"].fillna(False).sum()),
        "match_distance_km": (
            _distribution(matched["match_distance_km"]) if len(matched) else None
        ),
        "coord_spread_km": _distribution(joined["coord_spread_km"]),
    }

    temporal = {
        "rule": (
            "Closed interval from each pair's own first survey date to its own second survey "
            "date, inclusive, on daily CRW records."
        ),
        "defined_independently_of_response": True,
        "no_post_hoc_event_window": (
            "The interval is the observed survey spacing. No window was chosen to bracket the "
            "observed decline, and no alternative window was tried."
        ),
        "future_information_excluded": (
            "No CRW observation dated after a pair's second survey enters that pair's exposure."
        ),
        "backward_reach_caveat": (
            "DHW is a 12-week backward accumulation, so a DHW value dated shortly after the "
            "first survey partly accumulates heat from before it. Over intervals of 703-722 days "
            "this affects only the first ~84 days of each series and cannot import heat from "
            "after the second survey. Recorded, not corrected."
        ),
        "date_precision": (
            "Survey dates are day-level integers (YYYYMMDD) in the publisher's table, and CRW is "
            "daily, so both sides align at day granularity with no interpolation. No survey time "
            "of day is published; a survey is treated as occupying its whole date, which is the "
            "least-assumptive reading."
        ),
        "metrics": {
            "max_dhw": "maximum Degree Heating Week over the interval, degree_Celsius_weeks",
            "max_hotspot": "maximum Coral Bleaching HotSpot over the interval, degree_C",
            "secondary_metrics": (
                "None. Two pre-specified metrics only; a larger family would permit selecting "
                "whichever correlates best."
            ),
        },
        "exposure_distributions": {
            "max_dhw": _distribution(matched["max_dhw"]) if len(matched) else None,
            "max_hotspot": _distribution(matched["max_hotspot"]) if len(matched) else None,
        },
        "missing_day_fraction": {
            "max_dhw": float(matched["missing_dhw_day_fraction"].max()) if len(matched) else None,
            "max_hotspot": (
                float(matched["missing_hotspot_day_fraction"].max()) if len(matched) else None
            ),
        },
        "date_of_max_dhw_range": (
            [str(matched["date_of_max_dhw"].min()), str(matched["date_of_max_dhw"].max())]
            if len(matched)
            else None
        ),
        "date_of_max_hotspot_range": (
            [str(matched["date_of_max_hotspot"].min()), str(matched["date_of_max_hotspot"].max())]
            if len(matched)
            else None
        ),
    }

    limitations = {
        "causal": (
            "NON-CAUSAL. This is an association between a thermal-exposure predictor and a change "
            "in an image-derived cover estimate. It does not establish that thermal stress caused "
            "the change, and the words caused, bleaching caused the decline, thermal stress "
            "caused mortality and proves do not apply to it. There is no control, no bleaching "
            "observation, and no exclusion of storm damage, disease, predation, transect "
            "re-navigation or classifier variation."
        ),
        "geography": (
            "MALDIVES_NOT_INDIA. All 26 transects are Maldivian. No Indian reef is involved, no "
            "Indian model is validated, and nothing here is Indian data. The nearest Indian reef "
            "system, Lakshadweep, is roughly 400 km from the nearest transect."
        ),
        "response_is_estimated": (
            "pr_hard_coral is a classifier output, not measured cover and not ground truth. The "
            "published 97 % classifier validation is real and a well-validated estimate is still "
            "an estimate."
        ),
        "sample_size": (
            "n = 26 transects. Photo-quadrats within a transect are pseudo-replicates and are not "
            "used as independent observations. Confidence intervals at this size are wide and the "
            "effect sizes, not the p-values, carry the result."
        ),
        "restricted_exposure_range": (
            "All 26 transects sit inside one Maldivian box roughly 0.9 deg across and share the "
            "same 2016 thermal event, so between-transect exposure barely varies. The design "
            "therefore has little power to detect an exposure-response gradient, and a weak "
            "correlation is as consistent with 'no contrast to measure' as with 'no "
            "relationship'. See association.exposure_contrast for the actual ranges."
        ),
        "spatial_dependence": (
            "The 26 transects are spatially clustered and share the same regional 2016 "
            "heat-stress event. Consequently, the correlation p-values and bootstrap intervals "
            "are interpreted descriptively and should not be treated as inference from 26 "
            "independent climatic replicates. No spatial model, clustered resampling or "
            "permutation scheme was fitted to correct for this; the dependence is disclosed "
            "rather than adjusted for."
        ),
        "no_labels_created": (
            "No reef_health class, no restoration_suitability class, no bleached/not-bleached "
            "class, no healthy/unhealthy class. CRW values were not thresholded into any "
            "reef-condition label."
        ),
        "registered_models_untouched": (
            "The registered synthetic champions, artifacts/mlruns.db and "
            "data/raw/observations.csv are not read, modified or retrained by this analysis."
        ),
    }

    PAIRS_SUMMARY_JSON.write_text(
        json.dumps(
            {
                "schema": "coralsense.analysis.seaview_maldives_pairs/v1",
                "provenance": provenance,
                "pair_selection": checks,
                "paired_change": change,
                "pair_table": PAIRS_CSV.relative_to(PROJECT_ROOT).as_posix(),
                "limitations": limitations,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    ASSOCIATION_JSON.write_text(
        json.dumps(
            {
                "schema": "coralsense.analysis.seaview_maldives_crw_association/v1",
                "provenance": provenance,
                "spatial_matching": spatial,
                "temporal_exposure": temporal,
                "association": association,
                "limitations": limitations,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    written = [PAIRS_CSV, PAIRS_SUMMARY_JSON, ASSOCIATION_JSON]
    if args.figures:
        written.extend(write_figures(joined, matched))

    print()
    for path in written:
        print(f"Wrote {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
