#!/usr/bin/env python
"""
Acquire the NOAA Coral Reef Watch 5 km v3.1 HISTORICAL extension for the
Seaview Maldives repeat-transect analysis.

This is a **second, separately scoped acquisition** of the same NOAA product
already held in this repository.  It does not replace, extend, overwrite or
reinterpret the existing one.  Keeping them apart matters:

    existing acquisition   data/external/metadata/noaa_crw_5km_v3_1.manifest.json
        scope   four INDIAN reef systems (Lakshadweep, Gulf of Mannar,
                Gulf of Kutch, Andaman and Nicobar Islands)
        window  2018-01-01 .. 2024-12-31
        purpose environmental context for Indian reefs
        files   data/external/raw/noaa_crw_5km_v3_1/*.nc

    this acquisition       data/external/metadata/
                           noaa_crw_5km_v3_1_maldives_seaview.manifest.json
        scope   one MALDIVES window covering the 26 repeated Seaview transects
        window  derived from the survey dates themselves (2015-03-29 ..
                2017-04-01), never hard-coded to a remembered range
        purpose thermal-exposure support for a paired observational analysis
        files   data/external/raw/noaa_crw_5km_v3_1/maldives_seaview/*.nc

**The Maldives is not India.**  Nothing acquired here describes an Indian reef,
and nothing here validates any model for Indian reefs.  The existing India
manifest is not modified by this script.

Only two products are requested — Coral Bleaching HotSpot and Degree Heating
Week — because those are the two pre-specified thermal-exposure predictors of
the association analysis.  SST and SST anomaly are not needed for it and are
not downloaded, so the acquisition stays minimal.

Everything about the product itself — the ERDDAP endpoints, the variable
semantics, the licence determination and, importantly, the CoralTemp
source-lineage policy floor — is imported from ``scripts/fetch_noaa_crw.py``
rather than restated.  One definition, one guard, no chance of the two drifting
apart.  In particular this script inherits
``FIRST_POST_OSTIA_BLEND_REQUEST_DATE`` and must never describe any CRW product
as "OSTIA-free".

Usage
-----
    python scripts/fetch_noaa_crw_maldives.py --dry-run
    python scripts/fetch_noaa_crw_maldives.py
    python scripts/fetch_noaa_crw_maldives.py --validate-only
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.parse
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fetch_noaa_crw as crw  # noqa: E402

from src.external.provenance import (  # noqa: E402
    SourceRecord,
    SubsetRecord,
    validate_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Distinct dataset id.  The scientific product is identical to
#: ``noaa_crw_5km_v3_1``; the id differs because the *acquisition* differs in
#: scope, window and purpose, and conflating the two manifests would let
#: Maldives files be read as Indian ones.
DATASET_ID = "noaa_crw_5km_v3_1_maldives_seaview"

#: Purpose tag carried through the manifest.  Read this before joining anything.
ANALYSIS_SCOPE = "SEAVIEW_MALDIVES_REPEAT_TRANSECTS"

RAW_DIR = PROJECT_ROOT / "data" / "external" / "raw" / "noaa_crw_5km_v3_1" / "maldives_seaview"
METADATA_DIR = PROJECT_ROOT / "data" / "external" / "metadata"
MANIFEST_PATH = METADATA_DIR / f"{DATASET_ID}.manifest.json"

SURVEYS_CSV = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "raw"
    / "seaview"
    / "tabular-data"
    / "seaviewsurvey_surveys.csv"
)

#: The two pre-specified thermal-exposure predictors, in the order they are
#: reported.  Taken from the existing product definitions so the variable names,
#: units and "what this is not" text cannot diverge between the two manifests.
PRODUCT_KEYS: tuple[str, ...] = ("hotspot", "dhw")

#: Products present in the India acquisition and deliberately skipped here.
OMITTED_PRODUCTS: dict[str, str] = {
    "sst": (
        "Not required. The pre-registered exposure metrics are maximum DHW and maximum HotSpot "
        "over each pair interval; absolute SST adds no term to either. Acquiring it would "
        "enlarge the download without entering the analysis. It remains available for the "
        "India window, and can be requested for this window later under its own record."
    ),
    "sst_anomaly": (
        "Not required, and partly redundant: HotSpot is already SST minus the site's Maximum "
        "Monthly Mean, which is the anomaly formulation relevant to coral heat stress. Adding a "
        "second, differently-baselined anomaly would invite selecting whichever of the two "
        "correlates better with the response — exactly the multiple-comparison problem the "
        "analysis plan forbids."
    ),
    "bleaching_alert_area_7d_max": (
        "Omitted for the same reasons as in the India acquisition (redundant with HotSpot/DHW, "
        "a 7-day maximum composite that does not align with daily variables, and a categorical "
        "scale NOAA revised on 2023-12-15). See the India manifest's omitted_products block."
    ),
}

# ---------------------------------------------------------------------------
# Spatial extent — derived from the surveys, not chosen
# ---------------------------------------------------------------------------

#: Buffer added around the transect bounding box, in degrees.  Two native CRW
#: cells (2 x 0.05 deg, roughly 11 km at this latitude).  Its job is narrow: the
#: spatial match in the analysis may walk from a transect to the nearest *valid
#: ocean* cell when the nominal cell is land- or coast-masked, and that search
#: must never run off the edge of the subset.  One cell of margin would already
#: cover the documented 5 km search radius; two gives an unambiguous margin
#: while keeping the download at a few megabytes.
BBOX_BUFFER_DEG = 0.10

#: Days of daily CRW data added on each side of the survey window.  Zero, on
#: purpose.  Section 11 of the analysis plan forbids using any CRW observation
#: dated after a pair's second survey, so acquiring past the last second survey
#: would only create an opportunity to leak future thermal information.  A DHW
#: value is itself a backward-looking 12-week accumulation, so no forward pad is
#: needed to make the metric well defined.
WINDOW_PAD_DAYS = 0


def _snap_out(value: float, *, up: bool) -> float:
    """Round *value* outward (up or down) onto the 0.05 degree CRW grid."""
    step = crw.GRID_STEP_DEG
    steps = math.ceil(value / step) if up else math.floor(value / step)
    return round(steps * step, 10)


def repeat_pairs(surveys_csv: Path = SURVEYS_CSV):
    """
    Return the Seaview Maldives repeat-transect pairs as a DataFrame.

    One row per transect surveyed exactly twice, with the first and second
    survey side by side.  This is the *same* selection the analysis script
    makes; it lives here too because the acquisition window is derived from it
    rather than remembered.

    Imported lazily so that importing this module never touches pandas or the
    (git-ignored) Seaview tables.
    """
    import pandas as pd

    frame = pd.read_csv(surveys_csv)
    maldives = frame[(frame["ocean"] == "IND") & (frame["country"] == "MDV")].copy()
    maldives["survey_date"] = pd.to_datetime(maldives["surveydate"].astype(str), format="%Y%m%d")

    counts = maldives["transectid"].value_counts()
    repeated = counts[counts == 2].index
    paired = maldives[maldives["transectid"].isin(repeated)].sort_values(
        ["transectid", "survey_date"]
    )

    grouped = paired.groupby("transectid")
    first = grouped.nth(0).set_index("transectid")
    second = grouped.nth(1).set_index("transectid")
    return first.join(second, rsuffix="_second", how="inner")


def derive_extent(surveys_csv: Path = SURVEYS_CSV) -> dict:
    """
    Derive the acquisition window from the repeat surveys themselves.

    Returns the bounding box (buffered and snapped to the CRW grid), the exact
    date range, and the un-buffered transect extent that produced them, so the
    manifest can show the derivation rather than assert the result.
    """
    pairs = repeat_pairs(surveys_csv)

    # Both ends of both visits: the box must contain the whole swum transect on
    # each occasion, not just the point the analysis later uses as its centre.
    lat_columns = ["lat_start", "lat_end", "lat_start_second", "lat_end_second"]
    lon_columns = ["lng_start", "lng_end", "lng_start_second", "lng_end_second"]
    transect_lat_min = float(pairs[lat_columns].to_numpy().min())
    transect_lat_max = float(pairs[lat_columns].to_numpy().max())
    transect_lon_min = float(pairs[lon_columns].to_numpy().min())
    transect_lon_max = float(pairs[lon_columns].to_numpy().max())

    start = pairs["survey_date"].min()
    end = pairs["survey_date_second"].max()

    return {
        "n_pairs": int(len(pairs)),
        "transect_bbox": (
            transect_lat_min,
            transect_lat_max,
            transect_lon_min,
            transect_lon_max,
        ),
        "bbox": (
            _snap_out(transect_lat_min - BBOX_BUFFER_DEG, up=False),
            _snap_out(transect_lat_max + BBOX_BUFFER_DEG, up=True),
            _snap_out(transect_lon_min - BBOX_BUFFER_DEG, up=False),
            _snap_out(transect_lon_max + BBOX_BUFFER_DEG, up=True),
        ),
        "buffer_deg": BBOX_BUFFER_DEG,
        "earliest_first_survey": start.strftime("%Y-%m-%d"),
        "latest_second_survey": end.strftime("%Y-%m-%d"),
        "start_date": (start.normalize()).strftime("%Y-%m-%d"),
        "end_date": (end.normalize()).strftime("%Y-%m-%d"),
        "pad_days": WINDOW_PAD_DAYS,
    }


BBOX_RATIONALE = (
    "Derived from the 26 repeated Maldivian Seaview transects, not chosen. The minimal box "
    "containing every start and end coordinate of both visits was buffered by "
    f"{BBOX_BUFFER_DEG} deg (two native CRW cells) and snapped outward to the 0.05 deg grid. The "
    "buffer exists so the nearest-valid-ocean-cell search used for spatial matching cannot run "
    "off the edge of the subset; it is not a reef mask and not an environmental context window. "
    "It deliberately does NOT reuse the four Indian acquisition windows, which describe "
    "different reef systems roughly 400-1900 km away."
)

WINDOW_RATIONALE = (
    "Derived from the survey dates themselves: earliest first survey through latest second "
    "survey across the 26 pairs. No padding is added after the last second survey, because the "
    "analysis may not use any CRW observation dated after a pair's own second survey. The window "
    "is fixed before any exposure metric is computed and is independent of the response, so it "
    "cannot be a post-hoc event window selected to fit the observed cover change."
)


def build_request_url(product: crw.Product, extent: dict, *, base: str, dataset: str) -> str:
    """Return the ERDDAP griddap URL for one product over the derived extent."""
    # Inherit the project's conservative CoralTemp lineage floor rather than
    # re-implementing it.  2015 is well past 2002-12-01, but the guard is
    # called so that any future widening of the window trips the same check.
    crw._guard_licence_window(extent["start_date"])

    lat_min, lat_max, lon_min, lon_max = extent["bbox"]
    selector = (
        f"[({extent['start_date']}T12:00:00Z):1:({extent['end_date']}T12:00:00Z)]"
        f"[({lat_min}):1:({lat_max})][({lon_min}):1:({lon_max})]"
    )
    query = urllib.parse.quote(f"{product.variable}{selector}", safe="[]():.,-")
    return f"{base}/{dataset}.nc?{query}"


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def build_source_record(retrieved_at: str, extent: dict) -> SourceRecord:
    """Return the product-level provenance record for this historical extension."""
    india = crw.build_source_record(retrieved_at)
    lat_min, lat_max, lon_min, lon_max = extent["bbox"]

    return SourceRecord(
        dataset_id=DATASET_ID,
        source_name=india.source_name,
        product_name=india.product_name,
        source_url=india.source_url,
        doi=india.doi,
        product_identifier=india.product_identifier,
        version=india.version,
        # Licence determination is a property of the product, not of the
        # window, so it is carried across verbatim rather than re-derived.
        licence_name=india.licence_name,
        licence_url=india.licence_url,
        licence_verified=india.licence_verified,
        licence_verified_via=(
            india.licence_verified_via
            + " HISTORICAL EXTENSION, verified 2026-08-30: the requested window "
            f"({extent['start_date']} to {extent['end_date']}) lies inside the NOAA reprocessed "
            "Geo-Polar Blended era and crosses the documented October 1-29, 2016 merge between "
            "the reprocessed and near-real-time analyses, which is recorded here and is NOT "
            "treated as a licence boundary. It starts well after the project's conservative "
            f"{crw.FIRST_POST_OSTIA_BLEND_REQUEST_DATE} policy floor. Availability was verified "
            "before any download against the NOAA PIFSC OceanWatch ERDDAP dataset metadata: "
            "CRW_hs_v1_0 declares time_coverage_start 1985-01-01T12:00:00Z and CRW_dhw_v1_0 "
            "declares 1985-03-25T12:00:00Z, both covering the requested range. As with the "
            "India window, no part of this acquisition may be described as OSTIA-free: NOAA "
            "states the Geo-Polar Blended SST product switched to OSTIA for bias correction in "
            "2016, which overlaps this window."
        ),
        redistribution_allowed=india.redistribution_allowed,
        raw_tracked_in_git=False,
        citation=(
            "Liu, Gang; Heron, Scott F.; Eakin, C. Mark; De La Cour, Jacqueline L.; Geiger, "
            "Erick F.; Tirak, Kyle V.; Skirving, William J.; Strong, Alan E. (2018). NOAA Coral "
            "Reef Watch (CRW) Daily Global 5-km (0.05 degree) Satellite Coral Bleaching Heat "
            "Stress Monitoring Product Suite. [Subset used: Coral Bleaching HotSpot and Degree "
            f"Heating Week, {extent['start_date']} to {extent['end_date']}, for a single "
            "Maldives window covering the 26 repeated Seaview transects]. NOAA National Centers "
            f"for Environmental Information. Dataset. https://doi.org/{crw.CRW_DOI}. Accessed "
            "2026-08-30."
        ),
        observation_type=india.observation_type,
        processing_level=india.processing_level,
        sensor_type=india.sensor_type,
        original_format=india.original_format,
        original_crs=india.original_crs,
        spatial_resolution=india.spatial_resolution,
        temporal_resolution=india.temporal_resolution,
        geographic_scope=(
            f"Source product is global. This acquisition is ONE window in the MALDIVES: "
            f"lat {lat_min} to {lat_max}, lon {lon_min} to {lon_max}. It contains no Indian "
            f"reef system and is roughly 400 km from Lakshadweep, the nearest. It must not be "
            f"described as Indian data, and it does not overlap the four Indian windows of the "
            f"noaa_crw_5km_v3_1 acquisition."
        ),
        temporal_scope=(
            f"Source product spans 1985-01-01 to present. Acquired window: "
            f"{extent['start_date']} to {extent['end_date']}, derived from the Seaview repeat "
            f"surveys. Disjoint from, and earlier than, the 2018-01-01 to 2024-12-31 window of "
            f"the India acquisition; neither replaces the other."
        ),
        retrieved_at_utc=retrieved_at,
        access_method=india.access_method,
        variable_units={
            "coral_bleaching_hotspot_c": "degree_C",
            "degree_heating_week": "degree_Celsius_weeks",
        },
        provides_variables=(
            "coral_bleaching_hotspot_c (instantaneous thermal stress)",
            "degree_heating_week (accumulated thermal stress over 12 weeks)",
        ),
        cannot_provide=india.cannot_provide,
        disclaimer=india.disclaimer,
        notes=(
            f"analysis_scope = {ANALYSIS_SCOPE}. Acquired to support ONE observational question: "
            f"whether greater thermal exposure between two Seaview visits is ASSOCIATED with a "
            f"larger change in the image-derived hard-coral-cover estimate across 26 repeated "
            f"Maldivian transects. It is not causal bleaching attribution, not an Indian "
            f"validation dataset, not training data for reef_health or "
            f"restoration_suitability, and it does not touch the registered synthetic champions. "
            f"This manifest is SEPARATE from noaa_crw_5km_v3_1.manifest.json, which remains the "
            f"record of the 2018-2024 India-region acquisition and is unmodified. The delivered "
            f"files carry no CRW 'mask' variable, so land and missing data both appear as NaN "
            f"and are not separable within them."
        ),
    )


def build_subset_record(
    product: crw.Product, path: Path, url: str, stats: dict, extent: dict, retrieved_at: str
) -> SubsetRecord:
    """Return the per-file provenance record for one acquired product."""
    return SubsetRecord(
        dataset_id=DATASET_ID,
        region=f"Maldives (Seaview repeat transects) / {product.label}",
        local_file=path.relative_to(PROJECT_ROOT).as_posix(),
        file_format="NetCDF-3 classic",
        file_size_bytes=path.stat().st_size,
        sha256=crw.sha256_of(path),
        retrieved_at_utc=retrieved_at,
        request_url=url,
        requested_bbox=tuple(extent["bbox"]),
        actual_bbox=stats["actual_bbox"],
        dimensions=stats["dimensions"],
        grid_spacing_deg=round(stats["lat_step_deg"], 12),
        bbox_rationale=BBOX_RATIONALE,
        variable_name=product.variable,
        variable_long_name=product.label,
        variable_units=stats["units"],
        variable_dtype=stats["variable_dtype"],
        fill_value=stats["fill_value"],
        requested_time_range=(
            f"{extent['start_date']}T12:00:00Z",
            f"{extent['end_date']}T12:00:00Z",
        ),
        actual_time_range=stats["actual_time_range"],
        n_time_steps=stats["n_time_steps"],
        time_spacing_days=round(stats["time_spacing_days"], 6),
        missing_dates=stats["missing_dates"],
        nan_percent=round(stats["nan_percent"], 6),
        min_value=stats["min_value"],
        max_value=stats["max_value"],
        mean_value=stats["mean_value"],
        median_value=stats["median_value"],
        latitude_order=stats["latitude_order"],
    )


def write_manifest(
    source: SourceRecord, subsets: list[SubsetRecord], extent: dict, diagnostics: dict
) -> Path:
    """Write the machine-readable manifest and return its path."""
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    products = {key: crw_product_by_key(key) for key in PRODUCT_KEYS}

    payload = {
        "schema": "coralsense.external.manifest/v1",
        "dataset_id": DATASET_ID,
        "analysis_scope": ANALYSIS_SCOPE,
        "relationship_to_existing_acquisitions": {
            "same_product_as": "noaa_crw_5km_v3_1",
            "same_product_note": (
                "Scientifically the identical NOAA CRW 5 km v3.1 suite, from the same NOAA "
                "ERDDAP endpoint, under the same licence determination. Only the acquisition "
                "scope differs."
            ),
            "existing_acquisition": {
                "dataset_id": "noaa_crw_5km_v3_1",
                "manifest": "data/external/metadata/noaa_crw_5km_v3_1.manifest.json",
                "purpose": "India-region environmental context",
                "geography": (
                    "Lakshadweep, Gulf of Mannar, Gulf of Kutch, Andaman and Nicobar Islands"
                ),
                "window": "2018-01-01 to 2024-12-31",
                "variables": ["sst", "sst_anomaly", "hotspot", "dhw"],
                "modified_by_this_acquisition": False,
            },
            "this_acquisition": {
                "dataset_id": DATASET_ID,
                "purpose": "Maldives Seaview temporal-analysis support, historical period",
                "geography": "One Maldives window covering the 26 repeated Seaview transects",
                "window": f"{extent['start_date']} to {extent['end_date']}",
                "variables": list(PRODUCT_KEYS),
            },
            "geographic_warning": (
                "MALDIVES_NOT_INDIA. These two acquisitions describe different countries. Files "
                "under data/external/raw/noaa_crw_5km_v3_1/maldives_seaview/ are Maldivian and "
                "must never be presented as Indian data, nor pooled with the India windows."
            ),
        },
        "source": source.to_dict(),
        "subsets": [record.to_dict() for record in subsets],
        "acquisition_window_derivation": {
            "derived_from": (
                "data/external/raw/seaview/tabular-data/seaviewsurvey_surveys.csv, "
                "ocean == 'IND' and country == 'MDV', transects with exactly two surveys"
            ),
            "n_repeat_transects": extent["n_pairs"],
            "earliest_first_survey": extent["earliest_first_survey"],
            "latest_second_survey": extent["latest_second_survey"],
            "requested_start_date": extent["start_date"],
            "requested_end_date": extent["end_date"],
            "pad_days": extent["pad_days"],
            "window_rationale": WINDOW_RATIONALE,
            "transect_bbox_unbuffered": list(extent["transect_bbox"]),
            "buffer_deg": extent["buffer_deg"],
            "requested_bbox": list(extent["bbox"]),
            "bbox_rationale": BBOX_RATIONALE,
            "hard_coded_dates": False,
        },
        "checksum_semantics": {
            "algorithm": "SHA-256",
            "scope": (
                "Hash of the subset file as retrieved by this project, computed locally at "
                "acquisition time."
            ),
            "is_publisher_canonical_checksum": False,
            "guarantees": "The local file has not been altered since acquisition.",
            "does_not_guarantee": (
                "That re-requesting the same logical subset reproduces identical bytes. ERDDAP "
                "writes a fresh NetCDF on every request and embeds generation metadata, so two "
                "requests for the same window will differ in bytes while encoding equivalent "
                "values."
            ),
            "authoritative_identity": (
                f"Scientific identity of the source product is pinned by version "
                f"'{source.version}' and DOI {crw.CRW_DOI}, not by these file hashes."
            ),
        },
        "variable_semantics": {
            "warning": (
                "Both variables below are thermal quantities. Neither is a biological "
                "observation. DHW != bleaching_percentage. HotSpot != bleaching_percentage. "
                "NOAA's DHW thresholds describe RISK, not observed outcome. Thresholding either "
                "into a reef-condition label is prohibited, and doing so here would be "
                "especially circular because these values are the PREDICTOR of the association "
                "analysis they were acquired for."
            ),
            "products": {
                key: {
                    "label": product.label,
                    "erddap_variable": product.variable,
                    "means": product.meaning,
                    "does_not_mean": product.not_a,
                }
                for key, product in products.items()
            },
        },
        "omitted_products": OMITTED_PRODUCTS,
        "diagnostics": diagnostics,
    }
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return MANIFEST_PATH


def crw_product_by_key(key: str) -> crw.Product:
    """Return the shared :class:`Product` definition for *key*."""
    for product in crw.PRODUCTS:
        if product.key == key:
            return product
    raise KeyError(f"unknown CRW product {key!r}")


def _plan(extent: dict) -> list[tuple[crw.Product, Path]]:
    """Return every (product, destination) pair to acquire."""
    del extent  # window affects the URL, not the file layout
    return [
        (
            crw_product_by_key(key),
            RAW_DIR / f"noaa_crw_5km_v3_1_{key}_maldives_seaview.nc",
        )
        for key in PRODUCT_KEYS
    ]


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1] if __doc__ else "")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan; fetch nothing.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Re-validate already-downloaded files and rewrite the manifest.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download files that already exist."
    )
    args = parser.parse_args()

    if not SURVEYS_CSV.is_file():
        print(
            f"Missing {SURVEYS_CSV.relative_to(PROJECT_ROOT)}; run scripts/fetch_seaview.py first"
        )
        return 1

    extent = derive_extent()
    crw._guard_licence_window(extent["start_date"])

    if extent["n_pairs"] != 26:
        print(
            f"Expected 26 repeated Maldives transects, found {extent['n_pairs']}. "
            f"Refusing to acquire against an unverified spine."
        )
        return 1

    retrieved_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    if args.validate_only and MANIFEST_PATH.is_file():
        retrieved_at = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["source"][
            "retrieved_at_utc"
        ]

    plan = _plan(extent)
    lat_min, lat_max, lon_min, lon_max = extent["bbox"]
    n_days = (
        datetime.strptime(extent["end_date"], "%Y-%m-%d")
        - datetime.strptime(extent["start_date"], "%Y-%m-%d")
    ).days + 1

    if args.dry_run:
        n_lat = round((lat_max - lat_min) / crw.GRID_STEP_DEG) + 1
        n_lon = round((lon_max - lon_min) / crw.GRID_STEP_DEG) + 1
        print(f"NOAA CRW 5km v3.1 historical extension  (DOI {crw.CRW_DOI})")
        print(f"  scope    : {ANALYSIS_SCOPE}  (MALDIVES, not India)")
        print(f"  endpoint : {crw.RETRIEVAL_ERDDAP}")
        print(f"  pairs    : {extent['n_pairs']} repeated transects")
        print(f"  window   : {extent['start_date']} .. {extent['end_date']}  ({n_days} days)")
        print(f"  bbox     : lat {lat_min}..{lat_max}, lon {lon_min}..{lon_max}")
        print(f"  products : {', '.join(PRODUCT_KEYS)}")
        print(f"  omitted  : {', '.join(OMITTED_PRODUCTS)}")
        for product, dest in plan:
            mb = n_days * n_lat * n_lon * 4 / 1e6
            print(f"\n  {product.key:8s} {n_days} x {n_lat} x {n_lon} cells (~{mb:.1f} MB)")
            print(f"    file : {dest.relative_to(PROJECT_ROOT)}")
        return 0

    subsets: list[SubsetRecord] = []
    per_file: dict = {}
    range_problems: list[str] = []

    for product, dest in plan:
        url = build_request_url(
            product, extent, base=crw.RETRIEVAL_ERDDAP, dataset=product.erddap_dataset
        )

        if args.validate_only:
            if not dest.is_file():
                print(f"  {product.key}: MISSING {dest}")
                return 1
        elif dest.is_file() and not args.force:
            print(f"Have {product.key}  (skipping; --force to re-download)")
        else:
            print(f"Fetching {product.key} ...", flush=True)
            crw.fetch_file(url, dest)

        stats = crw.validate_file(dest, product)
        subsets.append(build_subset_record(product, dest, url, stats, extent, retrieved_at))

        per_file[product.key] = {
            "n_time_steps": stats["n_time_steps"],
            "missing_dates": stats["missing_dates"],
            "time_spacing_days": stats["time_spacing_days"],
            "max_gap_days": stats["time_spacing_max_days"],
            "nan_percent": round(stats["nan_percent"], 4),
            "min": stats["min_value"],
            "median": stats["median_value"],
            "mean": stats["mean_value"],
            "max": stats["max_value"],
            "std": stats["std_value"],
            "range_check": stats["range_check"],
        }
        if not stats["range_check"]["ok"]:
            range_problems.append(f"{product.key}: {'; '.join(stats['range_check']['problems'])}")

        print(
            f"  {product.key}: {stats['n_time_steps']} steps, "
            f"{stats['dimensions']['lat']}x{stats['dimensions']['lon']} cells, "
            f"{stats['nan_percent']:.1f}% NaN, "
            f"range {stats['min_value']:.2f}..{stats['max_value']:.2f} {stats['units']}"
        )

    diagnostics = {
        "per_file": per_file,
        "range_problems": range_problems,
        "expected_days": n_days,
    }

    source = build_source_record(retrieved_at, extent)
    validate_manifest(source, subsets, project_root=PROJECT_ROOT, require_files=True)

    manifest_path = write_manifest(source, subsets, extent, diagnostics)
    print(f"\nManifest written: {manifest_path.relative_to(PROJECT_ROOT)}")
    if range_problems:
        print("\nRANGE CHECK PROBLEMS (reported, not corrected):")
        for problem in range_problems:
            print(f"  - {problem}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
