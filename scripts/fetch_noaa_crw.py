"""
scripts/fetch_noaa_crw.py — acquire NOAA Coral Reef Watch 5 km thermal products for four Indian reef regions.

What this does
--------------
1. Requests four CRW v3.1 daily 5 km variables (SST, SST anomaly, Coral
   Bleaching HotSpot, Degree Heating Week) for four regional windows over
   2018-01-01..2024-12-31, from a NOAA-operated ERDDAP griddap service.
2. Validates each returned file read-only (geometry, time axis, units, fill
   value, statistics, range plausibility).
3. Optionally cross-checks a sample against a second NOAA ERDDAP server.
4. Writes a provenance manifest under ``data/external/metadata/``.

What this deliberately does NOT do
----------------------------------
- It does not touch the synthetic prototype pipeline, its dataset, or its
  labels; nothing here reads or writes that file.
- It does not create observation rows, site tables, or joins to any other
  product (including GEBCO).
- It does not modify the DVC DAG, MLflow, or any model artefact.

Scientific position of these products
-------------------------------------
Every variable here is a **satellite-derived or algorithmically derived
thermal quantity**.  None of them is a biological observation.  In particular:

    Degree Heating Weeks  !=  bleaching percentage
    Coral Bleaching HotSpot != bleaching percentage
    Bleaching Alert Area  !=  observed bleaching class

They describe *heat stress*, which is a predictor of bleaching risk, not a
measurement of bleaching.  Thresholding any of them into a reef-condition class
would rebuild the circular-supervision defect recorded in
``docs/audits/dataset_scientific_audit_2026-08-19.md`` out of real-looking
numbers, which is worse than the synthetic version because it looks credible.
See ``docs/external_data.md``.

Usage
-----
    python scripts/fetch_noaa_crw.py               # fetch, validate, write manifest
    python scripts/fetch_noaa_crw.py --dry-run     # print the plan, fetch nothing
    python scripts/fetch_noaa_crw.py --validate-only    # re-validate local files
    python scripts/fetch_noaa_crw.py --cross-check      # compare the two NOAA servers
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.io import netcdf_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.external.provenance import (  # noqa: E402
    SourceRecord,
    SubsetRecord,
    validate_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ID = "noaa_crw_5km_v3_1"

# ---------------------------------------------------------------------------
# Official source
# ---------------------------------------------------------------------------
# The product is NOAA Coral Reef Watch's Daily Global 5 km Satellite Coral
# Bleaching Heat Stress Monitoring Product Suite, Version 3.1, produced by
# NOAA/NESDIS/STAR.  Two NOAA-operated ERDDAP servers redistribute it:
#
#   * NOAA CoastWatch central   (coastwatch.noaa.gov)
#   * NOAA PIFSC OceanWatch     (oceanwatch.pifsc.noaa.gov)
#
# Bulk retrieval uses PIFSC because the CoastWatch front end returns HTTP 502
# from its proxy on requests that take more than about 60 seconds, which a
# multi-year daily subset always does.  PIFSC served an entire seven-year
# window for the largest region in one request.  Both are NOAA; neither is a
# third-party mirror.  ``--cross-check`` verifies that the two servers return
# identical values for the same cells, so the choice of endpoint is a
# performance decision and not a data-provenance one.

RETRIEVAL_ERDDAP = "https://oceanwatch.pifsc.noaa.gov/erddap/griddap"
REFERENCE_ERDDAP = "https://coastwatch.noaa.gov/erddap/griddap"
PRODUCT_PAGE = "https://coralreefwatch.noaa.gov/product/5km/index.php"
CRW_DOI = "10.25921/6jgr-pt28"
NCEI_IDENTIFIER = "gov.noaa.nodc:CRW-5km-HeatStressProducts"

RAW_DIR = PROJECT_ROOT / "data" / "external" / "raw" / DATASET_ID
METADATA_DIR = PROJECT_ROOT / "data" / "external" / "metadata"

#: Nominal grid step, 0.05 degrees (~5 km).  Verified against the product
#: metadata (``geospatial_lat_resolution`` = 0.049999999999999996).
GRID_STEP_DEG = 0.05

# ---------------------------------------------------------------------------
# Temporal window
# ---------------------------------------------------------------------------

START_DATE = "2018-01-01"
END_DATE = "2024-12-31"

#: CoralTemp's SST lineage, as NOAA describes it (index_5km_sst.php, verified
#: 2026-08-22).  Note that these are NOT discrete day-level eras: NOAA writes
#: the handovers at month granularity apart from the two documented merge
#: windows.
#:
#:   Jan 1985 - Nov 2002   Met Office OSTIA reanalysis contributes directly.
#:   Nov 1-29, 2002        OSTIA and NOAA's reprocessed Geo-Polar Blended SST
#:                         are "linearly merged over a period of 29 days".
#:   Nov 2002 - Oct 2016   NOAA reprocessed Geo-Polar Blended is the primary
#:                         analysis.
#:   Oct 1-29, 2016        Reprocessed and near-real-time Geo-Polar merged
#:                         "over a 29 day period as well".
#:   Oct 2016 - present    NOAA operational near-real-time Geo-Polar Blended
#:                         is the primary analysis.
#:
#: OSTIA does not drop out after 2002.  NOAA states: "Bias corrections
#: originally used the NOAA National Centers for Environmental Prediction
#: (NCEP) real-time global SST.  However, in 2016, the NOAA 5km Geo-Polar
#: Blended SST product (which CRW's 5km satellite monitoring for coral reefs
#: is based on) switched to using OSTIA as the bias correction."  So OSTIA
#: informs the product this project actually acquires, as a bias-correction
#: input rather than as a directly merged source.  Nothing here may be
#: described as "OSTIA-free".
#:
#: The constant below is therefore a PROJECT POLICY floor, not a licence
#: boundary and not a source-purity claim.  Its scope is exactly: do not
#: request dates inside the historic direct-OSTIA period or the November 1-29,
#: 2002 linear-merge window, where the lineage is most entangled and hardest
#: to characterise.  It sits one day past the documented merge end, rounded to
#: a month boundary, because NOAA publishes the merge span but not its daily
#: weights.
#:
#: The redistribution decision for the acquired files is made separately, from
#: NOAA CRW's own published terms and the licence metadata delivered inside
#: the files.  It is NOT inferred from this date.
FIRST_POST_OSTIA_BLEND_REQUEST_DATE = "2002-12-01"

#: Last day of the documented direct OSTIA/Geo-Polar linear merge.  Recorded
#: for provenance; the policy floor above sits one day later.
OSTIA_MERGE_END_DATE = "2002-11-29"


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Product:
    """One CRW variable, and what it does and does not mean."""

    key: str
    erddap_dataset: str
    reference_dataset: str
    variable: str
    label: str
    #: What the number physically is.
    meaning: str
    #: The misreading this variable invites, stated so it travels with the data.
    not_a: str

    @property
    def filename_stem(self) -> str:
        """Stem used for this product's raw files."""
        return f"{DATASET_ID}_{self.key}"


PRODUCTS: tuple[Product, ...] = (
    Product(
        key="sst",
        erddap_dataset="CRW_sst_v3_1",
        reference_dataset="noaacrwsstDaily",
        variable="analysed_sst",
        label="Sea Surface Temperature (CoralTemp v3.1)",
        meaning=(
            "Daily night-only gap-free analysed sea surface temperature, in degrees Celsius. "
            "An L4 analysis: a model-assimilated blend of satellite and in-situ inputs, "
            "spatially complete by construction rather than by observation."
        ),
        not_a=(
            "Not an in-situ subsurface probe measurement. It is a skin/sub-skin SEA-SURFACE "
            "value covering a 5 km cell, so it is not interchangeable with a thermistor or CTD "
            "reading at reef depth, and must not be relabelled as one."
        ),
    ),
    Product(
        key="sst_anomaly",
        erddap_dataset="CRW_sst_anom_v1_0",
        reference_dataset="noaacrwsstanomalyDaily",
        variable="sea_surface_temperature_anomaly",
        label="Sea Surface Temperature Anomaly",
        meaning=(
            "Daily SST minus the long-term climatological mean for that location and day, in "
            "degrees Celsius. Positive means warmer than the climatology, negative cooler."
        ),
        not_a=(
            "Not a temperature, and not interchangeable with SST: it is a difference from a "
            "reference climatology, so its zero point is a baseline choice rather than a "
            "physical origin."
        ),
    ),
    Product(
        key="hotspot",
        erddap_dataset="CRW_hs_v1_0",
        reference_dataset="noaacrwhotspotDaily",
        variable="hotspot",
        label="Coral Bleaching HotSpot",
        meaning=(
            "Instantaneous coral bleaching heat stress, in degrees Celsius: daily SST minus the "
            "site's Maximum Monthly Mean (MMM) climatology. NOAA's published HotSpot maps show "
            "only positive values, but the archived variable carries the full signed difference "
            "(valid range -15 to +15), so negative values - cooler than the MMM - are normal and "
            "are not missing data. Values of 1 degC or more indicate heat stress capable of "
            "leading to bleaching, and only those accumulate into DHW."
        ),
        not_a=(
            "Not a bleaching observation and not a bleaching percentage. It quantifies present "
            "thermal stress; whether corals actually bleached is a biological outcome that "
            "requires in-water or photographic observation."
        ),
    ),
    Product(
        key="dhw",
        erddap_dataset="CRW_dhw_v1_0",
        reference_dataset="noaacrwdhwDaily",
        variable="degree_heating_week",
        label="Degree Heating Week",
        meaning=(
            "Accumulated coral bleaching heat stress over the preceding 12 weeks, in "
            "degC-weeks: the running accumulation of HotSpot values at or above 1 degC. NOAA "
            "associates 4 degC-weeks with a risk of bleaching and 8 degC-weeks with a risk of "
            "reef-wide bleaching plus mortality of heat-sensitive corals."
        ),
        not_a=(
            "Not a bleaching percentage and not a reef-condition label. NOAA's thresholds "
            "describe RISK, not measured outcome. Thresholding DHW into a reef_health class "
            "would manufacture a label out of a predictor - precisely the circular supervision "
            "the 2026-08-19 audit found in the synthetic dataset."
        ),
    ),
)

#: Deliberately not acquired.  Recorded here so the omission is a documented
#: decision rather than an oversight; see ``docs/external_data.md``.
OMITTED_PRODUCTS: dict[str, str] = {
    "bleaching_alert_area_7d_max": (
        "Omitted from this first acquisition for three independent reasons. (1) Redundant: BAA "
        "is a deterministic function of the HotSpot and DHW values already acquired, so it adds "
        "no information. (2) Temporally incompatible: the only BAA product on these servers is "
        "the 7-day MAXIMUM composite, a rolling window dated on its last day, which does not "
        "align with the daily variables and would smear a 7-day maximum across daily rows. "
        "(3) Non-stationary within our window: on 2023-12-15 NOAA revised the alert-level "
        "system, extending it from Alert Level 2 to Alert Level 5, and the ERDDAP variable "
        "metadata still declares the superseded 0-4 scheme (valid_max=4). A naive ingest would "
        "silently mix two different categorical scales inside one 2018-2024 series. It can be "
        "reconstructed later from HotSpot and DHW under a single explicit scheme."
    ),
}


# ---------------------------------------------------------------------------
# Acquisition windows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Region:
    """One acquisition window.

    Identical to the four GEBCO_2026 windows, deliberately: the two products
    describe the same four reef systems and a shared extent keeps them
    comparable.  ``tests/test_external_noaa_crw.py`` pins that equality against
    the GEBCO manifest so the two cannot drift apart silently.

    These are **acquisition windows, not reef masks**.  Most cells inside them
    are open ocean: the GEBCO bathymetry showed 98.8 % of marine cells in the
    Lakshadweep window are deeper than 100 m.  No observation coordinate may
    ever be drawn uniformly from one of these.
    """

    key: str
    label: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    rationale: str

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """``(lat_min, lat_max, lon_min, lon_max)``."""
        return (self.lat_min, self.lat_max, self.lon_min, self.lon_max)


REGIONS: tuple[Region, ...] = (
    Region(
        key="lakshadweep",
        label="Lakshadweep",
        lat_min=8.0,
        lat_max=12.6,
        lon_min=71.5,
        lon_max=74.2,
        rationale=(
            "Identical to the GEBCO_2026 acquisition window. Spans the archipelago from Minicoy "
            "(8 deg 15' N) to Cherbaniani Reef (12 deg 18' N, 71 deg 53' E). At 0.05 deg this is "
            "93 x 54 CRW cells, the great majority of which are open Arabian Sea rather "
            "than reef."
        ),
    ),
    Region(
        key="gulf_of_mannar",
        label="Gulf of Mannar",
        lat_min=8.5,
        lat_max=9.5,
        lon_min=78.0,
        lon_max=79.6,
        rationale=(
            "Identical to the GEBCO_2026 acquisition window. Covers the 21-island chain between "
            "Tuticorin and Dhanushkodi, held at 9.5 N to exclude Palk Bay, which is a distinct "
            "system. Roughly 21 x 33 CRW cells."
        ),
    ),
    Region(
        key="gulf_of_kutch",
        label="Gulf of Kutch",
        lat_min=22.0,
        lat_max=23.0,
        lon_min=68.9,
        lon_max=70.6,
        rationale=(
            "Identical to the GEBCO_2026 acquisition window. Covers the Marine National Park and "
            "the 42 islands along the Jamnagar coast. The northernmost of the four systems and "
            "the most seasonally extreme. Roughly 21 x 36 CRW cells."
        ),
    ),
    Region(
        key="andaman_nicobar",
        label="Andaman and Nicobar Islands",
        lat_min=6.5,
        lat_max=14.0,
        lon_min=92.0,
        lon_max=94.3,
        rationale=(
            "Identical to the GEBCO_2026 acquisition window. Landfall Island (13 deg 39' N) to "
            "Indira Point (6 deg 45' N). Intrinsically large and mostly open Andaman Sea; it is "
            "an acquisition extent only and must be reduced to reef polygons before any "
            "modelling use. Roughly 151 x 47 CRW cells."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def _guard_licence_window(start_date: str) -> None:
    """
    Refuse to request dates before this project's conservative lineage floor.

    This is a **project policy**, deliberately and narrowly scoped.  What it
    does: keep acquisition out of the historic period where OSTIA contributes
    directly to CoralTemp, and out of the November 1-29, 2002 window where
    OSTIA and Geo-Polar are linearly merged on weights NOAA does not publish
    per day.

    What it is *not*:

    * a NOAA-defined licence boundary -- NOAA draws no such line;
    * evidence that later CoralTemp is free of OSTIA influence -- it is not,
      NOAA states the Geo-Polar Blended product switched to OSTIA for bias
      correction in 2016, which covers the window this project acquires;
    * a claim that 2002-11-30 is scientifically "OSTIA-free".

    The redistribution decision for the acquired files rests on NOAA CRW's
    published terms and the delivered file metadata, not on this date.  See
    :data:`FIRST_POST_OSTIA_BLEND_REQUEST_DATE`.
    """
    if start_date < FIRST_POST_OSTIA_BLEND_REQUEST_DATE:
        raise SystemExit(
            f"Refusing to request CoralTemp data starting {start_date}, which is before "
            f"{FIRST_POST_OSTIA_BLEND_REQUEST_DATE}. This is a conservative project policy, "
            f"not a NOAA licence boundary: it keeps acquisition clear of the period where "
            f"the Met Office OSTIA reanalysis contributes directly to CoralTemp (January "
            f"1985 to November 2002) and of the November 1-29, 2002 window over which NOAA "
            f"linearly merged OSTIA into the reprocessed Geo-Polar Blended analysis. The "
            f"OSTIA reanalysis carries restrictive terms (academic use only, reproduction "
            f"licence application required, five-year cap), and its weighting through the "
            f"merge is not published per day. Moving this floor earlier requires re-reading "
            f"NOAA's source history, re-verifying the OSTIA terms, and recording the "
            f"outcome in the manifest before the guard is relaxed."
        )


def build_request_url(product: Product, region: Region, *, base: str, dataset: str) -> str:
    """Return the ERDDAP griddap request URL for one product/region file."""
    _guard_licence_window(START_DATE)
    # ERDDAP griddap constraints are positional and must not be percent-encoded
    # as a whole; only the variable name is user-ish data here and it is a fixed
    # identifier from PRODUCTS.
    selector = (
        f"[({START_DATE}T12:00:00Z):1:({END_DATE}T12:00:00Z)]"
        f"[({region.lat_min}):1:({region.lat_max})]"
        f"[({region.lon_min}):1:({region.lon_max})]"
    )
    query = urllib.parse.quote(f"{product.variable}{selector}", safe="[]():.,-")
    return f"{base}/{dataset}.nc?{query}"


def fetch_file(url: str, dest: Path, *, timeout: int = 900, attempts: int = 3) -> None:
    """
    Download *url* to *dest*, retrying transient server failures.

    Subsetting a multi-year daily window is expensive server-side, so a slow or
    briefly overloaded endpoint is an ordinary event rather than an error worth
    aborting a two-hour acquisition for.  The download lands on a ``.part`` file
    and is renamed only once complete, so an interrupted attempt can never be
    mistaken for a finished one.  Once renamed the file is immutable; nothing in
    this project rewrites it.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    request = urllib.request.Request(  # noqa: S310 - fixed https host, built above
        url, headers={"User-Agent": "coralsense-mlops/external-data-acquisition"}
    )
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status} from {url}")
                with tmp.open("wb") as handle:
                    while chunk := response.read(4 * 1024 * 1024):
                        handle.write(chunk)
            if tmp.stat().st_size == 0:
                raise RuntimeError(f"empty response from {url}")
        except Exception as error:  # noqa: BLE001 - retried below, re-raised on the last attempt
            tmp.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            wait = 15 * attempt
            print(f"    attempt {attempt}/{attempts} failed ({error}); retrying in {wait}s")
            time.sleep(wait)
        else:
            tmp.replace(dest)
            return


def sha256_of(path: Path) -> str:
    """Return the SHA-256 of *path* as lowercase hex."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Validation (read-only)
# ---------------------------------------------------------------------------


def _decode_time(handle) -> tuple[np.ndarray, str]:
    """Return time values as datetime64[s] plus the raw units string."""
    time_var = handle.variables["time"]
    units = time_var.units
    units = units.decode() if isinstance(units, bytes) else str(units)
    raw = np.asarray(time_var.data, dtype="float64")
    if "since 1970-01-01" not in units:
        raise ValueError(f"unexpected time units {units!r}")
    return np.asarray(raw, dtype="int64").astype("datetime64[s]"), units


def validate_file(path: Path, product: Product) -> dict:
    """
    Open *path* read-only and return geometry, time-axis and value statistics.

    Nothing is written, deleted, or repaired.  Values that look wrong are
    reported, not corrected: a decoded fill value masquerading as a 30000-degree
    temperature is a finding, and silently dropping it would hide the finding.
    """
    with netcdf_file(str(path), "r", mmap=False) as handle:
        var = handle.variables[product.variable]
        lat = np.asarray(handle.variables["latitude"].data, dtype="float64")
        lon = np.asarray(handle.variables["longitude"].data, dtype="float64")
        times, time_units = _decode_time(handle)
        delivered_dtype = np.dtype(var.data.dtype).name
        values = np.asarray(var.data, dtype="float64")

        attrs = {
            key: (value.decode() if isinstance(value, bytes) else value)
            for key, value in var._attributes.items()
        }
        global_attrs = {
            key: (value.decode() if isinstance(value, bytes) else value)
            for key, value in handle._attributes.items()
        }

    units = str(attrs.get("units", ""))
    fill_raw = attrs.get("_FillValue")
    fill_value = (
        "NaN"
        if fill_raw is None or np.isnan(np.asarray(fill_raw, dtype="float64"))
        else (f"{float(np.asarray(fill_raw, dtype='float64')):g}")
    )

    finite = np.isfinite(values)
    n_total = int(values.size)
    n_valid = int(finite.sum())

    lat_step = float(np.median(np.diff(lat))) if lat.size > 1 else float("nan")
    lon_step = float(np.median(np.diff(lon))) if lon.size > 1 else float("nan")

    day_deltas = np.diff(times).astype("timedelta64[s]").astype("float64") / 86400.0
    first, last = times[0], times[-1]
    expected_days = int((last - first).astype("timedelta64[D]").astype("int64")) + 1
    missing_dates = expected_days - int(times.size)

    stats: dict = {
        "dimensions": {"time": int(times.size), "lat": int(lat.size), "lon": int(lon.size)},
        "actual_bbox": (float(lat.min()), float(lat.max()), float(lon.min()), float(lon.max())),
        "latitude_order": "ascending" if lat.size > 1 and lat[1] > lat[0] else "descending",
        "lat_step_deg": abs(lat_step),
        "lon_step_deg": abs(lon_step),
        "time_units": time_units,
        "actual_time_range": (
            f"{np.datetime_as_string(first, unit='s')}Z",
            f"{np.datetime_as_string(last, unit='s')}Z",
        ),
        "n_time_steps": int(times.size),
        "time_spacing_days": float(np.median(day_deltas)) if day_deltas.size else float("nan"),
        "time_spacing_max_days": float(day_deltas.max()) if day_deltas.size else float("nan"),
        "expected_days": expected_days,
        "missing_dates": missing_dates,
        "units": units,
        "variable_dtype": delivered_dtype,
        "fill_value": fill_value,
        "n_cells_total": n_total,
        "n_cells_valid": n_valid,
        "nan_percent": float((1.0 - n_valid / n_total) * 100.0) if n_total else 0.0,
        "global_attrs": global_attrs,
        "variable_attrs": {k: str(v) for k, v in attrs.items()},
    }

    if n_valid:
        valid = values[finite]
        stats.update(
            {
                "min_value": float(valid.min()),
                "max_value": float(valid.max()),
                "mean_value": float(valid.mean()),
                "median_value": float(np.median(valid)),
                "std_value": float(valid.std()),
            }
        )
    else:
        stats.update(
            dict.fromkeys(("min_value", "max_value", "mean_value", "median_value", "std_value"))
        )

    stats["range_check"] = _range_check(product, stats)
    return stats


#: Physically sensible bounds per product.  Deliberately wider than the values
#: we expect: the point is to catch a decoded fill value or a unit slip, not to
#: police tropical oceanography.
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "sst": (-2.0, 40.0),
    "sst_anomaly": (-15.0, 15.0),
    "hotspot": (-15.0, 15.0),
    "dhw": (0.0, 100.0),
}


def _range_check(product: Product, stats: dict) -> dict:
    """Compare observed extrema against plausible physical bounds."""
    low, high = PLAUSIBLE_RANGES[product.key]
    observed_min, observed_max = stats["min_value"], stats["max_value"]
    problems: list[str] = []
    if observed_min is None:
        problems.append("no valid values at all")
    else:
        if observed_min < low:
            problems.append(f"minimum {observed_min:g} below plausible {low:g}")
        if observed_max > high:
            problems.append(
                f"maximum {observed_max:g} above plausible {high:g} "
                f"(a decoded fill value would look like this)"
            )
    return {"bounds": [low, high], "ok": not problems, "problems": problems}


#: Tolerance for declaring two servers' values equal.  CRW stores these
#: quantities as scaled 16-bit integers, so a float round-trip through two
#: different ERDDAP installations can differ in the last bit or two.  1e-5 degC
#: is several orders of magnitude below anything physically meaningful and still
#: far tighter than any real disagreement would be.
CROSS_CHECK_TOLERANCE = 1e-5


def _load_for_comparison(url: str, variable: str, tmp: Path) -> tuple[np.ndarray, np.ndarray]:
    """Fetch *url* and return ``(values, valid_mask)`` in north-up orientation."""
    fetch_file(url, tmp, timeout=300)
    try:
        with netcdf_file(str(tmp), "r", mmap=False) as handle:
            lat = np.asarray(handle.variables["latitude"].data, dtype="float64")
            var = handle.variables[variable]
            values = np.asarray(var.data, dtype="float64")
            fill = var._attributes.get("_FillValue")
            # Normalise latitude direction before comparing; the two servers
            # publish the same grid in opposite order for some variables.
            if lat.size > 1 and lat[1] < lat[0]:
                values = values[:, ::-1, :]
    finally:
        tmp.unlink(missing_ok=True)

    valid = np.isfinite(values)
    if fill is not None:
        fill = float(np.asarray(fill, dtype="float64"))
        if np.isfinite(fill):
            valid &= values != fill
    return values, valid


def cross_check(product: Product, region: Region, *, days: int = 5) -> dict:
    """
    Fetch a small sample from both NOAA servers and compare them.

    This is what makes the choice of retrieval endpoint a performance decision
    rather than a provenance one: if PIFSC and CoastWatch return the same
    numbers for the same cells, the faster server is not a different dataset.

    Two things are compared, and they are reported separately on purpose:

    ``values_agree``
        Do the two servers report the same number for every cell that both
        consider valid?  This is the question that matters for provenance.

    ``fill_encoding_matches``
        Do they mark missing data the same way?  They do not: CoastWatch
        declares ``_FillValue = -327.68`` while PIFSC converts to ``NaN``.  That
        is a serialisation difference, not a data difference, and conflating the
        two would report a spurious disagreement.
    """
    end = (datetime.strptime(START_DATE, "%Y-%m-%d") + timedelta(days=days - 1)).strftime(
        "%Y-%m-%d"
    )
    selector = (
        f"[({START_DATE}T12:00:00Z):1:({end}T12:00:00Z)]"
        f"[({region.lat_min}):1:({region.lat_max})]"
        f"[({region.lon_min}):1:({region.lon_max})]"
    )
    query = urllib.parse.quote(f"{product.variable}{selector}", safe="[]():.,-")

    loaded: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, base, dataset in (
        ("retrieval", RETRIEVAL_ERDDAP, product.erddap_dataset),
        ("reference", REFERENCE_ERDDAP, product.reference_dataset),
    ):
        loaded[name] = _load_for_comparison(
            f"{base}/{dataset}.nc?{query}",
            product.variable,
            RAW_DIR / f".crosscheck_{name}_{product.key}_{region.key}.nc",
        )

    (a, a_valid), (b, b_valid) = loaded["retrieval"], loaded["reference"]
    if a.shape != b.shape:
        return {"values_agree": False, "reason": f"shape {a.shape} != {b.shape}"}

    both = a_valid & b_valid
    max_abs = float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0
    return {
        "values_agree": bool(max_abs <= CROSS_CHECK_TOLERANCE),
        "max_abs_difference": max_abs,
        "tolerance": CROSS_CHECK_TOLERANCE,
        "cells_compared": int(both.sum()),
        "shape": list(a.shape),
        "fill_encoding_matches": bool(int((a_valid != b_valid).sum()) == 0),
        "valid_mask_mismatches": int((a_valid != b_valid).sum()),
        "fill_encoding_note": (
            "The two servers encode missing data differently: CoastWatch declares "
            "_FillValue = -327.68, PIFSC delivers NaN. Cells invalid on one server and not the "
            "other are excluded from the value comparison. This is a serialisation difference, "
            "not a difference in the underlying CRW data."
        ),
        "retrieval_server": RETRIEVAL_ERDDAP,
        "reference_server": REFERENCE_ERDDAP,
    }


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def build_source_record(retrieved_at: str) -> SourceRecord:
    """Return the product-level provenance record for the CRW 5 km suite."""
    return SourceRecord(
        dataset_id=DATASET_ID,
        source_name="NOAA Coral Reef Watch (NOAA/NESDIS/STAR)",
        product_name=(
            "NOAA Coral Reef Watch Daily Global 5km (0.05 degree) Satellite Coral Bleaching "
            "Heat Stress Monitoring Product Suite"
        ),
        source_url=PRODUCT_PAGE,
        doi=CRW_DOI,
        product_identifier=NCEI_IDENTIFIER,
        version="v3.1 (released 2018-08-01)",
        licence_name=(
            "US Government public domain, attribution requested (NOAA Coral Reef Watch). "
            "Basis: NOAA CRW's published terms plus the licence metadata delivered inside "
            "the acquired files. Separately, and NOT as the basis for this determination, "
            "the project declines to acquire dates before 2002-12-01, keeping clear of the "
            "period where the restrictively licensed Met Office OSTIA reanalysis contributes "
            "directly to CoralTemp and of the November 1-29, 2002 merge window."
        ),
        licence_url="https://coralreefwatch.noaa.gov/satellite/docs/recommendations_crw_citation.php",
        licence_verified=True,
        licence_verified_via=(
            "Verified 2026-08-20 from first-party NOAA sources. (1) NOAA Coral Reef Watch's own "
            "disclaimer and citation page states: 'NOAA Coral Reef Watch (CRW) data posted on "
            "the internet are freely available to the public. All content on this website is "
            "considered to be in the public domain and may be distributed freely. We rely on "
            "the ethics and integrity of the user to ensure that the source of data and "
            "products is appropriately cited and credited.' (2) The 'license' global attribute "
            "delivered inside the NetCDF files themselves repeats the CRW statement that the "
            "data 'are available for use without restriction' with credit requested. (3) For "
            "CoralTemp SST specifically, that same attribute additionally carries an OSTIA "
            "Usage Statement and a GHRSST statement ('free and open'), reflecting CoralTemp's "
            "multi-source lineage. The determination above rests on (1) and (2) -- NOAA CRW's "
            "own publication terms for the product it distributes, and the licence metadata "
            "delivered with the files -- and is NOT inferred from the acquisition window's "
            "position in that lineage. SOURCE LINEAGE, recorded separately for provenance and "
            "re-verified 2026-08-22 against NOAA's CoralTemp source-history page "
            "(coralreefwatch.noaa.gov/product/5km/index_5km_sst.php): OSTIA reanalysis "
            "contributes directly January 1985 to November 2002; 'The OSTIA reanalysis and "
            "NOAA's reprocessed 5km Geo-Polar Blended SST dataset were linearly merged over a "
            "period of 29 days, from November 1-29, 2002'; NOAA reprocessed Geo-Polar Blended "
            "is the primary analysis from November 2002 to October 2016; a second 29-day merge "
            "runs October 1-29, 2016; NOAA operational near-real-time Geo-Polar Blended is "
            "primary from October 2016 onward. OSTIA does NOT drop out of the later record: "
            "NOAA states 'Bias corrections originally used the NOAA National Centers for "
            "Environmental Prediction (NCEP) real-time global SST. However, in 2016, the NOAA "
            "5km Geo-Polar Blended SST product (which CRW's 5km satellite monitoring for coral "
            "reefs is based on) switched to using OSTIA as the bias correction.' The acquired "
            "2018-2024 window therefore rests on a product that uses OSTIA as a bias-correction "
            "input, and must not be described as OSTIA-free. scripts/fetch_noaa_crw.py enforces "
            "a conservative project-policy floor of 2002-12-01 -- excluding the direct-OSTIA "
            "period and the 2002 linear-merge window only -- which is a provenance policy, not "
            "a licence boundary and not a source-purity claim."
        ),
        redistribution_allowed=True,
        raw_tracked_in_git=False,
        citation=(
            "Liu, Gang; Heron, Scott F.; Eakin, C. Mark; De La Cour, Jacqueline L.; Geiger, "
            "Erick F.; Tirak, Kyle V.; Skirving, William J.; Strong, Alan E. (2018). NOAA Coral "
            "Reef Watch (CRW) Daily Global 5-km (0.05 degree) Satellite Coral Bleaching Heat "
            "Stress Monitoring Product Suite. [Subset used: SST, SST Anomaly, Coral Bleaching "
            "HotSpot and Degree Heating Week, 2018-01-01 to 2024-12-31, for four Indian reef "
            f"regions]. NOAA National Centers for Environmental Information. Dataset. "
            f"https://doi.org/{CRW_DOI}. Accessed 2026-08-20."
        ),
        observation_type="derived",
        processing_level="L4",
        sensor_type=(
            "Composite. SST inputs: AVHRR, VIIRS, GOES Imager, MTSAT Imager, AHI, ABI, SEVIRI, "
            "ATSR series, plus moored/drifting/TAO buoys and ship intake. HotSpot, SST Anomaly "
            "and DHW are computed from the resulting CoralTemp v3.1 SST analysis and are not "
            "separately sensed."
        ),
        original_format="NetCDF-3 classic (server-side subset of a NetCDF-4 source)",
        original_crs="EPSG:4326 (WGS 84 geographic), equirectangular, grid-cell centres",
        spatial_resolution="0.05 degrees (~5 km)",
        temporal_resolution="daily",
        geographic_scope=(
            "Source product is global (-180/180, -90/90). Four regional subsets acquired: "
            "Lakshadweep, Gulf of Mannar, Gulf of Kutch, Andaman and Nicobar Islands."
        ),
        temporal_scope=(
            f"Source product spans 1985-01-01 to present. Acquired window: {START_DATE} to "
            f"{END_DATE} (seven complete calendar years)."
        ),
        retrieved_at_utc=retrieved_at,
        access_method=(
            f"ERDDAP griddap server-side subsetting. Bulk retrieval from the NOAA PIFSC "
            f"OceanWatch ERDDAP at {RETRIEVAL_ERDDAP}; values cross-verified against the NOAA "
            f"CoastWatch ERDDAP at {REFERENCE_ERDDAP}. Both are NOAA-operated; no third-party "
            f"mirror was used."
        ),
        variable_units={
            "sea_surface_temperature_c": "degree_C",
            "sea_surface_temperature_anomaly_c": "degree_C",
            "coral_bleaching_hotspot_c": "degree_C",
            "degree_heating_week": "degree_Celsius_weeks",
        },
        provides_variables=(
            "sea_surface_temperature_c (satellite L4 analysis, surface, 5 km cell)",
            "sea_surface_temperature_anomaly_c (departure from climatology)",
            "coral_bleaching_hotspot_c (instantaneous thermal stress)",
            "degree_heating_week (accumulated thermal stress over 12 weeks)",
        ),
        cannot_provide=(
            "coral_cover_percentage",
            "bleaching_percentage",
            "disease_percentage",
            "reef_health",
            "restoration_suitability",
            "sonar_backscatter",
            "acoustic_complexity_index",
            "rugosity_index",
            "depth_m",
            "ph",
            "turbidity_ntu",
            "dissolved_oxygen_mg_l",
            "salinity_ppt",
            "in-situ subsurface water_temperature_c at reef depth",
        ),
        disclaimer=(
            "These are thermal-stress PREDICTORS, not biological observations. DHW is not a "
            "bleaching percentage; HotSpot is not a bleaching percentage; Bleaching Alert Area "
            "is not an observed bleaching class. NOAA's DHW thresholds (4 degC-weeks, "
            "8 degC-weeks) describe risk, not an observed outcome. Deriving a "
            "reef_health or restoration_suitability label by thresholding any variable here "
            "would recreate the label-construction leakage documented in the 2026-08-19 audit "
            "using real-looking inputs. SST is a sea-SURFACE analysis and is not an in-situ "
            "subsurface measurement at reef depth."
        ),
        notes=(
            "Acquired as an isolated real thermal reference, source #2 after GEBCO_2026. Not "
            "joined to the synthetic prototype dataset, not joined to GEBCO, not used for "
            "modelling, and carrying no labels. The Bleaching Alert Area product was "
            "deliberately omitted; see the manifest's omitted_products block. The delivered "
            "files carry no CRW 'mask' variable, so land and missing data are not separable "
            "from each other within them - both appear as NaN."
        ),
    )


def build_subset_record(
    product: Product, region: Region, path: Path, url: str, stats: dict, retrieved_at: str
) -> SubsetRecord:
    """Return the per-file provenance record for one product/region pair."""
    return SubsetRecord(
        dataset_id=DATASET_ID,
        region=f"{region.label} / {product.label}",
        local_file=path.relative_to(PROJECT_ROOT).as_posix(),
        file_format="NetCDF-3 classic",
        file_size_bytes=path.stat().st_size,
        sha256=sha256_of(path),
        retrieved_at_utc=retrieved_at,
        request_url=url,
        requested_bbox=region.bbox,
        actual_bbox=stats["actual_bbox"],
        dimensions=stats["dimensions"],
        grid_spacing_deg=round(stats["lat_step_deg"], 12),
        bbox_rationale=region.rationale,
        variable_name=product.variable,
        variable_long_name=product.label,
        variable_units=stats["units"],
        variable_dtype=stats["variable_dtype"],
        fill_value=stats["fill_value"],
        requested_time_range=(f"{START_DATE}T12:00:00Z", f"{END_DATE}T12:00:00Z"),
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
    source: SourceRecord,
    subsets: list[SubsetRecord],
    diagnostics: dict,
) -> Path:
    """Write the machine-readable manifest and return its path."""
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = METADATA_DIR / f"{DATASET_ID}.manifest.json"

    payload = {
        "schema": "coralsense.external.manifest/v1",
        "dataset_id": DATASET_ID,
        "source": source.to_dict(),
        "subsets": [record.to_dict() for record in subsets],
        "checksum_semantics": {
            "algorithm": "SHA-256",
            "scope": (
                "Hash of the regional subset file as retrieved by this project, computed "
                "locally at acquisition time."
            ),
            "is_publisher_canonical_checksum": False,
            "guarantees": "The local file has not been altered since acquisition.",
            "does_not_guarantee": (
                "That re-requesting the same logical subset reproduces identical bytes. ERDDAP "
                "writes a fresh NetCDF on every request and embeds generation metadata, so two "
                "requests for the same window will differ in bytes while encoding equivalent "
                "values. A mismatch after re-fetch means 'different bytes', not necessarily "
                "'different data'."
            ),
            "authoritative_identity": (
                f"Scientific identity of the source product is pinned by version "
                f"'{source.version}' and DOI {CRW_DOI}, not by these file hashes."
            ),
        },
        "variable_semantics": {
            "warning": (
                "Every variable below is a thermal quantity. None is a biological observation. "
                "DHW != bleaching_percentage. HotSpot != bleaching_percentage. Bleaching Alert "
                "Area != observed bleaching class. These indicate thermal stress and bleaching "
                "RISK. Thresholding any of them into a reef-condition label is prohibited."
            ),
            "products": {
                product.key: {
                    "label": product.label,
                    "erddap_variable": product.variable,
                    "means": product.meaning,
                    "does_not_mean": product.not_a,
                }
                for product in PRODUCTS
            },
        },
        "omitted_products": OMITTED_PRODUCTS,
        "diagnostics": diagnostics,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _plan() -> list[tuple[Product, Region, Path]]:
    """Return every (product, region, destination) triple to acquire."""
    return [
        (product, region, RAW_DIR / f"{product.filename_stem}_{region.key}.nc")
        for product in PRODUCTS
        for region in REGIONS
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
        "--cross-check",
        action="store_true",
        help="Also compare a sample against the second NOAA ERDDAP server.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download files that already exist."
    )
    args = parser.parse_args()

    _guard_licence_window(START_DATE)

    retrieved_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    previous_cross_check: dict | None = None
    if args.validate_only:
        # Re-validation is not re-acquisition: keep the original retrieval
        # timestamp so provenance continues to describe when the bytes were
        # actually fetched.  Carry forward the cross-server result too, so a
        # plain --validate-only does not silently delete evidence it simply did
        # not re-gather.
        existing = METADATA_DIR / f"{DATASET_ID}.manifest.json"
        if existing.is_file():
            payload = json.loads(existing.read_text(encoding="utf-8"))
            retrieved_at = payload["source"]["retrieved_at_utc"]
            previous_cross_check = payload.get("diagnostics", {}).get("cross_server_check")

    plan = _plan()

    if args.dry_run:
        n_days = (
            datetime.strptime(END_DATE, "%Y-%m-%d") - datetime.strptime(START_DATE, "%Y-%m-%d")
        ).days + 1
        print(f"NOAA Coral Reef Watch 5km v3.1 acquisition plan  (DOI {CRW_DOI})")
        print(f"  endpoint : {RETRIEVAL_ERDDAP}")
        print(f"  window   : {START_DATE} .. {END_DATE}  ({n_days} days)")
        print(f"  products : {', '.join(p.key for p in PRODUCTS)}")
        print(f"  omitted  : {', '.join(OMITTED_PRODUCTS)}")
        total = 0.0
        for product, region, dest in plan:
            n_lat = round((region.lat_max - region.lat_min) / GRID_STEP_DEG) + 1
            n_lon = round((region.lon_max - region.lon_min) / GRID_STEP_DEG) + 1
            mb = n_lat * n_lon * n_days * 8 / 1e6
            total += mb
            print(f"\n  {product.key:12s} {region.label}")
            print(f"    approx : {n_days} x {n_lat} x {n_lon} cells (<= ~{mb:.0f} MB)")
            print(f"    file   : {dest.relative_to(PROJECT_ROOT)}")
        # Upper bound: assumes 8 bytes per cell.  In practice only SST is
        # delivered as float64; the three stress products arrive as float32, so
        # the real total lands near 60 % of this figure.
        print(f"\n  upper bound: ~{total / 1000:.2f} GB across {len(plan)} files")
        return 0

    subsets: list[SubsetRecord] = []
    per_file: dict = {}
    range_problems: list[str] = []

    for product, region, dest in plan:
        url = build_request_url(
            product, region, base=RETRIEVAL_ERDDAP, dataset=product.erddap_dataset
        )

        if args.validate_only:
            if not dest.is_file():
                print(f"  {product.key} / {region.label}: MISSING {dest}")
                return 1
        elif dest.is_file() and not args.force:
            print(f"Have {product.key} / {region.label}  (skipping; --force to re-download)")
        else:
            print(f"Fetching {product.key} / {region.label} ...", flush=True)
            fetch_file(url, dest)

        stats = validate_file(dest, product)
        subsets.append(build_subset_record(product, region, dest, url, stats, retrieved_at))

        key = f"{product.key}/{region.key}"
        per_file[key] = {
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
            range_problems.append(f"{key}: {'; '.join(stats['range_check']['problems'])}")

        print(
            f"  {product.key}/{region.key}: {stats['n_time_steps']} steps, "
            f"{stats['dimensions']['lat']}x{stats['dimensions']['lon']} cells, "
            f"{stats['nan_percent']:.1f}% NaN, "
            f"range {stats['min_value']:.2f}..{stats['max_value']:.2f} {stats['units']}"
        )

    diagnostics: dict = {"per_file": per_file, "range_problems": range_problems}
    if previous_cross_check is not None:
        diagnostics["cross_server_check"] = previous_cross_check

    if args.cross_check:
        print("\nCross-checking NOAA servers ...", flush=True)
        checks = {}
        for product in PRODUCTS:
            region = REGIONS[1]  # Gulf of Mannar: smallest window, cheapest sample
            result = cross_check(product, region)
            checks[product.key] = result
            verdict = "AGREE" if result["values_agree"] else "DIFFER"
            fill = "same" if result.get("fill_encoding_matches") else "differs"
            print(
                f"  {product.key}: values {verdict} "
                f"(max abs diff {result.get('max_abs_difference'):.3g} over "
                f"{result.get('cells_compared')} cells); fill encoding {fill}"
            )
        diagnostics["cross_server_check"] = checks

    source = build_source_record(retrieved_at)
    validate_manifest(source, subsets, project_root=PROJECT_ROOT, require_files=True)

    manifest_path = write_manifest(source, subsets, diagnostics)
    print(f"\nManifest written: {manifest_path.relative_to(PROJECT_ROOT)}")
    if range_problems:
        print("\nRANGE CHECK PROBLEMS (reported, not corrected):")
        for problem in range_problems:
            print(f"  - {problem}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
