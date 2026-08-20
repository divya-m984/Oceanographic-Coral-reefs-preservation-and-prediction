"""
scripts/fetch_gebco_2026.py — acquire GEBCO_2026 bathymetry subsets for the four Indian reef regions.

What this does
--------------
1. Requests four regional subsets of the GEBCO_2026 Grid from the official
   THREDDS NetCDF Subset Service hosted by CEDA on behalf of BODC/GEBCO.
2. Validates each returned file read-only (geometry, units, statistics).
3. Writes a provenance manifest under ``data/external/metadata/``.

What this deliberately does NOT do
----------------------------------
- It does not touch ``data/raw/observations.csv`` or anything else in the
  synthetic prototype pipeline.
- It does not create observation rows, coordinates, or labels.
- It does not join GEBCO to any other table.
- It does not modify the DVC DAG, MLflow, or any model artefact.

GEBCO is bathymetry.  It supplies terrain context and can support a derived
``depth_m``.  It cannot supply coral cover, bleaching, disease, reef health,
restoration suitability, or sonar backscatter.  See ``docs/external_data.md``.

Usage
-----
    python scripts/fetch_gebco_2026.py            # fetch, validate, write manifest
    python scripts/fetch_gebco_2026.py --dry-run  # print the plan, fetch nothing
    python scripts/fetch_gebco_2026.py --validate-only   # re-validate local files
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
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

DATASET_ID = "gebco_2026"

# ---------------------------------------------------------------------------
# Official source
# ---------------------------------------------------------------------------
# The GEBCO_2026 Grid is held at BODC on behalf of GEBCO and mirrored in the
# CEDA archive, which exposes a THREDDS NetCDF Subset Service.  NCSS is used
# rather than the interactive download app because it is scriptable, and rather
# than the global file because that is ~7 GB and we need four small windows.
#
# NCSS returns NetCDF-3 classic, which is readable with scipy.io.netcdf_file —
# no new runtime dependency is introduced.

NCSS_BASE = (
    "https://dap.ceda.ac.uk/thredds/ncss/grid/bodc/gebco/global/gebco_2026"
    "/ice_surface_elevation/netcdf/GEBCO_2026.nc"
)
OPENDAP_URL = (
    "https://dap.ceda.ac.uk/thredds/dodsC/bodc/gebco/global/gebco_2026"
    "/ice_surface_elevation/netcdf/GEBCO_2026.nc"
)
PRODUCT_PAGE = "https://www.gebco.net/data-products-gridded-bathymetry-data/gebco2026-grid"
GEBCO_DOI = "10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa"

RAW_DIR = PROJECT_ROOT / "data" / "external" / "raw" / DATASET_ID
METADATA_DIR = PROJECT_ROOT / "data" / "external" / "metadata"

#: Nominal grid step, 15 arc-seconds.  Verified against the product metadata
#: (``geospatial_lat_resolution`` = 0.004166666666666667).
GRID_STEP_DEG = 1.0 / 240.0


# ---------------------------------------------------------------------------
# Acquisition windows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Region:
    """One acquisition window.

    These are **acquisition windows**, not observation-sampling boxes.  GEBCO is
    a continuous raster, so a rectangle is the natural request shape.  No
    observation coordinate may ever be drawn uniformly from one of these — that
    is exactly the defect the 2026-08-19 audit found in the synthetic dataset.

    Windows must be refined against real reef masks and survey site locations
    before any modelling use.
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
            "Archipelago spans roughly 8-12 deg 13' N, 71-74 deg E. The southern bound of 8.0 "
            "deliberately includes Minicoy (8 deg 15'-8 deg 20' N, 73 deg 01'-73 deg 05' E), "
            "which the earlier synthetic box [10.0, 12.5, 72.0, 74.0] excluded entirely. The "
            "northern and western bounds cover Cherbaniani Reef (12 deg 18' N, 71 deg 53' E), "
            "the north-westernmost feature. Margin ~0.3 deg on each side for surrounding water."
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
            "The Gulf runs 8 deg 35' N-9 deg 25' N and 78 deg 08' E-79 deg 30' E; the 21-island "
            "chain lies 1-10 km offshore over the 160 km between Tuticorin and Dhanushkodi, with "
            "the marine national park at 9.1375 N, 79.4725 E. Northern bound held at 9.5 to "
            "exclude Palk Bay, which is a distinct system that should be acquired separately if "
            "needed. The earlier synthetic box [8.5, 10.5, 78.0, 80.5] reached inland Tamil Nadu "
            "and across the strait toward Sri Lanka."
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
            "Marine National Park and Sanctuary centred at 22.467 N, 69.617 E, comprising 42 "
            "islands along the Jamnagar coast with reefs bordering 33 of them over ~170 km. The "
            "earlier synthetic box [22.0, 24.5, 68.0, 71.0] extended to 24.5 N, roughly 170 km "
            "inland into mainland Gujarat; the northern bound is cut to 23.0 accordingly."
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
            "Territory spans 6-14 deg N, 92-94 deg E, from Landfall Island (13 deg 39' N, "
            "93 deg 00' E) to Indira Point (6 deg 45' 10\" N, 93 deg 49' 36\" E). The eastern "
            "bound is widened to 94.3 so Indira Point and the outer eastern reefs are not clipped. "
            "This window is intrinsically large and mostly open water; it is an acquisition "
            "extent only and must be reduced to reef polygons before any modelling use."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def build_request_url(region: Region) -> str:
    """Return the NCSS request URL for *region*."""
    query = urllib.parse.urlencode(
        {
            "var": "elevation",
            "north": f"{region.lat_max}",
            "south": f"{region.lat_min}",
            "west": f"{region.lon_min}",
            "east": f"{region.lon_max}",
            "horizStride": "1",
            "accept": "netcdf",
        }
    )
    return f"{NCSS_BASE}?{query}"


def fetch_region(region: Region, dest: Path, *, timeout: int = 600) -> str:
    """
    Download one regional subset to *dest*.  Returns the request URL.

    The downloaded file is treated as immutable from this point on; nothing in
    this project rewrites it.
    """
    url = build_request_url(region)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    request = urllib.request.Request(  # noqa: S310 - fixed https host, built above
        url, headers={"User-Agent": "coralsense-mlops/external-data-acquisition"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        if response.status != 200:
            raise RuntimeError(f"{region.key}: HTTP {response.status} from {url}")
        tmp.write_bytes(response.read())

    tmp.replace(dest)
    return url


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


def validate_file(path: Path) -> dict:
    """
    Open *path* read-only and return its geometry and elevation statistics.

    The source file is never modified.  ``depth_m = -elevation`` is *not*
    written anywhere; depth is a derived quantity that applies only to marine
    cells and belongs to a later analysis step.
    """
    with netcdf_file(str(path), "r", mmap=False) as handle:
        elevation_var = handle.variables["elevation"]
        lat = np.asarray(handle.variables["lat"].data, dtype="float64")
        lon = np.asarray(handle.variables["lon"].data, dtype="float64")
        elevation = np.asarray(elevation_var.data)
        units = elevation_var.units
        units = units.decode() if isinstance(units, bytes) else str(units)

        global_attrs = {
            key: (value.decode() if isinstance(value, bytes) else value)
            for key, value in handle._attributes.items()
        }

    elevation = elevation.astype("int32")

    # GEBCO_2026 has no _FillValue on elevation: it is a continuous global
    # terrain model with a value in every cell.  Count anyway so a future
    # product that does use a fill value is not silently mis-summarised.
    nodata_count = int(np.count_nonzero(~np.isfinite(elevation.astype("float64"))))

    marine = elevation < 0
    land = elevation >= 0

    lat_step = float(np.median(np.diff(lat))) if lat.size > 1 else float("nan")
    lon_step = float(np.median(np.diff(lon))) if lon.size > 1 else float("nan")

    marine_depths = -elevation[marine].astype("float64")  # derived, in-memory only
    bands = {
        "0_10m": float(np.mean((marine_depths >= 0) & (marine_depths < 10)) * 100)
        if marine_depths.size
        else 0.0,
        "10_30m": float(np.mean((marine_depths >= 10) & (marine_depths < 30)) * 100)
        if marine_depths.size
        else 0.0,
        "30_100m": float(np.mean((marine_depths >= 30) & (marine_depths < 100)) * 100)
        if marine_depths.size
        else 0.0,
        "gt_100m": float(np.mean(marine_depths >= 100) * 100) if marine_depths.size else 0.0,
    }

    return {
        "dimensions": {"lat": int(lat.size), "lon": int(lon.size)},
        "actual_bbox": (
            float(lat.min()),
            float(lat.max()),
            float(lon.min()),
            float(lon.max()),
        ),
        "lat_step_deg": lat_step,
        "lon_step_deg": lon_step,
        "elevation_units": units,
        "min_elevation_m": float(elevation.min()),
        "max_elevation_m": float(elevation.max()),
        "median_elevation_m": float(np.median(elevation)),
        "marine_cell_count": int(marine.sum()),
        "land_cell_count": int(land.sum()),
        "nodata_count": nodata_count,
        "percent_marine": float(marine.mean() * 100),
        "depth_bands_percent_of_marine": bands,
        "global_attrs": global_attrs,
    }


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def build_source_record(retrieved_at: str) -> SourceRecord:
    """Return the product-level provenance record for GEBCO_2026."""
    return SourceRecord(
        dataset_id=DATASET_ID,
        source_name="GEBCO / Nippon Foundation-GEBCO Seabed 2030 Project",
        product_name=(
            "The GEBCO_2026 Grid - a continuous terrain model for oceans and land "
            "at 15 arc-second intervals"
        ),
        source_url=PRODUCT_PAGE,
        doi=GEBCO_DOI,
        version="GEBCO_2026",
        licence_name="Public domain (GEBCO Grid terms of use)",
        licence_url="https://www.gebco.net/",
        licence_verified=True,
        licence_verified_via=(
            "Verified 2026-08-19 from two authoritative places: the official GEBCO_2026 product "
            "page, and the 'license' global attribute carried inside the delivered NetCDF itself "
            "- 'The GEBCO Grid is placed in the public domain and may be used free of charge.'"
        ),
        redistribution_allowed=True,
        raw_tracked_in_git=False,
        citation=(
            "GEBCO Bathymetric Compilation Group 2026 (2026). The GEBCO_2026 Grid - a continuous "
            "terrain model of the global oceans and land. NERC EDS British Oceanographic Data "
            f"Centre NOC. doi:{GEBCO_DOI}"
        ),
        observation_type="compiled",
        processing_level="L4",
        sensor_type=(
            "Composite: multibeam and singlebeam echosounder soundings, satellite-derived "
            "predicted bathymetry (SRTM15+ v2.8 base), and land topography"
        ),
        original_format="NetCDF-3 classic (subset via THREDDS NCSS from a NetCDF-4 source)",
        original_crs="EPSG:4326 (WGS 84 geographic)",
        vertical_crs="EPSG:5831 (instantaneous water level, positive up)",
        spatial_resolution="15 arc-seconds (~0.00416667 deg, ~450 m at the equator)",
        temporal_resolution="static (single 2026 compilation; annual product releases)",
        geographic_scope=(
            "Source product is global (-180/180, -90/90). Four regional subsets acquired: "
            "Lakshadweep, Gulf of Mannar, Gulf of Kutch, Andaman and Nicobar Islands."
        ),
        temporal_scope="Compilation published 2026-04-17; represents no single observation epoch",
        retrieved_at_utc=retrieved_at,
        access_method=(
            f"THREDDS NetCDF Subset Service (NCSS) at {NCSS_BASE} ; "
            f"grid metadata cross-checked via OPeNDAP at {OPENDAP_URL}"
        ),
        provides_variables=("elevation_m", "depth_m (derived, marine cells only)"),
        cannot_provide=(
            "coral_cover_percentage",
            "bleaching_percentage",
            "disease_percentage",
            "reef_health",
            "restoration_suitability",
            "sonar_backscatter",
            "acoustic_complexity_index",
            "rugosity_index (at the project's intended fine scale)",
        ),
        disclaimer=(
            "GEBCO states the grid must not be used for navigation or any purpose relating to "
            "safety at sea. At ~450 m the grid is far too coarse to reproduce the synthetic "
            "fine-scale rugosity_index; any future terrain-roughness feature must carry a new, "
            "resolution-aware definition."
        ),
        notes=(
            "Acquired as an isolated real bathymetric reference. Not joined to the synthetic "
            "prototype dataset, not used for modelling, and carries no labels. Raw files are "
            "kept out of Git (see data/external/raw/.gitignore) so this layer can later be "
            "placed under DVC without rewriting history; redistribution is nevertheless "
            "permitted because the product is public domain."
        ),
    )


def build_subset_record(
    region: Region, path: Path, url: str, stats: dict, retrieved_at: str
) -> SubsetRecord:
    """Return the per-subset provenance record for one region."""
    return SubsetRecord(
        dataset_id=DATASET_ID,
        region=region.label,
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
        elevation_units=stats["elevation_units"],
        min_elevation_m=stats["min_elevation_m"],
        max_elevation_m=stats["max_elevation_m"],
        median_elevation_m=stats["median_elevation_m"],
        marine_cell_count=stats["marine_cell_count"],
        land_cell_count=stats["land_cell_count"],
        nodata_count=stats["nodata_count"],
        bbox_rationale=region.rationale,
    )


def write_manifest(
    source: SourceRecord, subsets: list[SubsetRecord], stats_by_region: dict
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
                "That re-requesting the same logical subset reproduces identical bytes. "
                "Server-side subsetting may return a byte-different NetCDF serialisation "
                "(library version, chunking, attribute order, generation timestamp) while "
                "encoding equivalent grid values. A mismatch after re-fetch means 'different "
                "bytes', not necessarily 'different data'."
            ),
            "authoritative_identity": (
                f"Scientific identity of the source product is pinned by version GEBCO_2026 "
                f"and DOI {GEBCO_DOI}, not by these file hashes."
            ),
        },
        "bathymetric_context": {
            "note": (
                "Depth-band percentages describe bathymetry only. Shallow water is NOT a reef "
                "label and must never be treated as one."
            ),
            "convention": "negative elevation = below mean sea level; depth_m = -elevation",
            "regions": stats_by_region,
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1] if __doc__ else "")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan; fetch nothing.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Re-validate already-downloaded files and rewrite the manifest.",
    )
    args = parser.parse_args()

    retrieved_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    if args.validate_only:
        # Re-validation is not re-acquisition: keep the original retrieval
        # timestamp so provenance continues to describe when the bytes were
        # actually fetched.
        existing = METADATA_DIR / f"{DATASET_ID}.manifest.json"
        if existing.is_file():
            payload = json.loads(existing.read_text(encoding="utf-8"))
            retrieved_at = payload["source"]["retrieved_at_utc"]

    if args.dry_run:
        print(f"GEBCO_2026 acquisition plan  (DOI {GEBCO_DOI})")
        print(f"  endpoint: {NCSS_BASE}")
        for region in REGIONS:
            n_lat = round((region.lat_max - region.lat_min) / GRID_STEP_DEG)
            n_lon = round((region.lon_max - region.lon_min) / GRID_STEP_DEG)
            print(f"\n  {region.label}")
            print(f"    bbox   : {region.bbox}")
            print(f"    approx : {n_lat} x {n_lon} cells (~{n_lat * n_lon * 2 / 1e6:.1f} MB int16)")
            print(f"    url    : {build_request_url(region)}")
        return 0

    subsets: list[SubsetRecord] = []
    stats_by_region: dict = {}

    for region in REGIONS:
        dest = RAW_DIR / f"{DATASET_ID}_{region.key}.nc"

        if args.validate_only:
            if not dest.is_file():
                print(f"  {region.label}: MISSING {dest}")
                return 1
            url = build_request_url(region)
        else:
            print(f"Fetching {region.label} ...", flush=True)
            url = fetch_region(region, dest)

        stats = validate_file(dest)
        subsets.append(build_subset_record(region, dest, url, stats, retrieved_at))
        stats_by_region[region.key] = {
            "label": region.label,
            "percent_marine": round(stats["percent_marine"], 3),
            "depth_bands_percent_of_marine": {
                k: round(v, 3) for k, v in stats["depth_bands_percent_of_marine"].items()
            },
        }
        print(
            f"  {region.label}: {stats['dimensions']['lat']}x{stats['dimensions']['lon']} cells, "
            f"{stats['percent_marine']:.1f}% marine, "
            f"elev {stats['min_elevation_m']:.0f}..{stats['max_elevation_m']:.0f} m"
        )

    source = build_source_record(retrieved_at)
    validate_manifest(source, subsets, project_root=PROJECT_ROOT, require_files=True)

    manifest_path = write_manifest(source, subsets, stats_by_region)
    print(f"\nManifest written: {manifest_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
