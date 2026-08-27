#!/usr/bin/env python
"""
scripts/fetch_seaview.py — acquire the Seaview Survey *tabular* data.

Real biological data source #1.  This is the first external product in the
project that carries a biological quantity rather than a physical covariate:
the proportional cover of benthic groups on a reef, at ~1 m² photo-quadrat
resolution.  The imagery and the coordinates are real field records; the cover
values are *image-derived benthic-cover estimates* produced from them by an
image classifier.  Those are three different things and this module keeps them
apart — see :func:`_variable_semantics` and its ``evidence_layers`` block.

What this script deliberately does **not** fetch
-----------------------------------------------
The published collection at ``data.qld.edu.au/public/Q1281`` also contains
``photo-quadrats/`` (over one million JPEG images, ~860 per-survey ZIPs),
``annotated-images/`` and ``survey-previews/``.  None of them is downloaded.
This project needs the cover *values*, not the imagery they were derived from,
and the image archive is orders of magnitude larger than the tables.
:data:`FORBIDDEN_REMOTE_PATHS` records that refusal in code so it survives the
next person reading the directory listing and wondering why.

The two acquired objects are ``tabular-data.zip`` (18 CSVs) and the publisher's
own dataset documentation PDF, which is the authoritative description of the
table schemas and is small enough to keep alongside them.

Cover values are *not* direct field measurements
------------------------------------------------
98.3% of the Indian Ocean quadrat cover values in this product were produced by
a convolutional neural network (VGG-D 16) classifying 50 points per quadrat,
not by a human looking at the image.  The remaining 1.7% are the human-annotated
train/test quadrats.  They are therefore ML-estimated benthic cover, never
"directly observed", "field-measured" or "biological ground truth" — the
published classifier validation is preserved in the manifest, but validation
accuracy does not rename a prediction into a measurement.  The manifest and
:mod:`src.external.provenance` record that distinction; see
``docs/external_data.md`` §8.4.

Usage
-----
    python scripts/fetch_seaview.py --dry-run       # print the plan, fetch nothing
    python scripts/fetch_seaview.py                 # download, extract, validate, write manifest
    python scripts/fetch_seaview.py --validate-only # re-validate on-disk files, rewrite manifest
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.external.provenance import (  # noqa: E402
    SourceRecord,
    SubsetRecord,
    validate_manifest,
)

DATASET_ID = "seaview_survey"
RAW_DIR = PROJECT_ROOT / "data" / "external" / "raw" / "seaview"
TABLE_DIR = RAW_DIR / "tabular-data"
MANIFEST_PATH = PROJECT_ROOT / "data" / "external" / "metadata" / "seaview_survey.manifest.json"

#: The publisher serves this collection over plain HTTP only; the host does not
#: complete a TLS handshake on port 443.  Recorded rather than silently worked
#: around — it means the transfer is unauthenticated, which is why every file's
#: SHA-256 is pinned in the manifest.
BASE_URL = "http://data.qld.edu.au/public/Q1281"

ARCHIVE_NAME = "tabular-data.zip"
DOC_NAME = "Seaview_Survey_photoquadrat_data_collection.pdf"

#: Remote subtrees this script must never request.  See the module docstring.
FORBIDDEN_REMOTE_PATHS: tuple[str, ...] = (
    "photo-quadrats/",
    "annotated-images/",
    "survey-previews/",
)

DOI = "10.14264/uql.2019.930"
PAPER_DOI = "10.1038/s41597-020-00698-6"

#: The licence exactly as the publisher's own record words it.  Kept verbatim
#: and separately from our reading of it: UQ did write this, and rewriting the
#: publisher's field into our preferred wording would erase evidence.
SOURCE_REPORTED_LICENCE_LABEL = "Creative Commons Attribution 3.0 International (CC BY 3.0)"

#: Our normalized reading.  The label above is imprecise — CC BY 3.0 was issued
#: as *Unported* and in ported national forms, and "International" only begins
#: at 4.0 — but the licence *URI* the record supplies is unambiguous, and it
#: resolves to CC BY 3.0 Unported.  The URI governs.
NORMALIZED_LICENCE = "CC BY 3.0 Unported"
LICENCE_URL = "https://creativecommons.org/licenses/by/3.0/"

#: Ocean code → the countries/territories it actually contains, taken from the
#: delivered ``seaviewsurvey_surveys.csv``.  ``IND`` is the *Indian Ocean*
#: basin code, not India: no survey in this dataset has ``country == 'IND'``.
#: See :data:`GEOGRAPHIC_TRANSFER_STATUS`.
INDIAN_OCEAN_TERRITORIES: dict[str, str] = {
    "MDV": "Maldives",
    "CHA": "Chagos Archipelago (British Indian Ocean Territory)",
}

#: The project's four target reef systems, all Indian.  None of them is
#: surveyed by this dataset; the nearest Seaview transect is ~386 km from the
#: southern edge of Lakshadweep.  Any model fitted here and applied there is
#: performing a geographic transfer, and must say so.
GEOGRAPHIC_TRANSFER_STATUS = "INDIAN_OCEAN_NOT_INDIA"


@dataclass(frozen=True)
class Table:
    """One CSV in the tabular package."""

    filename: str
    unit: str
    description: str
    primary_key: str
    foreign_keys: str
    lat_col: str | None = None
    lng_col: str | None = None


TABLES: tuple[Table, ...] = (
    Table(
        "seaviewsurvey_surveys.csv",
        "survey",
        "One row per survey (transect visit). Carries the survey date, the ocean "
        "and country codes, start/end coordinates, and the five merged benthic "
        "group proportions aggregated to survey level.",
        "surveyid",
        "transectid groups repeat visits to the same transect",
        "lat_start",
        "lng_start",
    ),
    Table(
        "seaviewsurvey_quadrats.csv",
        "quadrat",
        "The survey → image → quadrat hierarchy, one row per quadrat. This is the "
        "table that makes the one-to-many structure explicit.",
        "quadratid",
        "surveyid → surveys.surveyid; imageid is the parent raw image",
        None,
        None,
    ),
    Table(
        "seaviewsurvey_labelsets.csv",
        "label",
        "Benthic label definitions per region: short code, functional group, full "
        "name, merged label, and worked examples. The join key from a cover column "
        "to its functional group.",
        "(region, label)",
        "label ← the column names of the reefcover tables",
        None,
        None,
    ),
    Table(
        "seaviewsurvey_annotations.csv",
        "annotation point",
        "Automated point annotations: one row per classified point. 55,229,185 rows. "
        "This is the CNN output the cover proportions are computed from.",
        "(quadratid, y, x)",
        "quadratid → quadrats.quadratid",
        None,
        None,
    ),
    Table(
        "seaviewsurvey_reefcover_indianocean.csv",
        "quadrat",
        "Central Indian Ocean quadrat-level proportional cover, one column per "
        "detailed benthic label. The Indian Ocean subset this project cares about.",
        "quadratid",
        "surveyid, imageid → surveys / quadrats",
        "lat",
        "lng",
    ),
    Table(
        "seaviewsurvey_reefcover_atlantic.csv",
        "quadrat",
        "Western Atlantic quadrat-level proportional cover.",
        "quadratid",
        "surveyid, imageid → surveys / quadrats",
        "lat",
        "lng",
    ),
    Table(
        "seaviewsurvey_reefcover_southeastasia.csv",
        "quadrat",
        "Southeast Asia quadrat-level proportional cover.",
        "quadratid",
        "surveyid, imageid → surveys / quadrats",
        "lat",
        "lng",
    ),
    Table(
        "seaviewsurvey_reefcover_pacificaustralia.csv",
        "quadrat",
        "Eastern Australia (Great Barrier Reef) quadrat-level proportional cover.",
        "quadratid",
        "surveyid, imageid → surveys / quadrats",
        "lat",
        "lng",
    ),
    Table(
        "seaviewsurvey_reefcover_pacifichawaii.csv",
        "quadrat",
        "Central Pacific (Hawaii) quadrat-level proportional cover.",
        "quadratid",
        "surveyid, imageid → surveys / quadrats",
        "lat",
        "lng",
    ),
) + tuple(
    Table(
        f"annotations_{code}.csv",
        "annotation point (human)",
        f"Human expert point annotations for {label}, used to train and test the "
        f"classifiers. Carries method (random/target) and data_set (train/test).",
        "(quadratid, y, x)",
        "quadratid → quadrats.quadratid",
        None,
        None,
    )
    for code, label in (
        ("ATL", "the Western Atlantic"),
        ("IND_CHA", "the Chagos Archipelago"),
        ("IND_MDV", "the Maldives"),
        ("PAC_AUS", "eastern Australia"),
        ("PAC_IDN_PHL", "Indonesia and the Philippines"),
        ("PAC_SLB", "the Solomon Islands"),
        ("PAC_TLS", "Timor-Leste"),
        ("PAC_TWN", "Taiwan"),
        ("PAC_USA", "Hawaii"),
    )
)


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------


def sha256_of(path: Path, *, chunk: int = 1 << 20) -> str:
    """Return the lowercase hex SHA-256 of *path*, streamed."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _guard_remote_path(url: str) -> None:
    """Refuse any URL that reaches into the image archive."""
    for forbidden in FORBIDDEN_REMOTE_PATHS:
        if forbidden in url:
            raise SystemExit(
                f"Refusing to request {url!r}: {forbidden!r} is part of the photo-quadrat "
                f"image archive. This script acquires tabular data only; the imagery is "
                f"deliberately out of scope (see the module docstring)."
            )


def fetch_file(url: str, dest: Path, *, timeout: int = 1800) -> None:
    """Download *url* to *dest*, atomically."""
    _guard_remote_path(url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "coralsense-mlops/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out, length=1 << 20)
    except urllib.error.URLError as exc:  # pragma: no cover - network path
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"Download failed for {url}: {exc}") from exc
    tmp.replace(dest)


def extract_tables(archive: Path, dest: Path) -> list[str]:
    """Extract every CSV member of *archive* into *dest*, flattening one level."""
    dest.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            if not member.endswith(".csv"):
                continue
            name = Path(member).name
            with zf.open(member) as src, (dest / name).open("wb") as out:
                shutil.copyfileobj(src, out, length=1 << 20)
            written.append(name)
    return sorted(written)


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


def inspect_table(path: Path, table: Table) -> dict:
    """Return row/column counts and, where present, the coordinate envelope."""
    csv.field_size_limit(10_000_000)
    lat_min = lng_min = float("inf")
    lat_max = lng_max = float("-inf")
    rows = 0
    blank_cells = 0

    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        lat_i = header.index(table.lat_col) if table.lat_col in header else None
        lng_i = header.index(table.lng_col) if table.lng_col in header else None
        for row in reader:
            rows += 1
            blank_cells += sum(1 for cell in row if cell == "")
            if lat_i is not None and lng_i is not None:
                try:
                    lat, lng = float(row[lat_i]), float(row[lng_i])
                except (ValueError, IndexError):
                    continue
                lat_min, lat_max = min(lat_min, lat), max(lat_max, lat)
                lng_min, lng_max = min(lng_min, lng), max(lng_max, lng)

    envelope = None
    if lat_min != float("inf"):
        envelope = (
            round(lat_min, 6),
            round(lat_max, 6),
            round(lng_min, 6),
            round(lng_max, 6),
        )
    return {
        "n_rows": rows,
        "n_columns": len(header),
        "columns": header,
        "blank_cells": blank_cells,
        "envelope": envelope,
    }


# ---------------------------------------------------------------------------
# Provenance records
# ---------------------------------------------------------------------------


def build_source_record(retrieved_at: str) -> SourceRecord:
    """Build the product-level provenance record."""
    return SourceRecord(
        dataset_id=DATASET_ID,
        source_name="University of Queensland",
        product_name="Seaview Survey Photo-quadrat and Image Classification Dataset",
        source_url=BASE_URL,
        version="2019 release; tabular-data.zip published 2019-12-09",
        doi=DOI,
        product_identifier="UQ eSpace UQ:734799",
        licence_name=NORMALIZED_LICENCE,
        licence_url=LICENCE_URL,
        citation=(
            "González-Rivero, M., Rodriguez-Ramirez, A., Beijbom, O., Dalton, P., "
            "Kennedy, E. V., Neal, B. P., Vercelloni, J., Bongaerts, P., Ganase, A., "
            "Bryant, D. E. P., et al. (2019). Seaview Survey Photo-quadrat and Image "
            "Classification Dataset. The University of Queensland. "
            "https://doi.org/10.14264/uql.2019.930 — described by "
            "Rodriguez-Ramirez, A., González-Rivero, M., Beijbom, O., et al. (2020). "
            "A contemporary baseline record of the world's coral reefs. "
            "Scientific Data 7, 355. https://doi.org/10.1038/s41597-020-00698-6"
        ),
        licence_verified=True,
        licence_verified_via=(
            "Read from the authoritative dataset record in UQ eSpace (UQ:734799), whose "
            f"licence field reads verbatim '{SOURCE_REPORTED_LICENCE_LABEL}' with access "
            "conditions 'Open Access', and which supplies the licence URI "
            f"{LICENCE_URL}. Retrieved 2026-08-25 from "
            "the eSpace record API. Cross-checked against the peer-reviewed data descriptor "
            "(Sci Data 7, 355), whose Usage Notes state the dataset is released under a "
            "'Creative Commons Attribution license (CC BY 3.0)'. The two agree.\n\n"
            "NOT established by: the Crossref DOI metadata for 10.14264/uql.2019.930, which "
            "registers title, publisher, year and creators but carries no licence field at "
            "all. A DOI landing page resolving successfully is not a licence statement.\n\n"
            "Label vs URI: the publisher's textual label is imprecise. CC BY 3.0 was issued "
            "in 'Unported' and in ported national forms; there is no CC BY '3.0 "
            "International' — that wording only begins at version 4.0, which was not "
            "granted here. The supplied licence URI is unambiguous and governs: "
            f"{LICENCE_URL} is Creative Commons Attribution 3.0 Unported. This record "
            f"therefore normalizes to '{NORMALIZED_LICENCE}' while preserving the "
            "publisher's own wording verbatim in the licence_normalization block. Both "
            "readings grant the rights this project relies on (redistribution, adaptation, "
            "commercial use, subject to attribution), so the normalization changes the "
            "wording and not the decision."
        ),
        redistribution_allowed=True,
        raw_tracked_in_git=False,
        is_synthetic=False,
        observation_type="derived",
        processing_level="L2",
        original_format="CSV (18 tables) distributed as a single ZIP archive",
        original_crs="EPSG:4326",
        spatial_resolution=(
            "~1 m² photo-quadrat; quadrats lie along transects of 1.5–2.0 km at a "
            "standard depth of 10 m (±2 m)"
        ),
        temporal_resolution=(
            "Irregular. One survey per transect visit; 26 Indian Ocean transects were "
            "visited twice, the rest once."
        ),
        geographic_scope=(
            "860 transects across five reef regions: Western Atlantic, Eastern Australia, "
            "Central Indian Ocean, Southeast Asia, Central Pacific. The Central Indian "
            "Ocean component is the Maldives and the Chagos Archipelago ONLY — it contains "
            "no Indian territorial waters. See geographic_transfer_status."
        ),
        temporal_scope=(
            "2012-09-16 to 2018-05-05 globally; the Central Indian Ocean surveys run "
            "2015-02-12 to 2017-04-01."
        ),
        retrieved_at_utc=retrieved_at,
        access_method=(
            "Direct HTTP GET of tabular-data.zip and the dataset documentation PDF from "
            "the publisher's open directory at data.qld.edu.au/public/Q1281, linked from "
            "the UQ eSpace record. No authentication, no access control, no scraping. "
            "The photo-quadrat, annotated-image and survey-preview subtrees were not "
            "requested."
        ),
        variable_units={
            "hard_coral_cover": "proportion (0-1)",
            "soft_coral_cover": "proportion (0-1)",
            "algae_cover": "proportion (0-1)",
            "other_invertebrates_cover": "proportion (0-1)",
            "other_substrate_cover": "proportion (0-1)",
        },
        provides_variables=(
            "hard_coral_cover",
            "soft_coral_cover",
            "algae_cover",
            "other_invertebrates_cover",
            "other_substrate_cover",
            "benthic_label_cover",
            "survey_date",
            "survey_coordinates",
        ),
        cannot_provide=(
            "reef_health",
            "restoration_suitability",
            "bleaching_percentage",
            "disease_percentage",
            "water_temperature",
            "depth_m",
            "any observation inside Indian territorial waters",
        ),
        disclaimer=(
            "Benthic cover here is an IMAGE-DERIVED BENTHIC-COVER ESTIMATE produced by an "
            "image classifier, not a direct field observation and not a field-measured "
            "quantity. 135,400 of the 137,698 Indian Ocean quadrats "
            "(98.33%) were labelled by a VGG-D 16 convolutional neural network at 50 "
            "points per quadrat; only 2,298 (1.67%) carry human expert annotations — and "
            "those annotations are themselves readings of a photograph, not in-water "
            "measurements. The published validation reports 97% agreement with human "
            "observers and errors of <2%–7%, which is good; validation accuracy is a "
            "property of the classifier and does not convert its predictions into "
            "measured coral cover or biological ground truth.\n\n"
            "hard_coral_cover IS NOT reef_health. A benthic class IS NOT "
            "restoration_suitability. Neither project target exists in this dataset and "
            "neither may be manufactured from it."
        ),
        notes=(
            "observation_type is 'derived', not 'measured': the photograph is the "
            "measurement, the cover proportion is a model output computed from it. "
            "processing_level is L2 for the same reason — a derived variable at the "
            "resolution of the source observation, not a gridded (L3) or physically "
            "modelled (L4) field.\n\n"
            "The publisher serves this collection over plain HTTP; data.qld.edu.au does "
            "not answer on port 443. The transfer is therefore unauthenticated, which is "
            "why every acquired file carries a pinned SHA-256 below."
        ),
    )


def build_subset_record(table: Table, path: Path, stats: dict, retrieved_at: str) -> SubsetRecord:
    """Build one provenance record for one acquired CSV table."""
    envelope = stats["envelope"]
    if envelope is None:
        bbox = (0.0, 0.0, 0.0, 0.0)
        rationale = (
            "This table carries no coordinate columns; it joins to geography through "
            "surveyid/quadratid. The zero bbox is 'not applicable', not an extent."
        )
    else:
        bbox = envelope
        rationale = (
            "Envelope of the coordinates actually present in this table, computed from "
            "the delivered rows. Not a requested window — nothing was subset server-side."
        )

    return SubsetRecord(
        dataset_id=DATASET_ID,
        region=table.filename,
        local_file=str(path.relative_to(PROJECT_ROOT).as_posix()),
        file_format="CSV",
        file_size_bytes=path.stat().st_size,
        sha256=sha256_of(path),
        retrieved_at_utc=retrieved_at,
        request_url=f"{BASE_URL}/{ARCHIVE_NAME}",
        actual_bbox=bbox,
        requested_bbox=(0.0, 0.0, 0.0, 0.0),
        dimensions={"rows": stats["n_rows"], "columns": stats["n_columns"]},
        bbox_rationale=rationale,
        variable_name=table.unit,
        variable_long_name=table.description,
        is_synthetic=False,
    )


def write_manifest(source: SourceRecord, subsets: list[SubsetRecord], extra: dict) -> None:
    """Validate and write the manifest JSON."""
    validate_manifest(source, subsets, project_root=PROJECT_ROOT, require_files=False)
    payload = {
        "schema": "coralsense.external.manifest/v1",
        "dataset_id": DATASET_ID,
        "source": source.to_dict(),
        "subsets": [s.to_dict() for s in subsets],
        **extra,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _plan() -> list[tuple[str, Path]]:
    return [
        (f"{BASE_URL}/{ARCHIVE_NAME}", RAW_DIR / ARCHIVE_NAME),
        (f"{BASE_URL}/{DOC_NAME}", RAW_DIR / DOC_NAME),
    ]


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--dry-run", action="store_true", help="Print the plan; fetch nothing.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Re-inspect files already on disk and rewrite the manifest. No network.",
    )
    args = parser.parse_args()

    if args.dry_run:
        print(f"Would acquire into {RAW_DIR.relative_to(PROJECT_ROOT)}/ :")
        for url, dest in _plan():
            print(f"  {url}\n    -> {dest.relative_to(PROJECT_ROOT)}")
        print("\nWould NOT request (image archive, deliberately out of scope):")
        for forbidden in FORBIDDEN_REMOTE_PATHS:
            print(f"  {BASE_URL}/{forbidden}")
        return 0

    if not args.validate_only:
        for url, dest in _plan():
            print(f"fetching {url}")
            fetch_file(url, dest)
        print(f"extracting {ARCHIVE_NAME}")
        extract_tables(RAW_DIR / ARCHIVE_NAME, TABLE_DIR)

    archive = RAW_DIR / ARCHIVE_NAME
    if not archive.is_file():
        raise SystemExit(f"{archive} is missing. Run without --validate-only to acquire it first.")

    retrieved_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    if args.validate_only:
        existing = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.is_file() else {}
        retrieved_at = existing.get("source", {}).get("retrieved_at_utc", retrieved_at)

    subsets: list[SubsetRecord] = []
    diagnostics: dict[str, dict] = {}
    for table in TABLES:
        path = TABLE_DIR / table.filename
        if not path.is_file():
            raise SystemExit(f"expected table missing: {path}")
        stats = inspect_table(path, table)
        subsets.append(build_subset_record(table, path, stats, retrieved_at))
        diagnostics[table.filename] = {
            "observational_unit": table.unit,
            "primary_key": table.primary_key,
            "foreign_keys": table.foreign_keys,
            "n_rows": stats["n_rows"],
            "n_columns": stats["n_columns"],
            "blank_cells": stats["blank_cells"],
            "coordinate_envelope": stats["envelope"],
            "description": table.description,
        }
        print(f"  {table.filename}: {stats['n_rows']:,} rows x {stats['n_columns']} cols")

    extra = {
        "acquired_archive": {
            "filename": ARCHIVE_NAME,
            "url": f"{BASE_URL}/{ARCHIVE_NAME}",
            "local_file": str((RAW_DIR / ARCHIVE_NAME).relative_to(PROJECT_ROOT).as_posix()),
            "file_size_bytes": archive.stat().st_size,
            "sha256": sha256_of(archive),
            "publisher_last_modified": "2019-12-09T23:08:38Z",
        },
        "acquired_documentation": _documentation_block(),
        "not_acquired": {
            "reason": (
                "This project needs benthic cover values, not the imagery they were "
                "derived from. The image archive is orders of magnitude larger than the "
                "tables and adds nothing the tables do not already carry."
            ),
            "paths": list(FORBIDDEN_REMOTE_PATHS),
        },
        "checksum_semantics": _checksum_semantics(),
        "licence_normalization": _licence_normalization(),
        "geographic_transfer_status": _geographic_transfer_status(),
        "temporal_design": _temporal_design(),
        "observational_units": _observational_units(),
        "variable_semantics": _variable_semantics(),
        "target_compatibility": _target_compatibility(),
        "known_data_defects": _known_data_defects(),
        "diagnostics": {"per_table": diagnostics},
    }

    source = build_source_record(retrieved_at)
    write_manifest(source, subsets, extra)
    print(f"\nwrote {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    return 0


def _documentation_block() -> dict:
    doc = RAW_DIR / DOC_NAME
    block = {
        "filename": DOC_NAME,
        "url": f"{BASE_URL}/{DOC_NAME}",
        "role": (
            "The publisher's own dataset documentation: the authoritative description of "
            "the table schemas, label sets and directory layout."
        ),
    }
    if doc.is_file():
        block["local_file"] = str(doc.relative_to(PROJECT_ROOT).as_posix())
        block["file_size_bytes"] = doc.stat().st_size
        block["sha256"] = sha256_of(doc)
    return block


def _checksum_semantics() -> dict:
    return {
        "algorithm": "SHA-256",
        "scope": (
            "Hash of each file as retrieved and extracted by this project, computed "
            "locally at acquisition time."
        ),
        "is_publisher_canonical_checksum": False,
        "guarantees": "The local file has not been altered since acquisition.",
        "does_not_guarantee": (
            "That the publisher will serve identical bytes in future. Unlike the NOAA "
            "ERDDAP case these are static files rather than server-generated "
            "serialisations, so a re-fetch SHOULD reproduce these hashes exactly — but "
            "the publisher may reissue the archive, and a mismatch would mean 'the "
            "published file changed', not 'our copy is corrupt'. The archive's own "
            "Last-Modified is recorded above so the two can be told apart."
        ),
        "authoritative_identity": (
            f"Scientific identity is pinned by DOI {DOI} and the data descriptor "
            f"{PAPER_DOI}, not by these file hashes."
        ),
        "transport_caveat": (
            "The publisher serves this collection over plain HTTP only, so the transfer "
            "was not authenticated in transit. These hashes are the integrity record."
        ),
    }


def _licence_normalization() -> dict:
    """
    Publisher wording and our reading of it, kept as two separate facts.

    The UQ record labels the licence 'Creative Commons Attribution 3.0
    International' and points at ``creativecommons.org/licenses/by/3.0/``.  No
    such licence as CC BY '3.0 International' exists — 3.0 shipped as Unported
    plus ported national forms — but the URI it supplies is unambiguous.  So the
    label is normalized and the original is preserved beside it, rather than one
    being overwritten by the other.
    """
    return {
        "source_reported_licence_label": SOURCE_REPORTED_LICENCE_LABEL,
        "source_reported_by": "UQ eSpace record UQ:734799, licence field, read 2026-08-25",
        "normalized_licence": NORMALIZED_LICENCE,
        "licence_url": LICENCE_URL,
        "canonical_licence_uri_resolves_to": "Creative Commons Attribution 3.0 Unported",
        "normalization_basis": (
            "The publisher's textual label is imprecise: Creative Commons issued version "
            "3.0 as 'Unported' and in ported national forms, and the 'International' "
            "wording only begins at version 4.0, which is not what this record grants. "
            "The licence URI the same record supplies is unambiguous, so the URI governs "
            "and the label is read as a naming error rather than as a different licence."
        ),
        "version_granted": "3.0",
        "version_not_granted": (
            "4.0. Nothing in the eSpace record, the data descriptor or the licence URI "
            "grants version 4.0, and this dataset must not be described under it."
        ),
        "publisher_wording_preserved": True,
        "licence_verified": True,
        "redistribution_allowed": True,
        "effect_of_the_discrepancy": (
            "None on the rights relied on here. Unported and any ported 3.0 form both "
            "permit redistribution, adaptation and commercial use subject to attribution. "
            "The discrepancy is recorded because provenance should show what the publisher "
            "said as well as what we concluded, not because it changes the permission."
        ),
    }


def _geographic_transfer_status() -> dict:
    return {
        "status": GEOGRAPHIC_TRANSFER_STATUS,
        "headline": "INDIAN OCEAN IS NOT INDIA.",
        "ocean_code_warning": (
            "The 'ocean' column uses IND for the Indian Ocean basin. IND is also the "
            "ISO 3166-1 alpha-3 code for India, and the human-annotation files are named "
            "annotations_IND_MDV.csv and annotations_IND_CHA.csv. Nothing in this dataset "
            "refers to India. Verified: no row of seaviewsurvey_surveys.csv has "
            "country == 'IND'."
        ),
        "territories_surveyed": INDIAN_OCEAN_TERRITORIES,
        "surveys_by_territory": {"MDV": 63, "CHA": 29},
        "coordinate_envelope": {
            "lat_min": -6.699465,
            "lat_max": 4.522577,
            "lng_min": 71.234587,
            "lng_max": 73.588981,
        },
        "project_target_regions_covered": [],
        "project_target_regions_not_covered": {
            "Lakshadweep": "0 surveys. Nearest Seaview transect ~386 km south.",
            "Gulf of Mannar": "0 surveys. Nearest Seaview transect ~654 km away.",
            "Gulf of Kutch": "0 surveys. Nearest Seaview transect ~1,942 km away.",
            "Andaman and Nicobar Islands": ("0 surveys. Nearest Seaview transect ~2,046 km away."),
        },
        "consequence": (
            "This dataset does NOT validate any model for Lakshadweep, the Gulf of Mannar, "
            "the Gulf of Kutch, or the Andaman and Nicobar Islands. Fitting on Maldives and "
            "Chagos observations and applying the result to Indian reefs is a geographic "
            "transfer across a minimum 386 km gap and a different reef province. It may "
            "still be scientifically useful, but it must be reported as transfer, never as "
            "Indian validation."
        ),
    }


def _temporal_design() -> dict:
    """
    What the 26 twice-visited Maldivian transects are, and what they are not.

    They are a paired pre/post contrast around the 2016 mass-bleaching period.
    They are not a measured bleaching response: this dataset records no bleaching
    observation at all, and a difference in cover between two visits is a
    difference in cover, whatever caused it.
    """
    return {
        "repeat_transects_maldives": 26,
        "repeat_transects_chagos": 0,
        "distinct_indian_ocean_transects": 66,
        "visited_once": 40,
        "visited_twice": 26,
        "visited_three_or_more_times": 0,
        "revisit_interval_days": {"min": 703, "max": 722},
        "epochs": {
            "MDV": "two — 2015-03-29 to 2015-04 and 2017-03 to 2017-04-01",
            "CHA": "one — 2015-02-12 to 2015-02-24, so no paired comparison is possible",
        },
        "what_this_supports": (
            "A paired pre/post survey change around the 2016 mass-bleaching period, on 26 "
            "Maldivian transects, approximately 703-722 days apart. It is a real temporal "
            "contrast between two image-derived benthic-cover estimates at the same "
            "transect."
        ),
        "what_this_is_not": (
            "NOT a bleaching response and NOT a bleaching measurement. This dataset "
            "contains no bleaching observation, no thermal covariate and no experimental "
            "control. A change in image-derived cover between two visits does not by "
            "itself demonstrate that bleaching caused the change: other explanations "
            "(storm damage, disease, crown-of-thorns predation, sampling and classifier "
            "variation, transect re-navigation) are not excluded by the design."
        ),
        "thermal_exposure_not_linked": (
            "NOAA CRW HotSpot and Degree Heating Week exposure should eventually be linked "
            "to these transects by site and date, from the CRW product already acquired in "
            "this repository. That linkage has NOT been performed — no join exists between "
            "Seaview and CRW."
        ),
        "causal_status_after_linkage": (
            "OBSERVATIONAL — ASSOCIATION ONLY. Even once CRW exposure is attached by site "
            "and date, the design remains an observational before/after comparison without "
            "controls or randomisation. It can support an association between thermal "
            "exposure and cover change; it does not license automatic causal attribution."
        ),
        "required_wording": (
            "Describe this as 'paired pre/post survey change around the 2016 "
            "mass-bleaching period', never as 'bleaching response'."
        ),
    }


def _observational_units() -> dict:
    return {
        "hierarchy": "survey -> image -> quadrat -> annotation point",
        "units": {
            "survey": (
                "One transect visit, 1.5-2.0 km, at ~10 m depth. 5-digit surveyid. "
                "860 globally, 92 in the Central Indian Ocean."
            ),
            "image": (
                "One raw photograph. 9-digit imageid = surveyid + 4-digit frame number. "
                "81,773 in the Indian Ocean subset."
            ),
            "quadrat": (
                "One standardised ~1 m² crop of a raw image. 11-digit quadratid = "
                "imageid + 2-digit quadrat number. 1,082,324 globally, 137,698 in the "
                "Indian Ocean subset."
            ),
            "annotation_point": (
                "One classified pixel location. 50 points per quadrat for the automated "
                "classifier; 50-300 for human-annotated training/test quadrats."
            ),
        },
        "cardinality_indian_ocean": {
            "images_per_survey": "mean 889, range 470-1137",
            "quadrats_per_survey": "mean 1497, range 646-2742",
            "quadrats_per_image": {
                "1": 31292,
                "2": 49156,
                "4": 1,
                "5": 3,
                "6": 1306,
                "15": 1,
                "16": 14,
            },
        },
        "independence_warning": (
            "Multiple quadrats cropped from ONE image are not independent geographic "
            "locations — they are neighbouring patches of one photograph, sharing its "
            "position, altitude, exposure and moment in time. 49,156 Indian Ocean images "
            "yield 2 quadrats each and 1,306 yield 6. Treating 137,698 quadrats as "
            "137,698 independent sites would overstate the sample by roughly an order of "
            "magnitude: the real spatial sample is 92 transects."
        ),
        "aggregation_rule": (
            "The rule stated by the data descriptor (Sci Data 7, 355): (1) per quadrat, "
            "proportional cover = classified points for a label / total points; (2) "
            "average quadrat cover within each image; (3) average image cover within each "
            "survey; (4) merge detailed labels into the five functional groups by summing "
            "within group, using seaviewsurvey_labelsets.csv. Reproducing this from the "
            "quadrat table recovers the published survey-level pr_hard_coral to a mean "
            "absolute difference of 0.0025 (max 0.040, 1 survey of 92 above 0.02)."
        ),
    }


def _variable_semantics() -> dict:
    return {
        "warning": (
            "These are IMAGE-DERIVED BENTHIC-COVER ESTIMATES, and they are the first real "
            "biological variables in this repository. They are neither field-measured "
            "cover nor reef-condition labels. hard_coral_cover != reef_health. "
            "algal_cover != reef_health. A benthic class != restoration_suitability."
        ),
        "preferred_terminology": {
            "use": [
                "image-derived benthic-cover estimate",
                "ML-estimated benthic cover",
                "image-derived biological response estimate",
            ],
            "do_not_use": [
                "directly observed coral cover",
                "field-measured coral cover",
                "measured coral cover",
                "biological ground truth",
            ],
            "why": (
                "The imagery and coordinates are real field records; the cover values are a "
                "classifier's reading of them. Real imagery does not make CNN output a "
                "direct field measurement, and the published validation accuracy does not "
                "rename a prediction as ground truth."
            ),
        },
        "evidence_layers": {
            "A_field_survey_imagery_and_coordinates": (
                "REAL FIELD RECORD. Diver-propelled cameras photographed real reef along "
                "real transects. The photographs, the survey dates, the transect "
                "coordinates and the ~10 m survey depth are direct field observations, and "
                "nothing here disputes that."
            ),
            "B_human_image_annotations": (
                "HUMAN ANNOTATION OF IMAGERY. Expert annotators scored points on a small "
                "training/test subset — 2,298 of 137,698 Indian Ocean quadrats (1.67%), "
                "held in the annotations_*.csv tables. This is an expert judgement about a "
                "photograph, which is the strongest biological evidence in the product, but "
                "it is still a reading of an image rather than an in-water measurement."
            ),
            "C_ml_classified_benthic_cover": (
                "MODEL OUTPUT. The cover columns of the reefcover tables and the "
                "survey-level pr_* columns are ML-estimated benthic cover: VGG-D 16 "
                "classifications of 50 points per quadrat, covering 98.33% of Indian Ocean "
                "quadrats. These are the values this project would actually use, and they "
                "are estimates, not measurements."
            ),
            "why_the_distinction_matters": (
                "A is real, B is a human reading of A, and C is a model's reading of A. "
                "Collapsing the three — 'we used real observed coral cover' — is not "
                "accurate: it overstates the evidence by two steps. The correct "
                "description of layer C is an image-derived benthic-cover estimate."
            ),
        },
        "how_values_were_produced": {
            "method": (
                "Deep learning point classification. Nine VGG-D 16 convolutional neural "
                "networks, one per country/region (the Western Atlantic shares one), "
                "fine-tuned for ~40,000 iterations. Each network classifies 50 points per "
                "quadrat; cover proportions are point counts divided by total points."
            ),
            "human_share_indian_ocean": {
                "quadrats_with_human_annotations": 2298,
                "quadrats_total": 137698,
                "human_percent": 1.67,
                "machine_percent": 98.33,
            },
            "published_validation": (
                "97% agreement between human and automated annotations; errors of the "
                "automated estimates between <2% and 7%; R² = 0.97 (P < 0.001) between "
                "automated and observer estimates; critical difference ~4% across labels "
                "and regions. Mean absolute error by group ranged from <1% (Other "
                "Invertebrates) to 5% (Algae, Southeast Asia). Preserved as published; it "
                "is a property of the classifier, not a licence to rename its predictions."
            ),
            "validation_does_not_promote_predictions": (
                "A well-validated estimate is still an estimate. These figures justify "
                "calling the values a good image-derived benthic-cover estimate; they do "
                "not justify calling them measured cover, directly observed cover or "
                "biological ground truth."
            ),
            "required_qualification": (
                "ML-estimated benthic cover must not be called ground truth, and must not "
                "be called directly observed, field-measured or measured coral cover. It is "
                "a classifier output validated against human image annotation. The "
                "human-annotated subset is the only part of this product carrying a direct "
                "expert judgement, and even that judgement is made on a photograph."
            ),
        },
        "indian_ocean_functional_groups": {
            "Hard Coral": 19,
            "Other Invertebrates": 10,
            "Algae": 9,
            "Other": 4,
            "Soft Coral": 3,
        },
        "survey_level_columns": {
            "pr_hard_coral": "proportion of points classified as any hard coral label",
            "pr_algae": "proportion classified as any algal label",
            "pr_soft_coral": "proportion classified as any soft coral label",
            "pr_oth_invert": "proportion classified as any other-invertebrate label",
            "pr_other": "proportion classified as any other/substrate label",
        },
    }


def _target_compatibility() -> dict:
    return {
        "reef_health": {
            "verdict": "PARTIAL BIOLOGICAL EVIDENCE ONLY — NOT A TARGET",
            "reasoning": (
                "This dataset supplies image-derived benthic-composition estimates, which "
                "are genuine biological evidence and a real advance on thermal covariates. "
                "It does not supply the project's reef_health class. reef_health in the "
                "synthetic pipeline is a constructed condition judgement; image-derived "
                "hard-coral cover is an estimate of one component of reef state. "
                "Thresholding cover into a health class "
                "would rebuild exactly the label-construction leakage the 2026-08-19 audit "
                "found, this time out of real numbers, which makes it harder to spot rather "
                "than safer."
            ),
        },
        "restoration_suitability": {
            "verdict": "NO DIRECT TARGET",
            "reasoning": (
                "Nothing in this dataset expresses restoration suitability. It contains no "
                "intervention, no restoration outcome, no site-selection judgement and no "
                "management variable. A benthic class is not a suitability score and must "
                "not be mapped onto one."
            ),
        },
        "legitimate_use": (
            "Survey-level hard-coral cover is an IMAGE-DERIVED CONTINUOUS BIOLOGICAL "
            "RESPONSE ESTIMATE and may be used as the response variable in a separate "
            "real-data analysis, described as such. That is a different modelling problem "
            "from the registered synthetic classifiers, and it does not change, retrain or "
            "supersede them."
        ),
        "recommended_response_field": {
            "field": "pr_hard_coral",
            "table": "seaviewsurvey_surveys.csv",
            "unit": "survey (transect visit)",
            "why": (
                "For the FIRST biological-response analysis, use the publisher-provided "
                "survey-level pr_hard_coral rather than reconstructing hard-coral cover "
                "from the quadrat label columns of "
                "seaviewsurvey_reefcover_indianocean.csv, which are known to be defective "
                "(see known_data_defects.missing_label_column_indian_ocean). The "
                "survey-level values do not show that shortfall, and the survey is also "
                "the honest spatial unit — 92 transects, not 137,698 pseudo-independent "
                "quadrats."
            ),
            "status": "RECOMMENDATION ONLY — no such analysis has been performed.",
        },
        "prohibited": [
            "Deriving reef_health from any cover variable.",
            "Deriving restoration_suitability from any benthic class.",
            "Joining these rows to data/raw/observations.csv.",
            "Describing Maldives or Chagos observations as Indian validation.",
            "Describing ML-classified benthic cover as directly observed, field-measured, "
            "measured coral cover or biological ground truth.",
            "Describing the 26 repeat-transect pairs as a measured bleaching response, or "
            "otherwise implying causality from the paired design alone.",
        ],
    }


def _known_data_defects() -> dict:
    return {
        "missing_label_column_indian_ocean": {
            "what": (
                "seaviewsurvey_reefcover_indianocean.csv has no MASE_MEA_L column, although "
                "MASE_MEA_L (Lobophyllia, functional group Hard Coral) IS defined for the "
                "Indian Ocean in seaviewsurvey_labelsets.csv and DOES appear in "
                "seaviewsurvey_annotations.csv. The same table instead carries MASE_LRG_I "
                "(Isopora), a label defined only for Southeast Asia, which is all zeros "
                "across all 137,698 rows."
            ),
            "evidence": (
                "All 135,185 Indian Ocean quadrats with no MASE_MEA_L annotation sum to "
                "exactly 1.00 across label columns. All 2,513 quadrats that DO have "
                "MASE_MEA_L points sum to less than 1.00 (mean 0.963, min 0.44). The "
                "correspondence is exact in both directions — every short-summing row is a "
                "row with dropped Lobophyllia points, and no other row is short."
            ),
            "impact": (
                "Quadrat-level hard coral cover is UNDER-REPORTED for 2,513 of 137,698 "
                "Indian Ocean quadrats (1.83%), by 4,669 classified points in total; the "
                "worst affected quadrat loses 56% of its cover. The survey-level "
                "pr_hard_coral values in seaviewsurvey_surveys.csv do NOT show this "
                "shortfall and appear to include the dropped label."
            ),
            "handling": (
                "Recorded, not repaired. The CSV has not been edited, nothing has been "
                "imputed, dropped or rescaled, and no row has been removed. The raw bytes "
                "are unchanged and their SHA-256 is pinned above. For survey-level hard "
                "coral cover, prefer seaviewsurvey_surveys.csv. For "
                "quadrat-level work, either accept the documented 1.83% under-count or "
                "recover MASE_MEA_L points from seaviewsurvey_annotations.csv."
            ),
            "recommendation_for_first_analysis": (
                "For the FIRST future biological-response analysis, use the "
                "publisher-provided survey-level pr_hard_coral from "
                "seaviewsurvey_surveys.csv rather than reconstructing hard-coral cover "
                "from these known-defective quadrat label columns. Not performed here."
            ),
        },
        "no_outliers_removed": (
            "No outlier removal, filtering or cleaning has been applied to any table. "
            "Every delivered row is present as published."
        ),
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
