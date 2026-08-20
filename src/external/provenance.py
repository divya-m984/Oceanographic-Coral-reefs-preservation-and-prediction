"""
src/external/provenance.py — provenance schema and licence gate for external data.

Scope
-----
This module governs *real* external datasets only.  It is never applied to the
synthetic prototype pipeline in ``src/data``; that dataset is frozen and is
described by ``docs/audits/dataset_scientific_audit_2026-08-19.md``.

Two record types
----------------
``SourceRecord``
    One record per acquired external **product** — the dataset as a whole:
    who published it, under what licence, at what version, and whether its raw
    files may be redistributed.

``SubsetRecord``
    One record per acquired **subset** of that product — for a gridded global
    product this is one regional extraction.  Each subset carries the checksum
    and geometry of exactly one local file.

There is deliberately no per-row provenance at this stage.  Copying the same
twenty fields onto millions of raster cells would be storage without
information; the subset record is the finest unit that currently differs.

The licence gate
----------------
``validate_source`` refuses any external product that has not had its licence
positively verified against the publisher's own terms, and that has not made an
explicit yes/no statement about redistribution.  ``licence_verified`` defaults
to ``False``: a product is untrusted until someone has read the terms.

``redistribution_allowed=False`` does not block *use*.  It blocks
*publication*: the metadata may describe the source, but the raw files must not
be committed or redistributed.  ``assert_publishable`` expresses that check for
callers that are about to export or commit data.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------

ObservationType = Literal["measured", "modelled", "remotely_sensed", "derived", "compiled"]
"""How the values came to exist.  ``compiled`` covers products such as GEBCO
that fuse measured soundings with interpolated and modelled estimates."""

ProcessingLevel = Literal["L0", "L1", "L2", "L3", "L4"]
"""L0 raw · L1 calibrated · L2 derived geophysical · L3 gridded · L4 modelled."""

VALID_OBSERVATION_TYPES: tuple[str, ...] = (
    "measured",
    "modelled",
    "remotely_sensed",
    "derived",
    "compiled",
)

VALID_PROCESSING_LEVELS: tuple[str, ...] = ("L0", "L1", "L2", "L3", "L4")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ISO_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

#: Target columns of the synthetic prototype pipeline.  No external product may
#: claim to supply these; they are reef-condition judgements, not measurements.
#: Enforced by :func:`validate_source` so a future contributor cannot quietly
#: assert that a bathymetry or satellite product is reef-health ground truth.
FORBIDDEN_TARGET_CLAIMS: tuple[str, ...] = ("reef_health", "restoration_suitability")


class ProvenanceError(ValueError):
    """A provenance record is structurally invalid."""


class LicenceGateError(ProvenanceError):
    """A provenance record failed the licence gate."""


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceRecord:
    """
    Provenance for one externally acquired product.

    Every field is required except those with defaults.  ``licence_verified``
    defaults to ``False`` on purpose — an unverified licence must be an active
    decision to override, never an oversight.
    """

    # ── Identity ────────────────────────────────────────────────────────────
    dataset_id: str
    source_name: str
    product_name: str
    source_url: str
    version: str

    # ── Licence ─────────────────────────────────────────────────────────────
    licence_name: str
    licence_url: str
    citation: str

    # ── Scientific description ──────────────────────────────────────────────
    observation_type: ObservationType
    processing_level: ProcessingLevel
    original_format: str
    original_crs: str
    spatial_resolution: str
    temporal_resolution: str
    geographic_scope: str
    temporal_scope: str

    # ── Acquisition ─────────────────────────────────────────────────────────
    retrieved_at_utc: str
    access_method: str

    # ── Optional / defaulted ────────────────────────────────────────────────
    doi: str | None = None
    sensor_type: str | None = None
    vertical_crs: str | None = None
    licence_verified: bool = False
    licence_verified_via: str | None = None
    redistribution_allowed: bool | None = None
    raw_tracked_in_git: bool = False
    is_synthetic: bool = False
    #: Variables this product genuinely supplies to the project.
    provides_variables: tuple[str, ...] = ()
    #: Variables it explicitly cannot supply — recorded so the limitation
    #: travels with the data rather than living only in prose.
    cannot_provide: tuple[str, ...] = ()
    disclaimer: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-serialisable dict with tuples flattened to lists."""
        out = asdict(self)
        for key in ("provides_variables", "cannot_provide"):
            out[key] = list(out[key])
        return out


@dataclass(frozen=True)
class SubsetRecord:
    """
    Provenance for one acquired subset (one local file) of a product.

    ``local_file`` must be a repository-relative POSIX path.  Absolute paths are
    rejected: committed metadata must not leak a machine-specific home
    directory, and must resolve on any checkout.

    Checksum semantics
    ------------------
    ``sha256`` is the hash of **the file this project retrieved**, computed
    locally at acquisition time.  It is *not* a publisher-issued canonical
    checksum, and it must not be presented as one unless the publisher actually
    distributes that exact digest.

    What it does guarantee: the local file has not been altered since it was
    acquired.  What it does **not** guarantee: that re-requesting the same
    logical subset reproduces the same bytes.  Server-side subsetting services
    may legitimately return a byte-different serialisation — different library
    version, chunking, attribute ordering, or generation timestamp — while
    encoding identical grid values.  A hash mismatch after a re-fetch therefore
    means "these are different bytes", not necessarily "the data changed".
    Establishing value-level equivalence would need array-level comparison,
    which is deliberately out of scope here.
    """

    dataset_id: str
    region: str
    local_file: str
    file_format: str
    file_size_bytes: int
    sha256: str
    retrieved_at_utc: str
    request_url: str

    #: Requested window, ``[lat_min, lat_max, lon_min, lon_max]``.
    requested_bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    #: Window actually returned, snapped to the source grid.
    actual_bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    #: ``{"lat": n, "lon": m}``.
    dimensions: dict[str, int] = field(default_factory=dict)

    grid_spacing_deg: float | None = None
    elevation_units: str | None = None
    min_elevation_m: float | None = None
    max_elevation_m: float | None = None
    median_elevation_m: float | None = None
    marine_cell_count: int | None = None
    land_cell_count: int | None = None
    nodata_count: int | None = None
    bbox_rationale: str = ""
    is_synthetic: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict with tuples flattened to lists."""
        out = asdict(self)
        out["requested_bbox"] = list(out["requested_bbox"])
        out["actual_bbox"] = list(out["actual_bbox"])
        return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_source(record: SourceRecord, *, project_root: Path | None = None) -> None:
    """
    Validate a :class:`SourceRecord`, including the licence gate.

    Parameters
    ----------
    record:
        The record to check.
    project_root:
        Unused today; accepted so callers can pass a root without changing when
        path-resolving checks are added.

    Raises
    ------
    LicenceGateError
        ``licence_verified`` is not ``True``, or ``redistribution_allowed`` was
        left unstated.
    ProvenanceError
        Any other structural problem.
    """
    del project_root  # reserved

    for name in ("dataset_id", "source_name", "product_name", "source_url", "version"):
        if not getattr(record, name):
            raise ProvenanceError(f"{name!r} must be a non-empty string")

    if record.is_synthetic:
        raise ProvenanceError(
            "SourceRecord describes external data; is_synthetic must be False. "
            "The synthetic prototype dataset is not part of this layer."
        )

    if record.observation_type not in VALID_OBSERVATION_TYPES:
        raise ProvenanceError(
            f"observation_type {record.observation_type!r} not in {VALID_OBSERVATION_TYPES}"
        )
    if record.processing_level not in VALID_PROCESSING_LEVELS:
        raise ProvenanceError(
            f"processing_level {record.processing_level!r} not in {VALID_PROCESSING_LEVELS}"
        )

    if not _ISO_UTC_RE.match(record.retrieved_at_utc):
        raise ProvenanceError(
            f"retrieved_at_utc must be ISO-8601 UTC 'YYYY-MM-DDTHH:MM:SSZ'; "
            f"got {record.retrieved_at_utc!r}"
        )

    # ── The licence gate ────────────────────────────────────────────────────
    if not record.licence_verified:
        raise LicenceGateError(
            f"{record.dataset_id}: licence_verified is False. An external dataset may not be "
            f"used inside this repository until its licence has been read from the publisher's "
            f"own terms and licence_verified set to True."
        )
    if not record.licence_name:
        raise LicenceGateError(f"{record.dataset_id}: licence_verified is True but no licence_name")
    if not record.licence_verified_via:
        raise LicenceGateError(
            f"{record.dataset_id}: licence_verified is True but licence_verified_via is empty — "
            f"record where the terms were read."
        )
    if record.redistribution_allowed is None:
        raise LicenceGateError(
            f"{record.dataset_id}: redistribution_allowed must be explicitly True or False, "
            f"never left unstated."
        )

    # ── No external product may claim to supply a project target ────────────
    claimed = {v.lower() for v in record.provides_variables}
    forbidden = claimed.intersection(FORBIDDEN_TARGET_CLAIMS)
    if forbidden:
        raise ProvenanceError(
            f"{record.dataset_id}: provides_variables claims target(s) {sorted(forbidden)}. "
            f"No external dataset supplies reef-condition ground truth; see the dataset audit "
            f"of 2026-08-19, sections 18 and 12."
        )


def validate_subset(
    record: SubsetRecord,
    *,
    project_root: Path | None = None,
    require_file: bool = False,
) -> None:
    """
    Validate a :class:`SubsetRecord`.

    Parameters
    ----------
    record:
        The record to check.
    project_root:
        Repository root, used to resolve ``local_file`` when *require_file*.
    require_file:
        When ``True``, the referenced file must exist and its size must match
        ``file_size_bytes``.  Left ``False`` on checkouts where the raw
        (git-ignored) files are absent.
    """
    if not record.dataset_id or not record.region:
        raise ProvenanceError("dataset_id and region must be non-empty")

    if record.is_synthetic:
        raise ProvenanceError(f"{record.region}: SubsetRecord must have is_synthetic False")

    if not _SHA256_RE.match(record.sha256):
        raise ProvenanceError(
            f"{record.region}: sha256 must be 64 lowercase hex characters; got {record.sha256!r}"
        )

    if record.file_size_bytes <= 0:
        raise ProvenanceError(f"{record.region}: file_size_bytes must be positive")

    local = PurePosixPathLike(record.local_file)
    if local.is_absolute:
        raise ProvenanceError(
            f"{record.region}: local_file must be a repository-relative path, "
            f"got absolute {record.local_file!r}"
        )
    if ".." in local.parts:
        raise ProvenanceError(f"{record.region}: local_file must not escape the repository root")

    lat_min, lat_max, lon_min, lon_max = record.actual_bbox
    if not (-90.0 <= lat_min <= lat_max <= 90.0):
        raise ProvenanceError(
            f"{record.region}: actual_bbox latitudes invalid: {record.actual_bbox}"
        )
    if not (-180.0 <= lon_min <= lon_max <= 180.0):
        raise ProvenanceError(
            f"{record.region}: actual_bbox longitudes invalid: {record.actual_bbox}"
        )

    if require_file:
        if project_root is None:
            raise ProvenanceError("project_root is required when require_file=True")
        path = project_root / record.local_file
        if not path.is_file():
            raise ProvenanceError(f"{record.region}: local_file not found at {path}")
        actual = path.stat().st_size
        if actual != record.file_size_bytes:
            raise ProvenanceError(
                f"{record.region}: file_size_bytes {record.file_size_bytes} != on-disk {actual}"
            )


class PurePosixPathLike:
    """
    Minimal POSIX path inspector for validating recorded relative paths.

    ``pathlib.PurePosixPath`` would treat a Windows-style ``C:\\x`` as relative,
    so absolute-path detection is done explicitly here instead.
    """

    def __init__(self, raw: str) -> None:
        self.raw = raw
        self.parts = tuple(p for p in raw.replace("\\", "/").split("/") if p)

    @property
    def is_absolute(self) -> bool:
        """True for POSIX-absolute, Windows-drive, or UNC paths."""
        raw = self.raw
        return raw.startswith("/") or raw.startswith("\\\\") or bool(re.match(r"^[A-Za-z]:", raw))


def validate_manifest(
    source: SourceRecord,
    subsets: list[SubsetRecord],
    *,
    project_root: Path | None = None,
    require_files: bool = False,
) -> None:
    """Validate a product record together with all of its subset records."""
    validate_source(source, project_root=project_root)

    if not subsets:
        raise ProvenanceError(f"{source.dataset_id}: manifest contains no subsets")

    seen: set[str] = set()
    for sub in subsets:
        if sub.dataset_id != source.dataset_id:
            raise ProvenanceError(
                f"subset {sub.region!r} has dataset_id {sub.dataset_id!r}, "
                f"expected {source.dataset_id!r}"
            )
        if sub.region in seen:
            raise ProvenanceError(f"{source.dataset_id}: duplicate region {sub.region!r}")
        seen.add(sub.region)
        validate_subset(sub, project_root=project_root, require_file=require_files)


def assert_publishable(source: SourceRecord) -> None:
    """
    Raise unless this product's raw files may be committed or redistributed.

    Call this immediately before exporting, committing, or publishing external
    raw data.  Verification alone is not permission.
    """
    validate_source(source)
    if not source.redistribution_allowed:
        raise LicenceGateError(
            f"{source.dataset_id}: redistribution_allowed is False. Metadata may describe this "
            f"source, but its raw data must not be committed or published."
        )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_source(path: Path) -> SourceRecord:
    """
    Load a :class:`SourceRecord` from a YAML or JSON file.

    Accepts either a file whose top level *is* the source mapping, or a manifest
    that nests it under a ``"source"`` key.  Supporting both keeps the manifest
    the single source of truth for a product while still allowing a future
    product to ship a standalone source file.
    """
    payload = _load_mapping(path)
    if "source" in payload and isinstance(payload["source"], dict):
        payload = dict(payload["source"])
    for key in ("provides_variables", "cannot_provide"):
        if key in payload and payload[key] is not None:
            payload[key] = tuple(payload[key])
    known = SourceRecord.__dataclass_fields__.keys()
    unknown = set(payload) - set(known)
    if unknown:
        raise ProvenanceError(f"{path}: unknown field(s) {sorted(unknown)}")
    return SourceRecord(**payload)


def load_manifest(path: Path) -> list[SubsetRecord]:
    """Load the list of :class:`SubsetRecord` from a manifest JSON file."""
    payload = _load_mapping(path)
    raw_subsets = payload.get("subsets")
    if not isinstance(raw_subsets, list):
        raise ProvenanceError(f"{path}: manifest must contain a 'subsets' list")

    records: list[SubsetRecord] = []
    known = set(SubsetRecord.__dataclass_fields__.keys())
    for entry in raw_subsets:
        unknown = set(entry) - known
        if unknown:
            raise ProvenanceError(f"{path}: unknown subset field(s) {sorted(unknown)}")
        entry = dict(entry)
        for key in ("requested_bbox", "actual_bbox"):
            if key in entry and entry[key] is not None:
                entry[key] = tuple(entry[key])
        records.append(SubsetRecord(**entry))
    return records


def _load_mapping(path: Path) -> dict[str, Any]:
    """Read a YAML or JSON file and return its top-level mapping."""
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        import yaml

        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ProvenanceError(f"{path}: expected a mapping at the top level")
    return payload
