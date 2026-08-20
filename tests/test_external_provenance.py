"""
Tests for the external (real) data provenance layer.

Scope
-----
These tests cover ``src/external`` and the GEBCO_2026 manifest only.  They never
train a model, never touch MLflow, never run DVC, and never read or write the
synthetic prototype dataset.

The point of the layer is that real external data cannot enter the project
without verified provenance, and cannot masquerade as reef-condition ground
truth.  These tests pin both properties.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path

import pytest

from src.external.provenance import (
    LicenceGateError,
    ProvenanceError,
    SourceRecord,
    SubsetRecord,
    assert_publishable,
    load_manifest,
    load_source,
    validate_manifest,
    validate_source,
    validate_subset,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "external" / "metadata" / "gebco_2026.manifest.json"

EXPECTED_REGIONS = {
    "Lakshadweep",
    "Gulf of Mannar",
    "Gulf of Kutch",
    "Andaman and Nicobar Islands",
}

#: Variables no external product may ever claim to supply.
PROJECT_TARGETS = ("reef_health", "restoration_suitability")


# ---------------------------------------------------------------------------
# Source-inspection helpers
# ---------------------------------------------------------------------------


def _imported_modules(path: Path) -> list[str]:
    """Return every module name imported by the Python file at *path*."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def _code_string_literals(path: Path) -> list[str]:
    """
    Return string literals from *path*, excluding docstrings.

    Docstrings are prose about the code and are allowed to name things the code
    must not touch; only literals that participate in execution are returned.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))

    docstring_nodes: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ) and (body := getattr(node, "body", None)):
            first = body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                if isinstance(first.value.value, str):
                    docstring_nodes.add(id(first.value))

    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstring_nodes
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _minimal_source(**overrides) -> SourceRecord:
    """A structurally valid SourceRecord, for negative testing."""
    base = {
        "dataset_id": "test_product",
        "source_name": "Test Provider",
        "product_name": "Test Product",
        "source_url": "https://example.org/product",
        "version": "v1",
        "licence_name": "Public domain",
        "licence_url": "https://example.org/terms",
        "citation": "Test Provider (2026). Test Product.",
        "observation_type": "measured",
        "processing_level": "L2",
        "original_format": "NetCDF",
        "original_crs": "EPSG:4326",
        "spatial_resolution": "1 km",
        "temporal_resolution": "daily",
        "geographic_scope": "global",
        "temporal_scope": "2020-2026",
        "retrieved_at_utc": "2026-08-19T12:00:00Z",
        "access_method": "https download",
        "licence_verified": True,
        "licence_verified_via": "Read from the publisher terms page on 2026-08-19",
        "redistribution_allowed": True,
    }
    base.update(overrides)
    return SourceRecord(**base)


def _minimal_subset(**overrides) -> SubsetRecord:
    """A structurally valid SubsetRecord, for negative testing."""
    base = {
        "dataset_id": "test_product",
        "region": "Test Region",
        "local_file": "data/external/raw/test_product/test.nc",
        "file_format": "NetCDF-3 classic",
        "file_size_bytes": 1024,
        "sha256": "a" * 64,
        "retrieved_at_utc": "2026-08-19T12:00:00Z",
        "request_url": "https://example.org/subset",
        "requested_bbox": (8.0, 9.0, 78.0, 79.0),
        "actual_bbox": (8.0, 9.0, 78.0, 79.0),
        "dimensions": {"lat": 10, "lon": 10},
    }
    base.update(overrides)
    return SubsetRecord(**base)


@pytest.fixture(scope="module")
def manifest_payload() -> dict:
    assert MANIFEST_PATH.is_file(), f"GEBCO manifest missing at {MANIFEST_PATH}"
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def gebco_source() -> SourceRecord:
    return load_source(MANIFEST_PATH)


@pytest.fixture(scope="module")
def gebco_subsets() -> list[SubsetRecord]:
    return load_manifest(MANIFEST_PATH)


# ---------------------------------------------------------------------------
# The licence gate
# ---------------------------------------------------------------------------


class TestLicenceGate:
    """An external product is untrusted until its licence has been read."""

    def test_unverified_licence_is_rejected(self):
        with pytest.raises(LicenceGateError, match="licence_verified is False"):
            validate_source(_minimal_source(licence_verified=False))

    def test_licence_verified_defaults_to_false(self):
        """The default must be the safe one, so omission cannot pass the gate."""
        assert SourceRecord.__dataclass_fields__["licence_verified"].default is False

    def test_verified_without_evidence_is_rejected(self):
        with pytest.raises(LicenceGateError, match="licence_verified_via"):
            validate_source(_minimal_source(licence_verified_via=None))

    def test_unstated_redistribution_is_rejected(self):
        with pytest.raises(LicenceGateError, match="redistribution_allowed"):
            validate_source(_minimal_source(redistribution_allowed=None))

    def test_redistribution_false_is_valid_but_not_publishable(self):
        """Non-redistributable data may still be used — just never published."""
        record = _minimal_source(redistribution_allowed=False)
        validate_source(record)  # usable
        with pytest.raises(LicenceGateError, match="must not be committed or published"):
            assert_publishable(record)

    def test_redistribution_true_is_publishable(self):
        assert_publishable(_minimal_source(redistribution_allowed=True))


class TestNoFabricatedTargets:
    """No external product may claim to supply a reef-condition target."""

    @pytest.mark.parametrize("target", PROJECT_TARGETS)
    def test_claiming_a_target_is_rejected(self, target: str):
        with pytest.raises(ProvenanceError, match="claims target"):
            validate_source(_minimal_source(provides_variables=(target,)))

    def test_external_record_cannot_be_marked_synthetic(self):
        with pytest.raises(ProvenanceError, match="is_synthetic must be False"):
            validate_source(_minimal_source(is_synthetic=True))


class TestRecordStructure:
    """Structural rules that keep committed metadata portable and honest."""

    def test_absolute_local_file_is_rejected(self):
        """Committed metadata must not leak a machine-specific path."""
        with pytest.raises(ProvenanceError, match="repository-relative"):
            validate_subset(_minimal_subset(local_file="/home/someone/secret/test.nc"))

    def test_parent_traversal_is_rejected(self):
        with pytest.raises(ProvenanceError, match="escape the repository root"):
            validate_subset(_minimal_subset(local_file="data/external/../../etc/passwd"))

    def test_bad_sha256_is_rejected(self):
        with pytest.raises(ProvenanceError, match="64 lowercase hex"):
            validate_subset(_minimal_subset(sha256="not-a-hash"))

    def test_uppercase_sha256_is_rejected(self):
        with pytest.raises(ProvenanceError, match="64 lowercase hex"):
            validate_subset(_minimal_subset(sha256="A" * 64))

    def test_impossible_bbox_is_rejected(self):
        with pytest.raises(ProvenanceError, match="latitudes invalid"):
            validate_subset(_minimal_subset(actual_bbox=(50.0, 10.0, 78.0, 79.0)))

    def test_bad_timestamp_is_rejected(self):
        with pytest.raises(ProvenanceError, match="ISO-8601 UTC"):
            validate_source(_minimal_source(retrieved_at_utc="19 August 2026"))

    def test_duplicate_regions_are_rejected(self):
        source = _minimal_source()
        subsets = [_minimal_subset(), _minimal_subset()]
        with pytest.raises(ProvenanceError, match="duplicate region"):
            validate_manifest(source, subsets)

    def test_mismatched_dataset_id_is_rejected(self):
        source = _minimal_source()
        subsets = [_minimal_subset(dataset_id="other_product")]
        with pytest.raises(ProvenanceError, match="expected"):
            validate_manifest(source, subsets)

    def test_empty_manifest_is_rejected(self):
        with pytest.raises(ProvenanceError, match="no subsets"):
            validate_manifest(_minimal_source(), [])


# ---------------------------------------------------------------------------
# The GEBCO_2026 manifest
# ---------------------------------------------------------------------------


class TestGebcoManifest:
    """The real acquired product must satisfy every rule the layer enforces."""

    def test_manifest_validates(self, gebco_source, gebco_subsets):
        validate_manifest(gebco_source, gebco_subsets, project_root=PROJECT_ROOT)

    def test_all_four_regions_present(self, gebco_subsets):
        assert {s.region for s in gebco_subsets} == EXPECTED_REGIONS

    def test_source_is_not_synthetic(self, gebco_source):
        assert gebco_source.is_synthetic is False

    def test_every_subset_is_not_synthetic(self, gebco_subsets):
        assert all(s.is_synthetic is False for s in gebco_subsets)

    def test_licence_is_verified_with_evidence(self, gebco_source):
        assert gebco_source.licence_verified is True
        assert gebco_source.licence_verified_via
        assert "public domain" in gebco_source.licence_name.lower()

    def test_redistribution_is_explicit(self, gebco_source):
        assert gebco_source.redistribution_allowed is True

    def test_doi_and_version_pinned(self, gebco_source):
        assert gebco_source.version == "GEBCO_2026"
        assert gebco_source.doi == "10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa"

    def test_checksums_well_formed(self, gebco_subsets):
        for subset in gebco_subsets:
            assert re.fullmatch(r"[0-9a-f]{64}", subset.sha256), subset.region

    def test_checksums_unique(self, gebco_subsets):
        """Four distinct regions must not share a checksum."""
        hashes = [s.sha256 for s in gebco_subsets]
        assert len(set(hashes)) == len(hashes)

    def test_local_paths_are_relative_and_under_external_raw(self, gebco_subsets):
        for subset in gebco_subsets:
            assert not subset.local_file.startswith("/"), subset.region
            assert subset.local_file.startswith("data/external/raw/"), subset.region

    def test_no_home_directory_leaked_anywhere(self):
        """No committed metadata may embed a machine-specific absolute path."""
        text = MANIFEST_PATH.read_text(encoding="utf-8")
        assert "/home/" not in text
        assert not re.search(r"[A-Za-z]:\\\\", text)

    def test_elevation_units_are_metres(self, gebco_subsets):
        assert all(s.elevation_units == "m" for s in gebco_subsets)

    def test_grid_spacing_is_15_arcseconds(self, gebco_subsets):
        for subset in gebco_subsets:
            assert subset.grid_spacing_deg == pytest.approx(1.0 / 240.0, rel=1e-6), subset.region

    def test_bathymetry_convention_negative_is_below_sea_level(self, gebco_subsets):
        """Every marine region must contain negative elevations."""
        for subset in gebco_subsets:
            assert subset.min_elevation_m < 0, subset.region
            assert subset.marine_cell_count and subset.marine_cell_count > 0, subset.region

    def test_actual_bbox_within_requested_plus_one_cell(self, gebco_subsets):
        """NCSS snaps to grid cell centres; allow one cell of slack, no more."""
        cell = 1.0 / 240.0
        for subset in gebco_subsets:
            req, act = subset.requested_bbox, subset.actual_bbox
            for i in range(4):
                assert abs(act[i] - req[i]) <= cell * 1.5, f"{subset.region} bound {i}"

    def test_minicoy_inside_lakshadweep_window(self, gebco_subsets):
        """The old synthetic box excluded Minicoy; this one must not."""
        lak = next(s for s in gebco_subsets if s.region == "Lakshadweep")
        lat_min, lat_max, lon_min, lon_max = lak.actual_bbox
        minicoy_lat, minicoy_lon = 8.28, 73.05
        assert lat_min <= minicoy_lat <= lat_max
        assert lon_min <= minicoy_lon <= lon_max

    def test_gulf_of_kutch_window_is_not_far_inland(self, gebco_subsets):
        """The old synthetic box reached 24.5 N, ~170 km inland."""
        kutch = next(s for s in gebco_subsets if s.region == "Gulf of Kutch")
        assert kutch.actual_bbox[1] <= 23.1

    def test_every_region_records_its_rationale(self, gebco_subsets):
        for subset in gebco_subsets:
            assert len(subset.bbox_rationale) > 80, subset.region

    def test_cannot_provide_lists_the_targets(self, gebco_source):
        """The limitation must travel with the data, not just live in prose."""
        cannot = {v.lower() for v in gebco_source.cannot_provide}
        for target in PROJECT_TARGETS:
            assert target in cannot
        for biological in ("coral_cover_percentage", "bleaching_percentage", "disease_percentage"):
            assert biological in cannot
        assert "sonar_backscatter" in cannot

    def test_provides_only_bathymetric_variables(self, gebco_source):
        joined = " ".join(gebco_source.provides_variables).lower()
        assert "elevation" in joined
        for target in PROJECT_TARGETS:
            assert target not in joined

    def test_manifest_does_not_claim_reef_ground_truth(self, manifest_payload):
        """No prose anywhere in the manifest may assert reef-condition truth."""
        text = json.dumps(manifest_payload).lower()
        for phrase in (
            "reef health ground truth",
            "ground truth for reef",
            "shallow water = reef",
            "shallow means reef",
        ):
            assert phrase not in text

    def test_manifest_warns_shallow_is_not_reef(self, manifest_payload):
        note = manifest_payload["bathymetric_context"]["note"].lower()
        assert "not a reef label" in note

    def test_resolution_caveat_recorded(self, gebco_source):
        """The rugosity resolution limitation must be machine-readable."""
        blob = f"{gebco_source.disclaimer} {' '.join(gebco_source.cannot_provide)}".lower()
        assert "rugosity" in blob


# ---------------------------------------------------------------------------
# Raw files (present only where the acquisition has been run)
# ---------------------------------------------------------------------------


class TestRawFiles:
    """Validate on-disk files when they exist; skip cleanly when they do not."""

    def test_referenced_files_match_recorded_size(self, gebco_source, gebco_subsets):
        missing = [s.region for s in gebco_subsets if not (PROJECT_ROOT / s.local_file).is_file()]
        if missing:
            pytest.skip(f"raw GEBCO files not present locally: {missing}")
        validate_manifest(
            gebco_source, gebco_subsets, project_root=PROJECT_ROOT, require_files=True
        )

    def test_recorded_checksums_match_files(self, gebco_subsets):
        import hashlib

        for subset in gebco_subsets:
            path = PROJECT_ROOT / subset.local_file
            if not path.is_file():
                pytest.skip(f"raw file not present: {subset.local_file}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == subset.sha256, subset.region


# ---------------------------------------------------------------------------
# Repository safety
# ---------------------------------------------------------------------------


class TestRepositorySafety:
    """The external layer must not endanger the synthetic pipeline or Git."""

    def test_raw_external_directory_is_git_ignored(self):
        result = subprocess.run(
            ["git", "check-ignore", "-q", "data/external/raw/"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, "data/external/raw/ is not git-ignored"

    def test_metadata_directory_is_not_git_ignored(self):
        """Provenance must remain trackable even though raw data is not."""
        result = subprocess.run(
            ["git", "check-ignore", "-q", "data/external/metadata/gebco_2026.manifest.json"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0, "the manifest must not be git-ignored"

    def test_synthetic_dataset_path_unchanged(self):
        """The frozen synthetic benchmark must still be where it always was."""
        assert (PROJECT_ROOT / "data" / "raw" / "observations.csv").is_file()

    def test_external_layer_performs_no_network_access(self):
        """
        Importing ``src.external`` must never touch the network.

        Acquisition is an explicit command (``scripts/fetch_gebco_2026.py``).
        Running the test suite, or importing the provenance layer in CI, must
        not download anything — a clean CI run has no GEBCO files and must not
        try to obtain them.
        """
        network_modules = {"urllib", "urllib.request", "http", "http.client", "socket", "ftplib"}
        for module in (PROJECT_ROOT / "src" / "external").glob("*.py"):
            imported = set(_imported_modules(module))
            offending = imported.intersection(network_modules) | {
                name
                for name in imported
                if name.split(".")[0] in {"requests", "httpx", "aiohttp", "urllib3"}
            }
            assert not offending, f"{module.name} imports network module(s) {sorted(offending)}"

    def test_this_test_module_performs_no_network_access(self):
        """The tests themselves must not be able to download anything."""
        imported = set(_imported_modules(Path(__file__)))
        for name in ("urllib", "urllib.request", "requests", "httpx", "socket"):
            assert name not in imported, f"test module imports {name}"

    def test_external_layer_does_not_import_synthetic_pipeline(self):
        """Isolation is enforced in the import graph, not just by convention."""
        for module in (PROJECT_ROOT / "src" / "external").glob("*.py"):
            for name in _imported_modules(module):
                assert not name.startswith("src.data"), (
                    f"{module.name} imports the synthetic pipeline ({name})"
                )

    def test_external_code_does_not_reference_synthetic_data_paths(self):
        """
        No *executable* reference to the synthetic dataset.

        Docstrings are excluded deliberately: they describe the isolation and
        must be free to name the file they are isolating from.
        """
        modules = list((PROJECT_ROOT / "src" / "external").glob("*.py"))
        modules.append(PROJECT_ROOT / "scripts" / "fetch_gebco_2026.py")
        for module in modules:
            for literal in _code_string_literals(module):
                assert "observations.csv" not in literal, f"{module.name}: {literal!r}"
                assert "data/raw/" not in literal, f"{module.name}: {literal!r}"
