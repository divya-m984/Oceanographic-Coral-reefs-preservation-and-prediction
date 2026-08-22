"""
Tests for the NOAA Coral Reef Watch external data source.

Scope
-----
These tests cover the CRW 5 km v3.1 manifest and the acquisition script's
static properties.  They never train a model, never touch MLflow, never run
DVC, never read the synthetic prototype dataset, and never reach the network.

Why a separate module
---------------------
``test_external_provenance.py`` pins the *layer* — the schema, the licence
gate, and the GEBCO_2026 product.  This module pins the second acquired
product.  The split keeps per-source rules next to the source they describe;
the generic rules are not duplicated here.

The central property under test is not schema conformance.  It is that a
thermal-stress product cannot quietly become a biological label.  Degree
Heating Weeks is a predictor of bleaching risk; it is not a bleaching
percentage, and no amount of thresholding makes it one.  Several tests below
exist solely to make that confusion fail loudly.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from src.external.provenance import (
    FORBIDDEN_BIOLOGICAL_CLAIMS,
    FORBIDDEN_TARGET_CLAIMS,
    ProvenanceError,
    load_manifest,
    load_source,
    validate_manifest,
    validate_source,
)
from tests.test_external_provenance import _code_string_literals, _imported_modules

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_DIR = PROJECT_ROOT / "data" / "external" / "metadata"
CRW_MANIFEST = METADATA_DIR / "noaa_crw_5km_v3_1.manifest.json"
GEBCO_MANIFEST = METADATA_DIR / "gebco_2026.manifest.json"
FETCH_SCRIPT = PROJECT_ROOT / "scripts" / "fetch_noaa_crw.py"

EXPECTED_REGIONS = {
    "Lakshadweep",
    "Gulf of Mannar",
    "Gulf of Kutch",
    "Andaman and Nicobar Islands",
}

EXPECTED_PRODUCTS = {"sst", "sst_anomaly", "hotspot", "dhw"}


def _normalised(text: str) -> str:
    """
    Lowercase, collapse whitespace, drop emphasis markers and quote marks.

    Phrase assertions below run against prose that is hard-wrapped and, in the
    docs, bold-marked and quoted.  Without this, ``"bias correction"`` fails
    purely because a line break landed between the two words, and
    ``described as "OSTIA-free"`` fails because the sentence's full stop sits
    inside the closing quote.  Both would test wording, not meaning.
    """
    stripped = text.replace("*", "").replace("—", "--")
    for quote in ('"', "'", "“", "”"):
        stripped = stripped.replace(quote, "")
    return " ".join(stripped.lower().split())


#: Units that must survive into the metadata.  A DHW reported in ``degree_C``
#: rather than ``degree_Celsius_weeks`` is a different physical quantity.
EXPECTED_UNITS = {
    "analysed_sst": "degree_C",
    "sea_surface_temperature_anomaly": "degree_C",
    "hotspot": "degree_C",
    "degree_heating_week": "degree_Celsius_weeks",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def payload() -> dict:
    assert CRW_MANIFEST.is_file(), f"CRW manifest missing at {CRW_MANIFEST}"
    return json.loads(CRW_MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def source():
    return load_source(CRW_MANIFEST)


@pytest.fixture(scope="module")
def subsets():
    return load_manifest(CRW_MANIFEST)


# ---------------------------------------------------------------------------
# Manifest structure
# ---------------------------------------------------------------------------


class TestManifestSchema:
    """The CRW product must satisfy every rule the external layer enforces."""

    def test_manifest_validates(self, source, subsets):
        validate_manifest(source, subsets, project_root=PROJECT_ROOT)

    def test_schema_version_matches_the_layer(self, payload):
        assert payload["schema"] == "coralsense.external.manifest/v1"

    def test_dataset_id_is_version_pinned(self, payload, source):
        assert payload["dataset_id"] == "noaa_crw_5km_v3_1"
        assert source.dataset_id == payload["dataset_id"]

    def test_source_is_not_synthetic(self, source):
        assert source.is_synthetic is False

    def test_every_subset_is_not_synthetic(self, subsets):
        assert all(s.is_synthetic is False for s in subsets)

    def test_product_identity_is_recorded(self, source):
        assert "Coral Reef Watch" in source.source_name
        assert "NESDIS" in source.source_name
        assert "5km" in source.product_name or "5 km" in source.product_name
        assert source.version.startswith("v3.1")
        assert source.doi == "10.25921/6jgr-pt28"
        assert source.product_identifier == "gov.noaa.nodc:CRW-5km-HeatStressProducts"

    def test_observation_type_is_not_measured(self, source):
        """
        CRW products are derived, not directly measured.

        ``measured`` would imply an instrument read these numbers off the reef.
        Nothing did: SST is an L4 analysis, and HotSpot/DHW/anomaly are computed
        from it.
        """
        assert source.observation_type == "derived"
        assert source.processing_level == "L4"

    def test_grid_and_resolution_recorded(self, source):
        assert "0.05" in source.spatial_resolution
        assert "EPSG:4326" in source.original_crs
        assert source.temporal_resolution == "daily"


class TestLicence:
    """The licence gate, and the era-dependent CoralTemp complication."""

    def test_licence_is_verified_with_first_party_evidence(self, source):
        assert source.licence_verified is True
        evidence = source.licence_verified_via
        assert evidence
        assert "coralreefwatch.noaa.gov" in evidence or "Coral Reef Watch" in evidence
        assert "public domain" in evidence.lower()

    def test_redistribution_is_explicit(self, source):
        assert source.redistribution_allowed is True

    def test_raw_is_not_tracked_in_git(self, source):
        assert source.raw_tracked_in_git is False

    def test_ostia_lineage_is_recorded_not_glossed_over(self, source):
        """
        CoralTemp has a multi-source lineage; the metadata must say so.

        The earliest segment is Met Office OSTIA and is restrictive.  Recording
        only "public domain" would be a true-sounding half-statement, so the
        lineage must appear -- and it must state the period accurately, ending
        at November 2002 rather than at 2002 flat, which would read as though
        January 2002 were already outside it.
        """
        blob = f"{source.licence_name} {source.licence_verified_via}"
        assert "OSTIA" in blob
        assert "1985" in blob
        assert "November 2002" in blob
        assert "1985-2002" not in blob

    def test_metadata_records_continued_ostia_bias_correction(self, source):
        """
        OSTIA does not drop out of CoralTemp after 2002.

        NOAA states the Geo-Polar Blended product switched to OSTIA for bias
        correction in 2016 -- upstream of everything this project acquired.  So
        the metadata must carry that fact *and* explicitly reject the
        "OSTIA-free" framing.

        This is asserted positively, not as a banned substring.  Every mention
        of "OSTIA-free" in this repository is a denial of the phrase, and
        substring matching cannot tell a denial from a claim -- the same
        limitation already documented for the bleaching disclaimer in
        ``test_manifest_never_asserts_observed_bleaching``.  Requiring the
        denial to be present is the property that actually protects a reader.
        """
        blob = _normalised(f"{source.licence_name} {source.licence_verified_via}")
        assert "bias correction" in blob
        assert "must not be described as ostia-free" in blob

    def test_redistribution_rests_on_noaa_terms_not_on_the_date_floor(self, source):
        """
        The licence determination and the lineage policy are separate things.

        ``redistribution_allowed`` follows from what NOAA CRW publishes about
        the product it distributes and from the licence metadata delivered in
        the files.  It must not be presented as a consequence of where the
        acquisition window sits in the source lineage -- that reasoning would
        collapse the moment anyone noticed the 2016 bias-correction switch.
        """
        assert source.licence_verified is True
        assert source.redistribution_allowed is True
        blob = _normalised(f"{source.licence_name} {source.licence_verified_via}")
        assert "not inferred from" in blob or "not as the basis" in blob
        assert "delivered" in blob

    def test_acquisition_respects_the_lineage_policy_floor(self, source, subsets):
        """
        No acquired day falls inside the direct-OSTIA or merge periods.

        This is the project's conservative provenance policy, not a licence
        test: the floor is 2002-12-01 because NOAA merged OSTIA into the
        Geo-Polar Blended reanalysis linearly over November 1-29, 2002.  See
        ``TestLineagePolicyGuard``.
        """
        for subset in subsets:
            assert subset.actual_time_range is not None, subset.region
            assert subset.actual_time_range[0] >= "2002-12-01", subset.region
        assert "2018-01-01" in source.temporal_scope

    def test_citation_is_present_and_names_the_doi(self, source):
        assert "Coral Reef Watch" in source.citation
        assert "10.25921/6jgr-pt28" in source.citation


# ---------------------------------------------------------------------------
# The point of the whole exercise
# ---------------------------------------------------------------------------


class TestThermalStressIsNotABiologicalLabel:
    """Heat stress is a predictor. It is not an observed reef outcome."""

    @pytest.mark.parametrize("target", FORBIDDEN_TARGET_CLAIMS)
    def test_does_not_claim_a_project_target(self, source, target: str):
        joined = " ".join(source.provides_variables).lower()
        assert target not in joined

    @pytest.mark.parametrize("biological", FORBIDDEN_BIOLOGICAL_CLAIMS)
    def test_does_not_claim_a_biological_observation(self, source, biological: str):
        joined = " ".join(source.provides_variables).lower()
        assert biological not in joined

    def test_cannot_provide_lists_the_targets_and_the_biology(self, source):
        """The limitation must travel with the data, not just live in prose."""
        cannot = {v.lower() for v in source.cannot_provide}
        for name in FORBIDDEN_TARGET_CLAIMS + FORBIDDEN_BIOLOGICAL_CLAIMS:
            assert name in cannot, name

    def test_cannot_provide_disclaims_in_situ_temperature(self, source):
        """
        Satellite SST is not the synthetic pipeline's ``water_temperature_c``.

        It is a sea-surface analysis over a 5 km cell, not a probe at reef
        depth, and silently substituting one for the other would misdescribe
        what was measured.
        """
        blob = " ".join(source.cannot_provide).lower()
        assert "in-situ" in blob
        assert "water_temperature_c" in blob

    def test_the_gate_would_reject_a_dhw_derived_bleaching_claim(self, source):
        """
        The prohibition is enforced, not merely documented.

        A future contributor writing ``provides_variables=("bleaching_percentage
        (derived from DHW)",)`` must be stopped by code.
        """
        forged = type(source)(
            **{
                **source.to_dict(),
                "provides_variables": ("bleaching_percentage (derived from DHW thresholds)",),
                "variable_units": dict(source.variable_units),
            }
        )
        with pytest.raises(ProvenanceError, match="claims target"):
            validate_source(forged)

    def test_manifest_states_the_inequalities_explicitly(self, payload):
        warning = payload["variable_semantics"]["warning"].lower()
        assert "dhw != bleaching_percentage" in warning
        assert "hotspot != bleaching_percentage" in warning
        assert "bleaching alert area != observed bleaching class" in warning

    def test_every_product_records_what_it_does_not_mean(self, payload):
        products = payload["variable_semantics"]["products"]
        assert set(products) == EXPECTED_PRODUCTS
        for key, entry in products.items():
            assert entry["means"].strip(), key
            assert len(entry["does_not_mean"]) > 60, key

    def test_dhw_disclaimer_names_risk_not_outcome(self, payload):
        dhw = payload["variable_semantics"]["products"]["dhw"]
        assert "risk" in dhw["does_not_mean"].lower()
        assert "bleaching percentage" in dhw["does_not_mean"].lower()

    def test_source_disclaimer_forbids_thresholding(self, source):
        assert source.disclaimer is not None
        assert "threshold" in source.disclaimer.lower()

    def test_manifest_never_asserts_observed_bleaching(self, payload):
        """
        Only unambiguous *claims* are banned, not denials.

        Substring matching cannot tell "is bleaching ground truth" from "is not
        bleaching ground truth", so the phrases below are ones that would only
        ever appear as an assertion.
        """
        text = json.dumps(payload).lower()
        for phrase in (
            "observed bleaching percentage",
            "bleaching ground truth",
            "reef health ground truth",
            "dhw is bleaching",
            "equivalent to bleaching",
        ):
            assert phrase not in text, phrase


# ---------------------------------------------------------------------------
# Variables, units, geography, time
# ---------------------------------------------------------------------------


class TestVariablesAndUnits:
    def test_all_four_products_present_for_all_four_regions(self, subsets):
        assert len(subsets) == 16
        variables = {s.variable_name for s in subsets}
        assert variables == set(EXPECTED_UNITS)

    def test_units_are_recorded_per_file(self, subsets):
        for subset in subsets:
            assert subset.variable_units == EXPECTED_UNITS[subset.variable_name], subset.region

    def test_dhw_units_are_degree_weeks_not_degrees(self, subsets):
        """A DHW in plain degrees would be a different physical quantity."""
        dhw = [s for s in subsets if s.variable_name == "degree_heating_week"]
        assert dhw
        assert all(s.variable_units == "degree_Celsius_weeks" for s in dhw)

    def test_product_level_units_map_is_populated(self, source):
        assert source.variable_units
        assert source.variable_units["degree_heating_week"] == "degree_Celsius_weeks"

    def test_fill_value_recorded_per_file(self, subsets):
        for subset in subsets:
            assert subset.fill_value, subset.region

    def test_delivered_dtype_is_recorded(self, subsets):
        """
        The server does not deliver one uniform type.

        SST arrives as float64 and the three stress products as float32, even
        though the source metadata declares all four as double. That sets the
        precision floor for any later value comparison, so it is recorded rather
        than assumed.
        """
        by_variable = {s.variable_name: s.variable_dtype for s in subsets}
        assert by_variable["analysed_sst"] == "float64"
        for variable in ("hotspot", "degree_heating_week", "sea_surface_temperature_anomaly"):
            assert by_variable[variable] == "float32", variable

    def test_missing_mask_limitation_is_recorded(self, source):
        """
        The delivered files carry no CRW ``mask`` variable.

        That means land and genuinely missing data are indistinguishable — both
        are NaN.  A silent limitation is a trap, so it must be written down.
        """
        assert "mask" in source.notes.lower()


class TestGeography:
    def test_every_expected_region_appears(self, subsets):
        regions = {s.region.split(" / ")[0] for s in subsets}
        assert regions == EXPECTED_REGIONS

    def test_windows_match_the_gebco_windows_exactly(self, subsets):
        """
        The two products describe the same four systems.

        If one set of windows is ever edited without the other, the products
        stop being comparable — so the equality is pinned rather than trusted.
        """
        gebco = {s.region: s.requested_bbox for s in load_manifest(GEBCO_MANIFEST)}
        for subset in subsets:
            region = subset.region.split(" / ")[0]
            assert subset.requested_bbox == gebco[region], region

    def test_actual_bbox_within_requested_plus_one_cell(self, subsets):
        """ERDDAP snaps to grid-cell centres; allow one cell of slack, no more."""
        cell = 0.05
        for subset in subsets:
            req, act = subset.requested_bbox, subset.actual_bbox
            for i in range(4):
                assert abs(act[i] - req[i]) <= cell * 1.5, f"{subset.region} bound {i}"

    def test_grid_spacing_is_five_km(self, subsets):
        """
        Tolerance is float32-sized, not arbitrary.

        ERDDAP delivers the latitude axis as float32, so the differenced spacing
        carries ~2e-7 of representation error. Demanding exact 0.05 would fail on
        correct data.
        """
        for subset in subsets:
            assert subset.grid_spacing_deg == pytest.approx(0.05, abs=1e-5), subset.region

    def test_latitude_order_is_recorded(self, subsets):
        """
        The two NOAA servers publish some variables north-up and some south-up.

        Recording the delivered order is what stops a future reader flipping a
        grid without noticing.
        """
        for subset in subsets:
            assert subset.latitude_order in ("ascending", "descending"), subset.region

    def test_every_region_records_its_rationale(self, subsets):
        for subset in subsets:
            assert len(subset.bbox_rationale) > 80, subset.region

    def test_windows_are_described_as_windows_not_reef_masks(self, subsets):
        """
        A rectangle over open ocean is an acquisition extent, not a reef.

        Treating one as a reef mask is how the synthetic dataset ended up
        sampling coordinates uniformly over deep water and dry land.
        """
        for subset in subsets:
            assert "acquisition window" in subset.bbox_rationale.lower(), subset.region


class TestTemporalCoverage:
    def test_requested_window_is_the_documented_one(self, subsets):
        for subset in subsets:
            assert subset.requested_time_range == (
                "2018-01-01T12:00:00Z",
                "2024-12-31T12:00:00Z",
            ), subset.region

    def test_actual_window_covers_seven_calendar_years(self, subsets):
        """
        The end may overshoot the request by one day, and legitimately does.

        NOAA's daily archive has no 2024-12-31, and ERDDAP's ``(time)`` selector
        snaps to the *nearest available* step — so asking for 2024-12-31 returns
        2025-01-01. Pinning the requested date here would encode a false
        expectation; pinning the tolerance records the real behaviour.
        """
        for subset in subsets:
            start, end = subset.actual_time_range
            assert start.startswith("2018-01-01"), subset.region
            assert end[:10] in ("2024-12-30", "2024-12-31", "2025-01-01"), subset.region

    def test_missing_dates_are_identical_across_regions(self, subsets):
        """
        A gap must come from the archive, not from our subsetting.

        Four regions of the same product are four independent requests. If they
        disagreed about which days exist, that would point at an indexing bug
        rather than at NOAA. They agree.
        """
        by_variable: dict[str, set[int]] = {}
        for subset in subsets:
            by_variable.setdefault(subset.variable_name, set()).add(subset.n_time_steps)
        for variable, counts in by_variable.items():
            assert len(counts) == 1, f"{variable}: regions disagree on step count {counts}"

    def test_time_step_count_is_plausible_for_daily_data(self, subsets):
        """2018-01-01..2024-12-31 is 2557 days; gaps are allowed but not chasms."""
        for subset in subsets:
            assert subset.n_time_steps is not None
            assert 2400 <= subset.n_time_steps <= 2557, f"{subset.region}: {subset.n_time_steps}"

    def test_spacing_is_daily(self, subsets):
        for subset in subsets:
            assert subset.time_spacing_days == pytest.approx(1.0, abs=0.01), subset.region

    def test_missing_dates_are_counted_not_hidden(self, subsets):
        for subset in subsets:
            assert subset.missing_dates is not None, subset.region
            assert subset.missing_dates >= 0, subset.region

    def test_source_temporal_scope_states_both_windows(self, source):
        assert "1985" in source.temporal_scope
        assert "2018-01-01" in source.temporal_scope
        assert "2024-12-31" in source.temporal_scope


class TestValueStatistics:
    """Recorded statistics must be physically possible."""

    def test_sst_is_in_a_plausible_ocean_range(self, subsets):
        for subset in subsets:
            if subset.variable_name != "analysed_sst":
                continue
            assert -2.0 <= subset.min_value <= 40.0, subset.region
            assert -2.0 <= subset.max_value <= 40.0, subset.region

    def test_no_decoded_fill_value_leaked_into_the_data(self, subsets):
        """
        A decoded sentinel is the classic silent corruption.

        CRW's own sentinel is -327.68; ERDDAP re-encodes to NaN.  If either ever
        survives as a real value, the extrema make it obvious.
        """
        for subset in subsets:
            assert subset.min_value > -300.0, subset.region
            assert subset.max_value < 300.0, subset.region

    def test_dhw_is_non_negative(self, subsets):
        """Accumulated heat stress cannot be negative by construction."""
        for subset in subsets:
            if subset.variable_name == "degree_heating_week":
                assert subset.min_value >= 0.0, subset.region

    def test_anomaly_and_hotspot_straddle_zero_sensibly(self, subsets):
        for subset in subsets:
            if subset.variable_name in ("sea_surface_temperature_anomaly", "hotspot"):
                assert -15.0 <= subset.min_value <= 15.0, subset.region
                assert -15.0 <= subset.max_value <= 15.0, subset.region

    def test_nan_percentage_is_recorded_and_sane(self, subsets):
        for subset in subsets:
            assert subset.nan_percent is not None, subset.region
            assert 0.0 <= subset.nan_percent <= 100.0, subset.region

    def test_regions_are_not_identical_series(self, subsets):
        """
        Four distinct regions must not share a mean.

        Identical statistics across regions would indicate a subsetting or
        indexing bug rather than a physical coincidence.
        """
        for variable in EXPECTED_UNITS:
            means = [s.mean_value for s in subsets if s.variable_name == variable]
            assert len(set(means)) == len(means), variable

    def test_checksums_are_well_formed_and_unique(self, subsets):
        hashes = [s.sha256 for s in subsets]
        for digest in hashes:
            assert re.fullmatch(r"[0-9a-f]{64}", digest)
        assert len(set(hashes)) == len(hashes)


class TestChecksumSemantics:
    def test_not_claimed_as_publisher_canonical(self, payload):
        semantics = payload["checksum_semantics"]
        assert semantics["is_publisher_canonical_checksum"] is False
        assert semantics["algorithm"] == "SHA-256"

    def test_reproducibility_limit_is_stated(self, payload):
        text = payload["checksum_semantics"]["does_not_guarantee"].lower()
        assert "identical bytes" in text

    def test_identity_is_pinned_by_version_and_doi(self, payload):
        assert "10.25921/6jgr-pt28" in payload["checksum_semantics"]["authoritative_identity"]


class TestOmittedProducts:
    """An omission must be a recorded decision, not a gap."""

    def test_bleaching_alert_area_omission_is_documented(self, payload):
        omitted = payload["omitted_products"]
        assert "bleaching_alert_area_7d_max" in omitted
        reason = omitted["bleaching_alert_area_7d_max"]
        assert "2023-12-15" in reason
        assert "redundant" in reason.lower()
        assert "7-day" in reason.lower()


# ---------------------------------------------------------------------------
# Repository safety
# ---------------------------------------------------------------------------


class TestRepositorySafety:
    def test_raw_crw_directory_is_git_ignored(self):
        result = subprocess.run(
            ["git", "check-ignore", "-q", "data/external/raw/noaa_crw_5km_v3_1/"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, "raw CRW data is not git-ignored"

    def test_crw_manifest_is_not_git_ignored(self):
        result = subprocess.run(
            ["git", "check-ignore", "-q", "data/external/metadata/noaa_crw_5km_v3_1.manifest.json"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0, "the CRW manifest must not be git-ignored"

    def test_no_home_directory_leaked(self):
        text = CRW_MANIFEST.read_text(encoding="utf-8")
        assert "/home/" not in text
        assert not re.search(r"[A-Za-z]:\\\\", text)

    def test_local_paths_are_relative_and_under_external_raw(self, subsets):
        for subset in subsets:
            assert not subset.local_file.startswith("/"), subset.region
            assert subset.local_file.startswith("data/external/raw/"), subset.region

    def test_gebco_manifest_still_validates(self):
        """
        Adding a second source must not disturb the first.

        The schema was extended for time-varying data; GEBCO is static and must
        continue to pass unchanged.
        """
        validate_manifest(
            load_source(GEBCO_MANIFEST),
            load_manifest(GEBCO_MANIFEST),
            project_root=PROJECT_ROOT,
        )

    def test_the_two_manifests_are_separate_products(self):
        gebco = json.loads(GEBCO_MANIFEST.read_text(encoding="utf-8"))
        crw = json.loads(CRW_MANIFEST.read_text(encoding="utf-8"))
        assert gebco["dataset_id"] != crw["dataset_id"]

    def test_synthetic_dataset_is_untouched_by_this_layer(self):
        assert (PROJECT_ROOT / "data" / "raw" / "observations.csv").is_file()

    def test_fetch_script_does_not_reference_synthetic_paths(self):
        """
        No *executable* reference to the synthetic dataset.

        Docstrings are excluded deliberately: they describe the isolation and
        must be free to name the file they are isolating from.
        """
        for literal in _code_string_literals(FETCH_SCRIPT):
            assert "observations.csv" not in literal, literal
            assert "data/raw/" not in literal, literal

    def test_fetch_script_does_not_import_synthetic_pipeline(self):
        for name in _imported_modules(FETCH_SCRIPT):
            assert not name.startswith("src.data"), name

    def test_this_test_module_performs_no_network_access(self):
        """Running the suite must never download CRW data."""
        imported = set(_imported_modules(Path(__file__)))
        for name in ("urllib", "urllib.request", "requests", "httpx", "socket", "http"):
            assert name not in imported, f"test module imports {name}"

    def test_no_endpoint_url_is_embedded_in_test_code(self):
        """
        Endpoints may be named only inside the manifest these tests read.

        A literal URL in test code would be one edit away from a test that
        downloads. The check looks for URL-shaped literals rather than a
        server name, so it cannot trip over its own guard string.
        """
        url_with_host = re.compile(r"^[a-z]+://[\w.-]+\.\w+", re.IGNORECASE)
        for literal in _code_string_literals(Path(__file__)):
            assert not url_with_host.match(literal), literal

    def test_download_happens_only_inside_a_function(self):
        """
        Importing the fetch script must not fetch anything.

        Every network call must sit inside a function body, so that importing
        the module — which the clean-checkout test does — cannot reach out.
        """
        tree = ast.parse(FETCH_SCRIPT.read_text(encoding="utf-8"))
        function_spans = [
            (node.lineno, node.end_lineno)
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        ]

        network_calls = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"urlopen", "urlretrieve", "get", "post"}
        ]
        assert network_calls, "expected the fetch script to contain a network call"
        for line in network_calls:
            assert any(start <= line <= end for start, end in function_spans), (
                f"network call at module scope, line {line}"
            )

    def test_lineage_policy_floor_is_enforced_in_code(self):
        """
        The floor is enforced, not just described.

        This reads the constant statically rather than importing the script,
        because importing it would pull in its network machinery.
        """
        tree = ast.parse(FETCH_SCRIPT.read_text(encoding="utf-8"))
        boundaries = [
            node.value.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "FIRST_POST_OSTIA_BLEND_REQUEST_DATE"
                for t in node.targets
            )
            and isinstance(node.value, ast.Constant)
        ]
        assert boundaries == ["2002-12-01"], boundaries


# ---------------------------------------------------------------------------
# The CoralTemp lineage-policy guard
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fetch_module():
    """
    Import the fetch script so the guard can be exercised, not just read.

    Importing is safe: ``test_download_happens_only_inside_a_function`` pins
    that every network call in the script lives inside a function body, so
    module execution touches no socket.  The module must be registered in
    ``sys.modules`` before ``exec_module``, because ``@dataclass`` resolves
    annotations through ``sys.modules[cls.__module__]``.
    """
    spec = importlib.util.spec_from_file_location("_fetch_noaa_crw", FETCH_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


class TestLineagePolicyGuard:
    """
    The guard excludes the *direct*-OSTIA period.  That is all it claims.

    NOAA's CoralTemp source history is written at month granularity apart from
    two documented merge windows: OSTIA contributes directly January 1985 to
    November 2002, then::

        "The OSTIA reanalysis and NOAA's reprocessed 5km Geo-Polar Blended SST
        dataset were linearly merged over a period of 29 days, from
        November 1-29, 2002."

    So November 2002 is *part* OSTIA, day by day, on weights NOAA does not
    publish.  The guard rejects the whole month rather than resolve the ramp.

    Crucially, passing the guard does NOT mean a date is free of OSTIA
    influence.  NOAA states the Geo-Polar Blended product "switched to using
    OSTIA as the bias correction" in 2016 -- which covers the entire window
    this project acquires.  The guard is a conservative provenance policy about
    *direct* lineage, not a licence boundary and not a purity certificate.
    These tests pin that reading, so the constant cannot quietly drift back
    into being treated as either.
    """

    @pytest.mark.parametrize(
        "start_date",
        [
            "1985-01-01",  # first CoralTemp day, direct OSTIA
            "1998-06-15",  # mid direct-OSTIA period
            "2002-01-01",  # an earlier, incorrect floor -- still direct OSTIA
            "2002-06-01",
            "2002-10-01",  # direct OSTIA right up to the merge
            "2002-11-01",  # merge begins: part OSTIA
            "2002-11-29",  # merge ends: still part OSTIA
            "2002-11-30",  # past the merge, still refused by project policy
        ],
    )
    def test_rejects_dates_in_the_direct_ostia_and_merge_periods(self, fetch_module, start_date):
        with pytest.raises(SystemExit) as excinfo:
            fetch_module._guard_licence_window(start_date)
        assert "OSTIA" in str(excinfo.value)

    @pytest.mark.parametrize(
        "start_date",
        [
            "2002-12-01",  # the policy floor itself
            "2003-01-01",
            "2016-10-15",  # inside the second merge, which is not an OSTIA handover
            "2018-01-01",  # the acquired window
            "2024-12-31",
        ],
    )
    def test_allows_dates_after_the_direct_blend_period(self, fetch_module, start_date):
        fetch_module._guard_licence_window(start_date)

    def test_the_acquired_window_passes_the_policy_floor(self, fetch_module):
        """The 2018-2024 acquisition is unaffected by the floor."""
        fetch_module._guard_licence_window(fetch_module.START_DATE)
        assert fetch_module.START_DATE == "2018-01-01"
        assert fetch_module.END_DATE == "2024-12-31"

    def test_the_floor_is_stricter_than_the_documented_merge_end(self, fetch_module):
        """
        The margin is intentional, not an off-by-one.

        NOAA documents the direct merge ending 2002-11-29.  The policy floor
        sits later than that on purpose, because the daily blend weights are
        not published.
        """
        assert fetch_module.OSTIA_MERGE_END_DATE == "2002-11-29"
        assert fetch_module.FIRST_POST_OSTIA_BLEND_REQUEST_DATE > fetch_module.OSTIA_MERGE_END_DATE

    def test_the_constant_is_named_as_a_policy_not_a_licence_or_purity_claim(self):
        """
        The name carries the semantics; a wrong name re-creates the error.

        ``FIRST_UNRESTRICTED_CORALTEMP_DATE`` implied NOAA had established an
        unrestricted date (it has not), and ``*_OSTIA_FREE_*`` would imply the
        later record contains no OSTIA (it does, via bias correction).  Both
        names are banned outright.
        """
        text = FETCH_SCRIPT.read_text(encoding="utf-8")
        assert "FIRST_POST_OSTIA_BLEND_REQUEST_DATE" in text
        for banned in (
            "OSTIA_LICENCE_BOUNDARY",
            "FIRST_UNRESTRICTED_CORALTEMP_DATE",
            "FIRST_OSTIA_FREE_DATE",
        ):
            assert banned not in text, banned

    def test_the_guard_documents_itself_as_a_conservative_project_policy(self, fetch_module):
        """The docstring must state what the guard is, and what it is not."""
        doc = _normalised(fetch_module._guard_licence_window.__doc__)
        assert "project policy" in doc
        assert "what it is not" in doc
        assert "noaa-defined licence boundary" in doc
        assert "bias correction" in doc
        assert "ostia-free" in doc  # present only as the denial

    def test_no_constant_claims_a_day_level_source_transition(self):
        """
        NOAA gives 29-day merge spans, not transition dates.

        Nothing in the script may assert that some specific day *is* the
        OSTIA/Geo-Polar changeover; the constant is named for what it is, a
        floor below which this project declines to request.
        """
        text = FETCH_SCRIPT.read_text(encoding="utf-8")
        assert "linearly merged" in text

    def test_the_script_records_continued_ostia_bias_correction(self):
        """
        OSTIA persists as a bias-correction input after 2016.

        The script must carry NOAA's own wording for that, so a future reader
        of the constant cannot conclude the later record is source-pure.
        Asserted positively for the reason given in
        ``TestLicence.test_metadata_records_continued_ostia_bias_correction``.
        """
        text = _normalised(FETCH_SCRIPT.read_text(encoding="utf-8"))
        assert "switched to using ostia as the bias correction" in text
        assert "described as ostia-free" in text


# ---------------------------------------------------------------------------
# Prose documentation
# ---------------------------------------------------------------------------


class TestLineageDocumentation:
    """
    The prose has to carry the same caveats the code does.

    A reader reaching for this dataset is far more likely to read
    ``docs/external_data.md`` than the fetch script, so the lineage caveat
    cannot live only in a Python comment.
    """

    @pytest.fixture(params=["docs/external_data.md", "data/external/README.md"])
    def doc(self, request):
        return _normalised((PROJECT_ROOT / request.param).read_text(encoding="utf-8"))

    def test_documents_continued_ostia_bias_correction(self, doc):
        """
        NOAA: the Geo-Polar Blended product "switched to using OSTIA as the
        bias correction" in 2016 -- inside the window this project acquired.
        """
        assert "switched to using ostia as the bias correction" in doc

    def test_explicitly_rejects_the_ostia_free_framing(self, doc):
        """
        The denial has to be on the page, not merely absent from it.

        Asserted positively rather than by banning the substring: every
        occurrence of "OSTIA-free" in this repository is a rejection of the
        phrase, and substring matching cannot distinguish a denial from a
        claim.
        """
        assert "ostia-free" in doc
        assert "described as ostia-free" in doc or "is ostia-free" in doc

    def test_separates_redistribution_basis_from_the_lineage(self, doc):
        """
        Redistribution follows from NOAA CRW's published terms and the
        delivered file metadata -- not from where the window sits in the
        lineage.  The prose must say so rather than leave the reader to infer
        the (invalid) purity argument.
        """
        assert "not inferred from" in doc
        assert "project policy" in doc or "provenance policy" in doc


# ---------------------------------------------------------------------------
# Raw files (present only where the acquisition has been run)
# ---------------------------------------------------------------------------


class TestRawFiles:
    """Validate on-disk files when they exist; skip cleanly when they do not."""

    def test_referenced_files_match_recorded_size(self, source, subsets):
        missing = [s.region for s in subsets if not (PROJECT_ROOT / s.local_file).is_file()]
        if missing:
            pytest.skip(f"raw CRW files not present locally: {len(missing)} of {len(subsets)}")
        validate_manifest(source, subsets, project_root=PROJECT_ROOT, require_files=True)

    def test_recorded_checksums_match_files(self, subsets):
        import hashlib

        for subset in subsets:
            path = PROJECT_ROOT / subset.local_file
            if not path.is_file():
                pytest.skip(f"raw file not present: {subset.local_file}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == subset.sha256, subset.region
