"""
tests/test_external_seaview.py — the Seaview Survey tabular product.

This is the first external product in the repository that carries a biological
quantity, and that makes it the first one that can be misread as a label.  Most
of what follows is not schema checking; it is the set of claims this dataset
must never be allowed to make quietly:

* the Central Indian Ocean surveys are Maldivian and Chagossian, not Indian;
* benthic cover is an image-derived estimate, not a diver's field note;
* the 26 repeat transects are a paired pre/post contrast, not a bleaching
  response and not a causal finding;
* the licence is CC BY 3.0 Unported, whatever the publisher's label says;
* neither project target exists here and neither may be manufactured.

Every test runs from the tracked manifest alone.  The raw tables are
git-ignored and absent on a clean checkout, so nothing here reads them and
nothing here reaches the network.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from src.external.provenance import (
    FORBIDDEN_BIOLOGICAL_CLAIMS,
    FORBIDDEN_TARGET_CLAIMS,
    assert_publishable,
    load_manifest,
    load_source,
    validate_manifest,
)
from tests.test_external_provenance import _code_string_literals, _imported_modules

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_DIR = PROJECT_ROOT / "data" / "external" / "metadata"
SEAVIEW_MANIFEST = METADATA_DIR / "seaview_survey.manifest.json"
CRW_MANIFEST = METADATA_DIR / "noaa_crw_5km_v3_1.manifest.json"
GEBCO_MANIFEST = METADATA_DIR / "gebco_2026.manifest.json"
FETCH_SCRIPT = PROJECT_ROOT / "scripts" / "fetch_seaview.py"

#: The four reef systems this project actually targets.  None is surveyed.
PROJECT_REEF_SYSTEMS = (
    "Lakshadweep",
    "Gulf of Mannar",
    "Gulf of Kutch",
    "Andaman and Nicobar Islands",
)

#: Counts published in the data descriptor (Sci Data 7, 355) and reproduced
#: exactly from the delivered tables at acquisition.  If a re-acquisition
#: changes any of these, the product changed and the manifest is stale.
EXPECTED_ROWS = {
    "seaviewsurvey_surveys.csv": 860,
    "seaviewsurvey_quadrats.csv": 1_082_324,
    "seaviewsurvey_labelsets.csv": 228,
    "seaviewsurvey_annotations.csv": 55_229_185,
    "seaviewsurvey_reefcover_indianocean.csv": 137_698,
}

#: SHA-256 of every file acquired at 2026-08-25, pinned here as well as in the
#: manifest.  The raw bytes are never edited — not to repair the Lobophyllia
#: export defect, not for anything — so a change to any of these digests means
#: either the publisher reissued the archive or something local rewrote a file
#: it had no business touching.  Either way it must fail loudly.
EXPECTED_SHA256 = {
    "seaviewsurvey_surveys.csv": "d465b622f9a4927b64f0597ec6acf080d7f257b31a577168e29f2351341619e5",
    "seaviewsurvey_quadrats.csv": "bc63b5bb7c205e7764582b7d3caab1d26732133f03531c361ba9f68a2add58dc",
    "seaviewsurvey_labelsets.csv": "7e30efc87adf3cf45dc923d24a0bed796200c000a024402baf4067ca411db80a",
    "seaviewsurvey_annotations.csv": "fa55813a15bdea1cc5be69adefaec0788d24a1ab2b8307bfa67fb86d1a4a8218",
    "seaviewsurvey_reefcover_indianocean.csv": (
        "fb122d9bc5ba85c742239ef51a26409eedee25faf73e9ddf2f96366d0439bade"
    ),
    "seaviewsurvey_reefcover_atlantic.csv": (
        "59a86980009d55faed61ae2b70a00f0ed4c41ed6edc2ef169b83d45fed26b9c4"
    ),
    "seaviewsurvey_reefcover_southeastasia.csv": (
        "b0d3f1f56c3efe37a9bc9e4729ff3fa284e6ff0fc71d8a748c18435048fdac72"
    ),
    "seaviewsurvey_reefcover_pacificaustralia.csv": (
        "a2955f6f0053909d870e6ba0d3372cddda3233cd59ba74f987db4051b271179c"
    ),
    "seaviewsurvey_reefcover_pacifichawaii.csv": (
        "acd96440fc32c0801dedadcbc31bed489df6df002c2c07f9241418f28c03f784"
    ),
    "annotations_ATL.csv": "3b50be40abb5969c2aa0154265a27899a40a2d4a7300d4728541b908499d7690",
    "annotations_IND_CHA.csv": "90a4038d6b146fd914bdc15cebc26edc277eb35c335d3d1880ddda886900756f",
    "annotations_IND_MDV.csv": "f5bf39b65209892abad1e4574551ad6d2c45abd661924783653799556de8510e",
    "annotations_PAC_AUS.csv": "9a0b6f876bbf96aad9936bfbca9bdb1b677af84f8ed4b523da0dec5aa39d0017",
    "annotations_PAC_IDN_PHL.csv": "97da27ec18cf2ef438dfbd97dc4735b498d51100a2587e2532aa0c12e79a2dfc",
    "annotations_PAC_SLB.csv": "31157728d5f2c5b73c4bae7a3a84b7bc330f32e13e9502f01352226e4cd38bbe",
    "annotations_PAC_TLS.csv": "880fe82883a9fac1c015c3e6c5c5f4dca4436b181760155084396008e2e72f20",
    "annotations_PAC_TWN.csv": "8617a56fb855ee9115f180891cd77b3fda781bd1788f8c5740d99db6f476f075",
    "annotations_PAC_USA.csv": "0b0ee6ff9e7efa060cd88e17f84b12c2efa8ae702f4a936809042d409a67bcab",
}

ARCHIVE_SHA256 = "2954427f0746cc2e3245955f62676066d0c2b94eab17ec87b50286ab0e5bf5c5"
DOCUMENTATION_SHA256 = "52dd6f1322460c90179180ba787ab69d41c40ef329d8e2c25b9b992b86e8adb0"

#: The publisher's own licence wording, kept verbatim, and our normalization of
#: it.  The label is imprecise — CC BY 3.0 exists as Unported and in ported
#: national forms, and "International" only starts at 4.0 — but the licence URI
#: the same record supplies is unambiguous, so the URI governs.
SOURCE_REPORTED_LICENCE_LABEL = "Creative Commons Attribution 3.0 International (CC BY 3.0)"
NORMALIZED_LICENCE = "CC BY 3.0 Unported"
LICENCE_URL = "https://creativecommons.org/licenses/by/3.0/"

#: Descriptions of the cover values that would overstate what they are.  98.33%
#: of them are CNN output; real imagery underneath does not make a prediction a
#: field measurement, and the published validation accuracy does not either.
BANNED_COVER_CLAIMS = (
    "directly observed coral cover",
    "field-measured coral cover",
    "measured coral cover",
    "observed coral cover",
    "biological ground truth",
    "direct field measurement",
    "observed benthic composition",
    "observed continuous biological response",
)

#: Wording that would turn a two-visit contrast into a causal claim.
BANNED_TEMPORAL_CLAIMS = ("bleaching response",)

#: Wording that would turn Maldives/Chagos surveys into Indian evidence.
BANNED_GEOGRAPHIC_CLAIMS = (
    "indian validation",
    "indian reef observations",
    "validation of lakshadweep",
    "validation of the gulf of mannar",
    "validation of gulf of mannar",
    "validation of the gulf of kutch",
    "validation of gulf of kutch",
    "validation of the andaman",
    "validation of andaman",
)

#: Manifest paths that are allowed to spell a banned phrase out bare, because
#: their whole purpose is to enumerate the phrases nobody may use.
TERMINOLOGY_DENYLIST_PATHS = (
    "variable_semantics.preferred_terminology.do_not_use",
    "target_compatibility.prohibited",
)

_DENIAL_MARKERS = ("not ", "never", "n't", "!=", "rather than", "instead of")


def _normalised(text: str) -> str:
    """Lowercase, collapse whitespace, drop emphasis markers and quote marks."""
    stripped = text.replace("*", "").replace("—", "--")
    for quote in ('"', "'", "“", "”"):
        stripped = stripped.replace(quote, "")
    return " ".join(stripped.lower().split())


def _denies(text: str) -> bool:
    """True if *text* reads as a refusal rather than an assertion."""
    lowered = _normalised(text)
    return any(marker in lowered for marker in _DENIAL_MARKERS)


def _manifest_strings(node: object, path: str = "") -> list[tuple[str, str]]:
    """Every string value in the manifest, paired with its dotted JSON path."""
    found: list[tuple[str, str]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            found.extend(_manifest_strings(value, f"{path}.{key}" if path else str(key)))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_manifest_strings(value, f"{path}[{index}]"))
    elif isinstance(node, str):
        found.append((path, node))
    return found


def _seaview_section() -> str:
    """The normalised text of §8 of ``docs/external_data.md`` and nothing else."""
    text = (PROJECT_ROOT / "docs" / "external_data.md").read_text(encoding="utf-8")
    start = text.index("## 8. Seaview Survey")
    end = text.index("\n## 9.", start)
    return _normalised(text[start:end])


def _paragraphs(doc: Path) -> list[str]:
    """
    Split a document into blank-line-separated blocks.

    The paragraph, not the line, is the unit a claim is read in: a markdown
    table's disclaimer lives in its header row, and a sentence wraps across
    several lines. Checking line by line would flag honest prose and teach the
    next person to weaken the assertion.
    """
    return [block for block in doc.read_text(encoding="utf-8").split("\n\n") if block.strip()]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def payload() -> dict:
    assert SEAVIEW_MANIFEST.is_file(), f"Seaview manifest missing at {SEAVIEW_MANIFEST}"
    return json.loads(SEAVIEW_MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def source():
    return load_source(SEAVIEW_MANIFEST)


@pytest.fixture(scope="module")
def subsets():
    return load_manifest(SEAVIEW_MANIFEST)


# ---------------------------------------------------------------------------
# Manifest structure
# ---------------------------------------------------------------------------


class TestManifestSchema:
    def test_manifest_validates(self, source, subsets):
        validate_manifest(source, subsets, project_root=PROJECT_ROOT, require_files=False)

    def test_schema_and_id(self, payload):
        assert payload["schema"] == "coralsense.external.manifest/v1"
        assert payload["dataset_id"] == "seaview_survey"

    def test_every_table_has_a_subset_record(self, subsets):
        regions = {s.region for s in subsets}
        assert len(subsets) == 18, f"expected 18 tables, got {len(subsets)}"
        for name in EXPECTED_ROWS:
            assert name in regions, f"{name} has no subset record"

    def test_row_counts_match_the_published_dataset(self, payload):
        per_table = payload["diagnostics"]["per_table"]
        for name, expected in EXPECTED_ROWS.items():
            assert per_table[name]["n_rows"] == expected, (
                f"{name}: manifest records {per_table[name]['n_rows']} rows, "
                f"published/acquired value is {expected}"
            )

    def test_subset_dimensions_agree_with_diagnostics(self, payload, subsets):
        per_table = payload["diagnostics"]["per_table"]
        for subset in subsets:
            assert subset.dimensions["rows"] == per_table[subset.region]["n_rows"]
            assert subset.dimensions["columns"] == per_table[subset.region]["n_columns"]


# ---------------------------------------------------------------------------
# Real, not synthetic
# ---------------------------------------------------------------------------


class TestRealDataFlag:
    def test_source_is_not_synthetic(self, source):
        assert source.is_synthetic is False

    def test_no_subset_is_synthetic(self, subsets):
        for subset in subsets:
            assert subset.is_synthetic is False, subset.region

    def test_observation_type_is_derived_not_measured(self, source):
        """
        The photograph is the measurement; the cover proportion is a model output.

        Recording this as ``measured`` would be the single most consequential
        overclaim available in this manifest, so it is pinned.
        """
        assert source.observation_type == "derived"

    def test_not_joined_to_the_synthetic_benchmark(self, payload):
        prohibited = " ".join(payload["target_compatibility"]["prohibited"]).lower()
        assert "observations.csv" in prohibited


# ---------------------------------------------------------------------------
# Licence
# ---------------------------------------------------------------------------


class TestLicence:
    def test_licence_is_verified_and_named(self, source):
        assert source.licence_verified is True
        assert "CC BY 3.0" in source.licence_name

    def test_redistribution_is_explicitly_allowed(self, source):
        assert source.redistribution_allowed is True

    def test_product_is_publishable_under_its_licence(self, source):
        """CC BY 3.0 permits redistribution, so the gate must let this through."""
        assert_publishable(source)

    def test_raw_files_are_still_not_tracked_in_git(self, source):
        """
        Redistribution permission is a licence fact; tracking is a storage decision.

        The licence would allow us to commit 341 MB of CSV. We do not.
        """
        assert source.raw_tracked_in_git is False

    def test_licence_was_read_from_the_authoritative_dataset_record(self, source):
        via = _normalised(source.licence_verified_via)
        assert "espace" in via and "uq:734799" in via

    def test_licence_is_not_claimed_to_rest_on_the_doi_metadata(self, source):
        """
        Crossref registers this DOI with no licence field at all.

        A resolving DOI is not a licence statement, and the record has to say so
        rather than leaving a reader to assume the DOI settled it.
        """
        via = _normalised(source.licence_verified_via)
        assert "crossref" in via
        assert "no licence field" in via

    def test_the_version_wording_discrepancy_is_recorded(self, source):
        """
        eSpace labels it 'CC BY 3.0 International'. No such licence exists.

        3.0 came in Unported and ported forms; 'International' begins at 4.0.
        The discrepancy does not change the grant, but it must not be silently
        tidied into a version we were not actually given.
        """
        via = _normalised(source.licence_verified_via)
        assert "unported" in via
        assert "4.0" in via

    def test_citation_names_both_dois(self, source):
        assert "10.14264/uql.2019.930" in source.citation
        assert "10.1038/s41597-020-00698-6" in source.citation


# ---------------------------------------------------------------------------
# Geographic truthfulness — the central claim of this module
# ---------------------------------------------------------------------------


class TestGeographicTruthfulness:
    def test_transfer_status_is_declared(self, payload):
        block = payload["geographic_transfer_status"]
        assert block["status"] == "INDIAN_OCEAN_NOT_INDIA"

    def test_headline_states_indian_ocean_is_not_india(self, payload):
        headline = _normalised(payload["geographic_transfer_status"]["headline"])
        assert "indian ocean is not india" in headline

    def test_territories_are_maldives_and_chagos_only(self, payload):
        territories = payload["geographic_transfer_status"]["territories_surveyed"]
        assert set(territories) == {"MDV", "CHA"}
        assert "Maldives" in territories["MDV"]
        assert "Chagos" in territories["CHA"]

    def test_survey_counts_by_territory(self, payload):
        counts = payload["geographic_transfer_status"]["surveys_by_territory"]
        assert counts == {"MDV": 63, "CHA": 29}
        assert sum(counts.values()) == 92, "the paper reports 92 Central Indian Ocean surveys"

    def test_no_project_reef_system_is_covered(self, payload):
        block = payload["geographic_transfer_status"]
        assert block["project_target_regions_covered"] == []
        not_covered = block["project_target_regions_not_covered"]
        for system in PROJECT_REEF_SYSTEMS:
            assert system in not_covered, f"{system} must be listed as not covered"
            assert "0 surveys" in not_covered[system]

    @pytest.mark.parametrize("system", PROJECT_REEF_SYSTEMS)
    def test_no_validation_is_claimed_for_any_indian_reef(self, payload, system):
        """
        Every Indian reef system must appear only inside a denial.

        A future edit that adds a Lakshadweep coverage claim has to delete one of
        these strings to pass, which makes it a visible decision rather than a
        drifting sentence.
        """
        consequence = _normalised(payload["geographic_transfer_status"]["consequence"])
        assert "does not validate" in consequence
        assert _normalised(system) in consequence

    def test_the_ind_ocean_code_collision_is_documented(self, payload):
        """
        'IND' is the Indian Ocean basin code here AND the ISO code for India.

        The human-annotation files are literally named annotations_IND_MDV.csv.
        Anyone who reads that as 'India, Maldives' has misread the dataset, so
        the trap is written down.
        """
        warning = _normalised(payload["geographic_transfer_status"]["ocean_code_warning"])
        assert "iso 3166" in warning
        assert "country == ind" in warning or "country == 'ind'" in warning

    def test_coordinate_envelope_is_south_of_every_indian_reef(self, payload):
        """The northernmost Seaview transect sits below Lakshadweep's south edge."""
        envelope = payload["geographic_transfer_status"]["coordinate_envelope"]
        assert envelope["lat_max"] < 8.0, (
            "Lakshadweep begins near 8 degrees N; a Seaview latitude above that "
            "would mean the no-overlap finding is wrong"
        )
        assert 71.0 < envelope["lng_min"] < envelope["lng_max"] < 74.0

    def test_geographic_scope_prose_denies_indian_waters(self, source):
        scope = _normalised(source.geographic_scope)
        assert "maldives" in scope and "chagos" in scope
        assert "no indian territorial waters" in scope

    def test_cannot_provide_names_the_geographic_limit(self, source):
        joined = _normalised(" ".join(source.cannot_provide))
        assert "indian territorial waters" in joined


# ---------------------------------------------------------------------------
# Cover is a classifier output, not ground truth
# ---------------------------------------------------------------------------


class TestCoverIsNotGroundTruth:
    def test_disclaimer_states_values_are_classifier_output(self, source):
        text = _normalised(source.disclaimer)
        assert "image classifier" in text or "classifier" in text
        assert "not a direct field observation" in text

    def test_machine_and_human_shares_are_quantified(self, payload):
        share = payload["variable_semantics"]["how_values_were_produced"][
            "human_share_indian_ocean"
        ]
        assert share["quadrats_total"] == 137_698
        assert share["quadrats_with_human_annotations"] == 2_298
        assert share["machine_percent"] > 98.0
        assert abs(share["human_percent"] + share["machine_percent"] - 100.0) < 0.05

    def test_the_network_architecture_is_named(self, payload):
        method = _normalised(payload["variable_semantics"]["how_values_were_produced"]["method"])
        assert "vgg" in method
        assert "50 points" in method

    def test_ground_truth_qualification_is_required_in_writing(self, payload):
        """
        Every occurrence of 'ground truth' in this manifest must be a denial.

        Substring matching cannot tell a claim from its refusal, so the assertion
        is positive: the qualification sentence has to be present. This mirrors
        the approach already used for the CRW bleaching disclaimer.
        """
        qualification = _normalised(
            payload["variable_semantics"]["how_values_were_produced"]["required_qualification"]
        )
        assert "must not be called ground truth" in qualification

    def test_published_validation_accuracy_is_recorded(self, payload):
        validation = payload["variable_semantics"]["how_values_were_produced"][
            "published_validation"
        ]
        assert "97%" in validation

    def test_units_are_proportions_not_percentages(self, source):
        for name, unit in source.variable_units.items():
            assert "proportion" in unit, f"{name} recorded as {unit!r}"


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


class TestNoTargetIsClaimed:
    def test_provides_variables_claims_no_forbidden_name(self, source):
        banned = FORBIDDEN_TARGET_CLAIMS + FORBIDDEN_BIOLOGICAL_CLAIMS
        for claim in source.provides_variables:
            for name in banned:
                assert name not in claim.lower(), f"{claim!r} claims {name}"

    def test_hard_coral_cover_is_offered_as_a_response_variable(self, source):
        """The point of acquiring this product at all."""
        assert "hard_coral_cover" in source.provides_variables

    @pytest.mark.parametrize("target", FORBIDDEN_TARGET_CLAIMS)
    def test_both_project_targets_are_listed_as_unavailable(self, source, target):
        assert target in source.cannot_provide

    def test_reef_health_verdict_is_partial_evidence_only(self, payload):
        verdict = payload["target_compatibility"]["reef_health"]["verdict"]
        assert "PARTIAL" in verdict
        assert "NOT A TARGET" in verdict

    def test_restoration_suitability_verdict_is_no_direct_target(self, payload):
        verdict = payload["target_compatibility"]["restoration_suitability"]["verdict"]
        assert "NO DIRECT TARGET" in verdict

    def test_the_inequalities_are_stated_explicitly(self, payload):
        warning = _normalised(payload["variable_semantics"]["warning"])
        assert "hard_coral_cover != reef_health" in warning
        assert "algal_cover != reef_health" in warning
        assert "restoration_suitability" in warning

    def test_leakage_reasoning_cites_the_dataset_audit(self, payload):
        reasoning = _normalised(payload["target_compatibility"]["reef_health"]["reasoning"])
        assert "2026-08-19" in reasoning
        assert "leakage" in reasoning

    def test_prohibited_list_bans_deriving_either_target(self, payload):
        prohibited = _normalised(" ".join(payload["target_compatibility"]["prohibited"]))
        assert "deriving reef_health" in prohibited
        assert "deriving restoration_suitability" in prohibited

    def test_legitimate_use_does_not_disturb_the_registered_models(self, payload):
        text = _normalised(payload["target_compatibility"]["legitimate_use"])
        assert "does not change" in text and "retrain" in text


# ---------------------------------------------------------------------------
# Observational units
# ---------------------------------------------------------------------------


class TestObservationalUnits:
    def test_the_four_units_are_distinguished(self, payload):
        units = payload["observational_units"]["units"]
        assert set(units) == {"survey", "image", "quadrat", "annotation_point"}

    def test_hierarchy_is_recorded_in_order(self, payload):
        hierarchy = payload["observational_units"]["hierarchy"]
        assert hierarchy == "survey -> image -> quadrat -> annotation point"

    def test_quadrats_per_image_shows_a_genuine_one_to_many(self, payload):
        counts = payload["observational_units"]["cardinality_indian_ocean"]["quadrats_per_image"]
        assert counts["1"] > 0
        assert max(int(k) for k in counts) > 1, "no one-to-many structure recorded"
        assert sum(counts.values()) == 81_773, "must account for every Indian Ocean image"

    def test_quadrat_totals_reconcile_with_the_cover_table(self, payload):
        counts = payload["observational_units"]["cardinality_indian_ocean"]["quadrats_per_image"]
        total = sum(int(k) * v for k, v in counts.items())
        assert total == EXPECTED_ROWS["seaviewsurvey_reefcover_indianocean.csv"]

    def test_non_independence_is_warned_about(self, payload):
        warning = _normalised(payload["observational_units"]["independence_warning"])
        assert "not independent geographic locations" in warning
        assert "92 transects" in warning

    def test_aggregation_rule_comes_from_the_source_paper(self, payload):
        rule = _normalised(payload["observational_units"]["aggregation_rule"])
        assert "sci data 7, 355" in rule
        for step in ("quadrat", "image", "survey"):
            assert step in rule

    def test_aggregation_rule_was_verified_against_published_values(self, payload):
        rule = payload["observational_units"]["aggregation_rule"]
        assert "0.0025" in rule, "the reproduction check must record its residual"

    def test_primary_and_foreign_keys_recorded_for_every_table(self, payload):
        for name, entry in payload["diagnostics"]["per_table"].items():
            assert entry["primary_key"], f"{name} has no primary key recorded"
            assert entry["foreign_keys"], f"{name} has no foreign keys recorded"
            assert entry["observational_unit"], f"{name} has no observational unit"


# ---------------------------------------------------------------------------
# Known defects — recorded, never repaired
# ---------------------------------------------------------------------------


class TestKnownDataDefects:
    def test_the_missing_indian_ocean_label_is_recorded(self, payload):
        defect = payload["known_data_defects"]["missing_label_column_indian_ocean"]
        assert "MASE_MEA_L" in defect["what"]
        assert "MASE_LRG_I" in defect["what"]

    def test_the_defect_carries_its_evidence(self, payload):
        evidence = payload["known_data_defects"]["missing_label_column_indian_ocean"]["evidence"]
        assert "135,185" in evidence and "2,513" in evidence

    def test_the_defect_was_not_silently_repaired(self, payload):
        handling = _normalised(
            payload["known_data_defects"]["missing_label_column_indian_ocean"]["handling"]
        )
        assert "recorded, not repaired" in handling
        assert "nothing has been imputed" in handling

    def test_no_outliers_were_removed(self, payload):
        text = _normalised(payload["known_data_defects"]["no_outliers_removed"])
        assert "no outlier removal" in text


# ---------------------------------------------------------------------------
# Acquisition scope
# ---------------------------------------------------------------------------


class TestAcquisitionScope:
    def test_the_image_archive_was_not_acquired(self, payload):
        paths = payload["not_acquired"]["paths"]
        assert "photo-quadrats/" in paths
        assert "annotated-images/" in paths
        assert "survey-previews/" in paths

    def test_the_archive_hash_and_size_are_pinned(self, payload):
        archive = payload["acquired_archive"]
        assert archive["filename"] == "tabular-data.zip"
        assert re.fullmatch(r"[0-9a-f]{64}", archive["sha256"])
        assert archive["file_size_bytes"] == 357_966_014

    def test_documentation_pdf_is_recorded(self, payload):
        doc = payload["acquired_documentation"]
        assert doc["filename"].endswith(".pdf")
        assert re.fullmatch(r"[0-9a-f]{64}", doc["sha256"])

    def test_the_fetch_script_refuses_image_paths_in_code(self):
        """
        The refusal must live in executable strings, not only in the docstring.

        ``_code_string_literals`` excludes docstrings precisely so that prose
        cannot satisfy a guard assertion.
        """
        literals = _code_string_literals(FETCH_SCRIPT)
        for path in ("photo-quadrats/", "annotated-images/", "survey-previews/"):
            assert path in literals, f"{path} is not a code-level constant"


# ---------------------------------------------------------------------------
# Checksums
# ---------------------------------------------------------------------------


class TestChecksumSemantics:
    def test_checksum_semantics_block_is_present(self, payload):
        block = payload["checksum_semantics"]
        assert block["algorithm"] == "SHA-256"
        assert block["is_publisher_canonical_checksum"] is False

    def test_it_states_what_the_hash_does_not_guarantee(self, payload):
        text = _normalised(payload["checksum_semantics"]["does_not_guarantee"])
        assert "the published file changed" in text

    def test_identity_is_pinned_by_doi_not_by_hash(self, payload):
        text = payload["checksum_semantics"]["authoritative_identity"]
        assert "10.14264/uql.2019.930" in text

    def test_the_plain_http_transport_caveat_is_recorded(self, payload):
        """
        The publisher does not answer on port 443.

        That is a real integrity weakness, so it is stated next to the hashes
        rather than left for someone to rediscover.
        """
        text = _normalised(payload["checksum_semantics"]["transport_caveat"])
        assert "http" in text and "not authenticated" in text

    def test_every_subset_hash_is_well_formed_and_distinct(self, subsets):
        hashes = [s.sha256 for s in subsets]
        for value in hashes:
            assert re.fullmatch(r"[0-9a-f]{64}", value)
        assert len(set(hashes)) == len(hashes), "two tables share a checksum"


# ---------------------------------------------------------------------------
# Repository safety and clean-checkout behaviour
# ---------------------------------------------------------------------------


class TestRepositorySafety:
    def test_raw_seaview_directory_is_git_ignored(self):
        result = subprocess.run(
            ["git", "check-ignore", "-q", "data/external/raw/seaview/"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, "raw Seaview data is not git-ignored"

    def test_the_archive_itself_is_git_ignored(self):
        result = subprocess.run(
            ["git", "check-ignore", "-q", "data/external/raw/seaview/tabular-data.zip"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, "tabular-data.zip is not git-ignored"

    def test_seaview_manifest_is_not_git_ignored(self):
        result = subprocess.run(
            ["git", "check-ignore", "-q", "data/external/metadata/seaview_survey.manifest.json"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0, "the Seaview manifest must not be git-ignored"

    def test_no_home_directory_leaked(self):
        text = SEAVIEW_MANIFEST.read_text(encoding="utf-8")
        assert "/home/" not in text
        assert not re.search(r"[A-Za-z]:\\\\", text)

    def test_local_paths_are_relative_and_under_external_raw(self, subsets):
        for subset in subsets:
            assert not subset.local_file.startswith("/"), subset.region
            assert subset.local_file.startswith("data/external/raw/seaview/"), subset.region

    def test_manifest_validates_without_the_raw_files(self, source, subsets):
        """
        Clean-checkout behaviour: the raw tables are absent and that is fine.

        ``require_files=False`` is the checkout case; this asserts the manifest
        carries everything needed to describe the product without them.
        """
        validate_manifest(source, subsets, project_root=PROJECT_ROOT, require_files=False)

    def test_no_test_here_reads_the_raw_tables(self):
        literals = _code_string_literals(Path(__file__))
        offending = [s for s in literals if s.endswith(".csv") and "reefcover" in s]
        for name in offending:
            assert "data/external/raw" not in name


class TestNoNetworkAccess:
    def test_this_test_module_performs_no_network_access(self):
        imported = set(_imported_modules(Path(__file__)))
        for name in ("urllib", "urllib.request", "requests", "httpx", "socket"):
            assert name not in imported, f"test module imports {name}"

    def test_the_fetch_script_only_networks_from_an_explicit_command(self):
        """
        Importing the fetch script must not download anything.

        The network calls live inside ``fetch_file``, which only ``main`` calls,
        and ``main`` only runs under ``__main__``.
        """
        text = FETCH_SCRIPT.read_text(encoding="utf-8")
        assert 'if __name__ == "__main__":' in text
        module_level = [
            line
            for line in text.splitlines()
            if line and not line[0].isspace() and "urlopen(" in line
        ]
        assert not module_level, f"module-level network call: {module_level}"


# ---------------------------------------------------------------------------
# The prose has to agree with the manifest
# ---------------------------------------------------------------------------

DOC_FILES = (
    PROJECT_ROOT / "docs" / "external_data.md",
    PROJECT_ROOT / "data" / "external" / "README.md",
)


@pytest.mark.parametrize("doc", DOC_FILES, ids=lambda p: p.name)
class TestDocumentation:
    """
    A manifest nobody reads is not a safeguard.

    These assertions are positive — the denial must be *present* — because
    substring matching cannot distinguish a claim from its refusal, and every
    mention of India in these documents is a refusal.
    """

    def test_documents_state_indian_ocean_is_not_india(self, doc):
        text = _normalised(doc.read_text(encoding="utf-8"))
        assert "indian ocean is not india" in text

    def test_documents_name_maldives_and_chagos(self, doc):
        text = _normalised(doc.read_text(encoding="utf-8"))
        assert "maldives" in text and "chagos" in text

    def test_documents_warn_about_the_ind_code_collision(self, doc):
        text = _normalised(doc.read_text(encoding="utf-8"))
        assert "annotations_ind_mdv.csv" in text

    def test_documents_deny_indian_validation(self, doc):
        text = _normalised(doc.read_text(encoding="utf-8"))
        assert "does not validate any model for" in text

    def test_documents_state_cover_is_not_ground_truth(self, doc):
        text = _normalised(doc.read_text(encoding="utf-8"))
        assert "hard_coral_cover != reef_health" in text

    def test_documents_record_the_machine_share(self, doc):
        text = _normalised(doc.read_text(encoding="utf-8"))
        assert "98.33" in text


# ---------------------------------------------------------------------------
# The other two products must be undisturbed
# ---------------------------------------------------------------------------


class TestExistingProductsStillValid:
    @pytest.mark.parametrize("manifest", [GEBCO_MANIFEST, CRW_MANIFEST])
    def test_earlier_manifest_still_validates(self, manifest):
        validate_manifest(
            load_source(manifest),
            load_manifest(manifest),
            project_root=PROJECT_ROOT,
        )

    def test_the_three_products_are_separate(self):
        ids = {
            load_source(path).dataset_id
            for path in (GEBCO_MANIFEST, CRW_MANIFEST, SEAVIEW_MANIFEST)
        }
        assert len(ids) == 3, f"dataset_ids collide: {ids}"

    def test_seaview_is_the_only_product_supplying_benthic_cover(self, source):
        """
        GEBCO is terrain and CRW is thermal; neither may acquire a cover variable.

        The test is on ``cover``, not on ``coral``: CRW legitimately supplies
        ``coral_bleaching_hotspot_c``, a thermal quantity that merely has the
        word in its name. Banning the word would flag an honest variable and
        teach the next person to weaken the check.
        """
        assert "hard_coral_cover" in source.provides_variables
        for path in (GEBCO_MANIFEST, CRW_MANIFEST):
            other = load_source(path)
            offending = [v for v in other.provides_variables if "cover" in v.lower()]
            assert not offending, f"{other.dataset_id} claims cover variable(s) {offending}"


# ---------------------------------------------------------------------------
# Licence normalization — publisher wording and our reading of it, side by side
# ---------------------------------------------------------------------------


class TestLicenceNormalization:
    """
    UQ's label and UQ's licence URI do not describe the same thing.

    The label says "Creative Commons Attribution 3.0 International". No such
    licence was ever issued: 3.0 shipped as Unported plus ported national forms,
    and "International" begins at 4.0. The URI in the same record resolves to
    CC BY 3.0 Unported and is unambiguous, so the URI governs — but UQ did write
    that label, and normalizing it away entirely would delete evidence.
    """

    def test_canonical_licence_uri_is_the_3_0_one(self, source, payload):
        assert source.licence_url == LICENCE_URL
        assert payload["licence_normalization"]["licence_url"] == LICENCE_URL
        assert "/by/3.0/" in source.licence_url
        assert "4.0" not in source.licence_url

    def test_normalized_licence_is_cc_by_3_0_unported(self, source, payload):
        assert source.licence_name == NORMALIZED_LICENCE
        assert payload["licence_normalization"]["normalized_licence"] == NORMALIZED_LICENCE
        assert "unported" in _normalised(source.licence_name)

    def test_publishers_reported_label_is_preserved_verbatim(self, payload):
        """UQ used this wording. The record has to keep saying that it did."""
        block = payload["licence_normalization"]
        assert block["source_reported_licence_label"] == SOURCE_REPORTED_LICENCE_LABEL
        assert block["publisher_wording_preserved"] is True
        assert "espace" in _normalised(block["source_reported_by"])

    def test_the_label_is_called_imprecise_and_the_uri_unambiguous(self, payload):
        basis = _normalised(payload["licence_normalization"]["normalization_basis"])
        assert "imprecise" in basis
        assert "unported" in basis
        assert "unambiguous" in basis
        assert "uri governs" in basis

    def test_version_granted_is_3_0_and_4_0_is_explicitly_not_granted(self, payload):
        block = payload["licence_normalization"]
        assert block["version_granted"] == "3.0"
        assert "4.0" in block["version_not_granted"]
        assert _denies(block["version_not_granted"])

    def test_nothing_in_the_manifest_calls_this_cc_by_4_0(self, payload):
        """
        The strongest available invariant, so it is a flat prohibition.

        Any occurrence of the string at all — even inside a denial — fails, which
        keeps the check unambiguous. The denial is phrased as "version 4.0 was
        not granted" instead.
        """
        for path, text in _manifest_strings(payload):
            assert "cc by 4.0" not in text.lower(), f"{path} describes the licence as CC BY 4.0"

    def test_verification_and_redistribution_survive_the_normalization(self, source, payload):
        block = payload["licence_normalization"]
        assert source.licence_verified is True
        assert source.redistribution_allowed is True
        assert block["licence_verified"] is True
        assert block["redistribution_allowed"] is True

    def test_the_discrepancy_is_recorded_as_wording_not_permission(self, payload):
        effect = _normalised(payload["licence_normalization"]["effect_of_the_discrepancy"])
        assert "redistribution" in effect
        assert "attribution" in effect

    @pytest.mark.parametrize("doc", DOC_FILES, ids=lambda p: p.name)
    def test_documents_carry_the_normalized_licence(self, doc):
        text = _normalised(doc.read_text(encoding="utf-8"))
        assert "cc by 3.0 unported" in text

    def test_the_long_form_document_keeps_both_wordings(self):
        """
        Scoped to §8 on purpose.

        Elsewhere the document legitimately says 'CC BY 4.0' about other,
        unacquired sources — HICORDIS, ReefNet. Banning the string document-wide
        would flag an honest sentence about a different dataset.
        """
        section = _seaview_section()
        assert _normalised(SOURCE_REPORTED_LICENCE_LABEL) in section
        assert "cc by 3.0 unported" in section
        assert "cc by 4.0" not in section


# ---------------------------------------------------------------------------
# Cover terminology — A, B and C are three different things
# ---------------------------------------------------------------------------


class TestCoverTerminology:
    """
    Real imagery does not make CNN output a direct field measurement.

    The survey imagery and coordinates are a real field record (A); the human
    annotations are expert judgements about photographs (B); the cover columns
    this project would consume are model output (C). Every assertion here exists
    to stop those three being collapsed into "observed coral cover".
    """

    def test_the_three_evidence_layers_are_distinguished(self, payload):
        layers = payload["variable_semantics"]["evidence_layers"]
        assert set(layers) == {
            "A_field_survey_imagery_and_coordinates",
            "B_human_image_annotations",
            "C_ml_classified_benthic_cover",
            "why_the_distinction_matters",
        }

    def test_layer_a_is_credited_as_a_real_field_record(self, payload):
        text = _normalised(
            payload["variable_semantics"]["evidence_layers"][
                "A_field_survey_imagery_and_coordinates"
            ]
        )
        assert "real field record" in text
        assert "coordinates" in text

    def test_layer_b_is_an_image_annotation_not_an_in_water_measurement(self, payload):
        text = _normalised(
            payload["variable_semantics"]["evidence_layers"]["B_human_image_annotations"]
        )
        assert "1.67%" in text.replace(" %", "%")
        assert "in-water measurement" in text
        assert _denies(text)

    def test_layer_c_is_declared_model_output(self, payload):
        text = _normalised(
            payload["variable_semantics"]["evidence_layers"]["C_ml_classified_benthic_cover"]
        )
        assert "model output" in text
        assert "98.33%" in text.replace(" %", "%")
        assert "estimates, not measurements" in text

    def test_the_preferred_terminology_is_written_down(self, payload):
        block = payload["variable_semantics"]["preferred_terminology"]
        assert "image-derived benthic-cover estimate" in block["use"]
        assert "ML-estimated benthic cover" in block["use"]
        assert "image-derived biological response estimate" in block["use"]
        for phrase in (
            "directly observed coral cover",
            "field-measured coral cover",
            "measured coral cover",
            "biological ground truth",
        ):
            assert phrase in block["do_not_use"]

    def test_validation_accuracy_does_not_promote_predictions(self, payload):
        produced = payload["variable_semantics"]["how_values_were_produced"]
        assert "97%" in produced["published_validation"], "published validation must be preserved"
        promoted = _normalised(produced["validation_does_not_promote_predictions"])
        assert "still an estimate" in promoted
        assert "ground truth" in promoted
        assert _denies(promoted)

    def test_the_disclaimer_uses_the_precise_term(self, source):
        text = _normalised(source.disclaimer)
        assert "image-derived benthic-cover estimate" in text
        assert "not a direct field observation" in text

    @pytest.mark.parametrize("phrase", BANNED_COVER_CLAIMS)
    def test_no_manifest_string_asserts_a_direct_field_measurement(self, payload, phrase):
        for path, text in _manifest_strings(payload):
            if phrase not in _normalised(text):
                continue
            if path.startswith(TERMINOLOGY_DENYLIST_PATHS):
                continue
            assert _denies(text), f"{path} claims {phrase!r}: {text!r}"

    @pytest.mark.parametrize("doc", DOC_FILES, ids=lambda p: p.name)
    @pytest.mark.parametrize("phrase", BANNED_COVER_CLAIMS)
    def test_no_document_paragraph_asserts_a_direct_field_measurement(self, doc, phrase):
        for block in _paragraphs(doc):
            if phrase in _normalised(block):
                assert _denies(block), f"{doc.name} claims {phrase!r} in:\n{block}"

    @pytest.mark.parametrize("doc", DOC_FILES, ids=lambda p: p.name)
    def test_documents_name_the_three_layers(self, doc):
        text = _normalised(doc.read_text(encoding="utf-8"))
        assert "image-derived" in text
        assert "human image annotation" in text or "human image annotations" in text
        assert "model output" in text


# ---------------------------------------------------------------------------
# The response variable, and the upstream defect that decides which one to use
# ---------------------------------------------------------------------------


class TestBiologicalResponseClassification:
    def test_neither_project_target_is_manufactured(self, payload):
        targets = payload["target_compatibility"]
        assert (
            targets["reef_health"]["verdict"] == "PARTIAL BIOLOGICAL EVIDENCE ONLY — NOT A TARGET"
        )
        assert targets["restoration_suitability"]["verdict"] == "NO DIRECT TARGET"

    def test_reef_health_reasoning_no_longer_calls_cover_measured(self, payload):
        reasoning = _normalised(payload["target_compatibility"]["reef_health"]["reasoning"])
        assert "image-derived benthic-composition estimates" in reasoning
        assert "image-derived hard-coral cover is an estimate" in reasoning

    def test_the_legitimate_use_is_an_estimate_not_an_observation(self, payload):
        text = _normalised(payload["target_compatibility"]["legitimate_use"])
        assert "image-derived continuous biological response estimate" in text
        assert "observed continuous biological response" not in text

    def test_the_recommended_first_response_field_is_survey_level(self, payload):
        block = payload["target_compatibility"]["recommended_response_field"]
        assert block["field"] == "pr_hard_coral"
        assert block["table"] == "seaviewsurvey_surveys.csv"
        assert block["unit"].startswith("survey")
        assert "recommendation only" in _normalised(block["status"])
        why = _normalised(block["why"])
        assert "defective" in why
        assert "92 transects" in why

    @pytest.mark.parametrize("doc", DOC_FILES, ids=lambda p: p.name)
    def test_no_document_promises_either_target(self, doc):
        text = _normalised(doc.read_text(encoding="utf-8"))
        assert "hard_coral_cover != reef_health" in text
        assert "restoration_suitability" in text


class TestUpstreamIndianOceanTableDefect:
    """
    The published quadrat table drops Lobophyllia and carries an all-zero
    Southeast-Asian label in its place. Recorded exactly; never repaired.
    """

    def test_the_defect_is_stated_with_both_label_codes(self, payload):
        what = payload["known_data_defects"]["missing_label_column_indian_ocean"]["what"]
        assert "MASE_MEA_L" in what and "Lobophyllia" in what
        assert "MASE_LRG_I" in what and "Isopora" in what
        assert "Southeast Asia" in what
        assert "all zeros" in what

    def test_the_affected_quadrat_count_is_pinned(self, payload):
        defect = payload["known_data_defects"]["missing_label_column_indian_ocean"]
        assert "2,513" in defect["evidence"]
        assert "2,513" in defect["impact"]
        assert "under-reported" in _normalised(defect["impact"])

    def test_nothing_was_repaired_imputed_rescaled_or_dropped(self, payload):
        handling = _normalised(
            payload["known_data_defects"]["missing_label_column_indian_ocean"]["handling"]
        )
        assert "recorded, not repaired" in handling
        assert "not been edited" in handling
        assert "imputed, dropped or rescaled" in handling
        assert "raw bytes are unchanged" in handling

    def test_the_first_analysis_is_pointed_at_the_survey_level_field(self, payload):
        text = _normalised(
            payload["known_data_defects"]["missing_label_column_indian_ocean"][
                "recommendation_for_first_analysis"
            ]
        )
        assert "pr_hard_coral" in text
        assert "seaviewsurvey_surveys.csv" in text
        assert "not performed here" in text

    def test_the_raw_hashes_are_unchanged(self, payload, subsets):
        """
        Recording a defect must never become quietly fixing it.

        These digests are the evidence that the CSV on disk is still the byte
        sequence the publisher served, defect included.
        """
        by_region = {s.region: s.sha256 for s in subsets}
        for name, digest in EXPECTED_SHA256.items():
            assert by_region[name] == digest, f"{name}: raw bytes changed since acquisition"
        assert payload["acquired_archive"]["sha256"] == ARCHIVE_SHA256
        assert payload["acquired_documentation"]["sha256"] == DOCUMENTATION_SHA256

    @pytest.mark.parametrize("doc", DOC_FILES, ids=lambda p: p.name)
    def test_the_long_form_document_records_the_defect(self, doc):
        if doc.name != "external_data.md":
            pytest.skip("the summary README defers the defect detail to docs/external_data.md")
        text = _normalised(doc.read_text(encoding="utf-8"))
        assert "mase_mea_l" in text
        assert "recorded, not repaired" in text
        assert "pr_hard_coral" in text


# ---------------------------------------------------------------------------
# Repeat transects — a temporal contrast, not a cause
# ---------------------------------------------------------------------------


class TestRepeatTransectLanguage:
    def test_the_paired_design_is_described_by_size_and_interval(self, payload):
        block = payload["temporal_design"]
        assert block["repeat_transects_maldives"] == 26
        assert block["repeat_transects_chagos"] == 0
        assert block["revisit_interval_days"] == {"min": 703, "max": 722}
        assert (
            block["visited_once"] + block["visited_twice"]
            == (block["distinct_indian_ocean_transects"])
        )
        assert block["visited_three_or_more_times"] == 0

    def test_it_is_called_a_paired_pre_post_change(self, payload):
        text = _normalised(payload["temporal_design"]["what_this_supports"])
        assert "paired pre/post survey change" in text
        assert "2016 mass-bleaching period" in text
        assert "real temporal contrast" in text

    def test_it_is_explicitly_not_a_bleaching_response(self, payload):
        text = _normalised(payload["temporal_design"]["what_this_is_not"])
        assert "not a bleaching response" in text
        assert "does not by itself demonstrate that bleaching caused the change" in text

    def test_crw_exposure_is_named_as_future_work_and_not_yet_joined(self, payload):
        text = _normalised(payload["temporal_design"]["thermal_exposure_not_linked"])
        assert "hotspot" in text and "degree heating week" in text
        assert "site and date" in text
        assert "has not been performed" in text
        assert "no join exists" in text

    def test_even_after_linkage_the_design_stays_observational(self, payload):
        text = _normalised(payload["temporal_design"]["causal_status_after_linkage"])
        assert "observational" in text
        assert "association" in text
        assert "does not license automatic causal attribution" in text

    def test_the_required_wording_is_recorded(self, payload):
        text = _normalised(payload["temporal_design"]["required_wording"])
        assert "paired pre/post survey change around the 2016 mass-bleaching period" in text
        assert "never as bleaching response" in text

    @pytest.mark.parametrize("phrase", BANNED_TEMPORAL_CLAIMS)
    def test_no_manifest_string_claims_a_bleaching_response(self, payload, phrase):
        for path, text in _manifest_strings(payload):
            if phrase not in _normalised(text):
                continue
            if path.startswith(TERMINOLOGY_DENYLIST_PATHS):
                continue
            assert _denies(text), f"{path} claims {phrase!r}: {text!r}"

    @pytest.mark.parametrize("doc", DOC_FILES, ids=lambda p: p.name)
    @pytest.mark.parametrize("phrase", BANNED_TEMPORAL_CLAIMS)
    def test_no_document_paragraph_claims_a_bleaching_response(self, doc, phrase):
        for block in _paragraphs(doc):
            if phrase in _normalised(block):
                assert _denies(block), f"{doc.name} claims {phrase!r} in:\n{block}"

    @pytest.mark.parametrize("doc", DOC_FILES, ids=lambda p: p.name)
    def test_documents_use_the_non_causal_wording(self, doc):
        text = _normalised(doc.read_text(encoding="utf-8"))
        assert "paired pre/post survey change" in text
        assert "2016 mass-bleaching period" in text


# ---------------------------------------------------------------------------
# Geography, once more, as a denylist rather than a positive claim
# ---------------------------------------------------------------------------


class TestNoIndianEvidenceIsClaimed:
    @pytest.mark.parametrize("phrase", BANNED_GEOGRAPHIC_CLAIMS)
    def test_no_manifest_string_claims_indian_evidence(self, payload, phrase):
        for path, text in _manifest_strings(payload):
            if phrase not in _normalised(text):
                continue
            if path.startswith(TERMINOLOGY_DENYLIST_PATHS):
                continue
            assert _denies(text), f"{path} claims {phrase!r}: {text!r}"

    @pytest.mark.parametrize("doc", DOC_FILES, ids=lambda p: p.name)
    @pytest.mark.parametrize("phrase", BANNED_GEOGRAPHIC_CLAIMS)
    def test_no_document_paragraph_claims_indian_evidence(self, doc, phrase):
        for block in _paragraphs(doc):
            if phrase in _normalised(block):
                assert _denies(block), f"{doc.name} claims {phrase!r} in:\n{block}"

    def test_the_transfer_status_still_says_indian_ocean_not_india(self, payload):
        assert payload["geographic_transfer_status"]["status"] == "INDIAN_OCEAN_NOT_INDIA"
        counts = payload["geographic_transfer_status"]["surveys_by_territory"]
        assert counts == {"MDV": 63, "CHA": 29}
        assert "IND" not in counts, "India has no surveys and must not acquire a count"
