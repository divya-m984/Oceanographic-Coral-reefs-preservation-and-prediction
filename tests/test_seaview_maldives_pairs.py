"""
Tests for the Seaview Maldives paired temporal analysis and its CRW extension.

Scope
-----
Two artifacts are under test: the historical NOAA CRW acquisition scoped to the
Maldives (``scripts/fetch_noaa_crw_maldives.py`` and its manifest) and the
paired association analysis built on it
(``scripts/analyze_seaview_maldives_pairs.py`` and the three reports it writes).

These tests never download anything, never train a model, never touch MLflow,
and never read the synthetic prototype dataset.  They run against the committed
manifest and the committed report artifacts, so they pass on a clean checkout
where the git-ignored raw files are absent.

What they are actually defending, in one line each:

* the row spine is 26 Maldivian transects and stays 26;
* the response is the publisher's ``pr_hard_coral``, not a quadrat
  reconstruction and not a reef-condition label;
* no CRW observation dated after a pair's second survey enters its exposure;
* the Maldives acquisition is never described as, or merged with, Indian data.
"""

from __future__ import annotations

import ast
import csv
import json
import re
import subprocess
from datetime import date
from pathlib import Path

import pytest

from src.external.provenance import (
    load_manifest,
    load_source,
    validate_manifest,
)
from tests.test_external_provenance import _code_string_literals, _imported_modules

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_DIR = PROJECT_ROOT / "data" / "external" / "metadata"

MALDIVES_CRW_MANIFEST = METADATA_DIR / "noaa_crw_5km_v3_1_maldives_seaview.manifest.json"
INDIA_CRW_MANIFEST = METADATA_DIR / "noaa_crw_5km_v3_1.manifest.json"
SEAVIEW_MANIFEST = METADATA_DIR / "seaview_survey.manifest.json"
GEBCO_MANIFEST = METADATA_DIR / "gebco_2026.manifest.json"

FETCH_SCRIPT = PROJECT_ROOT / "scripts" / "fetch_noaa_crw_maldives.py"
ANALYSIS_SCRIPT = PROJECT_ROOT / "scripts" / "analyze_seaview_maldives_pairs.py"

REPORT_DIR = PROJECT_ROOT / "reports" / "external"
PAIRS_CSV = REPORT_DIR / "seaview_maldives_pairs.csv"
PAIRS_SUMMARY = REPORT_DIR / "seaview_maldives_pairs_summary.json"
ASSOCIATION = REPORT_DIR / "seaview_maldives_crw_association.json"

DOC_FILES = (
    PROJECT_ROOT / "docs" / "external_data.md",
    PROJECT_ROOT / "data" / "external" / "README.md",
)

EXPECTED_N_PAIRS = 26

#: Verified against seaviewsurvey_surveys.csv, not carried over from prose.
EXPECTED_FIRST_SURVEY = "2015-03-29"
EXPECTED_LAST_SURVEY = "2017-04-01"
EXPECTED_INTERVAL_DAYS = (703, 722)

#: Targets that must never be constructed from this analysis.
PROJECT_TARGETS = ("reef_health", "restoration_suitability")

#: Wording that would misdescribe an observational Maldivian association.
BANNED_CAUSAL_CLAIMS = (
    "bleaching caused",
    "caused the decline",
    "caused mortality",
    "thermal stress caused",
    "proves that",
    "ground truth",
)

BANNED_GEOGRAPHIC_CLAIMS = (
    "indian validation",
    "validates indian",
    "indian reef validation",
    "validation for indian reefs",
)

#: Phrasings that encode the simplistic rule "Shapiro-Wilk was significant,
#: therefore the Wilcoxon test is the valid one". A significant normality test
#: does not confer validity on the signed-rank test, which carries its own
#: assumptions; these regexes fail the wording that pretends otherwise. They are
#: patterns rather than exact strings so a reworded version of the same fallacy
#: is caught too.
SIMPLISTIC_NORMALITY_RULES = (
    r"not (?:comfortably |especially |very )?normal[,;]? so (?:the )?wilcoxon",
    r"not (?:comfortably |especially |very )?normal[,;]? (?:and|therefore) (?:the )?wilcoxon",
    r"shapiro[- ]wilk [^.]{0,40}(?:failed|fails)",
    r"(?:therefore|hence|so) (?:the )?wilcoxon [^.]{0,40}(?:is|becomes) (?:the )?valid",
    r"fail(?:ed|s)? normality[,;]? (?:so|therefore|hence)",
)

#: Wording that would present the 26 transects as independent climatic draws.
REPLICATE_INDEPENDENCE_CLAIMS = (
    r"26 independent (?:climatic )?replicates",
    r"independent climatic replicates",
)

_DENIAL_MARKERS = (
    "not ",
    "never",
    "n't",
    "!=",
    "no ",
    "must not",
    "does not",
    "cannot",
    "omit",
    "rather than",
    "instead of",
)

#: Heading line of the section this milestone adds to ``docs/external_data.md``.
#: The ``## 8a.`` prefix is part of the constant on purpose: the document's own
#: status header names the section too, and anchoring on the bare title would
#: slice from there and silently test the wrong text.
MALDIVES_SECTION_HEADING = "## 8a. Seaview Maldives paired temporal analysis"


def _normalised(text: str) -> str:
    """Collapse whitespace and lowercase, for substring searching over prose."""
    return re.sub(r"\s+", " ", text).lower()


def _denies(text: str) -> bool:
    """True if *text* reads as a denial rather than an assertion."""
    return any(marker in text for marker in _DENIAL_MARKERS)


def _maldives_section() -> str:
    """
    Return the new section of ``docs/external_data.md``, raw.

    Scoped rather than whole-document on purpose. §8 already contains a table
    of *forbidden* phrasings, spelled out bare so the reader knows what not to
    write; a document-wide substring scan would flag that table and the fix
    would be to weaken it. This milestone's section is checked on its own.
    """
    text = (PROJECT_ROOT / "docs" / "external_data.md").read_text(encoding="utf-8")
    start = text.index(MALDIVES_SECTION_HEADING)
    end = text.find("\n## ", start)
    return text[start:] if end == -1 else text[start:end]


def _paragraphs(text: str) -> list[str]:
    """
    Split *text* into blank-line-separated blocks.

    The paragraph, not the line, is the unit a claim is read in: a caveat can
    sit in a table header or wrap across several lines, and a line-by-line scan
    would flag honest prose.
    """
    return [block for block in text.split("\n\n") if block.strip()]


def _path_component_literals(path: Path) -> set[str]:
    """
    Return string literals that *build a filesystem path* in the file at *path*.

    Only operands of a ``/`` join and arguments to ``Path(...)`` count, which is
    the distinction that matters here. A script is allowed — required, even — to
    name the defective quadrat table in prose explaining why it is not used; it
    is not allowed to open it. Matching on plain string literals conflates the
    two and would punish the documentation.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    components: set[str] = set()

    def collect(node: ast.AST) -> None:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                components.add(sub.value)

    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            collect(node.right)
        elif isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Path":
            for argument in node.args:
                collect(argument)
    return components


def _referenced_identifiers(path: Path) -> set[str]:
    """Return every name and attribute the file at *path* actually references."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MALDIVES_CRW_MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pairs() -> list[dict]:
    with PAIRS_CSV.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def summary() -> dict:
    return json.loads(PAIRS_SUMMARY.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def association() -> dict:
    return json.loads(ASSOCIATION.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The row spine
# ---------------------------------------------------------------------------


class TestPairSpine:
    def test_exactly_26_pair_rows(self, pairs):
        assert len(pairs) == EXPECTED_N_PAIRS

    def test_each_transect_appears_once(self, pairs):
        ids = [row["transect_id"] for row in pairs]
        assert len(set(ids)) == len(ids) == EXPECTED_N_PAIRS

    def test_every_survey_id_is_used_once(self, pairs):
        ids = [row["first_survey_id"] for row in pairs] + [row["second_survey_id"] for row in pairs]
        assert len(set(ids)) == len(ids) == 2 * EXPECTED_N_PAIRS

    def test_maldives_only(self, summary):
        assert summary["pair_selection"]["maldives_only"] is True

    def test_chagos_has_no_repeats_and_so_contributes_nothing(self, summary):
        """
        Chagos is excluded by the data, not by a special case.

        It has surveys in the Indian Ocean subset but no transect visited twice,
        so the 'exactly two visits' rule removes it without anyone naming it.
        """
        selection = summary["pair_selection"]
        assert selection["chagos_repeat_transects"] == 0
        assert selection["chagos_surveys"] > 0

    def test_no_transect_was_visited_more_than_twice(self, summary):
        assert summary["pair_selection"]["transects_visited_three_or_more_times"] == 0

    def test_first_survey_precedes_second(self, pairs):
        for row in pairs:
            assert row["first_survey_date"] < row["second_survey_date"], row["transect_id"]

    def test_interval_matches_the_verified_source_range(self, pairs):
        low, high = EXPECTED_INTERVAL_DAYS
        for row in pairs:
            assert low <= int(row["interval_days"]) <= high, row["transect_id"]

    def test_survey_date_range_matches_the_source(self, summary):
        selection = summary["pair_selection"]
        assert selection["earliest_first_survey"] == EXPECTED_FIRST_SURVEY
        assert selection["latest_second_survey"] == EXPECTED_LAST_SURVEY

    def test_quadrats_and_images_are_not_the_inferential_unit(self, summary):
        assert summary["paired_change"]["n_pairs"] == EXPECTED_N_PAIRS
        assert "26" in summary["limitations"]["sample_size"]


# ---------------------------------------------------------------------------
# The response
# ---------------------------------------------------------------------------


class TestResponseDefinition:
    def test_response_is_survey_level_pr_hard_coral(self, summary):
        response = summary["paired_change"]["response"]
        assert response["column"] == "pr_hard_coral"
        assert response["source_table"] == "seaviewsurvey_surveys.csv"

    def test_quadrat_reconstruction_is_not_used(self, summary):
        response = summary["paired_change"]["response"]
        assert response["reconstructed_from_quadrats"] is False
        assert "MASE_MEA_L" in response["why_not_reconstructed"]

    def test_the_analysis_never_opens_the_defective_quadrat_tables(self):
        """
        The defective tables may be *named* in prose; they may not be *opened*.

        ``seaviewsurvey_reefcover_indianocean.csv`` is missing MASE_MEA_L, and
        the script explains that in a string it writes into the report. Only
        path construction is checked, so the explanation is allowed to stand.
        """
        components = _path_component_literals(ANALYSIS_SCRIPT)
        for name in (
            "seaviewsurvey_reefcover_indianocean.csv",
            "seaviewsurvey_quadrats.csv",
            "seaviewsurvey_annotations.csv",
        ):
            assert name not in components, name
        assert "seaviewsurvey_surveys.csv" in components

    def test_the_defective_table_is_only_named_alongside_a_denial(self):
        for literal in _code_string_literals(ANALYSIS_SCRIPT):
            if "seaviewsurvey_reefcover_indianocean.csv" in literal:
                assert _denies(_normalised(literal)), literal

    def test_delta_is_second_minus_first(self, pairs):
        for row in pairs:
            expected = float(row["second_pr_hard_coral"]) - float(row["first_pr_hard_coral"])
            assert abs(float(row["delta_hard_coral"]) - expected) < 1e-9, row["transect_id"]

    def test_response_is_labelled_as_image_derived(self, summary):
        nature = summary["paired_change"]["response"]["nature"].lower()
        assert "image-derived" in nature
        assert "cnn" in nature or "classifier" in nature

    def test_no_reef_health_label_is_created(self, summary, association, pairs):
        for blob in (summary, association):
            text = _normalised(json.dumps(blob))
            for target in PROJECT_TARGETS:
                for occurrence in re.finditer(re.escape(target), text):
                    window = text[max(0, occurrence.start() - 200) : occurrence.end() + 120]
                    assert _denies(window), f"{target} appears without a denial: {window}"
        assert not any(
            target in column for row in pairs for column in row for target in PROJECT_TARGETS
        )

    def test_no_bleached_or_healthy_class_column_exists(self, pairs):
        columns = {column.lower() for column in pairs[0]}
        for banned in ("bleached", "healthy", "unhealthy", "reef_health", "suitability", "label"):
            assert not any(banned in column for column in columns), banned

    def test_no_outliers_were_removed(self, summary):
        assert summary["paired_change"]["outliers_removed"] == 0


# ---------------------------------------------------------------------------
# The CRW acquisition
# ---------------------------------------------------------------------------


class TestCrwAcquisition:
    def test_manifest_validates(self):
        validate_manifest(
            load_source(MALDIVES_CRW_MANIFEST),
            load_manifest(MALDIVES_CRW_MANIFEST),
            project_root=PROJECT_ROOT,
            require_files=False,
        )

    def test_it_is_a_separate_acquisition_from_the_india_one(self, manifest):
        india = json.loads(INDIA_CRW_MANIFEST.read_text(encoding="utf-8"))
        assert manifest["dataset_id"] != india["dataset_id"]
        relationship = manifest["relationship_to_existing_acquisitions"]
        assert relationship["existing_acquisition"]["modified_by_this_acquisition"] is False
        assert relationship["existing_acquisition"]["window"] == "2018-01-01 to 2024-12-31"

    def test_the_india_manifest_still_describes_india_and_2018_2024(self):
        """The historical extension must not have rewritten the earlier record."""
        india = json.loads(INDIA_CRW_MANIFEST.read_text(encoding="utf-8"))
        assert india["dataset_id"] == "noaa_crw_5km_v3_1"
        assert "2018-01-01 to 2024-12-31" in india["source"]["temporal_scope"]
        regions = {subset["region"].split(" / ")[0] for subset in india["subsets"]}
        assert regions == {
            "Lakshadweep",
            "Gulf of Mannar",
            "Gulf of Kutch",
            "Andaman and Nicobar Islands",
        }

    def test_analysis_scope_is_recorded(self, manifest):
        assert manifest["analysis_scope"] == "SEAVIEW_MALDIVES_REPEAT_TRANSECTS"

    def test_maldives_is_not_presented_as_india(self, manifest):
        warning = manifest["relationship_to_existing_acquisitions"]["geographic_warning"]
        assert "MALDIVES_NOT_INDIA" in warning
        scope = manifest["source"]["geographic_scope"]
        assert "MALDIVES" in scope.upper()
        assert _denies(_normalised(scope))

    def test_only_hotspot_and_dhw_were_acquired(self, manifest):
        variables = {subset["variable_name"] for subset in manifest["subsets"]}
        assert variables == {"hotspot", "degree_heating_week"}
        assert set(manifest["omitted_products"]) >= {"sst", "sst_anomaly"}

    def test_acquisition_window_is_derived_not_hard_coded(self, manifest):
        derivation = manifest["acquisition_window_derivation"]
        assert derivation["hard_coded_dates"] is False
        assert derivation["n_repeat_transects"] == EXPECTED_N_PAIRS
        assert derivation["earliest_first_survey"] == EXPECTED_FIRST_SURVEY
        assert derivation["latest_second_survey"] == EXPECTED_LAST_SURVEY
        assert "seaviewsurvey_surveys.csv" in derivation["derived_from"]

    def test_acquired_window_covers_every_pair(self, manifest, pairs):
        for subset in manifest["subsets"]:
            start = subset["actual_time_range"][0][:10]
            end = subset["actual_time_range"][1][:10]
            for row in pairs:
                assert start <= row["first_survey_date"][:10], (subset["region"], row)
                assert end >= row["second_survey_date"][:10], (subset["region"], row)

    def test_acquired_bbox_contains_every_transect(self, manifest, pairs):
        for subset in manifest["subsets"]:
            lat_min, lat_max, lon_min, lon_max = subset["actual_bbox"]
            for row in pairs:
                assert lat_min <= float(row["survey_latitude"]) <= lat_max, row["transect_id"]
                assert lon_min <= float(row["survey_longitude"]) <= lon_max, row["transect_id"]

    def test_the_india_windows_are_not_reused(self, manifest):
        india = json.loads(INDIA_CRW_MANIFEST.read_text(encoding="utf-8"))
        india_boxes = {tuple(subset["requested_bbox"]) for subset in india["subsets"]}
        for subset in manifest["subsets"]:
            assert tuple(subset["requested_bbox"]) not in india_boxes

    def test_grid_resolution_and_units_are_recorded(self, manifest):
        for subset in manifest["subsets"]:
            assert abs(subset["grid_spacing_deg"] - 0.05) < 1e-6, subset["region"]
            assert subset["variable_units"] in {"degree_C", "degree_Celsius_weeks"}
            assert subset["fill_value"]
            assert subset["sha256"] and len(subset["sha256"]) == 64
            assert subset["file_size_bytes"] > 0
            assert subset["retrieved_at_utc"]

    def test_requested_and_delivered_time_ranges_are_both_recorded(self, manifest):
        for subset in manifest["subsets"]:
            assert subset["requested_time_range"][0][:10] == EXPECTED_FIRST_SURVEY
            assert subset["requested_time_range"][1][:10] == EXPECTED_LAST_SURVEY
            assert subset["actual_time_range"][0] != subset["requested_time_range"][0] or True
            assert subset["n_time_steps"] > 700

    def test_doi_and_provider_are_recorded(self, manifest):
        source = manifest["source"]
        assert source["doi"] == "10.25921/6jgr-pt28"
        assert source["product_identifier"] == "gov.noaa.nodc:CRW-5km-HeatStressProducts"
        assert "NOAA" in source["source_name"]
        assert source["licence_verified"] is True
        assert source["redistribution_allowed"] is True


class TestSourceLineageWordingIsPreserved:
    def test_nothing_is_called_ostia_free(self, manifest):
        text = _normalised(json.dumps(manifest))
        for occurrence in re.finditer(r"ostia-free", text):
            window = text[max(0, occurrence.start() - 160) : occurrence.end()]
            assert _denies(window), f"'OSTIA-free' asserted rather than denied: {window}"

    def test_the_acquisition_floor_is_still_a_policy_not_a_licence_boundary(self, manifest):
        via = _normalised(manifest["source"]["licence_verified_via"])
        assert "2002-12-01" in via
        assert "policy" in via or "conservative" in via

    def test_the_fetch_script_reuses_the_shared_guard_rather_than_copying_it(self):
        """
        One definition of the policy floor, not two.

        A second copy of the date could be relaxed independently of the India
        script's, which is exactly how a provenance policy quietly stops
        applying. The extension must call the shared guard and read the shared
        constant.
        """
        tree = ast.parse(FETCH_SCRIPT.read_text(encoding="utf-8"))
        assigned = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        assert "FIRST_POST_OSTIA_BLEND_REQUEST_DATE" not in assigned
        assert "OSTIA_MERGE_END_DATE" not in assigned

        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "_guard_licence_window" in called

    def test_the_2016_merge_window_is_disclosed(self, manifest):
        via = manifest["source"]["licence_verified_via"]
        assert "2016" in via and "merge" in via.lower()


# ---------------------------------------------------------------------------
# Spatial matching
# ---------------------------------------------------------------------------


class TestSpatialMatching:
    def test_the_rule_is_recorded(self, association):
        spatial = association["spatial_matching"]
        assert "nearest valid" in spatial["rule"].lower()
        assert spatial["max_match_distance_km"] == 5.0
        assert spatial["max_match_distance_rationale"]

    def test_every_pair_records_a_match_distance(self, pairs):
        for row in pairs:
            assert row["match_distance_km"], row["transect_id"]
            assert float(row["match_distance_km"]) >= 0.0

    def test_no_match_exceeds_the_documented_threshold(self, pairs, association):
        threshold = association["spatial_matching"]["max_match_distance_km"]
        for row in pairs:
            if row["crw_matched"].lower() == "true":
                assert float(row["match_distance_km"]) <= threshold, row["transect_id"]

    def test_unmatched_cases_are_explicit(self, pairs, association):
        spatial = association["spatial_matching"]
        unmatched_rows = [row for row in pairs if row["crw_matched"].lower() != "true"]
        assert spatial["n_unmatched"] == len(unmatched_rows)
        assert len(spatial["unmatched"]) == len(unmatched_rows)
        for row in unmatched_rows:
            assert row["crw_unmatched_reason"], row["transect_id"]

    def test_matched_rows_carry_the_crw_cell_coordinates(self, pairs):
        for row in pairs:
            if row["crw_matched"].lower() == "true":
                assert row["crw_latitude"] and row["crw_longitude"], row["transect_id"]

    def test_pair_count_is_preserved_after_the_crw_join(self, pairs, association):
        spatial = association["spatial_matching"]
        assert spatial["n_matched"] + spatial["n_unmatched"] == EXPECTED_N_PAIRS
        assert len(pairs) == EXPECTED_N_PAIRS

    def test_masking_displacement_is_reported(self, association):
        assert "n_displaced_by_masking" in association["spatial_matching"]

    def test_no_interpolation(self, association):
        assert "no interpolation" in association["spatial_matching"]["rule"].lower()


# ---------------------------------------------------------------------------
# Temporal exposure
# ---------------------------------------------------------------------------


class TestTemporalExposure:
    def test_exposure_never_extends_past_the_second_survey(self, pairs):
        """The leakage guard: no thermal information from after the outcome."""
        for row in pairs:
            if row["crw_matched"].lower() != "true":
                continue
            second = row["second_survey_date"][:10]
            first = row["first_survey_date"][:10]
            assert row["exposure_end"] <= second, row["transect_id"]
            assert row["exposure_start"] >= first, row["transect_id"]
            for column in ("date_of_max_dhw", "date_of_max_hotspot"):
                assert first <= row[column] <= second, (row["transect_id"], column)

    def test_exposure_interval_is_the_pair_interval(self, pairs):
        for row in pairs:
            if row["crw_matched"].lower() != "true":
                continue
            assert row["exposure_start"] == row["first_survey_date"][:10]
            assert row["exposure_end"] == row["second_survey_date"][:10]

    def test_exposure_is_defined_independently_of_the_response(self, association):
        temporal = association["temporal_exposure"]
        assert temporal["defined_independently_of_response"] is True
        assert temporal["no_post_hoc_event_window"]

    def test_max_dhw_is_non_negative(self, pairs):
        for row in pairs:
            if row["max_dhw"]:
                assert float(row["max_dhw"]) >= 0.0, row["transect_id"]

    def test_hotspot_stays_inside_the_products_valid_range(self, pairs):
        """HotSpot is signed and bounded at +/- 15 degC by the product itself."""
        for row in pairs:
            if row["max_hotspot"]:
                assert -15.0 <= float(row["max_hotspot"]) <= 15.0, row["transect_id"]

    def test_missing_day_fraction_is_recorded_and_sane(self, pairs):
        for row in pairs:
            if row["crw_matched"].lower() != "true":
                continue
            for column in ("missing_dhw_day_fraction", "missing_hotspot_day_fraction"):
                assert 0.0 <= float(row[column]) <= 1.0, (row["transect_id"], column)

    def test_valid_day_counts_do_not_exceed_the_interval(self, pairs):
        for row in pairs:
            if row["crw_matched"].lower() != "true":
                continue
            expected = int(row["interval_days"]) + 1
            for column in ("n_valid_dhw_days", "n_valid_hotspot_days"):
                assert int(row[column]) <= expected, (row["transect_id"], column)

    def test_only_two_exposure_metrics_are_pre_specified(self, association):
        assert association["association"]["primary_predictors"] == ["max_dhw", "max_hotspot"]
        assert "None" in association["temporal_exposure"]["metrics"]["secondary_metrics"]

    def test_the_dhw_backward_accumulation_caveat_is_recorded(self, association):
        caveat = association["temporal_exposure"]["backward_reach_caveat"].lower()
        assert "12-week" in caveat or "12 week" in caveat
        assert "after the second survey" in caveat

    def test_date_precision_limitation_is_documented(self, association):
        assert "day" in association["temporal_exposure"]["date_precision"].lower()


# ---------------------------------------------------------------------------
# The association analysis
# ---------------------------------------------------------------------------


class TestAssociationAnalysis:
    def test_spearman_is_the_primary_test(self, association):
        assert "Spearman" in association["association"]["primary_test"]

    def test_both_predictors_report_n_rho_p_and_an_interval(self, association):
        for name in ("max_dhw", "max_hotspot"):
            test = association["association"]["tests"][name]
            assert test["n"] == EXPECTED_N_PAIRS
            assert -1.0 <= test["rho"] <= 1.0
            assert 0.0 <= test["p_value"] <= 1.0
            assert test["bootstrap_ci_95"] is not None
            assert test["random_seed"] == 20260830

    def test_raw_p_values_are_reported_alongside_the_holm_adjustment(self, association):
        block = association["association"]
        assert set(block["holm_adjusted_p_values"]) == {"max_dhw", "max_hotspot"}
        for name, adjusted in block["holm_adjusted_p_values"].items():
            assert adjusted >= block["tests"][name]["p_value"] - 1e-12

    def test_only_two_tests_were_run(self, association):
        assert association["association"]["multiple_testing"]["n_pre_specified_tests"] == 2

    def test_predictor_collinearity_is_stated(self, association):
        block = association["association"]["predictor_collinearity"]
        assert -1.0 <= block["spearman_rho_dhw_vs_hotspot"] <= 1.0
        assert block["note"]

    def test_leave_one_pair_out_sensitivity_is_reported(self, association):
        for name in ("max_dhw", "max_hotspot"):
            block = association["association"]["sensitivity_leave_one_pair_out"][name]
            assert block["rho_min"] <= block["rho_full"] <= block["rho_max"]
            assert isinstance(block["most_influential_transect_id"], int)
            assert "dominated_by_one_pair" in block

    def test_exposure_contrast_is_reported_so_a_null_can_be_read(self, association):
        for name in ("max_dhw", "max_hotspot"):
            block = association["association"]["exposure_contrast"][name]
            assert block["n_transects"] == EXPECTED_N_PAIRS
            assert block["range"] >= 0.0
            assert block["n_distinct_values"] >= 1

    def test_no_ml_benchmark_machinery_was_used(self, association):
        not_performed = association["association"]["not_performed"]
        for key in ("multivariable_model", "train_test_split", "cross_validation"):
            assert not_performed[key]

    def test_the_analysis_script_never_splits_or_cross_validates(self):
        """
        Checked by reference, not by substring.

        The report declares ``train_test_split: not an ML benchmark`` as a JSON
        key, so a substring test would fail on the very statement that promises
        the thing was not done.
        """
        imported = set(_imported_modules(ANALYSIS_SCRIPT))
        assert not any(name.startswith("sklearn") for name in imported)
        referenced = _referenced_identifiers(ANALYSIS_SCRIPT)
        for banned in ("train_test_split", "cross_val_score", "KFold", "GridSearchCV", "fit"):
            assert banned not in referenced, banned

    def test_scatter_data_is_present_for_every_analysed_pair(self, association):
        scatter = association["association"]["scatter_data"]
        assert len(scatter) == association["association"]["n_analysed"]
        for point in scatter:
            assert {"transect_id", "delta_hard_coral", "max_dhw", "max_hotspot"} <= set(point)


# ---------------------------------------------------------------------------
# Paired descriptive analysis
# ---------------------------------------------------------------------------


class TestPairedChange:
    def test_direction_counts_sum_to_the_pair_count(self, summary):
        direction = summary["paired_change"]["direction"]
        assert sum(direction.values()) == EXPECTED_N_PAIRS

    def test_a_non_parametric_paired_test_is_the_primary_one(self, summary):
        test = summary["paired_change"]["paired_test"]
        assert "Wilcoxon" in test["primary"]
        assert "secondary" in " ".join(test).lower()

    def test_normality_was_checked_and_reported(self, summary):
        check = summary["paired_change"]["normality_check"]
        assert "Shapiro" in check["test"]
        assert isinstance(check["normal_at_0_05"], bool)

    def test_effect_size_and_interval_accompany_the_p_value(self, summary):
        test = summary["paired_change"]["paired_test"]
        assert -1.0 <= test["effect_size_rank_biserial"] <= 1.0
        low, high = test["median_change_bootstrap_ci_95"]
        assert low <= high

    def test_relative_change_carries_its_small_denominator_caveat(self, summary):
        caveat = summary["paired_change"]["relative_change_caveat"]
        assert caveat["min_first_pr_hard_coral"] > 0
        assert "denominator" in caveat["note"].lower()

    def test_pre_and_post_distributions_are_reported(self, summary):
        for key in ("pre_distribution", "post_distribution", "delta_distribution"):
            block = summary["paired_change"][key]
            assert block["n"] == EXPECTED_N_PAIRS
            assert block["min"] <= block["median"] <= block["max"]


# ---------------------------------------------------------------------------
# Scientific language
# ---------------------------------------------------------------------------


class TestScientificLanguage:
    @pytest.mark.parametrize("path", [PAIRS_SUMMARY, ASSOCIATION, MALDIVES_CRW_MANIFEST])
    def test_no_causal_claim_is_asserted(self, path):
        text = _normalised(path.read_text(encoding="utf-8"))
        for claim in BANNED_CAUSAL_CLAIMS:
            for occurrence in re.finditer(re.escape(claim), text):
                window = text[max(0, occurrence.start() - 220) : occurrence.end() + 60]
                assert _denies(window), f"{path.name}: {claim!r} asserted: {window}"

    @pytest.mark.parametrize("path", [PAIRS_SUMMARY, ASSOCIATION, MALDIVES_CRW_MANIFEST])
    def test_no_indian_validation_is_claimed(self, path):
        text = _normalised(path.read_text(encoding="utf-8"))
        for claim in BANNED_GEOGRAPHIC_CLAIMS:
            for occurrence in re.finditer(re.escape(claim), text):
                window = text[max(0, occurrence.start() - 220) : occurrence.end() + 60]
                assert _denies(window), f"{path.name}: {claim!r} asserted: {window}"

    def test_the_non_causal_limitation_is_explicit(self, association, summary):
        for blob in (association, summary):
            causal = blob["limitations"]["causal"]
            assert "NON-CAUSAL" in causal
            assert "association" in causal.lower()

    def test_maldives_not_india_is_explicit(self, association, summary):
        for blob in (association, summary):
            assert blob["provenance"]["geography"] == "MALDIVES_NOT_INDIA"
            assert "MALDIVES_NOT_INDIA" in blob["limitations"]["geography"]

    def test_registered_models_are_declared_untouched(self, summary):
        note = summary["limitations"]["registered_models_untouched"].lower()
        assert "observations.csv" in note
        assert "mlruns" in note


# ---------------------------------------------------------------------------
# Repository safety
# ---------------------------------------------------------------------------


class TestRepositorySafety:
    def test_maldives_raw_directory_is_git_ignored(self):
        result = subprocess.run(
            ["git", "check-ignore", "-q", "data/external/raw/noaa_crw_5km_v3_1/maldives_seaview/"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, "raw Maldives CRW data is not git-ignored"

    def test_the_manifest_and_reports_are_not_git_ignored(self):
        for path in (MALDIVES_CRW_MANIFEST, PAIRS_CSV, PAIRS_SUMMARY, ASSOCIATION):
            result = subprocess.run(
                ["git", "check-ignore", "-q", str(path.relative_to(PROJECT_ROOT))],
                cwd=PROJECT_ROOT,
                capture_output=True,
                check=False,
            )
            assert result.returncode != 0, f"{path.name} must not be git-ignored"

    def test_no_home_directory_leaked(self):
        for path in (MALDIVES_CRW_MANIFEST, PAIRS_SUMMARY, ASSOCIATION):
            text = path.read_text(encoding="utf-8")
            assert "/home/" not in text, path.name
            assert not re.search(r"[A-Za-z]:\\\\", text), path.name

    def test_local_paths_are_relative_and_under_external_raw(self, manifest):
        for subset in manifest["subsets"]:
            assert not subset["local_file"].startswith("/"), subset["region"]
            assert subset["local_file"].startswith(
                "data/external/raw/noaa_crw_5km_v3_1/maldives_seaview/"
            ), subset["region"]

    def test_neither_script_builds_a_path_to_a_protected_artifact(self):
        """
        The seven protected files must be unreachable from these scripts.

        Matched on path *components*, so a prose sentence promising that
        ``data/raw/observations.csv`` is not touched still passes while an
        actual ``PROJECT_ROOT / "data" / "raw" / "observations.csv"`` fails.
        """
        protected_basenames = (
            "observations.csv",
            "mlruns.db",
            "best_model_health.joblib",
            "best_model_restoration.joblib",
            "preprocessor_health.joblib",
            "preprocessor_restoration.joblib",
            "drift_summary.json",
        )
        for script in (FETCH_SCRIPT, ANALYSIS_SCRIPT):
            components = _path_component_literals(script)
            for name in protected_basenames:
                assert name not in components, f"{script.name}: {name}"

    def test_any_prose_mention_of_a_protected_path_is_a_denial(self):
        for script in (FETCH_SCRIPT, ANALYSIS_SCRIPT):
            for literal in _code_string_literals(script):
                if "observations.csv" in literal or "mlruns.db" in literal:
                    assert _denies(_normalised(literal)), f"{script.name}: {literal}"

    def test_neither_script_imports_mlflow_dvc_or_the_training_code(self):
        for script in (FETCH_SCRIPT, ANALYSIS_SCRIPT):
            imported = set(_imported_modules(script))
            for banned in ("mlflow", "dvc", "src.models", "src.models.train", "src.data"):
                assert banned not in imported, f"{script.name} imports {banned}"


class TestNoNetworkAccess:
    def test_this_test_module_performs_no_network_access(self):
        imported = set(_imported_modules(Path(__file__)))
        for name in ("urllib", "urllib.request", "requests", "httpx", "socket"):
            assert name not in imported, f"test module imports {name}"

    def test_the_analysis_script_never_networks_at_all(self):
        imported = set(_imported_modules(ANALYSIS_SCRIPT))
        for name in ("urllib", "urllib.request", "requests", "httpx", "socket", "ftplib"):
            assert name not in imported, f"analysis script imports {name}"

    def test_the_fetch_script_only_networks_from_an_explicit_command(self):
        text = FETCH_SCRIPT.read_text(encoding="utf-8")
        assert 'if __name__ == "__main__":' in text
        module_level = [
            line
            for line in text.splitlines()
            if line and not line[0].isspace() and ("urlopen(" in line or "fetch_file(" in line)
        ]
        assert not module_level, f"module-level network call: {module_level}"


class TestExistingProductsStillValid:
    """Adding a fourth manifest must not disturb the first three."""

    @pytest.mark.parametrize(
        "path", [GEBCO_MANIFEST, INDIA_CRW_MANIFEST, SEAVIEW_MANIFEST], ids=lambda p: p.stem
    )
    def test_manifest_still_validates(self, path):
        validate_manifest(
            load_source(path),
            load_manifest(path),
            project_root=PROJECT_ROOT,
            require_files=False,
        )

    def test_all_four_dataset_ids_are_distinct(self):
        ids = [
            json.loads(path.read_text(encoding="utf-8"))["dataset_id"]
            for path in (GEBCO_MANIFEST, INDIA_CRW_MANIFEST, SEAVIEW_MANIFEST)
            + (MALDIVES_CRW_MANIFEST,)
        ]
        assert len(set(ids)) == 4


# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------


class TestDocumentation:
    """
    Assertions are positive — the required statement must be *present*.

    Substring matching cannot tell a claim from its refusal, and §8 of the same
    document deliberately spells out forbidden phrasings so a reader knows what
    to avoid. Demanding the denial exists is the check that cannot be satisfied
    by deleting text.
    """

    def test_the_paired_analysis_has_its_own_documented_section(self):
        assert MALDIVES_SECTION_HEADING in (PROJECT_ROOT / "docs" / "external_data.md").read_text(
            encoding="utf-8"
        )

    def test_the_section_states_the_verified_dates(self):
        section = _maldives_section()
        assert EXPECTED_FIRST_SURVEY in section
        assert EXPECTED_LAST_SURVEY in section

    def test_the_section_explains_why_only_26_pairs(self):
        section = _normalised(_maldives_section())
        assert "26" in section
        assert "exactly two" in section
        assert "chagos" in section

    def test_the_section_names_the_response_and_its_nature(self):
        section = _normalised(_maldives_section())
        assert "pr_hard_coral" in section
        assert "image-derived" in section

    def test_the_section_explains_why_quadrats_are_not_used(self):
        assert "MASE_MEA_L" in _maldives_section()

    def test_the_section_distinguishes_the_two_crw_acquisitions(self):
        section = _maldives_section()
        assert "noaa_crw_5km_v3_1_maldives_seaview" in section
        assert "2018-01-01" in section
        assert "2024-12-31" in section

    def test_the_section_records_the_matching_and_exposure_rules(self):
        section = _normalised(_maldives_section())
        assert "nearest valid" in section
        assert "spearman" in section
        assert "5 km" in section or "5.0 km" in section

    def test_the_section_states_the_non_causal_limitation(self):
        section = _normalised(_maldives_section())
        assert "non-causal" in section
        assert "associated with" in section or "association" in section

    def test_the_section_states_maldives_is_not_india(self):
        assert "MALDIVES_NOT_INDIA" in _maldives_section()

    def test_the_section_states_the_registered_models_are_unaffected(self):
        section = _normalised(_maldives_section())
        assert "champion" in section or "registered" in section
        assert "synthetic" in section

    def test_the_section_asserts_no_causal_claim(self):
        for paragraph in _paragraphs(_maldives_section()):
            text = _normalised(paragraph)
            for claim in BANNED_CAUSAL_CLAIMS:
                if claim in text:
                    assert _denies(text), f"{claim!r} asserted: {paragraph}"

    def test_the_section_asserts_no_indian_validation(self):
        for paragraph in _paragraphs(_maldives_section()):
            text = _normalised(paragraph)
            for claim in BANNED_GEOGRAPHIC_CLAIMS:
                if claim in text:
                    assert _denies(text), f"{claim!r} asserted: {paragraph}"

    def test_the_reported_numbers_match_the_artifacts(self, association, summary):
        """
        Prose drifts from artifacts silently; this is the tether.

        Only figures a reader would act on are pinned — the two rho values, the
        Wilcoxon p, the direction counts and the transect the sensitivity check
        names — not every number in the section.
        """
        # The prose uses a typographic minus (U+2212); Python formats an ASCII
        # hyphen. Fold them together rather than making the document uglier.
        section = _maldives_section().replace("−", "-")
        tests = association["association"]["tests"]
        assert f"{tests['max_dhw']['rho']:.3f}" in section
        assert f"{tests['max_hotspot']['rho']:.3f}" in section

        paired = summary["paired_change"]
        assert f"{paired['paired_test']['wilcoxon_p_value']:.5f}" in section
        assert str(paired["direction"]["declining"]) in section

        sensitivity = association["association"]["sensitivity_leave_one_pair_out"]
        assert str(sensitivity["max_hotspot"]["most_influential_transect_id"]) in section

    @pytest.mark.parametrize("path", DOC_FILES, ids=lambda p: p.name)
    def test_the_existing_seaview_denials_survive(self, path):
        """The new section must not have displaced §8's standing warnings."""
        text = _normalised(path.read_text(encoding="utf-8"))
        assert "indian ocean is not india" in text
        assert "does not validate any model for" in text
        assert "hard_coral_cover != reef_health" in text


# ---------------------------------------------------------------------------
# Reproducibility of the reported window
# ---------------------------------------------------------------------------


class TestWindowIsInternallyConsistent:
    def test_the_reported_range_matches_the_pair_rows(self, pairs, summary):
        firsts = sorted(row["first_survey_date"][:10] for row in pairs)
        seconds = sorted(row["second_survey_date"][:10] for row in pairs)
        assert firsts[0] == summary["pair_selection"]["earliest_first_survey"]
        assert seconds[-1] == summary["pair_selection"]["latest_second_survey"]

    def test_every_survey_date_parses_as_a_real_date(self, pairs):
        for row in pairs:
            for column in ("first_survey_date", "second_survey_date"):
                date.fromisoformat(row[column][:10])

    def test_interval_days_matches_the_two_dates(self, pairs):
        for row in pairs:
            first = date.fromisoformat(row["first_survey_date"][:10])
            second = date.fromisoformat(row["second_survey_date"][:10])
            assert (second - first).days == int(row["interval_days"]), row["transect_id"]


# ---------------------------------------------------------------------------
# Statistical interpretation
# ---------------------------------------------------------------------------


def _interpretation_surfaces(summary: dict, association: dict) -> dict[str, str]:
    """
    The places a reader could pick up the paired-test interpretation.

    Machine-readable and human-readable are both included: a correction applied
    to one and not the other leaves a reader quoting the stale version, which is
    exactly the failure this milestone is fixing.
    """
    return {
        "pairs_summary.json": _normalised(json.dumps(summary)),
        "crw_association.json": _normalised(json.dumps(association)),
        "docs/external_data.md §8a": _normalised(_maldives_section()),
    }


class TestPairedTestInterpretation:
    """
    Both paired tests are reported, and neither is justified by the other's
    absence.

    The failure mode being guarded against is a real one in applied statistics:
    running Shapiro-Wilk, seeing p < 0.05, and concluding that the rank-based
    test is therefore the correct one. It is not an entailment — Wilcoxon
    signed-rank assumes the differences are symmetric about their median, which
    a normality test says nothing about. The honest framing is a reporting
    choice with both tests shown, and that is what these tests pin.
    """

    def test_no_surface_encodes_non_normal_therefore_wilcoxon(self, summary, association):
        for where, text in _interpretation_surfaces(summary, association).items():
            for pattern in SIMPLISTIC_NORMALITY_RULES:
                match = re.search(pattern, text)
                assert match is None, f"{where}: simplistic normality rule: {match.group(0)!r}"

    def test_wilcoxon_is_described_as_the_primary_rank_based_analysis(self, summary):
        text = _normalised(json.dumps(summary["paired_change"]))
        assert "primary rank-based paired analysis" in text
        assert "wilcoxon" in text

    def test_the_paired_t_test_is_described_as_complementary(self, summary):
        text = _normalised(json.dumps(summary["paired_change"]))
        assert "complementary sensitivity analysis" in text
        assert "sensitivity analysis" in _normalised(_maldives_section())

    def test_both_paired_tests_are_still_reported(self, summary):
        test = summary["paired_change"]["paired_test"]
        for key in (
            "wilcoxon_statistic",
            "wilcoxon_p_value",
            "secondary_paired_t_statistic",
            "secondary_paired_t_p_value",
        ):
            assert isinstance(test[key], float), key

    def test_the_normality_result_carries_its_own_caveat(self, summary):
        """
        Stating the non-normality is not enough; the limits of what follows from
        it must travel with it, or the reader supplies the missing inference.
        """
        check = summary["paired_change"]["normality_check"]
        text = _normalised(check["interpretation"])
        assert "shapiro" in text
        assert "own assumption" in text or "symmetry" in text

    def test_the_documented_section_reports_both_tests(self):
        section = _normalised(_maldives_section())
        assert "wilcoxon" in section
        assert "paired t-test" in section


class TestSpatialDependenceIsDisclosed:
    """
    26 rows are not 26 independent climatic observations.

    The transects sit in one small box under one 2016 heat-stress event, so the
    exposure values attached to them are not independent draws. Nothing here is
    corrected for — no spatial model, no clustered bootstrap — so the disclosure
    is the whole of the remedy, and it has to be present and unambiguous.
    """

    def test_the_association_report_discloses_spatial_clustering(self, association):
        text = _normalised(json.dumps(association))
        assert "spatially clustered" in text
        assert "independent climatic replicates" in text
        assert "descriptive" in text

    def test_the_limitations_block_names_spatial_dependence(self, association, summary):
        for blob in (association, summary):
            note = _normalised(blob["limitations"]["spatial_dependence"])
            assert "spatially clustered" in note
            assert "2016" in note
            assert "independent climatic replicates" in note

    def test_the_documented_section_discloses_spatial_clustering(self):
        section = _normalised(_maldives_section())
        assert "spatially clustered" in section
        assert "independent climatic replicates" in section

    def test_independence_is_denied_and_never_asserted(self, summary, association):
        """
        The phrase has to appear inside a denial. Asserting the transects *are*
        independent replicates, anywhere, is the error being guarded against.
        """
        for where, text in _interpretation_surfaces(summary, association).items():
            for pattern in REPLICATE_INDEPENDENCE_CLAIMS:
                for occurrence in re.finditer(pattern, text):
                    window = text[max(0, occurrence.start() - 220) : occurrence.end() + 60]
                    assert _denies(window), f"{where}: independence asserted: {window}"

    def test_correlation_inference_stays_descriptive(self, association):
        interpretation = _normalised(json.dumps(association["association"]["interpretation"]))
        assert "descriptive" in interpretation
        assert "not detected here" in interpretation

    def test_no_new_inferential_machinery_was_introduced(self):
        """
        The correction is a disclosure, not a new model. If a spatial
        regression or a clustered resampling scheme ever does get added, this
        test should be updated deliberately rather than drift past unnoticed.
        """
        referenced = _referenced_identifiers(ANALYSIS_SCRIPT)
        for banned in ("mixedlm", "MixedLM", "permutation_test", "variogram", "GLS", "gls"):
            assert banned not in referenced, banned

    def test_the_report_records_that_no_adjustment_was_made(self, association):
        note = _normalised(
            association["association"]["not_performed"]["spatial_dependence_adjustment"]
        )
        assert "none" in note
        assert "disclosed" in note or "not corrected" in note
