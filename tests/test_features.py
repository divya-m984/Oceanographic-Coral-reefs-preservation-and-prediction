"""
tests/test_features.py — Tests for src/features/build_features.py (M4).

Coverage
--------
- All six derived features are created
- Formula correctness for each derived feature
- Boundary / clipping behaviour
- Original columns are preserved
- NaN propagation is correct
- Column-list helpers return expected values
- No target leakage in get_feature_columns()
- Feature metadata structure
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.config import get_config, reset_config
from src.data.generate_data import generate_observations
from src.features.build_features import (
    ALL_FEATURE_COLUMNS,
    DERIVED_FEATURE_NAMES,
    METADATA_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    TARGET_COLUMNS,
    add_derived_features,
    build_feature_metadata,
    get_feature_columns,
    get_target_column,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg():
    reset_config()
    return get_config()


@pytest.fixture(scope="module")
def raw_df(cfg):
    return generate_observations(n_samples=300, seed=99, cfg=cfg)


@pytest.fixture(scope="module")
def feat_df(raw_df):
    return add_derived_features(raw_df)


# ---------------------------------------------------------------------------
# TestDerivedFeaturesPresent
# ---------------------------------------------------------------------------


class TestDerivedFeaturesPresent:
    """add_derived_features() must add all six derived columns."""

    def test_all_derived_columns_added(self, feat_df: pd.DataFrame) -> None:
        for col in DERIVED_FEATURE_NAMES:
            assert col in feat_df.columns, f"Derived feature '{col}' not found"

    def test_exactly_six_derived_features(self) -> None:
        assert len(DERIVED_FEATURE_NAMES) == 6

    def test_original_columns_preserved(self, raw_df, feat_df) -> None:
        for col in raw_df.columns:
            assert col in feat_df.columns, f"Original column '{col}' lost after feature engineering"

    def test_output_is_copy_not_inplace(self, raw_df) -> None:
        original_cols = list(raw_df.columns)
        _ = add_derived_features(raw_df)
        assert list(raw_df.columns) == original_cols, "add_derived_features must not mutate input"

    def test_row_count_unchanged(self, raw_df, feat_df) -> None:
        assert len(feat_df) == len(raw_df)


# ---------------------------------------------------------------------------
# TestDerivedFeatureFormulas
# ---------------------------------------------------------------------------


class TestDerivedFeatureFormulas:
    """Spot-check formula correctness on known input values."""

    def _make_row(self, **kwargs) -> pd.DataFrame:
        """Build a 1-row DataFrame with default safe values, overriding with kwargs."""
        defaults = {
            "water_temperature_c": 27.0,
            "dissolved_oxygen_mg_l": 7.0,
            "ph": 8.1,
            "turbidity_ntu": 5.0,
            "hard_substrate_percentage": 60.0,
            "rugosity_index": 3.0,
            "acoustic_complexity_index": 0.7,
            # other columns required by the function — dummy values
            "depth_m": 10.0,
            "salinity_ppt": 35.0,
            "light_intensity": 500.0,
            "current_speed_m_s": 0.2,
            "sonar_backscatter": -15.0,
            "coral_cover_percentage": 50.0,
            "bleaching_percentage": 5.0,
            "disease_percentage": 2.0,
            "timestamp": "2022-01-01",
            "latitude": 11.0,
            "longitude": 73.0,
            "region": "Lakshadweep",
            "reef_health": "healthy",
            "restoration_suitability": "suitable",
        }
        defaults.update(kwargs)
        return pd.DataFrame([defaults])

    def test_thermal_stress_index_no_stress(self) -> None:
        """Temperature below threshold → index = 0."""
        df = add_derived_features(self._make_row(water_temperature_c=25.0))
        assert df["thermal_stress_index"].iloc[0] == pytest.approx(0.0)

    def test_thermal_stress_index_max_stress(self) -> None:
        """Temperature at threshold + normaliser → index = 1."""
        df = add_derived_features(self._make_row(water_temperature_c=28.5 + 4.5))
        assert df["thermal_stress_index"].iloc[0] == pytest.approx(1.0)

    def test_thermal_stress_index_partial(self) -> None:
        """Temperature halfway above threshold → index ≈ 0.5."""
        df = add_derived_features(self._make_row(water_temperature_c=28.5 + 2.25))
        assert df["thermal_stress_index"].iloc[0] == pytest.approx(0.5, abs=1e-6)

    def test_thermal_stress_clipped_above_one(self) -> None:
        """Temperature well above threshold → clipped at 1.0."""
        df = add_derived_features(self._make_row(water_temperature_c=42.0))
        assert df["thermal_stress_index"].iloc[0] == pytest.approx(1.0)

    def test_oxygen_stress_no_stress(self) -> None:
        """DO above healthy level → index = 0."""
        df = add_derived_features(self._make_row(dissolved_oxygen_mg_l=8.0))
        assert df["oxygen_stress_index"].iloc[0] == pytest.approx(0.0)

    def test_oxygen_stress_max(self) -> None:
        """DO at 6.0 - 3.5 = 2.5 mg/L → index = 1."""
        df = add_derived_features(self._make_row(dissolved_oxygen_mg_l=2.5))
        assert df["oxygen_stress_index"].iloc[0] == pytest.approx(1.0)

    def test_oxygen_stress_clipped(self) -> None:
        """DO = 0 → clipped at 1.0."""
        df = add_derived_features(self._make_row(dissolved_oxygen_mg_l=0.0))
        assert df["oxygen_stress_index"].iloc[0] == pytest.approx(1.0)

    def test_acidity_deviation_at_reference(self) -> None:
        """pH at reference (8.1) → deviation = 0."""
        df = add_derived_features(self._make_row(ph=8.1))
        assert df["acidity_deviation"].iloc[0] == pytest.approx(0.0, abs=1e-9)

    def test_acidity_deviation_acidic(self) -> None:
        """pH = 7.6 → deviation = |7.6 - 8.1| = 0.5."""
        df = add_derived_features(self._make_row(ph=7.6))
        assert df["acidity_deviation"].iloc[0] == pytest.approx(0.5, abs=1e-9)

    def test_acidity_deviation_alkaline(self) -> None:
        """pH = 8.6 → deviation = |8.6 - 8.1| = 0.5 (symmetric)."""
        df = add_derived_features(self._make_row(ph=8.6))
        assert df["acidity_deviation"].iloc[0] == pytest.approx(0.5, abs=1e-9)

    def test_water_quality_index_optimal_conditions(self) -> None:
        """Near-optimal conditions → WQI close to 1."""
        df = add_derived_features(
            self._make_row(
                water_temperature_c=25.0,
                dissolved_oxygen_mg_l=8.0,
                ph=8.1,
                turbidity_ntu=0.0,
            )
        )
        assert df["water_quality_index"].iloc[0] > 0.9

    def test_water_quality_index_worst_conditions(self) -> None:
        """Maximum stress → WQI close to 0."""
        df = add_derived_features(
            self._make_row(
                water_temperature_c=42.0,
                dissolved_oxygen_mg_l=0.0,
                ph=7.0,
                turbidity_ntu=100.0,
            )
        )
        assert df["water_quality_index"].iloc[0] < 0.15

    def test_water_quality_index_bounded(self, feat_df) -> None:
        """WQI must always be in [0, 1]."""
        wqi = feat_df["water_quality_index"]
        assert (wqi >= 0.0).all()
        assert (wqi <= 1.0).all()

    def test_substrate_stability_flat_no_hard(self) -> None:
        """No hard substrate, flat rugosity → score = 0."""
        df = add_derived_features(self._make_row(hard_substrate_percentage=0.0, rugosity_index=1.0))
        assert df["substrate_stability_score"].iloc[0] == pytest.approx(0.0, abs=1e-9)

    def test_substrate_stability_maximum(self) -> None:
        """Full hard substrate, maximum rugosity → score = 1."""
        df = add_derived_features(
            self._make_row(hard_substrate_percentage=100.0, rugosity_index=10.0)
        )
        assert df["substrate_stability_score"].iloc[0] == pytest.approx(1.0, abs=1e-6)

    def test_substrate_stability_bounded(self, feat_df) -> None:
        score = feat_df["substrate_stability_score"]
        assert (score >= 0.0).all()
        assert (score <= 1.0).all()

    def test_structural_complexity_score_bounded(self, feat_df) -> None:
        score = feat_df["structural_complexity_score"]
        assert (score >= 0.0).all()
        assert (score <= 1.0).all()

    def test_structural_complexity_minimum(self) -> None:
        """ACI = 0 and rugosity = 1 → score = 0."""
        df = add_derived_features(self._make_row(acoustic_complexity_index=0.0, rugosity_index=1.0))
        assert df["structural_complexity_score"].iloc[0] == pytest.approx(0.0, abs=1e-9)

    def test_structural_complexity_maximum(self) -> None:
        """ACI = 1 and rugosity = 10 → score = 1."""
        df = add_derived_features(
            self._make_row(acoustic_complexity_index=1.0, rugosity_index=10.0)
        )
        assert df["structural_complexity_score"].iloc[0] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# TestNaNPropagation
# ---------------------------------------------------------------------------


class TestNaNPropagation:
    """NaN in a source column must propagate into dependent derived features."""

    def test_nan_temperature_propagates_to_thermal_stress(self, raw_df) -> None:
        df = raw_df.copy()
        df.loc[0, "water_temperature_c"] = float("nan")
        out = add_derived_features(df)
        assert math.isnan(out["thermal_stress_index"].iloc[0])

    def test_nan_temperature_propagates_to_wqi(self, raw_df) -> None:
        df = raw_df.copy()
        df.loc[0, "water_temperature_c"] = float("nan")
        out = add_derived_features(df)
        assert math.isnan(out["water_quality_index"].iloc[0])

    def test_nan_ph_propagates_to_acidity_deviation(self, raw_df) -> None:
        df = raw_df.copy()
        df.loc[0, "ph"] = float("nan")
        out = add_derived_features(df)
        assert math.isnan(out["acidity_deviation"].iloc[0])

    def test_non_affected_rows_unchanged(self, raw_df) -> None:
        df = raw_df.copy()
        df.loc[0, "ph"] = float("nan")
        out_nan = add_derived_features(df)
        out_full = add_derived_features(raw_df)
        # Row 1 should be identical across both outputs
        assert out_nan["acidity_deviation"].iloc[1] == pytest.approx(
            out_full["acidity_deviation"].iloc[1]
        )


# ---------------------------------------------------------------------------
# TestColumnListHelpers
# ---------------------------------------------------------------------------


class TestColumnListHelpers:
    """Column-list functions must return correct and consistent values."""

    def test_metadata_columns_count(self) -> None:
        assert len(METADATA_COLUMNS) == 3
        assert "timestamp" in METADATA_COLUMNS
        assert "latitude" in METADATA_COLUMNS
        assert "longitude" in METADATA_COLUMNS

    def test_target_columns_count(self) -> None:
        assert len(TARGET_COLUMNS) == 2
        assert "reef_health" in TARGET_COLUMNS
        assert "restoration_suitability" in TARGET_COLUMNS

    def test_numeric_feature_columns_count(self) -> None:
        # 15 raw + 6 derived = 21
        assert len(NUMERIC_FEATURE_COLUMNS) == 21

    def test_all_feature_columns_count(self) -> None:
        # 21 numeric + 1 categorical (region) = 22
        assert len(ALL_FEATURE_COLUMNS) == 22

    def test_get_feature_columns_health(self) -> None:
        cols = get_feature_columns("health")
        assert "reef_health" not in cols
        assert "restoration_suitability" not in cols
        assert "region" in cols

    def test_get_feature_columns_restoration(self) -> None:
        cols = get_feature_columns("restoration")
        assert "reef_health" not in cols
        assert "restoration_suitability" not in cols
        assert "region" in cols

    def test_get_feature_columns_same_for_both_tasks(self) -> None:
        assert get_feature_columns("health") == get_feature_columns("restoration")

    def test_get_feature_columns_invalid_task(self) -> None:
        with pytest.raises(ValueError, match="task must be"):
            get_feature_columns("unknown")  # type: ignore[arg-type]

    def test_get_target_column_health(self) -> None:
        assert get_target_column("health") == "reef_health"

    def test_get_target_column_restoration(self) -> None:
        assert get_target_column("restoration") == "restoration_suitability"

    def test_get_target_column_invalid(self) -> None:
        with pytest.raises(ValueError, match="task must be"):
            get_target_column("xyz")  # type: ignore[arg-type]

    def test_no_overlap_between_features_and_targets(self) -> None:
        feat_cols = set(ALL_FEATURE_COLUMNS)
        tgt_cols = set(TARGET_COLUMNS)
        assert feat_cols.isdisjoint(tgt_cols)

    def test_no_overlap_between_features_and_metadata(self) -> None:
        feat_cols = set(ALL_FEATURE_COLUMNS)
        meta_cols = set(METADATA_COLUMNS)
        assert feat_cols.isdisjoint(meta_cols)

    def test_derived_features_in_numeric_columns(self) -> None:
        for col in DERIVED_FEATURE_NAMES:
            assert col in NUMERIC_FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# TestFeatureMetadata
# ---------------------------------------------------------------------------


class TestFeatureMetadata:
    """build_feature_metadata() must return a well-formed dict."""

    @pytest.fixture(scope="class")
    def meta(self):
        return build_feature_metadata()

    def test_metadata_has_required_keys(self, meta) -> None:
        required = {
            "metadata_columns",
            "target_columns",
            "raw_numeric_columns",
            "derived_feature_names",
            "numeric_feature_columns",
            "categorical_feature_columns",
            "all_feature_columns",
            "derived_feature_formulas",
            "synthetic_leakage_notes",
            "domain_constants",
        }
        assert required.issubset(meta.keys())

    def test_metadata_feature_count_consistent(self, meta) -> None:
        assert len(meta["all_feature_columns"]) == len(ALL_FEATURE_COLUMNS)

    def test_metadata_leakage_notes_both_tasks(self, meta) -> None:
        assert "reef_health" in meta["synthetic_leakage_notes"]
        assert "restoration_suitability" in meta["synthetic_leakage_notes"]

    def test_derived_formulas_count(self, meta) -> None:
        assert len(meta["derived_feature_formulas"]) == 6

    def test_domain_constants_are_numbers(self, meta) -> None:
        for key, val in meta["domain_constants"].items():
            assert isinstance(val, (int, float)), f"Constant {key!r} should be numeric"

    def test_metadata_is_json_serialisable(self, meta) -> None:
        import json

        serialised = json.dumps(meta)
        assert len(serialised) > 100
