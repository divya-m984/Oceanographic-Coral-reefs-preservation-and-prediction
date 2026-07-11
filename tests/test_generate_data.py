"""
tests/test_generate_data.py — Unit tests for src/data/generate_data.py (M2).

Covers:
  - Row count and column presence
  - Reproducibility (same seed → identical output)
  - No missing values
  - Field ranges (all features within documented scientific bounds)
  - Coordinates within region bounding boxes
  - All four regions and all class labels present
  - Minimum class coverage (no degenerate distributions)
  - No single column is a near-perfect predictor of either target
  - Temporal range of timestamps
  - Positive correlation between temperature and bleaching
  - CLI main() function
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Spearman helper — avoids scipy dependency
# ---------------------------------------------------------------------------


def _spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman rank-order correlation computed as Pearson correlation of ranks.

    Mathematically equivalent to scipy.stats.spearmanr but requires only pandas.
    """
    return float(a.rank().corr(b.rank()))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg():
    """Load project config once per test session."""
    from src.config import Config

    return Config.load(PROJECT_ROOT)


@pytest.fixture(scope="module")
def df(cfg):
    """Small dataset generated with a fixed seed — used for fast unit tests."""
    from src.data.generate_data import generate_observations

    return generate_observations(n_samples=800, seed=42, cfg=cfg)


@pytest.fixture(scope="module")
def df_large(cfg):
    """Larger dataset used for distribution and correlation tests."""
    from src.data.generate_data import generate_observations

    return generate_observations(n_samples=3000, seed=42, cfg=cfg)


# ---------------------------------------------------------------------------
# Expected column set
# ---------------------------------------------------------------------------
_EXPECTED_COLUMNS = [
    "timestamp",
    "latitude",
    "longitude",
    "region",
    "depth_m",
    "water_temperature_c",
    "ph",
    "salinity_ppt",
    "dissolved_oxygen_mg_l",
    "turbidity_ntu",
    "light_intensity",
    "current_speed_m_s",
    "sonar_backscatter",
    "rugosity_index",
    "hard_substrate_percentage",
    "acoustic_complexity_index",
    "coral_cover_percentage",
    "bleaching_percentage",
    "disease_percentage",
    "reef_health",
    "restoration_suitability",
]

_HEALTH_CLASSES = ["healthy", "stressed", "bleached", "severely_degraded"]
_RESTORATION_CLASSES = ["suitable", "moderately_suitable", "unsuitable"]


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


class TestShape:
    @pytest.mark.unit
    def test_row_count_matches_requested(self, df):
        assert len(df) == 800

    @pytest.mark.unit
    def test_column_count(self, df):
        assert df.shape[1] == len(_EXPECTED_COLUMNS)

    @pytest.mark.unit
    def test_all_expected_columns_present(self, df):
        missing = [c for c in _EXPECTED_COLUMNS if c not in df.columns]
        assert missing == [], f"Missing columns: {missing}"

    @pytest.mark.unit
    def test_no_extra_latent_columns(self, df):
        """Internal latent variables (_stress, _structure) must not appear."""
        latent = [c for c in df.columns if c.startswith("_")]
        assert latent == [], f"Latent columns leaked into output: {latent}"


# ---------------------------------------------------------------------------
# Reproducibility tests
# ---------------------------------------------------------------------------


class TestReproducibility:
    @pytest.mark.unit
    def test_same_seed_produces_identical_dataframe(self, cfg):
        from src.data.generate_data import generate_observations

        df1 = generate_observations(n_samples=200, seed=99, cfg=cfg)
        df2 = generate_observations(n_samples=200, seed=99, cfg=cfg)
        pd.testing.assert_frame_equal(df1, df2)

    @pytest.mark.unit
    def test_different_seeds_produce_different_data(self, cfg):
        from src.data.generate_data import generate_observations

        df1 = generate_observations(n_samples=200, seed=1, cfg=cfg)
        df2 = generate_observations(n_samples=200, seed=2, cfg=cfg)
        # At least the numeric temperatures should differ
        assert not np.allclose(
            df1["water_temperature_c"].values,
            df2["water_temperature_c"].values,
        )

    @pytest.mark.unit
    def test_sample_count_param_is_respected(self, cfg):
        from src.data.generate_data import generate_observations

        for n in (100, 500, 1200):
            assert len(generate_observations(n_samples=n, seed=0, cfg=cfg)) == n


# ---------------------------------------------------------------------------
# Missing-value tests
# ---------------------------------------------------------------------------


class TestMissingValues:
    @pytest.mark.unit
    def test_no_null_values_anywhere(self, df):
        null_counts = df.isnull().sum()
        bad = null_counts[null_counts > 0]
        assert bad.empty, f"Columns with nulls:\n{bad}"

    @pytest.mark.unit
    def test_no_inf_in_numeric_columns(self, df):
        numeric = df.select_dtypes(include="number")
        assert not np.isinf(numeric.values).any(), "Infinite values found in numeric columns"


# ---------------------------------------------------------------------------
# Field-range tests
# ---------------------------------------------------------------------------

# (min_allowed, max_allowed, column_name)
_RANGE_SPECS = [
    (0.0, 45.0, "depth_m"),
    (15.0, 39.0, "water_temperature_c"),
    (7.55, 8.55, "ph"),
    (26.0, 46.0, "salinity_ppt"),
    (1.5, 11.5, "dissolved_oxygen_mg_l"),
    (0.0, 34.0, "turbidity_ntu"),
    (4.0, 2201.0, "light_intensity"),
    (0.0, 1.0, "current_speed_m_s"),
    (-37.0, -1.0, "sonar_backscatter"),
    (0.9, 5.0, "rugosity_index"),
    (0.0, 100.0, "hard_substrate_percentage"),
    (0.0, 1.01, "acoustic_complexity_index"),
    (0.0, 91.0, "coral_cover_percentage"),
    (0.0, 100.0, "bleaching_percentage"),
    (0.0, 62.0, "disease_percentage"),
    (-91.0, 91.0, "latitude"),
    (-181.0, 181.0, "longitude"),
]


class TestFieldRanges:
    @pytest.mark.unit
    @pytest.mark.parametrize("lo,hi,col", _RANGE_SPECS)
    def test_column_within_bounds(self, df_large, lo, hi, col):
        below = (df_large[col] < lo).sum()
        above = (df_large[col] > hi).sum()
        assert below == 0, f"{col}: {below} values below {lo}"
        assert above == 0, f"{col}: {above} values above {hi}"


# ---------------------------------------------------------------------------
# Coordinate / region tests
# ---------------------------------------------------------------------------


class TestCoordinates:
    @pytest.mark.unit
    def test_all_four_regions_present(self, df):
        expected = {
            "Lakshadweep",
            "Gulf of Mannar",
            "Gulf of Kutch",
            "Andaman and Nicobar Islands",
        }
        assert set(df["region"].unique()) == expected

    @pytest.mark.unit
    def test_latitudes_within_region_bounds(self, df, cfg):
        for region, bounds in cfg.region_bounds.items():
            lat_min, lat_max, _, _ = bounds
            mask = df["region"] == region
            lats = df.loc[mask, "latitude"]
            assert (lats >= lat_min).all(), f"{region}: latitude below {lat_min}"
            assert (lats <= lat_max).all(), f"{region}: latitude above {lat_max}"

    @pytest.mark.unit
    def test_longitudes_within_region_bounds(self, df, cfg):
        for region, bounds in cfg.region_bounds.items():
            _, _, lon_min, lon_max = bounds
            mask = df["region"] == region
            lons = df.loc[mask, "longitude"]
            assert (lons >= lon_min).all(), f"{region}: longitude below {lon_min}"
            assert (lons <= lon_max).all(), f"{region}: longitude above {lon_max}"

    @pytest.mark.unit
    def test_each_region_has_minimum_samples(self, df):
        counts = df["region"].value_counts()
        for region in counts.index:
            assert counts[region] >= 50, f"{region} has fewer than 50 samples"


# ---------------------------------------------------------------------------
# Target-class distribution tests
# ---------------------------------------------------------------------------


class TestTargetDistributions:
    @pytest.mark.unit
    def test_all_health_classes_present(self, df):
        present = set(df["reef_health"].unique())
        missing = set(_HEALTH_CLASSES) - present
        assert not missing, f"Missing health classes: {missing}"

    @pytest.mark.unit
    def test_all_restoration_classes_present(self, df):
        present = set(df["restoration_suitability"].unique())
        missing = set(_RESTORATION_CLASSES) - present
        assert not missing, f"Missing restoration classes: {missing}"

    @pytest.mark.unit
    def test_health_classes_minimum_coverage(self, df_large):
        """Each health class must represent at least 5% of the dataset."""
        for cls in _HEALTH_CLASSES:
            pct = 100.0 * (df_large["reef_health"] == cls).sum() / len(df_large)
            assert pct >= 5.0, f"Class '{cls}' coverage too low: {pct:.1f}%"

    @pytest.mark.unit
    def test_restoration_classes_minimum_coverage(self, df_large):
        """Each restoration class must represent at least 8% of the dataset.

        'unsuitable' is naturally rarer given the region mix (Andaman 42% of
        data is generally suitable).  8% still yields ~1 200 samples at the
        default dataset size of 15 000 — sufficient for model training.
        """
        for cls in _RESTORATION_CLASSES:
            pct = 100.0 * (df_large["restoration_suitability"] == cls).sum() / len(df_large)
            assert pct >= 8.0, f"Class '{cls}' coverage too low: {pct:.1f}%"

    @pytest.mark.unit
    def test_health_labels_are_valid_strings(self, df):
        invalid = df["reef_health"][~df["reef_health"].isin(_HEALTH_CLASSES)]
        assert invalid.empty, f"Invalid health labels: {invalid.unique()}"

    @pytest.mark.unit
    def test_restoration_labels_are_valid_strings(self, df):
        invalid = df["restoration_suitability"][
            ~df["restoration_suitability"].isin(_RESTORATION_CLASSES)
        ]
        assert invalid.empty, f"Invalid restoration labels: {invalid.unique()}"


# ---------------------------------------------------------------------------
# Data-quality / scientific plausibility tests
# ---------------------------------------------------------------------------


class TestDataQuality:
    @pytest.mark.unit
    def test_timestamps_in_expected_range(self, df):
        ts = pd.to_datetime(df["timestamp"])
        assert ts.min() >= pd.Timestamp("2018-01-01"), "Timestamps start before 2018"
        assert ts.max() <= pd.Timestamp("2025-01-01"), "Timestamps end after 2024"

    @pytest.mark.unit
    def test_bleaching_positively_correlates_with_temperature(self, df_large):
        """Higher temperature should increase bleaching (Spearman r > 0.15)."""
        r = _spearman(df_large["water_temperature_c"], df_large["bleaching_percentage"])
        assert r > 0.15, f"Temp-bleaching Spearman r too low: {r:.3f}"

    @pytest.mark.unit
    def test_disease_correlates_with_stress_indicators(self, df_large):
        """Disease percentage should increase as pH drops (negative correlation)."""
        r = _spearman(df_large["ph"], df_large["disease_percentage"])
        assert r < -0.05, f"pH-disease Spearman r unexpected: {r:.3f}"

    @pytest.mark.unit
    def test_sonar_backscatter_correlates_with_hard_substrate(self, df_large):
        """Harder substrate → higher (less negative) sonar return."""
        r = _spearman(df_large["hard_substrate_percentage"], df_large["sonar_backscatter"])
        assert r > 0.50, f"Substrate-backscatter Spearman r too low: {r:.3f}"

    @pytest.mark.unit
    def test_rugosity_correlates_with_hard_substrate(self, df_large):
        """Complex reef structure correlates with hard substrate."""
        r = _spearman(df_large["hard_substrate_percentage"], df_large["rugosity_index"])
        assert r > 0.50, f"Substrate-rugosity Spearman r too low: {r:.3f}"

    @pytest.mark.unit
    def test_light_decreases_with_depth(self, df_large):
        """Light intensity should decrease as depth increases."""
        r = _spearman(df_large["depth_m"], df_large["light_intensity"])
        assert r < -0.10, f"Depth-light Spearman r unexpected: {r:.3f}"

    @pytest.mark.unit
    def test_no_feature_perfectly_predicts_health(self, df_large):
        """No single numeric column should have |Spearman r| >= 0.92 with the
        ordinal-encoded health label.  This ensures models must learn
        multi-feature interactions."""
        label_map = {
            "healthy": 0,
            "stressed": 1,
            "bleached": 2,
            "severely_degraded": 3,
        }
        encoded = df_large["reef_health"].map(label_map)
        numeric_cols = df_large.select_dtypes(include="number").columns
        for col in numeric_cols:
            r = abs(_spearman(df_large[col], encoded))
            assert r < 0.92, (
                f"Column '{col}' has |r|={r:.3f} with health label — "
                "too close to a perfect predictor."
            )

    @pytest.mark.unit
    def test_no_feature_perfectly_predicts_restoration(self, df_large):
        """No single numeric column should have |Spearman r| >= 0.92 with the
        ordinal-encoded restoration label."""
        label_map = {"unsuitable": 0, "moderately_suitable": 1, "suitable": 2}
        encoded = df_large["restoration_suitability"].map(label_map)
        numeric_cols = df_large.select_dtypes(include="number").columns
        for col in numeric_cols:
            r = abs(_spearman(df_large[col], encoded))
            assert r < 0.92, (
                f"Column '{col}' has |r|={r:.3f} with restoration label — "
                "too close to a perfect predictor."
            )

    @pytest.mark.unit
    def test_gulf_of_kutch_higher_salinity_than_andaman(self, df_large):
        """Gulf of Kutch is hypersaline; Andaman should be much lower."""
        kutch = df_large.loc[df_large["region"] == "Gulf of Kutch", "salinity_ppt"].mean()
        andaman = df_large.loc[
            df_large["region"] == "Andaman and Nicobar Islands", "salinity_ppt"
        ].mean()
        assert kutch > andaman + 3.0, (
            f"Gulf of Kutch salinity ({kutch:.1f}) should be >3 ppt above Andaman ({andaman:.1f})"
        )

    @pytest.mark.unit
    def test_andaman_healthier_than_gulf_of_kutch(self, df_large):
        """Andaman should have significantly more 'healthy' reefs than Gulf of Kutch."""
        andaman_healthy = (
            df_large.loc[df_large["region"] == "Andaman and Nicobar Islands", "reef_health"]
            == "healthy"
        ).mean()
        kutch_healthy = (
            df_large.loc[df_large["region"] == "Gulf of Kutch", "reef_health"] == "healthy"
        ).mean()
        assert andaman_healthy > kutch_healthy, (
            f"Andaman healthy fraction ({andaman_healthy:.2f}) should exceed "
            f"Gulf of Kutch ({kutch_healthy:.2f})"
        )


# ---------------------------------------------------------------------------
# CLI / integration tests
# ---------------------------------------------------------------------------


class TestCLI:
    @pytest.mark.integration
    def test_main_creates_output_file(self, tmp_path, monkeypatch):
        """CLI main() should write a CSV to the requested path."""
        import sys

        output = tmp_path / "obs.csv"
        monkeypatch.setattr(
            sys,
            "argv",
            ["generate_data", "--samples", "150", "--seed", "7", "--output", str(output)],
        )
        from src.config import reset_config
        from src.data.generate_data import main

        reset_config()
        result = main()
        assert result == 0
        assert output.exists()

    @pytest.mark.integration
    def test_main_output_has_correct_shape(self, tmp_path, monkeypatch):
        """CSV written by main() should match requested row count and column set."""
        import sys

        output = tmp_path / "obs2.csv"
        monkeypatch.setattr(
            sys,
            "argv",
            ["generate_data", "--samples", "250", "--seed", "13", "--output", str(output)],
        )
        from src.config import reset_config
        from src.data.generate_data import main

        reset_config()
        main()
        loaded = pd.read_csv(output)
        assert len(loaded) == 250
        missing_cols = [c for c in _EXPECTED_COLUMNS if c not in loaded.columns]
        assert missing_cols == [], f"Missing columns in saved CSV: {missing_cols}"

    @pytest.mark.integration
    def test_default_output_path_exists_after_generation(self, tmp_path, monkeypatch):
        """The CLI must create a non-empty CSV at the requested output path.

        Uses tmp_path so no real project file is touched during testing.
        """
        import sys

        output = tmp_path / "observations.csv"
        monkeypatch.setattr(
            sys,
            "argv",
            ["generate_data", "--samples", "100", "--seed", "42", "--output", str(output)],
        )
        from src.config import reset_config
        from src.data.generate_data import main

        reset_config()
        result = main()
        assert result == 0
        assert output.exists()
        assert output.stat().st_size > 0
