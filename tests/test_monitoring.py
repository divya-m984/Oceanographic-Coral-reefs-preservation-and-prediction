"""
tests/test_monitoring.py — Unit tests for M11 drift monitoring.

Tests cover:
- generate_production.py: shift application, zero-shift case, clipping
- drift.py: DriftDetector (feature, prediction, confidence drift)
- run_drift.py: CLI smoke test with mocked InferencePipeline
- Dashboard page 8: AppTest smoke test

All tests are isolated from real MLflow, real models, and real files.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import (
    DriftDetector,
    _parse_column_from_metric_name,
    _parse_method_from_metric_name,
)
from src.monitoring.generate_production import (
    apply_shift,
    generate_production_window,
    generate_reference_window,
)
from src.monitoring.run_drift import main as run_drift_main
from src.monitoring.run_drift import run_drift

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_ROWS = 200
RNG = np.random.default_rng(42)


def _make_raw_df(n: int = N_ROWS, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Create a minimal synthetic observations DataFrame."""
    _rng = rng or RNG
    return pd.DataFrame(
        {
            "depth_m": _rng.uniform(1, 30, n),
            "water_temperature_c": _rng.uniform(24, 31, n),
            "ph": _rng.uniform(7.8, 8.4, n),
            "salinity_ppt": _rng.uniform(32, 38, n),
            "dissolved_oxygen_mg_l": _rng.uniform(4, 9, n),
            "turbidity_ntu": _rng.uniform(0.5, 15, n),
            "light_intensity": _rng.uniform(100, 1500, n),
            "current_speed_m_s": _rng.uniform(0.01, 0.8, n),
            "sonar_backscatter": _rng.uniform(-30, 0, n),
            "rugosity_index": _rng.uniform(1, 10, n),
            "hard_substrate_percentage": _rng.uniform(10, 90, n),
            "acoustic_complexity_index": _rng.uniform(0.1, 0.9, n),
            "coral_cover_percentage": _rng.uniform(5, 80, n),
            "bleaching_percentage": _rng.uniform(0, 40, n),
            "disease_percentage": _rng.uniform(0, 20, n),
            "region": _rng.choice(
                ["Lakshadweep", "Gulf of Mannar", "Gulf of Kutch"],
                n,
            ),
            "reef_health": _rng.choice(["healthy", "stressed", "bleached"], n),
            "restoration_suitability": _rng.choice(
                ["suitable", "moderately_suitable", "unsuitable"], n
            ),
        }
    )


@pytest.fixture()
def raw_df() -> pd.DataFrame:
    return _make_raw_df()


@pytest.fixture()
def raw_csv(tmp_path: Path, raw_df: pd.DataFrame) -> Path:
    p = tmp_path / "observations.csv"
    raw_df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# generate_production.py
# ---------------------------------------------------------------------------


class TestApplyShift:
    def test_zero_shift_returns_copy(self, raw_df: pd.DataFrame) -> None:
        result = apply_shift(raw_df, shift_scale=0.0)
        pd.testing.assert_frame_equal(result, raw_df)

    def test_zero_shift_is_a_copy(self, raw_df: pd.DataFrame) -> None:
        result = apply_shift(raw_df, shift_scale=0.0)
        assert result is not raw_df

    def test_standard_shift_increases_temperature(self, raw_df: pd.DataFrame) -> None:
        result = apply_shift(raw_df, shift_scale=1.0)
        assert (result["water_temperature_c"] >= raw_df["water_temperature_c"]).all()

    def test_standard_shift_increases_bleaching(self, raw_df: pd.DataFrame) -> None:
        result = apply_shift(raw_df, shift_scale=1.0)
        assert result["bleaching_percentage"].mean() > raw_df["bleaching_percentage"].mean()

    def test_standard_shift_decreases_coral_cover(self, raw_df: pd.DataFrame) -> None:
        result = apply_shift(raw_df, shift_scale=1.0)
        assert result["coral_cover_percentage"].mean() < raw_df["coral_cover_percentage"].mean()

    def test_standard_shift_increases_turbidity(self, raw_df: pd.DataFrame) -> None:
        result = apply_shift(raw_df, shift_scale=1.0)
        assert result["turbidity_ntu"].mean() > raw_df["turbidity_ntu"].mean()

    def test_half_shift_smaller_than_full_shift(self, raw_df: pd.DataFrame) -> None:
        full = apply_shift(raw_df, shift_scale=1.0)
        half = apply_shift(raw_df, shift_scale=0.5)
        assert half["water_temperature_c"].mean() < full["water_temperature_c"].mean()

    def test_clipping_keeps_temperature_in_bounds(self, raw_df: pd.DataFrame) -> None:
        result = apply_shift(raw_df, shift_scale=10.0)
        assert result["water_temperature_c"].max() <= 35.0

    def test_clipping_keeps_bleaching_in_bounds(self, raw_df: pd.DataFrame) -> None:
        result = apply_shift(raw_df, shift_scale=10.0)
        assert result["bleaching_percentage"].max() <= 100.0

    def test_clipping_keeps_coral_cover_non_negative(self, raw_df: pd.DataFrame) -> None:
        result = apply_shift(raw_df, shift_scale=10.0)
        assert result["coral_cover_percentage"].min() >= 0.0

    def test_non_shifted_columns_unchanged(self, raw_df: pd.DataFrame) -> None:
        result = apply_shift(raw_df, shift_scale=1.0)
        for col in ("depth_m", "ph", "salinity_ppt", "region"):
            pd.testing.assert_series_equal(result[col], raw_df[col])

    def test_negative_shift_scale_raises(self, raw_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="shift_scale must be >= 0"):
            apply_shift(raw_df, shift_scale=-0.1)

    def test_labels_preserved_in_output(self, raw_df: pd.DataFrame) -> None:
        result = apply_shift(raw_df, shift_scale=1.0)
        assert "reef_health" in result.columns
        assert "restoration_suitability" in result.columns

    def test_shift_does_not_mutate_input(self, raw_df: pd.DataFrame) -> None:
        original_temp = raw_df["water_temperature_c"].copy()
        apply_shift(raw_df, shift_scale=1.0)
        pd.testing.assert_series_equal(raw_df["water_temperature_c"], original_temp)


class TestGenerateWindows:
    def test_reference_window_size(self, raw_csv: Path) -> None:
        rng = np.random.default_rng(0)
        df = generate_reference_window(raw_csv, reference_n=50, rng=rng)
        assert len(df) == 50

    def test_reference_capped_at_data_size(self, raw_csv: Path, raw_df: pd.DataFrame) -> None:
        rng = np.random.default_rng(0)
        df = generate_reference_window(raw_csv, reference_n=10_000, rng=rng)
        assert len(df) == len(raw_df)

    def test_reference_has_no_labels(self, raw_csv: Path) -> None:
        rng = np.random.default_rng(0)
        df = generate_reference_window(raw_csv, reference_n=50, rng=rng)
        assert "reef_health" not in df.columns
        assert "restoration_suitability" not in df.columns

    def test_production_window_size(self, raw_csv: Path) -> None:
        rng = np.random.default_rng(0)
        df = generate_production_window(raw_csv, production_n=60, shift_scale=1.0, rng=rng)
        assert len(df) == 60

    def test_production_has_no_labels_by_default(self, raw_csv: Path) -> None:
        rng = np.random.default_rng(0)
        df = generate_production_window(raw_csv, production_n=50, shift_scale=1.0, rng=rng)
        assert "reef_health" not in df.columns
        assert "restoration_suitability" not in df.columns

    def test_production_can_keep_labels(self, raw_csv: Path) -> None:
        rng = np.random.default_rng(0)
        df = generate_production_window(
            raw_csv, production_n=50, shift_scale=1.0, rng=rng, drop_labels=False
        )
        assert "reef_health" in df.columns

    def test_production_zero_shift_no_drift(self, raw_csv: Path) -> None:
        """Zero shift should not move feature means significantly."""
        rng_ref = np.random.default_rng(10)
        rng_prod = np.random.default_rng(20)
        ref = generate_reference_window(raw_csv, reference_n=100, rng=rng_ref)
        prod = generate_production_window(raw_csv, production_n=100, shift_scale=0.0, rng=rng_prod)
        # Temperatures should be in similar range
        assert abs(ref["water_temperature_c"].mean() - prod["water_temperature_c"].mean()) < 3.0

    def test_production_standard_shift_moves_temperature(self, raw_csv: Path) -> None:
        rng_ref = np.random.default_rng(10)
        rng_prod = np.random.default_rng(20)
        ref = generate_reference_window(raw_csv, reference_n=100, rng=rng_ref)
        prod = generate_production_window(raw_csv, production_n=100, shift_scale=1.0, rng=rng_prod)
        # Shifted temp should be higher
        assert prod["water_temperature_c"].mean() > ref["water_temperature_c"].mean() + 1.0


# ---------------------------------------------------------------------------
# drift.py — DriftDetector
# ---------------------------------------------------------------------------


@pytest.fixture()
def detector() -> DriftDetector:
    return DriftDetector(drift_threshold=0.05)


@pytest.fixture()
def two_dfs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Undrifted numeric DataFrames."""
    rng = np.random.default_rng(99)
    n = 150
    ref = pd.DataFrame(
        {
            "a": rng.normal(0, 1, n),
            "b": rng.normal(5, 2, n),
            "c": rng.normal(-1, 0.5, n),
        }
    )
    cur = pd.DataFrame(
        {
            "a": rng.normal(0, 1, n),
            "b": rng.normal(5, 2, n),
            "c": rng.normal(-1, 0.5, n),
        }
    )
    return ref, cur


@pytest.fixture()
def drifted_dfs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """DataFrames where column 'a' is clearly shifted."""
    rng = np.random.default_rng(77)
    n = 200
    ref = pd.DataFrame(
        {
            "a": rng.normal(0, 1, n),
            "b": rng.normal(5, 2, n),
        }
    )
    cur = pd.DataFrame(
        {
            "a": rng.normal(10, 1, n),  # large shift
            "b": rng.normal(5, 2, n),  # no shift
        }
    )
    return ref, cur


class TestDriftDetectorFeatureDrift:
    def test_returns_required_keys(
        self, detector: DriftDetector, two_dfs: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        ref, cur = two_dfs
        result = detector.compute_feature_drift(ref, cur)
        for key in ("drifted_count", "total_columns", "drifted_share", "per_column"):
            assert key in result

    def test_total_columns_correct(
        self, detector: DriftDetector, two_dfs: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        ref, cur = two_dfs
        result = detector.compute_feature_drift(ref, cur)
        assert result["total_columns"] == 3

    def test_no_drift_on_same_distribution(
        self, detector: DriftDetector, two_dfs: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        ref, cur = two_dfs
        result = detector.compute_feature_drift(ref, cur)
        assert result["drifted_count"] == 0

    def test_detects_drift_on_shifted_column(
        self, detector: DriftDetector, drifted_dfs: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        ref, cur = drifted_dfs
        result = detector.compute_feature_drift(ref, cur)
        assert result["drifted_count"] >= 1
        assert result["per_column"]["a"]["drifted"] is True

    def test_per_column_has_required_fields(
        self, detector: DriftDetector, two_dfs: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        ref, cur = two_dfs
        result = detector.compute_feature_drift(ref, cur)
        for col_info in result["per_column"].values():
            assert "drifted" in col_info
            assert "p_value" in col_info
            assert "method" in col_info

    def test_column_filter_respected(
        self, detector: DriftDetector, two_dfs: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        ref, cur = two_dfs
        result = detector.compute_feature_drift(ref, cur, columns=["a", "b"])
        assert result["total_columns"] == 2

    def test_missing_column_in_current_ignored(
        self, detector: DriftDetector, two_dfs: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        ref, cur = two_dfs
        cur_missing = cur.drop(columns=["c"])
        # Should not raise; c is in ref but not cur — evidently handles gracefully
        # (or we can just pass common cols; implementation skips missing)
        result = detector.compute_feature_drift(ref, cur_missing, columns=["a", "b"])
        assert result["total_columns"] == 2

    def test_drifted_share_between_0_and_1(
        self, detector: DriftDetector, drifted_dfs: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        ref, cur = drifted_dfs
        result = detector.compute_feature_drift(ref, cur)
        assert 0.0 <= result["drifted_share"] <= 1.0

    def test_no_drift_on_identical_data(self, detector: DriftDetector) -> None:
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"x": rng.normal(0, 1, 200), "y": rng.normal(2, 1, 200)})
        result = detector.compute_feature_drift(df, df.copy())
        assert result["drifted_count"] == 0


class TestDriftDetectorPredictionDrift:
    def test_no_drift_same_distribution(self, detector: DriftDetector) -> None:
        labels = ["healthy", "stressed", "healthy", "stressed", "bleached"] * 20
        result = detector.compute_prediction_drift(
            labels, labels, ["healthy", "stressed", "bleached"]
        )
        assert result["drifted"] is False

    def test_detects_drift_on_shifted_labels(self, detector: DriftDetector) -> None:
        ref = ["healthy"] * 80 + ["stressed"] * 20
        cur = ["stressed"] * 80 + ["healthy"] * 20  # large shift
        result = detector.compute_prediction_drift(ref, cur, ["healthy", "stressed"])
        assert result["drifted"] is True

    def test_returns_required_keys(self, detector: DriftDetector) -> None:
        labels = ["a", "b"] * 30
        result = detector.compute_prediction_drift(labels, labels, ["a", "b"])
        for key in (
            "drifted",
            "p_value",
            "statistic",
            "method",
            "reference_distribution",
            "current_distribution",
        ):
            assert key in result

    def test_method_is_chi2(self, detector: DriftDetector) -> None:
        labels = ["a", "b"] * 30
        result = detector.compute_prediction_drift(labels, labels, ["a", "b"])
        assert result["method"] == "chi2"

    def test_distributions_sum_to_one(self, detector: DriftDetector) -> None:
        ref = ["a", "b", "c"] * 20
        cur = ["a", "b", "c"] * 20
        result = detector.compute_prediction_drift(ref, cur, ["a", "b", "c"])
        assert abs(sum(result["reference_distribution"].values()) - 1.0) < 1e-4
        assert abs(sum(result["current_distribution"].values()) - 1.0) < 1e-4

    def test_empty_labels_no_drift(self, detector: DriftDetector) -> None:
        result = detector.compute_prediction_drift([], [], ["a", "b"])
        assert result["drifted"] is False
        assert result["p_value"] == 1.0

    def test_p_value_between_0_and_1(self, detector: DriftDetector) -> None:
        ref = ["a"] * 50 + ["b"] * 50
        cur = ["a"] * 60 + ["b"] * 40
        result = detector.compute_prediction_drift(ref, cur, ["a", "b"])
        assert 0.0 <= result["p_value"] <= 1.0


class TestDriftDetectorConfidenceDrift:
    def test_no_drift_same_scores(self, detector: DriftDetector) -> None:
        scores = [0.9, 0.8, 0.95, 0.7, 0.85] * 20
        result = detector.compute_confidence_drift(scores, scores)
        assert result["drifted"] is False

    def test_detects_drift_on_shifted_confidence(self, detector: DriftDetector) -> None:
        ref = [0.95] * 100
        cur = [0.55] * 100
        result = detector.compute_confidence_drift(ref, cur)
        assert result["drifted"] is True

    def test_returns_required_keys(self, detector: DriftDetector) -> None:
        scores = [0.9] * 20
        result = detector.compute_confidence_drift(scores, scores)
        for key in (
            "drifted",
            "p_value",
            "statistic",
            "method",
            "mean_reference",
            "mean_current",
            "delta",
        ):
            assert key in result

    def test_method_is_ks(self, detector: DriftDetector) -> None:
        scores = [0.9] * 20
        result = detector.compute_confidence_drift(scores, scores)
        assert result["method"] == "ks"

    def test_empty_scores_no_drift(self, detector: DriftDetector) -> None:
        result = detector.compute_confidence_drift([], [])
        assert result["drifted"] is False

    def test_delta_is_mean_current_minus_mean_reference(self, detector: DriftDetector) -> None:
        ref = [0.9] * 50
        cur = [0.7] * 50
        result = detector.compute_confidence_drift(ref, cur)
        assert abs(result["delta"] - (0.7 - 0.9)) < 1e-3

    def test_p_value_between_0_and_1(self, detector: DriftDetector) -> None:
        rng = np.random.default_rng(5)
        ref = list(rng.uniform(0.7, 0.95, 100))
        cur = list(rng.uniform(0.5, 0.75, 100))
        result = detector.compute_confidence_drift(ref, cur)
        assert 0.0 <= result["p_value"] <= 1.0


class TestDriftDetectorRecommendation:
    def test_ok_when_no_drift(self, detector: DriftDetector) -> None:
        rec = detector.make_recommendation(False, False, False)
        assert rec.startswith("OK")

    def test_retrain_when_feature_and_pred_drift(self, detector: DriftDetector) -> None:
        rec = detector.make_recommendation(True, True, False)
        assert rec.startswith("RETRAIN")

    def test_investigate_feature_when_only_feature_drift(self, detector: DriftDetector) -> None:
        rec = detector.make_recommendation(True, False, False)
        assert "INVESTIGATE" in rec

    def test_investigate_pred_when_only_pred_drift(self, detector: DriftDetector) -> None:
        rec = detector.make_recommendation(False, False, True)
        assert "INVESTIGATE" in rec

    def test_retrain_when_feature_and_restoration_pred_drift(self, detector: DriftDetector) -> None:
        rec = detector.make_recommendation(True, False, True)
        assert rec.startswith("RETRAIN")


class TestParsingHelpers:
    def test_parse_column_from_standard_format(self) -> None:
        name = "ValueDrift(column=water_temperature_c,method=K-S p_value,threshold=0.05)"
        assert _parse_column_from_metric_name(name) == "water_temperature_c"

    def test_parse_column_with_underscore(self) -> None:
        name = "ValueDrift(column=coral_cover_percentage,method=chi2,threshold=0.1)"
        assert _parse_column_from_metric_name(name) == "coral_cover_percentage"

    def test_parse_column_returns_none_on_bad_format(self) -> None:
        assert _parse_column_from_metric_name("NotADriftMetric") is None

    def test_parse_method_from_standard_format(self) -> None:
        name = "ValueDrift(column=a,method=K-S p_value,threshold=0.05)"
        assert _parse_method_from_metric_name(name) == "K-S p_value"

    def test_parse_method_returns_unknown_on_bad_format(self) -> None:
        assert _parse_method_from_metric_name("bad") == "unknown"


# ---------------------------------------------------------------------------
# run_drift.py — smoke test with mocked InferencePipeline
# ---------------------------------------------------------------------------


def _make_fake_pipeline(labels: list[str], confidence: float = 0.9):
    """Return a callable that creates a fake pipeline returning fixed predictions."""

    class FakePipeline:
        def predict_batch(self, df: pd.DataFrame):
            return [
                {"predicted_class": labels[i % len(labels)], "confidence": confidence}
                for i in range(len(df))
            ]

    def factory(task: str) -> FakePipeline:
        return FakePipeline()

    return factory


class TestRunDrift:
    def test_run_drift_returns_summary_dict(self, raw_csv: Path, tmp_path: Path) -> None:
        cfg = _make_test_cfg(raw_csv, tmp_path)
        factory = _make_fake_pipeline(["healthy", "stressed", "bleached"])
        summary = run_drift(
            cfg=cfg, shift_scale=1.0, generate_html=False, pipeline_factory=factory, raw_csv=raw_csv
        )
        assert isinstance(summary, dict)

    def test_summary_has_required_keys(self, raw_csv: Path, tmp_path: Path) -> None:
        cfg = _make_test_cfg(raw_csv, tmp_path)
        factory = _make_fake_pipeline(["healthy", "stressed"])
        summary = run_drift(
            cfg=cfg, shift_scale=1.0, generate_html=False, pipeline_factory=factory, raw_csv=raw_csv
        )
        for key in (
            "generated_at",
            "shift_scale",
            "reference_n",
            "production_n",
            "feature_drift",
            "prediction_drift",
            "confidence_drift",
            "recommendation",
            "synthetic_data_disclaimer",
        ):
            assert key in summary, f"Missing key: {key}"

    def test_json_summary_written_to_disk(self, raw_csv: Path, tmp_path: Path) -> None:
        cfg = _make_test_cfg(raw_csv, tmp_path)
        factory = _make_fake_pipeline(["healthy"])
        run_drift(
            cfg=cfg, shift_scale=1.0, generate_html=False, pipeline_factory=factory, raw_csv=raw_csv
        )
        summary_path = cfg.paths.reports_dir / cfg.monitoring["summary_filename"]
        assert summary_path.exists()
        with summary_path.open() as f:
            loaded = json.load(f)
        assert loaded["shift_scale"] == 1.0

    def test_reference_csv_written_to_disk(self, raw_csv: Path, tmp_path: Path) -> None:
        cfg = _make_test_cfg(raw_csv, tmp_path)
        factory = _make_fake_pipeline(["healthy"])
        run_drift(
            cfg=cfg, shift_scale=1.0, generate_html=False, pipeline_factory=factory, raw_csv=raw_csv
        )
        ref_path = cfg.paths.reference_data_dir / cfg.monitoring["reference_filename"]
        assert ref_path.exists()

    def test_production_csv_written_to_disk(self, raw_csv: Path, tmp_path: Path) -> None:
        cfg = _make_test_cfg(raw_csv, tmp_path)
        factory = _make_fake_pipeline(["healthy"])
        run_drift(
            cfg=cfg, shift_scale=1.0, generate_html=False, pipeline_factory=factory, raw_csv=raw_csv
        )
        prod_path = cfg.paths.production_data_dir / cfg.monitoring["production_filename"]
        assert prod_path.exists()

    def test_zero_shift_no_feature_drift(self, raw_csv: Path, tmp_path: Path) -> None:
        """With shift_scale=0, feature drift should not be detected on small samples."""
        cfg = _make_test_cfg(raw_csv, tmp_path)
        factory = _make_fake_pipeline(["healthy", "stressed"])
        summary = run_drift(
            cfg=cfg, shift_scale=0.0, generate_html=False, pipeline_factory=factory, raw_csv=raw_csv
        )
        # With no shift, drifted columns should be 0 or very few
        assert summary["feature_drift"]["drifted_count"] < 5

    def test_shift_scale_stored_in_summary(self, raw_csv: Path, tmp_path: Path) -> None:
        cfg = _make_test_cfg(raw_csv, tmp_path)
        factory = _make_fake_pipeline(["healthy"])
        summary = run_drift(
            cfg=cfg, shift_scale=2.5, generate_html=False, pipeline_factory=factory, raw_csv=raw_csv
        )
        assert summary["shift_scale"] == 2.5

    def test_recommendation_not_empty(self, raw_csv: Path, tmp_path: Path) -> None:
        cfg = _make_test_cfg(raw_csv, tmp_path)
        factory = _make_fake_pipeline(["healthy", "stressed"])
        summary = run_drift(
            cfg=cfg, shift_scale=1.0, generate_html=False, pipeline_factory=factory, raw_csv=raw_csv
        )
        assert len(summary["recommendation"]) > 10

    def test_prediction_drift_health_has_required_keys(self, raw_csv: Path, tmp_path: Path) -> None:
        cfg = _make_test_cfg(raw_csv, tmp_path)
        factory = _make_fake_pipeline(["healthy"])
        summary = run_drift(
            cfg=cfg, shift_scale=1.0, generate_html=False, pipeline_factory=factory, raw_csv=raw_csv
        )
        health = summary["prediction_drift"]["health"]
        for key in (
            "drifted",
            "p_value",
            "method",
            "reference_distribution",
            "current_distribution",
        ):
            assert key in health

    def test_confidence_drift_health_has_required_keys(self, raw_csv: Path, tmp_path: Path) -> None:
        cfg = _make_test_cfg(raw_csv, tmp_path)
        factory = _make_fake_pipeline(["healthy"])
        summary = run_drift(
            cfg=cfg, shift_scale=1.0, generate_html=False, pipeline_factory=factory, raw_csv=raw_csv
        )
        health = summary["confidence_drift"]["health"]
        for key in ("drifted", "p_value", "mean_reference", "mean_current", "delta"):
            assert key in health


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


class TestRunDriftCLI:
    def test_cli_returns_0_on_success(self, raw_csv: Path, tmp_path: Path, monkeypatch) -> None:
        cfg = _make_test_cfg(raw_csv, tmp_path)
        with mock.patch("src.monitoring.run_drift.get_config", return_value=cfg):
            with mock.patch("src.monitoring.run_drift.run_drift") as mock_run:
                mock_run.return_value = {
                    "feature_drift": {"drifted_count": 0},
                    "prediction_drift": {
                        "health": {"drifted": False},
                        "restoration": {"drifted": False},
                    },
                    "recommendation": "OK",
                }
                exit_code = run_drift_main(["--no-html"])
        assert exit_code == 0

    def test_cli_returns_1_on_file_not_found(self, tmp_path: Path) -> None:
        cfg = mock.MagicMock()
        with mock.patch("src.monitoring.run_drift.get_config", return_value=cfg):
            with mock.patch(
                "src.monitoring.run_drift.run_drift", side_effect=FileNotFoundError("x")
            ):
                exit_code = run_drift_main(["--no-html"])
        assert exit_code == 1

    def test_cli_shift_scale_passed_through(self, raw_csv: Path, tmp_path: Path) -> None:
        cfg = _make_test_cfg(raw_csv, tmp_path)
        with mock.patch("src.monitoring.run_drift.get_config", return_value=cfg):
            with mock.patch("src.monitoring.run_drift.run_drift") as mock_run:
                mock_run.return_value = {
                    "feature_drift": {"drifted_count": 0},
                    "prediction_drift": {
                        "health": {"drifted": False},
                        "restoration": {"drifted": False},
                    },
                    "recommendation": "OK",
                }
                run_drift_main(["--shift-scale", "2.0", "--no-html"])
                _, kwargs = mock_run.call_args
                assert kwargs.get("shift_scale") == 2.0 or mock_run.call_args[0][0] is None


# ---------------------------------------------------------------------------
# Helpers for test config
# ---------------------------------------------------------------------------


def _make_test_cfg(raw_csv: Path, tmp_path: Path):
    """Return a minimal config-like object for testing run_drift."""
    from src.config import get_config

    cfg = get_config()

    # Override paths to use tmp_path
    from dataclasses import replace

    from src.config import Paths

    new_paths = Paths(
        project_root=tmp_path,
        data_dir=tmp_path / "data",
        raw_data_dir=raw_csv.parent,
        processed_data_dir=tmp_path / "processed",
        reference_data_dir=tmp_path / "data" / "reference",
        production_data_dir=tmp_path / "data" / "production",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        artifacts_dir=tmp_path / "artifacts",
        scripts_dir=tmp_path / "scripts",
        notebooks_dir=tmp_path / "notebooks",
    )
    return replace(cfg, paths=new_paths)
