"""
tests/test_validate.py — Pytest suite for src/data/validate.py (M3).

Coverage
--------
- valid generated dataset passes schema
- missing required column
- incorrect data type (non-numeric pH)
- out-of-range pH
- invalid temperature (below minimum)
- invalid percentage (> 100)
- invalid region string
- invalid target class (reef_health / restoration_suitability)
- missing values (NaN in non-nullable field)
- invalid coordinates (lat/lon outside region bounding box)
- CLI success (exit 0, output file written)
- CLI failure (exit 1, no output file)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.config import get_config, reset_config
from src.data.generate_data import generate_observations
from src.data.validate import (
    validate_dataframe,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_REQUIRED_COLUMNS: list[str] = [
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


@pytest.fixture(scope="module")
def cfg():
    reset_config()
    return get_config()


@pytest.fixture(scope="module")
def valid_df(cfg) -> pd.DataFrame:
    """Small but fully valid generated dataset (500 rows, seed=7)."""
    return generate_observations(n_samples=500, seed=7, cfg=cfg)


def _clone(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the index reset."""
    return df.copy().reset_index(drop=True)


# ---------------------------------------------------------------------------
# TestValidDataset
# ---------------------------------------------------------------------------


class TestValidDataset:
    """The generated dataset must pass schema validation without errors."""

    def test_valid_df_passes(self, valid_df: pd.DataFrame) -> None:
        is_valid, failures = validate_dataframe(valid_df, lazy=True)
        assert is_valid, f"Expected valid dataset to pass; failures:\n{failures}"

    def test_failures_none_on_success(self, valid_df: pd.DataFrame) -> None:
        _, failures = validate_dataframe(valid_df, lazy=True)
        assert failures is None

    def test_validate_returns_tuple(self, valid_df: pd.DataFrame) -> None:
        result = validate_dataframe(valid_df, lazy=True)
        assert isinstance(result, tuple) and len(result) == 2

    def test_all_21_columns_present(self, valid_df: pd.DataFrame) -> None:
        for col in _REQUIRED_COLUMNS:
            assert col in valid_df.columns, f"Column '{col}' missing from generated data"

    def test_extra_column_allowed(self, valid_df: pd.DataFrame) -> None:
        """Schema must not reject DataFrames with extra columns (strict=False)."""
        df = _clone(valid_df)
        df["extra_col"] = 0.0
        is_valid, _ = validate_dataframe(df, lazy=True)
        assert is_valid


# ---------------------------------------------------------------------------
# TestMissingColumn
# ---------------------------------------------------------------------------


class TestMissingColumn:
    """Dropping a required column must trigger a validation failure."""

    @pytest.mark.parametrize("col", ["reef_health", "timestamp", "region", "ph"])
    def test_missing_column_fails(self, valid_df: pd.DataFrame, col: str) -> None:
        df = _clone(valid_df).drop(columns=[col])
        is_valid, failures = validate_dataframe(df, lazy=True)
        assert not is_valid, f"Expected failure after dropping '{col}'"
        assert failures is not None and len(failures) > 0


# ---------------------------------------------------------------------------
# TestDtype
# ---------------------------------------------------------------------------


class TestDtype:
    """Non-numeric values in numeric columns must fail after coercion."""

    def test_string_in_ph_column_fails(self, valid_df: pd.DataFrame) -> None:
        df = _clone(valid_df)
        # Replace entire column with non-numeric strings that cannot be coerced
        df["ph"] = "not_a_number"
        is_valid, failures = validate_dataframe(df, lazy=True)
        assert not is_valid
        assert failures is not None


# ---------------------------------------------------------------------------
# TestOutOfRange
# ---------------------------------------------------------------------------


class TestOutOfRange:
    """Values outside declared Field bounds must be caught."""

    def test_ph_below_minimum_fails(self, valid_df: pd.DataFrame) -> None:
        df = _clone(valid_df)
        df.loc[0, "ph"] = 5.0  # below ge=7.0
        is_valid, failures = validate_dataframe(df, lazy=True)
        assert not is_valid
        assert failures is not None
        assert any(failures["column"].str.contains("ph", na=False))

    def test_ph_above_maximum_fails(self, valid_df: pd.DataFrame) -> None:
        df = _clone(valid_df)
        df.loc[0, "ph"] = 10.0  # above le=9.0
        is_valid, failures = validate_dataframe(df, lazy=True)
        assert not is_valid

    def test_temperature_below_minimum_fails(self, valid_df: pd.DataFrame) -> None:
        df = _clone(valid_df)
        df.loc[0, "water_temperature_c"] = -5.0  # below ge=10.0
        is_valid, failures = validate_dataframe(df, lazy=True)
        assert not is_valid
        assert failures is not None
        assert any(failures["column"].str.contains("water_temperature_c", na=False))

    def test_temperature_above_maximum_fails(self, valid_df: pd.DataFrame) -> None:
        df = _clone(valid_df)
        df.loc[0, "water_temperature_c"] = 50.0  # above le=42.0
        is_valid, failures = validate_dataframe(df, lazy=True)
        assert not is_valid

    def test_sonar_backscatter_above_zero_fails(self, valid_df: pd.DataFrame) -> None:
        df = _clone(valid_df)
        df.loc[0, "sonar_backscatter"] = 5.0  # above le=0.0
        is_valid, _ = validate_dataframe(df, lazy=True)
        assert not is_valid

    def test_depth_negative_fails(self, valid_df: pd.DataFrame) -> None:
        df = _clone(valid_df)
        df.loc[0, "depth_m"] = -1.0  # below ge=0.0
        is_valid, _ = validate_dataframe(df, lazy=True)
        assert not is_valid

    def test_rugosity_below_minimum_fails(self, valid_df: pd.DataFrame) -> None:
        df = _clone(valid_df)
        df.loc[0, "rugosity_index"] = 0.5  # below ge=1.0
        is_valid, _ = validate_dataframe(df, lazy=True)
        assert not is_valid

    def test_acoustic_complexity_above_one_fails(self, valid_df: pd.DataFrame) -> None:
        df = _clone(valid_df)
        df.loc[0, "acoustic_complexity_index"] = 1.5  # above le=1.0
        is_valid, _ = validate_dataframe(df, lazy=True)
        assert not is_valid


# ---------------------------------------------------------------------------
# TestInvalidPercentage
# ---------------------------------------------------------------------------


class TestInvalidPercentage:
    """Percentage columns are bounded [0, 100]; violations must be caught."""

    @pytest.mark.parametrize(
        "col",
        [
            "coral_cover_percentage",
            "bleaching_percentage",
            "disease_percentage",
            "hard_substrate_percentage",
        ],
    )
    def test_percentage_above_100_fails(self, valid_df: pd.DataFrame, col: str) -> None:
        df = _clone(valid_df)
        df.loc[0, col] = 105.0
        is_valid, failures = validate_dataframe(df, lazy=True)
        assert not is_valid
        assert failures is not None
        assert any(failures["column"].str.contains(col, na=False))

    @pytest.mark.parametrize(
        "col",
        [
            "coral_cover_percentage",
            "bleaching_percentage",
        ],
    )
    def test_percentage_below_zero_fails(self, valid_df: pd.DataFrame, col: str) -> None:
        df = _clone(valid_df)
        df.loc[0, col] = -1.0
        is_valid, _ = validate_dataframe(df, lazy=True)
        assert not is_valid


# ---------------------------------------------------------------------------
# TestInvalidRegion
# ---------------------------------------------------------------------------


class TestInvalidRegion:
    """Unknown region strings must be rejected."""

    def test_unknown_region_fails(self, valid_df: pd.DataFrame) -> None:
        df = _clone(valid_df)
        df.loc[0, "region"] = "Pacific Ocean"
        is_valid, failures = validate_dataframe(df, lazy=True)
        assert not is_valid
        assert failures is not None
        assert any(failures["column"].str.contains("region", na=False))

    def test_empty_region_fails(self, valid_df: pd.DataFrame) -> None:
        df = _clone(valid_df)
        df.loc[0, "region"] = ""
        is_valid, _ = validate_dataframe(df, lazy=True)
        assert not is_valid

    def test_all_four_valid_regions_accepted(self, valid_df: pd.DataFrame) -> None:
        """Check each region individually to ensure none are accidentally blocked."""
        valid_regions = [
            "Lakshadweep",
            "Gulf of Mannar",
            "Gulf of Kutch",
            "Andaman and Nicobar Islands",
        ]
        for region in valid_regions:
            mask = valid_df["region"] == region
            if not mask.any():
                continue
            subset = valid_df[mask].head(5).copy().reset_index(drop=True)
            is_valid, _ = validate_dataframe(subset, lazy=True)
            assert is_valid, f"Region '{region}' unexpectedly rejected"


# ---------------------------------------------------------------------------
# TestInvalidTargetClass
# ---------------------------------------------------------------------------


class TestInvalidTargetClass:
    """Invalid label strings for target variables must be caught."""

    def test_invalid_reef_health_fails(self, valid_df: pd.DataFrame) -> None:
        df = _clone(valid_df)
        df.loc[0, "reef_health"] = "critical"
        is_valid, failures = validate_dataframe(df, lazy=True)
        assert not is_valid
        assert failures is not None
        assert any(failures["column"].str.contains("reef_health", na=False))

    def test_invalid_restoration_class_fails(self, valid_df: pd.DataFrame) -> None:
        df = _clone(valid_df)
        df.loc[0, "restoration_suitability"] = "perfect"
        is_valid, failures = validate_dataframe(df, lazy=True)
        assert not is_valid
        assert failures is not None
        assert any(failures["column"].str.contains("restoration_suitability", na=False))

    def test_all_valid_health_classes_accepted(self, cfg) -> None:
        """Confirm each valid health label is not blocked."""
        from src.data.validate import _VALID_HEALTH_CLASSES

        valid_health = _VALID_HEALTH_CLASSES
        # Build a minimal 1-row DataFrame and test each class
        base = generate_observations(n_samples=5, seed=1, cfg=cfg).iloc[[0]].copy()
        for cls in valid_health:
            df = base.copy()
            df["reef_health"] = cls
            is_valid, _ = validate_dataframe(df, lazy=True)
            assert is_valid, f"Health class '{cls}' unexpectedly rejected"

    def test_all_valid_restoration_classes_accepted(self, cfg) -> None:
        from src.data.validate import _VALID_RESTORATION_CLASSES

        valid_rest = _VALID_RESTORATION_CLASSES
        base = generate_observations(n_samples=5, seed=1, cfg=cfg).iloc[[0]].copy()
        for cls in valid_rest:
            df = base.copy()
            df["restoration_suitability"] = cls
            is_valid, _ = validate_dataframe(df, lazy=True)
            assert is_valid, f"Restoration class '{cls}' unexpectedly rejected"


# ---------------------------------------------------------------------------
# TestMissingValues
# ---------------------------------------------------------------------------


class TestMissingValues:
    """NaN in nullable=False columns must fail validation."""

    @pytest.mark.parametrize(
        "col",
        [
            "timestamp",
            "latitude",
            "ph",
            "water_temperature_c",
            "reef_health",
            "region",
        ],
    )
    def test_nan_in_non_nullable_field_fails(self, valid_df: pd.DataFrame, col: str) -> None:
        df = _clone(valid_df).astype(object)  # cast to object to allow mixed NaN
        df.loc[0, col] = None
        is_valid, failures = validate_dataframe(df, lazy=True)
        assert not is_valid, f"Expected failure for NaN in non-nullable column '{col}'"
        assert failures is not None


# ---------------------------------------------------------------------------
# TestInvalidCoordinates
# ---------------------------------------------------------------------------


class TestInvalidCoordinates:
    """Lat/lon pairs outside a region's bounding box must fail the cross-field check."""

    def test_lakshadweep_wrong_coordinates_fails(self, valid_df: pd.DataFrame) -> None:
        df = _clone(valid_df)
        # Force first Lakshadweep row to use Gulf of Mannar coordinates
        mask = df["region"] == "Lakshadweep"
        if not mask.any():
            pytest.skip("No Lakshadweep rows in fixture")
        idx = df[mask].index[0]
        df.loc[idx, "latitude"] = 9.0  # Gulf of Mannar range, outside Lakshadweep [10,12.5]
        df.loc[idx, "longitude"] = 79.0  # Gulf of Mannar range, outside Lakshadweep [72,74]
        is_valid, failures = validate_dataframe(df, lazy=True)
        assert not is_valid
        assert failures is not None

    def test_gulf_of_kutch_wrong_coordinates_fails(self, valid_df: pd.DataFrame) -> None:
        df = _clone(valid_df)
        mask = df["region"] == "Gulf of Kutch"
        if not mask.any():
            pytest.skip("No Gulf of Kutch rows in fixture")
        idx = df[mask].index[0]
        df.loc[idx, "latitude"] = 5.0  # far south, below 22.0 minimum
        df.loc[idx, "longitude"] = 80.0  # east of 71.0 maximum
        is_valid, failures = validate_dataframe(df, lazy=True)
        assert not is_valid

    def test_coordinates_on_boundary_pass(self, valid_df: pd.DataFrame) -> None:
        """Exact boundary values (ge/le inclusive) must be accepted."""
        df = _clone(valid_df)
        mask = df["region"] == "Lakshadweep"
        if not mask.any():
            pytest.skip("No Lakshadweep rows in fixture")
        idx = df[mask].index[0]
        # Set coordinates exactly on the Lakshadweep boundary
        df.loc[idx, "latitude"] = 10.0  # lat_min exactly
        df.loc[idx, "longitude"] = 72.0  # lon_min exactly
        is_valid, failures = validate_dataframe(df, lazy=True)
        assert is_valid, f"Boundary coordinates should pass; failures:\n{failures}"

    def test_globally_valid_but_wrong_region_coordinates_fails(
        self, valid_df: pd.DataFrame
    ) -> None:
        """Coordinates valid globally (within ±90/±180) but wrong for assigned region."""
        df = _clone(valid_df)
        mask = df["region"] == "Andaman and Nicobar Islands"
        if not mask.any():
            pytest.skip("No Andaman rows in fixture")
        idx = df[mask].index[0]
        # Tasmania coordinates: globally valid, but not in Andaman bounding box [6.5-14, 92-94]
        df.loc[idx, "latitude"] = -43.0
        df.loc[idx, "longitude"] = 147.0
        is_valid, _ = validate_dataframe(df, lazy=True)
        assert not is_valid


# ---------------------------------------------------------------------------
# TestMultipleFailures
# ---------------------------------------------------------------------------


class TestMultipleFailures:
    """Lazy validation must collect ALL failures before raising."""

    def test_multiple_failures_reported(self, valid_df: pd.DataFrame) -> None:
        df = _clone(valid_df)
        df.loc[0, "ph"] = -1.0  # out of range
        df.loc[1, "reef_health"] = "zombie"  # invalid class
        df.loc[2, "region"] = "Mars"  # invalid region
        is_valid, failures = validate_dataframe(df, lazy=True)
        assert not is_valid
        assert failures is not None
        # Should have at least 3 distinct failures
        assert len(failures) >= 3


# ---------------------------------------------------------------------------
# TestCLI
# ---------------------------------------------------------------------------


class TestCLI:
    """CLI exit codes and output file behaviour."""

    _RAW_CSV = Path("data/raw/observations.csv")
    _MODULE = "src.data.validate"

    def _run_cli(self, *extra_args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", self._MODULE, *extra_args],
            capture_output=True,
            text=True,
        )

    def test_cli_success_exit_code(self) -> None:
        if not self._RAW_CSV.exists():
            pytest.skip("data/raw/observations.csv not present")
        result = self._run_cli("--input", str(self._RAW_CSV), "--no-output")
        assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    def test_cli_success_stdout_contains_passed(self) -> None:
        if not self._RAW_CSV.exists():
            pytest.skip("data/raw/observations.csv not present")
        result = self._run_cli("--input", str(self._RAW_CSV), "--no-output")
        assert "PASSED" in result.stdout

    def test_cli_success_writes_output_file(self, tmp_path: Path) -> None:
        if not self._RAW_CSV.exists():
            pytest.skip("data/raw/observations.csv not present")
        out_file = tmp_path / "validated.csv"
        result = self._run_cli(
            "--input",
            str(self._RAW_CSV),
            "--output",
            str(out_file),
        )
        assert result.returncode == 0
        assert out_file.exists(), "Validated output file was not created"
        df_out = pd.read_csv(out_file)
        df_in = pd.read_csv(self._RAW_CSV)
        assert len(df_out) == len(df_in)

    def test_cli_failure_exit_code(self, tmp_path: Path, cfg) -> None:
        """Writing a CSV with a bad value should yield exit code 1."""
        bad_csv = tmp_path / "bad.csv"
        df = generate_observations(n_samples=50, seed=42, cfg=cfg)
        df.loc[0, "ph"] = -99.0
        df.to_csv(bad_csv, index=False)
        result = self._run_cli("--input", str(bad_csv), "--no-output")
        assert result.returncode == 1

    def test_cli_failure_stdout_contains_failed(self, tmp_path: Path, cfg) -> None:
        bad_csv = tmp_path / "bad.csv"
        df = generate_observations(n_samples=50, seed=42, cfg=cfg)
        df.loc[0, "reef_health"] = "UNKNOWN"
        df.to_csv(bad_csv, index=False)
        result = self._run_cli("--input", str(bad_csv), "--no-output")
        assert "FAILED" in result.stdout

    def test_cli_failure_no_output_file_written(self, tmp_path: Path, cfg) -> None:
        bad_csv = tmp_path / "bad.csv"
        out_file = tmp_path / "should_not_exist.csv"
        df = generate_observations(n_samples=50, seed=42, cfg=cfg)
        df.loc[0, "ph"] = -1.0
        df.to_csv(bad_csv, index=False)
        result = self._run_cli(
            "--input",
            str(bad_csv),
            "--output",
            str(out_file),
        )
        assert result.returncode == 1
        assert not out_file.exists(), "Output file must not be written on validation failure"

    def test_cli_missing_input_file_exit_code(self) -> None:
        result = self._run_cli("--input", "nonexistent_file_xyz.csv", "--no-output")
        assert result.returncode == 2

    def test_cli_no_output_flag_suppresses_file(self) -> None:
        if not self._RAW_CSV.exists():
            pytest.skip("data/raw/observations.csv not present")
        default_output = self._RAW_CSV.parent / "observations_validated.csv"
        # Remove any pre-existing validated file to get a clean state
        existed_before = default_output.exists()
        result = self._run_cli("--input", str(self._RAW_CSV), "--no-output")
        assert result.returncode == 0
        if not existed_before:
            assert not default_output.exists(), "--no-output should not create default output"
