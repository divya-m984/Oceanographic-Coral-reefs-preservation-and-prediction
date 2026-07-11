"""
tests/test_preprocess.py — Tests for src/data/preprocess.py (M4).

Coverage
--------
- load_and_validate: valid input, missing file, schema failure
- add_derived_features: integration (via preprocess)
- split reproducibility and stratification (health + restoration)
- target leakage prevention (targets not in X, one target not predicting other)
- no NaN / Inf in processed transform output
- preprocessor fitted only on training data (no test-set leakage)
- correct output shapes
- unknown-category handling by OHE
- CLI success (exit 0, all artefacts written) and failure (exit 1, exit 2)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src.config import get_config, reset_config
from src.data.generate_data import generate_observations
from src.data.preprocess import (
    PreprocessResult,
    build_preprocessor,
    load_and_validate,
    run_preprocessing,
)
from src.features.build_features import (
    ALL_FEATURE_COLUMNS,
    CATEGORICAL_FEATURE_COLUMNS,
    DERIVED_FEATURE_NAMES,
    NUMERIC_FEATURE_COLUMNS,
    TARGET_COLUMNS,
    add_derived_features,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MODULE = "src.data.preprocess"


@pytest.fixture(scope="module")
def cfg():
    reset_config()
    return get_config()


@pytest.fixture(scope="module")
def raw_df(cfg):
    """600-row generated dataset for fast tests."""
    return generate_observations(n_samples=600, seed=11, cfg=cfg)


@pytest.fixture(scope="module")
def feat_df(raw_df):
    return add_derived_features(raw_df)


@pytest.fixture(scope="module")
def result(raw_df, cfg, tmp_path_factory):
    """Full preprocessing result from a generated 600-row dataset."""
    tmp = tmp_path_factory.mktemp("processed_module")
    # Write raw CSV to tmp dir so load_and_validate can read it
    raw_csv = tmp / "observations.csv"
    raw_df.to_csv(raw_csv, index=False)
    return run_preprocessing(raw_csv, tmp, cfg)


# ---------------------------------------------------------------------------
# TestLoadAndValidate
# ---------------------------------------------------------------------------


class TestLoadAndValidate:
    def test_valid_file_loads(self, raw_df, cfg, tmp_path) -> None:
        csv = tmp_path / "obs.csv"
        raw_df.to_csv(csv, index=False)
        df = load_and_validate(csv, cfg)
        assert len(df) == len(raw_df)

    def test_missing_file_raises_file_not_found(self, cfg) -> None:
        with pytest.raises(FileNotFoundError):
            load_and_validate(Path("nonexistent_xyz.csv"), cfg)

    def test_schema_failure_raises_value_error(self, raw_df, cfg, tmp_path) -> None:
        df = raw_df.copy()
        df.loc[0, "ph"] = -999.0  # invalid
        csv = tmp_path / "bad.csv"
        df.to_csv(csv, index=False)
        with pytest.raises(ValueError, match="Schema validation failed"):
            load_and_validate(csv, cfg)

    def test_returns_dataframe(self, raw_df, cfg, tmp_path) -> None:
        csv = tmp_path / "obs.csv"
        raw_df.to_csv(csv, index=False)
        df = load_and_validate(csv, cfg)
        assert isinstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# TestDerivedFeaturesIntegration
# ---------------------------------------------------------------------------


class TestDerivedFeaturesIntegration:
    """Derived features must be present in the X splits produced by run_preprocessing."""

    def test_derived_features_in_X_train_health(self, result: PreprocessResult) -> None:
        for col in DERIVED_FEATURE_NAMES:
            assert col in result.X_train_health.columns

    def test_derived_features_in_X_train_restoration(self, result: PreprocessResult) -> None:
        for col in DERIVED_FEATURE_NAMES:
            assert col in result.X_train_restoration.columns

    def test_X_feature_count(self, result: PreprocessResult) -> None:
        # 22 feature columns: 21 numeric + 1 categorical
        assert result.X_train_health.shape[1] == len(ALL_FEATURE_COLUMNS)

    def test_no_target_in_X(self, result: PreprocessResult) -> None:
        for tgt in TARGET_COLUMNS:
            assert tgt not in result.X_train_health.columns
            assert tgt not in result.X_train_restoration.columns


# ---------------------------------------------------------------------------
# TestTargetLeakagePrevention
# ---------------------------------------------------------------------------


class TestTargetLeakagePrevention:
    def test_reef_health_not_in_restoration_features(self, result: PreprocessResult) -> None:
        assert "reef_health" not in result.X_train_restoration.columns
        assert "reef_health" not in result.X_test_restoration.columns

    def test_restoration_not_in_health_features(self, result: PreprocessResult) -> None:
        assert "restoration_suitability" not in result.X_train_health.columns
        assert "restoration_suitability" not in result.X_test_health.columns

    def test_metadata_not_in_features(self, result: PreprocessResult) -> None:
        for meta_col in ["timestamp", "latitude", "longitude"]:
            assert meta_col not in result.X_train_health.columns
            assert meta_col not in result.X_train_restoration.columns


# ---------------------------------------------------------------------------
# TestSplitReproducibility
# ---------------------------------------------------------------------------


class TestSplitReproducibility:
    def test_health_split_reproducible(self, raw_df, cfg, tmp_path) -> None:
        csv = tmp_path / "obs.csv"
        raw_df.to_csv(csv, index=False)
        r1 = run_preprocessing(csv, tmp_path / "out1", cfg)
        r2 = run_preprocessing(csv, tmp_path / "out2", cfg)
        pd.testing.assert_frame_equal(r1.X_train_health, r2.X_train_health)
        pd.testing.assert_series_equal(r1.y_train_health, r2.y_train_health)

    def test_restoration_split_reproducible(self, raw_df, cfg, tmp_path) -> None:
        csv = tmp_path / "obs.csv"
        raw_df.to_csv(csv, index=False)
        r1 = run_preprocessing(csv, tmp_path / "out1", cfg)
        r2 = run_preprocessing(csv, tmp_path / "out2", cfg)
        pd.testing.assert_frame_equal(r1.X_train_restoration, r2.X_train_restoration)

    def test_different_seeds_produce_different_splits(self, raw_df, cfg, tmp_path) -> None:
        import dataclasses

        csv = tmp_path / "obs.csv"
        raw_df.to_csv(csv, index=False)
        # Temporarily create a cfg with different seed
        cfg2 = dataclasses.replace(cfg, random_seed=cfg.random_seed + 1)
        r1 = run_preprocessing(csv, tmp_path / "out_s1", cfg)
        r2 = run_preprocessing(csv, tmp_path / "out_s2", cfg2)
        # The training indices will differ with different seeds
        assert not r1.y_train_health.equals(r2.y_train_health)


# ---------------------------------------------------------------------------
# TestStratification
# ---------------------------------------------------------------------------


class TestStratification:
    """Class proportions in train and test must be approximately equal."""

    def _proportions(self, series: pd.Series) -> dict:
        return (series.value_counts(normalize=True)).to_dict()

    def test_health_class_proportions_preserved(self, result: PreprocessResult) -> None:
        train_p = self._proportions(result.y_train_health)
        test_p = self._proportions(result.y_test_health)
        for cls in train_p:
            if cls in test_p:
                assert abs(train_p[cls] - test_p[cls]) < 0.06, (
                    f"Health class '{cls}' proportion drift: "
                    f"train={train_p[cls]:.3f}, test={test_p[cls]:.3f}"
                )

    def test_restoration_class_proportions_preserved(self, result: PreprocessResult) -> None:
        train_p = self._proportions(result.y_train_restoration)
        test_p = self._proportions(result.y_test_restoration)
        for cls in train_p:
            if cls in test_p:
                assert abs(train_p[cls] - test_p[cls]) < 0.06

    def test_all_health_classes_in_train(self, result: PreprocessResult, cfg) -> None:
        present = set(result.y_train_health.unique())
        expected = set(cfg.health_classes)
        assert expected.issubset(present), f"Missing health classes in train: {expected - present}"

    def test_all_restoration_classes_in_train(self, result: PreprocessResult, cfg) -> None:
        present = set(result.y_train_restoration.unique())
        expected = set(cfg.restoration_classes)
        assert expected.issubset(present)


# ---------------------------------------------------------------------------
# TestOutputShapes
# ---------------------------------------------------------------------------


class TestOutputShapes:
    def test_total_rows_health(self, result: PreprocessResult) -> None:
        total = len(result.X_train_health) + len(result.X_test_health)
        assert total == 600

    def test_total_rows_restoration(self, result: PreprocessResult) -> None:
        total = len(result.X_train_restoration) + len(result.X_test_restoration)
        assert total == 600

    def test_test_size_approximately_correct(self, result: PreprocessResult, cfg) -> None:
        test_frac = len(result.X_test_health) / (
            len(result.X_train_health) + len(result.X_test_health)
        )
        assert abs(test_frac - cfg.test_size) < 0.02

    def test_X_and_y_lengths_match_health(self, result: PreprocessResult) -> None:
        assert len(result.X_train_health) == len(result.y_train_health)
        assert len(result.X_test_health) == len(result.y_test_health)

    def test_X_and_y_lengths_match_restoration(self, result: PreprocessResult) -> None:
        assert len(result.X_train_restoration) == len(result.y_train_restoration)
        assert len(result.X_test_restoration) == len(result.y_test_restoration)


# ---------------------------------------------------------------------------
# TestNoNaNOrInf
# ---------------------------------------------------------------------------


class TestNoNaNOrInf:
    """After preprocessing, transformed arrays must contain no NaN or Inf."""

    def _transform(self, preprocessor, X: pd.DataFrame) -> np.ndarray:
        return preprocessor.transform(X)

    def test_no_nan_in_transformed_health_train(self, result: PreprocessResult) -> None:
        arr = self._transform(result.preprocessor_health, result.X_train_health)
        assert not np.isnan(arr).any(), "NaN found in transformed health train data"

    def test_no_inf_in_transformed_health_train(self, result: PreprocessResult) -> None:
        arr = self._transform(result.preprocessor_health, result.X_train_health)
        assert not np.isinf(arr).any()

    def test_no_nan_in_transformed_health_test(self, result: PreprocessResult) -> None:
        arr = self._transform(result.preprocessor_health, result.X_test_health)
        assert not np.isnan(arr).any()

    def test_no_nan_in_transformed_restoration_train(self, result: PreprocessResult) -> None:
        arr = self._transform(result.preprocessor_restoration, result.X_train_restoration)
        assert not np.isnan(arr).any()

    def test_no_nan_in_transformed_restoration_test(self, result: PreprocessResult) -> None:
        arr = self._transform(result.preprocessor_restoration, result.X_test_restoration)
        assert not np.isnan(arr).any()


# ---------------------------------------------------------------------------
# TestPreprocessorFitOnTrainOnly
# ---------------------------------------------------------------------------


class TestPreprocessorFitOnTrainOnly:
    """The fitted preprocessor must encode only training-data statistics."""

    def test_scaler_mean_computed_from_train(self, result: PreprocessResult) -> None:
        """StandardScaler mean should equal the training-data column mean."""
        pre = result.preprocessor_health
        X_train = result.X_train_health
        # Extract the numeric transformer
        scaler = pre.named_transformers_["num"].named_steps["scaler"]
        numeric_means = X_train[NUMERIC_FEATURE_COLUMNS].mean().values
        np.testing.assert_allclose(scaler.mean_, numeric_means, rtol=1e-5)

    def test_scaler_mean_differs_from_test(self, result: PreprocessResult) -> None:
        """Scaler mean should NOT equal the test-set mean (which it would if fit on all data)."""
        pre = result.preprocessor_health
        scaler = pre.named_transformers_["num"].named_steps["scaler"]
        X_test = result.X_test_health
        test_means = X_test[NUMERIC_FEATURE_COLUMNS].mean().values
        # They won't be identical (different samples), confirming fit was on train only
        # (if fit on all data, mean would be the overall mean, not test mean)
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(scaler.mean_, test_means, rtol=1e-3)

    def test_preprocessors_are_fitted_sklearn_objects(self, result: PreprocessResult) -> None:
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(result.preprocessor_health)
        check_is_fitted(result.preprocessor_restoration)


# ---------------------------------------------------------------------------
# TestUnknownCategoryHandling
# ---------------------------------------------------------------------------


class TestUnknownCategoryHandling:
    def test_unknown_region_produces_zero_ohe_row(self, result: PreprocessResult) -> None:
        """OHE with handle_unknown='ignore' must produce all-zero row for unseen region."""
        pre = result.preprocessor_health
        # Build a 1-row DataFrame with an unseen region
        row = result.X_train_health.iloc[[0]].copy()
        row["region"] = "Atlantic Ocean"  # unseen category
        transformed = pre.transform(row)
        n_numeric = len(NUMERIC_FEATURE_COLUMNS)
        ohe_block = transformed[:, n_numeric:]
        assert (ohe_block == 0).all(), (
            "Unknown region should produce all-zero OHE block, got non-zero values"
        )

    def test_unknown_region_does_not_raise(self, result: PreprocessResult) -> None:
        """Transforming an unseen region must not raise any exception."""
        pre = result.preprocessor_health
        row = result.X_train_health.iloc[[0]].copy()
        row["region"] = "Coral Sea"
        try:
            pre.transform(row)
        except Exception as exc:
            pytest.fail(f"transform raised unexpectedly: {exc}")

    def test_known_region_produces_nonzero_ohe(self, result: PreprocessResult) -> None:
        """A known region must produce at least one non-zero OHE column."""
        pre = result.preprocessor_health
        row = result.X_train_health.iloc[[0]].copy()
        transformed = pre.transform(row)
        n_numeric = len(NUMERIC_FEATURE_COLUMNS)
        ohe_block = transformed[:, n_numeric:]
        assert ohe_block.sum() > 0


# ---------------------------------------------------------------------------
# TestSavedArtefacts
# ---------------------------------------------------------------------------


class TestSavedArtefacts:
    """run_preprocessing must persist all expected files to disk."""

    @pytest.fixture(scope="class")
    def saved_dir(self, raw_df, tmp_path_factory):
        reset_config()
        cfg = get_config()
        tmp = tmp_path_factory.mktemp("saved_artefacts")
        csv = tmp / "obs.csv"
        raw_df.to_csv(csv, index=False)
        run_preprocessing(csv, tmp, cfg)
        return tmp

    def test_X_train_health_csv_exists(self, saved_dir) -> None:
        assert (saved_dir / "X_train_health.csv").exists()

    def test_X_test_health_csv_exists(self, saved_dir) -> None:
        assert (saved_dir / "X_test_health.csv").exists()

    def test_y_train_health_csv_exists(self, saved_dir) -> None:
        assert (saved_dir / "y_train_health.csv").exists()

    def test_y_test_health_csv_exists(self, saved_dir) -> None:
        assert (saved_dir / "y_test_health.csv").exists()

    def test_X_train_restoration_csv_exists(self, saved_dir) -> None:
        assert (saved_dir / "X_train_restoration.csv").exists()

    def test_X_test_restoration_csv_exists(self, saved_dir) -> None:
        assert (saved_dir / "X_test_restoration.csv").exists()

    def test_y_train_restoration_csv_exists(self, saved_dir) -> None:
        assert (saved_dir / "y_train_restoration.csv").exists()

    def test_y_test_restoration_csv_exists(self, saved_dir) -> None:
        assert (saved_dir / "y_test_restoration.csv").exists()

    def test_preprocessor_health_joblib_exists(self, saved_dir) -> None:
        assert (saved_dir / "preprocessor_health.joblib").exists()

    def test_preprocessor_restoration_joblib_exists(self, saved_dir) -> None:
        assert (saved_dir / "preprocessor_restoration.joblib").exists()

    def test_feature_metadata_json_exists(self, saved_dir) -> None:
        assert (saved_dir / "feature_metadata.json").exists()

    def test_preprocessor_loads_correctly(self, saved_dir) -> None:
        pre = joblib.load(saved_dir / "preprocessor_health.joblib")
        assert isinstance(pre, ColumnTransformer)

    def test_feature_metadata_json_valid(self, saved_dir) -> None:
        with (saved_dir / "feature_metadata.json").open() as fh:
            meta = json.load(fh)
        assert "all_feature_columns" in meta
        assert len(meta["all_feature_columns"]) == len(ALL_FEATURE_COLUMNS)

    def test_saved_X_train_has_correct_columns(self, saved_dir) -> None:
        df = pd.read_csv(saved_dir / "X_train_health.csv")
        for col in ALL_FEATURE_COLUMNS:
            assert col in df.columns, f"Column '{col}' missing from saved X_train_health.csv"


# ---------------------------------------------------------------------------
# TestBuildPreprocessor
# ---------------------------------------------------------------------------


class TestBuildPreprocessor:
    """Unit tests for build_preprocessor()."""

    def test_returns_column_transformer(self) -> None:
        pre = build_preprocessor(NUMERIC_FEATURE_COLUMNS, CATEGORICAL_FEATURE_COLUMNS)
        assert isinstance(pre, ColumnTransformer)

    def test_has_numeric_transformer(self) -> None:
        pre = build_preprocessor(NUMERIC_FEATURE_COLUMNS, CATEGORICAL_FEATURE_COLUMNS)
        names = [name for name, _, _ in pre.transformers]
        assert "num" in names

    def test_has_categorical_transformer(self) -> None:
        pre = build_preprocessor(NUMERIC_FEATURE_COLUMNS, CATEGORICAL_FEATURE_COLUMNS)
        names = [name for name, _, _ in pre.transformers]
        assert "cat" in names

    def test_remainder_is_drop(self) -> None:
        pre = build_preprocessor(NUMERIC_FEATURE_COLUMNS, CATEGORICAL_FEATURE_COLUMNS)
        assert pre.remainder == "drop"


# ---------------------------------------------------------------------------
# TestCLI
# ---------------------------------------------------------------------------


class TestCLI:
    """CLI exit codes and artefact behaviour."""

    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", _MODULE, *args],
            capture_output=True,
            text=True,
        )

    def test_cli_success_exit_code(self, raw_df, tmp_path) -> None:
        csv = tmp_path / "obs.csv"
        out = tmp_path / "out"
        raw_df.to_csv(csv, index=False)
        result = self._run("--input", str(csv), "--output-dir", str(out))
        assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    def test_cli_success_creates_artefacts(self, raw_df, tmp_path) -> None:
        csv = tmp_path / "obs.csv"
        out = tmp_path / "out"
        raw_df.to_csv(csv, index=False)
        self._run("--input", str(csv), "--output-dir", str(out))
        assert (out / "X_train_health.csv").exists()
        assert (out / "preprocessor_health.joblib").exists()
        assert (out / "feature_metadata.json").exists()

    def test_cli_missing_input_exits_2(self) -> None:
        result = self._run("--input", "nonexistent_file_abc.csv")
        assert result.returncode == 2

    def test_cli_schema_failure_exits_1(self, raw_df, tmp_path) -> None:
        df = raw_df.copy()
        df.loc[0, "ph"] = -999.0
        csv = tmp_path / "bad.csv"
        out = tmp_path / "out"
        df.to_csv(csv, index=False)
        result = self._run("--input", str(csv), "--output-dir", str(out))
        assert result.returncode == 1

    def test_cli_stdout_contains_summary(self, raw_df, tmp_path) -> None:
        csv = tmp_path / "obs.csv"
        out = tmp_path / "out"
        raw_df.to_csv(csv, index=False)
        result = self._run("--input", str(csv), "--output-dir", str(out))
        assert "Preprocessing Summary" in result.stdout
