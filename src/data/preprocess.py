"""
src/data/preprocess.py — Preprocessing pipeline for CoralSense observations.

Pipeline overview
-----------------
1. Load raw CSV (default: data/raw/observations.csv).
2. Validate the DataFrame against CoralObservationSchema (Pandera).
3. Add six derived features via src.features.build_features.
4. Split into stratified train / test sets — separately for each classification
   task so that class proportions are preserved independently.
5. Fit a ColumnTransformer on the training split ONLY, then transform both.
6. Save raw feature splits (pre-transformer) and fitted transformers to disk.
7. Save feature_metadata.json describing all column decisions.

Saved artefacts (under data/processed/)
----------------------------------------
X_train_health.csv           X_test_health.csv
y_train_health.csv           y_test_health.csv
X_train_restoration.csv      X_test_restoration.csv
y_train_restoration.csv      y_test_restoration.csv
preprocessor_health.joblib   preprocessor_restoration.joblib
feature_metadata.json

Note: X_train/X_test files contain the RAW (pre-transformer) feature columns
including derived features but BEFORE StandardScaler / OneHotEncoder.  The
joblib preprocessors apply those transforms at model-training time, which is
the correct practice for cross-validation (each fold re-applies the
transformer fitted on that fold's training data).

Usage
-----
  python -m src.data.preprocess
  python -m src.data.preprocess --input data/raw/observations.csv
  python -m src.data.preprocess --input data/raw/observations.csv \\
      --output-dir data/processed
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import Config, get_config, setup_logging
from src.data.validate import validate_dataframe
from src.features.build_features import (
    CATEGORICAL_FEATURE_COLUMNS,
    DERIVED_FEATURE_NAMES,
    NUMERIC_FEATURE_COLUMNS,
    add_derived_features,
    build_feature_metadata,
    get_feature_columns,
    get_target_column,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class PreprocessResult:
    """
    Container for all outputs of a preprocessing run.

    Attributes
    ----------
    X_train_health, X_test_health:
        Raw (pre-transformer) feature DataFrames for reef-health task.
    y_train_health, y_test_health:
        Target Series for reef-health task.
    X_train_restoration, X_test_restoration:
        Raw feature DataFrames for restoration-suitability task.
    y_train_restoration, y_test_restoration:
        Target Series for restoration-suitability task.
    preprocessor_health, preprocessor_restoration:
        Fitted sklearn ColumnTransformer objects.
    feature_metadata:
        Dict describing all column decisions; also saved as JSON.
    """

    X_train_health: pd.DataFrame
    X_test_health: pd.DataFrame
    y_train_health: pd.Series
    y_test_health: pd.Series

    X_train_restoration: pd.DataFrame
    X_test_restoration: pd.DataFrame
    y_train_restoration: pd.Series
    y_test_restoration: pd.Series

    preprocessor_health: ColumnTransformer
    preprocessor_restoration: ColumnTransformer

    feature_metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def load_and_validate(input_path: Path, cfg: Config) -> pd.DataFrame:
    """
    Load *input_path* as a CSV and validate it with the Pandera schema.

    Parameters
    ----------
    input_path:
        Path to the raw observations CSV.
    cfg:
        Loaded Config object (used for logging only; validation is schema-driven).

    Returns
    -------
    Validated DataFrame.

    Raises
    ------
    FileNotFoundError:
        If *input_path* does not exist.
    ValueError:
        If the DataFrame fails Pandera schema validation.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info("Loading %s", input_path)
    df = pd.read_csv(input_path)
    logger.info("Loaded %d rows × %d columns", *df.shape)

    logger.info("Validating schema...")
    is_valid, failures = validate_dataframe(df, lazy=True)
    if not is_valid:
        assert failures is not None
        n = len(failures)
        sample = failures.head(5).to_string(index=False)
        raise ValueError(
            f"Schema validation failed with {n} failure(s).\nFirst failures:\n{sample}"
        )
    logger.info("Schema validation PASSED")
    return df


def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    """
    Build an unfitted sklearn ColumnTransformer.

    Numeric pipeline
    ~~~~~~~~~~~~~~~~
    1. SimpleImputer(strategy="median") — safe for skewed distributions.
    2. StandardScaler() — zero mean, unit variance.

    Categorical pipeline
    ~~~~~~~~~~~~~~~~~~~~
    1. SimpleImputer(strategy="most_frequent") — fills any missing region.
    2. OneHotEncoder(handle_unknown="ignore", sparse_output=False) — unknown
       region values at inference time produce all-zero OHE columns without
       raising an exception.

    Parameters
    ----------
    numeric_cols:
        Ordered list of numeric feature column names.
    categorical_cols:
        Ordered list of categorical feature column names.

    Returns
    -------
    Unfitted ColumnTransformer.
    """
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ohe",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def split_and_preprocess(
    df: pd.DataFrame,
    task: Literal["health", "restoration"],
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """
    Split *df* into stratified train/test sets and fit a preprocessor.

    The preprocessor is fitted on the training split ONLY to prevent data
    leakage from the test set into scaling statistics.

    Parameters
    ----------
    df:
        Full DataFrame with derived features already added.
    task:
        ``"health"`` or ``"restoration"``.
    cfg:
        Config object providing split ratios and random seed.

    Returns
    -------
    (X_train, X_test, y_train, y_test, fitted_preprocessor)
    """
    feature_cols = get_feature_columns(task)
    target_col = get_target_column(task)

    X = df[feature_cols]
    y = df[target_col]

    stratify_arg = y if cfg.stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_seed,
        stratify=stratify_arg,
    )

    logger.info(
        "[%s] Split → train %d / test %d (stratify=%s)",
        task,
        len(X_train),
        len(X_test),
        cfg.stratify,
    )

    preprocessor = build_preprocessor(NUMERIC_FEATURE_COLUMNS, CATEGORICAL_FEATURE_COLUMNS)
    preprocessor.fit(X_train)
    logger.info("[%s] Preprocessor fitted on training data only", task)

    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
        preprocessor,
    )


def run_preprocessing(
    input_path: Path,
    output_dir: Path,
    cfg: Config,
) -> PreprocessResult:
    """
    Execute the full preprocessing pipeline and save all artefacts.

    Parameters
    ----------
    input_path:
        Path to the raw observations CSV.
    output_dir:
        Directory under which all processed files are saved.
    cfg:
        Config object.

    Returns
    -------
    PreprocessResult containing all splits and fitted preprocessors.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load + validate ──────────────────────────────────────────────────
    df = load_and_validate(input_path, cfg)

    # ── 2. Feature engineering ──────────────────────────────────────────────
    logger.info("Adding %d derived features", len(DERIVED_FEATURE_NAMES))
    df = add_derived_features(df)
    logger.info("DataFrame shape after feature engineering: %s", df.shape)

    # ── 3. Split + preprocess for each task ─────────────────────────────────
    (X_train_h, X_test_h, y_train_h, y_test_h, pre_h) = split_and_preprocess(df, "health", cfg)

    (X_train_r, X_test_r, y_train_r, y_test_r, pre_r) = split_and_preprocess(df, "restoration", cfg)

    # ── 4. Save splits (raw feature DataFrames) ─────────────────────────────
    _save_splits(
        output_dir,
        task="health",
        X_train=X_train_h,
        X_test=X_test_h,
        y_train=y_train_h,
        y_test=y_test_h,
    )
    _save_splits(
        output_dir,
        task="restoration",
        X_train=X_train_r,
        X_test=X_test_r,
        y_train=y_train_r,
        y_test=y_test_r,
    )

    # ── 5. Save fitted preprocessors ────────────────────────────────────────
    _save_preprocessor(output_dir, "health", pre_h)
    _save_preprocessor(output_dir, "restoration", pre_r)

    # ── 6. Save feature metadata ─────────────────────────────────────────────
    metadata = build_feature_metadata()
    _save_metadata(output_dir, metadata)

    # ── 7. Log distribution summaries ───────────────────────────────────────
    _log_distributions("health", y_train_h, y_test_h)
    _log_distributions("restoration", y_train_r, y_test_r)

    logger.info("Preprocessing complete. Artefacts saved to %s", output_dir)

    return PreprocessResult(
        X_train_health=X_train_h,
        X_test_health=X_test_h,
        y_train_health=y_train_h,
        y_test_health=y_test_h,
        X_train_restoration=X_train_r,
        X_test_restoration=X_test_r,
        y_train_restoration=y_train_r,
        y_test_restoration=y_test_r,
        preprocessor_health=pre_h,
        preprocessor_restoration=pre_r,
        feature_metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _save_splits(
    out_dir: Path,
    task: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    X_train.to_csv(out_dir / f"X_train_{task}.csv", index=False)
    X_test.to_csv(out_dir / f"X_test_{task}.csv", index=False)
    y_train.to_frame().to_csv(out_dir / f"y_train_{task}.csv", index=False)
    y_test.to_frame().to_csv(out_dir / f"y_test_{task}.csv", index=False)
    logger.info(
        "[%s] Saved splits: X_train %s, X_test %s",
        task,
        X_train.shape,
        X_test.shape,
    )


def _save_preprocessor(out_dir: Path, task: str, preprocessor: ColumnTransformer) -> None:
    path = out_dir / f"preprocessor_{task}.joblib"
    joblib.dump(preprocessor, path)
    logger.info("[%s] Preprocessor saved to %s", task, path)


def _save_metadata(out_dir: Path, metadata: dict) -> None:
    path = out_dir / "feature_metadata.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    logger.info("Feature metadata saved to %s", path)


def _log_distributions(task: str, y_train: pd.Series, y_test: pd.Series) -> None:
    logger.info("[%s] Train target distribution:", task)
    for cls, cnt in y_train.value_counts().items():
        logger.info("    %-30s %5d  (%.1f%%)", cls, cnt, 100 * cnt / len(y_train))
    logger.info("[%s] Test target distribution:", task)
    for cls, cnt in y_test.value_counts().items():
        logger.info("    %-30s %5d  (%.1f%%)", cls, cnt, 100 * cnt / len(y_test))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """
    Command-line entry point for the preprocessing pipeline.

    Exit codes
    ----------
    0   Preprocessing completed successfully.
    1   Runtime error (validation failure, missing column, etc.).
    2   Input file not found.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the CoralSense preprocessing pipeline: validate, engineer features, "
            "split, fit ColumnTransformer, and save artefacts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        metavar="PATH",
        help="Input CSV path (default: data/raw/observations.csv from params.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Output directory (default: data/processed/ from params.yaml).",
    )
    args = parser.parse_args()
    setup_logging("coralsense.preprocess")

    cfg = get_config()
    input_path: Path = args.input if args.input is not None else cfg.raw_data_path
    output_dir: Path = (
        args.output_dir if args.output_dir is not None else cfg.paths.processed_data_dir
    )

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return 2

    try:
        result = run_preprocessing(input_path, output_dir, cfg)
    except ValueError as exc:
        logger.error("Preprocessing failed: %s", exc)
        return 1
    except Exception as exc:
        logger.error("Unexpected error: %s", exc)
        return 1

    _print_summary(result, input_path, output_dir)
    return 0


def _print_summary(result: PreprocessResult, input_path: Path, output_dir: Path) -> None:
    sep = "-" * 62
    total = len(result.X_train_health) + len(result.X_test_health)
    print(f"\n{sep}")
    print("  CoralSense -- Preprocessing Summary")
    print(sep)
    print(f"  Input file  : {input_path}")
    print(f"  Output dir  : {output_dir}")
    print(f"  Total rows  : {total:,}")
    print(f"  Features    : {len(result.X_train_health.columns)} columns")
    print(f"  Train rows  : {len(result.X_train_health):,}")
    print(f"  Test rows   : {len(result.X_test_health):,}")
    print(sep)
    print("  Reef health distribution (train):")
    for cls, cnt in result.y_train_health.value_counts().items():
        pct = 100 * cnt / len(result.y_train_health)
        print(f"    {cls:<30} {cnt:5,}  ({pct:.1f}%)")
    print("  Restoration suitability distribution (train):")
    for cls, cnt in result.y_train_restoration.value_counts().items():
        pct = 100 * cnt / len(result.y_train_restoration)
        print(f"    {cls:<30} {cnt:5,}  ({pct:.1f}%)")
    print(sep)
    print()


if __name__ == "__main__":
    sys.exit(main())
