"""
src/data/validate.py — Pandera schema validation for CoralSense observations.

DISCLAIMER
----------
The ranges validated here reflect the *documented prototype ranges* for the
CoralSense synthetic dataset.  They do not represent calibrated real-sensor
specifications.  When replacing synthetic data with real sensor readings in a
later development phase, review and update every Field bound documented below.

Pandera version
---------------
This module targets Pandera 0.20+ and uses the modern ``pandera.pandas``
import path (top-level ``import pandera as pa`` is deprecated in 0.20+ and
will be removed in a future release).

Usage
-----
  # Validate the default generated dataset:
  python -m src.data.validate

  # Validate a custom file and write validated output:
  python -m src.data.validate --input data/raw/observations.csv --output data/raw/observations_validated.csv

  # Validate without saving output:
  python -m src.data.validate --input data/raw/observations.csv --no-output
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError, SchemaErrors
from pandera.pandas import DataFrameModel, Field, dataframe_check
from pandera.typing import Series

from src.config import get_config, setup_logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------
# These match params.yaml and the data dictionary in docs/data_dictionary.md.
# Any change here should be reflected in both documents.

_VALID_REGIONS: list[str] = [
    "Lakshadweep",
    "Gulf of Mannar",
    "Gulf of Kutch",
    "Andaman and Nicobar Islands",
]

_VALID_HEALTH_CLASSES: list[str] = [
    "healthy",
    "stressed",
    "bleached",
    "severely_degraded",
]

_VALID_RESTORATION_CLASSES: list[str] = [
    "suitable",
    "moderately_suitable",
    "unsuitable",
]

# Bounding boxes: [lat_min, lat_max, lon_min, lon_max] — WGS-84 degrees.
# Duplicated here so the schema is self-contained and does not require the
# Config at import time.  Values must stay in sync with params.yaml.
_REGION_BOUNDS: dict[str, list[float]] = {
    "Lakshadweep":                  [10.0, 12.5, 72.0, 74.0],
    "Gulf of Mannar":               [8.5,  10.5, 78.0, 80.5],
    "Gulf of Kutch":                [22.0, 24.5, 68.0, 71.0],
    "Andaman and Nicobar Islands":  [6.5,  14.0, 92.0, 94.0],
}


# ---------------------------------------------------------------------------
# Pandera DataFrameModel schema
# ---------------------------------------------------------------------------


class CoralObservationSchema(DataFrameModel):
    """
    Pandera schema for CoralSense synthetic marine observation data.

    Field bounds
    ------------
    Ranges are deliberately wider than the generator clips so that marginal
    real-sensor values (e.g., a probe reading 41 °C during an extreme heat
    event) are not rejected without cause.  Clearly impossible values (negative
    pH, temperature > 42 °C in an open ocean reef context, etc.) are still
    rejected.

    Cross-field checks
    ------------------
    ``coordinates_within_region_bounds``: each observation's latitude and
    longitude must fall inside the documented bounding box for its assigned
    region.  Region-validity is enforced separately by ``Field(isin=...)``.
    """

    # ── Identifiers / spatial ────────────────────────────────────────────────
    timestamp: Series[pa.DateTime] = Field(
        nullable=False,
        title="Observation timestamp",
        description="UTC date-time of the sensor reading (2018-01-01 to 2024-12-31).",
    )
    latitude: Series[float] = Field(
        ge=-90.0,
        le=90.0,
        nullable=False,
        title="Latitude (°N)",
        description="WGS-84 latitude.  Valid reef locations in this dataset are between 6.5 and 24.5 °N.",
    )
    longitude: Series[float] = Field(
        ge=-180.0,
        le=180.0,
        nullable=False,
        title="Longitude (°E)",
        description="WGS-84 longitude.  Valid reef locations are between 68.0 and 94.0 °E.",
    )
    region: Series[str] = Field(
        isin=_VALID_REGIONS,
        nullable=False,
        title="Reef region",
        description="One of the four supported Indian reef regions.",
    )

    # ── Physical / oceanographic ─────────────────────────────────────────────
    depth_m: Series[float] = Field(
        ge=0.0,
        le=50.0,
        nullable=False,
        title="Depth (m)",
        description="Sensor depth below mean sea level.  Shallow reef systems only.",
    )
    water_temperature_c: Series[float] = Field(
        ge=10.0,
        le=42.0,
        nullable=False,
        title="Water temperature (°C)",
        description="In-situ temperature.  Indian Ocean bleaching threshold ≈ 28.5 °C.",
    )
    ph: Series[float] = Field(
        ge=7.0,
        le=9.0,
        nullable=False,
        title="pH (total scale)",
        description="Seawater pH.  Normal range 8.1–8.3.  Below 8.0 indicates stress.",
    )
    salinity_ppt: Series[float] = Field(
        ge=20.0,
        le=50.0,
        nullable=False,
        title="Salinity (ppt)",
        description="Practical salinity.  Gulf of Kutch reaches 43 ppt (hypersaline).",
    )
    dissolved_oxygen_mg_l: Series[float] = Field(
        ge=0.0,
        le=15.0,
        nullable=False,
        title="Dissolved oxygen (mg/L)",
        description="Hypoxia stress below ~4 mg/L; normal reef water 6–8 mg/L.",
    )
    turbidity_ntu: Series[float] = Field(
        ge=0.0,
        le=100.0,
        nullable=False,
        title="Turbidity (NTU)",
        description="Water column turbidity.  >5 NTU significantly reduces PAR.",
    )
    light_intensity: Series[float] = Field(
        ge=0.0,
        le=3000.0,
        nullable=False,
        title="Light intensity (µmol photons m⁻² s⁻¹)",
        description="PAR reaching substrate.  Coral compensation point ≈ 50 µmol.",
    )
    current_speed_m_s: Series[float] = Field(
        ge=0.0,
        le=5.0,
        nullable=False,
        title="Current speed (m/s)",
        description="Near-bottom current speed.  Optimal reef flushing 0.05–0.40 m/s.",
    )

    # ── Acoustic / structural ────────────────────────────────────────────────
    sonar_backscatter: Series[float] = Field(
        ge=-60.0,
        le=0.0,
        nullable=False,
        title="Sonar backscatter (dB)",
        description="Hard reef: −10 to −3 dB; soft sediment: −30 to −35 dB.",
    )
    rugosity_index: Series[float] = Field(
        ge=1.0,
        le=10.0,
        nullable=False,
        title="Rugosity index (dimensionless)",
        description="Surface complexity.  1.0 = flat; ≥3.5 = complex 3-D reef structure.",
    )
    hard_substrate_percentage: Series[float] = Field(
        ge=0.0,
        le=100.0,
        nullable=False,
        title="Hard substrate (%)",
        description="Fraction of benthic area covered by hard substrate.",
    )
    acoustic_complexity_index: Series[float] = Field(
        ge=0.0,
        le=1.0,
        nullable=False,
        title="Acoustic complexity index (normalised [0,1])",
        description="Higher = richer, more diverse soundscape.  Not a bleaching proxy.",
    )

    # ── Biological observations ──────────────────────────────────────────────
    coral_cover_percentage: Series[float] = Field(
        ge=0.0,
        le=100.0,
        nullable=False,
        title="Coral cover (%)",
        description="Live coral coverage of the benthic transect.",
    )
    bleaching_percentage: Series[float] = Field(
        ge=0.0,
        le=100.0,
        nullable=False,
        title="Bleaching (%)",
        description="Fraction of visible coral tissue showing bleaching.  Driven by thermal stress.",
    )
    disease_percentage: Series[float] = Field(
        ge=0.0,
        le=100.0,
        nullable=False,
        title="Disease prevalence (%)",
        description="Fraction of colonies with disease signs.",
    )

    # ── Target variables ─────────────────────────────────────────────────────
    reef_health: Series[str] = Field(
        isin=_VALID_HEALTH_CLASSES,
        nullable=False,
        title="Reef health class",
        description="Ordinal class: healthy → stressed → bleached → severely_degraded.",
    )
    restoration_suitability: Series[str] = Field(
        isin=_VALID_RESTORATION_CLASSES,
        nullable=False,
        title="Restoration suitability class",
        description="Site suitability for coral restoration: suitable, moderately_suitable, unsuitable.",
    )

    class Config:
        # Coerce types on ingestion: critical for parsing string timestamps from CSV
        # and for accepting integer-typed columns where float is declared.
        coerce = True
        # Allow extra columns — preprocessing steps may add columns without
        # invalidating the raw observation schema.
        strict = False

    # ── Cross-field checks ───────────────────────────────────────────────────

    @dataframe_check
    def coordinates_within_region_bounds(cls, df: pd.DataFrame) -> pd.Series:
        """
        Each observation's latitude and longitude must fall inside the
        documented bounding box for its assigned region.

        If the region value is not one of the four known regions, this check
        returns True (the ``Field(isin=...)`` constraint on ``region`` handles
        that case separately, avoiding duplicated error messages).
        """

        def _row_in_bounds(row: pd.Series) -> bool:
            region = row["region"]
            if region not in _REGION_BOUNDS:
                return True  # already caught by region field check
            lat_min, lat_max, lon_min, lon_max = _REGION_BOUNDS[region]
            return (
                lat_min <= row["latitude"] <= lat_max
                and lon_min <= row["longitude"] <= lon_max
            )

        return df.apply(_row_in_bounds, axis=1)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_dataframe(
    df: pd.DataFrame,
    lazy: bool = True,
) -> tuple[bool, pd.DataFrame | None]:
    """
    Validate *df* against :class:`CoralObservationSchema`.

    Parameters
    ----------
    df:
        The DataFrame to validate.
    lazy:
        When ``True`` (default), all validation failures are collected before
        raising, so every problem is reported at once.  Set to ``False`` for
        fast-fail behaviour.

    Returns
    -------
    (is_valid, failure_cases)
        ``is_valid`` is ``True`` iff validation passed.
        ``failure_cases`` is ``None`` on success, or a DataFrame of failing
        rows/checks on failure (columns: schema_context, column, check,
        check_number, failure_case, index).
    """
    try:
        CoralObservationSchema.validate(df, lazy=lazy)
        return True, None
    except SchemaErrors as e:
        return False, e.failure_cases
    except SchemaError as e:
        # Non-lazy mode raises SchemaError (singular)
        return False, pd.DataFrame(
            {
                "schema_context": [str(e.schema)],
                "column": [str(getattr(e, "schema_context", "unknown"))],
                "check": [str(e.check)],
                "check_number": [None],
                "failure_case": [str(e.failure_cases)],
                "index": [None],
            }
        )


def _format_failure_report(failures: pd.DataFrame, max_rows: int = 30) -> str:
    """Format the failure_cases DataFrame as a human-readable string."""
    lines: list[str] = []
    n = len(failures)
    shown = failures.head(max_rows)
    for _, row in shown.iterrows():
        col   = row.get("column", "?")
        check = row.get("check", "?")
        val   = row.get("failure_case", "?")
        idx   = row.get("index", "?")
        ctx   = row.get("schema_context", "")
        lines.append(f"  [{ctx}] column={col!r}  check={check!r}  value={val!r}  row={idx}")
    if n > max_rows:
        lines.append(f"  ... and {n - max_rows} more failure(s) not shown.")
    return "\n".join(lines)


def print_success_summary(df: pd.DataFrame, input_path: Path) -> None:
    """Print a concise validation-passed summary."""
    sep = "-" * 62
    print(f"\n{sep}")
    print("  CoralSense -- Schema Validation Summary")
    print(sep)
    print("  STATUS        : PASSED")
    print(f"  Input file    : {input_path}")
    print(f"  Rows          : {len(df):,}")
    print(f"  Columns       : {df.shape[1]}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print(sep)
    print()


def print_failure_report(
    failures: pd.DataFrame,
    input_path: Path,
) -> None:
    """Print a detailed validation-failed report to stdout."""
    sep = "-" * 62
    print(f"\n{sep}")
    print("  CoralSense -- Schema Validation Summary")
    print(sep)
    print("  STATUS        : FAILED")
    print(f"  Input file    : {input_path}")
    print(f"  Failure count : {len(failures)}")
    print(sep)
    print()
    print("  Validation failures:")
    print(_format_failure_report(failures))
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """
    Command-line entry point for schema validation.

    Exit codes
    ----------
    0   Validation passed.
    1   Validation failed (schema violations found).
    2   File not found or I/O error.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Validate a CoralSense observation CSV against the Pandera schema. "
            "Returns exit code 0 on success, 1 on validation failure."
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
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Write validated (type-coerced) CSV to this path on success.  "
            "Defaults to <input_dir>/observations_validated.csv.  "
            "Pass --no-output to suppress."
        ),
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        default=False,
        help="Do not write a validated output file even if validation passes.",
    )

    args = parser.parse_args()
    setup_logging("coralsense.validate")

    cfg = get_config()

    input_path: Path = args.input if args.input is not None else cfg.raw_data_path

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return 2

    # Derive default output path
    if args.no_output:
        output_path: Path | None = None
    elif args.output is not None:
        output_path = args.output
    else:
        output_path = input_path.parent / "observations_validated.csv"

    logger.info("Validating %s", input_path)

    # Read
    try:
        df = pd.read_csv(input_path)
    except Exception as exc:
        logger.error("Failed to read %s: %s", input_path, exc)
        return 2

    logger.info("Read %d rows x %d columns", *df.shape)

    # Validate
    is_valid, failures = validate_dataframe(df, lazy=True)

    if is_valid:
        print_success_summary(df, input_path)
        if output_path is not None:
            # Validate again (non-lazy) to get the coerced DataFrame for saving
            validated_df = CoralObservationSchema.validate(df, lazy=False)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            validated_df.to_csv(output_path, index=False)
            logger.info("Validated dataset saved to %s", output_path)
        return 0
    else:
        assert failures is not None
        print_failure_report(failures, input_path)
        logger.error(
            "Validation FAILED with %d failure(s). No output file written.", len(failures)
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
