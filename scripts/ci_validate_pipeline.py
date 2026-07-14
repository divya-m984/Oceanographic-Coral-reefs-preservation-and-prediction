"""
scripts/ci_validate_pipeline.py — CI pipeline validation for CoralSense MLOps.

Validates the DVC pipeline definition and exercises the data generation,
validation, and preprocessing steps in a fully isolated temporary environment.
Safe to run in CI: never writes to project data directories.

Checks
------
1. ``dvc.yaml`` is parseable YAML with all six required stages.
2. No stage command contains a hardcoded absolute user path.
3. 400 synthetic rows pass the Pandera schema validator.
4. Preprocessing produces the expected output files.

Exit codes
----------
0  All checks passed.
1  One or more checks failed.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import yaml

_REQUIRED_STAGES = {
    "generate",
    "validate",
    "preprocess",
    "train",
    "evaluate",
    "register_candidate",
}

_EXPECTED_OUTPUTS = [
    "X_train_health.csv",
    "X_test_health.csv",
    "y_train_health.csv",
    "y_test_health.csv",
    "X_train_restoration.csv",
    "X_test_restoration.csv",
    "y_train_restoration.csv",
    "y_test_restoration.csv",
    "preprocessor_health.joblib",
    "preprocessor_restoration.joblib",
]


def _check_dvc_yaml(project_root: Path) -> list[str]:
    """Return error strings for any structural problems in dvc.yaml."""
    errors: list[str] = []
    dvc_path = project_root / "dvc.yaml"

    if not dvc_path.exists():
        return ["dvc.yaml not found"]

    with dvc_path.open(encoding="utf-8") as fh:
        dvc: dict = yaml.safe_load(fh)

    stages: dict = dvc.get("stages", {})
    missing = _REQUIRED_STAGES - set(stages.keys())
    if missing:
        errors.append(f"dvc.yaml missing required stages: {sorted(missing)}")

    for name, stage in stages.items():
        cmd: str = stage.get("cmd", "")
        for pattern in ("/home/", "/Users/", "/root/"):
            if pattern in cmd:
                errors.append(f"Stage {name!r} contains a hardcoded absolute path: {cmd!r}")

    return errors


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    all_errors: list[str] = []

    print("=" * 60)
    print("CoralSense CI Pipeline Validation")
    print("=" * 60)

    # ── Check 1: dvc.yaml structure ──────────────────────────────────────
    print("\n[1/3] Checking dvc.yaml structure ...")
    errs = _check_dvc_yaml(project_root)
    if errs:
        all_errors.extend(errs)
        for e in errs:
            print(f"  ERROR: {e}")
    else:
        print(f"  OK — all {len(_REQUIRED_STAGES)} required stages present, no absolute paths")

    # ── Check 2: schema validation on temp data ──────────────────────────
    print("\n[2/3] Generating and validating 400 synthetic observations ...")
    from src.config import get_config, reset_config
    from src.data.generate_data import generate_observations
    from src.data.validate import validate_dataframe

    reset_config()
    cfg = get_config()

    df = generate_observations(n_samples=400, seed=123, cfg=cfg)
    try:
        validate_dataframe(df)
        print(f"  OK — {len(df)} rows passed schema validation")
    except Exception as exc:
        all_errors.append(f"Schema validation failed: {exc}")
        print(f"  ERROR: {all_errors[-1]}")

    # ── Check 3: preprocessing round-trip ───────────────────────────────
    print("\n[3/3] Preprocessing to temporary directory ...")
    from src.data.preprocess import run_preprocessing

    with tempfile.TemporaryDirectory(prefix="cs_ci_pipeline_") as tmp_str:
        tmp = Path(tmp_str)
        raw_csv = tmp / "observations.csv"
        processed_dir = tmp / "processed"

        df.to_csv(raw_csv, index=False)

        try:
            run_preprocessing(raw_csv, processed_dir, cfg)
        except Exception as exc:
            all_errors.append(f"Preprocessing failed: {exc}")
            print(f"  ERROR: {all_errors[-1]}")
        else:
            missing_files = [f for f in _EXPECTED_OUTPUTS if not (processed_dir / f).exists()]
            if missing_files:
                for f in missing_files:
                    all_errors.append(f"Expected output not found: {f}")
                print(f"  ERROR: {len(missing_files)} expected output(s) missing")
            else:
                print(
                    f"  OK — {len(_EXPECTED_OUTPUTS)} expected output files present"
                    f" in {processed_dir}"
                )

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if all_errors:
        print("PIPELINE VALIDATION FAILED")
        for e in all_errors:
            print(f"  ERROR: {e}")
        print("=" * 60)
        return 1

    print("PIPELINE VALIDATION PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
