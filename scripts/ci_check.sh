#!/usr/bin/env bash
# scripts/ci_check.sh — Local equivalent of the GitHub Actions CI pipeline.
#
# Runs the same checks as CI without modifying real project artifacts:
#   - code quality (ruff lint + format check)
#   - full pytest suite (with a disposable MLflow database)
#   - DVC pipeline validation (isolated temp data)
#   - ML smoke test (temp data, temp MLflow DB, no registry writes)
#
# Usage
# -----
#   bash scripts/ci_check.sh
#
# Requirements
# ------------
#   - Project virtual environment activated, OR run with:
#       .venv/bin/python  (the scripts call python directly)
#   - All dependencies from requirements.txt installed.
#
# Safety
# ------
#   - MLFLOW_TRACKING_URI is set to a temporary file for the test run.
#   - The canonical database (artifacts/mlruns.db) is never opened.
#   - No model is promoted; register_candidate is not invoked.
#   - All temporary files are cleaned up automatically by the scripts.
#
# Exit code: 0 if all checks pass, non-zero otherwise.

set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
RUFF="${RUFF:-.venv/bin/ruff}"

# Use a throwaway MLflow DB so the test suite never touches artifacts/mlruns.db.
export MLFLOW_TRACKING_URI="sqlite:///ci_check_mlruns.db"

cleanup() {
    rm -f ci_check_mlruns.db
}
trap cleanup EXIT

echo "========================================"
echo "CoralSense CI Check (local)"
echo "========================================"

echo ""
echo "[1/4] Code quality — ruff lint ..."
"$RUFF" check src/ tests/ scripts/
echo "      ruff lint OK"

echo ""
echo "[1/4] Code quality — ruff format check ..."
"$RUFF" format --check src/ tests/ scripts/
echo "      ruff format OK"

echo ""
echo "[2/4] Test suite ..."
"$PYTHON" -m pytest tests/ -q --tb=short
echo "      Tests OK"

echo ""
echo "[3/4] Pipeline validation ..."
"$PYTHON" scripts/ci_validate_pipeline.py
echo "      Pipeline validation OK"

echo ""
echo "[4/4] ML smoke test ..."
"$PYTHON" scripts/ci_smoke_test.py
echo "      Smoke test OK"

echo ""
echo "========================================"
echo "ALL CI CHECKS PASSED"
echo "========================================"
