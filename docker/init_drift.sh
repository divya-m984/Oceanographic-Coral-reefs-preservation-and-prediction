#!/bin/sh
# docker/init_drift.sh — One-shot drift monitoring for the 'drift' Compose profile.
#
# Generates a synthetic drift report without:
#   - modifying the canonical MLflow database
#   - registering any model version
#   - promoting any champion alias
#
# Output is written to /app/reports/ (mounted as a named volume or bind mount).
#
# SYNTHETIC-DATA DISCLAIMER
# Drift results are computed on SYNTHETIC data.  They do not represent real
# ocean conditions and must not be used to guide conservation decisions.

set -e

echo "[drift-init] Starting drift monitoring (synthetic demonstration) …"
echo "[drift-init] shift_scale=${DRIFT_SHIFT_SCALE:-1.0}"

python -m src.monitoring.run_drift \
    --no-html \
    --shift-scale "${DRIFT_SHIFT_SCALE:-1.0}"

echo "[drift-init] Drift report written to /app/reports/drift_summary.json"
echo "[drift-init] Synthetic demonstration complete."
