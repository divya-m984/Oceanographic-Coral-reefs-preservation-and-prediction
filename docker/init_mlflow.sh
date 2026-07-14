#!/bin/sh
# docker/init_mlflow.sh — MLflow container initialisation.
#
# Purpose
# -------
# This script runs ONCE before the MLflow server starts.  It:
#   1. Copies the read-only canonical database to a writable runtime location.
#   2. Rewrites host-absolute artifact paths to container paths — detected
#      automatically from the database itself, so no host path is hardcoded.
#   3. Starts the MLflow server on the runtime copy.
#
# The canonical database at /app/artifacts/mlruns.db is NEVER modified.
#
# LIMITATION
# ----------
# The runtime copy is a LOCAL DEMONSTRATION database only.
# It is not the authoritative registry.  After container shutdown the runtime
# copy is discarded (named volume can be cleaned with: docker compose down -v).
#
# SYNTHETIC-DATA DISCLAIMER
# The experiment metadata, run metrics and registered models all reflect
# performance on SYNTHETIC data.  They must not be used to guide real-world
# conservation decisions.

set -e

CANONICAL_DB="/app/artifacts/mlruns.db"
RUNTIME_DIR="/mlflow-runtime"
RUNTIME_DB="${RUNTIME_DIR}/mlruns.db"
ARTIFACTS_SRC="/app/mlruns"
CONTAINER_APP="/app"

echo "[mlflow-init] Starting MLflow initialisation …"

# ── 1. Validate canonical DB ────────────────────────────────────────────────
if [ ! -f "${CANONICAL_DB}" ]; then
    echo "[mlflow-init] ERROR: Canonical database not found at ${CANONICAL_DB}."
    echo "              Mount the project artifacts directory read-only."
    exit 1
fi

# ── 2. Copy canonical DB to runtime location ────────────────────────────────
mkdir -p "${RUNTIME_DIR}"
cp "${CANONICAL_DB}" "${RUNTIME_DB}"
echo "[mlflow-init] Runtime DB created at ${RUNTIME_DB}."

# ── 3. Detect and rewrite host-absolute paths ───────────────────────────────
python3 - <<'PYEOF'
import sqlite3, sys

RUNTIME_DB = "/mlflow-runtime/mlruns.db"
CONTAINER_APP = "/app"

conn = sqlite3.connect(RUNTIME_DB)
conn.execute("PRAGMA journal_mode=WAL")

# Detect old prefix from the first non-default experiment artifact_location.
old_prefix = None
row = conn.execute(
    "SELECT artifact_location FROM experiments WHERE experiment_id != 0 LIMIT 1"
).fetchone()
if row and row[0]:
    loc = row[0]
    # Location ends with /mlruns/<N> — find the prefix before /mlruns/
    idx = loc.find("/mlruns/")
    if idx > 0:
        old_prefix = loc[:idx]
    elif loc.startswith("/"):
        # Fallback: use the parent two levels above the last component
        old_prefix = "/".join(loc.rstrip("/").split("/")[:-2])

if old_prefix and old_prefix != CONTAINER_APP:
    print(f"[mlflow-init] Rewriting paths: {old_prefix!r} → {CONTAINER_APP!r}")
    rewrite_count = 0
    for tbl, col in [
        ("experiments", "artifact_location"),
        ("runs", "artifact_uri"),
        ("logged_models", "artifact_location"),
    ]:
        try:
            cur = conn.execute(
                f"UPDATE {tbl} SET {col} = REPLACE({col}, ?, ?) "
                f"WHERE {col} LIKE ?",
                (old_prefix, CONTAINER_APP, f"{old_prefix}%"),
            )
            rewrite_count += cur.rowcount
        except Exception as e:
            print(f"[mlflow-init] Warning: could not rewrite {tbl}.{col}: {e}")
    conn.commit()
    print(f"[mlflow-init] {rewrite_count} path(s) rewritten.")
else:
    print("[mlflow-init] No path rewrite needed (paths already portable or no prefix detected).")

conn.close()
PYEOF

echo "[mlflow-init] Initialisation complete."
echo "[mlflow-init] NOTE: This is a runtime copy for local demonstration only."
echo "[mlflow-init]       The canonical database was NOT modified."

# ── 4. Start MLflow server on runtime copy ──────────────────────────────────
exec mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri "sqlite:////mlflow-runtime/mlruns.db" \
    --default-artifact-root "${ARTIFACTS_SRC}" \
    --serve-artifacts
