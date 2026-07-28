#!/usr/bin/env bash
# scripts/start_mlflow.sh — Start (or stop/status) the MLflow tracking UI.
#
# Why Docker?
# -----------
# MLflow 3.14.0 uses `from importlib.abc import Traversable` in
# mlflow/assistant/skill_installer.py.  Python 3.14 removed Traversable from
# importlib.abc (it moved to importlib.resources.abc in Python 3.12), causing:
#
#   ImportError: cannot import name 'Traversable' from 'importlib.abc'
#
# The project's main .venv runs Python 3.14.6.  The Dockerfile.mlflow image
# uses Python 3.12-slim, which is unaffected by this regression.
#
# See upstream issue: https://github.com/mlflow/mlflow/issues/24155
#
# Database safety
# ---------------
# artifacts/mlruns.db is mounted read-only.  docker/init_mlflow.sh copies it
# to a writable named volume (mlflow-runtime) and rewrites host-absolute paths
# automatically.  The canonical database is NEVER modified.
#
# Usage
# -----
#   bash scripts/start_mlflow.sh            # start (default)
#   bash scripts/start_mlflow.sh start      # start explicitly
#   bash scripts/start_mlflow.sh stop       # stop the container
#   bash scripts/start_mlflow.sh status     # show container status
#   bash scripts/start_mlflow.sh logs       # tail container logs
#
# The MLflow UI will be available at http://127.0.0.1:5000

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ACTION="${1:-start}"

# Always run docker compose from the project root so relative volume paths resolve.
cd "${PROJECT_ROOT}"

case "${ACTION}" in
  start)
    echo "[start_mlflow] Starting MLflow UI via Docker (Python 3.12)..."
    docker compose up -d mlflow

    echo "[start_mlflow] Waiting for MLflow to become healthy (up to 60s)..."
    MAX_WAIT=60
    WAITED=0
    until curl -sf http://127.0.0.1:5000/health >/dev/null 2>&1; do
      if [ "${WAITED}" -ge "${MAX_WAIT}" ]; then
        echo "[start_mlflow] ERROR: MLflow did not become healthy within ${MAX_WAIT}s."
        echo "[start_mlflow] Last container logs:"
        docker compose logs --tail=20 mlflow
        exit 1
      fi
      sleep 2
      WAITED=$((WAITED + 2))
    done

    echo ""
    echo "[start_mlflow] MLflow UI ready at http://127.0.0.1:5000"
    echo "[start_mlflow] Stop with:   bash scripts/start_mlflow.sh stop"
    echo "[start_mlflow] Or:          docker compose stop mlflow"
    ;;

  stop)
    echo "[start_mlflow] Stopping MLflow container..."
    docker compose stop mlflow
    echo "[start_mlflow] Stopped. Runtime DB volume preserved (docker compose down -v to remove)."
    ;;

  status)
    docker compose ps mlflow
    ;;

  logs)
    docker compose logs -f mlflow
    ;;

  *)
    echo "Usage: $0 [start|stop|status|logs]"
    exit 1
    ;;
esac
