# Makefile — Oceanographic MLOps
#
# Project: Oceanographic: A Machine Learning-Driven Sonar Framework for
#          Real-Time Coral Reef Habitat Prediction and Marine Ecosystem Monitoring
#
# Safety rules:
#   - Never deletes: data/raw/, artifacts/mlruns.db, models/, .dvc/
#   - No target trains, registers, promotes, or rolls back a model
#   - clean-generated is narrowly scoped (only generated cache/build artifacts)
#
# Long-running targets are marked with [SLOW] in their help text.

PYTHON ?= python
PIP    := $(PYTHON) -m pip

.DEFAULT_GOAL := help

# ── Help ───────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo "Oceanographic MLOps — available targets"
	@echo ""
	@echo "  Setup"
	@echo "    install          Install all dependencies into the selected Python environment"
	@echo ""
	@echo "  Quality"
	@echo "    lint             Run ruff check on src/, tests/, scripts/"
	@echo "    format-check     Run ruff format --check on src/, tests/, scripts/"
	@echo "    test             Run full test suite (910 tests)  [SLOW ~8 min]"
	@echo "    test-fast        Run tests excluding slow marks"
	@echo "    ci-check         Run local equivalent of CI pipeline  [SLOW]"
	@echo ""
	@echo "  Preflight"
	@echo "    preflight        Read-only system and project integrity check"
	@echo "    preflight-json   Preflight with machine-readable JSON output"
	@echo ""
	@echo "  Models"
	@echo "    export-models    Export champion models to deploy/bundles/"
	@echo "    verify-models    Verify deployment bundle integrity (11 checks)"
	@echo ""
	@echo "  Services (local, no Docker)"
	@echo "    mlflow-ui        Start MLflow UI via Docker on :5000 (Python 3.14 workaround)"
	@echo "    mlflow-ui-stop   Stop the MLflow Docker container"
	@echo "    api              Start FastAPI inference service on :8000"
	@echo "    dashboard        Start Streamlit dashboard on :8501"
	@echo "    drift            Generate drift report (reports/drift_summary.json)"
	@echo ""
	@echo "  Docker demonstration"
	@echo "    docker-build     Build all Docker images"
	@echo "    demo             Run full demonstration (preflight + start + verify)"
	@echo "    demo-start       Start Docker services only"
	@echo "    demo-status      Show container health and service status"
	@echo "    demo-verify      Submit test prediction and verify all services"
	@echo "    demo-stop        Stop all Docker services cleanly"
	@echo ""
	@echo "  Evidence"
	@echo "    evidence         Generate reports/project_manifest.json"
	@echo ""
	@echo "  Cleanup"
	@echo "    clean-generated  Remove __pycache__, .pytest_cache, build/, dist/"
	@echo ""
	@echo "  NEVER deleted: data/raw/, artifacts/mlruns.db, models/, .dvc/"

# ── Setup ──────────────────────────────────────────────────────────────────────
.PHONY: install
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo "Installation complete for: $(PYTHON)"

# ── Quality ────────────────────────────────────────────────────────────────────
.PHONY: lint
lint:
	$(PYTHON) -m ruff check src/ tests/ scripts/

.PHONY: format-check
format-check:
	$(PYTHON) -m ruff format --check src/ tests/ scripts/

.PHONY: test
test:
	$(PYTHON) -m pytest tests/ -q

.PHONY: test-fast
test-fast:
	$(PYTHON) -m pytest tests/ -q -m "not slow"

.PHONY: ci-check
ci-check:
	bash scripts/ci_check.sh

# ── Preflight ──────────────────────────────────────────────────────────────────
.PHONY: preflight
preflight:
	$(PYTHON) scripts/preflight.py

.PHONY: preflight-json
preflight-json:
	$(PYTHON) scripts/preflight.py --json

# ── Models ─────────────────────────────────────────────────────────────────────
.PHONY: export-models
export-models:
	$(PYTHON) scripts/export_champions.py

.PHONY: verify-models
verify-models:
	$(PYTHON) scripts/verify_deployment_bundle.py

# ── Services (local) ───────────────────────────────────────────────────────────

# NOTE: The MLflow server cannot run directly under Python 3.14 because
# mlflow/assistant/skill_installer.py uses `from importlib.abc import Traversable`,
# which was removed in Python 3.14 (upstream: github.com/mlflow/mlflow/issues/24155).
# The mlflow-ui target launches the pre-built Docker image (Python 3.12) instead.
.PHONY: mlflow-ui
mlflow-ui:
	@echo "Starting MLflow UI via Docker (Python 3.12) on http://127.0.0.1:5000"
	@bash scripts/start_mlflow.sh start

.PHONY: mlflow-ui-stop
mlflow-ui-stop:
	@bash scripts/start_mlflow.sh stop

.PHONY: api
api:
	@echo "Starting FastAPI on http://127.0.0.1:8000  (Ctrl-C to stop)"
	$(PYTHON) -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000

.PHONY: dashboard
dashboard:
	@echo "Starting Streamlit on http://localhost:8501  (Ctrl-C to stop)"
	$(PYTHON) -m streamlit run src/dashboard/app.py

.PHONY: drift
drift:
	$(PYTHON) -m src.monitoring.run_drift --no-html
	@echo "Drift report written to reports/drift_summary.json"

# ── Docker ─────────────────────────────────────────────────────────────────────
.PHONY: docker-build
docker-build:
	docker compose build

.PHONY: demo
demo:
	$(PYTHON) scripts/demo.py start

.PHONY: demo-start
demo-start:
	$(PYTHON) scripts/demo.py start --skip-preflight

.PHONY: demo-status
demo-status:
	$(PYTHON) scripts/demo.py status

.PHONY: demo-verify
demo-verify:
	$(PYTHON) scripts/demo.py verify

.PHONY: demo-stop
demo-stop:
	$(PYTHON) scripts/demo.py stop

# ── Evidence ───────────────────────────────────────────────────────────────────
.PHONY: evidence
evidence:
	$(PYTHON) scripts/collect_evidence.py
	@echo "Manifest written to reports/project_manifest.json"

# ── Cleanup (narrowly scoped — never touches data/raw, models, .dvc, artifacts) ─
.PHONY: clean-generated
clean-generated:
	find . -type d -name __pycache__ \
	    -not -path "./.venv/*" \
	    -not -path "./.dvc/*" \
	    -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache \
	    -not -path "./.venv/*" \
	    -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	@echo "Cleaned: __pycache__, .pytest_cache, build/, dist/"
	@echo "Preserved: data/raw/, models/, artifacts/, .dvc/"
