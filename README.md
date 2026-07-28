# Oceanographic: A Machine Learning-Driven Sonar Framework for Real-Time Coral Reef Habitat Prediction and Marine Ecosystem Monitoring

> **DISCLAIMER — SYNTHETIC DATA ONLY**
> All observations used in this project are computer-generated using a documented
> synthetic data generator. Predictions produced by this system do **not**
> represent real conservation advice and must not be used to guide actual marine
> management decisions. All scientific assumptions are documented in
> `src/data/generate_data.py` and `params.yaml`.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [CI Status](#ci-status)
5. [Requirements](#requirements)
6. [Setup](#setup)
7. [Configuration](#configuration)
8. [Development Commands](#development-commands)
9. [API Service](#api-service)
10. [Milestone Commands](#milestone-commands)
11. [Running Tests](#running-tests)
12. [Docker](#docker)
13. [Classroom Demo](#classroom-demo)
14. [M13 — Controlled Retraining and Model Governance](#m13--controlled-retraining-and-model-governance)
15. [MLOps Maturity](#mlops-maturity)
16. [Project Structure](#project-structure)
17. [Scientific Assumptions](#scientific-assumptions)

---

## Quick Start

```bash
# 1. Install dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# 2. Run preflight check
make preflight

# 3. Run all tests
make test

# 4. Export models and start Docker demonstration
make export-models
python scripts/demo.py start
# URLs: MLflow :5000 | FastAPI :8000 | Streamlit :8501

# 5. Verify the demo
python scripts/demo.py verify

# 6. Stop demo
python scripts/demo.py stop
```

---

## Project Overview

The platform processes geotagged marine sensor observations and provides two
classification tasks:

| Task | Labels |
|---|---|
| **Reef Health** | `healthy`, `stressed`, `bleached`, `severely_degraded` |
| **Restoration Suitability** | `suitable`, `moderately_suitable`, `unsuitable` |

The platform covers four Indian reef regions:
- Lakshadweep
- Gulf of Mannar
- Gulf of Kutch
- Andaman and Nicobar Islands

---

## CI Status

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs five jobs on every push and pull request to `master` or `main`, and can also be triggered manually.

### What runs on every push / pull request

| Job | What it does | Timeout |
|---|---|---|
| **code-quality** | `ruff check` + `ruff format --check` on `src/`, `tests/`, `scripts/` | 10 min |
| **tests** | Full 498-test suite with a disposable MLflow database | 30 min |
| **pipeline-validation** | `dvc dag`, DVC YAML structure tests, isolated data round-trip (400 rows) | 10 min |
| **ml-smoke-test** | Quick training of both tasks on 500 rows; verifies predictions and metrics | 15 min |
| **build** | `python -m build` (wheel + sdist); uploads `dist/` artifact | 10 min |

### What is intentionally excluded from CI

- The full 10-minute DVC pipeline (`dvc repro`) — never run on push.
- The `register_candidate` stage — never invoked; no model versions are added.
- Champion promotion (`--promote`) — never run; canonical registry is read-only.
- The canonical MLflow database (`artifacts/mlruns.db`) — never opened; CI uses `sqlite:///ci_mlruns.db`.
- DVC data pull — no remote is configured; pipeline-validation uses isolated temp data.
- Model artifacts (`models/*.joblib`) — not uploaded; build artifact contains only the Python package.
- The `run_drift` DVC stage — never run in CI; requires real champion models and takes ~30s.

### Local equivalent

Run the same checks locally without modifying real project artifacts:

```bash
bash scripts/ci_check.sh
```

This script:
1. Runs `ruff check` and `ruff format --check`
2. Runs the full test suite with `MLFLOW_TRACKING_URI=sqlite:///ci_check_mlruns.db`
3. Runs `scripts/ci_validate_pipeline.py` (isolated temp data)
4. Runs `scripts/ci_smoke_test.py` (isolated temp MLflow DB)

The temporary MLflow database (`ci_check_mlruns.db`) is deleted on exit.

---

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full system diagram.

**Data sources (proposed):**
- **Sonar** captures reef structure: backscatter, rugosity, depth, hard-substrate fraction,
  acoustic complexity. Sonar does NOT directly measure bleaching, coral cover, or water chemistry.
- **Environmental sensors** capture temperature, pH, salinity, dissolved oxygen, turbidity, light, current.
- **Biological surveys** capture coral cover, bleaching %, disease %.
- **Current implementation:** synthetic data only (real sonar hardware not yet deployed).

```
Synthetic Data Generator  ─►  Validation (Pandera)
        │
        ▼
  DVC Pipeline — 7 stages (dvc.yaml)
  generate → validate → preprocess → train → evaluate
            → register_candidate → run_drift
        │
        ▼
  MLflow Tracking + Model Registry (artifacts/mlruns.db)
        │
        ├──► FastAPI  :8000  (7 endpoints, bundle mode)
        ├──► Streamlit :8501  (10 pages, reef map)
        ├──► Evidently Drift Monitoring
        └──► Controlled Retraining Governance (M13)
                │
                └──► Docker Compose (4 services)
```

---

## Requirements

- Python >= 3.11 (tested on 3.14.6 / Arch Linux)
- pip >= 24
- Docker + Docker Compose (optional, for containerised deployment)
- Git

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url> coralsense-mlops
cd coralsense-mlops
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

**M1 only (scaffold + config tests):**

```bash
pip install -r requirements-dev.txt
pip install -e .
```

**Full stack (M2 onwards — install when each milestone requires it):**

```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env if you need non-default ports or a remote MLflow server
```

---

## Configuration

All tunable parameters live in **`params.yaml`**.
Runtime secrets and service URLs live in **`.env`** (gitignored).
`src/config.py` merges both into a single `Config` dataclass.

```python
from src.config import get_config, setup_logging

logger = setup_logging(__name__)
cfg = get_config()
print(cfg.n_samples)           # 12000
print(cfg.paths.raw_data_dir)  # /abs/path/to/data/raw
```

---

## Development Commands

```bash
# Lint and format check
ruff check src/ tests/
ruff format --check src/ tests/

# Auto-fix lint issues
ruff check --fix src/ tests/
ruff format src/ tests/

# Run all tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=term-missing

# Run only unit tests (fast)
pytest -m unit

# Run only integration tests
pytest -m integration

# Run a specific test file
pytest tests/test_config.py -v
```

---

## API Service

The FastAPI inference service (`src/api/main.py`) exposes both champion models over HTTP.

### Launch

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Interactive docs available at `http://127.0.0.1:8000/docs` after startup.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Project index and documentation links |
| `GET` | `/health` | Liveness and model readiness probe |
| `GET` | `/model-info` | Champion model metadata (safe fields only) |
| `POST` | `/predict/reef-health` | Reef-health prediction for one observation |
| `POST` | `/predict/restoration` | Restoration-suitability prediction for one observation |
| `POST` | `/predict/both` | Both predictions for one observation |
| `POST` | `/predict/batch` | Batch predictions (max 50 observations) |

### Example request

```bash
curl -s -X POST http://127.0.0.1:8000/predict/both \
  -H "Content-Type: application/json" \
  -d '{
    "region": "Gulf of Mannar",
    "depth_m": 5.0,
    "water_temperature_c": 27.5,
    "ph": 8.1,
    "salinity_ppt": 35.0,
    "dissolved_oxygen_mg_l": 7.0,
    "turbidity_ntu": 2.0,
    "light_intensity": 800.0,
    "current_speed_m_s": 0.2,
    "sonar_backscatter": -15.0,
    "rugosity_index": 3.5,
    "hard_substrate_percentage": 60.0,
    "acoustic_complexity_index": 0.7,
    "coral_cover_percentage": 45.0,
    "bleaching_percentage": 5.0,
    "disease_percentage": 2.0
  }'
```

### Example response

```json
{
  "health": {
    "predicted_class": "healthy",
    "probabilities": {
      "bleached": 0.000004,
      "healthy": 0.984728,
      "severely_degraded": 0.0,
      "stressed": 0.015268
    },
    "confidence": 0.984728,
    "task": "health",
    "registered_model_name": "coralsense_reef_health",
    "model_version": "1",
    "model_alias": "champion",
    "synthetic_data_disclaimer": "Predictions generated by a model trained on synthetic data only. Do not use to guide real-world conservation decisions."
  },
  "restoration": {
    "predicted_class": "suitable",
    "probabilities": {
      "moderately_suitable": 0.008054,
      "suitable": 0.991880,
      "unsuitable": 0.000065
    },
    "confidence": 0.99188,
    "task": "restoration",
    "registered_model_name": "coralsense_restoration_suitability",
    "model_version": "1",
    "model_alias": "champion",
    "synthetic_data_disclaimer": "Predictions generated by a model trained on synthetic data only. Do not use to guide real-world conservation decisions."
  }
}
```

### Configuration via environment variables

| Variable | Default | Description |
|---|---|---|
| `CORALSENSE_HOST` | `127.0.0.1` | Bind host |
| `CORALSENSE_PORT` | `8000` | Bind port |
| `CORALSENSE_LOG_LEVEL` | `info` | Uvicorn log level |
| `CORALSENSE_MAX_BATCH` | `50` | Maximum batch size |
| `CORALSENSE_CORS_ORIGINS` | _(none)_ | Comma-separated allowed origins |

### Security

- No internal paths, tracebacks, or MLflow URIs are returned to clients.
- No model registration or promotion occurs at runtime.
- Preprocessors are used in transform-only mode (no `fit` calls).
- Returns 503 (not 500) when models fail to load, allowing degraded-mode operation.

---

## Milestone Commands

Each command below corresponds to one project milestone.
Run them in order after the milestone is implemented.

```bash
# M2 — Generate synthetic data
python -m src.data.generate_data

# M3 — Validate schema
python -m src.data.validate

# M4 — Preprocess + feature engineering
python -m src.data.preprocess
python -m src.features.build_features

# M5 — Train models (tracks to MLflow automatically)
python -m src.models.train

# M6 — Evaluate and register best model
python -m src.models.evaluate
python -m src.models.registry

# M7 — Reproduce full DVC pipeline
dvc repro

# M8 — Run CI checks locally
bash scripts/ci_check.sh

# M9 — Start the FastAPI inference server
uvicorn src.api.main:app --host 127.0.0.1 --port 8000

# M10 — Start the Streamlit dashboard (requires FastAPI running first)
# Terminal 1:
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
# Terminal 2:
streamlit run src/dashboard/app.py

# M11 — Run drift monitoring (feature, prediction, confidence drift)
python -m src.monitoring.run_drift                  # standard shift, with HTML reports
python -m src.monitoring.run_drift --no-html        # faster, JSON summary only
python -m src.monitoring.run_drift --shift-scale 0  # zero-shift baseline (no drift expected)
python -m src.monitoring.run_drift --shift-scale 2  # stronger simulated degradation

# M12 — Start all services via Docker Compose
docker compose up --build

# M13 — Controlled retraining (run drift first to get a recommendation)
python -m src.monitoring.run_drift --no-html        # produces reports/drift_summary.json

# Dry-run: validate labelled input and check retraining permission (no DB writes)
python scripts/run_retraining.py \
  --task health \
  --input data/raw/observations.csv \
  --drift-summary reports/drift_summary.json \
  --dry-run

# Full retrain + compare (registers challenger, writes comparison report; never promotes)
python scripts/run_retraining.py \
  --task health \
  --input data/raw/observations.csv \
  --drift-summary reports/drift_summary.json

# Promote challenger (requires explicit --approve, --approver, --reason)
python -m src.models.promote \
  --model coralsense_reef_health \
  --challenger-version <VERSION> \
  --approve \
  --approver "Your Name" \
  --reason "Challenger improves macro-F1 by 3 pp over champion"

# Rollback (--dry-run to preview, then run without it)
python -m src.models.rollback \
  --model coralsense_reef_health \
  --target-version 1 \
  --approver "Your Name" \
  --reason "Challenger underperforms on held-out reef transects" \
  --dry-run

# Generate a Markdown model card for any registered version
python -m src.models.model_card \
  --model coralsense_reef_health \
  --version 1
```

---

## Running Tests

```bash
# All tests
pytest

# With verbose output and coverage
pytest -v --cov=src --cov-report=term-missing --cov-report=html
# HTML report: htmlcov/index.html

# Exclude slow tests
pytest -m "not slow"
```

---

## Docker

**Prerequisites (run once after `dvc repro`):**

```bash
# Export champion models to a portable deployment bundle
python scripts/export_champions.py

# Verify bundle integrity (11 checks)
python scripts/verify_deployment_bundle.py
```

```bash
# Build and start all services
docker compose up --build -d

# Tail logs
docker compose logs -f

# Stop
docker compose down

# Stop and remove volumes
docker compose down -v
```

Services after startup:

| Service | URL |
|---|---|
| MLflow UI | http://localhost:5000 |
| FastAPI docs | http://localhost:8000/docs |
| Streamlit dashboard | http://localhost:8501 |

### MLflow UI — Python 3.14 compatibility note

MLflow 3.14.0 cannot start directly under Python 3.14 because
`mlflow/assistant/skill_installer.py` imports `Traversable` from
`importlib.abc`, which was removed in Python 3.14 (moved to
`importlib.resources.abc`).  See upstream issue
[mlflow#24155](https://github.com/mlflow/mlflow/issues/24155).

**Workaround:** use the pre-built Docker image (`Dockerfile.mlflow`), which
runs Python 3.12 and is unaffected.

```bash
# Start MLflow UI only (Docker, Python 3.12)
make mlflow-ui
# or
bash scripts/start_mlflow.sh

# Stop
make mlflow-ui-stop
# or
bash scripts/start_mlflow.sh stop
```

`artifacts/mlruns.db` is mounted **read-only**; `docker/init_mlflow.sh` copies
it to a writable named volume (`mlflow-runtime`) and rewrites host-absolute
paths automatically.  The canonical database is never modified.

---

## M13 — Controlled Retraining and Model Governance

### Why unlabelled drift data cannot be used for retraining

The M11 Evidently drift pipeline generates a **shifted, unlabelled production window**
(`data/production/`) to simulate data drift. This window contains sensor observations
without ground-truth reef-health or restoration-suitability labels.

Supervised learning requires labelled examples. Using unlabelled drift data for retraining
would be scientifically invalid — there are no labels to learn from. The drift window's
only role is to signal **when** retraining is warranted, not to supply training data.

### Labelled-data contract

`scripts/run_retraining.py` and `src/models/retrain.py` enforce the following rules before
any model is trained:

| Requirement | Detail |
|---|---|
| Both target columns present | `reef_health` (health task) and/or `restoration_suitability` (restoration task) must be non-null |
| No NaN targets | Rows with missing labels are rejected outright |
| Minimum row count | At least 200 labelled rows required (configurable in `params.yaml`) |
| All classes present | All label classes must appear in the input (health: 4 classes, restoration: 3 classes) |
| Minimum class count | Each class must have at least 5 examples |
| Valid feature columns | Input must contain the expected sensor feature columns |
| SHA-256 provenance | Input file hash is recorded in MLflow tags for audit purposes |
| Retraining permission | Drift summary must recommend RETRAIN **or** a manual reason must be supplied |

Attempting to pass the M11 production window (missing target columns) as retraining input
raises a `ValueError` and exits with code 1.

### Dry-run mode

Pass `--dry-run` to validate the input and permission check without writing anything to
the MLflow registry or filesystem:

```bash
python scripts/run_retraining.py \
  --task health \
  --input data/raw/observations.csv \
  --drift-summary reports/drift_summary.json \
  --dry-run
```

Exit codes: `0` = validation passed, `1` = validation error, `3` = permission denied.

### Challenger training

When run without `--dry-run`, the orchestrator:

1. Validates the labelled input (8 checks above).
2. Verifies retraining permission from the drift summary.
3. Fits a fresh preprocessor on the **training split only** (no leakage from holdout).
4. Trains all three algorithm variants (Logistic Regression, Random Forest, XGBoost)
   using the same hyperparameter grid as M5.
5. Selects the best challenger by CV macro-F1.
6. Evaluates the challenger on the holdout split.
7. Registers the challenger in MLflow **without** setting the `champion` alias.
8. Immediately runs champion-challenger comparison and writes a JSON report.

The champion alias is **never moved** by the retraining script.

### Comparison outcomes

`src/models/compare.py` applies four gates (thresholds configurable in `params.yaml`
under `retraining.comparison`):

| Outcome | Meaning |
|---|---|
| `eligible_for_promotion` | Challenger passes all gates; human may promote |
| `review_required` | Challenger meets minimum quality but does not clearly beat the champion |
| `reject` | Challenger regresses on macro-F1, balanced accuracy, or per-class recall |

The comparison outcome is printed to stdout and saved to
`reports/comparison_<model>_<timestamp>.json`. **Promotion is never triggered
automatically**, regardless of outcome.

### Explicit promotion

Promotion requires three mandatory arguments and is blocked unless the comparison
outcome is `eligible_for_promotion` (or `--force` is passed for `review_required` cases):

```bash
python -m src.models.promote \
  --model coralsense_reef_health \
  --challenger-version <VERSION> \
  --approve \
  --approver "Your Name" \
  --reason "Challenger improves macro-F1 by 3 pp and passes all quality gates"
```

Omitting `--approve`, `--approver`, or `--reason` raises an error. A promotion receipt
(`reports/promotion_receipt_<timestamp>.json`) is written on success.

### Rollback

Rolling back moves the `champion` alias to a previous version without deleting any
model version:

```bash
# Preview what would happen
python -m src.models.rollback \
  --model coralsense_reef_health \
  --target-version 1 \
  --approver "Your Name" \
  --reason "Challenger underperforms on held-out reef transects" \
  --dry-run

# Execute rollback
python -m src.models.rollback \
  --model coralsense_reef_health \
  --target-version 1 \
  --approver "Your Name" \
  --reason "Challenger underperforms on held-out reef transects"
```

A rollback receipt (`reports/rollback_receipt_<timestamp>.json`) is written on success.

### Model cards

Generate a Markdown model card for any registered version:

```bash
python -m src.models.model_card \
  --model coralsense_reef_health \
  --version 1
# Saved to reports/model_cards/coralsense_reef_health_v1.md
```

### Approval requirements summary

| Action | Required flags |
|---|---|
| Retrain + compare | `--task`, `--input` (drift summary or `--reason`) |
| Dry-run only | add `--dry-run` |
| Promote challenger | `--approve`, `--approver`, `--reason` |
| Force-promote `review_required` | add `--force` |
| Rollback | `--target-version`, `--approver`, `--reason` |

### Synthetic-data limitation

All metrics reported in comparison reports, promotion receipts, and model cards reflect
performance on the **synthetic** dataset. They do not indicate real-world
coral reef prediction accuracy. Replace `src/data/generate_data.py` with a real sensor
ingestion module and supply genuinely labelled field data before drawing any ecological
conclusions.

---

## Classroom Demo

```bash
make preflight          # pre-demo system check
make export-models      # export champion bundles
make drift              # generate drift report
python scripts/demo.py start    # start full Docker stack
python scripts/demo.py verify   # submit test prediction
python scripts/demo.py stop     # stop cleanly
```

See [`docs/demo_guide.md`](docs/demo_guide.md) for the full 8–12 minute demonstration plan.

---

## MLOps Maturity

| Level | Demonstrated capability |
|---|---|
| **Level 0** | `make test` (910 tests), `dvc repro`, `make lint` |
| **Level 1** | GitHub Actions CI (5 jobs, every push) |
| **Level 2** | `docker compose up` deploys all services; FastAPI + Streamlit |
| **Level 3** | Evidently drift monitoring, RETRAIN recommendation, governed challenger training, explicit approval promotion, rollback with receipt |

---

## Project Structure

```
coralsense-mlops/
├── README.md
├── Makefile                  # Safe build/test/demo targets (M14)
├── CHANGELOG.md              # Milestone history (M14)
├── requirements.txt          # Full dependency list
├── requirements-dev.txt      # M1 minimal deps (pytest, ruff, pyyaml)
├── pyproject.toml            # Build config, pytest, ruff settings
├── params.yaml               # All tunable knobs (single source of truth)
├── .env.example              # Environment variable template
├── dvc.yaml                  # Pipeline DAG (M7 — 7 stages)
├── docker-compose.yml        # Container orchestration (M12)
├── data/
│   ├── raw/                  # Generated observations (DVC-tracked)
│   ├── processed/            # Train/test splits
│   ├── reference/            # Evidently reference baseline
│   └── production/           # Shifted data for drift demo
├── models/                   # Serialised model artifacts
├── notebooks/                # Exploratory analysis
├── reports/                  # Evidently HTML reports, plots
├── artifacts/                # MLflow tracking store
├── scripts/
│   ├── preflight.py          # Read-only system check (M14)
│   ├── demo.py               # Demo orchestrator start/status/verify/stop (M14)
│   ├── collect_evidence.py   # Generates reports/project_manifest.json (M14)
│   ├── ci_check.sh           # Local CI equivalent (M8)
│   ├── ci_smoke_test.py      # ML smoke test (M8)
│   ├── ci_validate_pipeline.py # DVC pipeline validation (M8)
│   ├── export_champions.py   # Export deployment bundles (M12)
│   ├── verify_deployment_bundle.py # Bundle integrity checks (M12)
│   └── run_retraining.py     # Retrain + compare orchestrator (M13)
├── docs/
│   ├── architecture.md       # System architecture (M14)
│   ├── course_evidence.md    # Course activities evidence (M14)
│   ├── demo_guide.md         # 8-12 min classroom demo guide (M14)
│   └── progress.md           # Milestone progress log
├── src/
│   ├── config.py             # Central config (paths, params, logging)
│   ├── data/
│   │   ├── generate_data.py  # Synthetic data generator (M2)
│   │   ├── validate.py       # Pandera schema validation (M3)
│   │   └── preprocess.py     # Preprocessing pipeline (M4)
│   ├── features/
│   │   └── build_features.py # Feature engineering (M4)
│   ├── models/
│   │   ├── train.py          # Model training + MLflow (M5)
│   │   ├── evaluate.py       # Metrics + confusion matrix (M5)
│   │   ├── predict.py        # Inference helpers (M6)
│   │   ├── registry.py       # MLflow model registry (M6)
│   │   ├── retrain.py        # Challenger training + input validation (M13)
│   │   ├── compare.py        # Champion-challenger comparison engine (M13)
│   │   ├── promote.py        # Explicit promotion with approval guard (M13)
│   │   ├── rollback.py       # Champion rollback with dry-run (M13)
│   │   └── model_card.py     # Markdown model card generator (M13)
│   ├── monitoring/
│   │   ├── drift.py          # Evidently drift report (M10)
│   │   └── performance.py    # Production performance tracking (M10)
│   ├── api/
│   │   ├── main.py           # FastAPI app (M9)
│   │   ├── schemas.py        # Pydantic request/response models (M9)
│   │   └── model_loader.py   # Model loading singleton (M9)
│   └── dashboard/
│       ├── app.py            # Streamlit entry point (M10)
│       └── pages/            # Multi-page dashboard pages (M10)
└── tests/
    ├── test_config.py        # M1 — config unit tests
    ├── test_generate_data.py # M2
    ├── test_validate.py      # M3
    ├── test_preprocess.py    # M4
    ├── test_models.py        # M5
    ├── test_api.py           # M9
    └── test_retraining.py    # M13 — 70 tests (challenger, compare, promote, rollback)
```

---

## Scientific Assumptions

All assumptions baked into the synthetic data generator are documented in
`src/data/generate_data.py`. Key rules:

- Higher `water_temperature_c` (>30 °C) correlates with bleaching stress.
- Lower `ph` (<8.0) and higher `turbidity_ntu` push labels toward degraded.
- Higher `coral_cover_percentage` and `hard_substrate_percentage` favour restoration suitability.
- `bleaching_percentage` > 40 % and `disease_percentage` > 15 % indicate severely degraded reef.
- Controlled Gaussian noise (`noise_scale` in params.yaml) prevents unrealistically perfect accuracy.
- A fixed random seed (`random_seed` in params.yaml) ensures full reproducibility.

None of these thresholds replace expert ecological survey data.
Replace `src/data/generate_data.py` with a real sensor ingestion module when
physical data becomes available.
