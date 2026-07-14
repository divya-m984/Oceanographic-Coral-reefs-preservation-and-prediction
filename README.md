# CoralSense MLOps

**An MLOps-Driven Multi-Sensor Platform for Coral Reef Health Prediction and Restoration Planning**

> **DISCLAIMER — SYNTHETIC DATA ONLY**
> All observations used in this project are computer-generated using a documented
> synthetic data generator. Predictions produced by CoralSense models do **not**
> represent real conservation advice and must not be used to guide actual marine
> management decisions. All scientific assumptions are documented in
> `src/data/generate_data.py` and `params.yaml`.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [CI Status](#ci-status)
4. [Requirements](#requirements)
5. [Setup](#setup)
6. [Configuration](#configuration)
7. [Development Commands](#development-commands)
8. [API Service](#api-service)
9. [Milestone Commands](#milestone-commands)
10. [Running Tests](#running-tests)
11. [Docker](#docker)
12. [Project Structure](#project-structure)
13. [Scientific Assumptions](#scientific-assumptions)

---

## Project Overview

CoralSense processes geotagged marine sensor observations and provides two
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

```
Synthetic Data Generator
        │
        ▼
  DVC Pipeline (dvc.yaml)
  ┌─────────────────────────────────────────┐
  │  generate → validate → preprocess →     │
  │  build_features → train → evaluate      │
  └─────────────────────────────────────────┘
        │
        ▼
  MLflow Tracking + Model Registry
        │
        ├──► FastAPI  :8000  (POST /predict, GET /health)
        │
        └──► Streamlit :8501  (map, stats, predict, drift)
                │
                └──► Evidently Drift Reports
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

# M10 — Start the Streamlit dashboard
streamlit run src/dashboard/app.py --server.port 8501

# M11 — Generate Evidently drift report
python -m src.monitoring.drift

# M12 — Start all services via Docker Compose
docker compose up --build
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

---

## Project Structure

```
coralsense-mlops/
├── README.md
├── requirements.txt          # Full dependency list
├── requirements-dev.txt      # M1 minimal deps (pytest, ruff, pyyaml)
├── pyproject.toml            # Build config, pytest, ruff settings
├── params.yaml               # All tunable knobs (single source of truth)
├── .env.example              # Environment variable template
├── dvc.yaml                  # Pipeline DAG (M7)
├── docker-compose.yml        # Container orchestration (M11)
├── data/
│   ├── raw/                  # Generated observations (DVC-tracked)
│   ├── processed/            # Train/test splits
│   ├── reference/            # Evidently reference baseline
│   └── production/           # Shifted data for drift demo
├── models/                   # Serialised model artifacts
├── notebooks/                # Exploratory analysis
├── reports/                  # Evidently HTML reports, plots
├── artifacts/                # MLflow tracking store
├── scripts/                  # CLI helper scripts
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
│   │   └── registry.py       # MLflow model registry (M6)
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
    └── test_api.py           # M9
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
