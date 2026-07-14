# CoralSense MLOps — Project Progress

> **Synthetic-data disclaimer**: All metrics and predictions in this document reflect
> performance on a synthetic dataset generated for college project purposes only.
> They must NOT be interpreted as evidence of real-world coral reef conservation accuracy.

---

## Completed Milestones

### M1 — Project Foundation
- Python 3.14 virtual environment (`.venv/`)
- `pyproject.toml` with all dependencies
- `src/config.py` — typed `Config` dataclass loading `params.yaml`; absolute path resolution via `_PROJECT_ROOT`
- `src/__init__.py` package structure
- `pytest` configured in `pyproject.toml` (`tests/`)
- `requirements.txt`, `requirements-dev.txt`
- `.github/workflows/` scaffold (CI/CD placeholder)
- `dvc.yaml` pipeline descriptor
- `docker-compose.yml` service descriptor

### M2 — Synthetic Dataset
- `src/data/generate_data.py` — reproducible synthetic generator
- 15,000 observations, 21 columns, zero missing values
- Four Indian reef regions: Lakshadweep, Gulf of Mannar, Gulf of Kutch, Andaman and Nicobar Islands
- Two target labels: `reef_health` (4 classes), `restoration_suitability` (3 classes)
- Saved to `data/raw/observations.csv`

```bash
python -m src.data.generate_data
```

### M3 — Dataset Validation
- `src/data/validate_data.py` — Pandera schema with range checks, regex, allowed values
- `tests/test_data.py` — validation and CLI tests

```bash
python -m src.data.validate_data --input data/raw/observations.csv
pytest tests/test_data.py
```

### M4 — Preprocessing and Feature Engineering
- `src/data/preprocess.py` — stratified 80/20 split, `ColumnTransformer` (StandardScaler + OneHotEncoder), 6 derived features
- `src/data/build_features.py` — derived feature computation (thermal stress index, oxygen stress index, acidity deviation, water quality index, substrate stability score, structural complexity score)
- Artifacts saved to `data/processed/`:
  - `X_train_health.csv`, `X_test_health.csv`, `y_train_health.csv`, `y_test_health.csv`
  - `X_train_restoration.csv`, `X_test_restoration.csv`, `y_train_restoration.csv`, `y_test_restoration.csv`
  - `preprocessor_health.joblib`, `preprocessor_restoration.joblib`
- Split sizes: 12,000 train / 3,000 test (per task)

```bash
python -m src.data.preprocess --input data/raw/observations.csv
pytest tests/test_preprocessing.py
```

### M5 — Model Training and MLflow Experiment Tracking
- `src/models/train.py` — Logistic Regression, Random Forest, XGBoost with 5-fold CV
- MLflow experiment tracking at `sqlite:///artifacts/mlruns.db`
- Experiments: `coral_reef_health`, `coral_restoration_suitability`

**Reef-health results** (15,000 observations, 5-fold CV):

| Algorithm           | CV macro-F1 | CV bal-accuracy | Test macro-F1 | Test accuracy |
|---------------------|-------------|-----------------|---------------|---------------|
| Logistic Regression | 0.7612      | 0.7776          | **0.7871**    | 0.7867        |
| Random Forest       | 0.7604      | —               | —             | —             |
| XGBoost             | 0.7580      | —               | —             | —             |

Best model: **Logistic Regression** (selected by CV macro-F1)

**Restoration suitability results** (15,000 observations, 5-fold CV):

| Algorithm           | CV macro-F1 | CV bal-accuracy | Test macro-F1 | Test accuracy |
|---------------------|-------------|-----------------|---------------|---------------|
| XGBoost             | 0.7913      | 0.8016          | **0.8029**    | 0.8027        |
| Random Forest       | 0.7839      | —               | —             | —             |
| Logistic Regression | 0.7808      | —               | —             | —             |

Best model: **XGBoost** (selected by CV macro-F1)

Candidate models saved to `models/`:
- `models/best_model_health.joblib`
- `models/best_model_restoration.joblib`
- `models/evaluation_health.json`
- `models/evaluation_restoration.json`

```bash
python -m src.models.train --task all
pytest tests/test_models.py
```

### M6 — Model Registry and Production Prediction

#### Files created / modified

| File | Status |
|------|--------|
| `src/models/registry.py` | New (untracked → M6) |
| `src/models/predict.py` | New (untracked → M6) |
| `tests/test_registry.py` | New (untracked → M6) |
| `tests/test_predict.py` | New (untracked → M6) |
| `params.yaml` | Modified (added `quality_gates`, `champion_alias`, corrected model names) |
| `src/config.py` | Modified (added `mlflow_champion_alias`, `quality_gates`; absolute MLflow tracking URI) |
| `docs/progress.md` | New — this file |

#### Canonical MLflow tracking location
```
sqlite:///artifacts/mlruns.db    (absolute: <project_root>/artifacts/mlruns.db)
```
This SQLite database holds all M5 experiments and M6 registry entries.
The `mlruns/` directory contains the actual artifact files referenced by the database.

> Note: `mlflow.db` in the project root is an empty default database — it is not used.

#### Quality gates (configured in `params.yaml`)

| Task        | min CV macro-F1 | min CV balanced accuracy |
|-------------|-----------------|--------------------------|
| health      | 0.70            | 0.70                     |
| restoration | 0.73            | 0.73                     |

Promotion uses CV metrics only — never final test-set results.

#### Registered models and champion aliases

```bash
python -m src.models.registry --task all --register --promote
```

| Registered model name                 | Algorithm           | Version | CV macro-F1 | Gate | Champion alias |
|---------------------------------------|---------------------|---------|-------------|------|----------------|
| `coralsense_reef_health`              | Logistic Regression | 1       | 0.7612      | PASS | champion       |
| `coralsense_restoration_suitability`  | XGBoost             | 1       | 0.7913      | PASS | champion       |

#### API fixes required for MLflow 3.14.0
1. `mv.version` is now `int` → normalised to `str` via `str(mv.version)` in two places in `registry.py`.
2. `mv.tags` is now a plain `dict` → replaced `{t.key: t.value for t in mv.tags}` with dict-safe extraction.
3. `predict.py::_prepare_features` now uses `preprocessor.feature_names_in_` (pre-transform column names) rather than `payload["feature_names"]` (post-transform `num__*` / `cat__*` names).
4. `predict.py::predict_batch` now uses `estimator.classes_` / `label_encoder.classes_` to map probability columns instead of `payload["label_names"]` (order mismatch).

#### Path-migration fix
After moving the project from `~/Documents/Projects/` to `~/Projects/`, the SQLite database stored stale absolute paths. Fixed by updating three tables in `artifacts/mlruns.db`:
- `experiments.artifact_location` (3 rows)
- `runs.artifact_uri` (6 rows)
- `logged_models.artifact_location` (6 rows)

`src/config.py` now derives the MLflow tracking URI from `_PROJECT_ROOT` to prevent recurrence.

#### Test suite

```bash
pytest tests/ -q
# 363 passed in 282s
```

Individual M6 test counts:
- `tests/test_registry.py`: 39 tests
- `tests/test_predict.py`: 49 tests

All tests use temporary directories and temporary SQLite databases. No test touches the real registry, models, or dataset.

#### Example predictions

```bash
python -m src.models.predict --task health --input <input.json>
python -m src.models.predict --task restoration --input <input.json>
```

Sample sensor record (Gulf of Mannar, moderate conditions):

**Health prediction** — model: `coralsense_reef_health` v1 (champion = Logistic Regression)
```
predicted_class : healthy
confidence      : 0.9595
probabilities:
  healthy           : 0.9595
  stressed          : 0.0405
  bleached          : 0.0000
  severely_degraded : 0.0000
```

**Restoration prediction** — model: `coralsense_restoration_suitability` v1 (champion = XGBoost)
```
predicted_class : suitable
confidence      : 0.7594
probabilities:
  suitable             : 0.7594
  moderately_suitable  : 0.2387
  unsuitable           : 0.0019
```

#### Reproduction commands (from project root)

```bash
# Activate venv
source .venv/bin/activate

# Validate dataset
python -m src.data.validate_data --input data/raw/observations.csv

# Preprocess (if data/processed/ is missing)
python -m src.data.preprocess --input data/raw/observations.csv

# Train (if models/ are missing)
python -m src.models.train --task all

# Register and promote champions
python -m src.models.registry --task all --register --promote

# Run health prediction
python -m src.models.predict --task health --input <your_input.json>

# Run restoration prediction
python -m src.models.predict --task restoration --input <your_input.json>

# Full test suite
pytest tests/ -q
```

### M7 — Reproducible DVC Pipeline

#### DVC version
3.67.1

#### Files created / modified

| File | Status |
|------|--------|
| `dvc.yaml` | Rewritten — 6 functional stages replacing M1 scaffold |
| `dvc.lock` | Generated by `dvc repro` |
| `.dvcignore` | Updated — MLflow DB, mlruns/, mlartifacts/, .venv/ excluded |
| `.gitignore` | Updated — `data/processed/` excluded from git (DVC-tracked) |
| `src/models/run_evaluate.py` | New — evaluate stage: reads evaluation JSONs, writes `reports/metrics.json` |
| `src/models/run_register_candidate.py` | New — register_candidate stage: registers candidate, never promotes champion |
| `tests/test_dvc_pipeline.py` | New — 26 tests for new scripts and dvc.yaml structural checks |

#### Pipeline DAG

```
generate → validate → preprocess → train → evaluate
                                        ↓
                                  register_candidate
```

(register_candidate depends on both train outputs and evaluate output)

#### Stage summary

| Stage | Command | Key params | Outputs |
|-------|---------|------------|---------|
| `generate` | `run_data.generate_data` | `base.random_seed`, `data.n_samples`, `data.regions`, `data.noise_scale` | `data/raw/observations.csv` |
| `validate` | `src.data.validate` | — | `data/raw/observations_validated.csv` |
| `preprocess` | `src.data.preprocess` | `base.random_seed`, `split.*`, `features.*` | `data/processed/` |
| `train` | `src.models.train --task all` | `base.random_seed`, `models.cv_folds`, `models.health`, `models.restoration` | `models/best_model_*.joblib`, `models/evaluation_*.json` |
| `evaluate` | `src.models.run_evaluate` | — | `reports/metrics.json` (DVC metric) |
| `register_candidate` | `src.models.run_register_candidate` | — | `reports/candidate_registration.json` |

#### Commands

```bash
# Run the full pipeline
dvc repro

# Reproduce a single stage
dvc repro train

# Check which stages are stale
dvc status

# Display the pipeline graph
dvc dag

# Show tracked metrics
dvc metrics show

# Show param changes since last run
dvc params diff
```

#### Pipeline reproduction results

**Full run (first run):**
- All 6 stages executed sequentially.
- generate: 15,000 rows, seed=42.
- validate: PASSED, 15,000 rows, 0 missing values.
- preprocess: 12,000 train / 3,000 test, 22 feature columns.
- train: health best=logistic_regression CV macro-F1=0.7612; restoration best=xgboost CV macro-F1=0.7913.
- evaluate: wrote `reports/metrics.json`.
- register_candidate: registered v2 for both models, champion v1 unchanged.

**No-change run (second run):**
```
Stage 'generate' didn't change, skipping
Stage 'validate' didn't change, skipping
Stage 'preprocess' didn't change, skipping
Stage 'train' didn't change, skipping
Stage 'evaluate' didn't change, skipping
Stage 'register_candidate' didn't change, skipping
Data and pipelines are up to date.
```

**Partial rerun (models.health.logistic_regression.C: 1.0 → 1.1):**
```
Stage 'generate' didn't change, skipping
Stage 'validate' didn't change, skipping
Stage 'preprocess' didn't change, skipping   ← data stages unaffected
Running stage 'train': ...                   ← retrains
Running stage 'evaluate': ...                ← updates metrics
Running stage 'register_candidate': ...      ← registers v3
```

**Param restore (C: 1.1 → 1.0):**
- train and evaluate restored from DVC cache (no retraining).
- register_candidate ran again, registered v4 (MLflow DB is mutable; not DVC-cached).

#### DVC metrics

```
reports/metrics.json:
  health:
    best_algorithm:         logistic_regression
    cv_macro_f1:            0.76117
    cv_balanced_accuracy:   0.77765
    test_macro_f1:          0.7871
    test_balanced_accuracy: 0.8012
  restoration:
    best_algorithm:         xgboost
    cv_macro_f1:            0.79129
    cv_balanced_accuracy:   0.80156
    test_macro_f1:          0.8029
    test_balanced_accuracy: 0.8121
```

#### Dataset integrity after testing
- `data/raw/observations.csv`: 15,000 rows confirmed.
- DVC re-generated from seed=42 (identical to original M2 data).

#### Champion model integrity
- `coralsense_reef_health` v1 (Logistic Regression): champion alias unchanged.
- `coralsense_restoration_suitability` v1 (XGBoost): champion alias unchanged.
- Candidates v2, v3, v4 registered during pipeline runs (not promoted).

#### Test suite

```bash
pytest tests/ -q
# 389 passed in 274s  (363 original + 26 new DVC tests)
```

New test file: `tests/test_dvc_pipeline.py` (26 tests):
- `TestRunEvaluate` — 6 tests for `run_evaluate.py`
- `TestRunRegisterCandidate` — 4 tests, includes assertion that `promote=False` always
- `TestDvcYaml` — 8 structural tests (required stages, no absolute paths, etc.)
- `TestParamsYaml` — 7 tests verifying all DVC-referenced param keys exist

#### Design decisions
- `artifacts/mlruns.db` is excluded from DVC outputs. The MLflow database is mutable and would invalidate the pipeline on every run if tracked.
- `data/processed/` is tracked as a whole directory by the preprocess stage.
- Champion promotion is NOT part of the pipeline. `run_register_candidate.py` hard-codes `promote=False`.
- Stage commands use `.venv/bin/python` (relative path from project root) to ensure the correct interpreter is used without requiring venv activation.

---

---

### M8 — GitHub Actions CI/CD Foundation

#### Files created or modified

| File | Action |
|---|---|
| `.github/workflows/ci.yml` | Replaced scaffold with full 5-job workflow |
| `scripts/ci_smoke_test.py` | New — quick ML training smoke test (CI-safe) |
| `scripts/ci_validate_pipeline.py` | New — DVC YAML check + isolated data round-trip |
| `scripts/ci_check.sh` | New — local equivalent of CI |
| `tests/test_ci_workflow.py` | New — 26 structural tests for the workflow YAML |
| `tests/test_dvc_pipeline.py` | Fixed pre-existing ruff lint issues (F401, I001) |
| `tests/test_predict.py` | Fixed pre-existing ruff lint issue (I001) |
| `README.md` | Added CI Status section |
| `docs/progress.md` | Added M8 section (this document) |

#### Workflow triggers

- `push` → `master` or `main`
- `pull_request` → `master` or `main`
- `workflow_dispatch` (manual)

#### Jobs and dependencies

```
code-quality
├── tests ──────────────────────────────┐
├── pipeline-validation                  ├── build
└── ml-smoke-test ───────────────────────┘
```

| Job | Description | Timeout |
|---|---|---|
| code-quality | ruff check + ruff format --check | 10 min |
| tests | Full 415-test suite | 30 min |
| pipeline-validation | dvc dag + DVC structure tests + data round-trip | 10 min |
| ml-smoke-test | Quick ML training, prediction checks | 15 min |
| build | python -m build; uploads dist/ | 10 min |

#### Python version

Python **3.12** in CI. Project requires `>=3.11`; tested locally on 3.14.6.

#### Caching strategy

`actions/setup-python@v5` with `cache: pip` and `cache-dependency-path: requirements.txt`.
Subsequent runs on unchanged dependencies skip re-downloading (~3 GB of packages).

#### CI-safe MLflow strategy

- `MLFLOW_TRACKING_URI=sqlite:///ci_mlruns.db` set in the `tests` job env.
- `scripts/ci_smoke_test.py` creates its own `sqlite:///tmp.../mlruns.db` per run.
- Canonical `artifacts/mlruns.db` is never opened in CI.
- No model is registered or promoted in CI.

#### CI-safe dataset strategy

- No DVC remote is configured; `dvc pull` is not used.
- `scripts/ci_validate_pipeline.py` generates 400 isolated synthetic rows.
- `scripts/ci_smoke_test.py` generates 500 isolated synthetic rows.
- Real `data/raw/` and `data/processed/` are never read or written in CI.

#### Security restrictions

- `permissions: contents: read` (no write access to repository).
- No repository secrets required.
- No hardcoded absolute paths (`/home/BAAHbun` etc.).
- `--promote` flag never used.
- `dvc repro` never run (avoids `register_candidate` side effect).
- Build artifact contains only the Python package (`dist/`), not datasets or models.

#### Local equivalent command

```bash
bash scripts/ci_check.sh
```

#### YAML validation

Validated with `yaml.safe_load` in `tests/test_ci_workflow.py` (26 tests, all pass).

#### Test results

```
415 passed, 97 warnings in 284s
  (389 original tests + 26 new CI workflow tests)
```

New test file: `tests/test_ci_workflow.py` (26 tests):
- `TestWorkflowFile` — 3 tests: file exists, valid YAML, has name
- `TestTriggers` — 4 tests: push/PR/dispatch triggers, branch targets
- `TestPermissions` — 3 tests: permissions block, contents:read, no write
- `TestConcurrency` — 3 tests: cancel-in-progress, group includes workflow+ref
- `TestJobs` — 5 tests: required jobs, timeouts, build dependencies
- `TestSecurity` — 4 tests: no /home/ paths, no --promote, no dvc repro, no canonical DB
- `TestActions` — 4 tests: pinned checkout@v4, setup-python@v5, upload-artifact@v4, pip cache

#### Dataset integrity

`data/raw/observations.csv`: 15,000 rows confirmed (15,001 lines with header).

#### Model registry integrity

- `coralsense_reef_health` v1: champion alias unchanged.
- `coralsense_restoration_suitability` v1: champion alias unchanged.
- Candidate versions 2, 3, 4 from M7 remain; no new versions registered during M8.

---

### M9 — FastAPI Inference Service

#### Files created

| File | Description |
|---|---|
| `src/api/schemas.py` | Pydantic v2 request/response models with strict field validation |
| `src/api/model_loader.py` | `ModelLoader` singleton: graceful load, safe metadata, no fit/register |
| `src/api/main.py` | FastAPI app: 7 endpoints, lifespan startup, Annotated dependency pattern |
| `tests/test_api.py` | 83 isolated tests using `FakeModelLoader` + `FakeInferencePipeline` |

#### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Project index and documentation links |
| `GET` | `/health` | Liveness and model readiness (200 even when degraded) |
| `GET` | `/model-info` | Safe champion metadata — no paths or URIs exposed |
| `POST` | `/predict/reef-health` | Single reef-health prediction |
| `POST` | `/predict/restoration` | Single restoration-suitability prediction |
| `POST` | `/predict/both` | Both predictions for one observation |
| `POST` | `/predict/batch` | Batch predictions (max 50 observations, configurable) |

#### Key design decisions

- **Graceful degradation**: `ModelLoader.load()` catches exceptions silently; failed models return `None`. Endpoints return 503 for unavailable models rather than crashing the app.
- **Dependency injection**: `get_loader` is a FastAPI dependency injected via `app.dependency_overrides` in tests — no real models needed.
- **Test isolation**: `TestClient` used without context manager (`with`) to skip lifespan; `FakeModelLoader` / `FakeInferencePipeline` provide deterministic outputs.
- **Ruff B008 avoidance**: `LoaderDep = Annotated[ModelLoader, Depends(get_loader)]` type alias prevents B008 (no `Depends()` call in default parameter values).
- **No server state exposure**: `model_info()` returns only safe presentation fields; joblib paths, MLflow URIs, and internal config are excluded.
- **Finite-float guard**: `ObservationInput` model validator rejects `NaN` and `Inf` in any numeric field.
- **Extra-field rejection**: `ConfigDict(extra="forbid")` prevents target labels (`reef_health`, `restoration_suitability`) from being submitted as input features.

#### Input validation (ObservationInput)

All 16 inference features are required with domain-appropriate bounds matching `src/data/validate.py`:

| Feature | Type | Range |
|---|---|---|
| `region` | Literal | Lakshadweep, Gulf of Mannar, Gulf of Kutch, Andaman and Nicobar Islands |
| `depth_m` | float | [0, 50] m |
| `water_temperature_c` | float | [10, 42] °C |
| `ph` | float | [7.0, 9.0] |
| `salinity_ppt` | float | [20, 50] ppt |
| `dissolved_oxygen_mg_l` | float | [0, 15] mg/L |
| `turbidity_ntu` | float | [0, 100] NTU |
| `light_intensity` | float | [0, 3000] µmol m⁻² s⁻¹ |
| `current_speed_m_s` | float | [0, 5] m/s |
| `sonar_backscatter` | float | [−60, 0] dB |
| `rugosity_index` | float | [1, 10] |
| `hard_substrate_percentage` | float | [0, 100] % |
| `acoustic_complexity_index` | float | [0, 1] |
| `coral_cover_percentage` | float | [0, 100] % |
| `bleaching_percentage` | float | [0, 100] % |
| `disease_percentage` | float | [0, 100] % |

Optional spatial metadata (`timestamp`, `latitude`, `longitude`) is accepted but not used for inference.

#### Integration test results (Gulf of Mannar observation, real champion models)

```
GET /health → 200
{
  "status": "ok",
  "health_model_ready": true,
  "restoration_model_ready": true
}

POST /predict/both →
  health:      predicted_class="healthy",  confidence=0.9847
  restoration: predicted_class="suitable", confidence=0.9919
```

#### Test suite

```bash
pytest tests/test_api.py -q
# 83 passed in 4.05s

pytest tests/ -q
# 498 passed in 295s  (415 previous + 83 new API tests)
```

New test file: `tests/test_api.py` (83 tests across 10 classes):
- `TestRootEndpoint` — 4 tests: 200, required fields, endpoints list, disclaimer
- `TestHealthEndpoint` — 7 tests: ok/degraded status, per-model flags, timestamp, timestamp format, 503 when loader missing
- `TestModelInfoEndpoint` — 6 tests: both models available/unavailable, disclaimer, no path leakage
- `TestReefHealthEndpoint` — 11 tests: valid predict, 503 when unavailable, 422 on invalid input (OOB values, bad region, missing fields, NaN, Inf, extra fields), correct task field
- `TestRestorationEndpoint` — 8 tests: analogous to reef-health
- `TestBothEndpoint` — 8 tests: both predictions returned, 503 variants, invalid input
- `TestBatchEndpoint` — 12 tests: single/multi row, batch-too-large (422), empty batch (422), model unavailable (503), invalid obs in batch
- `TestProbabilityIntegrity` — 10 tests: proba keys match label sets for health and restoration in single and batch modes
- `TestPredictionResponseFields` — 9 tests: all required PredictionResponse fields present
- `TestRootEndpointWithoutLoader` — 1 test: root endpoint works even when loader is None (no model dependency)
- `TestServiceHealth503` — 7 tests: loader None → 503 (health, model-info, predict endpoints)

#### Dataset integrity

`data/raw/observations.csv`: 15,000 rows, 21 columns confirmed.

#### Registry integrity after M9

- `coralsense_reef_health` v1 (Logistic Regression): champion alias unchanged.
- `coralsense_restoration_suitability` v1 (XGBoost): champion alias unchanged.
- No new model versions registered during M9 (API is read-only).

---

## Current Limitations

- MLflow file store (`mlruns/`) is in maintenance mode as of MLflow 3.14.0; all tracking uses the SQLite backend.
- Artifact paths in `artifacts/mlruns.db` are absolute; if the project moves again, run the path-migration script documented in `MEMORY.md`.
- Quality gate thresholds are conservative; small-dataset test fixtures may not naturally pass them.
- `register_candidate` always re-runs because its output (`candidate_registration.json`, `cache: false`) cannot be restored from DVC cache. This is acceptable since registration is a side-effect on the MLflow DB, not a pure function of local files.
- `mlflow.db` at project root is unused — not deleted to avoid unintended changes.
- No DVC remote configured; CI uses isolated temp data instead of `dvc pull`.
- Python 3.14 is not yet officially supported by GitHub Actions; CI uses Python 3.12.
- Streamlit UI, drift monitoring, and Docker deployment not yet implemented.

---

## Planned Next Steps (M10+)

- M10: Streamlit dashboard (`src/dashboard/app.py`)
- M11: Evidently AI drift monitoring (`src/monitoring/drift.py`)
- M12: Docker containerisation and `docker compose up --build`
