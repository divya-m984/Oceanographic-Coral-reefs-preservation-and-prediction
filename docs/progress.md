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

## Current Limitations

- MLflow file store (`mlruns/`) is in maintenance mode as of MLflow 3.14.0; all tracking uses the SQLite backend.
- Artifact paths in `artifacts/mlruns.db` are absolute; if the project moves again, run the path-migration script documented in `MEMORY.md`.
- Quality gate thresholds are conservative; small-dataset test fixtures may not naturally pass them.
- `register_candidate` always re-runs because its output (`candidate_registration.json`, `cache: false`) cannot be restored from DVC cache. This is acceptable since registration is a side-effect on the MLflow DB, not a pure function of local files.
- `mlflow.db` at project root is unused — not deleted to avoid unintended changes.
- No FastAPI serving, Streamlit UI, drift monitoring, or Docker deployment yet.

---

## Planned Next Steps (M8+)

- M8: GitHub Actions CI/CD pipeline (`.github/workflows/` already scaffolded)
- M8: FastAPI inference endpoint (`src/api/`)
- M8: Evidently AI drift monitoring (`src/monitoring/`)
- M8: Docker containerisation (`docker-compose.yml` already scaffolded)
