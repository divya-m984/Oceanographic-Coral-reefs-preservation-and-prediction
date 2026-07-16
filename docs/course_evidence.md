# Oceanographic MLOps — Course Evidence

> **Project title (exact):**
> Oceanographic: A Machine Learning-Driven Sonar Framework for
> Real-Time Coral Reef Habitat Prediction and Marine Ecosystem Monitoring

> **Important:** This document provides concise, factual technical evidence
> to complement handwritten notebook entries. It does not replace the notebook.
> All items are marked as COMPLETED, DEMONSTRATED, or PLANNED.

> **Synthetic-data statement:** The current implementation uses a
> computer-generated synthetic dataset because real field-deployed sonar
> hardware and labelled marine survey data are not yet available. All
> reported metrics reflect synthetic-data performance only and must not be
> interpreted as evidence of real-world coral reef prediction accuracy.

---

## 1. Project Title

**Oceanographic: A Machine Learning-Driven Sonar Framework for
Real-Time Coral Reef Habitat Prediction and Marine Ecosystem Monitoring**

The project proposes an acoustic sonar-based platform for structural reef
assessment, combining sonar-derived structural information with environmental
and biological sensor readings to classify reef health and predict restoration
suitability. The current college implementation uses synthetic observations
that statistically represent the kinds of readings such a system would produce.

---

## 2. Topic Selection

**Motivation:** Coral reefs support approximately 25% of marine biodiversity
and protect coastlines from storm surge, yet current monitoring relies heavily
on manual diver surveys that are expensive, slow, and spatially limited.

**Research gap:** Most ML-for-coral studies use satellite or photogrammetry
inputs. Few have explored integrating underwater acoustic (sonar) structural
data with environmental sensors in a production-grade MLOps pipeline with
automated drift monitoring and controlled retraining governance.

**Sonar versus environmental distinction:**
- **Sonar captures:** seabed depth, acoustic backscatter, rugosity, substrate
  hardness, acoustic complexity — structural information about the physical
  reef habitat.
- **Sonar does NOT directly measure:** bleaching percentage, disease
  percentage, coral cover, water temperature, pH, or turbidity. Those
  require independent sensor types or observer surveys.

---

## 3. Existing Approaches and Research Gap

| Approach | Limitation addressed by this project |
|---|---|
| Satellite-based bleaching mapping (CoralWatch, NOAA) | Low spatial resolution, restricted to shallow reefs, cloud occlusion |
| Diver visual surveys | Labour-intensive, non-reproducible, small spatial coverage |
| Single-sensor ML (SST or RGB imagery only) | Misses structural habitat information |
| Research-only ML pipelines | No production inference service, drift monitoring, or governed retraining |

**This project's contribution (academic, synthetic-data level):**
- Combines sonar structural features with multi-sensor environmental inputs.
- Implements a full MLOps lifecycle: data → validation → training → registry
  → API → dashboard → monitoring → governed retraining → Docker deployment.
- Demonstrates drift-triggered, approval-gated retraining governance.

---

## 4. Activity 2 — MLOps Lifecycle

The following MLOps lifecycle stages are all COMPLETED and demonstrated:

| Stage | Implementation | Status |
|---|---|---|
| Data generation | `src/data/generate_data.py` | COMPLETED |
| Data validation | `src/data/validate.py` (Pandera) | COMPLETED |
| Feature engineering | `src/features/build_features.py` | COMPLETED |
| Model training | `src/models/train.py` (LR, RF, XGBoost) | COMPLETED |
| Experiment tracking | MLflow, `artifacts/mlruns.db` | COMPLETED |
| Model registration | `src/models/registry.py` | COMPLETED |
| Pipeline automation | DVC 7-stage DAG (`dvc.yaml`) | COMPLETED |
| Continuous integration | GitHub Actions 5-job workflow | COMPLETED |
| Inference serving | FastAPI `src/api/main.py` | COMPLETED |
| Monitoring dashboard | Streamlit 10-page dashboard | COMPLETED |
| Drift monitoring | Evidently `src/monitoring/drift.py` | COMPLETED |
| Controlled retraining | `src/models/retrain.py` + compare + promote | COMPLETED |
| Containerisation | Docker Compose 4-service stack | COMPLETED |

---

## 5. Activity 3 — Usage of ML and MLOps

### Machine learning component

**Task 1 — Reef habitat health:**
- 4-class classification: `healthy`, `stressed`, `bleached`, `severely_degraded`
- Champion model: Logistic Regression, CV macro-F1 = 0.7612 (SYNTHETIC)
- Test macro-F1 = 0.7871, test balanced accuracy = 0.8012 (SYNTHETIC)

**Task 2 — Restoration suitability:**
- 3-class classification: `suitable`, `moderately_suitable`, `unsuitable`
- Champion model: XGBoost, CV macro-F1 = 0.7913 (SYNTHETIC)
- Test macro-F1 = 0.8029, test balanced accuracy = 0.8121 (SYNTHETIC)

**Algorithm comparison:**
All three algorithms (Logistic Regression, Random Forest, XGBoost) were
trained and evaluated. Selection was by 5-fold cross-validation macro-F1 on
the training set. Test metrics were computed on a held-out 20% split.

### MLOps component

- DVC tracks data and model lineage across all pipeline stages.
- MLflow tracks every training run with parameters, metrics, and artifacts.
- The Model Registry holds 4 versions of each model; champion alias = v1.
- Evidently monitors feature, prediction, and confidence drift in production.
- Champion promotion requires explicit human approval; no automatic promotion.

---

## 6. Activity 4 — DevOps versus MLOps

| Dimension | Standard DevOps | Oceanographic MLOps (this project) |
|---|---|---|
| Versioned artifact | Code commit / Docker image | Code + data + model + preprocessor |
| CI pipeline test | Unit + integration tests | Unit + integration + ML smoke test + DVC validation |
| Deployment | Container image push | Model bundle export + container build |
| Production monitoring | Error rates, latency | Feature drift, prediction distribution drift, confidence drift |
| Rollback trigger | Failed health check / error spike | Drift recommendation + champion comparison regression |
| Rollback mechanism | Redeploy previous image | `python -m src.models.rollback` → champion alias moves to previous version |
| Data validation | Input schema check | Pandera schema + Evidently statistical tests |
| Model governance | N/A | Challenger comparison, quality gates, explicit approval, audit receipts |

**Key MLOps-specific additions:**
- Retraining requires labelled data with provenance hash — unlabelled drift
  data is explicitly rejected.
- The champion alias can only move via `src/models/promote.py` with
  `--approve --approver --reason` flags.
- Every promotion and rollback produces a timestamped JSON receipt.

---

## 7. Activity 5 — Work Completed

### Completed (all 14 milestones)

All 14 project milestones are implemented and tested:

- **M1** — Project foundation: Python 3.14 venv, pyproject.toml, config, logging
- **M2** — Synthetic dataset: 15,000 rows, 21 columns, 4 reef regions (Indian Ocean)
- **M3** — Data validation: Pandera schema with range checks and regex
- **M4** — Preprocessing: stratified split, ColumnTransformer, 6 derived features
- **M5** — Training: 3 algorithms × 2 tasks, 5-fold CV, MLflow logging
- **M6** — Registry: champion/challenger aliases, quality gates
- **M7** — DVC pipeline: 7-stage reproducible DAG, `params.yaml` single source of truth
- **M8** — CI/CD: 5-job GitHub Actions workflow, local equivalent `ci_check.sh`
- **M9** — FastAPI: 7 endpoints, Pydantic v2 validation, no MLflow calls at runtime
- **M10** — Streamlit: 10 pages, geographic reef map, model performance visualisation
- **M11** — Evidently drift: feature drift, prediction drift, confidence drift, RETRAIN recommendation
- **M12** — Docker: 4-service Compose stack, portable model bundles, checksum verification
- **M13** — Controlled retraining: 8-check data contract, challenger comparison, explicit approval gate
- **M14** — Release readiness: Makefile, preflight, demo orchestrator, evidence manifest, docs

### Test suite

- 910 tests, all passing (last verified on commit 8268e13)
- Test files: test_config, test_generate_data, test_validate, test_preprocess,
  test_models, test_registry, test_predict, test_dvc_pipeline, test_ci_workflow,
  test_api, test_dashboard, test_monitoring, test_docker, test_bundle,
  test_retraining, test_demo

### Current output

- Champion reef-health model: Logistic Regression v1, test macro-F1 = 0.7871 (SYNTHETIC)
- Champion restoration model: XGBoost v1, test macro-F1 = 0.8029 (SYNTHETIC)
- FastAPI service serving both champions via `/predict/both`
- Streamlit dashboard with 10 interactive pages
- Drift monitoring: 4 features drift with standard shift, RETRAIN recommended
- Docker stack: all 3 services healthy (mlflow, api, dashboard)

---

## 8. Current Output

### Model performance (SYNTHETIC DATA ONLY)

**Reef health — Logistic Regression champion:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| healthy | 0.66 | 0.74 | 0.70 | 318 |
| stressed | 0.94 | 0.86 | 0.90 | 1726 |
| bleached | 0.89 | 0.88 | 0.88 | 283 |
| severely_degraded | 0.63 | 0.72 | 0.67 | 673 |
| **macro avg** | **0.78** | **0.80** | **0.79** | 3000 |

**Restoration suitability — XGBoost champion:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| suitable | 0.71 | 0.73 | 0.72 | 907 |
| moderately_suitable | 0.90 | 0.87 | 0.89 | 1798 |
| unsuitable | 0.77 | 0.83 | 0.80 | 295 |
| **macro avg** | **0.79** | **0.81** | **0.80** | 3000 |

### Drift demonstration output

Running `python -m src.monitoring.run_drift` with standard shift (scale=1.0):
- Drifted features: 4 (`water_temperature_c`, `bleaching_percentage`,
  `coral_cover_percentage`, `turbidity_ntu`)
- Prediction distribution drift: detected for both tasks
- Confidence drift: detected
- Recommendation: **RETRAIN**

### Example Gulf of Mannar prediction (via FastAPI)

```json
{
  "health": {
    "predicted_class": "healthy",
    "confidence": 0.985,
    "model_version": "1",
    "model_alias": "champion"
  },
  "restoration": {
    "predicted_class": "suitable",
    "confidence": 0.992,
    "model_version": "1",
    "model_alias": "champion"
  }
}
```

---

## 9. Future Work

The following items are architectural objectives for a real-world Phase 2:

1. **Real sensor integration:** Replace `src/data/generate_data.py` with an
   ingestion module connected to deployed sonar hardware and multi-sensor
   buoys.
2. **Labelled field data:** Commission ecological surveys to produce genuinely
   labelled reef health and restoration suitability ground truth.
3. **DVC remote:** Configure S3 or GCS remote storage for large survey files.
4. **MLflow on PostgreSQL:** Move from SQLite to a persistent PostgreSQL backend
   for multi-user tracking.
5. **Cloud deployment:** Deploy FastAPI to a cloud VM or Kubernetes cluster
   near the Indian Ocean reef survey region.
6. **Expert validation:** Obtain domain expert ecological sign-off before using
   predictions to guide any conservation management decision.
7. **Spatial model:** Incorporate GPS coordinates as features or use a
   spatially-aware model that can generalise across reef transects.
8. **Active learning:** Implement a query strategy to select the most
   informative unlabelled observations for efficient expert labelling.

---

## 10. Activity 6 — Dataset Description

| Property | Value |
|---|---|
| Dataset type | **Synthetic** (computer-generated) |
| Generator | `src/data/generate_data.py` |
| Rows | 15,000 |
| Columns | 21 (16 sensor + region + lat + lon + 2 targets) |
| Reef regions | Lakshadweep, Gulf of Mannar, Gulf of Kutch, Andaman and Nicobar Islands |
| Target 1 | `reef_health` — 4 classes |
| Target 2 | `restoration_suitability` — 3 classes |
| Missing values | Zero (by construction) |
| Random seed | 42 (fully reproducible) |
| SHA-256 prefix | a03cb3e9 |
| Storage | `data/raw/observations.csv` |
| Tracking | DVC (`data/raw/observations.csv.dvc`) |

**Generator design:** The synthetic rules are documented in
`src/data/generate_data.py`. Key rules:
- Water temperature > 30°C pushes toward bleached / severely_degraded.
- pH < 8.0 or turbidity > 10 NTU pushes toward degraded labels.
- Coral cover > 40% and hard substrate > 50% favour `suitable` restoration.
- Bleaching > 40% or disease > 15% indicates `severely_degraded`.
- Gaussian noise (`noise_scale = 0.15`) prevents artificially perfect accuracy.

**Limitation:** Because the dataset is synthetic, the statistical rules
reflect a developer's modelling assumptions, not real coral reef ecology.
Published accuracy metrics are valid only for this synthetic benchmark.

---

## 11. Activity 7 — CI/CD

### CI pipeline (`.github/workflows/ci.yml`)

| Job | Trigger | What it does |
|---|---|---|
| `code-quality` | every push to master/main or PR | `ruff check` + `ruff format --check` on src/, tests/, scripts/ |
| `tests` | every push to master/main or PR | Full test suite; disposable MLflow DB (`sqlite:///ci_mlruns.db`) |
| `pipeline-validation` | every push to master/main or PR | DVC YAML structure check + isolated 400-row data round-trip |
| `ml-smoke-test` | every push to master/main or PR | Train both tasks on 500 rows; verify predictions and metrics |
| `build` | every push to master/main or PR | `python -m build` → wheel + sdist; uploaded as CI artifact |

**CI safety guarantees:**
- The canonical MLflow DB (`artifacts/mlruns.db`) is never opened in CI.
- No model version is registered in CI.
- Champion promotion is never triggered in CI.
- The full 10-minute DVC pipeline (`dvc repro`) is not run on every push.

### Local CI equivalent

```bash
bash scripts/ci_check.sh
```

---

## 12. Level 0 — Basic Automation

COMPLETED. Demonstrated:
- Single command to run the full test suite (`pytest` / `make test`).
- Reproducible pipeline with DVC (`dvc repro`).
- Automated lint and format checking (`ruff` via `make lint`).

---

## 13. Level 1 — Continuous Integration

COMPLETED. Demonstrated:
- GitHub Actions workflow runs on every push and pull request.
- 5 parallel/sequential jobs covering code quality, tests, pipeline
  validation, ML smoke testing, and packaging.
- No model is registered or promoted in CI — the registry is read-only.
- CI uses an isolated disposable MLflow database.

---

## 14. Level 2 — Automated Deployment

COMPLETED. Demonstrated:
- `docker compose up --build -d` starts the full three-service stack.
- FastAPI serves both champion models via 7 REST endpoints.
- Streamlit dashboard connects to the API automatically.
- MLflow UI exposes the experiment history and registry.
- The API uses bundle mode: models are loaded from `deploy/bundles/` without
  any network call to the MLflow service.
- Evidently drift reports are generated by the optional `drift` profile.

---

## 15. Level 3 — Full MLOps Maturity

COMPLETED. Demonstrated:
- Evidently drift monitoring detects distribution shift in production data.
- Drift summary recommends RETRAIN when shift is detected.
- `scripts/run_retraining.py` validates labelled input against an 8-check
  contract (rejects unlabelled drift data by design).
- `src/models/compare.py` compares challenger against champion on 4 gates.
- `src/models/promote.py` requires `--approve --approver --reason` and
  refuses to promote without a comparison report.
- Every promotion and rollback writes a timestamped JSON receipt.
- `src/models/rollback.py` moves the champion alias to a previous version
  without deleting any model version.
- `src/models/model_card.py` generates a Markdown model card for any version.
- The canonical MLflow database is never modified by any automated process.
