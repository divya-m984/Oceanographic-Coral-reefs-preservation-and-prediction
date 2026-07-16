# Oceanographic MLOps — Classroom Demonstration Guide

> **Project:** Oceanographic: A Machine Learning-Driven Sonar Framework for
> Real-Time Coral Reef Habitat Prediction and Marine Ecosystem Monitoring

> **Target duration:** 8–12 minutes
> **Audience:** Course faculty and peers

> **What NOT to claim during the demo:**
> - Do NOT claim sonar directly measures bleaching, coral cover, or disease.
> - Do NOT claim these metrics represent real-world coral reef accuracy.
> - Always state: "These are synthetic data metrics."
> - Do NOT claim the models are ready for deployment to guide conservation decisions.

---

## Prerequisites (run the evening before)

```bash
# 1. Verify environment
python scripts/preflight.py

# 2. Export champion models to deployment bundle
python scripts/export_champions.py
python scripts/verify_deployment_bundle.py

# 3. Generate drift report
python -m src.monitoring.run_drift --no-html

# 4. Build Docker images (slow first time)
docker compose build

# 5. Run full test suite
python -m pytest tests/ -q
```

---

## Demo Flow

### STEP 1 — Problem Statement (1 minute)

**Say:**
> "This project proposes an acoustic sonar platform for coral reef habitat
> assessment. Sonar captures the physical structure of the reef —
> backscatter intensity, rugosity, and hard-substrate percentage.
> These structural signals, combined with environmental sensors measuring
> temperature, pH, and turbidity, feed into two ML classifiers.
> The first classifies reef health into four states.
> The second predicts restoration suitability.
> The current implementation uses synthetic data because we don't yet have
> a deployed sonar rig in the Indian Ocean."

---

### STEP 2 — Repository Structure (30 seconds)

```bash
ls -1
```

**Point out:**
- `params.yaml` — all tunable parameters in one file
- `dvc.yaml` — 7-stage pipeline DAG
- `docker-compose.yml` — 4-service production stack
- `src/` — Python source code
- `tests/` — 910 automated tests
- `docs/` — architecture and course evidence

---

### STEP 3 — Dataset (30 seconds)

```bash
wc -l data/raw/observations.csv
head -1 data/raw/observations.csv
python -m src.data.generate_data --help
```

**Say:**
> "15,000 synthetic observations covering four Indian reef regions:
> Lakshadweep, Gulf of Mannar, Gulf of Kutch, Andaman and Nicobar Islands.
> 21 columns including sonar structural features and environmental sensors."

**Expected output:** `15001 data/raw/observations.csv` (header + 15,000 rows)

---

### STEP 4 — Validation (30 seconds)

```bash
python -m src.data.validate --input data/raw/observations.csv
```

**Say:**
> "Pandera schema validation enforces value ranges, allowed classes,
> and a zero-null contract before any model can be trained."

**Expected output:** `Validation passed — 15000 rows written to observations_validated.csv`

---

### STEP 5 — DVC Pipeline (1 minute)

```bash
dvc dag
dvc status
```

**Say:**
> "DVC tracks the full pipeline: generate → validate → preprocess → train →
> evaluate → register_candidate → run_drift.
> Each stage is re-run only when its inputs change.
> params.yaml is the single source of truth for all hyperparameters."

**Expected output:** ASCII DAG showing 7 stages

---

### STEP 6 — MLflow Experiments (1 minute)

Open browser: **http://localhost:5000** (after Docker starts)

**Navigate to:**
- Experiments → `coral_reef_health`
- Show run list with logged metrics

**Say:**
> "Every training run is tracked automatically. We can compare Logistic
> Regression, Random Forest, and XGBoost runs side by side.
> Logistic Regression achieved the highest CV macro-F1 of 0.76 on
> the synthetic reef-health task."

---

### STEP 7 — Champion Models (1 minute)

```bash
python -c "
import mlflow
mlflow.set_tracking_uri('sqlite:///artifacts/mlruns.db')
c = mlflow.tracking.MlflowClient()
for name in ['coralsense_reef_health', 'coralsense_restoration_suitability']:
    rm = c.get_registered_model(name)
    versions = c.search_model_versions(f\"name='{name}'\")
    champion = rm.aliases.get('champion')
    print(f'{name}: {len(versions)} versions, champion=v{champion}')
"
```

**Expected output:**
```
coralsense_reef_health: 4 versions, champion=v1
coralsense_restoration_suitability: 4 versions, champion=v1
```

**Say:**
> "The Model Registry holds four versions of each model.
> The champion alias points to version 1 for both tasks.
> Champion aliases can only be moved by an explicit promotion command
> with human approval — never automatically."

---

### STEP 8 — FastAPI Documentation (30 seconds)

Open browser: **http://localhost:8000/docs**

**Point out:**
- GET `/health` — liveness probe
- GET `/model-info` — champion metadata
- POST `/predict/both` — dual prediction

**Say:**
> "The FastAPI service loads models from a portable bundle on disk.
> It does not call the MLflow service at runtime, so API and MLflow
> are fully decoupled."

---

### STEP 9 — Live Prediction (1 minute)

```bash
curl -s -X POST http://localhost:8000/predict/both \
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
  }' | python3 -m json.tool
```

Or use the preflight demo verifier:

```bash
python scripts/demo.py verify
```

**Expected result:** `healthy` + `suitable`, both with high confidence (> 0.98)

**Say:**
> "This Gulf of Mannar observation has good sonar structural signals:
> rugosity 3.5, 60% hard substrate. Environmental conditions are normal.
> The model returns healthy and suitable, which is consistent with
> the synthetic rules. On real data, an ecological expert would need
> to validate these outputs."

---

### STEP 10 — Streamlit Dashboard (1 minute)

Open browser: **http://localhost:8501**

**Navigate through:**
1. **Home** — project overview and disclaimer
2. **Reef Map** — interactive geographic map with colour-coded health
3. **Habitat Health** — bar charts by region
4. **Predict (page 5)** — live prediction form connected to FastAPI
5. **MLOps Status (page 7)** — champion model metadata from registry

**Say:**
> "The dashboard shows the geographic distribution of our synthetic reef
> observations. Page 5 lets you submit a prediction through the UI.
> Page 7 shows the MLflow registry state directly."

---

### STEP 11 — Drift Report (1 minute)

```bash
cat reports/drift_summary.json | python3 -m json.tool | head -30
```

Or navigate to **Drift Monitoring** page in Streamlit.

**Say:**
> "We simulate a production distribution shift: water temperature rises
> 3°C, bleaching percentage rises 20 percentage points, coral cover drops
> 15 points, turbidity rises 5 NTU.
> Evidently detects drift in 4 features.
> The prediction distribution also shifts.
> The recommendation is RETRAIN."

---

### STEP 12 — RETRAIN Recommendation (30 seconds)

```bash
python3 -c "
import json
d = json.load(open('reports/drift_summary.json'))
print('Recommendation:', d['recommendation'])
print('Drifted features:', d.get('n_drifted_features', 'see detail'))
"
```

**Expected output:** `Recommendation: RETRAIN`

---

### STEP 13 — Retraining Dry-Run (1 minute)

```bash
python scripts/run_retraining.py \
  --task health \
  --input data/raw/observations.csv \
  --drift-summary reports/drift_summary.json \
  --dry-run
```

**Expected output:** Validation passed, permission granted, no DB writes.

**Say:**
> "The dry-run validates all 8 data contract checks without writing
> anything to the registry. In particular, it verifies that the input
> CSV has labelled target columns — the unlabelled drift window would
> be rejected here."

**To demonstrate rejection:**

```bash
python scripts/run_retraining.py \
  --task health \
  --input data/production/production.csv \
  --drift-summary reports/drift_summary.json \
  --dry-run
```

**Expected:** Exit code 1 — missing target column.

---

### STEP 14 — CI Workflow (30 seconds)

```bash
cat .github/workflows/ci.yml | grep -A2 "^jobs:"
```

Or open `.github/workflows/ci.yml` in editor.

**Say:**
> "GitHub Actions runs 5 jobs on every push: code quality, tests,
> pipeline validation, ML smoke test, and a package build.
> The canonical MLflow database is never touched in CI.
> No model version is ever registered in CI."

---

### STEP 15 — MLOps Maturity Levels (1 minute)

**Say:**

> **Level 0 (basic automation):** `make test` runs 910 tests. `dvc repro`
> reproduces the full pipeline from data to registered candidate.

> **Level 1 (continuous integration):** The GitHub Actions workflow validates
> code quality, data pipeline, and ML smoke tests on every push.

> **Level 2 (automated deployment):** `docker compose up --build -d` deploys
> the full stack. The API serves both champion models immediately.

> **Level 3 (full MLOps):** Drift monitoring detects distribution shift.
> Retraining requires labelled data, a drift recommendation, and human
> approval. No model can be promoted without `--approve --approver --reason`.
> Rollback is a single command that preserves all model versions.

---

## Shutdown

```bash
python scripts/demo.py stop
# or
docker compose down
```

**Confirm:** `docker ps` shows no CoralSense containers.

---

## Recovery Steps

| Problem | Fix |
|---|---|
| Port already in use | `lsof -i :8000` → `kill -9 <pid>` |
| Container not healthy | `docker compose logs api` to diagnose |
| Bundle missing | `python scripts/export_champions.py` |
| Drift summary missing | `python -m src.monitoring.run_drift --no-html` |
| Test failure | `python -m pytest tests/ -x -v` to isolate |
| Preflight failure | `python scripts/preflight.py` for detailed output |

---

## Likely Faculty Questions and Short Factual Answers

| Question | Answer |
|---|---|
| Why synthetic data? | Real field-deployed sonar hardware and labelled marine surveys are not yet available for this project. |
| Does sonar measure bleaching? | No. Sonar captures structural features (backscatter, rugosity, depth). Bleaching and coral cover require separate observer surveys or optical sensors. |
| How is the champion chosen? | By 5-fold cross-validation macro-F1 on the training set. Test metrics are computed on a separate held-out 20% split. |
| What happens if the model degrades? | Evidently detects drift and recommends RETRAIN. A human runs `run_retraining.py`, reviews the comparison report, and executes `promote.py` with explicit approval. |
| Can CI accidentally promote a model? | No. The canonical database is never opened in CI. Promotion requires `--approve --approver --reason`. |
| What is the champion version? | Version 1 for both tasks. Four total versions are registered (candidate v2–v4 from DVC pipeline runs). |
| Why Logistic Regression won for health? | On this synthetic dataset, Logistic Regression achieved the highest CV macro-F1 (0.7612) among the three algorithms. |
| What is the drift threshold? | p-value = 0.10 (configurable in params.yaml as `drift_threshold`). |
| What does RETRAIN mean? | It means 4 or more features have drifted beyond the p-value threshold. A human must still decide whether and when to retrain. |
| Is the system production-ready? | At the academic level, yes — it demonstrates all MLOps maturity levels. For real marine management, it would require real sensor data and ecological expert validation. |

---

## Commands Reference Card

```bash
# Pre-demo
python scripts/preflight.py                    # system check
make export-models                             # export bundles
make drift                                     # generate drift report

# Demo start
python scripts/demo.py start                   # full start (preflight + docker)
# or manually:
docker compose up --build -d

# URLs
#   MLflow:     http://localhost:5000
#   FastAPI:    http://localhost:8000/docs
#   Streamlit:  http://localhost:8501

# During demo
python scripts/demo.py verify                  # test prediction
python scripts/demo.py status                  # service status

# Demo stop
python scripts/demo.py stop
docker ps                                      # confirm empty

# Tests
make test                                      # 910 tests
make lint                                      # ruff check
```
