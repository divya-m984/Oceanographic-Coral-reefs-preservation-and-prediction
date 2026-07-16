# Oceanographic MLOps — Architecture

> **Project:** Oceanographic: A Machine Learning-Driven Sonar Framework for
> Real-Time Coral Reef Habitat Prediction and Marine Ecosystem Monitoring

> **Synthetic-data disclaimer:** All data used in this project is
> computer-generated. No sonar hardware or field survey data has been
> collected. Sonar features represent structural reef properties that
> **would** be captured by an underwater acoustic survey in a real deployment.

---

## 1. Proposed Data Sources

### 1.1 Acoustic / sonar (structural information)

Sonar is proposed as the primary source of reef structural data. In a physical
deployment, a multibeam echosounder or side-scan sonar tow-fish would capture:

| Feature | Description |
|---|---|
| `depth_m` | Seabed depth from sonar range |
| `sonar_backscatter` | Acoustic reflectance — proxy for substrate hardness |
| `rugosity_index` | Terrain roughness — computed from bathymetric grid |
| `hard_substrate_percentage` | Fraction of hard substrate inferred from backscatter |
| `acoustic_complexity_index` | Spectral complexity of the acoustic return |

Sonar does **not** directly measure bleaching, disease, coral cover, or water
chemistry. Sonar does not capture biological condition directly.
Those require independent observational or sensor inputs.

### 1.2 Environmental sensors

Accompanying in-situ sensors would supply:

| Feature | Description |
|---|---|
| `water_temperature_c` | CTD or moored thermometer |
| `ph` | In-situ pH probe |
| `salinity_ppt` | CTD (conductivity) |
| `dissolved_oxygen_mg_l` | Dissolved-oxygen optode |
| `turbidity_ntu` | Optical turbidity probe |
| `light_intensity` | PAR sensor |
| `current_speed_m_s` | ADCP or acoustic Doppler |

### 1.3 Biological observations

These require trained observer survey or photogrammetry:

| Feature | Description |
|---|---|
| `coral_cover_percentage` | Photo-quadrat or towed video |
| `bleaching_percentage` | Observer bleaching survey |
| `disease_percentage` | Observer disease survey |

### 1.4 Geographic

| Feature | Description |
|---|---|
| `region` | Named reef region (categorical) |
| `latitude_deg`, `longitude_deg` | GPS from survey vessel |

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DATA INGESTION LAYER                            │
│  Sonar survey → backscatter/rugosity/depth                          │
│  Environmental sensors → temperature/pH/salinity/DO/turbidity       │
│  Biological surveys → coral_cover/bleaching/disease                 │
│  (Currently: synthetic data generator — src/data/generate_data.py) │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    VALIDATION LAYER  (M3)                           │
│  src/data/validate.py — Pandera schema                              │
│  Range checks, regex, allowed values, zero-null contract            │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│           PREPROCESSING AND FEATURE ENGINEERING  (M4)               │
│  src/data/preprocess.py — stratified 80/20 split                   │
│  ColumnTransformer: StandardScaler (numeric) +                      │
│                     OneHotEncoder (region)                          │
│  src/features/build_features.py — 6 derived features:              │
│    thermal_stress_index, oxygen_stress_index, acidity_deviation,    │
│    water_quality_index, substrate_stability_score,                  │
│    structural_complexity_score                                       │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   MODEL TRAINING  (M5)                              │
│  src/models/train.py                                                │
│  Algorithms: Logistic Regression, Random Forest, XGBoost           │
│  Selection: 5-fold CV macro-F1 on training set                     │
│  Tasks:                                                             │
│    reef_health → {healthy, stressed, bleached, severely_degraded}  │
│    restoration_suitability → {suitable, moderately_suitable,       │
│                                unsuitable}                          │
│  All runs logged to MLflow (experiment per task)                   │
└───────────────┬─────────────────────────────────┬───────────────────┘
                │                                 │
                ▼                                 ▼
┌───────────────────────────┐    ┌───────────────────────────────────┐
│   MLFLOW TRACKING  (M5)   │    │  MLFLOW MODEL REGISTRY  (M6)      │
│  sqlite:///artifacts/     │    │  coralsense_reef_health           │
│  mlruns.db                │    │  coralsense_restoration_suitability│
│  Experiments:             │    │  champion alias → v1               │
│   coral_reef_health       │    │  candidate → v2/v3/v4 (no alias)  │
│   coral_restoration_      │    └───────────────────────────────────┘
│   suitability             │
└───────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    DVC PIPELINE  (M7)                               │
│  dvc.yaml — 7-stage DAG                                             │
│  generate → validate → preprocess → train → evaluate →             │
│  register_candidate → run_drift                                     │
│  params.yaml — single source of truth for all hyperparameters      │
│  reports/metrics.json — DVC-tracked evaluation metrics             │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  CI/CD PIPELINE  (M8)                               │
│  .github/workflows/ci.yml — 5 GitHub Actions jobs                  │
│  code-quality → tests → pipeline-validation → ml-smoke-test        │
│                       → build                                       │
│  Canonical DB never opened in CI                                    │
│  No model version registered in CI                                  │
└────────────────────────┬────────────────────────────────────────────┘
                         │
           ┌─────────────┼──────────────────────┐
           ▼             ▼                      ▼
┌──────────────┐  ┌─────────────────┐  ┌──────────────────────────┐
│  FASTAPI     │  │  STREAMLIT      │  │  DRIFT MONITORING  (M11) │
│  (M9)        │  │  DASHBOARD      │  │  src/monitoring/         │
│  :8000       │  │  (M10)  :8501   │  │  Evidently 0.7           │
│              │  │                 │  │  Feature drift           │
│  7 endpoints │  │  10 pages       │  │  Prediction drift        │
│  /predict/   │  │  reef map       │  │  Confidence drift        │
│  /health     │  │  governance     │  │  RETRAIN recommendation  │
│  /model-info │  │  drift page     │  └──────────┬───────────────┘
└──────┬───────┘  └────────┬────────┘             │
       │                   │               ┌───────▼─────────────────┐
       │                   │               │  CONTROLLED RETRAINING  │
       │                   │               │  (M13)                  │
       │                   │               │  src/models/retrain.py  │
       │                   │               │  src/models/compare.py  │
       │                   │               │  src/models/promote.py  │
       │                   │               │  src/models/rollback.py │
       │                   │               │  8-check data contract  │
       │                   │               │  Explicit approval gate │
       │                   │               │  No auto-promotion      │
       │                   │               └─────────────────────────┘
       │                   │
       ▼                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  DOCKER DEPLOYMENT  (M12)                           │
│  docker-compose.yml — 4 services                                    │
│  Dockerfile.api / Dockerfile.dashboard / Dockerfile.mlflow          │
│  Bundle mode: deploy/bundles/ (checksum-verified, no MLflow call)  │
│  artifacts/mlruns.db mounted :ro — never modified at runtime       │
│  Named volume mlflow-runtime for runtime DB copy                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. ML Classification Tasks

### 3.1 Reef Health

- **Target:** `reef_health`
- **Classes:** `healthy`, `stressed`, `bleached`, `severely_degraded`
- **Champion algorithm:** Logistic Regression (v1)
- **CV macro-F1:** 0.7612 | **Test macro-F1:** 0.7871
- **Test balanced accuracy:** 0.8012

### 3.2 Restoration Suitability

- **Target:** `restoration_suitability`
- **Classes:** `suitable`, `moderately_suitable`, `unsuitable`
- **Champion algorithm:** XGBoost (v1)
- **CV macro-F1:** 0.7913 | **Test macro-F1:** 0.8029
- **Test balanced accuracy:** 0.8121

All metrics are on the synthetic dataset and must not be
interpreted as evidence of real-world coral reef prediction accuracy.

---

## 4. Preprocessing Pipeline

```
Raw observation (16 sensor columns + region)
    │
    ├── Derived features (6 computed columns, src/features/build_features.py)
    │   ├── thermal_stress_index = max(0, water_temperature_c − 29) × 10
    │   ├── oxygen_stress_index  = max(0, 7 − dissolved_oxygen_mg_l)
    │   ├── acidity_deviation    = |ph − 8.2|
    │   ├── water_quality_index  = f(ph, turbidity, DO, salinity)
    │   ├── substrate_stability_score = f(rugosity, hard_substrate, backscatter)
    │   └── structural_complexity_score = f(acoustic_complexity, rugosity, depth)
    │
    └── ColumnTransformer (fitted on training set only — no leakage)
        ├── StandardScaler → 21 numeric columns
        └── OneHotEncoder  → region (4 categories → 4 binary columns)
                                               ↓
                                    25-column feature matrix
```

---

## 5. MLflow Governance

```
Experiment run
    │
    ├── Log params (algorithm, hyperparameters)
    ├── Log metrics (cv_macro_f1, test_macro_f1, balanced_accuracy)
    ├── Log model artifact (joblib pipeline)
    └── Register in Model Registry
            │
            ├── champion alias → v1  (never moved automatically)
            ├── candidate v2, v3, v4  (no alias)
            │
            └── Explicit promotion (M13):
                    requires --approve --approver --reason
                    requires comparison report
                    requires quality gate re-validation
                    writes promotion receipt
```

---

## 6. Drift Monitoring Pipeline

```
data/raw/observations.csv
    │
    ├── reference window (1,500 rows, unshifted) → data/reference/reference.csv
    └── production window (1,500 rows, shifted)  → data/production/production.csv
             Shifts at scale=1.0:
               +3°C water temperature
               +20pp bleaching
               −15pp coral cover
               +5 NTU turbidity
    │
    ├── Feature drift (Evidently DataDriftPreset → 4 columns drift)
    ├── Prediction drift (chi2_contingency on predicted class distributions)
    └── Confidence drift (KS test on model confidence scores)
                │
                └── reports/drift_summary.json
                         recommendation: RETRAIN
```

---

## 7. Controlled Retraining Flow

```
Trigger: drift_summary.json recommendation = RETRAIN
    │
    ├── scripts/run_retraining.py --task health --input <labelled_csv>
    │       │
    │       ├── Validate input (8 checks):
    │       │     labelled target column, no NaN, min 200 rows,
    │       │     all classes present, min 5 per class, valid features,
    │       │     SHA-256 provenance, drift permission / manual reason
    │       │
    │       ├── Fit fresh preprocessor on train split only
    │       ├── Train LR / RF / XGBoost challengers
    │       ├── Select best challenger by CV macro-F1
    │       ├── Evaluate on holdout split
    │       ├── Register challenger (NO champion alias)
    │       └── Run comparison → reports/comparison_*.json
    │               outcome: eligible_for_promotion | review_required | reject
    │
    └── Explicit promotion (requires human decision):
            python -m src.models.promote
              --model coralsense_reef_health
              --challenger-version <N>
              --approve
              --approver "Name"
              --reason "..."
```

---

## 8. Key Design Decisions

| Decision | Rationale |
|---|---|
| SQLite canonical DB | Portable; no server required for college project |
| Bundle mode in Docker | API does not call MLflow at runtime; avoids service coupling |
| Champion alias never auto-moved | Explicit approval prevents silent model regression |
| Unlabelled drift data rejected | Scientifically invalid for supervised retraining |
| DVC but no DVC remote | Reproducibility without requiring cloud storage |
| Synthetic data only | Real field sonar data not yet available for this project |

---

## 9. Future Real-Sensor Phase

When physical sonar hardware and labelled marine surveys become available:

1. Replace `src/data/generate_data.py` with a real sensor ingestion module.
2. Configure a DVC remote (S3 / GCS / Azure Blob) for large survey files.
3. Connect MLflow to a persistent PostgreSQL backend.
4. Deploy the FastAPI service to a cloud VM near the survey region.
5. Integrate with the existing champion/challenger governance flow.
6. Obtain ecological domain expert sign-off before using predictions for
   any conservation management decision.
