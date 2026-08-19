# Oceanographic — Dataset Inventory & Scientific Audit

**Repository:** `/home/BAAHbun/Projects/coralsense-mlops` · branch `main` @ `53f1430`
**Date:** 2026-08-19 · **Mode:** read-only inspection. No repository file was created, modified, staged, or committed.
**Working outputs:** `/tmp/oceanographic-dataset-audit/`

---

## Preservation note

This document is the permanent, unaltered record of the dataset inventory and
scientific audit carried out on **2026-08-19**. Its scientific conclusions have
not been revised. Only this preservation note was added when the report was
moved into version control.

**Audited canonical dataset**

| Field | Value |
|:------|:------|
| Path | `data/raw/observations.csv` |
| Rows × columns | 15,000 × 21 |
| Size | 2,471,475 bytes |
| Generator | `src/data/generate_data.py`, fixed seed 42 |
| Provenance | **Synthetic prototype dataset** — algorithmically generated labels |
| **SHA-256** | **`a03cb3e92ba1904ae07147da95f96aa689d092d56fc41b040b701a101ad8f458`** |

Every finding below applies to that exact file. If the hash of
`data/raw/observations.csv` no longer matches, this audit describes a different
dataset and must be re-run before its conclusions are relied upon.

### Two different sets of numbers appear in this report — do not conflate them

This report contains **audit diagnostic** measurements produced specifically for
the leakage investigation. They were computed with an ad-hoc
`HistGradientBoostingClassifier` under 5-fold cross-validation over the full
15,000 rows, purely to quantify circular supervision. **They are not the
project's registered model metrics, and they do not replace them.**

| | reef_health macro-F1 | restoration macro-F1 |
|:--|:--|:--|
| **Audit diagnostic** — closed-form reconstruction of the generator's scoring rules, no machine learning at all | **≈ 0.770** | **≈ 0.812** |
| **Audit diagnostic** — ad-hoc learned model, all 21 features, 5-fold CV | **≈ 0.760** | **≈ 0.791** |
| **Registered champion metrics** — canonical, from `reports/metrics.json` and the MLflow registry (held-out 20 % test split) | **0.7871** | **0.8029** |

The first two rows are the evidence for circular supervision: a closed-form
arithmetic re-computation of the label formula, with the generator's Gaussian
noise term removed, **matches or exceeds** what a trained model achieves. The
model is recovering the synthetic generation rules rather than learning an
independently observed ecological relationship.

The third row is the project's canonical champion performance and remains the
authoritative figure quoted in the README, model cards, dashboard, and API. The
audit did not retrain, re-register, or alter any model, and the closed-form
diagnostic must never be substituted for a registered metric.

---

## 1. Executive Summary

The current 15,000-row dataset is internally clean, fully documented, reproducible, and **scientifically unusable as evidence that the models predict reef condition**. It is a high-quality *software* artefact and a null *scientific* artefact.

The decisive finding is quantitative, not rhetorical. `src/data/generate_data.py` computes `reef_health` as a fixed weighted sum of seven columns, adds Gaussian noise of σ = 0.075, and thresholds. Those same seven columns are then handed to the model as features. I re-implemented the generator's scoring arithmetic, **deleted the noise term**, re-thresholded, and compared to the stored labels:

| Target | Closed-form formula recovers stored label | Trained GBM (all 21 features, 5-fold CV) |
|---|---|---|
| `reef_health` | **82.7 %** exact (100 % within one ordinal class) | 81.8 % accuracy / macro-F1 0.760 |
| `restoration_suitability` | **84.4 %** exact | 83.0 % accuracy / macro-F1 0.791 |

**A pocket calculator outperforms the trained model.** The model is not discovering a biological relationship; it is reverse-engineering a spreadsheet formula, and its accuracy ceiling is set exactly by the noise the generator injected. This is textbook circular supervision (severity **CRITICAL**).

Three corroborating results:

- **Ablation.** The 7 label-generating features alone reach macro-F1 0.759 vs 0.760 for all 21 features. The other 14 features contribute **+0.002**. Conversely, the 8 *non*-label-generating features reach only 0.470 — barely above the 0.575 majority baseline. Every feature that isn't in the label formula is nearly worthless, which is exactly what circular supervision predicts and not what a real ecological signal looks like.
- **Geography.** Random stratified CV reports macro-F1 0.763. Leave-one-region-out collapses to **0.41–0.68**. The random split overstates generalisation by up to 0.35 macro-F1 (severity **HIGH**).
- **Time.** Timestamps are decorative. Correlation of every environmental variable with time is |r| < 0.014; month-of-year η² ≈ 0.0005; year η² ≈ 0.0002. Class balance is flat across 2018–2024 (healthy 56.1 %–59.1 %) despite that window containing two global bleaching events. A temporal hold-out performs identically to random CV — not because the model generalises through time, but because **there is no temporal signal to fail on** (severity **HIGH** for scientific realism).

On deployment: 3 of the 16 required inference features — `coral_cover_percentage`, `bleaching_percentage`, `disease_percentage` — cannot be obtained from a boat-mounted sonar and water-sensor rig. They require a diver or a downward camera plus benthic image analysis. They are **biological ground truth, not predictors** (severity **CRITICAL** for the deployment story). A further three (`rugosity_index`, `acoustic_complexity_index`, and to a degree `sonar_backscatter`) require processing chains the project does not yet have.

On the external material: `data/audit_inbox/` contains **five PDF journal articles and zero datasets**. The IUCN zip named in the brief was not in the repository; I located it at `~/Downloads/` and audited it in place. It contains one polygon for *Catalaphyllia jardinei* with **no environmental, acoustic, or reef-condition variables**, and — verified by point-in-polygon test — **zero geographic overlap with any of the four project regions** (nearest range boundary 388 km from Lakshadweep, 1,758 km from Gulf of Kutch). It is reference data only, and IUCN terms **prohibit redistribution**, so it must never enter this Git repository.

The recommendation is **D — hybrid**, with a hard partition: keep the 15k set as a frozen, clearly-labelled synthetic CI/demo benchmark; build a separate real-data track for any scientific claim. Do not regenerate the synthetic set hoping to fix the leakage — the leakage is structural, not parametric. Any label built from the same variables you feed the model is circular at *any* noise level.

### Severity register

| # | Issue | Severity |
|---|---|---|
| 1 | Circular label construction (labels are a formula over the features) | **CRITICAL** |
| 2 | Direct target leakage — 7/8 label-generating columns served as features | **CRITICAL** |
| 3 | Deployment-unavailable features (`coral_cover`, `bleaching`, `disease`) | **CRITICAL** |
| 4 | Engineered proxy leakage (`thermal_stress_index`, `oxygen_stress_index`, `water_quality_index` re-derive score components) | **HIGH** |
| 5 | Random stratified split hides geographic non-generalisation | **HIGH** |
| 6 | Geographic leakage — region identity carries the signal; LORO collapses | **HIGH** |
| 7 | Synthetic-vs-real provenance — no real observation anywhere in the pipeline | **HIGH** |
| 8 | Coordinates uniform in rectangles; land/deep-ocean points; no reef geometry | **HIGH** |
| 9 | Distributional realism (mean bleaching 37 %, 399 rows at 37.5 °C) | **HIGH** |
| 10 | IUCN redistribution restriction if the zip is ever committed | **HIGH** (latent) |
| 11 | Temporal signal absent — no seasonality, no bleaching years | **MEDIUM** |
| 12 | Feature redundancy — 15 pairs at \|r\| > 0.8 | **MEDIUM** |
| 13 | 14 rows with coral cover = 0 but bleaching/disease > 0 | **LOW** |
| 14 | Clipping saturation (887 rows bleaching = 0, 786 = 100) | **LOW** |
| 15 | External inbox contains literature, not data | **INFO** |
| 16 | Temporal leakage | **NONE FOUND** |
| 17 | Missing values / duplicates / malformed records | **NONE FOUND** |

---

## 2. Repository Safety State

`pwd` confirmed `/home/BAAHbun/Projects/coralsense-mlops`. Branch `main`, level with `origin/main`, HEAD `53f1430`.

Pre-existing untracked user content, **left untouched**:

```
?? data/audit_inbox/          (5 PDFs, user-supplied)
?? matlab_crash_dump.283300-1
?? matlab_crash_dump.283487-1
?? matlab_crash_dump.283712-1
```

### Protected artefact hashes — before and after

| File | SHA-256 | Bytes | Changed? |
|---|---|---|---|
| `artifacts/mlruns.db` | `b76a401522754ad050793f392ba2cdf0e8f9e4b76140dc8ebb9f604c95f7c477` | 1,208,320 | No |
| `data/raw/observations.csv` | `a03cb3e92ba1904ae07147da95f96aa689d092d56fc41b040b701a101ad8f458` | 2,471,475 | No |
| `reports/drift_summary.json` | `252785f69805d593dea6ddbfa4e123759a176f19925a77bdcba2446d5f13eade` | 4,168 | No |
| `models/best_model_health.joblib` | `586096df9e164420363f459471c91ac2e5258ab9878e2e150ce3b291184f42d4` | 2,704 | No |
| `models/best_model_restoration.joblib` | `a93d71dbd6303363bbfcd27014d41254f3e34eebf4157c6249a6172f18865ca8` | 3,269,590 | No |
| `data/processed/preprocessor_health.joblib` | `e625aa747c8b6ac2f2b6e0a528279cc4c96906849046b8fd875238875eabb1d5` | 5,256 | No |
| `data/processed/preprocessor_restoration.joblib` | `2c45c9f43fb5aed5b92a1df5a83ef8cf2c6911d4dc731758282cf1701976a08e` | 5,256 | No |

**`observations.csv` matches the previously known canonical hash exactly** — verified, not assumed. `git diff HEAD` is empty; no tracked file was modified.

---

## 3. Current 15k Dataset Inventory

| Property | Value |
|---|---|
| Path | `data/raw/observations.csv` |
| Rows | 15,000 |
| Columns | 21 |
| File size | 2,471,475 bytes (2.36 MiB) |
| In-memory (deep) | 4,036,753 bytes |
| SHA-256 | `a03cb3e9…1ad8f458` |
| Provenance | 100 % synthetic — `src/data/generate_data.py`, seed 42, `n_samples: 15000` |

A byte-identical copy exists at `data/raw/observations_validated.csv` (same size; Pandera pass-through). Derived project data: `data/processed/` (train/test splits, preprocessors, `feature_metadata.json`), `data/reference/reference.csv` (215 KB) and `data/production/production.csv` (226 KB) — both drift-monitoring windows resampled from the same synthetic source with an artificial shift applied. **Every byte of data in this project traces back to one seeded RNG call.**

---

## 4. Schema and Data Quality

### Column order, dtypes, semantics

| # | Column | dtype | Semantic role | Deployment class (§13) |
|---|---|---|---|---|
| 1 | `timestamp` | datetime64[ns] | metadata (excluded from features) | C/system |
| 2 | `latitude` | float64 | metadata (excluded) | C — GPS |
| 3 | `longitude` | float64 | metadata (excluded) | C — GPS |
| 4 | `region` | object | categorical feature | C — GPS-derived |
| 5 | `depth_m` | float64 | numeric feature | A — echosounder |
| 6 | `water_temperature_c` | float64 | numeric feature | B — thermistor |
| 7 | `ph` | float64 | numeric feature | B — pH probe |
| 8 | `salinity_ppt` | float64 | numeric feature | B — CTD |
| 9 | `dissolved_oxygen_mg_l` | float64 | numeric feature | B — optode |
| 10 | `turbidity_ntu` | float64 | numeric feature | B — nephelometer |
| 11 | `light_intensity` | float64 | numeric feature | B — PAR sensor |
| 12 | `current_speed_m_s` | float64 | numeric feature | B — ADCP/tilt |
| 13 | `sonar_backscatter` | float64 | numeric feature | A — calibrated sonar |
| 14 | `rugosity_index` | float64 | numeric feature | A* — needs MBES + DEM chain |
| 15 | `hard_substrate_percentage` | float64 | numeric feature | A* — needs ARC classifier |
| 16 | `acoustic_complexity_index` | float64 | numeric feature | H* — needs hydrophone, not sonar |
| 17 | `coral_cover_percentage` | float64 | numeric feature **← label input** | **E/F — ground truth** |
| 18 | `bleaching_percentage` | float64 | numeric feature **← label input** | **E/F — ground truth** |
| 19 | `disease_percentage` | float64 | numeric feature **← label input** | **E — diver only** |
| 20 | `reef_health` | object | **target** | — |
| 21 | `restoration_suitability` | object | **target** | — |

Type split: 15 numeric features + 1 categorical (`region`) + 2 spatial floats + 1 timestamp + 2 string targets. `src/features/build_features.py` appends 6 engineered numerics before modelling, giving 21 numeric + 1 categorical = 22 model inputs.

### Quality — clean

| Check | Result |
|---|---|
| Missing values (all 21 columns) | **0** |
| Duplicate full rows | **0** |
| Duplicate coordinate pairs | **0** (all 15,000 unique) |
| Duplicate coordinate + timestamp | **0** |
| Duplicate timestamps | **0** (15,000 unique) |
| Constant columns | none |
| Near-constant columns (>95 % one value) | none |
| Malformed timestamps | 0 |
| Infinite values | 0 |
| Negative values where impossible | 0 |
| Percentages > 100 | 0 |
| Rows outside declared region bbox | **0** |

Data hygiene is exemplary. **INFO.**

### Quality — suspicious

| Finding | Count | Assessment |
|---|---|---|
| `coral_cover = 0` **and** `bleaching > 0` | **14** | Physically impossible — bleaching of coral that does not exist. Generator computes bleaching from temperature independently of cover. **LOW** |
| `coral_cover = 0` **and** `disease > 0` | **14** | Same 14 rows, same cause. **LOW** |
| `bleaching + cover > 100` | 3,553 | **Not an error** — different denominators (bleaching is % *of coral tissue*, cover is % *of benthos*). Flagged only because the units are easy to misread; document explicitly. **INFO** |
| `bleaching = 0.0` exactly | 887 | Lower-clip pile-up |
| `bleaching = 100.0` exactly | 786 | Upper-clip pile-up — 5.2 % of reefs *totally* bleached. **LOW** (realism) |
| `disease = 0.0` exactly | 1,153 | Lower-clip pile-up |
| `water_temperature = 37.5` exactly | **399** | Upper clip. Sustained 37.5 °C is lethal, not stressful, for scleractinian corals. **HIGH** (realism) |
| `water_temperature > 34 °C` | 870 (5.8 %) | Implausibly frequent |
| `turbidity = 0.05` exactly | 635 | Lower clip |
| `light_intensity = 5.0` exactly | 217 | Lower clip |
| `ph = 7.60` exactly | 123 | Lower clip |
| `ph < 7.8` | 1,464 (9.8 %) | Extreme for open-reef seawater |
| `salinity > 40 ppt` | 928 (6.2 %) | Plausible for Gulf of Kutch only; check regional attribution |

Clip saturation is a generator artefact: `np.clip` piles probability mass onto boundary values, creating spikes no real sensor produces. This is directly visible to the model as an exploitable signal.

---

## 5. Geographic Audit

### Per-region inventory

| Region | Rows | % | Declared bbox [lat_min, lat_max, lon_min, lon_max] | Observed lat | Observed lon | Unique coords |
|---|---|---|---|---|---|---|
| Andaman and Nicobar Islands | 6,300 | 42.0 | [6.5, 14.0, 92.0, 94.0] | 6.5000 – 13.9907 | 92.0010 – 93.9996 | 6,300 / 6,300 |
| Lakshadweep | 3,450 | 23.0 | [10.0, 12.5, 72.0, 74.0] | 10.0004 – 12.4994 | 72.0001 – 73.9998 | 3,450 / 3,450 |
| Gulf of Mannar | 3,000 | 20.0 | [8.5, 10.5, 78.0, 80.5] | 8.5001 – 10.4987 | 78.0010 – 80.4995 | 3,000 / 3,000 |
| Gulf of Kutch | 2,250 | 15.0 | [22.0, 24.5, 68.0, 71.0] | 22.0007 – 24.4996 | 68.0030 – 70.9943 | 2,250 / 2,250 |

Allocation matches `_REGION_WEIGHTS` exactly (0.42 / 0.23 / 0.20 / 0.15). All four regions are real Indian reef systems and the bounding boxes are broadly correct at the coarse level.

### Are coordinates spatially distributed or uniform-in-rectangle? — **Uniform in rectangle. Confirmed.**

Kolmogorov–Smirnov test of each coordinate against a Uniform distribution over its declared box:

| Region | KS p (lat) | KS p (lon) | Verdict |
|---|---|---|---|
| Andaman and Nicobar | 0.878 | 0.561 | indistinguishable from uniform |
| Gulf of Mannar | 0.886 | 0.425 | indistinguishable from uniform |
| Gulf of Kutch | 0.372 | 0.663 | indistinguishable from uniform |
| Lakshadweep | 0.271 | 0.261 | indistinguishable from uniform |

Source confirms it — `generate_data.py:228-229`:
```python
latitude  = rng.uniform(lat_min, lat_max, n)
longitude = rng.uniform(lon_min, lon_max, n)
```

Consequences (severity **HIGH**):

1. **No reef geometry.** The Andaman box (6.5–14 °N × 92–94 °E) is ~830 km × 220 km of mostly open Bay of Bengal, plus land. Reefs occupy a fraction of a percent of it. Thousands of "observations" sit over abyssal water or dry land.
2. **No spatial structure.** Correlation between coordinates and every environmental variable is |r| < 0.05 in all four regions (max observed 0.041). Latitude and longitude carry **zero** information beyond region membership.
3. **No spatial autocorrelation.** Nearest-neighbour label agreement exceeds chance by only 0.001–0.021:

| Region | NN label agreement (health) | Chance | Excess | Median NN distance |
|---|---|---|---|---|
| Andaman and Nicobar | 0.760 | 0.759 | **+0.001** | 0.023° (~2.6 km) |
| Gulf of Kutch | 0.359 | 0.338 | **+0.021** | 0.027° |
| Gulf of Mannar | 0.295 | 0.281 | **+0.014** | 0.020° |
| Lakshadweep | 0.548 | 0.537 | **+0.012** | 0.018° |

Real reefs show strong spatial autocorrelation over these distances. Here there is essentially none — the excess is attributable to shared region priors, not to space.

**Implication for CV design:** within-region spatial-block CV would change nothing, because there is no within-region spatial structure to leak. The geographic problem is entirely **between** regions (§12).

### Region as a target proxy

`region` alone predicts the targets strongly (health macro-F1 0.499; restoration macro-F1 **0.657**, the single best predictor of restoration suitability):

| Region | healthy | stressed | bleached | severely_degraded |
|---|---|---|---|---|
| Andaman and Nicobar | **86.1 %** | 13.1 % | 0.8 % | **0.0 %** (1 row) |
| Lakshadweep | 67.4 % | 28.5 % | 4.1 % | 0.1 % (3 rows) |
| Gulf of Mannar | 24.4 % | 34.7 % | 29.9 % | 11.0 % |
| Gulf of Kutch | 6.6 % | 23.0 % | 22.4 % | **48.0 %** |

Andaman has **1** severely-degraded row in 6,300; Kutch has 1,080 in 2,250. The per-region Beta latent (`_REGION_STRESS`) makes region a near-deterministic prior, and the model leans on it heavily.

---

## 6. Temporal Audit

| Property | Value |
|---|---|
| Earliest | 2018-01-01 06:13:19 |
| Latest | 2024-12-31 19:03:41 |
| Span | 2,556 days (7.0 years) |
| Unique timestamps | 15,000 / 15,000 |
| Median inter-observation gap | 10,167 s (2.82 h) |
| Max gap | 46.1 h |
| Gaps > 24 h | 55 |

Rows per year: 2018 = 2,099 · 2019 = 2,087 · 2020 = 2,201 · 2021 = 2,135 · 2022 = 2,200 · 2023 = 2,120 · 2024 = 2,158. Rows per month range 1,157 (Feb) – 1,295 (Jul), consistent with uniform draw. Per-region year counts are equally flat.

### Do timestamps affect environmental values? — **No. They are independent random metadata.**

| Variable | Pearson r vs time | month-of-year η² | year η² |
|---|---|---|---|
| `water_temperature_c` | +0.0044 | 0.00064 | 0.00009 |
| `bleaching_percentage` | −0.0010 | 0.00061 | 0.00011 |
| `coral_cover_percentage` | −0.0132 | 0.00041 | 0.00033 |
| `ph` | −0.0037 | 0.00047 | 0.00023 |
| `turbidity_ntu` | +0.0021 | 0.00042 | 0.00028 |
| `dissolved_oxygen_mg_l` | −0.0023 | 0.00067 | 0.00016 |

η² < 0.001 everywhere. Time explains under one tenth of one percent of variance in anything. Confirmed at source (`generate_data.py:232-233`) — timestamps are drawn `rng.integers(0, _TOTAL_SECONDS, n)` and never enter any feature or label computation.

Class balance by year is correspondingly flat (healthy 56.1 % → 59.1 %, no trend; severely_degraded 8.7 % → 10.0 %).

**Scientific significance (HIGH).** The 2018–2024 window contains the 2020 Indian Ocean warming event and the 2023–24 fourth global bleaching event. A realistic Indian-reef dataset over this period would show pronounced inter-annual bleaching spikes and a clear monsoon/pre-monsoon seasonal cycle in temperature and turbidity. This dataset shows a flat line. Any dashboard time-series is displaying noise.

**Corollary:** there is **no temporal leakage** — but only because there is no temporal signal at all. A temporal hold-out is currently a meaningless test (§12).

---

## 7. Class Balance

### `reef_health`

| Class | Count | % |
|---|---|---|
| healthy | 8,631 | 57.54 |
| stressed | 3,363 | 22.42 |
| bleached | 1,592 | 10.61 |
| severely_degraded | 1,414 | 9.43 |

Imbalance ratio (max/min) **6.10** · Entropy **1.607 bits** of max 2.000 (normalised 0.804) · Majority-class baseline accuracy **0.5754**

### `restoration_suitability`

| Class | Count | % |
|---|---|---|
| suitable | 8,991 | 59.94 |
| moderately_suitable | 4,534 | 30.23 |
| unsuitable | 1,475 | 9.83 |

Imbalance ratio **6.10** · Entropy **1.293 bits** of max 1.585 (normalised 0.816) · Majority baseline **0.5994**

Moderate imbalance, correctly handled by `class_weight: balanced` in `params.yaml`. Reported CV macro-F1 (~0.76 / 0.79) is genuinely above baseline **on this data** — the objection is to what the data represents, not to the metric arithmetic.

### By region

Health — see §5 table. Restoration:

| Region | suitable | moderately_suitable | unsuitable |
|---|---|---|---|
| Andaman and Nicobar | 82.2 % | 17.6 % | 0.2 % |
| Lakshadweep | 80.7 % | 19.2 % | 0.2 % |
| Gulf of Mannar | 32.1 % | 59.6 % | 8.3 % |
| Gulf of Kutch | 2.9 % | 43.5 % | **53.6 %** |

Andaman + Lakshadweep hold 13 + 6 = **19 of 1,475** `unsuitable` rows. Leave-one-region-out training on those regions leaves a minority class effectively unlearnable — the mechanism behind the LORO collapse in §12.

### By year

Flat (health: healthy 56.1–59.1 %; restoration: suitable 58.6–60.6 %). No inter-annual dynamics.

### Target cross-tabulation

| | suitable | moderately_suitable | unsuitable |
|---|---|---|---|
| healthy | 7,050 | 1,535 | 46 |
| stressed | 1,639 | 1,492 | 232 |
| bleached | 272 | 998 | 322 |
| severely_degraded | 30 | 509 | 875 |

Strongly diagonal. The two "independent" targets share `water_temperature_c`, `ph`, `turbidity_ntu`, and `coral_cover_percentage` as scoring inputs, so they are largely restatements of the same latent stress axis. `get_feature_columns()` correctly prevents *cross-target* leakage (neither target is a feature for the other) — that particular guard works and is the one leakage class the project already handles properly.

---

## 8. Feature Distribution Findings

| Feature | min | p01 | p25 | median | mean | p75 | p95 | p99 | max | std | IQR | outliers >3·IQR | skew |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `depth_m` | 0.50 | 0.76 | 6.67 | 12.70 | 14.28 | 19.99 | 32.79 | 37.19 | 38.00 | 9.50 | 13.32 | 0 | +0.64 |
| `water_temperature_c` | 20.85 | 25.24 | 28.40 | 29.36 | 29.76 | 30.43 | 34.68 | 37.50 | 37.50 | 2.29 | 2.03 | **535** | +1.41 |
| `ph` | 7.60 | 7.61 | 7.93 | 8.04 | 8.03 | 8.14 | 8.26 | 8.33 | 8.42 | 0.16 | 0.21 | 0 | −0.47 |
| `salinity_ppt` | 31.47 | 32.28 | 33.71 | 34.64 | 35.10 | 35.57 | 40.68 | 42.55 | 43.75 | 2.20 | 1.86 | **586** | +1.71 |
| `dissolved_oxygen_mg_l` | 2.00 | 2.96 | 5.86 | 6.80 | 6.70 | 7.72 | 8.72 | 9.29 | 10.12 | 1.40 | 1.86 | 0 | −0.50 |
| `turbidity_ntu` | 0.05 | 0.05 | 1.02 | 2.37 | 4.32 | 5.06 | 16.72 | 25.88 | 30.86 | 5.30 | 4.04 | **618** | +2.23 |
| `light_intensity` | 5.00 | 5.00 | 215.38 | 475.50 | 576.63 | 861.62 | 1420.9 | 1737.4 | 2034.7 | 441.27 | 646.24 | 0 | +0.77 |
| `current_speed_m_s` | 0.005 | 0.01 | 0.15 | 0.29 | 0.30 | 0.42 | 0.68 | 0.85 | 0.95 | 0.19 | 0.27 | 0 | +0.65 |
| `sonar_backscatter` | −36.00 | −34.25 | −22.57 | −16.57 | −17.43 | −11.71 | −6.34 | −4.38 | −2.00 | 7.28 | 10.86 | 0 | −0.37 |
| `rugosity_index` | 1.00 | 1.00 | 2.36 | 3.01 | 2.93 | 3.56 | 4.13 | 4.42 | 4.80 | 0.81 | 1.20 | 0 | −0.33 |
| `hard_substrate_percentage` | 0.00 | 0.70 | 33.60 | 50.10 | 48.08 | 63.80 | 79.30 | 86.00 | 100.00 | 20.34 | 30.20 | 0 | −0.32 |
| `acoustic_complexity_index` | 0.05 | 0.16 | 0.40 | 0.52 | 0.51 | 0.62 | 0.75 | 0.82 | 0.95 | 0.15 | 0.22 | 0 | −0.18 |
| `coral_cover_percentage` | 0.00 | 8.60 | 32.40 | 44.40 | 43.83 | 55.30 | 69.30 | 76.90 | 90.00 | 15.76 | 22.90 | 0 | −0.10 |
| `bleaching_percentage` | 0.00 | 0.00 | 13.30 | 29.10 | **37.09** | 54.40 | 96.60 | 100.00 | 100.00 | 29.98 | 41.10 | 0 | +0.78 |
| `disease_percentage` | 0.00 | 0.00 | 5.90 | 12.40 | **15.32** | 21.90 | 40.10 | 53.90 | 60.00 | 12.63 | 16.00 | 0 | +1.07 |

`latitude` shows 2,250 "3·IQR outliers" — this is simply the Gulf of Kutch cluster at 22–24.5 °N being disjoint from the 6.5–14 °N group. Multimodal, not erroneous.

### Scientific plausibility

**Plausible:** `depth_m` (0.5–38 m, appropriate for shallow reef work); `ph` (7.60–8.42, wide but defensible); `dissolved_oxygen_mg_l` (2.0–10.1 mg/L); `sonar_backscatter` (−36 to −2 dB, correct sign and magnitude for calibrated backscatter, hard reef higher than soft sediment); `rugosity_index` (1.0–4.8, matches published chain-and-tape ranges); `current_speed_m_s` (0.005–0.95 m/s); `salinity_ppt` (31.5–43.8; the hypersaline tail is genuinely characteristic of Gulf of Kutch); `light_intensity` (Beer–Lambert with a turbidity-dependent k_d, correctly implemented and explicitly corrected in a code comment).

**Implausible — severity HIGH:**

- **`bleaching_percentage`: mean 37.1 %, median 29.1 %, p75 54.4 %, 786 rows at exactly 100 %.** This says the *average* Indian reef, on an average day across seven years, has over a third of its coral tissue bleached, and 5.2 % of all observations are total bleach-outs. Published baseline bleaching prevalence on Indian reefs outside acute events is typically single-digit percent. This distribution describes a permanent global catastrophe, not a monitoring baseline. Root cause: `bleaching_base = thermal_excess × 20 + stress × 32` fires whenever temperature exceeds 28.5 °C — which, given regional baselines of 27–32 °C, is most of the time.
- **`water_temperature_c` = 37.5 °C in 399 rows; > 34 °C in 870.** Above roughly 33 °C corals bleach within days and die; 37.5 °C is acute lethal. These are not "stressed reef" readings, they are "no reef remains" readings. Root cause: Gulf of Kutch base range extends to 36 °C, and `thermal_push = stress × 0.40 × t_range` adds up to 6.4 °C more.
- **`disease_percentage`: mean 15.3 %, p99 53.9 %.** Coral disease prevalence above ~10 % is normally reported as an outbreak. A 15 % mean makes outbreak the resting state.
- **`turbidity_ntu` skew +2.23 with 635 rows pinned at the 0.05 lower clip.** The log-uniform base is a good modelling choice; the clip pile-up is not.

The generator's *directional* reasoning is sound and well-cited throughout. The *magnitudes* are not calibrated to any observational distribution, and the stress-driven pushes are applied to nearly every row rather than to episodic events.

---

## 9. Correlation and Redundancy Findings

15 feature pairs exceed |r| > 0.8 (Pearson or Spearman) — full matrices in `pearson_matrix.csv` / `spearman_matrix.csv`.

| Feature A | Feature B | Pearson | Spearman | Nature |
|---|---|---|---|---|
| `hard_substrate_percentage` | `substrate_stability_score` | **+0.994** | +0.994 | derived-from-source; near-duplicate |
| `bleaching_percentage` | `thermal_stress_index` | **+0.945** | +0.896 | **proxy leakage** — both encode thermal excess |
| `acoustic_complexity_index` | `structural_complexity_score` | +0.935 | +0.932 | derived-from-source |
| `rugosity_index` | `substrate_stability_score` | +0.929 | +0.927 | derived-from-source |
| `acoustic_complexity_index` | `coral_cover_percentage` | +0.929 | +0.927 | generator-imposed (ACI computed *from* cover) |
| `water_temperature_c` | `thermal_stress_index` | +0.924 | **+0.989** | pure monotone transform |
| `sonar_backscatter` | `substrate_stability_score` | +0.924 | +0.922 | shared `structure` latent |
| `coral_cover_percentage` | `structural_complexity_score` | +0.920 | +0.920 | transitive via ACI |
| `sonar_backscatter` | `hard_substrate_percentage` | +0.906 | +0.903 | shared `structure` latent |
| `sonar_backscatter` | `rugosity_index` | +0.904 | +0.901 | shared `structure` latent |
| `rugosity_index` | `hard_substrate_percentage` | +0.884 | +0.880 | shared `structure` latent |
| `thermal_stress_index` | `water_quality_index` | −0.886 | −0.833 | WQI is 40 % TSI by construction |
| `bleaching_percentage` | `water_quality_index` | −0.872 | −0.844 | **proxy leakage** |
| `water_temperature_c` | `bleaching_percentage` | +0.854 | +0.884 | generator-imposed |
| `water_temperature_c` | `water_quality_index` | −0.815 | −0.804 | proxy |

Notable 0.6–0.8 pairs: `ph`↔`disease_percentage` (−0.756, generator-imposed), `dissolved_oxygen_mg_l`↔`oxygen_stress_index` (−0.791), `depth_m`↔`light_intensity` (−0.632, the only *physically* motivated correlation in the list — Beer–Lambert).

### Redundancy verdict — severity MEDIUM

Four of the six engineered features are near-duplicates of their sources and carry no new information:

| Derived feature | Strongest source correlation | Adds information? |
|---|---|---|
| `substrate_stability_score` | `hard_substrate_percentage` +0.994 | No — 0.6× a rescaled copy |
| `structural_complexity_score` | `acoustic_complexity_index` +0.935 | No |
| `thermal_stress_index` | `water_temperature_c` Spearman +0.989 | No — clipped monotone map |
| `oxygen_stress_index` | `dissolved_oxygen_mg_l` −0.791 | Marginal (clipping only) |
| `acidity_deviation` | `ph` −0.695 | Yes — |·| makes it non-monotone |
| `water_quality_index` | composite | **Negative value** — see §11 |

Three raw features (`sonar_backscatter`, `rugosity_index`, `hard_substrate_percentage`) all share the single `structure` Beta latent and are mutually correlated at ~0.9. In effect this dataset has roughly **three independent latent dimensions** (`stress`, `structure`, `clarity`) plus depth and noise — not 21 features. That is why adding 14 features to the 7 label-generating ones improves macro-F1 by 0.002.

### Mutual information (nats; target entropy: health 1.114, restoration 0.897)

| Rank | `reef_health` | MI | % of H(Y) | | `restoration_suitability` | MI | % of H(Y) |
|---|---|---|---|---|---|---|---|
| 1 | `water_quality_index` | 0.621 | **55.7 %** | | `turbidity_ntu` | 0.296 | 33.0 % |
| 2 | `bleaching_percentage` | 0.520 | 46.7 % | | `water_quality_index` | 0.273 | 30.4 % |
| 3 | `water_temperature_c` | 0.461 | 41.4 % | | `structural_complexity_score` | 0.268 | 29.8 % |
| 4 | `thermal_stress_index` | 0.430 | 38.6 % | | `coral_cover_percentage` | 0.231 | 25.8 % |
| 5 | `disease_percentage` | 0.315 | 28.3 % | | `ph` | 0.223 | 24.8 % |
| 6 | `ph` | 0.274 | 24.6 % | | `water_temperature_c` | 0.208 | 23.2 % |
| 7 | `acoustic_complexity_index` | 0.255 | 22.9 % | | `hard_substrate_percentage` | 0.208 | 23.2 % |

A single engineered feature, `water_quality_index`, carries **55.7 % of the entire entropy of `reef_health`**. No real environmental covariate behaves this way. This is the signature of a variable that is algebraically related to the label.

---

## 10. Label Generation Analysis

Verified line-by-line against `src/data/generate_data.py`. Not summarised from comments.

### Latent structure

Each region draws two Beta latents per row (`generate_data.py:206-217`):

| Region | `stress` α,β (mean) | `structure` α,β (mean) |
|---|---|---|
| Lakshadweep | 1.8, 3.5 (0.34) | 3.5, 1.8 (0.66) |
| Gulf of Mannar | 2.5, 2.5 (0.50) | 2.5, 2.5 (0.50) |
| Gulf of Kutch | 4.0, 2.0 (0.67) | 1.5, 3.5 (0.30) |
| Andaman and Nicobar | 1.5, 4.0 (0.27) | 4.0, 1.8 (0.69) |

A third latent is derived: `clarity = clip(1 − 0.55·stress + 0.25·structure + N(0, 0.10), 0, 1)`.

### Feature generation — complete specification

| Feature | Formula | Distribution | Depends on | Independent? |
|---|---|---|---|---|
| `latitude`/`longitude` | `U(bbox)` | Uniform | region bbox only | Independent |
| `timestamp` | `TS_START + U{0, 220 838 399} s` | Uniform | nothing | **Independent of everything** |
| `depth_m` | `U(d_min, d_max)` | Uniform | region | Independent |
| `water_temperature_c` | `U(t_min,t_max) + stress·0.40·t_range + N(0,0.35)`, clip `[t_min−1.5, t_max+1.5]` | Uniform + shift | **stress**, region | Derived |
| `ph` | `U(ph_min,ph_max) − 0.22·stress + N(0,0.04)`, clip `[7.60, 8.50]` | Uniform − shift | **stress**, region | Derived |
| `salinity_ppt` | `U(s_min,s_max) + N(0,0.35)`, clip ±1.5 | Uniform | region only | **Independent of stress** |
| `dissolved_oxygen_mg_l` | `U(do_min,do_max) − 1.6·stress + 0.5·clarity + N(0,0.30)`, clip `[2,11]` | Uniform + 2 shifts | stress, clarity | Derived |
| `turbidity_ntu` | `exp(U(ln(t_min+.05), ln(t_max+.05))) + (1−clarity)·6.0 + N(0,0.4)`, clip `[0.05,32]` | Log-uniform + shift | clarity | Derived |
| `light_intensity` | `U(1200,2100)·exp(−k·depth) + N(0,12)`, `k = 0.04 + 0.02·turbidity`, clip `[5,2200]` | Beer–Lambert | depth, turbidity | Derived (physical) |
| `current_speed_m_s` | `U(c_min,c_max) + N(0,0.04)`, clip `[0.005,0.95]` | Uniform | region only | **Independent of stress** |
| `hard_substrate_percentage` | `structure·82 + N(0,7)`, clip `[0,100]` | Beta-scaled | **structure** | Derived |
| `rugosity_index` | `1 + structure·3.3 + N(0,0.28)`, clip `[1,4.8]` | Beta-scaled | **structure** | Derived |
| `sonar_backscatter` | `−35 + structure·30 + N(0,2)`, clip `[−36,−2]` | Beta-scaled | **structure** | Derived |
| `coral_cover_percentage` | `(1 − 0.72·stress)·78·(0.45 + 0.55·structure) + N(0,7.5)`, clip `[0,90]` | multiplicative | **stress × structure** | Derived |
| `acoustic_complexity_index` | `0.10 + N(0,0.05) + (cover/90)·0.62 + (1−stress)·0.18`, clip `[0.05,1]` | composite | **coral cover**, stress | Derived — *computed from a biological variable, not acoustically* |
| `bleaching_percentage` | `clip(max(0, T−28.5)·20 + stress·32, 0, 100) + N(0,8)`, clip `[0,100]` | composite | **temperature, stress** | Derived |
| `disease_percentage` | `stress·28 + max(0, 8.05−pH)·38 + max(0, 5.5−DO)·7 + N(0,5)`, clip `[0,60]` | composite | **stress, pH, DO** | Derived |

Only `salinity_ppt` and `current_speed_m_s` are independent of the stress latent — and both are near-useless predictors (§11), exactly as this structure implies.

Note `acoustic_complexity_index`: it is generated *from* `coral_cover_percentage`. In reality ACI is a hydrophone-derived soundscape metric that would be measured independently and might *correlate with* cover. Here the causal arrow is inverted and hard-coded, which is why the two correlate at +0.929.

### `reef_health` label formula — **VERIFIED, weights confirmed exactly as stated in the brief**

`_compute_health_score()`, `generate_data.py:456-474`:

```python
thermal_stress   = clip((temp - 28.5) / 4.5,   0, 1)
ph_stress        = clip((8.10 - ph)   / 0.55,  0, 1)
do_stress        = clip((6.00 - do)   / 3.50,  0, 1)
turbidity_stress = clip(turb / 20.0,           0, 1)
bleach_factor    = bl / 100.0
disease_factor   = clip(dis / 45.0,            0, 1)
low_cover_factor = clip(1.0 - cov / 72.0,      0, 1)

score = (0.20*thermal_stress + 0.12*ph_stress + 0.08*do_stress
       + 0.10*turbidity_stress + 0.25*bleach_factor
       + 0.12*disease_factor  + 0.13*low_cover_factor)
score = clip(score + N(0, noise_scale * 0.50), 0, 1)     # noise_scale = 0.15 → σ = 0.075
```

Thresholds `_HEALTH_THRESHOLDS = (0.28, 0.47, 0.66)`: `< 0.28` healthy · `[0.28, 0.47)` stressed · `[0.47, 0.66)` bleached · `≥ 0.66` severely_degraded.

Brief's claimed weights vs code — **all seven match exactly**:

| Component | Brief | Code | Match |
|---|---|---|---|
| `bleaching_percentage` | 0.25 | 0.25 | ✓ |
| `water_temperature_c` | 0.20 | 0.20 | ✓ |
| `coral_cover_percentage` | 0.13 | 0.13 | ✓ |
| `disease_percentage` | 0.12 | 0.12 | ✓ |
| `ph` | 0.12 | 0.12 | ✓ |
| `turbidity_ntu` | 0.10 | 0.10 | ✓ |
| `dissolved_oxygen_mg_l` | 0.08 | 0.08 | ✓ |

Weights sum to 1.00. **Confirmed from current source, not assumed.**

### `restoration_suitability` label formula — VERIFIED

`_compute_restoration_score()`, `generate_data.py:529-550`:

```python
temp_stability    = clip(1 - |temp - 28.0| / 6.5,          0, 1)   # 0.15
ph_acceptability  = clip((ph - 7.75) / 0.55,               0, 1)   # 0.12
light_adequacy    = clip(light / 500.0,                    0, 1)   # 0.10
substrate_quality = clip(hard / 100.0,                     0, 1)   # 0.22
current_quality   = clip(1 - |current - 0.18| / 0.45,      0, 1)   # 0.10
turbidity_clarity = clip(1 - turb / 18.0,                  0, 1)   # 0.12
depth_suitability = clip(1 - max(0, depth - 15.0) / 25.0,  0, 1)   # 0.09
coral_proximity   = clip(coral / 65.0,                     0, 1)   # 0.10
score = clip(Σ w_i·c_i + N(0, noise_scale * 0.45), 0, 1)           # σ = 0.0675
```

Thresholds `(0.40, 0.63)`: `< 0.40` unsuitable · `[0.40, 0.63)` moderately_suitable · `≥ 0.63` suitable. Weights sum to 1.00. Note this score is *higher = better*, the opposite polarity to the health score.

### The noise is too small to break determinism

| Target | Noise σ | Narrowest class band | σ / band |
|---|---|---|---|
| `reef_health` | 0.075 | 0.19 (stressed) | 0.39 |
| `restoration_suitability` | 0.0675 | 0.23 (moderately_suitable) | 0.29 |

The docstring claims noise ensures "no single input feature perfectly predicts either target." That narrow claim is true. But the *joint* function of the seven inputs predicts the target 82.7 % of the time, which is the property that actually matters — and the noise does not prevent it. Raising `noise_scale` would only degrade the labels toward randomness; it cannot make a formula over the features into an independent observation.

---

## 11. Target Leakage / Circular Supervision Findings

### Classification of every feature

| Leakage class | `reef_health` | `restoration_suitability` |
|---|---|---|
| **(1) Direct target leakage** — feature is an input to the label formula | `bleaching_percentage` (0.25), `water_temperature_c` (0.20), `coral_cover_percentage` (0.13), `disease_percentage` (0.12), `ph` (0.12), `turbidity_ntu` (0.10), `dissolved_oxygen_mg_l` (0.08) — **7 features, 100 % of label weight** | `hard_substrate_percentage` (0.22), `water_temperature_c` (0.15), `ph` (0.12), `turbidity_ntu` (0.12), `light_intensity` (0.10), `current_speed_m_s` (0.10), `coral_cover_percentage` (0.10), `depth_m` (0.09) — **8 features, 100 % of label weight** |
| **(2) Proxy leakage** — algebraically reproduces a label component | `thermal_stress_index` (**identical expression** to `thermal_stress`), `oxygen_stress_index` (**identical** to `do_stress`), `water_quality_index` (composite of 4 of 7 components), `acidity_deviation` (≈ `ph_stress` unsigned) | `substrate_stability_score` (0.6 × `substrate_quality` + rugosity) |
| **(3) Label-construction leakage / circular supervision** | **The entire dataset.** No independent observation of reef state exists. | Same. |
| **(4) Normal correlated predictors** | `sonar_backscatter`, `rugosity_index`, `acoustic_complexity_index`, `structural_complexity_score`, `salinity_ppt` (via shared latents only — none is a label input) | `sonar_backscatter`, `rugosity_index`, `acoustic_complexity_index`, `salinity_ppt`, `dissolved_oxygen_mg_l`, `bleaching_percentage`, `disease_percentage` |

### (2) Proxy leakage — exact algebraic identity, severity HIGH

Two engineered features are **character-for-character** the same expression as label-score components:

| `build_features.py` | `generate_data.py::_compute_health_score` |
|---|---|
| `thermal_stress_index = clip((water_temperature_c − 28.5)/4.5, 0, 1)` | `thermal_stress = clip((temp − 28.5)/4.5, 0, 1)` — weight 0.20 |
| `oxygen_stress_index = clip((6.0 − dissolved_oxygen_mg_l)/3.5, 0, 1)` | `do_stress = clip((6.00 − do)/3.50, 0, 1)` — weight 0.08 |

Same constants (28.5, 4.5, 6.0, 3.5), same clip. These are not approximations of the label components — they *are* the label components, handed to the model pre-computed.

`water_quality_index` is worse:
```
WQI = 1 − clip(0.40·thermal_stress_index + 0.30·oxygen_stress_index
             + 0.20·clip(|ph−8.1|/0.55,0,1) + 0.10·clip(turbidity/20,0,1), 0, 1)
```
Its four components correspond to health-score components carrying combined weight 0.20 + 0.08 + 0.12 + 0.10 = **0.50**. It is a re-weighted, sign-flipped half of the label. Its measured MI with `reef_health` is 0.621 nats = **55.7 % of the target's total entropy**, and it is the strongest single predictor of health (macro-F1 0.700 alone). That is not a coincidence — it is the arithmetic showing through.

`build_features.py` documents this honestly in `SYNTHETIC_LEAKAGE_NOTES`, which is genuinely good practice. But documenting leakage does not remove it from the training matrix, and every downstream metric is computed with these columns present.

### (3) Circular supervision — the decisive test

Method: re-implement `_compute_health_score` and `_compute_restoration_score` from the seven/eight supplied predictor columns, **omit the Gaussian noise term**, re-apply the published thresholds, compare to the stored labels. No model, no fitting, no training data.

| Metric | `reef_health` | `restoration_suitability` |
|---|---|---|
| Exact label match, closed form, zero learning | **82.72 %** | **84.39 %** |
| Macro-F1 of closed form vs stored labels | **0.770** | **0.812** |
| Within one ordinal class | **100.0 %** | — |
| Spearman ρ (noiseless score vs ordinal class) | 0.843 | 0.796 |
| **Trained GBM, all 21 features, 5-fold CV — accuracy** | 0.8177 | 0.8302 |
| **Trained GBM — macro-F1** | 0.7603 | 0.7911 |

The trained model does **not beat arithmetic** on health (0.760 vs 0.770 macro-F1) and barely differs on restoration. The residual 17 % is precisely the injected label noise, which is irreducible by construction. The model has converged to the Bayes-optimal solution of a problem whose answer was written down in `generate_data.py`.

### Ablation — the corroboration

| Feature set | health macro-F1 | restoration macro-F1 |
|---|---|---|
| All 21 numeric (raw + engineered) | 0.7603 | 0.7911 |
| Raw 15 only | 0.7576 | 0.7933 |
| **Label-generating features only** (7 / 8) | **0.7587** | **0.7889** |
| **Non-label-generating features only** (8 / 7) | **0.4704** | 0.6889 |
| Sensor-only, no biological (12) | 0.7352 | 0.7936 |
| Engineered 6 only | 0.7384 | 0.7152 |
| Biological only — cover/bleach/disease (3) | 0.7253 | 0.6365 |

Read the health column: the 7 label inputs alone give 0.7587; adding 14 more features gains **+0.0016**. Remove them and performance collapses to 0.4704, barely above the 0.5754 majority baseline. **All predictive signal lives in, and only in, the columns used to write the label.**

The restoration column tells a subtler story: `sensor_only_no_biological` (0.7936) actually *ties* the full set, because 6 of the 8 restoration-label inputs are sensor variables. Restoration is therefore the more salvageable of the two tasks from a deployment standpoint — though still fully circular.

### Verdict

**The model is learning a deterministic-plus-noise scoring function, not an independently observed biological phenomenon. Severity: CRITICAL.**

Reported metrics (macro-F1 ≈ 0.76 / 0.79, quality gates at 0.70 / 0.73) measure how well a gradient-booster can invert a weighted sum through a small noise floor. They carry **no information** about whether the system can assess a real reef. Every model card, dashboard performance page, and drift report currently states these numbers without this qualification.

To be fair to the project: `build_features.py` already documents the leakage, `get_feature_columns()` correctly blocks cross-target leakage, and the generator docstring carries an explicit synthetic-data disclaimer. The design flaw is architectural, not careless.

---

## 12. Train/Test Split Assessment

### Verification of documented behaviour — confirmed

`preprocess.py::split_and_preprocess` (lines 246-252):
```python
train_test_split(X, y, test_size=cfg.test_size,      # 0.20
                 random_state=cfg.random_seed,       # 42
                 stratify=y if cfg.stratify else None)  # stratify: true
```
15,000 × 0.80 = **12,000 train / 3,000 test**, stratified on the target, seed 42 — exactly as documented. Split runs **independently per task**, so health and restoration have different row assignments. The preprocessor is fitted on the training split only (line 263), which correctly prevents scaling leakage. Raw pre-transformer splits are persisted so each CV fold refits the transformer — also correct.

The mechanics are right. The **design** is not.

### Is random stratification appropriate? — **No. Severity HIGH.**

| Split strategy | health macro-F1 | health bal-acc | restoration macro-F1 | restoration bal-acc |
|---|---|---|---|---|
| **Random stratified 5-fold** (current) | **0.7632** | 0.7580 | **0.7911** | 0.7829 |
| Leave-one-region-out: Andaman (n=6,300) | **0.4107** | 0.3962 | **0.4793** | 0.4571 |
| Leave-one-region-out: Lakshadweep (n=3,450) | 0.4804 | 0.4600 | 0.4975 | 0.4822 |
| Leave-one-region-out: Gulf of Kutch (n=2,250) | 0.6410 | 0.6488 | 0.5968 | 0.5937 |
| Leave-one-region-out: Gulf of Mannar (n=3,000) | 0.6825 | 0.6664 | 0.6308 | 0.6084 |
| Temporal hold-out (train ≤2022, test 2023–24) | 0.7529 | 0.7446 | 0.7929 | 0.7808 |

**Geographic leakage confirmed.** Random CV reports 0.763; the worst region hold-out gives 0.411. The current split **overstates generalisation to a new reef system by up to 0.35 macro-F1**. Mechanism: random splitting places rows from the same region — sharing the same `_REGION_STRESS`/`_REGION_STRUCTURE` Beta parameters — on both sides of the split, so the model memorises region-level priors and is tested on rows drawn from the identical distribution. Andaman degrades worst because it is 42 % of the data and holds only 1 severely-degraded and 13 unsuitable rows; training without it means never seeing its distribution, and testing on it means predicting a near-single-class population.

**No temporal leakage** — the temporal hold-out matches random CV (0.753 vs 0.763). But this is a *null* result, not a pass: §6 established that time explains η² < 0.001 of every variable, so there is nothing for a temporal split to detect. On a real dataset with seasonal cycles and bleaching years, this test would be highly informative; here it is vacuous.

**Spatial autocorrelation is absent within regions** (§5), so spatial-block CV would currently behave like random CV. The problem is entirely between-region.

### Alternative designs — assessed, not implemented

| | Design | What it tests | Strength on current data | Strength on real data | Cost |
|---|---|---|---|---|---|
| **A** | Region hold-out (leave-one-region-out) | transfer to an unseen reef system | **High** — already exposes a 0.35 F1 gap | High | Low — 4 folds |
| **B** | Spatial-block CV (grid/cluster blocks) | transfer across space at sub-region scale | **Low** — no within-region spatial structure exists | High | Medium |
| **C** | Temporal hold-out (train past, test future) | forecasting under environmental change | **None** — no temporal signal | High | Low |
| **D** | Region + temporal hold-out (unseen region *and* unseen period) | joint spatial-temporal transfer — closest to real deployment | Medium (inherits A only) | **Highest** | Medium |
| **E** | External validation on an independent real dataset | whether anything transfers to reality at all | **N/A — currently impossible** | **Decisive** | High |

**Recommendation.** For the current synthetic dataset, adopt **A (region hold-out)** immediately — it costs almost nothing and is the only design that reveals a real weakness in the present data. Report both numbers side by side (random CV *and* LORO) so the gap is visible rather than hidden.

For the eventual real-data track, **E is the only strategy that constitutes genuine evidence of generalisation**, with **D** as the strongest internal proxy. No amount of clever internal splitting can validate a model whose labels were authored by the same script that authored its features — external data breaks that circle, and nothing else does. Until E exists, every reported metric should be labelled *in-distribution synthetic performance*.

---

## 13. Deployment Feature Availability

### Verified required inputs

`src/api/schemas.py::ObservationInput` — docstring states "All 16 inference features are required." `timestamp`, `latitude`, `longitude` are `| None = None` (optional metadata). The 16 required fields match `params.yaml::features.numeric` (15) + `region` exactly, and match `ALL_FEATURE_COLUMNS`. The 6 engineered features are computed server-side by `add_derived_features()`, so the caller supplies 16 values. **Confirmed from current source.**

### Acquisition classification

| Feature | Class | Realistic acquisition | Available from proposed sonar + sensor rig? |
|---|---|---|---|
| `region` | **C** | GPS + lookup table | **Yes** — trivial |
| `depth_m` | **A** | Single-beam echosounder | **Yes** — the easiest sonar product |
| `water_temperature_c` | **B** | Thermistor / CTD (~$50–300) | **Yes** |
| `ph` | **B** | pH probe (~$200–2 000; needs frequent calibration, drifts) | **Yes**, with maintenance burden |
| `salinity_ppt` | **B** | Conductivity cell / CTD | **Yes** |
| `dissolved_oxygen_mg_l` | **B** | Optical DO optode (~$500–2 000) | **Yes** |
| `turbidity_ntu` | **B** | Nephelometer (~$300–1 500) | **Yes** |
| `light_intensity` | **B** / D | PAR quantum sensor at depth; or modelled from surface PAR + depth + k_d | **Yes**, but *at-substrate* PAR needs a lowered sensor, not a hull mount |
| `current_speed_m_s` | **B** | ADCP (~$5–20 k) or tilt-current meter (~$500) | **Yes**, cost-dependent |
| `sonar_backscatter` | **A** | Calibrated single/multibeam backscatter (dB) | **Yes — but requires calibration.** Uncalibrated echosounders give relative, not absolute, dB. Non-trivial. |
| `rugosity_index` | **A*** | Multibeam DEM → surface-to-planar area ratio | **Partially.** Needs MBES (~$30 k+) plus a bathymetric-processing chain the project does not have. Not obtainable from a single-beam sounder. |
| `hard_substrate_percentage` | **A*** | Acoustic seabed classification (QTC/RoxAnn-style) from backscatter | **Partially.** Requires a *trained and ground-truthed* classifier — itself a supervised problem needing diver/grab validation. Circular unless solved first. |
| `acoustic_complexity_index` | **H*** | **Passive** hydrophone recording + soundscape ACI computation | **No.** ACI is passive-acoustic; a sonar is an *active* instrument. Different sensor, different deployment (long moored recordings), different processing. Mislabelled as a sonar product. |
| **`coral_cover_percentage`** | **E / F** | Diver line-intercept/photo-quadrat, or downward camera + CoralNet-style point classification | **NO** |
| **`bleaching_percentage`** | **E / F** | Diver visual assessment or colour-calibrated imagery (CoralWatch chart, spectral analysis) | **NO** |
| **`disease_percentage`** | **E** | Diver colony-level visual diagnosis; often needs histology/microbiology to confirm | **NO** |

### The critical distinction — severity CRITICAL

**Predictors available at deployment (13 of 16):** region, depth, temperature, pH, salinity, DO, turbidity, light, current, sonar backscatter, and — with additional engineering investment — rugosity, hard substrate, ACI.

**Biological ground-truth variables available only during surveys (3 of 16):** `coral_cover_percentage`, `bleaching_percentage`, `disease_percentage`.

These three are the *output* of the survey the system is supposed to replace. Requiring them as *inputs* inverts the value proposition: if a diver has already been down to count bleached colonies, the reef-health class is known and no model is needed.

They are also, by the leakage analysis, the most heavily weighted label inputs — bleaching alone carries 0.25 of the health score, and `biological_only` (just these three) reaches macro-F1 0.7253 versus 0.7603 for all 21 features. **95 % of the model's apparent skill on reef health comes from three variables that cannot be measured at inference time.**

Quantified: dropping the three biological variables costs only 0.025 macro-F1 on health (0.7603 → 0.7352) and *nothing* on restoration (0.7911 → 0.7936) — but that is a property of this synthetic dataset, where all variables descend from the same two Beta latents. On real data the drop would be far larger, because the sensor variables would no longer be deterministic functions of the same hidden state.

**Recommended reframing (do not implement yet):**

- **Deployable model** — inputs restricted to classes A + B + C (13 features, ideally 10 without the specialist-processing three). This is the real product.
- **Biological variables** — reclassified as ground-truth/targets, or as auxiliary supervision available at training time only, never at inference.
- **`restoration_suitability` is the better near-term target**: 6 of its 8 label inputs are already sensor-obtainable, and sensor-only performance matches full-feature performance exactly.

---

## 14. External Dataset Inventory

### `data/audit_inbox/` — 5 files, all PDFs, **zero datasets**

Recursive inventory: 5 files, 30,804,751 bytes total, no subdirectories, no hidden files. No `.csv`, `.tsv`, `.json`, `.jsonl`, `.parquet`, `.xlsx`, `.zip`, `.tar`, `.gz`, `.geojson`, `.shp`, `.gpkg`, `.nc`, `.h5`, `.tif` — verified by extension sweep across the entire repository (excluding `.venv`/`.git`), which returned nothing.

| # | File | Size | SHA-256 (first 16) | Identification |
|---|---|---|---|---|
| 1 | `1-s2.0-S2352340916304607-main-1.pdf` | 4,087,194 | `50a5743223882047` | Caldwell et al. (2016), *Data in Brief* 8:1054–1058 — **HICORDIS data descriptor** |
| 2 | `A-coral-disease-outbreak-…_2023_iSci.pdf` | 4,075,280 | `c2aca65860ede0c4` | Page et al. (2023), *iScience* 26:106205 — coral disease outbreak, Norfolk Island |
| 3 | `Coral-black-band-disease-in-Indonesia…_2024_…pdf` | 3,591,626 | `1525264d068e561b` | Pribawastuti et al. (2024), *Egypt. J. Aquat. Res.* 50:103–109 — BBD review |
| 4 | `Fish-metacommunity-dynamics…_2026_…pdf` | 4,533,473 | `488cf8207e5ff629` | Oliveira-Silva et al. (2026), *J. Nature Conservation* 89:127112 — Brazilian freshwater fish |
| 5 | `Pathology-of-green-sea-turtle…_2026_…pdf` | 14,517,178 | `1aa672bd3569b664` | Hannon et al. (2026), *J. Comp. Pathology* 227:4–13 — green turtle mortality, Gladstone Harbour |

#### Individual assessments

**#1 — HICORDIS descriptor (the valuable one).** Describes the Hawai'i Coral Disease Database: 286,071 coral colonies, 1,819 transects, 660 sites, 17 Hawaiian islands/atolls, 2005–2015, 60 species / 22 genera, 21 health conditions. Acquisition: underwater visual surveys on SCUBA and snorkel. Variables per colony: species ID, colony measurements, health condition, **GPS coordinates and depth**. Licence: **CC BY 4.0** (open access). Data availability: "within this article (Table S1)" — i.e. supplementary material of the DOI, not yet in this repository. Real, not synthetic.

This is a *pointer to a genuinely relevant dataset*, and the most useful item in the inbox. Compatibility: provides real observed `disease_percentage`-analogue (colony-level prevalence, aggregable to transect), partial `coral_cover` (species composition), latitude/longitude/depth. Provides **no** sonar, temperature, pH, salinity, DO, turbidity, or restoration variables. Geography is Hawai'i, not India — usable as external validation for a *disease* model, not for Indian reef transfer.
Classification: **EXTERNAL VALIDATION DATA** (disease sub-problem) / **REFERENCE DATA**.

**#2 — Page et al. 2023.** Single-site disease outbreak study, Norfolk Island lagoon (high-latitude, SW Pacific). Reports 60 % community disease prevalence linked to 2020 heat stress and pollution. Narrative/figures only; no tabular dataset in the PDF. Value: **methodological** — an empirical anchor for what real disease prevalence and its environmental drivers look like, directly relevant to recalibrating the generator's `disease_percentage` distribution (currently mean 15.3 %).
Classification: **REFERENCE DATA (literature)**.

**#3 — Pribawastuti et al. 2024.** Review of black-band disease in Indonesia. No primary dataset. Value: domain background on disease–water-quality links.
Classification: **REFERENCE DATA (literature)**.

**#4 — Oliveira-Silva et al. 2026.** Freshwater fish metacommunities, Mamanguape River, semi-arid NE Brazil. **No relevance whatsoever** to coral reefs, sonar, or either target. Different realm (freshwater), different continent, different taxa.
Classification: **UNSUITABLE FOR CURRENT TARGETS**.

**#5 — Hannon et al. 2026.** Green sea turtle necropsy pathology, Gladstone Harbour, Queensland. Marine, and reef-adjacent geographically, but the observational unit is an individual turtle carcass. No reef-condition, sonar, or environmental variables usable for either target.
Classification: **UNSUITABLE FOR CURRENT TARGETS** (marginal **BIODIVERSITY CONTEXT**).

#### Inbox verdict — severity INFO (with a caution)

`data/audit_inbox/` contains **literature, not data**. Nothing here is loadable, joinable, or trainable. Two of the five are off-topic (freshwater fish, turtle pathology) and appear to have been collected by keyword rather than by relevance.

**Caution:** all five are Elsevier-published PDFs. Items 2–5 are almost certainly **subscription-access, all-rights-reserved** (item 1 is explicitly CC BY). They must **not** be committed to this Git repository — `data/audit_inbox/` is currently untracked, which is correct and should stay that way. Add it to `.gitignore` (proposal only; not done). See §21.

### The IUCN archive — not where the brief expected it

The brief anticipated `redlist_species_data_938f7263-9446-4750-a806-d46d4133a98a.zip` under `data/audit_inbox/`. **It is not in the repository.** Located at:

```
/home/BAAHbun/Downloads/redlist_species_data_938f7263-9446-4750-a806-d46d4133a98a.zip
sha256: b188920c03e182acad2c3592b4e1c73d510b75b894027a95fb413be67d52b5ac
size:   4,830,579 bytes (listed before extraction; original archive unmodified)
```
Extracted read-only to `/tmp/oceanographic-dataset-audit/iucn/`. Full assessment in §15.

---

## 15. IUCN Dataset Assessment

### Archive contents (listed before extraction, per instruction)

| File | Bytes | Purpose |
|---|---|---|
| `data_0.shp` | 4,198,600 | Polygon geometry |
| `data_0.dbf` | 1,157 | Attribute table |
| `data_0.shx` | 108 | Shape index — **(108−100)/8 = 1 record** |
| `data_0.prj` | 256 | CRS definition |
| `data_0.cpg` | 5 | Encoding: `UTF-8` |
| `ReadMe.txt` | 5,114 | IUCN standard readme |
| `METADATA for Digital Distribution Maps…pdf` | 510,341 | Field definitions |
| `IUCN Red List_Terms and Conditions of Use_v3.pdf` | 114,998 | **Licence** |

### Verified contents — **brief's expectation confirmed**

**Species: _Catalaphyllia jardinei_** (Elegance coral, Euphylliidae) — verified by direct DBF parse, not assumed.

**Record count: 1.** A single multipart polygon feature.

| DBF field | Value |
|---|---|
| `ASSESSMENT` | 165616621 |
| `ID_NO` | 132890 |
| `SCI_NAME` | **Catalaphyllia jardinei** |
| `PRESENCE` | 1 (Extant) |
| `ORIGIN` | 1 (Native) |
| `SEASONAL` | 1 (Resident) |
| `LEGEND` | Extant (resident) |
| `COMPILER` | IUCN Marine Biodiversity Unit |
| `YRCOMPILED` | 2024 |
| `CITATION` | IUCN Marine Biodiversity Unit |
| `SUBSPECIES` / `SUBPOP` / `DIST_COMM` / `ISLAND` / `TAX_COMM` | all empty |

**Geometry:** Shapefile type 5 (Polygon). **5,356 parts**, **261,064 vertices** — a multipart polygon of many island/coastal range fragments. Extent: longitude −179.999 to 179.999 (dateline-crossing), latitude −30.124 to +35.687.

**CRS:** `GEOGCS["WGS 84", … AUTHORITY["EPSG","4326"]]` — **EPSG:4326, WGS-84 geographic degrees.** Matches the project's coordinate convention exactly.

**Temporal coverage:** None. Compiled 2024; the polygon is a static range assessment, not a time series.

**Observational unit:** species range extent. Not a survey, not a measurement, not an observation of any site.

### Does it contain any variable we need? — **No. Verified field by field.**

| Requested variable | Present? |
|---|---|
| sonar / acoustic | **No** |
| temperature | **No** |
| salinity | **No** |
| pH | **No** |
| dissolved oxygen | **No** |
| turbidity | **No** |
| coral cover | **No** |
| bleaching | **No** |
| disease | **No** |
| reef health | **No** |
| restoration suitability | **No** |

All 15 DBF fields are taxonomic/administrative metadata. There are **zero** measured environmental, acoustic, or biological-condition attributes. The file answers one question — "where does this species occur?" — and nothing else.

### Geographic overlap with the project — **ZERO. This is the decisive finding.**

The bounding box overlaps all four project regions, but a bounding box spanning the entire Indo-Pacific is meaningless. I ran an even-odd ray-casting point-in-polygon test against all 5,356 rings.

Validation of the test first (known *C. jardinei* habitat vs known non-habitat):

| Probe point | Expected | Result |
|---|---|---|
| Java Sea, Indonesia (115 °E, 5 °S) | inside | **inside** ✓ |
| Philippines (121 °E, 12 °N) | inside | **inside** ✓ |
| Great Barrier Reef (146.5 °E, 18.5 °S) | inside | **inside** ✓ |
| Solomon Islands (159 °E, 9 °S) | inside | **inside** ✓ |
| Mid-Pacific open ocean (140 °W, 0 °N) | outside | **outside** ✓ |
| Sahara desert (10 °E, 25 °N) | outside | **outside** ✓ |

Test validated. Now the project regions:

| Project region | Centroid inside range? | bbox corners inside (of 4) | Polygon vertices inside region bbox | Min distance, centroid → range |
|---|---|---|---|---|
| Lakshadweep | **No** | 0 / 4 | **0** | **388 km** |
| Gulf of Mannar | **No** | 0 / 4 | **0** | **622 km** |
| Andaman and Nicobar Islands | **No** | 0 / 4 | **0** | **674 km** |
| Gulf of Kutch | **No** | 0 / 4 | **0** | **1,758 km** |

And against the actual data: of 600 systematically sampled observation coordinates from `observations.csv`, **0 (0.0 %)** fall inside the species range.

*C. jardinei*'s documented range is the Indo-West Pacific from the eastern Indian Ocean through the Coral Triangle to Australia and Vanuatu. It does not extend to the Indian subcontinent's reef systems. The overlap with this project's study area is **exactly zero**, verified geometrically.

### Licensing — severity HIGH (latent)

From the IUCN Red List Terms and Conditions (v3, bundled in the archive) and the current published terms:

- Free for **non-commercial** use only. "Commercial Use" includes use by or on behalf of a for-profit entity, *and* use by any individual or non-profit for revenue generation.
- **Redistribution is strictly prohibited without prior written permission from IUCN** — explicitly including web downloads, APIs, interactive web maps granting download access, KML, FTP, digital storage, "or any other electronic media or device."
- The prohibition explicitly covers redistribution "alone or combined with other data, **including within Derivative Works**."
- Citation of the assessment and the Red List is required.

**Practical consequences:**

1. **Do not commit this archive, its extracted contents, or the polygon geometry to this repository.** The repo is MIT-licensed with a public GitHub remote — committing would be redistribution to an unlimited third-party audience. Currently it lives in `~/Downloads`, outside the repo. Keep it there.
2. Do not serve the polygon through the FastAPI service or make it downloadable from the Streamlit dashboard — an interactive map granting download access is named as prohibited redistribution.
3. Rendering a **derived, non-downloadable** cartographic layer for internal/academic display, with full IUCN attribution, is the most that is defensible — and even that warrants checking against the bundled terms PDF before implementation.
4. Since geographic overlap is zero, there is **no scientific reason to take that risk at all**.

### Classification

**REFERENCE DATA / BIODIVERSITY CONTEXT — and, for this project's four regions, effectively UNSUITABLE.**

It is authentic, authoritative, well-documented, correctly projected data. It is simply about a different question (species range) in a different place (Indo-West Pacific, not India). It contains no variable this project models, provides no ground truth for either target, and cannot be joined to any observation in `observations.csv` because no observation falls within it.

**Recommendation: exclude entirely.** If species-range context for *Indian* reefs is genuinely wanted later, request the appropriate species set from IUCN with correct geographic filtering — and even then keep it outside version control.

---

## 16. Recommended Public Datasets

Research conducted 2026-08-19. Authoritative sources only. **Nothing was downloaded.**

### 1. NOAA Coral Reef Watch — Daily Global 5 km Bleaching Heat Stress Suite v3.1

- **URL:** https://coralreefwatch.noaa.gov/product/5km/ · NCEI archive: https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.nodc%3ACRW-5km-HeatStressProducts
- **Product:** CoralTemp SST, SST Anomaly, Coral Bleaching HotSpot, **Degree Heating Week (DHW)**, Bleaching Alert Area (BAA), 7-day max BAA, 7-day SST trend
- **Availability:** Operational, near-real-time; last updated 2026-08-01. Version 3.1 since 2018-08-01.
- **Type:** Satellite-derived gridded raster (NetCDF), daily
- **Spatial resolution:** 0.05° (~5 km) · **Temporal:** daily, 1985–present · **Coverage:** global
- **Variables:** SST, SSTA, HotSpot, DHW (°C-weeks), BAA (0–4). DHW ≥ 4 = bleaching likely; ≥ 8 = reef-wide bleaching with mortality; ≥ 12 = multi-species mortality.
- **Licence:** US Government work — public domain, free and open. Citation requested.
- **Access:** HTTPS/FTP from CoralReefWatch, NCEI THREDDS/OPeNDAP, ERDDAP; also PacIOOS. Regional Virtual Stations exist **for Lakshadweep** (https://coralreefwatch.noaa.gov/product/vs/gauges/lakshadweep.php).
- **Size:** ~1–3 MB/day global NetCDF; an Indian-Ocean subset 2018–2026 is a few GB.
- **Relevance:** **Highest of any source.** Directly supplies the thermal-stress variable currently *fabricated* by `thermal_push = stress × 0.40 × t_range`.
- **Role:** **Primary real environmental covariate + the physically correct thermal-stress feature.** Replaces `water_temperature_c` and `thermal_stress_index` with observations. Enables real seasonality and real bleaching-event years, fixing the §6 finding.

### 2. Allen Coral Atlas — Global Benthic & Geomorphic Habitat Maps

- **URL:** https://allencoralatlas.org/ · methods: https://allencoralatlas.org/methods/
- **Product:** Geomorphic zonation (~12 classes) and benthic composition (hard coral, soft coral, macroalgae, seagrass, sand, rubble, rock); plus reef extent
- **Availability:** Live; 2022 map version current
- **Type:** Satellite-derived (Planet Dove) classified raster / vector polygons
- **Resolution:** **5 m pixels** · **Temporal:** static maps, 2022 baseline (a separate near-real-time bleaching monitoring layer exists) · **Coverage:** 30 °N – 30 °S — **includes all four project regions**
- **Variables:** benthic class (to ~10 m depth), geomorphic zone (to ~15 m depth), reef extent
- **Licence:** **Requires verification before use.** Site carries "© 2026 Arizona State University, all rights reserved"; the methods page does not state a licence name. Historically CC BY 4.0 for map downloads, but **confirm current terms on the download page before ingesting.** Registration required.
- **Access:** Interactive download by area from the Atlas web app; GIS-ready formats
- **Relevance:** Very high — supplies real `hard_substrate_percentage` and habitat context.
- **Role:** **Habitat map / geomorphology layer.** Real substrate composition to replace the `structure`-Beta-derived `hard_substrate_percentage`, and a **reef mask** to fix the §5 problem of coordinates over open ocean and land.

### 3. GEBCO_2025 Grid (IHO/IOC, Seabed 2030)

- **URL:** https://www.gebco.net/data-products-gridded-bathymetry-data/gebco2025-grid · download app: https://download.gebco.net
- **Product:** Global bathymetric/topographic terrain model
- **Availability:** Released 2025; annual releases
- **Type:** Gridded elevation raster (NetCDF, GeoTIFF, Esri ASCII)
- **Resolution:** **15 arc-second** (~450 m at equator), 43,200 × 86,400 = 3.73 × 10⁹ cells · **Temporal:** static · **Coverage:** global ocean + land
- **Variables:** elevation/depth (m). A Type Identifier grid indicates data provenance per cell.
- **Licence:** **Public domain, free public use.** Attribution requested.
- **Access:** GEBCO download app (user-defined areas or 8 × 90°×90° tiles), NERC CEDA direct download
- **Size:** ~7.5 GB global NetCDF; an Indian-reef-region subset is tens of MB.
- **Relevance:** High — supplies real `depth_m`, currently `rng.uniform(0.5, 38)`.
- **Role:** **Bathymetry / geomorphology layer.** Real depth per coordinate; derived slope and terrain-ruggedness index give a physically grounded proxy for `rugosity_index`. Caveat: 450 m resolution is far coarser than reef rugosity scale — good for regional context, insufficient for colony-scale structure.

### 4. AIMS Long-Term Monitoring Program (Great Barrier Reef)

- **URL:** https://www.aims.gov.au/research-topics/monitoring-and-discovery/monitoring-great-barrier-reef/long-term-monitoring-program · data: https://eatlas.org.au/gbr/ltmp-data · https://apps.aims.gov.au/reef-monitoring/
- **Product:** Manta-tow surveys (COTS + benthos) and fixed-site photo-transect surveys; 40+ years
- **Availability:** Live, updated annually; 2024/25 condition report published
- **Type:** Tabular survey records
- **Resolution:** reef- and site-level; manta tow ≈ 2,000 m² per 2-minute tow · **Temporal:** annual, ~1985–present · **Coverage:** Great Barrier Reef (Australia)
- **Variables:** **percent hard coral cover** (manta tow in ordinal bands: 0, >0–10, >10–30, >30–50, >50–75, >75–100 %), soft coral, algae, COTS counts; fixed-site transects give finer percent cover by benthic category
- **Licence:** CC BY (per AIMS/eAtlas data policy) — confirm the per-download readme, which states conditions of use
- **Access:** eAtlas portal download, AIMS reef-monitoring web app
- **Relevance:** High for method, **low for geographic transfer** (GBR ≠ Indian Ocean)
- **Role:** **Biological ground-truth survey exemplar and external validation.** The gold standard for what a real `coral_cover_percentage` distribution looks like — use it to recalibrate the generator and to test whether a model trained elsewhere transfers. Note the ordinal banding: manta-tow cover is **not** a continuous percentage, which has direct implications for how the target should be defined.

### 5. NOAA NCEI Water-Column Sonar Data Archive

- **URL:** https://www.ncei.noaa.gov/products/water-column-sonar-data · AWS: https://registry.opendata.aws/ncei-wcsd-archive/ · docs: https://cires.gitbook.io/ncei-wcsd-archive
- **Product:** Raw and processed water-column active-acoustic data (Simrad EK60/EK80, 18–710 kHz)
- **Availability:** Live, continuously growing
- **Type:** Raw sonar `.raw` files; cloud-native Zarr for processed products
- **Resolution:** ping-level, sub-metre vertical · **Temporal:** per-cruise, 2000s–present · **Coverage:** global, following NOAA vessel tracks
- **Variables:** volume backscattering strength Sv, target strength TS, calibration parameters, navigation
- **Licence:** **Public domain** (US Government, NOAA Open Data Dissemination)
- **Access:** Anonymous S3 bucket `noaa-wcsd-pds` (boto3); structure ship → cruise → instrument → file. Process with `echopype`.
- **Size:** Archive is many TB; individual cruises GB-scale.
- **Relevance:** **The only authoritative source of real calibrated acoustic backscatter available to this project.** Critical caveat: this is *water-column* (pelagic, fisheries-oriented) sonar, not *seabed-classification* backscatter. It answers "what is in the water," not "what is the bottom made of." Reef-substrate work needs multibeam seabed backscatter — a related but distinct product.
- **Role:** **Acoustic-method development data.** Learn real Sv/TS magnitudes, calibration handling, and file formats; **not** a drop-in replacement for `sonar_backscatter` as currently defined.

### 6. CoralNet (and ReefNet)

- **URL:** https://coralnet.ucsd.edu/ · ReefNet: https://huggingface.co/datasets/ReefNet/ReefNet-1.0 · toolbox: https://jordan-pierce.github.io/CoralNet-Toolbox/
- **Product:** Benthic image repository with point-count annotations and trained classifiers
- **Availability:** Live; free and open source (NSF/NOAA supported)
- **Type:** Images + point-label annotations
- **Resolution:** photo-quadrat scale · **Temporal:** varies by source, 2010s–present · **Coverage:** global, several hundred public sources
- **Variables:** benthic point labels (coral genus/species, algae, substrate) → **derived percent cover**; **bleaching classifiers exist** (NOAA InPort: "semi-automated CoralNet Bleaching Classifier")
- **Licence:** Per-source; CoralNet itself is open, but individual sources carry their own terms — **check each source before use.** ReefNet aggregates 76 curated CoralNet sources (~925,000 genus-level hard-coral annotations, WoRMS-mapped).
- **Access:** Web UI; community download scripts; ReefNet via Hugging Face
- **Relevance:** High — the realistic path to obtaining `coral_cover_percentage` and `bleaching_percentage` *without a diver*, which is precisely the §13 blocker.
- **Role:** **Computer-vision data.** Train a downward-camera → benthic-cover model, converting two currently-unavailable inputs into camera-derived measurements. This is the single highest-leverage change to the deployment story.

### 7. GCRMN — `gcrmndb_benthos` / Status of Coral Reefs of the World

- **URL:** https://gcrmn.net/ · https://icriforum.org/ · code/data: https://github.com/JWicquart
- **Product:** Harmonised global benthic-cover monitoring database underpinning the GCRMN status reports
- **Availability:** 2020 report (first global quantitative assessment) and 2025 report published; database access via GCRMN/ICRI
- **Type:** Tabular harmonised survey records
- **Resolution:** site/transect · **Temporal:** 1978–2019 in the 2020 compilation, extended since · **Coverage:** global — **~2 million observations, 12,000+ sites, 300+ contributors**; includes the South Asia region
- **Variables:** benthic cover by category (hard coral, algae, etc.), site coordinates, date, method
- **Licence:** Varies by contributing dataset; access typically requires request/agreement — **not a straightforward open download**
- **Relevance:** **Highest of any biological source for this project**, because it is global and includes Indian Ocean sites — the only realistic path to real `coral_cover_percentage` *for the actual study regions*.
- **Role:** **Primary biological ground truth / training labels** for a real-data track. Pursue access early; lead time is likely months.

### 8. Reef Life Survey

- **URL:** https://reeflifesurvey.com/survey-data/ · AODN portal (search "RLS") · GBIF: https://www.gbif.org/dataset/38f06820-08c5-42b2-94f6-47cc3e83a54a
- **Product:** Standardised global reef fish, invertebrate, and habitat-quadrat surveys by trained divers
- **Availability:** Live; housed in the AODN portal
- **Type:** Tabular + photo quadrats
- **Resolution:** 50 m transects (50 × 5 m swathes) · **Temporal:** ~2006–present · **Coverage:** global shallow rocky and coral reefs
- **Variables:** fish/invertebrate species abundance and size, **habitat quadrat percent cover**, downloadable photo quadrats
- **Licence:** AODN data are generally CC BY; **confirm per layer** (contact reeflife.survey@utas.edu.au)
- **Access:** AODN portal layers; GBIF for occurrences
- **Relevance:** Medium-high — real benthic cover plus a biodiversity dimension the project does not yet model
- **Role:** **Biological ground truth + external validation + photo-quadrat CV training data.**

### 9. OBIS — Ocean Biodiversity Information System

- **URL:** https://obis.org/data/access/ · manual: https://manual.obis.org/access.html · AWS: https://registry.opendata.aws/obis/
- **Product:** Integrated global marine species occurrence records
- **Availability:** Live
- **Type:** Occurrence records (Darwin Core); GeoParquet and TSV
- **Resolution:** point occurrences · **Temporal:** historical–present · **Coverage:** global
- **Variables:** taxon, coordinates, date, depth, dataset provenance; **no environmental measurements of reef condition**
- **Licence:** **Source datasets are CC0 / CC BY / CC BY-NC; the integrated OBIS dataset as a whole is CC BY-NC.** Per-dataset licences at `s3://obis-open-data/licenses.tsv`. **The CC BY-NC on the aggregate is a real constraint** — check it before any commercial or revenue-generating use.
- **Access:** OBIS API, `robis` R package, GeoParquet from `s3://obis-open-data/occurrence`
- **Relevance:** Medium — biodiversity context, coral species presence for Indian regions
- **Role:** **Biodiversity context.** Confirms which coral taxa actually occur at a site; not a source of reef-condition labels.

### 10. UNESCO / NOAA CoRTAD v6 — Coral Reef Temperature Anomaly Database

- **URL:** https://www.ncei.noaa.gov/products/coral-reef-temperature-anomaly-database · accession: https://accession.nodc.noaa.gov/NCEI-CoRTADv6
- **Product:** Weekly SST and thermal-stress metrics for coral reef research (from Pathfinder v5.3)
- **Availability:** **DISCONTINUED — no updates beyond 2022-12-31.** Archived data remain available.
- **Type:** Gridded NetCDF time series
- **Resolution:** 4 km · **Temporal:** weekly, 1982-01-02 – 2022-12-30 (41 years) · **Coverage:** global
- **Variables:** weekly SST, SST anomaly, thermal stress anomaly (TSA), SSTA Degree Heating Week, SSTA Frequency, TSA DHW, TSA Frequency, sea-ice concentration, marine wind speed
- **Licence:** Public domain (NOAA)
- **Access:** NCEI THREDDS, FTP
- **Relevance:** High for **historical** analysis — its 1982-onward record is longer than CRW's operational 5 km series and is purpose-built for reef thermal-stress work.
- **Role:** **Historical thermal-stress baseline / climatology.** Use for long-term context and for defining what constitutes an anomaly; use CRW (#1) for anything current, since CoRTAD is frozen at 2022 and cannot cover the project's 2023–2024 window.

### 11. RECIFS — Reef Environment Centralized InFormation System

- **URL:** https://recifs.epfl.ch · paper: https://onlinelibrary.wiley.com/doi/10.1111/geb.13657 (Selmoni et al. 2023, *Global Ecology and Biogeography*)
- **Product:** Integrated geo-environmental database for coral reef research (EPFL + ENTROPIE)
- **Availability:** Live web application
- **Type:** Harmonised gridded environmental time series
- **Resolution:** 5–25 km · **Temporal:** monthly, past 3–4 decades · **Coverage:** global reef environments
- **Variables:** **chlorophyll, degree heating week, iron, oxygen, pH, nitrate, phosphate, sea current velocity, suspended matter, salinity, temperature** — near-surface
- **Licence:** Open access for research; cite Selmoni et al. 2023
- **Access:** Interactive web app, region/variable selection
- **Relevance:** **Very high — the single best fit for this project's feature list.** It supplies real analogues for `ph`, `salinity_ppt`, `dissolved_oxygen_mg_l`, `current_speed_m_s`, `turbidity_ntu` (via suspended matter), and temperature/DHW — six or seven of the ten deployable features, pre-harmonised, in one place.
- **Role:** **Primary real environmental covariate layer.** Start here for the environmental block; use CRW for high-resolution daily thermal stress on top.

### Bonus — HICORDIS (surfaced by the audit_inbox literature)

- **URL:** https://doi.org/10.1016/j.dib.2016.07.025 (Caldwell et al. 2016, Table S1)
- 286,071 coral colonies · 1,819 transects · 660 sites · 2005–2015 · Hawaiian archipelago · 60 species, 21 health conditions, with GPS and depth
- **Licence: CC BY 4.0** — genuinely open
- **Role:** **External validation for a disease/health sub-model**, and a real-world calibration reference for `disease_percentage`. Small enough to work with immediately, and already in hand as a citation.

### India-specific note

For the actual study regions, Indian institutional data is held by **NCSCM** (reef mapping, 1,439 km² mapped under CRZ 2011/2019; sentinel sites; the **CReON** program with data buoys and AWS at Gulf of Mannar, Andaman, and Lakshadweep sites — https://creon.ncscm.res.in/), **ZSI**, **SAC/ISRO** (1:25,000 reef mapping), and **NCCR/MoES** (bleaching monitoring in Palk Bay, Andaman, Lakshadweep). These are generally **not open-download** and require institutional data requests. NOAA CRW already operates a Lakshadweep Virtual Station, which is the immediately available proxy. Pursuing an NCSCM/CReON data agreement is the highest-value long-lead action for making this project genuinely about Indian reefs.

---

## 17. Dataset-to-Feature Compatibility Matrix

Which sources can supply each of the 16 required inference features?
**●** direct measurement · **◐** derivable/proxy · **○** none

| Feature | Current (synthetic) | CRW #1 | Allen #2 | GEBCO #3 | AIMS #4 | NCEI Sonar #5 | CoralNet #6 | GCRMN #7 | RLS #8 | OBIS #9 | CoRTAD #10 | RECIFS #11 | HICORDIS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `region` | generated | ◐ | ◐ | ◐ | ● | ◐ | ◐ | ● | ● | ● | ◐ | ◐ | ● |
| `depth_m` | `U(0.5,38)` | ○ | ◐ | **●** | ◐ | ● | ◐ | ◐ | ● | ◐ | ○ | ○ | ● |
| `water_temperature_c` | stress-shifted | **●** | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | **●** | **●** | ○ |
| `ph` | stress-shifted | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | **●** | ○ |
| `salinity_ppt` | region only | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | **●** | ○ |
| `dissolved_oxygen_mg_l` | stress+clarity | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | **●** | ○ |
| `turbidity_ntu` | clarity-shifted | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ◐ (susp. matter) | ○ |
| `light_intensity` | Beer–Lambert | ○ | ○ | ◐ (via depth) | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ◐ | ○ |
| `current_speed_m_s` | region only | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ◐ (wind) | **●** | ○ |
| `sonar_backscatter` | structure latent | ○ | ○ | ◐ | ○ | **●** (water column, not seabed) | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| `rugosity_index` | structure latent | ○ | ◐ | ◐ (TRI, coarse) | ○ | ◐ | ◐ | ○ | ◐ | ○ | ○ | ○ | ○ |
| `hard_substrate_percentage` | structure latent | ○ | **●** | ○ | ● | ○ | ● | ● | ● | ○ | ○ | ○ | ◐ |
| `acoustic_complexity_index` | from coral cover | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| `coral_cover_percentage` | stress × structure | ○ | ◐ (benthic class) | ○ | **●** | **●** | **●** | **●** | ◐ | ○ | ○ | ○ | ◐ |
| `bleaching_percentage` | temp + stress | ◐ (DHW→risk, **not** observed) | ◐ (monitoring layer) | ○ | ◐ | ○ | **●** (classifier) | ◐ | ◐ | ○ | ◐ (TSA) | ◐ (DHW) | ● (health cond.) |
| `disease_percentage` | stress+pH+DO | ○ | ○ | ○ | ○ | ○ | ◐ | ○ | ○ | ○ | ○ | ○ | **●** |

**Reading the matrix:**

- **RECIFS (#11) is the highest-coverage single source** — it alone can supply real pH, salinity, DO, current, temperature, and a turbidity proxy: 5–6 of the 10 deployable features.
- **`acoustic_complexity_index` has no source in any column.** Nothing recommended here provides passive-acoustic ACI. It would require an original hydrophone deployment. Given that it is currently *computed from coral cover* rather than measured, **the honest move is to drop it** until real passive acoustics exist.
- **`sonar_backscatter` has only one ● — and it is the wrong kind.** NCEI's archive is water-column, not seabed-classification, backscatter. There is no readily available open source of calibrated *seabed* backscatter for Indian reefs. This is the largest genuine data gap in the project, and it sits directly on the framework's central claim.
- **`bleaching_percentage` has no true ● except CoralNet's classifier and HICORDIS's health conditions.** CRW's DHW is *thermal stress*, i.e. bleaching **risk**, not observed bleaching. Treating DHW as a bleaching measurement would recreate the current circularity in real clothing — this is the trap to avoid.
- **`disease_percentage` has exactly one real source: HICORDIS** (Hawai'i only). For Indian reefs there is no open disease dataset.

---

## 18. Dataset-to-Target Compatibility Matrix

Can each source **responsibly** provide ground truth? Answering §12's two questions separately for every real dataset.

| Source | `reef_health` ground truth? | `restoration_suitability` ground truth? |
|---|---|---|
| **IUCN Red List (in hand)** | **NO** | **NO** |
| **audit_inbox PDFs (in hand)** | **NO** | **NO** |
| **HICORDIS (in hand as citation)** | **PARTIALLY** | **NO** |
| NOAA CRW #1 | **NO** | **NO** |
| Allen Coral Atlas #2 | **PARTIALLY** | **PARTIALLY** |
| GEBCO #3 | **NO** | **NO** |
| AIMS LTMP #4 | **PARTIALLY** | **NO** |
| NCEI Sonar #5 | **NO** | **NO** |
| CoralNet #6 | **PARTIALLY** | **NO** |
| GCRMN #7 | **PARTIALLY** | **NO** |
| Reef Life Survey #8 | **PARTIALLY** | **NO** |
| OBIS #9 | **NO** | **NO** |
| CoRTAD #10 | **NO** | **NO** |
| RECIFS #11 | **NO** | **NO** |

### Justifications

**IUCN Red List — NO / NO.** Contains zero condition variables and has zero geographic overlap with the study regions (§15). It cannot label anything, anywhere in this project.

**audit_inbox PDFs — NO / NO.** They are documents, not datasets. Item #1 *points to* HICORDIS, assessed separately; items #4 and #5 are not even about coral reefs.

**HICORDIS — PARTIALLY / NO.** Provides real, colony-level, expert-assessed health condition with GPS and depth for 286,071 colonies — genuine observed biological state, exactly the kind of measurement this project lacks. Limits: Hawai'i only (no Indian transfer), 2005–2015 only, and its 21 health conditions are **not** the project's four ordinal classes. Mapping condition codes onto `healthy / stressed / bleached / severely_degraded` would be an expert-informed **derived label**, not observed ground truth. For restoration: no substrate, light, current, or restoration-outcome data — **NO**.

**NOAA CRW — NO / NO.** DHW is a *driver*, not a *state*. High DHW means bleaching is likely, not that bleaching occurred; reefs vary enormously in thermal tolerance, and CRW itself frames its products as bleaching-risk alerts. Thresholding DHW into health classes would be **exactly the circularity this audit identifies**, rebuilt from satellite data — an environmental predictor converted into a label and then fed back as a feature. **Use CRW as a feature. Never as a label.** For restoration it says nothing about substrate or larval supply — **NO**.

**Allen Coral Atlas — PARTIALLY / PARTIALLY.** The benthic layer gives real hard-coral presence at 5 m over the actual study regions; the Atlas also runs a bleaching monitoring product. It supports a coarse *habitat-state* classification and could genuinely inform restoration suitability, since substrate quality carries the largest weight (0.22) in the current restoration score and the Atlas measures substrate directly. Limits: static 2022 baseline (no time series for health dynamics), depth-limited (~10 m benthic / ~15 m geomorphic), classes are habitat types not condition states, and satellite classification carries per-class accuracy caveats. Any health or suitability label built from it is a **derived label**. Confirm the licence first.

**GEBCO — NO / NO.** Depth is a covariate. Bathymetry cannot indicate reef condition. It does contribute to a restoration *suitability index* via the 0.09 depth term — but a component of a derived index is not ground truth for it.

**AIMS LTMP — PARTIALLY / NO.** Real diver-observed hard-coral cover over 40 years is the strongest empirical anchor available for what a health target should mean, and its time series capture real disturbance events. Limits: Great Barrier Reef only; manta-tow cover is **ordinal bands**, not continuous percent; and cover alone is not "health" — a low-cover reef may be naturally low-cover rather than degraded. Converting cover trajectories into the four classes is a **proxy label**, and one that requires local baselines. For restoration: no restoration trials, so **NO**.

**NCEI Water-Column Sonar — NO / NO.** Acoustic backscatter of the water column contains no information about benthic condition. It is method-development material for the acoustic side of the framework, nothing more.

**CoralNet — PARTIALLY / NO.** Point-count annotations yield real percent cover, and bleaching classifiers yield real bleaching state from imagery — genuinely observed, image-verified biological variables. This is the most promising route to real values for two of the three currently-unavailable deployment features. Limits: images are unevenly distributed geographically and temporally; source licences vary; annotation quality varies by source; and cover + bleaching still do not constitute a validated four-class health scale. Any four-class label is a **derived label**. For restoration: images say nothing about current, light, or larval supply — **NO**.

**GCRMN — PARTIALLY / NO.** ~2 million observations across 12,000+ sites including South Asia, harmonised, and specifically designed for status-and-trend assessment — the closest thing to authoritative global reef-condition ground truth, and the only realistic source of real benthic cover *for the project's own regions*. Limits: benthic cover only (not a health class), methodological heterogeneity across contributors, and access requires agreement. Deriving four classes needs a documented, expert-reviewed rule with region-specific baselines — a **weak/derived label**, honestly named. For restoration: **NO**.

**Reef Life Survey — PARTIALLY / NO.** Standardised habitat quadrats give real cover; the fish and invertebrate data add an ecological-condition dimension (herbivore biomass, trophic structure) that is arguably a *better* health indicator than coral cover alone. Same limits: not a validated health class; global but sparse for Indian reefs. For restoration: **NO**.

**OBIS — NO / NO.** Presence records. A species being recorded at a site says nothing about that site's condition, and absence in OBIS reflects sampling effort, not ecology.

**CoRTAD — NO / NO.** Same reasoning as CRW, with the added limitation of being frozen at 2022.

**RECIFS — NO / NO.** An environmental covariate database by design and by its authors' own description. Its variables are precisely the *predictors* this project needs — which is exactly why they must never become the label. Thresholding RECIFS pH or DO into "reef health" would reproduce the current circularity with real numbers, which is more dangerous than the synthetic version because it would look credible.

### The rule this table encodes

**No available dataset provides observed ground truth for `restoration_suitability`. Not one.** Restoration suitability is not a measurable state of nature — it is a management judgement about intervention prospects. Real ground truth would require **restoration outcome data**: sites where coral was actually transplanted, with survival and growth tracked over years. Nothing in this audit surfaces such a dataset at scale. Until one exists, `restoration_suitability` can only ever be an **expert-informed derived label**, and it must be labelled as such everywhere — model card, dashboard, API docs, and report.

For `reef_health`, the honest ceiling with currently obtainable data is a **weak label** derived from real observed benthic cover and bleaching (GCRMN / AIMS / CoralNet / HICORDIS) under a documented, expert-reviewed, region-calibrated rule — with the rule's inputs **excluded from the feature set**. That single exclusion is what breaks the circle, and it is the difference between a derived label that supports valid inference and one that does not.

**Terminology to enforce, per §12 of the brief:** any label built by thresholding environmental variables is a **synthetic label** (current state); one built from expert rules over real observations is a **derived label** or **weak label**; one built from a single imperfect real proxy is a **proxy label**. **Observed ground truth** should be reserved for a value a human or instrument actually measured at that site and time. The current dataset has none.

---

## 19. Scientifically Defensible Integration Strategy

Do not concatenate. Every rule below exists because a naive merge would silently manufacture a false observation.

**Rule 1 — Never row-concatenate datasets with different observational units.** `observations.csv` rows are (synthetic) point-in-time sensor readings. AIMS rows are reef-level annual summaries. HICORDIS rows are individual coral colonies. GCRMN rows are transects. CoralNet rows are image points. RECIFS cells are monthly 5–25 km grid means. Stacking these produces a table where a single "row" means six incompatible things. Column-name similarity is not schema compatibility.

**Rule 2 — Join on space-time keys with explicit, documented tolerances.** Every join must declare its spatial and temporal matching radius, and that radius must be carried forward as an uncertainty attribute, not discarded.

| Join | Key | Spatial tolerance | Temporal tolerance | Rationale |
|---|---|---|---|---|
| Survey → CRW thermal stress | lat/lon → 5 km cell | ≤ 5 km (1 cell) | ±1 day; DHW needs a 12-week accumulation window preceding the survey | CRW native grid; DHW is by definition backward-looking |
| Survey → RECIFS environment | lat/lon → 5–25 km cell | ≤ 25 km | same calendar month | RECIFS is monthly; sub-monthly precision is unsupported |
| Survey → GEBCO depth | lat/lon → 450 m cell | ≤ 450 m | none (static) | Bathymetry is time-invariant at project timescales |
| Survey → Allen Coral Atlas | point-in-polygon | exact containment, or ≤ 50 m to nearest reef polygon | none (2022 static) | 5 m pixels; a 50 m buffer handles GPS error |
| Survey ↔ Survey (cross-programme) | reef/site ID first, coordinates only as fallback | ≤ 500 m same-site | same survey season | Site identifiers are authoritative; coordinates drift between programmes |
| Image → benthic cover | image ID → transect ID | exact | exact | Must be the same physical transect, never a nearby one |
| Sonar ping → benthic sample | vessel track → sample point | ≤ 10 m | ≤ 1 h | Acoustic footprint scale; substrate is heterogeneous at metres |

**Rule 3 — Aggregate up, never disaggregate down.** Colony → transect → site → reef is valid, with the number of contributing units retained as a weight. Assigning a reef-level cover value down to individual points invents precision that was never measured. The finest common unit across candidate sources is the **transect/site**, so that is the natural row unit of a real dataset — meaning a real version of this project would have thousands of rows, not 15,000, and each would be a genuine survey.

**Rule 4 — Match the coarsest resolution in the join, and say so.** Joining a 25 km RECIFS monthly mean to a 50 m transect does not give a 50 m environmental measurement. Carry `env_spatial_resolution_km` and `env_temporal_window_days` as explicit columns so downstream users and reviewers can see the mismatch.

**Rule 5 — Reproject everything to EPSG:4326 before joining, and record the source CRS.** The IUCN shapefile is already WGS-84. GEBCO and CRW are geographic. Allen Coral Atlas tiles may arrive in a projected CRS. A silent CRS mismatch produces plausible-looking coordinates hundreds of metres wrong — invisible in a table, fatal in a join.

**Rule 6 — Label-source columns must be excluded from the feature matrix.** Whatever variables construct the label are barred from `ALL_FEATURE_COLUMNS`. This is the one rule that directly fixes the CRITICAL finding, and it must be enforced in code (an assertion in `build_features.py`), not in documentation.

**Rule 7 — Every row carries provenance.** Minimum: `data_source`, `is_synthetic` (bool), `observation_type` (measured / modelled / derived / synthetic), `label_type` (observed / derived / weak / proxy / synthetic), `source_licence`, `citation`. Without these, synthetic and real rows become indistinguishable after one merge — the failure mode most likely to end up in a publication.

**Rule 8 — Never mix synthetic and real rows in the same training table.** Real data validates; synthetic data demonstrates. If synthetic augmentation is ever wanted, it belongs behind an explicit flag with results reported both ways.

**Rule 9 — Sonar requires its own calibration path.** Water-column Sv (NCEI) and seabed backscatter are different physical quantities. Do not substitute one for the other, and do not compare uncalibrated dB across instruments or surveys.

---

## 20. Proposed Future Data Architecture

Proposed only. **Nothing here is implemented.**

```
                 ┌──────────────────────────────────────────────────┐
                 │  LAYER E — BIOLOGICAL GROUND TRUTH (label source) │
                 │  GCRMN · AIMS LTMP · RLS quadrats · HICORDIS      │
                 │  NCSCM/CReON (India — request access)             │
                 │  unit: transect/site · REAL · CANONICAL ROW UNIT  │
                 └───────────────────────┬──────────────────────────┘
                                         │ site_id + date  ← THE SPINE
   ┌──────────────┬──────────────┬───────┴───────┬──────────────┬──────────────┐
   │              │              │               │              │              │
┌──┴────────┐ ┌───┴──────┐ ┌─────┴─────┐ ┌───────┴────┐ ┌───────┴────┐ ┌───────┴────┐
│ LAYER B   │ │ LAYER F  │ │ LAYER C   │ │ LAYER G    │ │ LAYER D    │ │ LAYER I    │
│ in-situ   │ │ satellite│ │ bathymetry│ │ habitat    │ │ acoustic   │ │ image-     │
│ environment│ │ thermal │ │ geomorph. │ │ maps       │ │ sonar      │ │ derived    │
│ RECIFS    │ │ CRW DHW  │ │ GEBCO     │ │ Allen Atlas│ │ NCEI/MBES  │ │ CoralNet   │
│ own sensors│ │ CoRTAD  │ │ own MBES  │ │ NCSCM/SAC  │ │ own surveys│ │ own camera │
│ ±25km/month│ │ ±5km/day│ │ ±450m     │ │ ±5m poly   │ │ ±10m/1h    │ │ same transect│
└───────────┘ └──────────┘ └───────────┘ └────────────┘ └────────────┘ └────────────┘
   │              │              │               │              │              │
   └──────────────┴──────────────┴───────┬───────┴──────────────┴──────────────┘
                                         │
                       ┌─────────────────┴──────────────────┐
                       │  ANALYSIS TABLE (real track)        │
                       │  one row = one survey transect      │
                       │  features: Layers B,C,D,F,G,I only  │
                       │  label: from Layer E only           │
                       │  ── strict separation enforced ──   │
                       └─────────────────────────────────────┘

     ┌─────────────────────────────────────────────────────────────────────┐
     │  LAYER A — SYNTHETIC PROTOTYPE (current 15k)                        │
     │  FROZEN · CI, tests, dashboard demo, MLOps regression ONLY          │
     │  NEVER joined to, concatenated with, or validated against the above │
     └─────────────────────────────────────────────────────────────────────┘

     ┌─────────────────────────────────────────────────────────────────────┐
     │  LAYER H — BIODIVERSITY / SPECIES CONTEXT                           │
     │  OBIS · IUCN (external to repo, licence-restricted)                 │
     │  display and interpretation only — never a feature, never a label   │
     └─────────────────────────────────────────────────────────────────────┘
```

**Layer E is the spine.** This is the architectural inversion that fixes the project. Today, generated *features* come first and labels are computed from them. In the proposed design, real biological *surveys* come first and define which rows exist; every other layer attaches to them. You cannot have a row without a real observation — which makes circular supervision structurally impossible rather than merely discouraged.

**Join keys, in priority order:** (1) `site_id` / `reef_id` / `transect_id` where programmes share identifiers — always preferred; (2) `latitude` + `longitude` + `date` with the §19 tolerances; (3) `spatial_cell` (H3 or a fixed grid) for gridded environmental products; (4) `depth` as a secondary discriminator when multiple transects share a site.

**Spatial and temporal tolerance is a first-class attribute, not an implementation detail.** Each joined column carries its own resolution and lag metadata. A DHW value matched at 5 km/12 weeks and a substrate class matched at 5 m/static are not equally precise, and the analysis table must be able to say so.

**Layer A stays completely disconnected.** No arrow reaches it. That isolation is the point: it keeps the MLOps demonstration working while removing any possibility that synthetic rows leak into a scientific claim.

---

## 21. Risks / Licensing / Provenance Concerns

| # | Risk | Severity | Detail | Mitigation (proposed, not executed) |
|---|---|---|---|---|
| 1 | **IUCN redistribution breach** | **HIGH** (latent) | Terms prohibit redistribution "alone or combined with other data, including within Derivative Works" without written IUCN permission. Repo is MIT-licensed with a public remote. Committing the zip or its polygon would breach. | Keep in `~/Downloads`, outside the repo. Add `data/audit_inbox/` and `*.shp`/`*.zip` to `.gitignore`. Do not serve via API or dashboard. Given zero geographic overlap, exclude entirely. |
| 2 | **Copyrighted PDFs in the working tree** | **MEDIUM** | 4 of 5 inbox PDFs are Elsevier subscription content, all rights reserved. Currently untracked (correct). | `.gitignore` `data/audit_inbox/`. Keep citations, not files, in version control. |
| 3 | **Synthetic data presented as scientific evidence** | **CRITICAL** | Model cards, dashboard, and drift reports carry macro-F1 ≈ 0.76/0.79 without stating these measure recovery of a formula. A reader or examiner would reasonably infer reef-prediction capability. | Add a prominent qualifier wherever a metric appears: *in-distribution synthetic performance; labels are algorithmically derived from the features*. Report LORO alongside random CV. |
| 4 | **Provenance loss on first merge** | **HIGH** | With no `is_synthetic` / `data_source` columns, one concatenation makes synthetic and real rows permanently indistinguishable. | Mandate the §19 Rule 7 provenance block **before** any real data is ingested. |
| 5 | **Allen Coral Atlas licence unverified** | **MEDIUM** | Site asserts ASU copyright; methods page states no licence. Historically CC BY 4.0. | Confirm on the download page and record the terms before ingesting. |
| 6 | **OBIS aggregate is CC BY-NC** | **MEDIUM** | Individual sources are CC0/CC BY/CC BY-NC, but the integrated dataset is CC BY-NC. Blocks commercial or revenue-generating use. | If commercial use is ever contemplated, filter to CC0/CC BY sources using `s3://obis-open-data/licenses.tsv`. |
| 7 | **GCRMN access lead time** | **MEDIUM** | The best biological ground truth for Indian regions requires a data-sharing agreement. | Initiate the request now; assume months, not weeks. |
| 8 | **DHW-as-label trap** | **HIGH** | The most likely next mistake: thresholding real CRW DHW into "reef health." This rebuilds the exact circularity in real clothing — more credible-looking, equally invalid. | Enforce §19 Rule 6 in code. CRW is a **feature**, permanently. |
| 9 | **Deployment claim unsupported** | **CRITICAL** | 3 of 16 required inputs need a diver. The "real-time sonar framework" claim does not currently hold. | Reframe per §13: deployable model on classes A/B/C only; biological variables become targets or training-time auxiliaries. |
| 10 | **Seabed backscatter gap** | **HIGH** | No open source of calibrated *seabed* backscatter for Indian reefs. NCEI is water-column. This sits on the framework's central claim. | Scope an original MBES survey, or seek NCSCM/SAC/NIO acoustic holdings. Do not substitute water-column Sv. |
| 11 | **CoralNet per-source licences** | **LOW** | Aggregated sources carry heterogeneous terms. | Record the licence per source at ingestion. |
| 12 | **CoRTAD discontinued** | **LOW** | Frozen at 2022-12-31; cannot cover 2023–24. | Historical baseline only; CRW for anything current. |
| 13 | **Matlab crash dumps in working tree** | **INFO** | 3 untracked `matlab_crash_dump.*` files at repo root, unrelated to this project. | User's call — left untouched. |
| 14 | **No DOI/version pinning for external data** | **MEDIUM** | GEBCO, CRW, Allen Atlas all revise. Unpinned versions make results irreproducible. | Record product version, DOI, and access date per ingested file; add to DVC. |

---

## 22. Prioritized Next Actions

Approval required before any of these — this audit performed no changes.

### P0 — Truthfulness (do before anything else; hours)

1. **Qualify every reported metric.** Add to the model cards, dashboard Model Performance page, README, and course evidence: *labels are algorithmically derived from the same features supplied to the model; reported scores measure recovery of a scoring function, not reef prediction.* This is the single highest-value action in the list and it costs almost nothing.
2. **Publish the closed-form benchmark alongside model metrics.** State plainly that a noiseless re-computation of the generator formula scores 0.770 / 0.812 macro-F1 versus the model's 0.760 / 0.791. It is the clearest possible statement of what the model is doing, and disclosing it is far stronger than having it found.
3. **Add `data/audit_inbox/` to `.gitignore`** and confirm the IUCN archive stays outside the repository.

### P1 — Evaluation honesty (days)

4. **Add leave-one-region-out evaluation** and report it beside random CV everywhere. The 0.763 → 0.411 gap is the most informative number this dataset can produce about generalisation.
5. **Enforce label-source exclusion in code.** An assertion in `build_features.py` that no label-constructing column appears in `ALL_FEATURE_COLUMNS`, so the invariant survives future edits.
6. **Add a leakage-diagnostic test to CI:** if any single feature exceeds a macro-F1 threshold, or if the closed-form formula beats the model, fail the build. This converts the audit's finding into a standing guard.
7. **Split the feature sets by deployability** — a `deployment_features` list (classes A/B/C) versus `survey_features` — and report metrics for both. Evidence is already in hand: sensor-only costs 0.025 macro-F1 on health and nothing on restoration.

### P2 — Data acquisition (weeks; start the long-lead items now)

8. **Request GCRMN `gcrmndb_benthos` access.** Longest lead time, highest value: the only realistic source of real benthic cover for the project's own regions.
9. **Contact NCSCM / CReON** about Indian reef monitoring and buoy data for Gulf of Mannar, Andaman, and Lakshadweep. Also long lead; makes the project genuinely about Indian reefs.
10. **Ingest RECIFS** for the four regions — highest coverage per unit effort (5–6 real environmental features in one source, open access).
11. **Ingest NOAA CRW DHW** for the four regions, 2018–2026, including the existing Lakshadweep Virtual Station. Public domain, immediate.
12. **Ingest GEBCO_2025** subset for real depth and terrain-ruggedness. Public domain, immediate.
13. **Obtain Allen Coral Atlas** benthic/geomorphic layers after confirming the licence — supplies real substrate and a **reef mask** to fix the coordinates-over-open-ocean problem.
14. **Retrieve HICORDIS Table S1** (CC BY 4.0, already cited in the inbox). Small, open, immediately usable as a disease-model external validation set.

### P3 — Architecture (months; only after P2 lands)

15. **Build the Layer-E-first analysis table** per §20 — survey rows first, environment attached, tolerances recorded. This is the structural fix.
16. **Freeze the current 15k dataset** as `data/synthetic/observations_v1.csv` with a written charter: CI, tests, dashboard demo, MLOps regression only; never a scientific claim.
17. **Prototype the CoralNet-derived camera → benthic-cover model.** This is what converts `coral_cover_percentage` and `bleaching_percentage` from unavailable inputs into measurable ones, and it is the key to the deployment story in §13.
18. **Scope real acoustics.** Either an MBES survey with ground-truthed seabed classification, or an institutional data request. The framework's central claim rests on this, and it is currently the largest genuine gap.
19. **Re-specify the targets.** `reef_health` → a documented, expert-reviewed, region-calibrated **derived label** from real cover and bleaching, with its inputs excluded from features. `restoration_suitability` → explicitly and permanently labelled an **expert-informed derived label**, since no observed ground truth for it exists anywhere in this audit.
20. **Decide on `acoustic_complexity_index`.** It is currently computed from coral cover, is obtainable from no recommended source, and requires passive acoustics the project does not have. Either commit to a hydrophone deployment or drop it.

### Recommendation on the current 15k dataset — §14 of the brief

**D — HYBRID REAL + SYNTHETIC ARCHITECTURE, with a strict partition.** Specifically **A + D**: keep the current dataset unchanged as a legacy synthetic benchmark, and build a separate real-data track for all scientific claims.

**Do not choose B (regenerate).** The leakage is structural, not parametric. Any label computed from variables that are then supplied as features is circular at any noise level, under any weights, with any thresholds. Regenerating would cost real effort and fix nothing — and would risk creating a *more* convincing artefact with the same fatal property. (If the generator is ever revised for other reasons, the fix that would matter is generating an independent latent reef state, deriving the label from *that*, and exposing only noisy sensor observations of it as features. That is a different program from tuning weights.)

**Do not choose C (replace outright).** The current dataset does real work: it keeps DVC, MLflow, CI, drift monitoring, the API, and the dashboard all functioning and testable, with zero licensing risk and instant reproducibility from a seed. Deleting it would break the MLOps demonstration that milestones M1–M14 were built to show, and gain nothing scientifically that the real track will not gain anyway.

**Keep the 15k dataset available for:**

| Use | Keep? | Why |
|---|---|---|
| CI / testing | **Yes** | Deterministic, seeded, fast, no licence risk. Ideal fixture. |
| MLOps demonstration | **Yes** | The pipeline architecture is genuinely sound and worth demonstrating; the data's provenance does not detract from that. |
| Dashboard demonstration | **Yes**, with a visible banner | Must be labelled synthetic wherever a user could mistake it for observation. |
| Regression tests | **Yes** | Fixed hash makes it a perfect regression baseline. |
| Model benchmarking | **Only relative** | Valid for comparing algorithms against each other. Invalid as evidence of reef-prediction capability. |
| **Scientific validation** | **No** | Categorically. |
| **Any conservation or policy claim** | **No** | The generator docstring already says this; the rest of the project should say it as loudly. |

**Required separation:** move to `data/synthetic/`, mark every row `is_synthetic = true`, and make the real track physically distinct — separate directories, separate DVC stages, separate MLflow experiments, separate model registry names. The two must never meet in a training table.

---

## Appendix A — Files inspected

**Source code (read in full):** `src/data/generate_data.py` (765 L) · `src/data/validate.py` (486 L) · `src/data/preprocess.py` (493 L) · `src/features/build_features.py` (334 L) · `params.yaml` (171 L)
**Source code (targeted inspection):** `src/api/schemas.py` (deployment feature contract)
**Data:** `data/raw/observations.csv` (15,000 × 21, full quantitative audit)
**Data (inventoried, not analysed):** `data/raw/observations_validated.csv` · `data/processed/{X,y}_{train,test}_{health,restoration}.csv` · `data/processed/feature_metadata.json` · `data/processed/preprocessor_{health,restoration}.joblib` · `data/reference/reference.csv` · `data/production/production.csv` · `models/best_model_{health,restoration}.joblib` · `artifacts/mlruns.db` · `reports/drift_summary.json`
**External:** 5 PDFs in `data/audit_inbox/` (metadata + text extraction) · `~/Downloads/redlist_species_data_938f7263-….zip` (all 8 members; DBF and SHP parsed binary; PRJ, CPG, ReadMe read in full)

## Appendix B — Datasets discovered

| Location | Item | Type | Verdict |
|---|---|---|---|
| `data/raw/observations.csv` | CoralSense synthetic observations | 15,000 × 21 CSV | Canonical, synthetic, hash-verified unchanged |
| `data/audit_inbox/` | 5 journal-article PDFs | Literature | **No datasets.** 1 useful data descriptor (HICORDIS), 2 relevant reviews, 2 off-topic |
| `~/Downloads/redlist_…zip` | IUCN *C. jardinei* range polygon | ESRI shapefile, 1 record, EPSG:4326 | Reference only; zero overlap with study regions; redistribution prohibited |
| `data/{processed,reference,production}/` | Derived project data | CSV/joblib | All descend from the same synthetic source |

**No other dataset in any supported format exists anywhere in the repository** — verified by extension sweep across `.csv .tsv .json .jsonl .parquet .xlsx .zip .tar .gz .geojson .shp .dbf .shx .prj .gpkg .nc .h5 .hdf5 .tif .tiff` (excluding `.venv` and `.git`).

## Appendix C — Files that could not be read

**None.** Every file encountered was read successfully.

Two limitations worth recording (neither blocked the audit):
- The `.venv` lacks `shapely`, `geopandas`, `fiona`, `pyshp`, `rasterio`, `xarray`, `netCDF4`, and `h5py`. The IUCN shapefile was therefore parsed with a hand-written binary reader and an even-odd ray-casting point-in-polygon routine — **validated against six known-inside/known-outside probe points** before its results were used (§15). Installing `geopandas` would be advisable before any serious geospatial work.
- The two bundled IUCN PDFs (terms, metadata) were not text-extracted; their terms were cross-checked against the current published IUCN Red List Terms and Conditions instead.

## Appendix D — Commands executed

All read-only except writes confined to `/tmp/oceanographic-dataset-audit/`.

```
pwd ; git branch --show-current ; git status --short ; git status -sb ; git rev-parse HEAD
sha256sum <7 protected artefacts>                      # start and end
find data -type f -printf '%s\t%p\n' | sort -rn ; ls -la data/audit_inbox/
find . -not -path '*/.venv/*' -not -path '*/.git/*' \( -iname '*redlist*' -o -iname '*.zip' … \)
find /home/BAAHbun -maxdepth 4 -iname '*redlist*' ; ls /home/BAAHbun/Downloads
wc -l src/data/generate_data.py src/data/validate.py src/data/preprocess.py src/features/build_features.py params.yaml
mkdir -p /tmp/oceanographic-dataset-audit
unzip -l  ~/Downloads/redlist_species_data_…zip        # listed BEFORE extracting
sha256sum ~/Downloads/redlist_species_data_…zip
unzip -o -q ~/Downloads/redlist_species_data_…zip -d /tmp/oceanographic-dataset-audit/iucn/
cat iucn/data_0.prj ; cat iucn/data_0.cpg ; cat iucn/ReadMe.txt
python3  <hand-written DBF header/record parser>
python3  <SHP header + multipart polygon parser; ring areas; ray-casting PIP; probe validation; distance calc>
pdfinfo / pdftotext  <5 audit_inbox PDFs>              # metadata + text extraction
.venv/bin/python /tmp/oceanographic-dataset-audit/audit_core.py       # structure, quality, distributions, geo, temporal, targets
.venv/bin/python /tmp/oceanographic-dataset-audit/audit_leakage.py    # reconstruction, correlations, MI, ablations, splits, spatial NN
python3  <JSON report readers>
.venv/bin/python -c "<geospatial library availability check>"
git status --short ; git status -sb ; git diff --stat HEAD ; sha256sum <7 protected artefacts>
```

WebSearch (§11 research): NOAA CRW · Allen Coral Atlas · GEBCO 2025 · AIMS LTMP · NCEI water-column sonar · CoralNet · Reef Life Survey · CoRTAD v6 · GCRMN/ICRI · RECIFS · OBIS · IUCN terms of use · India (NCSCM/CReON/ZSI/SAC).
WebFetch: `allencoralatlas.org/methods/`.

**No command was run that writes to the repository.** No `dvc repro`, no training, no MLflow call, no model registration, no `git add`/`commit`/`amend`/`push`. No large external dataset was downloaded.

## Appendix E — Did any repository file change?

**No.**

- `git diff HEAD` — empty. No tracked file modified.
- `git status --short` — identical before and after: the same 4 pre-existing untracked entries (`data/audit_inbox/` and 3 matlab crash dumps), all of which existed before this audit and were left untouched.
- Nothing staged, committed, amended, or pushed.
- All audit outputs written exclusively to `/tmp/oceanographic-dataset-audit/`.
- The IUCN archive in `~/Downloads/` was read and listed before extraction; the original file is unmodified (hash recorded).

## Appendix F — Final git status

```
## main...origin/main
?? data/audit_inbox/
?? matlab_crash_dump.283300-1
?? matlab_crash_dump.283487-1
?? matlab_crash_dump.283712-1
```

Identical to the pre-audit state. Branch `main`, level with `origin/main`, HEAD `53f1430`.

## Appendix G — Final protected SHA-256 values

| File | SHA-256 | Matches start? |
|---|---|---|
| `artifacts/mlruns.db` | `b76a401522754ad050793f392ba2cdf0e8f9e4b76140dc8ebb9f604c95f7c477` | **Yes** |
| `data/raw/observations.csv` | `a03cb3e92ba1904ae07147da95f96aa689d092d56fc41b040b701a101ad8f458` | **Yes** — and matches the documented canonical value |
| `reports/drift_summary.json` | `252785f69805d593dea6ddbfa4e123759a176f19925a77bdcba2446d5f13eade` | **Yes** |
| `models/best_model_health.joblib` | `586096df9e164420363f459471c91ac2e5258ab9878e2e150ce3b291184f42d4` | **Yes** |
| `models/best_model_restoration.joblib` | `a93d71dbd6303363bbfcd27014d41254f3e34eebf4157c6249a6172f18865ca8` | **Yes** |
| `data/processed/preprocessor_health.joblib` | `e625aa747c8b6ac2f2b6e0a528279cc4c96906849046b8fd875238875eabb1d5` | **Yes** |
| `data/processed/preprocessor_restoration.joblib` | `2c45c9f43fb5aed5b92a1df5a83ef8cf2c6911d4dc731758282cf1701976a08e` | **Yes** |

Also recorded — external, outside the repository:
`~/Downloads/redlist_species_data_938f7263-9446-4750-a806-d46d4133a98a.zip` → `b188920c03e182acad2c3592b4e1c73d510b75b894027a95fb413be67d52b5ac` (unmodified)

## Appendix H — Working artefacts

Retained in `/tmp/oceanographic-dataset-audit/`:

| File | Contents |
|---|---|
| `AUDIT_REPORT.md` | This report |
| `audit_core.py` / `audit_core.json` | Structure, quality, distributions, geography, temporal, targets |
| `audit_leakage.py` / `audit_leakage.json` | Label reconstruction, correlations, MI, single-feature diagnostics, ablations, split comparison, spatial NN test |
| `pearson_matrix.csv` / `spearman_matrix.csv` | Full 21 × 21 correlation matrices |
| `iucn/` | Extracted IUCN archive (read-only working copy) |

**Sources consulted (§11):**
[NOAA Coral Reef Watch 5km](https://coralreefwatch.noaa.gov/product/5km/) · [CRW NCEI metadata](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.nodc%3ACRW-5km-HeatStressProducts) · [CRW Lakshadweep Virtual Station](https://coralreefwatch.noaa.gov/product/vs/gauges/lakshadweep.php) · [Allen Coral Atlas](https://allencoralatlas.org/) · [Allen Coral Atlas Methods](https://allencoralatlas.org/methods/) · [GEBCO_2025 Grid](https://www.gebco.net/data-products-gridded-bathymetry-data/gebco2025-grid) · [GEBCO download](https://download.gebco.net) · [AIMS LTMP](https://www.aims.gov.au/research-topics/monitoring-and-discovery/monitoring-great-barrier-reef/long-term-monitoring-program) · [AIMS LTMP data (eAtlas)](https://eatlas.org.au/gbr/ltmp-data) · [NCEI Water Column Sonar](https://www.ncei.noaa.gov/products/water-column-sonar-data) · [NCEI WCSD on AWS](https://registry.opendata.aws/ncei-wcsd-archive/) · [CoralNet](https://coralnet.ucsd.edu/about/) · [ReefNet-1.0](https://huggingface.co/datasets/ReefNet/ReefNet-1.0) · [GCRMN](https://gcrmn.net/) · [ICRI](https://icriforum.org/) · [Reef Life Survey](https://reeflifesurvey.com/survey-data/) · [RLS on GBIF](https://www.gbif.org/dataset/38f06820-08c5-42b2-94f6-47cc3e83a54a) · [OBIS data access](https://obis.org/data/access/) · [OBIS manual](https://manual.obis.org/access.html) · [CoRTAD (NCEI)](https://www.ncei.noaa.gov/products/coral-reef-temperature-anomaly-database) · [CoRTAD v6 accession](https://accession.nodc.noaa.gov/NCEI-CoRTADv6) · [RECIFS paper](https://onlinelibrary.wiley.com/doi/10.1111/geb.13657) · [IUCN Red List Terms of Use](https://www.iucnredlist.org/terms/terms-of-use) · [IUCN spatial data](https://www.iucnredlist.org/resources/spatial-data-download) · [NCSCM CReON](https://creon.ncscm.res.in/)

---

**END OF AUDIT — no repository file was changed. Stopping for manual review.**
