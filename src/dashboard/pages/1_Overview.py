"""
Page 1 — Project Overview.

Displays the system objective, sonar-vs-environmental feature distinction,
dataset statistics, champion model summaries, and the MLOps pipeline overview.
"""

from __future__ import annotations

import streamlit as st

from src.dashboard.components import (
    CORAL,
    DARK_BLUE,
    DISCLAIMER_FULL,
    HEALTH_COLORS,
    RESTORATION_COLORS,
    TEAL,
    render_sidebar,
    set_page,
)
from src.dashboard.data_loader import load_evaluation, load_observations

set_page("Overview")

render_sidebar(show_region_filter=False)

st.title("Project Overview")
st.caption(
    "Oceanographic: A Machine Learning-Driven Sonar Framework for "
    "Real-Time Coral Reef Habitat Prediction and Marine Ecosystem Monitoring"
)

st.error(
    "**Synthetic data only.** This dashboard visualises computer-generated "
    "observations. It does not represent a field-deployed ocean monitoring system."
)

# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

st.header("System Objective")
st.markdown(
    """
    CoralSense demonstrates an end-to-end MLOps pipeline for classifying coral-reef
    habitat condition and assessing restoration suitability using simulated sonar and
    environmental sensor data.

    The project covers the full ML lifecycle:
    data generation → validation → preprocessing → training → registry →
    DVC automation → CI/CD → API serving → dashboard.
    """
)

# ---------------------------------------------------------------------------
# Sonar vs environmental features
# ---------------------------------------------------------------------------

st.header("Sonar Structure vs Environmental Indicators")

col_s, col_e = st.columns(2)
with col_s:
    st.markdown(
        f"<div style='background:{DARK_BLUE}; border-radius:10px; "
        f"padding:1rem 1.2rem; border-left:4px solid {TEAL};'>"
        f"<div style='color:{TEAL}; font-weight:700; margin-bottom:0.5rem;'>"
        f"Acoustic / Structural (Sonar)</div>"
        f"<ul style='color:#ccd6f6; font-size:0.88rem; margin:0; padding-left:1.2rem;'>"
        f"<li>Sonar Backscatter (dB) — hard substrate reflectance</li>"
        f"<li>Rugosity Index — benthic surface complexity</li>"
        f"<li>Acoustic Complexity Index — biotic sound richness</li>"
        f"<li>Hard Substrate Percentage — structural cover</li>"
        f"</ul>"
        f"<div style='color:#8892b0; font-size:0.78rem; margin-top:0.5rem;'>"
        f"Primary structural indicators from acoustic surveys</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

with col_e:
    st.markdown(
        f"<div style='background:{DARK_BLUE}; border-radius:10px; "
        f"padding:1rem 1.2rem; border-left:4px solid {CORAL};'>"
        f"<div style='color:{CORAL}; font-weight:700; margin-bottom:0.5rem;'>"
        f"Environmental / Ecological</div>"
        f"<ul style='color:#ccd6f6; font-size:0.88rem; margin:0; padding-left:1.2rem;'>"
        f"<li>Water Temperature (°C)</li>"
        f"<li>pH (total scale)</li>"
        f"<li>Dissolved Oxygen (mg/L)</li>"
        f"<li>Turbidity (NTU)</li>"
        f"<li>Light Intensity (PAR)</li>"
        f"<li>Salinity (ppt)</li>"
        f"<li>Current Speed (m/s)</li>"
        f"<li>Depth (m)</li>"
        f"<li>Coral Cover, Bleaching, Disease (%)</li>"
        f"</ul>"
        f"<div style='color:#8892b0; font-size:0.78rem; margin-top:0.5rem;'>"
        f"In-situ water quality and biological condition</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

st.header("Dataset Statistics")

try:
    df = load_observations()
    n_obs = len(df)
    health_dist = df["reef_health"].value_counts()
    rest_dist = df["restoration_suitability"].value_counts()
    data_ok = True
except Exception as exc:
    st.warning(f"Dataset unavailable: {exc}")
    data_ok = False

if data_ok:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Observations", f"{n_obs:,}")
    with c2:
        st.metric("Regions", df["region"].nunique())
    with c3:
        st.metric("Health Classes", df["reef_health"].nunique())
    with c4:
        st.metric("Restoration Classes", df["restoration_suitability"].nunique())

    col_h, col_r = st.columns(2)
    with col_h:
        st.subheader("Reef Health Distribution")
        for cls in ("healthy", "stressed", "bleached", "severely_degraded"):
            count = health_dist.get(cls, 0)
            pct = count / n_obs * 100
            color = HEALTH_COLORS.get(cls, TEAL)
            st.markdown(
                f"<div style='margin:0.25rem 0;'>"
                f"<div style='display:flex; justify-content:space-between; "
                f"font-size:0.85rem;'>"
                f"<span style='color:{color};'>{cls.replace('_', ' ').title()}</span>"
                f"<span style='color:#8892b0;'>{count:,} ({pct:.1f}%)</span>"
                f"</div>"
                f"<div style='background:#1a2f4e; border-radius:3px; height:8px;'>"
                f"<div style='background:{color}; height:8px; border-radius:3px; "
                f"width:{pct:.1f}%;'></div></div></div>",
                unsafe_allow_html=True,
            )
    with col_r:
        st.subheader("Restoration Suitability Distribution")
        for cls in ("suitable", "moderately_suitable", "unsuitable"):
            count = rest_dist.get(cls, 0)
            pct = count / n_obs * 100
            color = RESTORATION_COLORS.get(cls, TEAL)
            st.markdown(
                f"<div style='margin:0.25rem 0;'>"
                f"<div style='display:flex; justify-content:space-between; "
                f"font-size:0.85rem;'>"
                f"<span style='color:{color};'>{cls.replace('_', ' ').title()}</span>"
                f"<span style='color:#8892b0;'>{count:,} ({pct:.1f}%)</span>"
                f"</div>"
                f"<div style='background:#1a2f4e; border-radius:3px; height:8px;'>"
                f"<div style='background:{color}; height:8px; border-radius:3px; "
                f"width:{pct:.1f}%;'></div></div></div>",
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# Champion model summaries
# ---------------------------------------------------------------------------

st.header("Selected Champion Models")

eval_h = load_evaluation("health")
eval_r = load_evaluation("restoration")

m1, m2 = st.columns(2)
with m1:
    st.subheader("Reef Health — Logistic Regression")
    if eval_h:
        best = eval_h["models"][eval_h["best_model_name"]]
        st.markdown(
            f"| Metric | CV (mean) | Test |\n"
            f"|---|---|---|\n"
            f"| Macro F1 | {best['cv_macro_f1_mean']:.4f} | {best['test_macro_f1']:.4f} |\n"
            f"| Bal. Accuracy | {best['cv_balanced_accuracy_mean']:.4f} | "
            f"{best['test_balanced_accuracy']:.4f} |\n"
            f"| Accuracy | — | {best['test_accuracy']:.4f} |"
        )
    else:
        st.caption("Evaluation file not found.")

with m2:
    st.subheader("Restoration Suitability — XGBoost")
    if eval_r:
        best = eval_r["models"][eval_r["best_model_name"]]
        st.markdown(
            f"| Metric | CV (mean) | Test |\n"
            f"|---|---|---|\n"
            f"| Macro F1 | {best['cv_macro_f1_mean']:.4f} | {best['test_macro_f1']:.4f} |\n"
            f"| Bal. Accuracy | {best['cv_balanced_accuracy_mean']:.4f} | "
            f"{best['test_balanced_accuracy']:.4f} |\n"
            f"| Accuracy | — | {best['test_accuracy']:.4f} |"
        )
    else:
        st.caption("Evaluation file not found.")

st.caption(
    "Selection criterion: best cross-validated macro F1. All metrics computed on synthetic data."
)

# ---------------------------------------------------------------------------
# MLOps pipeline overview
# ---------------------------------------------------------------------------

st.header("MLOps Pipeline")

steps = [
    ("M2", "Data Generation", "15,000 synthetic sonar + environmental observations", True),
    ("M3", "Schema Validation", "Pandera schema — ranges, types, allowed values", True),
    (
        "M4",
        "Preprocessing",
        "ColumnTransformer: StandardScaler + OneHotEncoder, 6 derived features",
        True,
    ),
    ("M5", "Model Training", "LR / RF / XGBoost with 5-fold CV; MLflow tracking", True),
    ("M6", "Model Registry", "Quality gates; champion promotion via MLflow Registry", True),
    ("M7", "DVC Pipeline", "6-stage reproducible pipeline (generate→register_candidate)", True),
    ("M8", "CI / CD", "GitHub Actions: 5 jobs — lint, test, validate, smoke-test, build", True),
    ("M9", "FastAPI Serving", "7 endpoints; lifespan startup; Pydantic v2 validation", True),
    ("M10", "Dashboard", "Streamlit multi-page dashboard (current milestone)", False),
    ("M11", "Drift Monitoring", "Evidently AI statistical drift reports (planned)", False),
    ("M12", "Docker", "docker compose up --build (planned)", False),
]

for milestone, name, description, done in steps:
    if done:
        icon, color = "✓", "#2ecc71"
    elif name == "Dashboard":
        icon, color = "●", TEAL
    else:
        icon, color = "○", "#8892b0"

    st.markdown(
        f"<div style='background:{DARK_BLUE}; border-radius:8px; "
        f"padding:0.6rem 1rem; margin:0.3rem 0; "
        f"border-left:3px solid {color}; display:flex; gap:1rem;'>"
        f"<span style='color:{color}; font-weight:700; min-width:35px;'>{icon} {milestone}</span>"
        f"<span style='color:#ccd6f6; font-weight:600; min-width:180px;'>{name}</span>"
        f"<span style='color:#8892b0; font-size:0.85rem;'>{description}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.divider()
st.markdown(
    f"<div style='background:#1a0a0a; border-radius:8px; padding:1rem 1.2rem; "
    f"border:1px solid {CORAL}40;'>"
    f"<strong style='color:{CORAL};'>Disclaimer:</strong> "
    f"<span style='color:#c0a090; font-size:0.85rem;'>{DISCLAIMER_FULL}</span>"
    f"</div>",
    unsafe_allow_html=True,
)
