"""
src/dashboard/app.py — CoralSense home page.

Launch
------
    streamlit run src/dashboard/app.py

This is the entry point for the multi-page Streamlit dashboard.
Streamlit automatically discovers pages in src/dashboard/pages/.

SYNTHETIC-DATA DISCLAIMER
--------------------------
All observations and predictions shown in this dashboard are based on
synthetically generated data. They must NOT be used to guide real-world
coral-reef conservation decisions.
"""

from __future__ import annotations

import streamlit as st

from src.dashboard.components import (
    CORAL,
    DARK_BLUE,
    DISCLAIMER_FULL,
    HEALTH_COLORS,
    NAVY,
    REGIONS,
    RESTORATION_COLORS,
    TEAL,
    render_sidebar,
    set_page,
)
from src.dashboard.data_loader import load_observations

set_page("Home", layout="wide")

render_sidebar(show_region_filter=False)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    f"<div style='background:linear-gradient(135deg, {NAVY} 0%, #0d2040 100%); "
    f"border-radius:12px; padding:2rem 2.5rem; margin-bottom:1.5rem; "
    f"border:1px solid #1a3a5c;'>"
    f"<div style='color:{TEAL}; font-size:0.85rem; font-weight:600; "
    f"letter-spacing:2px; text-transform:uppercase;'>Oceanographic Research Platform</div>"
    f"<h1 style='color:#ccd6f6; font-size:2rem; font-weight:700; margin:0.5rem 0;'>"
    f"CoralSense — Sonar Habitat Intelligence</h1>"
    f"<p style='color:#8892b0; font-size:0.95rem; max-width:750px; margin:0;'>"
    f"A Machine Learning-Driven Sonar Framework for Real-Time Coral Reef Habitat "
    f"Prediction and Marine Ecosystem Monitoring</p>"
    f"</div>",
    unsafe_allow_html=True,
)

# Disclaimer banner
st.error(
    "**SYNTHETIC DATA ONLY** — All observations in this dashboard are computer-generated "
    "for academic demonstration purposes. These are not real ocean sensor readings."
)

# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

try:
    df = load_observations()
    n_obs = len(df)
    n_regions = df["region"].nunique()
    n_health = df["reef_health"].nunique()
    n_rest = df["restoration_suitability"].nunique()
    n_train = int(n_obs * 0.8)
    n_test = n_obs - n_train
    data_ok = True
except Exception:
    n_obs = n_regions = n_health = n_rest = n_train = n_test = 0
    data_ok = False

col1, col2, col3, col4, col5 = st.columns(5)
metric_style = (
    f"background:{DARK_BLUE}; border-radius:10px; padding:1rem 1.2rem; "
    f"text-align:center; border:1px solid #1a3a5c;"
)

with col1:
    st.markdown(
        f"<div style='{metric_style}'>"
        f"<div style='color:{TEAL}; font-size:0.75rem; font-weight:600; "
        f"letter-spacing:1px;'>OBSERVATIONS</div>"
        f"<div style='color:#ccd6f6; font-size:2rem; font-weight:700;'>{n_obs:,}</div>"
        f"<div style='color:#8892b0; font-size:0.75rem;'>synthetic records</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"<div style='{metric_style}'>"
        f"<div style='color:{TEAL}; font-size:0.75rem; font-weight:600; "
        f"letter-spacing:1px;'>REGIONS</div>"
        f"<div style='color:#ccd6f6; font-size:2rem; font-weight:700;'>{n_regions}</div>"
        f"<div style='color:#8892b0; font-size:0.75rem;'>prototype reef zones</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"<div style='{metric_style}'>"
        f"<div style='color:{TEAL}; font-size:0.75rem; font-weight:600; "
        f"letter-spacing:1px;'>HEALTH CLASSES</div>"
        f"<div style='color:#ccd6f6; font-size:2rem; font-weight:700;'>{n_health}</div>"
        f"<div style='color:#8892b0; font-size:0.75rem;'>reef condition labels</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"<div style='{metric_style}'>"
        f"<div style='color:{TEAL}; font-size:0.75rem; font-weight:600; "
        f"letter-spacing:1px;'>RESTORATION CLASSES</div>"
        f"<div style='color:#ccd6f6; font-size:2rem; font-weight:700;'>{n_rest}</div>"
        f"<div style='color:#8892b0; font-size:0.75rem;'>suitability labels</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

with col5:
    st.markdown(
        f"<div style='{metric_style}'>"
        f"<div style='color:{TEAL}; font-size:0.75rem; font-weight:600; "
        f"letter-spacing:1px;'>TRAIN / TEST</div>"
        f"<div style='color:#ccd6f6; font-size:1.4rem; font-weight:700;'>"
        f"{n_train:,} / {n_test:,}</div>"
        f"<div style='color:#8892b0; font-size:0.75rem;'>80 / 20 stratified split</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.divider()

# ---------------------------------------------------------------------------
# System overview
# ---------------------------------------------------------------------------

left, right = st.columns([3, 2])

with left:
    st.subheader("Project Objective")
    st.markdown(
        """
        CoralSense is a proof-of-concept MLOps platform that demonstrates how
        acoustic sonar data combined with environmental sensor readings can be used
        to classify coral-reef habitat condition and assess restoration suitability.

        **Two classification tasks are supported:**

        - **Reef Health** — Assigns each observation to one of four condition classes:
          *Healthy*, *Stressed*, *Bleached*, or *Severely Degraded*.
        - **Restoration Suitability** — Classifies whether a reef site is *Suitable*,
          *Moderately Suitable*, or *Unsuitable* for intervention.

        **Sonar-first design:**
        The platform treats acoustic backscatter, rugosity index, and acoustic
        complexity index as primary structural indicators of substrate condition.
        Environmental parameters (temperature, pH, dissolved oxygen, turbidity)
        serve as complementary ecological indicators. This distinction reflects
        how sonar surveys are typically used in reef mapping before in-situ
        biological surveys are conducted.
        """
    )

    st.subheader("Classification Labels")
    hcol, rcol = st.columns(2)
    with hcol:
        st.markdown("**Reef Health**")
        for cls in ("healthy", "stressed", "bleached", "severely_degraded"):
            color = HEALTH_COLORS[cls]
            label = cls.replace("_", " ").title()
            st.markdown(
                f"<span style='background:{color}20; color:{color}; "
                f"padding:2px 10px; border-radius:12px; font-size:0.85rem; "
                f"border:1px solid {color}60;'>{label}</span>&nbsp;",
                unsafe_allow_html=True,
            )
            st.markdown("")
    with rcol:
        st.markdown("**Restoration Suitability**")
        for cls in ("suitable", "moderately_suitable", "unsuitable"):
            color = RESTORATION_COLORS[cls]
            label = cls.replace("_", " ").title()
            st.markdown(
                f"<span style='background:{color}20; color:{color}; "
                f"padding:2px 10px; border-radius:12px; font-size:0.85rem; "
                f"border:1px solid {color}60;'>{label}</span>&nbsp;",
                unsafe_allow_html=True,
            )
            st.markdown("")

with right:
    st.subheader("Champion Models")
    st.markdown(
        f"<div style='background:{DARK_BLUE}; border-radius:10px; "
        f"padding:1rem 1.2rem; margin-bottom:0.8rem; border-left:4px solid {TEAL};'>"
        f"<div style='color:{TEAL}; font-size:0.75rem; font-weight:600; "
        f"letter-spacing:1px;'>REEF HEALTH</div>"
        f"<div style='color:#ccd6f6; font-size:1rem; font-weight:600; margin-top:4px;'>"
        f"Logistic Regression v1</div>"
        f"<div style='color:#8892b0; font-size:0.82rem;'>CV macro-F1: 0.761 | "
        f"Test macro-F1: 0.787</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='background:{DARK_BLUE}; border-radius:10px; "
        f"padding:1rem 1.2rem; margin-bottom:0.8rem; border-left:4px solid {CORAL};'>"
        f"<div style='color:{CORAL}; font-size:0.75rem; font-weight:600; "
        f"letter-spacing:1px;'>RESTORATION SUITABILITY</div>"
        f"<div style='color:#ccd6f6; font-size:1rem; font-weight:600; margin-top:4px;'>"
        f"XGBoost v1</div>"
        f"<div style='color:#8892b0; font-size:0.82rem;'>CV macro-F1: 0.791 | "
        f"Test macro-F1: 0.803</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.subheader("MLOps Pipeline")
    pipeline_steps = [
        ("Data Generation", "M2", "#2ecc71"),
        ("Schema Validation", "M3", "#2ecc71"),
        ("Preprocessing", "M4", "#2ecc71"),
        ("Model Training", "M5", "#2ecc71"),
        ("Model Registry", "M6", "#2ecc71"),
        ("DVC Pipeline", "M7", "#2ecc71"),
        ("CI / CD", "M8", "#2ecc71"),
        ("FastAPI Serving", "M9", "#2ecc71"),
        ("Dashboard", "M10", TEAL),
        ("Drift Monitoring", "M11", "#8892b0"),
        ("Docker Deploy", "M12", "#8892b0"),
    ]
    for step, milestone, color in pipeline_steps:
        icon = "✓" if color == "#2ecc71" else ("●" if color == TEAL else "○")
        st.markdown(
            f"<div style='display:flex; align-items:center; "
            f"margin:0.2rem 0; font-size:0.85rem;'>"
            f"<span style='color:{color}; width:18px;'>{icon}</span>"
            f"<span style='color:#ccd6f6; flex:1;'>{step}</span>"
            f"<span style='color:#8892b0; font-size:0.75rem;'>{milestone}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.subheader("Covered Reef Regions")
    for r in REGIONS:
        st.markdown(f"- {r}")

st.divider()

# Full disclaimer
st.markdown(
    f"<div style='background:#1a0a0a; border:1px solid {CORAL}40; "
    f"border-radius:8px; padding:1rem 1.2rem;'>"
    f"<div style='color:{CORAL}; font-size:0.8rem; font-weight:700; "
    f"margin-bottom:0.4rem;'>SYNTHETIC DATA DISCLAIMER</div>"
    f"<div style='color:#c0a090; font-size:0.82rem;'>{DISCLAIMER_FULL}</div>"
    f"</div>",
    unsafe_allow_html=True,
)
