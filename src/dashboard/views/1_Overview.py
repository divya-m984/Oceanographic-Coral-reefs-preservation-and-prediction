"""
Page 1 — Project Overview.

Displays the system objective, sonar-vs-environmental feature distinction,
dataset statistics, champion model summaries, and the MLOps pipeline overview.
"""

from __future__ import annotations

import streamlit as st

from src.dashboard import theme
from src.dashboard.components import (
    HEALTH_COLORS,
    RESTORATION_COLORS,
    render_disclaimer_banner,
    render_sidebar,
    set_page,
)
from src.dashboard.data_loader import load_evaluation, load_observations
from src.dashboard.viz import mountain_chart

set_page("Overview")

render_sidebar(show_region_filter=False)

# No hero band here.  The cinematic first viewport belongs to the landing; an
# interior page gets the standard header over the quiet still backdrop that
# set_page() rendered above.
theme.page_header(
    "Project Overview",
    "A machine-learning-driven sonar framework for real-time coral-reef habitat "
    "prediction and marine ecosystem monitoring.",
    eyebrow="Oceanographic",
)

st.error(
    "**Synthetic data only.** This dashboard visualises computer-generated "
    "observations. It does not represent a field-deployed ocean monitoring system."
)

# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

theme.section("System Objective", kicker="Purpose")
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

theme.section(
    "Sonar Structure vs Environmental Indicators",
    "Acoustic features describe the reef's physical structure; environmental features "
    "describe the water it sits in.",
    kicker="Feature families",
)

col_s, col_e = st.columns(2, gap="large")
with col_s:
    theme.panel(
        "<ul>"
        "<li>Sonar Backscatter (dB) — hard substrate reflectance</li>"
        "<li>Rugosity Index — benthic surface complexity</li>"
        "<li>Acoustic Complexity Index — biotic sound richness</li>"
        "<li>Hard Substrate Percentage — structural cover</li>"
        "</ul>"
        '<div style="margin-top:0.7rem;font-size:0.78rem;color:' + theme.TEXT_DIM + '">'
        "Primary structural indicators from acoustic surveys</div>",
        label="Acoustic / structural (sonar)",
        accent=theme.AQUA,
    )

with col_e:
    theme.panel(
        "<ul>"
        "<li>Water Temperature (°C)</li>"
        "<li>pH (total scale)</li>"
        "<li>Dissolved Oxygen (mg/L)</li>"
        "<li>Turbidity (NTU)</li>"
        "<li>Light Intensity (PAR)</li>"
        "<li>Salinity (ppt)</li>"
        "<li>Current Speed (m/s)</li>"
        "<li>Depth (m)</li>"
        "<li>Coral Cover, Bleaching, Disease (%)</li>"
        "</ul>"
        '<div style="margin-top:0.7rem;font-size:0.78rem;color:' + theme.TEXT_DIM + '">'
        "In-situ water quality and biological condition</div>",
        label="Environmental / ecological",
        accent=theme.CORAL,
    )

# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

theme.section("Dataset Statistics", kicker="Corpus")

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
    theme.stat_row(
        [
            {
                "label": "Total observations",
                "value": f"{n_obs:,}",
                "caption": "synthetic records",
                "accent": theme.AQUA,
            },
            {
                "label": "Regions",
                "value": f"{df['region'].nunique()}",
                "caption": "prototype reef zones",
                "accent": theme.TURQUOISE,
            },
            {
                "label": "Health classes",
                "value": f"{df['reef_health'].nunique()}",
                "caption": "condition labels",
                "accent": theme.CYAN_SOFT,
            },
            {
                "label": "Restoration classes",
                "value": f"{df['restoration_suitability'].nunique()}",
                "caption": "suitability labels",
                "accent": theme.CORAL_SOFT,
            },
        ]
    )

    theme.spacer("1.4rem")

    # Class balance as layered mountain ridges. Each summit is the exact
    # observation count for that class — the number printed above the peak and
    # the number in the hover are the same value the ridge is drawn from.
    col_h, col_r = st.columns(2, gap="large")
    with col_h:
        health_classes = ["healthy", "stressed", "bleached", "severely_degraded"]
        st.plotly_chart(
            mountain_chart(
                [cls.replace("_", " ").title() for cls in health_classes],
                [int(health_dist.get(cls, 0)) for cls in health_classes],
                value_name="Observations",
                digits=0,
                title="Reef health distribution",
                height=340,
            ),
            use_container_width=True,
        )
        theme.sonar_card(
            [
                (
                    cls.replace("_", " ").title(),
                    f"{health_dist.get(cls, 0) / n_obs * 100:.1f}%",
                    HEALTH_COLORS.get(cls, theme.AQUA),
                )
                for cls in health_classes
            ],
            accent=theme.AQUA,
        )
    with col_r:
        rest_classes = ["suitable", "moderately_suitable", "unsuitable"]
        st.plotly_chart(
            mountain_chart(
                [cls.replace("_", " ").title() for cls in rest_classes],
                [int(rest_dist.get(cls, 0)) for cls in rest_classes],
                value_name="Observations",
                digits=0,
                title="Restoration suitability distribution",
                height=340,
            ),
            use_container_width=True,
        )
        theme.sonar_card(
            [
                (
                    cls.replace("_", " ").title(),
                    f"{rest_dist.get(cls, 0) / n_obs * 100:.1f}%",
                    RESTORATION_COLORS.get(cls, theme.AQUA),
                )
                for cls in rest_classes
            ],
            accent=theme.CORAL,
        )

# ---------------------------------------------------------------------------
# Champion model summaries
# ---------------------------------------------------------------------------

theme.section(
    "Selected Champion Models",
    "Selection criterion: best cross-validated macro F1. All metrics computed on synthetic data.",
    kicker="Model registry",
)

eval_h = load_evaluation("health")
eval_r = load_evaluation("restoration")

m1, m2 = st.columns(2, gap="large")
with m1:
    st.markdown("##### Reef Health — Logistic Regression")
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
    st.markdown("##### Restoration Suitability — XGBoost")
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

# ---------------------------------------------------------------------------
# MLOps pipeline overview
# ---------------------------------------------------------------------------

theme.section("MLOps Pipeline", kicker="Lifecycle")

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
        icon, colour, status = "✓", theme.SUCCESS, "Complete"
    elif name == "Dashboard":
        icon, colour, status = "●", theme.AQUA, "Current"
    else:
        icon, colour, status = "○", theme.TEXT_DIM, "Planned"

    theme.status_row(
        name=name,
        description=description,
        accent=colour,
        icon=icon,
        tag=milestone,
        status=status,
    )

theme.spacer("1.8rem")
render_disclaimer_banner()
