"""
Page 7 — MLOps Status.

Displays the factual completion status of each milestone in the CoralSense
MLOps pipeline and fetches safe champion metadata from GET /model-info when
the API is available.

No internal paths, database URIs or sensitive configuration are exposed.
Drift monitoring is shown as "planned" only — no fabricated drift results.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.dashboard.api_client import APIClient, APIError
from src.dashboard.components import (
    CORAL,
    CYAN,
    DARK_BLUE,
    TEAL,
    render_sidebar,
    set_page,
)

set_page("MLOps Status")

render_sidebar(show_region_filter=False)

st.title("MLOps Pipeline Status")
st.caption(
    "Factual status of every milestone in the CoralSense end-to-end "
    "machine learning operations pipeline."
)

# ---------------------------------------------------------------------------
# Pipeline status table
# ---------------------------------------------------------------------------

STATUS_DONE = "complete"
STATUS_CURRENT = "current"
STATUS_PLANNED = "planned"

PIPELINE = [
    (
        STATUS_DONE,
        "M2",
        "Data Generation",
        "Synthetic Dataset",
        "15,000 sonar + environmental observations (21 columns). "
        "4 Indian reef regions. 2 target labels. Reproducible seed=42.",
        "src/data/generate_data.py",
    ),
    (
        STATUS_DONE,
        "M3",
        "Validation",
        "Schema Validation",
        "Pandera schema: type checks, range constraints, allowed categorical values. "
        "Zero missing values enforced.",
        "src/data/validate.py",
    ),
    (
        STATUS_DONE,
        "M4",
        "Preprocessing",
        "Feature Engineering",
        "ColumnTransformer: StandardScaler (numeric), OneHotEncoder (region). "
        "6 derived features: thermal stress, oxygen stress, acidity deviation, "
        "water quality index, substrate stability, structural complexity.",
        "src/data/preprocess.py, src/features/build_features.py",
    ),
    (
        STATUS_DONE,
        "M5",
        "Model Training",
        "Experiment Tracking",
        "Logistic Regression, Random Forest, XGBoost. 5-fold cross-validation. "
        "MLflow tracking (SQLite backend). Champion selected by CV macro F1.",
        "src/models/train.py",
    ),
    (
        STATUS_DONE,
        "M6",
        "Model Registry",
        "Champion Promotion",
        "MLflow Model Registry. Quality gates (CV macro F1 ≥ 0.70 / 0.73). "
        "Champion alias set. Registry never modified at inference time.",
        "src/models/registry.py, src/models/predict.py",
    ),
    (
        STATUS_DONE,
        "M7",
        "DVC Pipeline",
        "Reproducible Automation",
        "6-stage DVC pipeline: generate → validate → preprocess → train → "
        "evaluate → register_candidate. dvc repro reproduces full pipeline.",
        "dvc.yaml",
    ),
    (
        STATUS_DONE,
        "M8",
        "CI / CD",
        "GitHub Actions",
        "5 jobs: code-quality, tests, pipeline-validation, ml-smoke-test, build. "
        "Canonical MLflow DB never touched. No promotion in CI.",
        ".github/workflows/ci.yml",
    ),
    (
        STATUS_DONE,
        "M9",
        "FastAPI Serving",
        "Inference API",
        "7 HTTP endpoints. Pydantic v2 validation. Lifespan startup. "
        "Graceful 503 degradation. No paths/URIs exposed to clients.",
        "src/api/main.py",
    ),
    (
        STATUS_CURRENT,
        "M10",
        "Dashboard",
        "Streamlit UI",
        "7-page Streamlit dashboard. Interactive reef map, habitat analysis, "
        "restoration planning, live prediction form, model performance, MLOps status.",
        "src/dashboard/",
    ),
    (
        STATUS_PLANNED,
        "M11",
        "Drift Monitoring",
        "Evidently AI",
        "Statistical drift detection on synthetic production data. "
        "Population Stability Index, feature drift reports. (Not yet implemented.)",
        "src/monitoring/drift.py (planned)",
    ),
    (
        STATUS_PLANNED,
        "M12",
        "Docker",
        "Containerisation",
        "docker-compose.yml with FastAPI + Streamlit + MLflow services. (Not yet implemented.)",
        "docker-compose.yml (planned)",
    ),
    (
        STATUS_PLANNED,
        "—",
        "Real Sensor Integration",
        "Field Deployment",
        "Replace synthetic generator with a real sonar + environmental sensor "
        "ingestion module. Requires domain expert involvement and real survey data.",
        "src/data/ (future)",
    ),
]

STATUS_CONFIG = {
    STATUS_DONE: ("#2ecc71", "✓", "Complete"),
    STATUS_CURRENT: (TEAL, "●", "Current Milestone"),
    STATUS_PLANNED: ("#8892b0", "○", "Planned"),
}

for status, milestone, name, category, description, files in PIPELINE:
    color, icon, label = STATUS_CONFIG[status]
    st.markdown(
        f"<div style='background:{DARK_BLUE}; border-radius:10px; "
        f"padding:0.8rem 1.2rem; margin:0.4rem 0; "
        f"border-left:4px solid {color};'>"
        f"<div style='display:flex; align-items:flex-start; gap:0.8rem;'>"
        f"<div style='min-width:30px; color:{color}; font-size:1.1rem; "
        f"font-weight:700; padding-top:2px;'>{icon}</div>"
        f"<div style='flex:1;'>"
        f"<div style='display:flex; align-items:center; gap:0.5rem; margin-bottom:2px;'>"
        f"<span style='color:{CYAN}; font-size:0.75rem; font-weight:600; "
        f"background:#0a1628; padding:1px 6px; border-radius:4px; "
        f"border:1px solid {color}40;'>{milestone}</span>"
        f"<span style='color:#ccd6f6; font-weight:700;'>{name}</span>"
        f"<span style='color:#8892b0; font-size:0.78rem;'>— {category}</span>"
        f"<span style='color:{color}; font-size:0.72rem; margin-left:auto; "
        f"background:{color}15; padding:1px 8px; border-radius:10px; "
        f"border:1px solid {color}40;'>{label}</span>"
        f"</div>"
        f"<div style='color:#8892b0; font-size:0.85rem;'>{description}</div>"
        f"<div style='color:#4a5568; font-size:0.75rem; margin-top:2px;'>{files}</div>"
        f"</div></div></div>",
        unsafe_allow_html=True,
    )

st.divider()

# ---------------------------------------------------------------------------
# Live champion metadata from API
# ---------------------------------------------------------------------------

st.header("Champion Model Registry")
st.caption("Safe metadata fetched from GET /model-info (no paths or URIs exposed).")

try:
    client = APIClient()
    info = client.model_info()
    api_ok = True
except APIError as exc:
    st.warning(f"API not available: {exc}")
    api_ok = False
    info = {}

if api_ok:
    col_h, col_r = st.columns(2)

    def _render_model_card(task_info: dict[str, Any], task_label: str, accent: str) -> None:
        if not task_info.get("available", False):
            st.warning(f"{task_label} champion not available.")
            return
        st.markdown(
            f"<div style='background:#0a1628; border-radius:10px; "
            f"padding:1rem 1.2rem; border-left:4px solid {accent};'>"
            f"<div style='color:{accent}; font-size:0.75rem; font-weight:600; "
            f"letter-spacing:1px; text-transform:uppercase;'>{task_label}</div>"
            f"<div style='color:#ccd6f6; font-size:1.05rem; font-weight:700; "
            f"margin:4px 0;'>{task_info.get('registered_model_name', '—')}</div>"
            f"<div style='color:#8892b0; font-size:0.85rem;'>"
            f"Version: <strong style='color:#ccd6f6;'>{task_info.get('version', '—')}</strong>"
            f" &nbsp;|&nbsp; Alias: "
            f"<strong style='color:{accent};'>{task_info.get('alias', '—')}</strong>"
            f"</div>"
            f"<div style='color:#8892b0; font-size:0.85rem; margin-top:4px;'>"
            f"Algorithm: <strong style='color:#ccd6f6;'>"
            f"{task_info.get('algo_name', '—').replace('_', ' ').title()}</strong>"
            f"</div>"
            f"<div style='color:#8892b0; font-size:0.85rem;'>"
            f"CV Macro F1: <strong style='color:#ccd6f6;'>"
            f"{task_info.get('cv_macro_f1', 0):.4f}</strong>"
            f"</div>"
            f"<div style='color:#8892b0; font-size:0.85rem;'>"
            f"Run ID: <code style='color:#4a5568; font-size:0.78rem;'>"
            f"{task_info.get('run_id', '—')[:16]}…</code>"
            f"</div>"
            f"<div style='color:#8892b0; font-size:0.78rem; margin-top:6px;'>"
            f"Labels: {', '.join(task_info.get('label_names', []))}"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_h:
        _render_model_card(info.get("health", {}), "Reef Health", TEAL)
    with col_r:
        _render_model_card(info.get("restoration", {}), "Restoration Suitability", CORAL)

    st.caption(
        "Source: GET /model-info. No internal paths, MLflow URIs or tracking "
        "database details are included in this response."
    )

# ---------------------------------------------------------------------------
# Planned items note
# ---------------------------------------------------------------------------

st.divider()
st.markdown(
    f"<div style='background:#0a1628; border:1px solid #1a3a5c; "
    f"border-radius:8px; padding:1rem 1.2rem;'>"
    f"<strong style='color:{TEAL};'>Drift Monitoring (M11 — Planned)</strong><br>"
    f"<span style='color:#8892b0; font-size:0.85rem;'>"
    f"Statistical drift detection using Evidently AI is planned for M11. "
    f"This page will display Population Stability Index (PSI), feature drift "
    f"heatmaps, and prediction distribution shifts between reference and production "
    f"windows. No drift results are shown here because M11 is not yet implemented."
    f"</span></div>",
    unsafe_allow_html=True,
)
