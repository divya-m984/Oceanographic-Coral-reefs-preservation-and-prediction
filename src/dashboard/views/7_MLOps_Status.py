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

from src.dashboard import theme
from src.dashboard.api_client import APIClient, APIError
from src.dashboard.components import (
    render_sidebar,
    set_page,
)

set_page("MLOps Status")

render_sidebar(show_region_filter=False)

theme.page_header(
    "MLOps Pipeline Status",
    "Factual status of every milestone in the CoralSense end-to-end "
    "machine learning operations pipeline.",
    eyebrow="Operations",
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
        STATUS_DONE,
        "M10",
        "Dashboard",
        "Streamlit UI",
        "8-page Streamlit dashboard. Interactive reef map, habitat analysis, "
        "restoration planning, live prediction form, model performance, MLOps status, drift monitoring.",
        "src/dashboard/",
    ),
    (
        STATUS_CURRENT,
        "M11",
        "Drift Monitoring",
        "Evidently AI",
        "Statistical drift detection on synthetic production window. Feature drift (KS test), "
        "prediction-distribution drift (chi-squared), confidence drift (KS test). "
        "Configurable shift magnitude. JSON summary + optional HTML reports. "
        "DVC run_drift stage. CLI: python -m src.monitoring.run_drift",
        "src/monitoring/drift.py, src/monitoring/run_drift.py",
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
    STATUS_DONE: (theme.SUCCESS, "✓", "Complete"),
    STATUS_CURRENT: (theme.AQUA, "●", "Current"),
    STATUS_PLANNED: (theme.TEXT_DIM, "○", "Planned"),
}

_completed = sum(1 for row in PIPELINE if row[0] == STATUS_DONE)
theme.stat_row(
    [
        {
            "label": "Milestones complete",
            "value": f"{_completed} / {len(PIPELINE)}",
            "caption": "delivered end to end",
            "accent": theme.SUCCESS,
        },
        {
            "label": "In progress",
            "value": f"{sum(1 for row in PIPELINE if row[0] == STATUS_CURRENT)}",
            "caption": "current milestone",
            "accent": theme.AQUA,
        },
        {
            "label": "Planned",
            "value": f"{sum(1 for row in PIPELINE if row[0] == STATUS_PLANNED)}",
            "caption": "not yet implemented",
            "accent": theme.TEXT_DIM,
        },
    ]
)

theme.section("Pipeline Milestones", kicker="Delivery log")

for status, milestone, name, category, description, files in PIPELINE:
    color, icon, label = STATUS_CONFIG[status]
    theme.status_row(
        name=name,
        description=description,
        accent=color,
        icon=icon,
        tag=milestone,
        category=category,
        status=label,
        files=files,
    )

st.divider()

# ---------------------------------------------------------------------------
# Live champion metadata from API
# ---------------------------------------------------------------------------

theme.section(
    "Champion Model Registry",
    "Safe metadata fetched from GET /model-info (no paths or URIs exposed).",
    kicker="Live registry",
)

try:
    client = APIClient()
    info = client.model_info()
    api_ok = True
except APIError as exc:
    st.warning(f"API not available: {exc}")
    api_ok = False
    info = {}

if api_ok:
    col_h, col_r = st.columns(2, gap="large")

    def _render_model_card(task_info: dict[str, Any], task_label: str, accent: str) -> None:
        if not task_info.get("available", False):
            st.warning(f"{task_label} champion not available.")
            return
        rows = [
            ("Version", str(task_info.get("version", "—"))),
            ("Alias", str(task_info.get("alias", "—"))),
            ("Algorithm", str(task_info.get("algo_name", "—")).replace("_", " ").title()),
            ("CV macro F1", f"{task_info.get('cv_macro_f1', 0):.4f}"),
            ("Run ID", f"{task_info.get('run_id', '—')[:16]}…"),
            ("Labels", ", ".join(task_info.get("label_names", []))),
        ]
        body = "".join(
            f'<div style="display:flex;gap:0.8rem;padding:0.22rem 0;font-size:0.85rem">'
            f'<span style="color:{theme.TEXT_DIM};min-width:96px">{key}</span>'
            f'<span style="color:{theme.TEXT}">{value}</span></div>'
            for key, value in rows
        )
        theme.panel(
            body,
            label=task_label,
            title=str(task_info.get("registered_model_name", "—")),
            accent=accent,
        )

    with col_h:
        _render_model_card(info.get("health", {}), "Reef Health", theme.AQUA)
    with col_r:
        _render_model_card(info.get("restoration", {}), "Restoration Suitability", theme.CORAL)

    st.caption(
        "Source: GET /model-info. No internal paths, MLflow URIs or tracking "
        "database details are included in this response."
    )

# ---------------------------------------------------------------------------
# Planned items note
# ---------------------------------------------------------------------------

st.divider()
theme.panel(
    "Statistical drift detection using Evidently AI is planned for M11. "
    "This page will display Population Stability Index (PSI), feature drift "
    "heatmaps, and prediction distribution shifts between reference and production "
    "windows. No drift results are shown here because M11 is not yet implemented.",
    label="Roadmap",
    title="Drift Monitoring (M11 — Planned)",
    accent=theme.AQUA,
)
