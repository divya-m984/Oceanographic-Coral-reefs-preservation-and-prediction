"""
Page 8 — Drift Monitoring.

Loads the drift summary JSON produced by:
    python -m src.monitoring.run_drift [--no-html] [--shift-scale N]

Displays:
- Feature drift table and per-column p-value heatmap
- Prediction-distribution comparison (reference vs production)
- Confidence-score distribution summary
- Overall recommendation

If the summary file is not found, instructions are shown instead.

SYNTHETIC-DATA DISCLAIMER
--------------------------
All drift results are computed on SYNTHETIC data.
They do not reflect real ocean conditions.
No accuracy or F1 scores are shown for the production window
because the window is intentionally UNLABELED.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import get_config
from src.dashboard.components import (
    CORAL,
    DARK_BLUE,
    DISCLAIMER_FULL,
    TEAL,
    render_sidebar,
    set_page,
)

set_page("Drift Monitoring")
render_sidebar(show_region_filter=False)

# Resolve report locations through the shared Config singleton (the same
# convention used by src/monitoring/run_drift.py) rather than re-deriving them
# from __file__.  This keeps a single source of truth for reports_dir and lets
# tests point the page at an isolated directory instead of the real reports/.
_REPORTS_DIR = get_config().paths.reports_dir
_SUMMARY_PATH = get_config().drift_summary_path

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Drift Monitoring")
st.caption(
    "Statistical drift detection between a reference window (training distribution) "
    "and a synthetic production window. Based on Evidently AI."
)
st.info(
    "**Synthetic data only.** All drift analysis is performed on computer-generated "
    "observations. Results do not represent real ocean conditions. "
    "No accuracy or performance metrics are shown for the production window "
    "because it is intentionally unlabeled.",
    icon="ℹ️",
)

# ---------------------------------------------------------------------------
# Load summary or show instructions
# ---------------------------------------------------------------------------


def _load_summary() -> dict[str, Any] | None:
    if not _SUMMARY_PATH.exists():
        return None
    try:
        with _SUMMARY_PATH.open() as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


summary = _load_summary()

if summary is None:
    st.warning(
        "Drift summary not found. Run the monitoring pipeline first:",
        icon="⚠️",
    )
    st.code("python -m src.monitoring.run_drift", language="bash")
    st.markdown(
        "**Options:**\n"
        "- `--shift-scale 0` — zero-shift baseline (no drift expected)\n"
        "- `--shift-scale 2` — stronger simulated degradation event\n"
        "- `--no-html` — skip HTML report generation (faster)\n"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Overview metrics
# ---------------------------------------------------------------------------

feat = summary.get("feature_drift", {})
pred_h = summary.get("prediction_drift", {}).get("health", {})
pred_r = summary.get("prediction_drift", {}).get("restoration", {})
conf_h = summary.get("confidence_drift", {}).get("health", {})
conf_r = summary.get("confidence_drift", {}).get("restoration", {})

drifted_cols = feat.get("drifted_count", 0)
total_cols = feat.get("total_columns", 0)
pred_h_drifted = pred_h.get("drifted", False)
pred_r_drifted = pred_r.get("drifted", False)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Drifted Features", f"{drifted_cols} / {total_cols}")
col2.metric("Health Pred Drift", "Yes" if pred_h_drifted else "No")
col3.metric("Restoration Pred Drift", "Yes" if pred_r_drifted else "No")
col4.metric("Shift Scale", summary.get("shift_scale", "—"))

st.markdown(
    f"<div style='font-size:0.8rem; color:#8892b0;'>"
    f"Reference: {summary.get('reference_n', '—')} rows &nbsp;|&nbsp; "
    f"Production: {summary.get('production_n', '—')} rows &nbsp;|&nbsp; "
    f"Generated: {summary.get('generated_at', '—')[:19].replace('T', ' ')} UTC"
    f"</div>",
    unsafe_allow_html=True,
)

st.divider()

# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

recommendation = summary.get("recommendation", "")
rec_color = TEAL
rec_icon = "✓"
if recommendation.startswith("RETRAIN"):
    rec_color = "#e74c3c"
    rec_icon = "⚠"
elif recommendation.startswith("INVESTIGATE"):
    rec_color = "#f39c12"
    rec_icon = "●"

st.markdown(
    f"<div style='background:{DARK_BLUE}; border-radius:10px; "
    f"padding:1rem 1.2rem; border-left:4px solid {rec_color};'>"
    f"<div style='color:{rec_color}; font-weight:700; font-size:1rem; margin-bottom:4px;'>"
    f"{rec_icon} Recommendation</div>"
    f"<div style='color:#ccd6f6;'>{recommendation}</div>"
    f"</div>",
    unsafe_allow_html=True,
)

st.divider()

# ---------------------------------------------------------------------------
# Feature Drift
# ---------------------------------------------------------------------------

st.header("Feature Drift")
st.caption(
    f"Drift threshold: p < {summary.get('drift_threshold', 0.10)}. "
    "Method: Kolmogorov-Smirnov (continuous features)."
)

per_col: dict[str, Any] = feat.get("per_column", {})
if per_col:
    rows = []
    for col, info in sorted(per_col.items()):
        rows.append(
            {
                "Feature": col.replace("_", " ").title(),
                "Drifted": "Yes" if info.get("drifted") else "No",
                "p-value": round(info.get("p_value", 1.0), 5),
                "Method": info.get("method", "—"),
            }
        )
    df_feat = pd.DataFrame(rows).sort_values("p-value")

    st.dataframe(
        df_feat.style.apply(
            lambda row: [
                f"color: {'#e74c3c' if v == 'Yes' else '#2ecc71'};" if col == "Drifted" else ""
                for col, v in zip(row.index, row, strict=False)
            ],
            axis=1,
        ),
        use_container_width=True,
        hide_index=True,
    )

    # p-value bar chart
    fig_feat = px.bar(
        df_feat,
        x="p-value",
        y="Feature",
        color="Drifted",
        color_discrete_map={"Yes": "#e74c3c", "No": "#2ecc71"},
        orientation="h",
        title="Feature Drift p-values (lower = more drift)",
    )
    threshold = summary.get("drift_threshold", 0.10)
    fig_feat.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="#f39c12",
        annotation_text=f"threshold={threshold}",
        annotation_font_color="#f39c12",
    )
    fig_feat.update_layout(
        paper_bgcolor=DARK_BLUE,
        plot_bgcolor="#0a1628",
        font_color="#ccd6f6",
        margin=dict(t=50, b=10),
        xaxis=dict(gridcolor="#1a3a5c", range=[0, 1]),
        yaxis=dict(gridcolor="#1a3a5c"),
    )
    st.plotly_chart(fig_feat, use_container_width=True)
else:
    st.info("No per-column drift data available.")

st.divider()

# ---------------------------------------------------------------------------
# Prediction Distribution Drift
# ---------------------------------------------------------------------------

st.header("Prediction Distribution Drift")
st.caption(
    "Distribution of predicted classes in the reference vs production window. "
    "No accuracy is shown — the production window is unlabeled."
)


def _render_pred_drift(task_drift: dict[str, Any], task_label: str, accent: str) -> None:
    drifted = task_drift.get("drifted", False)
    p_val = task_drift.get("p_value", 1.0)
    method = task_drift.get("method", "—")
    ref_dist: dict[str, float] = task_drift.get("reference_distribution", {})
    cur_dist: dict[str, float] = task_drift.get("current_distribution", {})

    status_color = "#e74c3c" if drifted else "#2ecc71"
    status_label = "DRIFTED" if drifted else "STABLE"
    st.markdown(
        f"<div style='background:#0a1628; border-radius:8px; "
        f"padding:0.6rem 1rem; border-left:3px solid {status_color}; margin-bottom:0.5rem;'>"
        f"<span style='color:{status_color}; font-weight:700;'>{status_label}</span>"
        f"<span style='color:#8892b0; font-size:0.85rem;'>"
        f" &nbsp;|&nbsp; p={p_val:.4f} ({method})</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if ref_dist and cur_dist:
        labels = sorted(set(ref_dist) | set(cur_dist))
        df_pred = pd.DataFrame(
            {
                "Class": labels * 2,
                "Share": [ref_dist.get(lbl, 0) for lbl in labels]
                + [cur_dist.get(lbl, 0) for lbl in labels],
                "Window": ["Reference"] * len(labels) + ["Production"] * len(labels),
            }
        )
        fig = px.bar(
            df_pred,
            x="Class",
            y="Share",
            color="Window",
            barmode="group",
            color_discrete_map={"Reference": TEAL, "Production": CORAL},
            title=f"{task_label} — Predicted Class Distribution",
        )
        fig.update_layout(
            paper_bgcolor=DARK_BLUE,
            plot_bgcolor="#0a1628",
            font_color="#ccd6f6",
            margin=dict(t=50, b=10),
            yaxis=dict(gridcolor="#1a3a5c", range=[0, 1]),
            xaxis=dict(gridcolor="#1a3a5c"),
        )
        st.plotly_chart(fig, use_container_width=True)


col_h, col_r = st.columns(2)
with col_h:
    st.subheader("Reef Health")
    _render_pred_drift(pred_h, "Reef Health", TEAL)
with col_r:
    st.subheader("Restoration Suitability")
    _render_pred_drift(pred_r, "Restoration Suitability", CORAL)

st.divider()

# ---------------------------------------------------------------------------
# Confidence Drift
# ---------------------------------------------------------------------------

st.header("Confidence Score Drift")
st.caption(
    "Mean model confidence (max class probability) in reference vs production window. "
    "Decreasing confidence may indicate that the production data is dissimilar from "
    "the training distribution."
)


def _render_conf_drift(conf: dict[str, Any], task_label: str, accent: str) -> None:
    drifted = conf.get("drifted", False)
    p_val = conf.get("p_value", 1.0)
    mean_ref = conf.get("mean_reference", 0.0)
    mean_cur = conf.get("mean_current", 0.0)
    delta = conf.get("delta", 0.0)
    status_color = "#e74c3c" if drifted else "#2ecc71"
    status_label = "DRIFTED" if drifted else "STABLE"
    delta_sign = "▲" if delta >= 0 else "▼"
    st.markdown(
        f"<div style='background:#0a1628; border-radius:10px; "
        f"padding:1rem 1.2rem; border-left:4px solid {accent};'>"
        f"<div style='color:{accent}; font-weight:700; margin-bottom:6px;'>{task_label}</div>"
        f"<div style='display:flex; gap:2rem;'>"
        f"<div><div style='color:#8892b0; font-size:0.78rem;'>REFERENCE</div>"
        f"<div style='color:#ccd6f6; font-size:1.2rem; font-weight:700;'>{mean_ref:.3f}</div></div>"
        f"<div><div style='color:#8892b0; font-size:0.78rem;'>PRODUCTION</div>"
        f"<div style='color:#ccd6f6; font-size:1.2rem; font-weight:700;'>{mean_cur:.3f}</div></div>"
        f"<div><div style='color:#8892b0; font-size:0.78rem;'>DELTA</div>"
        f"<div style='color:{status_color}; font-size:1.2rem; font-weight:700;'>"
        f"{delta_sign} {abs(delta):.3f}</div></div>"
        f"<div><div style='color:#8892b0; font-size:0.78rem;'>KS p-value</div>"
        f"<div style='color:#ccd6f6;'>{p_val:.4f}</div></div>"
        f"<div><div style='color:#8892b0; font-size:0.78rem;'>STATUS</div>"
        f"<div style='color:{status_color}; font-weight:700;'>{status_label}</div></div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )


_render_conf_drift(conf_h, "Reef Health", TEAL)
st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
_render_conf_drift(conf_r, "Restoration Suitability", CORAL)

st.divider()

# ---------------------------------------------------------------------------
# HTML report links
# ---------------------------------------------------------------------------

st.header("Evidently HTML Reports")
st.caption(
    "Full interactive Evidently reports (if generated with the CLI). "
    "Open these files in a browser for detailed per-feature visualisations."
)

report_health = _REPORTS_DIR / "drift_report_health.html"
report_restoration = _REPORTS_DIR / "drift_report_restoration.html"

if report_health.exists():
    st.success("Reef Health report: `reports/drift_report_health.html`")
else:
    st.info("Health HTML report not found. Run CLI without `--no-html` to generate it.")

if report_restoration.exists():
    st.success("Restoration report: `reports/drift_report_restoration.html`")
else:
    st.info("Restoration HTML report not found. Run CLI without `--no-html` to generate it.")

st.code(
    "# Generate HTML reports:\npython -m src.monitoring.run_drift\n\n"
    "# Skip HTML (faster):\npython -m src.monitoring.run_drift --no-html",
    language="bash",
)

st.divider()
st.caption(DISCLAIMER_FULL)
