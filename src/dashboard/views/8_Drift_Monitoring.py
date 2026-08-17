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
import streamlit as st

from src.config import get_config
from src.dashboard import theme
from src.dashboard.components import (
    DISCLAIMER_FULL,
    render_sidebar,
    set_page,
)
from src.dashboard.viz import mirrored_stream, mountain_chart

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

theme.page_header(
    "Drift Monitoring",
    "Statistical drift detection between a reference window (training distribution) "
    "and a synthetic production window. Based on Evidently AI.",
    eyebrow="Model health",
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

theme.stat_row(
    [
        {
            "label": "Drifted features",
            "value": f"{drifted_cols} / {total_cols}",
            "caption": "columns past the threshold",
            "accent": theme.DANGER if drifted_cols else theme.SUCCESS,
        },
        {
            "label": "Health pred drift",
            "value": "Yes" if pred_h_drifted else "No",
            "caption": "predicted-class distribution",
            "accent": theme.DANGER if pred_h_drifted else theme.SUCCESS,
        },
        {
            "label": "Restoration pred drift",
            "value": "Yes" if pred_r_drifted else "No",
            "caption": "predicted-class distribution",
            "accent": theme.DANGER if pred_r_drifted else theme.SUCCESS,
        },
        {
            "label": "Shift scale",
            "value": str(summary.get("shift_scale", "—")),
            "caption": "simulated degradation",
            "accent": theme.AQUA,
        },
    ]
)

st.markdown(
    f"<div style='font-size:0.78rem; color:{theme.TEXT_DIM}; margin-top:0.9rem'>"
    f"Reference: {summary.get('reference_n', '—')} rows &nbsp;·&nbsp; "
    f"Production: {summary.get('production_n', '—')} rows &nbsp;·&nbsp; "
    f"Generated: {summary.get('generated_at', '—')[:19].replace('T', ' ')} UTC"
    f"</div>",
    unsafe_allow_html=True,
)

theme.spacer("1.4rem")

# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

recommendation = summary.get("recommendation", "")
rec_color = theme.SUCCESS
rec_icon = "✓"
if recommendation.startswith("RETRAIN"):
    rec_color = theme.DANGER
    rec_icon = "⚠"
elif recommendation.startswith("INVESTIGATE"):
    rec_color = theme.WARNING
    rec_icon = "●"

theme.panel(
    f"<span style='color:{theme.TEXT};font-size:0.95rem'>{recommendation}</span>",
    label=f"{rec_icon} Recommendation",
    accent=rec_color,
)

st.divider()

# ---------------------------------------------------------------------------
# Feature Drift
# ---------------------------------------------------------------------------

theme.section(
    "Feature Drift",
    f"Drift threshold: p < {summary.get('drift_threshold', 0.10)}. "
    "Method: Kolmogorov-Smirnov (continuous features).",
    kicker="Input distribution",
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
                f"color: {theme.DANGER if v == 'Yes' else theme.SUCCESS};"
                if col == "Drifted"
                else ""
                for col, v in zip(row.index, row, strict=False)
            ],
            axis=1,
        ),
        use_container_width=True,
        hide_index=True,
    )

    # p-values as a mountain range: each summit is the feature's exact p-value,
    # and the shoreline marks the drift threshold. A ridge that fails to reach
    # the threshold line is a drifted feature.
    threshold = float(summary.get("drift_threshold", 0.10))
    fig_feat = mountain_chart(
        list(df_feat["Feature"]),
        [float(v) for v in df_feat["p-value"]],
        value_name="p-value",
        digits=5,
        y_range=(0, 1),
        title="Feature drift p-values (lower = more drift)",
        height=460,
    )
    fig_feat.add_hline(
        y=threshold,
        line_dash="dash",
        line_color=theme.WARNING,
        annotation_text=f"threshold = {threshold}",
        annotation_font_color=theme.WARNING,
        annotation_position="top left",
    )
    fig_feat.update_layout(
        margin=dict(t=70, b=120, l=64, r=26),
        xaxis=dict(tickangle=-40, tickfont=dict(size=10)),
    )
    st.plotly_chart(fig_feat, use_container_width=True)
else:
    st.info("No per-column drift data available.")

st.divider()

# ---------------------------------------------------------------------------
# Prediction Distribution Drift
# ---------------------------------------------------------------------------

theme.section(
    "Prediction Distribution Drift",
    "Distribution of predicted classes in the reference vs production window. "
    "No accuracy is shown — the production window is unlabeled.",
    kicker="Output distribution",
)


def _render_pred_drift(task_drift: dict[str, Any], task_label: str, accent: str) -> None:
    drifted = task_drift.get("drifted", False)
    p_val = task_drift.get("p_value", 1.0)
    method = task_drift.get("method", "—")
    ref_dist: dict[str, float] = task_drift.get("reference_distribution", {})
    cur_dist: dict[str, float] = task_drift.get("current_distribution", {})

    status_color = theme.DANGER if drifted else theme.SUCCESS
    status_label = "DRIFTED" if drifted else "STABLE"
    st.markdown(
        f"<div class='cs-status' style='margin-bottom:0.6rem'>"
        f"<span class='cs-status__dot' style='background:{status_color};"
        f"box-shadow:0 0 0 3px {status_color}33'></span>"
        f"<span class='cs-status__text' style='color:{status_color}'>{status_label}</span>"
        f"<span class='cs-status__sub'>p={p_val:.4f} ({method})</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if ref_dist and cur_dist:
        labels = sorted(set(ref_dist) | set(cur_dist))
        # Mirrored stream: the two halves are two genuinely different
        # measurements of the same classes — the reference window above the
        # baseline and the production window below it.  Both sets of shares are
        # positive; the downward direction is a drawing convention and the
        # hover and axis ticks both report the true positive share.
        fig = mirrored_stream(
            labels,
            {"share": [float(ref_dist.get(lbl, 0.0)) for lbl in labels]},
            {"share": [float(cur_dist.get(lbl, 0.0)) for lbl in labels]},
            up_name="Reference window",
            down_name="Production window",
            colors={"share": accent},
            value_name="Share",
            value_format=".3f",
            title=f"{task_label} — predicted class distribution",
            height=360,
        )
        st.plotly_chart(fig, use_container_width=True)

        theme.sonar_card(
            [
                (
                    lbl.replace("_", " ").title(),
                    f"{float(ref_dist.get(lbl, 0.0)):.3f} → {float(cur_dist.get(lbl, 0.0)):.3f}",
                    theme.DANGER
                    if abs(float(cur_dist.get(lbl, 0.0)) - float(ref_dist.get(lbl, 0.0))) >= 0.05
                    else theme.TEXT_BRIGHT,
                )
                for lbl in labels
            ],
            accent=accent,
        )


col_h, col_r = st.columns(2, gap="large")
with col_h:
    st.markdown("##### Reef Health")
    _render_pred_drift(pred_h, "Reef Health", theme.AQUA)
with col_r:
    st.markdown("##### Restoration Suitability")
    _render_pred_drift(pred_r, "Restoration Suitability", theme.CORAL)

st.divider()

# ---------------------------------------------------------------------------
# Confidence Drift
# ---------------------------------------------------------------------------

theme.section(
    "Confidence Score Drift",
    "Mean model confidence (max class probability) in reference vs production window. "
    "Decreasing confidence may indicate that the production data is dissimilar from "
    "the training distribution.",
    kicker="Certainty",
)


def _render_conf_drift(conf: dict[str, Any], task_label: str, accent: str) -> None:
    drifted = conf.get("drifted", False)
    p_val = conf.get("p_value", 1.0)
    mean_ref = conf.get("mean_reference", 0.0)
    mean_cur = conf.get("mean_current", 0.0)
    delta = conf.get("delta", 0.0)
    status_color = theme.DANGER if drifted else theme.SUCCESS
    status_label = "DRIFTED" if drifted else "STABLE"
    delta_sign = "▲" if delta >= 0 else "▼"

    st.markdown(
        f'<div class="cs-panel__label" style="color:{accent};margin-bottom:0.4rem">'
        f"{task_label}</div>",
        unsafe_allow_html=True,
    )
    theme.sonar_card(
        [
            ("Reference", f"{mean_ref:.3f}", theme.TEXT_BRIGHT),
            ("Production", f"{mean_cur:.3f}", theme.TEXT_BRIGHT),
            ("Delta", f"{delta_sign} {abs(delta):.3f}", status_color),
            ("KS p-value", f"{p_val:.4f}", theme.TEXT_BRIGHT),
            ("Status", status_label, status_color),
        ],
        accent=accent,
    )


_render_conf_drift(conf_h, "Reef Health", theme.AQUA)
theme.spacer("0.7rem")
_render_conf_drift(conf_r, "Restoration Suitability", theme.CORAL)

st.divider()

# ---------------------------------------------------------------------------
# HTML report links
# ---------------------------------------------------------------------------

theme.section(
    "Evidently HTML Reports",
    "Full interactive Evidently reports (if generated with the CLI). "
    "Open these files in a browser for detailed per-feature visualisations.",
    kicker="Deep dive",
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
