"""
Page 6 — Model Performance.

Reads existing evaluation JSON files (models/evaluation_health.json and
models/evaluation_restoration.json) to display model comparison tables and
per-class performance metrics.

No models are loaded, trained or registered by this page.
All metrics are from synthetic data.
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.dashboard.components import (
    DARK_BLUE,
    HEALTH_COLORS,
    RESTORATION_COLORS,
    TEAL,
    render_sidebar,
    set_page,
)
from src.dashboard.data_loader import load_evaluation

set_page("Model Performance")

render_sidebar(show_region_filter=False)

st.title("Model Performance")
st.caption(
    "Cross-validation and held-out test metrics for all trained algorithms. "
    "Champion model is selected by cross-validated macro F1."
)
st.info(
    "All metrics are computed on **synthetic data** generated for this project. "
    "They do not represent real-world predictive performance on actual reef surveys."
)

# ---------------------------------------------------------------------------
# Why macro F1?
# ---------------------------------------------------------------------------

with st.expander("Why macro F1 is the primary selection metric"):
    st.markdown(
        """
        **Macro F1** averages the per-class F1 score with equal weight for every class,
        regardless of class frequency. This makes it suitable here because:

        - The reef-health dataset has imbalanced classes (stressed >> healthy ≈ severely degraded).
        - Accuracy is misleading for imbalanced classes: a model predicting only "stressed"
          would achieve ~57 % accuracy but be useless.
        - Macro F1 penalises the model equally for poor performance on minority classes
          (e.g. *healthy* or *bleached*), ensuring the champion generalises across all conditions.
        - Cross-validated macro F1 (5-fold CV) is used for selection to avoid overfitting
          to the test split.
        - The test-set macro F1 is reported for verification only; it is never used
          for champion selection.
        """
    )

# ---------------------------------------------------------------------------
# Shared comparison helper
# ---------------------------------------------------------------------------

ALGO_DISPLAY = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


def _render_task(task: str, champion_name: str, class_colors: dict[str, str]) -> None:
    eval_data = load_evaluation(task)
    if eval_data is None:
        st.warning(f"Evaluation file for '{task}' not found. Run the DVC pipeline first.")
        return

    models = eval_data.get("models", {})
    best_name = eval_data.get("best_model_name", champion_name)
    # ---------- Comparison table ----------
    st.subheader("Algorithm Comparison (all trained models)")

    rows = []
    for algo, metrics in models.items():
        rows.append(
            {
                "Algorithm": ALGO_DISPLAY.get(algo, algo),
                "Champion": "Yes" if algo == best_name else "No",
                "CV Macro F1": round(metrics.get("cv_macro_f1_mean", 0), 4),
                "CV Macro F1 (std)": round(metrics.get("cv_macro_f1_std", 0), 4),
                "CV Bal. Accuracy": round(metrics.get("cv_balanced_accuracy_mean", 0), 4),
                "Test Macro F1": round(metrics.get("test_macro_f1", 0), 4),
                "Test Bal. Accuracy": round(metrics.get("test_balanced_accuracy", 0), 4),
                "Test Accuracy": round(metrics.get("test_accuracy", 0), 4),
                "Test Weighted F1": round(metrics.get("test_weighted_f1", 0), 4),
            }
        )

    import pandas as pd

    df_cmp = pd.DataFrame(rows).sort_values("CV Macro F1", ascending=False)
    st.dataframe(
        df_cmp.style.apply(
            lambda row: ["background-color: #1a3a1a;" if v == "Yes" else "" for v in row],
            axis=1,
        ),
        use_container_width=True,
        hide_index=True,
    )

    # ---------- CV F1 bar chart ----------
    fig_cv = px.bar(
        df_cmp,
        x="Algorithm",
        y="CV Macro F1",
        color="Champion",
        color_discrete_map={"Yes": "#2ecc71", "No": TEAL},
        error_y="CV Macro F1 (std)",
        title="Cross-Validated Macro F1 (5-fold) — selection metric",
        labels={"CV Macro F1": "CV Macro F1", "Algorithm": ""},
    )
    fig_cv.update_layout(
        paper_bgcolor=DARK_BLUE,
        plot_bgcolor="#0a1628",
        font_color="#ccd6f6",
        showlegend=True,
        margin=dict(t=50, b=10),
        yaxis=dict(gridcolor="#1a3a5c", range=[0, 1]),
        xaxis=dict(gridcolor="#1a3a5c"),
    )
    st.plotly_chart(fig_cv, use_container_width=True)

    # ---------- Champion per-class results ----------
    champion_metrics = models.get(best_name, {})
    per_class = champion_metrics.get("test_per_class", {})
    champion_display = ALGO_DISPLAY.get(best_name, best_name)

    if per_class:
        st.subheader(f"Champion ({champion_display}) — Per-Class Results")

        cls_rows = []
        for cls, m in per_class.items():
            cls_rows.append(
                {
                    "Class": cls.replace("_", " ").title(),
                    "Precision": round(m.get("precision", 0), 4),
                    "Recall": round(m.get("recall", 0), 4),
                    "F1": round(m.get("f1", 0), 4),
                    "Support": int(m.get("support", 0)),
                }
            )
        df_cls = pd.DataFrame(cls_rows)
        st.dataframe(df_cls, use_container_width=True, hide_index=True)

        # Per-class F1 bar
        fig_cls = px.bar(
            df_cls,
            x="Class",
            y="F1",
            color="Class",
            color_discrete_map={
                cls.replace("_", " ").title(): color for cls, color in class_colors.items()
            },
            title=f"{champion_display} — Per-Class F1 (test set)",
        )
        fig_cls.update_layout(
            paper_bgcolor=DARK_BLUE,
            plot_bgcolor="#0a1628",
            font_color="#ccd6f6",
            showlegend=False,
            margin=dict(t=50, b=10),
            yaxis=dict(gridcolor="#1a3a5c", range=[0, 1]),
            xaxis=dict(gridcolor="#1a3a5c"),
        )
        st.plotly_chart(fig_cls, use_container_width=True)

    st.caption(
        f"Selection: {champion_display} selected by CV macro F1 = "
        f"{champion_metrics.get('cv_macro_f1_mean', 0):.4f}. "
        "All metrics on synthetic data."
    )


# ---------------------------------------------------------------------------
# Reef health
# ---------------------------------------------------------------------------

st.header("Reef Health Task")
_render_task("health", "logistic_regression", HEALTH_COLORS)

st.divider()

# ---------------------------------------------------------------------------
# Restoration suitability
# ---------------------------------------------------------------------------

st.header("Restoration Suitability Task")
_render_task("restoration", "xgboost", RESTORATION_COLORS)
