"""
Page 6 — Model Performance.

Reads existing evaluation JSON files (models/evaluation_health.json and
models/evaluation_restoration.json) to display model comparison tables and
per-class performance metrics.

No models are loaded, trained or registered by this page.
All metrics are from synthetic data.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard import theme
from src.dashboard.components import (
    HEALTH_COLORS,
    RESTORATION_COLORS,
    render_sidebar,
    set_page,
)
from src.dashboard.data_loader import load_evaluation
from src.dashboard.viz import mountain_chart, sonar_wireframe

set_page("Model Performance")

render_sidebar(show_region_filter=False)

theme.page_header(
    "Model Performance",
    "Cross-validation and held-out test metrics for all trained algorithms. "
    "The champion model is selected by cross-validated macro F1.",
    eyebrow="Evaluation",
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
    champion_metrics = models.get(best_name, {})
    champion_display = ALGO_DISPLAY.get(best_name, best_name)

    # ---------- Champion headline ----------
    theme.stat_row(
        [
            {
                "label": "Champion",
                "value": champion_display,
                "caption": "selected by CV macro F1",
                "accent": theme.AQUA,
                "compact": True,
            },
            {
                "label": "CV macro F1",
                "value": f"{champion_metrics.get('cv_macro_f1_mean', 0):.4f}",
                "caption": f"± {champion_metrics.get('cv_macro_f1_std', 0):.4f} (5-fold)",
                "accent": theme.TURQUOISE,
            },
            {
                "label": "Test macro F1",
                "value": f"{champion_metrics.get('test_macro_f1', 0):.4f}",
                "caption": "held-out split",
                "accent": theme.CYAN_SOFT,
            },
            {
                "label": "Test accuracy",
                "value": f"{champion_metrics.get('test_accuracy', 0):.4f}",
                "caption": "reported for verification only",
                "accent": theme.CORAL_SOFT,
            },
        ]
    )

    # ---------- Comparison table ----------
    theme.section("Algorithm Comparison", "All trained models for this task.", kicker="Leaderboard")

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

    df_cmp = pd.DataFrame(rows).sort_values("CV Macro F1", ascending=False)
    st.dataframe(
        df_cmp.style.apply(
            lambda row: [
                f"background-color: {theme.SUCCESS}1f;" if v == "Yes" else "" for v in row
            ],
            axis=1,
        ),
        use_container_width=True,
        hide_index=True,
    )

    # ---------- CV macro F1 as a mountain range ----------
    #
    # The selection metric, drawn as layered topography instead of bars.  Each
    # summit sits at exactly the algorithm's CV macro F1: the translucent
    # ridges behind it are fixed fractions of that same number, and the value
    # printed above the peak, the hover readout and the leaderboard table above
    # are all the same figure straight out of models/evaluation_*.json.  The
    # dotted whisker through the summit is the 5-fold standard deviation.
    st.plotly_chart(
        mountain_chart(
            list(df_cmp["Algorithm"]),
            [float(v) for v in df_cmp["CV Macro F1"]],
            value_name="CV Macro F1",
            errors=[float(v) for v in df_cmp["CV Macro F1 (std)"]],
            error_name="± std (5-fold)",
            highlight=ALGO_DISPLAY.get(best_name, best_name),
            y_range=(0, 1),
            title="Cross-Validated Macro F1 (5-fold) — selection metric",
            height=440,
        ),
        use_container_width=True,
        key=f"cv-mountain-{task}",
    )
    st.caption(
        "Peak height is the exact cross-validated macro F1; the layers behind "
        "each ridge are decorative depth scaled from that same value."
    )

    # ---------- Algorithm × metric surface ----------
    #
    # A genuine 2-D matrix (three algorithms × six metrics), so the sonar
    # wireframe is the right member of the system here — every node is a real
    # cell of the leaderboard table, not a synthesised third dimension.
    metric_columns = [
        "CV Macro F1",
        "CV Bal. Accuracy",
        "Test Macro F1",
        "Test Bal. Accuracy",
        "Test Accuracy",
        "Test Weighted F1",
    ]
    st.plotly_chart(
        sonar_wireframe(
            metric_columns,
            list(df_cmp["Algorithm"]),
            [[float(row[col]) for col in metric_columns] for _, row in df_cmp.iterrows()],
            value_name="Score",
            value_format=".4f",
            x_title="Metric",
            y_title="Algorithm",
            z_title="Score",
            title="Algorithm × metric surface",
            height=520,
        ),
        use_container_width=True,
        key=f"algo-metric-surface-{task}",
    )

    # ---------- Champion per-class results ----------
    per_class = champion_metrics.get("test_per_class", {})

    if per_class:
        theme.section(
            f"Champion ({champion_display}) — Per-Class Results",
            "Precision, recall and F1 for every label on the held-out test split.",
            kicker="Breakdown",
        )

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

        # Exact per-class F1 in the semantic class colours, as a compact
        # instrument readout beside the charts.
        theme.sonar_card(
            [
                (
                    cls.replace("_", " ").title(),
                    f"{metrics.get('f1', 0):.4f}",
                    class_colors.get(cls, theme.AQUA),
                )
                for cls, metrics in per_class.items()
            ],
            accent=theme.AQUA,
        )

        # Per-class F1 as a smaller mountain range — same guarantee: each
        # summit is the exact test-set F1 for that class.
        st.plotly_chart(
            mountain_chart(
                list(df_cls["Class"]),
                [float(v) for v in df_cls["F1"]],
                value_name="Test F1",
                y_range=(0, 1),
                title=f"{champion_display} — Per-Class F1 (test set)",
                height=380,
            ),
            use_container_width=True,
            key=f"per-class-mountain-{task}",
        )

        # Class × metric surface: precision / recall / F1 per label.
        st.plotly_chart(
            sonar_wireframe(
                ["Precision", "Recall", "F1"],
                list(df_cls["Class"]),
                [
                    [float(row["Precision"]), float(row["Recall"]), float(row["F1"])]
                    for _, row in df_cls.iterrows()
                ],
                value_name="Score",
                value_format=".4f",
                x_title="Metric",
                y_title="Class",
                z_title="Score",
                title=f"{champion_display} — per-class precision / recall / F1 surface",
                height=480,
            ),
            use_container_width=True,
            key=f"per-class-surface-{task}",
        )

    st.caption(
        f"Selection: {champion_display} selected by CV macro F1 = "
        f"{champion_metrics.get('cv_macro_f1_mean', 0):.4f}. "
        "All metrics on synthetic data."
    )


# ---------------------------------------------------------------------------
# Reef health
# ---------------------------------------------------------------------------

theme.section("Reef Health Task", kicker="Task 1")
_render_task("health", "logistic_regression", HEALTH_COLORS)

st.divider()

# ---------------------------------------------------------------------------
# Restoration suitability
# ---------------------------------------------------------------------------

theme.section("Restoration Suitability Task", kicker="Task 2")
_render_task("restoration", "xgboost", RESTORATION_COLORS)
