"""
Page 4 — Restoration Planning Analysis.

Visualises restoration-suitability patterns across regions and environmental
parameters using synthetic data. The "top suitable observations" are labelled
clearly as model-development records, not real restoration recommendations.
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.dashboard.components import (
    DARK_BLUE,
    RESTORATION_COLORS,
    TEAL,
    render_sidebar,
    set_page,
)
from src.dashboard.data_loader import load_observations

set_page("Restoration Planning")

filters = render_sidebar(show_region_filter=True)
selected_regions = filters["selected_regions"]

st.title("Restoration Planning Analysis")
st.caption(
    "Distribution and environmental correlates of restoration-suitability "
    "classifications across synthetic sonar and sensor observations."
)

st.warning(
    "The observations shown below are model-development data generated synthetically. "
    "They do not represent real restoration recommendations or survey findings."
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

try:
    df_all = load_observations()
    data_ok = True
except Exception as exc:
    st.error(f"Dataset unavailable: {exc}")
    data_ok = False

if not data_ok:
    st.stop()

df = df_all[df_all["region"].isin(selected_regions)].copy()

if df.empty:
    st.info("No observations match the selected regions.")
    st.stop()

REST_ORDER = ["suitable", "moderately_suitable", "unsuitable"]
cat_order = {"restoration_suitability": REST_ORDER}

# ---------------------------------------------------------------------------
# Suitability distribution
# ---------------------------------------------------------------------------

st.header("Restoration Suitability Distribution")

col_pie, col_reg = st.columns(2)

with col_pie:
    counts = df["restoration_suitability"].value_counts()
    fig_pie = px.pie(
        values=counts.values,
        names=counts.index,
        color=counts.index,
        color_discrete_map=RESTORATION_COLORS,
        title="Suitability Class Proportions",
        hole=0.45,
    )
    fig_pie.update_layout(
        paper_bgcolor=DARK_BLUE,
        font_color="#ccd6f6",
        legend=dict(bgcolor=DARK_BLUE),
        margin=dict(t=40, b=10),
    )
    fig_pie.update_traces(textinfo="percent+label", textfont_size=12)
    st.plotly_chart(fig_pie, use_container_width=True)

with col_reg:
    region_rest = df.groupby(["region", "restoration_suitability"]).size().reset_index(name="count")
    fig_reg = px.bar(
        region_rest,
        x="region",
        y="count",
        color="restoration_suitability",
        color_discrete_map=RESTORATION_COLORS,
        category_orders=cat_order,
        barmode="stack",
        title="Suitability by Region",
        labels={
            "region": "Region",
            "count": "Observations",
            "restoration_suitability": "Suitability",
        },
    )
    fig_reg.update_layout(
        paper_bgcolor=DARK_BLUE,
        plot_bgcolor="#0a1628",
        font_color="#ccd6f6",
        legend=dict(bgcolor=DARK_BLUE, title_font_color=TEAL),
        margin=dict(t=40, b=10),
        xaxis=dict(gridcolor="#1a3a5c", tickangle=-15),
        yaxis=dict(gridcolor="#1a3a5c"),
    )
    st.plotly_chart(fig_reg, use_container_width=True)

# ---------------------------------------------------------------------------
# Environmental and structural comparisons
# ---------------------------------------------------------------------------

st.header("Feature Comparisons by Suitability Class")

scatter_features = [
    ("depth_m", "Depth (m)", "Depth vs Suitability"),
    ("hard_substrate_percentage", "Hard Substrate (%)", "Substrate vs Suitability"),
    ("rugosity_index", "Rugosity Index", "Rugosity vs Suitability"),
    ("turbidity_ntu", "Turbidity (NTU)", "Turbidity vs Suitability"),
    ("water_temperature_c", "Water Temperature (°C)", "Temperature vs Suitability"),
    ("coral_cover_percentage", "Coral Cover (%)", "Coral Cover vs Suitability"),
]

for i in range(0, len(scatter_features), 2):
    col_a, col_b = st.columns(2)
    for col, (feat, label, title) in zip([col_a, col_b], scatter_features[i : i + 2], strict=False):
        with col:
            fig = px.box(
                df,
                x="restoration_suitability",
                y=feat,
                color="restoration_suitability",
                color_discrete_map=RESTORATION_COLORS,
                category_orders=cat_order,
                title=title,
                labels={"restoration_suitability": "", feat: label},
            )
            fig.update_layout(
                paper_bgcolor=DARK_BLUE,
                plot_bgcolor="#0a1628",
                font_color="#ccd6f6",
                showlegend=False,
                margin=dict(t=40, b=10, l=40, r=10),
                xaxis=dict(gridcolor="#1a3a5c", tickfont_size=10),
                yaxis=dict(gridcolor="#1a3a5c"),
            )
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Top prototype "suitable" observations
# ---------------------------------------------------------------------------

st.header("Top Prototype Suitable Observations")
st.caption(
    "These are synthetic model-development records scored as 'suitable' by "
    "the original label-generation rules. They are NOT real restoration site "
    "recommendations and must not be used as such."
)

suitable_df = (
    df[df["restoration_suitability"] == "suitable"]
    .sort_values("coral_cover_percentage", ascending=False)
    .head(10)[
        [
            "region",
            "latitude",
            "longitude",
            "depth_m",
            "coral_cover_percentage",
            "hard_substrate_percentage",
            "rugosity_index",
            "water_temperature_c",
            "reef_health",
        ]
    ]
    .reset_index(drop=True)
)
suitable_df.index += 1
suitable_df.columns = [
    "Region",
    "Lat",
    "Lon",
    "Depth (m)",
    "Coral Cover (%)",
    "Hard Substrate (%)",
    "Rugosity",
    "Temp (°C)",
    "Reef Health",
]
st.dataframe(suitable_df, use_container_width=True)

st.caption("Sorted by coral cover percentage. All values are synthetic.")
