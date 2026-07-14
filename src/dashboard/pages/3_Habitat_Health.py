"""
Page 3 — Habitat Health Analysis.

Visualises the distribution of reef-health classes and the relationship between
environmental/sonar features and health condition.

Sonar structural features (backscatter, rugosity, ACI) are clearly distinguished
from ecological/environmental indicators.
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.dashboard.components import (
    DARK_BLUE,
    HEALTH_COLORS,
    TEAL,
    render_sidebar,
    set_page,
)
from src.dashboard.data_loader import load_observations

set_page("Habitat Health")

filters = render_sidebar(show_region_filter=True)
selected_regions = filters["selected_regions"]

st.title("Habitat Health Analysis")
st.caption(
    "Distribution and environmental correlates of reef-health classifications "
    "across synthetic sonar and sensor observations."
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
    df_all = None

if not data_ok:
    st.stop()

df = df_all[df_all["region"].isin(selected_regions)].copy()

if df.empty:
    st.info("No observations match the selected regions.")
    st.stop()

# Health class order
HEALTH_ORDER = ["healthy", "stressed", "bleached", "severely_degraded"]
cat_order = {"reef_health": HEALTH_ORDER}

# ---------------------------------------------------------------------------
# Class distribution
# ---------------------------------------------------------------------------

st.header("Health Class Distribution")

col_pie, col_bar = st.columns(2)

with col_pie:
    counts = df["reef_health"].value_counts()
    fig_pie = px.pie(
        values=counts.values,
        names=counts.index,
        color=counts.index,
        color_discrete_map=HEALTH_COLORS,
        title="Health Class Proportions",
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

with col_bar:
    fig_bar = px.histogram(
        df,
        x="reef_health",
        color="reef_health",
        color_discrete_map=HEALTH_COLORS,
        category_orders=cat_order,
        title="Observation Count by Health Class",
        labels={"reef_health": "Health Class", "count": "Count"},
    )
    fig_bar.update_layout(
        paper_bgcolor=DARK_BLUE,
        plot_bgcolor="#0a1628",
        font_color="#ccd6f6",
        showlegend=False,
        margin=dict(t=40, b=10),
        xaxis=dict(gridcolor="#1a3a5c"),
        yaxis=dict(gridcolor="#1a3a5c"),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------------------------------------
# Environmental indicators by health class
# ---------------------------------------------------------------------------

st.header("Environmental Indicators by Health Class")

env_features = [
    ("water_temperature_c", "Water Temperature (°C)"),
    ("ph", "pH"),
    ("dissolved_oxygen_mg_l", "Dissolved Oxygen (mg/L)"),
    ("turbidity_ntu", "Turbidity (NTU)"),
    ("coral_cover_percentage", "Coral Cover (%)"),
    ("bleaching_percentage", "Bleaching (%)"),
]

for i in range(0, len(env_features), 2):
    col_a, col_b = st.columns(2)
    for col, (feat, label) in zip([col_a, col_b], env_features[i : i + 2], strict=False):
        with col:
            fig = px.box(
                df,
                x="reef_health",
                y=feat,
                color="reef_health",
                color_discrete_map=HEALTH_COLORS,
                category_orders=cat_order,
                title=label,
                labels={"reef_health": "", feat: label},
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
# Sonar / structural indicators
# ---------------------------------------------------------------------------

st.header("Sonar and Structural Indicators by Health Class")
st.caption(
    "Acoustic backscatter, rugosity and acoustic complexity index are the primary "
    "structural features derived from sonar surveys."
)

sonar_features = [
    ("sonar_backscatter", "Sonar Backscatter (dB)"),
    ("rugosity_index", "Rugosity Index"),
    ("acoustic_complexity_index", "Acoustic Complexity Index"),
    ("hard_substrate_percentage", "Hard Substrate (%)"),
]

for i in range(0, len(sonar_features), 2):
    col_a, col_b = st.columns(2)
    for col, (feat, label) in zip([col_a, col_b], sonar_features[i : i + 2], strict=False):
        with col:
            fig = px.violin(
                df,
                x="reef_health",
                y=feat,
                color="reef_health",
                color_discrete_map=HEALTH_COLORS,
                category_orders=cat_order,
                box=True,
                title=label,
                labels={"reef_health": "", feat: label},
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
# Region × health breakdown
# ---------------------------------------------------------------------------

st.header("Health Class by Region")

region_health = df.groupby(["region", "reef_health"]).size().reset_index(name="count")

fig_rh = px.bar(
    region_health,
    x="region",
    y="count",
    color="reef_health",
    color_discrete_map=HEALTH_COLORS,
    category_orders={"reef_health": HEALTH_ORDER},
    barmode="group",
    title="Health Class Distribution per Region",
    labels={"region": "Region", "count": "Observations", "reef_health": "Health"},
)
fig_rh.update_layout(
    paper_bgcolor=DARK_BLUE,
    plot_bgcolor="#0a1628",
    font_color="#ccd6f6",
    legend=dict(bgcolor=DARK_BLUE, title_font_color=TEAL),
    margin=dict(t=40, b=10),
    xaxis=dict(gridcolor="#1a3a5c", tickangle=-15),
    yaxis=dict(gridcolor="#1a3a5c"),
)
st.plotly_chart(fig_rh, use_container_width=True)

# ---------------------------------------------------------------------------
# Summary statistics table
# ---------------------------------------------------------------------------

st.header("Summary Statistics by Health Class")

summary = (
    df.groupby("reef_health")[
        [
            "water_temperature_c",
            "ph",
            "dissolved_oxygen_mg_l",
            "coral_cover_percentage",
            "bleaching_percentage",
            "sonar_backscatter",
            "rugosity_index",
        ]
    ]
    .mean()
    .round(3)
)
summary.index.name = "Health Class"
summary.columns = [
    "Temp (°C)",
    "pH",
    "DO (mg/L)",
    "Coral Cover (%)",
    "Bleaching (%)",
    "Backscatter (dB)",
    "Rugosity",
]
st.dataframe(summary, use_container_width=True)
st.caption("All statistics computed on synthetic data.")
