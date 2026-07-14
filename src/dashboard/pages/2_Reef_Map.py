"""
Page 2 — Interactive Reef Map.

Displays geographic positions of synthetic observations across four Indian reef
regions using Plotly's OpenStreetMap-based map (no API key required).

Performance note: at most 2,000 points are rendered by default. Adjust the
sampling slider if needed.
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.dashboard.components import (
    HEALTH_COLORS,
    RESTORATION_COLORS,
    render_sidebar,
    set_page,
)
from src.dashboard.data_loader import load_observations, sample_for_display

set_page("Reef Map")

filters = render_sidebar(show_region_filter=True)
selected_regions = filters["selected_regions"]

st.title("Interactive Reef Map")
st.caption(
    "Geographic distribution of synthetic sonar and environmental observations "
    "across four Indian reef prototype zones. "
    "Map tiles: OpenStreetMap (free, no API key required)."
)

# ---------------------------------------------------------------------------
# Load and filter data
# ---------------------------------------------------------------------------

try:
    df_all = load_observations()
    data_ok = True
except Exception as exc:
    st.error(f"Dataset unavailable: {exc}")
    data_ok = False
    df_all = None

if data_ok:
    df = df_all[df_all["region"].isin(selected_regions)].copy()

    # ---------------------------------------------------------------------------
    # Controls row
    # ---------------------------------------------------------------------------

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 2, 2, 1])

    with ctrl1:
        color_mode = st.selectbox(
            "Colour observations by",
            options=["Reef Health", "Restoration Suitability"],
        )

    with ctrl2:
        health_filter = st.multiselect(
            "Health class filter",
            options=["healthy", "stressed", "bleached", "severely_degraded"],
            default=["healthy", "stressed", "bleached", "severely_degraded"],
        )

    with ctrl3:
        rest_filter = st.multiselect(
            "Restoration filter",
            options=["suitable", "moderately_suitable", "unsuitable"],
            default=["suitable", "moderately_suitable", "unsuitable"],
        )

    with ctrl4:
        max_pts = st.select_slider(
            "Max points",
            options=[500, 1000, 2000, 5000],
            value=2000,
        )

    # Apply class filters
    df = df[df["reef_health"].isin(health_filter)] if health_filter else df
    df = df[df["restoration_suitability"].isin(rest_filter)] if rest_filter else df

    st.caption(f"Showing {min(len(df), max_pts):,} of {len(df):,} filtered observations.")

    if df.empty:
        st.info("No observations match the current filter combination.")
    else:
        plot_df = sample_for_display(df, max_n=max_pts)

        # Build colour map and column based on mode
        if color_mode == "Reef Health":
            color_col = "reef_health"
            color_map = HEALTH_COLORS
            title_suffix = "Reef Health Condition"
        else:
            color_col = "restoration_suitability"
            color_map = RESTORATION_COLORS
            title_suffix = "Restoration Suitability"

        hover_data = {
            "reef_health": True,
            "restoration_suitability": True,
            "depth_m": ":.1f",
            "water_temperature_c": ":.1f",
            "coral_cover_percentage": ":.1f",
            "region": True,
            "latitude": False,
            "longitude": False,
        }

        fig = px.scatter_map(
            plot_df,
            lat="latitude",
            lon="longitude",
            color=color_col,
            color_discrete_map=color_map,
            hover_name="region",
            hover_data=hover_data,
            zoom=4,
            center={"lat": 12.5, "lon": 80.0},
            height=580,
            title=f"Synthetic Observation Map — {title_suffix}",
            map_style="open-street-map",
        )
        fig.update_layout(
            paper_bgcolor="#0a1628",
            plot_bgcolor="#0a1628",
            font_color="#ccd6f6",
            legend=dict(
                bgcolor="#112240",
                bordercolor="#1a3a5c",
                borderwidth=1,
                title_font_color="#00b4d8",
            ),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        fig.update_traces(marker_size=6, marker_opacity=0.75)
        st.plotly_chart(fig, use_container_width=True)

        # Region summary table
        st.subheader("Observations by Region")
        region_summary = (
            df.groupby("region")
            .agg(
                count=("reef_health", "count"),
                healthy_pct=(
                    "reef_health",
                    lambda s: (s == "healthy").mean() * 100,
                ),
                suitable_pct=(
                    "restoration_suitability",
                    lambda s: (s == "suitable").mean() * 100,
                ),
            )
            .reset_index()
            .rename(
                columns={
                    "region": "Region",
                    "count": "Observations",
                    "healthy_pct": "Healthy (%)",
                    "suitable_pct": "Suitable (%)",
                }
            )
        )
        region_summary["Healthy (%)"] = region_summary["Healthy (%)"].round(1)
        region_summary["Suitable (%)"] = region_summary["Suitable (%)"].round(1)
        st.dataframe(region_summary, use_container_width=True, hide_index=True)

    st.caption(
        "Coordinates are synthetic and represent approximate region centroids "
        "with Gaussian scatter. They do not represent real survey waypoints."
    )
