"""
Page 2 — Interactive Reef Map (bathymetric).

Renderer
--------
Leaflet, via folium / streamlit-folium (see :mod:`src.dashboard.reefmap`).  The
ocean floor is the GEBCO bathymetric WMS; land, boundaries and place labels come
from a packaged Natural Earth 1:50m subset.  There is no street basemap.

Filtering contract
------------------
Every control on this page filters the DataFrame *before* it reaches the map
builder, and the same filtered frame feeds the region summary table below.
The map, the caption count and the table therefore always describe the same
rows.  The controls and their semantics are unchanged from the previous
version of this page:

    colour observations by · health class filter · restoration filter ·
    max points · sidebar region filter · observations-by-region table

Synthetic data only — coordinates are region centroids with Gaussian scatter
and do not represent real survey waypoints.
"""

from __future__ import annotations

import streamlit as st

from src.dashboard import reefmap, theme
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

theme.page_header(
    "Interactive Reef Map",
    "Geographic distribution of synthetic sonar and environmental observations "
    "across four Indian reef prototype zones, over GEBCO ocean-floor bathymetry.",
    eyebrow="Bathymetric survey",
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
    # Controls row — semantics identical to the previous renderer
    # ---------------------------------------------------------------------------

    with theme.card():
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

        if color_mode == "Reef Health":
            color_col = "reef_health"
            color_map = HEALTH_COLORS
        else:
            color_col = "restoration_suitability"
            color_map = RESTORATION_COLORS

        # ------------------------------------------------------------------
        # Render
        #
        # Any failure — folium missing, WMS unreachable, Leaflet unhappy — is
        # contained here.  The page reports it and carries on to the table
        # rather than showing a traceback.
        # ------------------------------------------------------------------
        try:
            from streamlit_folium import st_folium

            fmap, bathymetry_used = reefmap.build_reef_map(
                plot_df,
                color_col=color_col,
                color_map=color_map,
            )
            if not bathymetry_used:
                st.warning(
                    "GEBCO bathymetry service is unreachable — showing the styled "
                    "abyssal background with coastlines, labels and observations "
                    "intact. No street map is substituted.",
                    icon="🌊",
                )
            # returned_objects=[] keeps pan/zoom from triggering a Streamlit
            # rerun; the map stays fully interactive in the browser.
            st_folium(
                fmap,
                height=620,
                use_container_width=True,
                returned_objects=[],
                key=f"reefmap-{color_col}-{max_pts}-{len(plot_df)}",
            )
        except ImportError:
            st.error(
                "The map renderer is not installed. Install the dashboard runtime "
                "(`pip install -r requirements-dashboard.txt`) to enable it."
            )
        except Exception as exc:  # pragma: no cover - defensive, never fatal
            st.error(f"Map could not be rendered: {exc}")

        # Legend for the marker semantics.
        legend_classes = (
            ["healthy", "stressed", "bleached", "severely_degraded"]
            if color_col == "reef_health"
            else ["suitable", "moderately_suitable", "unsuitable"]
        )
        st.markdown(
            '<div style="display:flex;flex-wrap:wrap;gap:0.55rem;margin-top:0.6rem">'
            + "".join(
                theme.badge(cls.replace("_", " ").title(), color_map.get(cls, theme.AQUA))
                for cls in legend_classes
            )
            + "</div>",
            unsafe_allow_html=True,
        )
        theme.attribution(reefmap.ATTRIBUTIONS)

        # Region summary table — same filtered frame as the map.
        theme.section("Observations by Region", kicker="Breakdown")
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
