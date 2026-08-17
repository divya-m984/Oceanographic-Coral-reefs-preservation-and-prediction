"""
Page 3 — Habitat Health Analysis.

Visualises the distribution of reef-health classes and the relationship between
environmental/sonar features and health condition.

Sonar structural features (backscatter, rugosity, ACI) are clearly distinguished
from ecological/environmental indicators.

Visual language (src/dashboard/viz)
-----------------------------------
* class balance        → mountain ridge, summit = exact observation count
* composition by region→ layered streamgraph, band thickness = exact count
* feature by class     → distribution mountains (histograms, unsmoothed)
* region × class       → sonar wireframe over the real contingency matrix
* depth × temperature  → bathymetric contour of mean bleaching, with per-cell
                         sample counts in the hover

No aggregate on this page is smoothed, resampled or interpolated; empty bins
stay empty. All data is synthetic.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard import theme
from src.dashboard.components import (
    HEALTH_COLORS,
    render_sidebar,
    set_page,
)
from src.dashboard.data_loader import load_observations
from src.dashboard.viz import (
    bathymetric_contour,
    mountain_chart,
    ridgeline_chart,
    sonar_wireframe,
    stream_chart,
)

set_page("Habitat Health")

filters = render_sidebar(show_region_filter=True)
selected_regions = filters["selected_regions"]

theme.page_header(
    "Habitat Health Analysis",
    "Distribution and environmental correlates of reef-health classifications "
    "across synthetic sonar and sensor observations.",
    eyebrow="Condition analysis",
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

HEALTH_ORDER = ["healthy", "stressed", "bleached", "severely_degraded"]
present_classes = [cls for cls in HEALTH_ORDER if (df["reef_health"] == cls).any()]

# ---------------------------------------------------------------------------
# Class distribution
# ---------------------------------------------------------------------------

theme.section("Health Class Distribution", kicker="Overview")

counts = df["reef_health"].value_counts()

col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.plotly_chart(
        mountain_chart(
            [cls.replace("_", " ").title() for cls in HEALTH_ORDER],
            [int(counts.get(cls, 0)) for cls in HEALTH_ORDER],
            value_name="Observations",
            digits=0,
            title="Observation count by health class",
            height=380,
        ),
        use_container_width=True,
    )

with col_right:
    regions = sorted(df["region"].unique())
    region_class = (
        df.groupby(["region", "reef_health"]).size().unstack(fill_value=0).reindex(regions)
    )
    st.plotly_chart(
        stream_chart(
            regions,
            {
                cls: [int(region_class.get(cls, {}).get(region, 0)) for region in regions]
                for cls in present_classes
            },
            keys=present_classes,
            colors=HEALTH_COLORS,
            value_name="Observations",
            title="Health composition by region",
            height=380,
        ),
        use_container_width=True,
    )

theme.sonar_card(
    [
        (
            cls.replace("_", " ").title(),
            f"{int(counts.get(cls, 0)):,}",
            HEALTH_COLORS.get(cls, theme.AQUA),
        )
        for cls in HEALTH_ORDER
    ]
    + [("Total", f"{len(df):,}", theme.FOAM)],
    accent=theme.AQUA,
)

# ---------------------------------------------------------------------------
# Environmental indicators by health class
# ---------------------------------------------------------------------------

theme.section(
    "Environmental Indicators by Health Class",
    "In-situ water quality and biological condition as distribution mountains — "
    "each ridge is a histogram of the real rows, with the exact median marked.",
    kicker="Ecological signal",
)

env_features = [
    ("water_temperature_c", "Water Temperature (°C)"),
    ("ph", "pH"),
    ("dissolved_oxygen_mg_l", "Dissolved Oxygen (mg/L)"),
    ("turbidity_ntu", "Turbidity (NTU)"),
    ("coral_cover_percentage", "Coral Cover (%)"),
    ("bleaching_percentage", "Bleaching (%)"),
]


def _series_by_class(feature: str) -> dict[str, list[float]]:
    """Return ``{health class: raw values}`` — no aggregation, no filtering."""
    return {cls: df.loc[df["reef_health"] == cls, feature].tolist() for cls in present_classes}


def _render_ridge_grid(features: list[tuple[str, str]], digits: int) -> None:
    for i in range(0, len(features), 2):
        col_a, col_b = st.columns(2, gap="large")
        for col, (feat, label) in zip([col_a, col_b], features[i : i + 2], strict=False):
            with col:
                st.plotly_chart(
                    ridgeline_chart(
                        present_classes,
                        _series_by_class(feat),
                        colors=HEALTH_COLORS,
                        axis_title=label,
                        title=label,
                        height=330,
                        digits=digits,
                    ),
                    use_container_width=True,
                    key=f"ridge-{feat}",
                )


_render_ridge_grid(env_features, digits=2)

# ---------------------------------------------------------------------------
# Sonar / structural indicators
# ---------------------------------------------------------------------------

theme.section(
    "Sonar and Structural Indicators by Health Class",
    "Acoustic backscatter, rugosity and acoustic complexity index are the primary "
    "structural features derived from sonar surveys.",
    kicker="Acoustic signal",
)

sonar_features = [
    ("sonar_backscatter", "Sonar Backscatter (dB)"),
    ("rugosity_index", "Rugosity Index"),
    ("acoustic_complexity_index", "Acoustic Complexity Index"),
    ("hard_substrate_percentage", "Hard Substrate (%)"),
]

_render_ridge_grid(sonar_features, digits=3)

# ---------------------------------------------------------------------------
# Region × health as a sonar surface
# ---------------------------------------------------------------------------

theme.section(
    "Health Class by Region",
    "The region × class contingency table read as a sonar return. Every node is "
    "an exact observation count; rotate the scene to compare ridges.",
    kicker="Geography",
)

matrix = [
    [int(region_class.get(cls, {}).get(region, 0)) for cls in present_classes] for region in regions
]
st.plotly_chart(
    sonar_wireframe(
        [cls.replace("_", " ").title() for cls in present_classes],
        regions,
        matrix,
        value_name="Observations",
        value_format=",.0f",
        x_title="Health class",
        y_title="Region",
        z_title="Observations",
        height=520,
    ),
    use_container_width=True,
)

# ---------------------------------------------------------------------------
# Environmental surface: depth × temperature → mean bleaching
# ---------------------------------------------------------------------------

theme.section(
    "Bleaching Surface — Depth × Water Temperature",
    "Mean bleaching percentage in each depth/temperature cell, drawn as depth "
    "contours. Cells with no observations are left blank rather than filled in; "
    "hover shows the cell mean and how many rows it was computed from.",
    kicker="Environmental surface",
)

DEPTH_BINS = 10
TEMP_BINS = 12

depth_cut = pd.cut(df["depth_m"], bins=DEPTH_BINS)
temp_cut = pd.cut(df["water_temperature_c"], bins=TEMP_BINS)
grouped = df.groupby([depth_cut, temp_cut], observed=False)["bleaching_percentage"]
cell_mean = grouped.mean().unstack()
cell_count = grouped.count().unstack()

depth_labels = [round(interval.mid, 1) for interval in cell_mean.index]
temp_labels = [round(interval.mid, 1) for interval in cell_mean.columns]

# A cell backed by fewer than five observations is reported as missing: its
# mean would be noise dressed up as a surface.
MIN_CELL_N = 5
z_values = [
    [
        None
        if (cell_count.iloc[j, i] < MIN_CELL_N or pd.isna(cell_mean.iloc[j, i]))
        else float(cell_mean.iloc[j, i])
        for i in range(len(temp_labels))
    ]
    for j in range(len(depth_labels))
]
z_counts = [
    [int(cell_count.iloc[j, i]) for i in range(len(temp_labels))] for j in range(len(depth_labels))
]

st.plotly_chart(
    bathymetric_contour(
        temp_labels,
        depth_labels,
        z_values,
        counts=z_counts,
        x_title="Water temperature (°C)",
        y_title="Depth (m)",
        value_name="Mean bleaching (%)",
        value_format=".1f",
        height=470,
    ),
    use_container_width=True,
)
st.caption(
    f"Cells with fewer than {MIN_CELL_N} observations are shown as gaps. "
    "No value is interpolated across a gap."
)

# ---------------------------------------------------------------------------
# Summary statistics table
# ---------------------------------------------------------------------------

theme.section("Summary Statistics by Health Class", kicker="Aggregates")

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
