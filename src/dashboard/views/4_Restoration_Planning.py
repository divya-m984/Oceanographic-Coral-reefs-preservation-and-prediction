"""
Page 4 — Restoration Planning Analysis.

Visualises restoration-suitability patterns across regions and environmental
parameters using synthetic data. The "top suitable observations" are labelled
clearly as model-development records, not real restoration recommendations.

Visual language (src/dashboard/viz)
-----------------------------------
* suitability balance    → mountain ridge, summit = exact observation count
* composition by region  → layered streamgraph, band thickness = exact count
* feature by class       → distribution mountains (histograms, unsmoothed)
* depth × hard substrate → bathymetric contour of the suitable share, with
                           per-cell sample counts in the hover

Every number drawn here is a count or a share computed directly from the
filtered rows; nothing is smoothed and empty cells stay empty.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard import theme
from src.dashboard.components import (
    RESTORATION_COLORS,
    render_sidebar,
    set_page,
)
from src.dashboard.data_loader import load_observations
from src.dashboard.viz import (
    bathymetric_contour,
    mountain_chart,
    ridgeline_chart,
    stream_chart,
)

set_page("Restoration Planning")

filters = render_sidebar(show_region_filter=True)
selected_regions = filters["selected_regions"]

theme.page_header(
    "Restoration Planning Analysis",
    "Distribution and environmental correlates of restoration-suitability "
    "classifications across synthetic sonar and sensor observations.",
    eyebrow="Intervention planning",
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
present_classes = [cls for cls in REST_ORDER if (df["restoration_suitability"] == cls).any()]

# ---------------------------------------------------------------------------
# Suitability distribution
# ---------------------------------------------------------------------------

theme.section("Restoration Suitability Distribution", kicker="Overview")

counts = df["restoration_suitability"].value_counts()
regions = sorted(df["region"].unique())
region_class = (
    df.groupby(["region", "restoration_suitability"]).size().unstack(fill_value=0).reindex(regions)
)

col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.plotly_chart(
        mountain_chart(
            [cls.replace("_", " ").title() for cls in REST_ORDER],
            [int(counts.get(cls, 0)) for cls in REST_ORDER],
            value_name="Observations",
            digits=0,
            title="Observation count by suitability class",
            height=380,
        ),
        use_container_width=True,
    )

with col_right:
    st.plotly_chart(
        stream_chart(
            regions,
            {
                cls: [int(region_class.get(cls, {}).get(region, 0)) for region in regions]
                for cls in present_classes
            },
            keys=present_classes,
            colors=RESTORATION_COLORS,
            value_name="Observations",
            title="Suitability composition by region",
            height=380,
        ),
        use_container_width=True,
    )

theme.sonar_card(
    [
        (
            cls.replace("_", " ").title(),
            f"{int(counts.get(cls, 0)):,}",
            RESTORATION_COLORS.get(cls, theme.AQUA),
        )
        for cls in REST_ORDER
    ]
    + [("Total", f"{len(df):,}", theme.FOAM)],
    accent=theme.CORAL,
)

# ---------------------------------------------------------------------------
# Environmental and structural comparisons
# ---------------------------------------------------------------------------

theme.section(
    "Feature Comparisons by Suitability Class",
    "How each sonar and environmental feature separates the three suitability "
    "labels. Each ridge is a histogram of the real rows; the exact median is "
    "marked on every one.",
    kicker="Drivers",
)

scatter_features = [
    ("depth_m", "Depth (m)"),
    ("hard_substrate_percentage", "Hard Substrate (%)"),
    ("rugosity_index", "Rugosity Index"),
    ("turbidity_ntu", "Turbidity (NTU)"),
    ("water_temperature_c", "Water Temperature (°C)"),
    ("coral_cover_percentage", "Coral Cover (%)"),
]

for i in range(0, len(scatter_features), 2):
    col_a, col_b = st.columns(2, gap="large")
    for col, (feat, label) in zip([col_a, col_b], scatter_features[i : i + 2], strict=False):
        with col:
            st.plotly_chart(
                ridgeline_chart(
                    present_classes,
                    {
                        cls: df.loc[df["restoration_suitability"] == cls, feat].tolist()
                        for cls in present_classes
                    },
                    colors=RESTORATION_COLORS,
                    axis_title=label,
                    title=f"{label} vs Suitability",
                    height=330,
                    digits=2,
                ),
                use_container_width=True,
                key=f"ridge-{feat}",
            )

# ---------------------------------------------------------------------------
# Suitability surface: depth × hard substrate
# ---------------------------------------------------------------------------

theme.section(
    "Suitability Surface — Depth × Hard Substrate",
    "Share of observations labelled *suitable* in each depth/substrate cell, "
    "drawn as bathymetric contours. Sparse cells are left blank rather than "
    "filled in; hover reports the exact share and the cell's sample count.",
    kicker="Environmental surface",
)

DEPTH_BINS = 10
SUBSTRATE_BINS = 12
MIN_CELL_N = 5

depth_cut = pd.cut(df["depth_m"], bins=DEPTH_BINS)
substrate_cut = pd.cut(df["hard_substrate_percentage"], bins=SUBSTRATE_BINS)
is_suitable = (df["restoration_suitability"] == "suitable").astype(float)
grouped = is_suitable.groupby([depth_cut, substrate_cut], observed=False)
cell_share = grouped.mean().unstack()
cell_count = grouped.count().unstack()

depth_labels = [round(interval.mid, 1) for interval in cell_share.index]
substrate_labels = [round(interval.mid, 1) for interval in cell_share.columns]

z_values = [
    [
        None
        if (cell_count.iloc[j, i] < MIN_CELL_N or pd.isna(cell_share.iloc[j, i]))
        else float(cell_share.iloc[j, i])
        for i in range(len(substrate_labels))
    ]
    for j in range(len(depth_labels))
]
z_counts = [
    [int(cell_count.iloc[j, i]) for i in range(len(substrate_labels))]
    for j in range(len(depth_labels))
]

st.plotly_chart(
    bathymetric_contour(
        substrate_labels,
        depth_labels,
        z_values,
        counts=z_counts,
        x_title="Hard substrate (%)",
        y_title="Depth (m)",
        value_name="Share suitable",
        value_format=".3f",
        height=470,
    ),
    use_container_width=True,
)
st.caption(
    f"Cells with fewer than {MIN_CELL_N} observations are shown as gaps. "
    "No value is interpolated across a gap."
)

# ---------------------------------------------------------------------------
# Top prototype "suitable" observations
# ---------------------------------------------------------------------------

theme.section(
    "Top Prototype Suitable Observations",
    "These are synthetic model-development records scored as 'suitable' by "
    "the original label-generation rules. They are NOT real restoration site "
    "recommendations and must not be used as such.",
    kicker="Sample records",
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
