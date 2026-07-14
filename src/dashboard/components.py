"""
src/dashboard/components.py — Shared UI components for the CoralSense dashboard.

Provides:
- Colour maps for reef-health and restoration-suitability classes.
- A shared sidebar rendered consistently on every page.
- Disclaimer banners.
- Prediction-result rendering helpers.
- Payload-building helper that explicitly excludes target labels.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

REGIONS: tuple[str, ...] = (
    "Lakshadweep",
    "Gulf of Mannar",
    "Gulf of Kutch",
    "Andaman and Nicobar Islands",
)

HEALTH_CLASSES: tuple[str, ...] = (
    "healthy",
    "stressed",
    "bleached",
    "severely_degraded",
)

RESTORATION_CLASSES: tuple[str, ...] = (
    "suitable",
    "moderately_suitable",
    "unsuitable",
)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

# Marine UI palette
NAVY = "#0a1628"
DARK_BLUE = "#112240"
TEAL = "#00b4d8"
CYAN = "#90e0ef"
CORAL = "#ff6b6b"

# Semantic class colours
HEALTH_COLORS: dict[str, str] = {
    "healthy": "#2ecc71",
    "stressed": "#f39c12",
    "bleached": "#e67e22",
    "severely_degraded": "#e74c3c",
}

RESTORATION_COLORS: dict[str, str] = {
    "suitable": "#2ecc71",
    "moderately_suitable": "#f39c12",
    "unsuitable": "#e74c3c",
}

# Plotly discrete colour sequence for region facets
REGION_COLORS: dict[str, str] = {
    "Lakshadweep": "#00b4d8",
    "Gulf of Mannar": "#48cae4",
    "Gulf of Kutch": "#0096c7",
    "Andaman and Nicobar Islands": "#023e8a",
}

# ---------------------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------------------

DISCLAIMER_SHORT = (
    "**Synthetic data only.** Predictions must not guide real conservation decisions."
)

DISCLAIMER_FULL = (
    "All observations used in this dashboard are computer-generated using a "
    "documented synthetic data generator. No real marine sensor data is displayed. "
    "Predictions produced by CoralSense models are for demonstration and academic "
    "purposes only and must **not** be used to guide actual coral-reef conservation "
    "or marine management decisions."
)

DASHBOARD_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Page configuration helper
# ---------------------------------------------------------------------------


def set_page(title: str, layout: str = "wide") -> None:
    """Call st.set_page_config with consistent project branding."""
    st.set_page_config(
        page_title=f"{title} | CoralSense",
        page_icon=":fish:",
        layout=layout,
    )


# ---------------------------------------------------------------------------
# Shared sidebar
# ---------------------------------------------------------------------------


def render_sidebar(
    *,
    show_region_filter: bool = True,
    show_date_filter: bool = False,
    df: Any = None,
) -> dict[str, Any]:
    """
    Render the standard sidebar for all CoralSense dashboard pages.

    Returns a dict of selected filter values:
        selected_regions: list[str]
        date_range: tuple | None
    """
    from src.dashboard.api_client import APIClient, APIError

    with st.sidebar:
        st.markdown(
            "<div style='text-align:center; padding:0.5rem 0'>"
            "<span style='color:#00b4d8; font-size:1.15rem; font-weight:700'>"
            "CoralSense</span><br>"
            "<span style='color:#90e0ef; font-size:0.75rem'>"
            "Sonar Habitat Intelligence</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        # API status
        st.markdown("**API Status**")
        try:
            client = APIClient()
            health = client.health()
            if health.get("status") == "ok":
                st.success("Connected — models ready")
            else:
                st.warning("Connected — models degraded")
        except APIError:
            st.error("API offline")

        st.divider()

        # Disclaimer
        st.markdown(
            f"<div style='background:#1a2f4e; padding:0.6rem 0.8rem; "
            f"border-radius:6px; border-left:3px solid #ff6b6b; "
            f"font-size:0.78rem; color:#ffd6d6;'>"
            f"{DISCLAIMER_SHORT}"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.divider()

        selected_regions: list[str] = list(REGIONS)
        date_range = None

        if show_region_filter:
            st.markdown("**Region Filter**")
            selected_regions = st.multiselect(
                "Select regions",
                options=list(REGIONS),
                default=list(REGIONS),
                label_visibility="collapsed",
            )
            if not selected_regions:
                st.caption("No region selected — showing all.")
                selected_regions = list(REGIONS)

        if show_date_filter and df is not None and "timestamp" in df.columns:
            st.markdown("**Date Range**")
            ts = df["timestamp"]
            min_date = ts.min().date()
            max_date = ts.max().date()
            date_range = st.date_input(
                "Date range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                label_visibility="collapsed",
            )

        st.divider()
        st.caption(f"Dashboard v{DASHBOARD_VERSION}")
        st.caption("Navigate using the pages above.")

    return {"selected_regions": selected_regions, "date_range": date_range}


# ---------------------------------------------------------------------------
# Disclaimer banners
# ---------------------------------------------------------------------------


def render_disclaimer(expanded: bool = False) -> None:
    """Render the synthetic-data disclaimer as an info callout."""
    with st.expander("Synthetic Data Disclaimer", expanded=expanded):
        st.info(DISCLAIMER_FULL)


# ---------------------------------------------------------------------------
# Prediction result rendering
# ---------------------------------------------------------------------------


def render_prediction(result: dict[str, Any], task_label: str) -> None:
    """
    Render a single PredictionResponse dict as a dashboard card.

    Parameters
    ----------
    result:
        Dict matching PredictionResponse schema from the FastAPI service.
    task_label:
        Human-readable task name, e.g. "Reef Health" or "Restoration Suitability".
    """
    predicted = result.get("predicted_class", "unknown")
    confidence = result.get("confidence", 0.0)
    probabilities = result.get("probabilities", {})
    model_name = result.get("registered_model_name", "—")
    model_version = result.get("model_version", "—")
    model_alias = result.get("model_alias", "—")
    timestamp = result.get("prediction_timestamp", "—")

    # Choose colour map
    if "health" in (result.get("task") or ""):
        color = HEALTH_COLORS.get(predicted, TEAL)
    else:
        color = RESTORATION_COLORS.get(predicted, TEAL)

    st.markdown(
        f"<div style='background:{DARK_BLUE}; border-radius:10px; "
        f"padding:1.2rem; border-left:5px solid {color}; margin-bottom:0.8rem;'>"
        f"<div style='color:{CYAN}; font-size:0.85rem; font-weight:600; "
        f"text-transform:uppercase; letter-spacing:1px;'>{task_label}</div>"
        f"<div style='font-size:1.8rem; font-weight:700; color:{color}; "
        f"margin:0.3rem 0;'>{predicted.replace('_', ' ').title()}</div>"
        f"<div style='color:#90e0ef;'>Confidence: "
        f"<strong>{confidence:.1%}</strong></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Probability bars
    if probabilities:
        st.markdown("**Class Probabilities**")
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        for cls, prob in sorted_probs:
            if "health" in (result.get("task") or ""):
                bar_color = HEALTH_COLORS.get(cls, TEAL)
            else:
                bar_color = RESTORATION_COLORS.get(cls, TEAL)
            st.markdown(
                f"<div style='margin:0.15rem 0;'>"
                f"<span style='color:{CYAN}; font-size:0.85rem; min-width:200px; "
                f"display:inline-block;'>{cls.replace('_', ' ').title()}</span>"
                f"<div style='background:#1a2f4e; border-radius:4px; height:18px; "
                f"width:100%; position:relative; margin-top:2px;'>"
                f"<div style='background:{bar_color}; height:18px; border-radius:4px; "
                f"width:{prob * 100:.1f}%;'></div>"
                f"<span style='position:absolute; right:6px; top:0; font-size:0.75rem; "
                f"line-height:18px; color:#ccd6f6;'>{prob:.3f}</span>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

    # Metadata footer
    st.caption(
        f"Model: {model_name} v{model_version} ({model_alias}) | "
        f"Predicted: {timestamp[:19] if timestamp != '—' else '—'}"
    )


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------

_TARGET_LABELS = frozenset({"reef_health", "restoration_suitability"})


def build_predict_payload(form_values: dict[str, Any]) -> dict[str, Any]:
    """
    Build a POST /predict/both payload from form values.

    Explicitly removes any target label columns so that the API's
    extra='forbid' validation never triggers.
    """
    return {k: v for k, v in form_values.items() if k not in _TARGET_LABELS}
