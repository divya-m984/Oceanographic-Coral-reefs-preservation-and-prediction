"""
src/dashboard/components.py — Shared UI components for the CoralSense dashboard.

Provides:
- Colour maps for reef-health and restoration-suitability classes.
- A shared sidebar rendered consistently on every page.
- Disclaimer banners.
- Prediction-result rendering helpers.
- Payload-building helper that explicitly excludes target labels.

Visual language
---------------
All colours, spacing and CSS live in :mod:`src.dashboard.theme`. This module
consumes those tokens and never hard-codes a hex value of its own, so a change
to the theme propagates everywhere.

``set_page`` also renders the background media stage
(:mod:`src.dashboard.cinema`), which is how every page shares one visual
identity while only the landing pays for video.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.dashboard import cinema, theme

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
#
# Names kept for backwards compatibility with existing page imports; the values
# now come from the central theme so there is one source of truth.
# ---------------------------------------------------------------------------

NAVY = theme.NAVY
DARK_BLUE = theme.SURFACE
TEAL = theme.AQUA
CYAN = theme.CYAN
CORAL = theme.CORAL

# Semantic class colours
HEALTH_COLORS: dict[str, str] = {
    "healthy": theme.SUCCESS,
    "stressed": theme.WARNING,
    "bleached": theme.CORAL_SOFT,
    "severely_degraded": theme.DANGER,
}

RESTORATION_COLORS: dict[str, str] = {
    "suitable": theme.SUCCESS,
    "moderately_suitable": theme.WARNING,
    "unsuitable": theme.DANGER,
}

# Plotly discrete colour sequence for region facets
REGION_COLORS: dict[str, str] = {
    "Lakshadweep": theme.AQUA,
    "Gulf of Mannar": theme.CYAN_SOFT,
    "Gulf of Kutch": theme.TURQUOISE,
    "Andaman and Nicobar Islands": theme.TEAL,
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


def set_page(
    title: str, layout: str = "wide", *, icon: str = ":fish:", motion: bool = False
) -> str:
    """
    Set up one page: inject the stylesheet and render the background stage.

    Deliberately does **not** call ``st.set_page_config``.  Under
    ``st.navigation`` the entrypoint (``src/dashboard/app.py``) owns the page
    config, and calling it again from a view raises.  ``title``, ``layout`` and
    ``icon`` are kept in the signature because every view already passes them
    and because the page title now comes from the matching ``st.Page``.

    ``motion`` is the media budget, and only the landing sets it.  Analytical
    pages get the still stage so that Plotly, Leaflet and a video decoder are
    never competing for the same frame.

    Returns the stage mode actually rendered (``"video"``/``"still"``/
    ``"gradient"``) so a page — and the tests — can assert what was loaded.

    Order matters: the stage must be the FIRST element in the main block,
    because the stylesheet promotes every following block to its own positioned
    layer.  That, plus DOM order, is what keeps the media behind all content
    without any page needing a z-index of its own.
    """
    theme.inject_theme()
    return cinema.stage(motion=motion)


# ---------------------------------------------------------------------------
# Shared sidebar
# ---------------------------------------------------------------------------


def _render_api_status() -> None:
    """Render the API connection indicator as a themed status pill."""
    from src.dashboard.api_client import APIClient, APIError

    try:
        health = APIClient().health()
        if health.get("status") == "ok":
            colour, label, detail, live = theme.SUCCESS, "API connected", "Models ready", True
        else:
            colour, label, detail, live = theme.WARNING, "API degraded", "Models unavailable", True
    except APIError:
        colour, label, detail, live = (
            theme.DANGER,
            "API offline",
            "Start the FastAPI service",
            False,
        )

    pulse = " cs-status__dot--live" if live else ""
    st.markdown(
        f'<div class="cs-status">'
        f'<span class="cs-status__dot{pulse}" style="background:{colour};'
        f'box-shadow:0 0 0 3px {colour}33"></span>'
        f"<span>"
        f'<span class="cs-status__text" style="color:{colour}">{label}</span><br>'
        f'<span class="cs-status__sub">{detail}</span>'
        f"</span></div>",
        unsafe_allow_html=True,
    )


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
    # Nothing to show means no sidebar at all: seven of the ten pages take no
    # filters, and an empty rail is what made the app read as academic.
    if not show_region_filter and not (show_date_filter and df is not None):
        return {"selected_regions": list(REGIONS), "date_range": None}

    with st.sidebar:
        st.markdown('<div class="cs-side-label">Filters</div>', unsafe_allow_html=True)

        selected_regions: list[str] = list(REGIONS)
        date_range = None

        if show_region_filter:
            theme.spacer("1rem")
            st.markdown('<div class="cs-side-label">Region filter</div>', unsafe_allow_html=True)
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
            theme.spacer("0.6rem")
            st.markdown('<div class="cs-side-label">Date range</div>', unsafe_allow_html=True)
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

        theme.spacer("1.1rem")
        st.markdown(
            f'<div style="border-top:1px solid {theme.BORDER_SOFT};padding-top:0.7rem;'
            f'font-size:0.72rem;color:{theme.TEXT_DIM};line-height:1.6">'
            f"Dashboard v{DASHBOARD_VERSION}<br>Synthetic data · academic demonstration"
            f"</div>",
            unsafe_allow_html=True,
        )

    return {"selected_regions": selected_regions, "date_range": date_range}


# ---------------------------------------------------------------------------
# Disclaimer banners
# ---------------------------------------------------------------------------


def render_disclaimer(expanded: bool = False) -> None:
    """Render the synthetic-data disclaimer as an info callout."""
    with st.expander("Synthetic Data Disclaimer", expanded=expanded):
        st.info(DISCLAIMER_FULL)


def render_disclaimer_banner() -> None:
    """Render the full disclaimer as a coral-accented panel."""
    st.markdown(
        f'<div style="background:linear-gradient(150deg,{theme.CORAL}1c,{theme.CORAL}08);'
        f"border:1px solid {theme.CORAL}3d;border-left:3px solid {theme.CORAL};"
        f'border-radius:{theme.RADIUS};padding:1.1rem 1.3rem">'
        f'<div style="color:{theme.CORAL};font-size:0.68rem;font-weight:700;'
        f'letter-spacing:0.18em;text-transform:uppercase;margin-bottom:0.45rem">'
        f"Synthetic data disclaimer</div>"
        f'<div style="color:#e8c9bf;font-size:0.85rem;line-height:1.65">{DISCLAIMER_FULL}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


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

    is_health = "health" in (result.get("task") or "")
    colour_map = HEALTH_COLORS if is_health else RESTORATION_COLORS
    colour = colour_map.get(predicted, theme.AQUA)

    conf_pct = max(0.0, min(1.0, float(confidence))) * 100
    stamp = timestamp[:19].replace("T", " ") if timestamp != "—" else "—"

    st.markdown(
        f'<div class="cs-pred" style="--cs-pred-accent:{colour}">'
        f'<div class="cs-pred__inner">'
        f'<div class="cs-pred__task">{task_label}</div>'
        f'<div class="cs-pred__class">{predicted.replace("_", " ").title()}</div>'
        f'<div class="cs-pred__confhead"><span>Confidence</span>'
        f'<span class="cs-pred__confval">{confidence:.1%}</span></div>'
        f'<div class="cs-pred__track"><div class="cs-pred__fill" '
        f'style="width:{conf_pct:.1f}%;background:linear-gradient(90deg,{colour}99,{colour})">'
        f"</div></div>"
        f'<div class="cs-pred__meta">'
        f'<span class="cs-pred__metaitem">Model <b>{model_name}</b></span>'
        f'<span class="cs-pred__metaitem">Version <b>v{model_version}</b></span>'
        f'<span class="cs-pred__metaitem">Alias <b>{model_alias}</b></span>'
        f'<span class="cs-pred__metaitem">Predicted <b>{stamp}</b></span>'
        f"</div></div></div>",
        unsafe_allow_html=True,
    )

    # Probability breakdown, as a small mountain range.
    #
    # Each summit is the exact class probability returned by the API — the same
    # number appears above the peak, in the hover, and in the instrument
    # readout beneath, so the shape never has to be trusted for a value.
    if probabilities:
        theme.spacer("0.9rem")
        st.markdown(
            f'<div class="cs-side-label" style="color:{theme.TEXT_MUTED}">'
            f"Class probabilities</div>",
            unsafe_allow_html=True,
        )
        sorted_probs = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)

        from src.dashboard.viz import mountain_chart

        st.plotly_chart(
            mountain_chart(
                [cls.replace("_", " ").title() for cls, _ in sorted_probs],
                [float(prob) for _, prob in sorted_probs],
                value_name="Probability",
                digits=3,
                y_range=(0, 1),
                highlight=predicted.replace("_", " ").title(),
                highlight_color=colour,
                height=280,
            ),
            use_container_width=True,
            # Both prediction cards render on one page; identical probabilities
            # would otherwise collide on Streamlit's auto-generated element ID.
            key=f"pred-mountain-{task_label}",
        )
        theme.sonar_card(
            [
                (
                    cls.replace("_", " ").title(),
                    f"{float(prob):.3f}",
                    colour_map.get(cls, theme.AQUA),
                )
                for cls, prob in sorted_probs
            ],
            accent=colour,
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
