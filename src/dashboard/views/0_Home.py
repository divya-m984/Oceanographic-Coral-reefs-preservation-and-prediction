"""
src/dashboard/views/0_Home.py — the landing.

Two parts, in this order:

1. **The first viewport.**  One screen tall, with licensed underwater footage
   filling it edge to edge and the copy laid over the footage — no card, no
   bordered hero, no rectangle of ocean inside a dark dashboard.  Composition:
   page index, cinematic title, lede, rule and run metadata on the left; a
   system fact bottom-left; one translucent feature card bottom-right.

2. **The dashboard.**  Everything that was already here — dataset statistics,
   objective, champion models, pipeline state, coverage — restyled as
   translucent panels over the now-quiet backdrop.

This is the only page that loads video (``motion=True``).  Every other view
takes the still stage; see ``src/dashboard/cinema.py``.

SYNTHETIC-DATA DISCLAIMER
--------------------------
All observations shown here are computer-generated.
"""

from __future__ import annotations

import streamlit as st

from src.dashboard import cinema, theme
from src.dashboard.components import (
    HEALTH_COLORS,
    REGIONS,
    RESTORATION_COLORS,
    render_disclaimer_banner,
    render_sidebar,
    set_page,
)
from src.dashboard.data_loader import load_observations

STAGE_MODE = set_page("Home", motion=True)

render_sidebar(show_region_filter=False)

# ---------------------------------------------------------------------------
# 1. The first viewport
# ---------------------------------------------------------------------------

try:
    df = load_observations()
    n_obs = len(df)
    n_regions = df["region"].nunique()
    n_health = df["reef_health"].nunique()
    n_rest = df["restoration_suitability"].nunique()
    n_train = int(n_obs * 0.8)
    n_test = n_obs - n_train
except Exception:
    n_obs = n_regions = n_health = n_rest = n_train = n_test = 0

cinema.landing(
    index="01 / 09",
    title_lines=("OCEAN", "INTELLIGENCE"),
    lede=(
        "Sonar-driven habitat intelligence for reef health, restoration suitability "
        "and marine ecosystem monitoring — a complete, reproducible MLOps pipeline "
        "from data generation through to monitored deployment."
    ),
    meta=(
        ("Regions", f"{n_regions or 4} Indian reef zones"),
        ("Corpus", f"{n_obs:,} synthetic observations" if n_obs else "Synthetic corpus"),
        ("Serving", "FastAPI · MLflow registry"),
    ),
)

_cta_left, _cta_right, _ = st.columns([1, 1, 3], gap="small")
with _cta_left:
    st.page_link("views/5_Predict.py", label="Run a prediction", icon=":material/play_arrow:")
with _cta_right:
    st.page_link("views/2_Reef_Map.py", label="Explore the reef map", icon=":material/map:")

# Bottom band of the first viewport: a system fact at the left, the feature card
# at the right.  Kept in normal flow rather than absolutely positioned, so it
# cannot overlap the copy on a short window.
#
# NOTE: there is deliberately no wrapper <div> around this row.  Streamlit
# renders every element into its own container, so an opening tag emitted by one
# st.markdown call is closed by the parser at the end of that container and
# cannot wrap the blocks that follow.  The first viewport gets its height from
# .cs-landing itself, which IS a single block.
_fact, _spacer, _card = st.columns([5, 2, 4], gap="large")

with _fact:
    cinema.footnote(
        "Why sonar first",
        "Acoustic backscatter, rugosity and acoustic complexity describe reef "
        "structure directly, so a survey can assess substrate condition before "
        "any in-situ biological sampling takes place.",
    )

with _card:
    cinema.feature_card(
        label="Live reef intelligence",
        rows=(
            ("Reef Health", "Logistic Regression v1"),
            ("Restoration", "XGBoost v1"),
        ),
    )
    st.page_link(
        "views/5_Predict.py",
        label="Open prediction console",
        icon=":material/arrow_forward:",
    )

cinema.scroll_cue()

# ---------------------------------------------------------------------------
# 2. The dashboard
# ---------------------------------------------------------------------------

st.error(
    "**SYNTHETIC DATA ONLY** — All observations in this dashboard are computer-generated "
    "for academic demonstration purposes. These are not real ocean sensor readings."
)

theme.section(
    "Dataset at a Glance",
    "Synthetic sonar and environmental observations across four Indian reef prototype zones.",
    kicker="Corpus",
)

theme.stat_row(
    [
        {
            "label": "Observations",
            "value": f"{n_obs:,}",
            "caption": "synthetic records",
            "accent": theme.AQUA,
        },
        {
            "label": "Regions",
            "value": f"{n_regions}",
            "caption": "prototype reef zones",
            "accent": theme.TURQUOISE,
        },
        {
            "label": "Health classes",
            "value": f"{n_health}",
            "caption": "reef condition labels",
            "accent": theme.CYAN_SOFT,
        },
        {
            "label": "Restoration classes",
            "value": f"{n_rest}",
            "caption": "suitability labels",
            "accent": theme.CORAL_SOFT,
        },
        {
            "label": "Train / test",
            "value": f"{n_train:,} / {n_test:,}",
            "caption": "80 / 20 stratified split",
            "accent": theme.CORAL,
            "compact": True,
        },
    ]
)

left, right = st.columns([3, 2], gap="large")

with left:
    theme.section("Project Objective", kicker="What this is")
    st.markdown(
        """
        CoralSense is a proof-of-concept MLOps platform that demonstrates how
        acoustic sonar data combined with environmental sensor readings can be used
        to classify coral-reef habitat condition and assess restoration suitability.

        **Two classification tasks are supported:**

        - **Reef Health** — Assigns each observation to one of four condition classes:
          *Healthy*, *Stressed*, *Bleached*, or *Severely Degraded*.
        - **Restoration Suitability** — Classifies whether a reef site is *Suitable*,
          *Moderately Suitable*, or *Unsuitable* for intervention.

        **Sonar-first design:**
        The platform treats acoustic backscatter, rugosity index, and acoustic
        complexity index as primary structural indicators of substrate condition.
        Environmental parameters (temperature, pH, dissolved oxygen, turbidity)
        serve as complementary ecological indicators. This distinction reflects
        how sonar surveys are typically used in reef mapping before in-situ
        biological surveys are conducted.
        """
    )

    theme.section("Classification Labels", kicker="Targets")
    hcol, rcol = st.columns(2)
    with hcol:
        badges = "".join(
            theme.badge(cls.replace("_", " ").title(), HEALTH_COLORS[cls])
            for cls in ("healthy", "stressed", "bleached", "severely_degraded")
        )
        theme.panel(
            f'<div style="display:flex;flex-wrap:wrap;gap:0.45rem">{badges}</div>',
            label="Reef health",
            accent=theme.AQUA,
        )
    with rcol:
        badges = "".join(
            theme.badge(cls.replace("_", " ").title(), RESTORATION_COLORS[cls])
            for cls in ("suitable", "moderately_suitable", "unsuitable")
        )
        theme.panel(
            f'<div style="display:flex;flex-wrap:wrap;gap:0.45rem">{badges}</div>',
            label="Restoration suitability",
            accent=theme.CORAL,
        )

with right:
    theme.section("Champion Models", kicker="In production")
    theme.panel(
        "CV macro-F1 <b>0.761</b> &nbsp;·&nbsp; Test macro-F1 <b>0.787</b>",
        label="Reef health",
        title="Logistic Regression v1",
        accent=theme.AQUA,
    )
    theme.spacer("0.7rem")
    theme.panel(
        "CV macro-F1 <b>0.791</b> &nbsp;·&nbsp; Test macro-F1 <b>0.803</b>",
        label="Restoration suitability",
        title="XGBoost v1",
        accent=theme.CORAL,
    )

    theme.section("MLOps Pipeline", kicker="Lifecycle")
    pipeline_steps = [
        ("Data Generation", "M2", theme.SUCCESS),
        ("Schema Validation", "M3", theme.SUCCESS),
        ("Preprocessing", "M4", theme.SUCCESS),
        ("Model Training", "M5", theme.SUCCESS),
        ("Model Registry", "M6", theme.SUCCESS),
        ("DVC Pipeline", "M7", theme.SUCCESS),
        ("CI / CD", "M8", theme.SUCCESS),
        ("FastAPI Serving", "M9", theme.SUCCESS),
        ("Dashboard", "M10", theme.AQUA),
        ("Drift Monitoring", "M11", theme.TEXT_DIM),
        ("Docker Deploy", "M12", theme.TEXT_DIM),
    ]
    rows = []
    for step, milestone, colour in pipeline_steps:
        icon = "✓" if colour == theme.SUCCESS else ("●" if colour == theme.AQUA else "○")
        rows.append(
            f'<div style="display:flex;align-items:center;gap:0.6rem;'
            f'padding:0.22rem 0;font-size:0.86rem">'
            f'<span style="color:{colour};width:14px">{icon}</span>'
            f'<span style="flex:1;color:{theme.TEXT}">{step}</span>'
            f'<span style="color:{theme.TEXT_DIM};font-size:0.74rem">{milestone}</span>'
            f"</div>"
        )
    theme.panel("".join(rows), accent=theme.TURQUOISE)

    theme.section("Covered Reef Regions", kicker="Coverage")
    region_rows = "".join(
        f'<div style="display:flex;align-items:center;gap:0.6rem;'
        f'padding:0.22rem 0;font-size:0.88rem;color:{theme.TEXT}">'
        f'<span style="color:{theme.CYAN_SOFT}">◆</span>{region}</div>'
        for region in REGIONS
    )
    theme.panel(region_rows, accent=theme.CYAN_SOFT)

theme.spacer("2rem")

render_disclaimer_banner()

# The attribution both background clips are licensed under (CC BY).
cinema.media_credits()
