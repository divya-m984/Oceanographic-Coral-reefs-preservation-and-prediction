"""
src/dashboard/theme.py — Centralised design system for the CoralSense dashboard.

This module is the single source of truth for the dashboard's visual language.
It is presentation-only: nothing here reads data, loads a model, calls the API
or touches the filesystem.  It emits CSS and HTML strings, and applies a shared
palette to Plotly figures.

Layers
------
1. Tokens        Colours, radii and shadows as plain Python constants.
2. Global CSS    One injected stylesheet, driven by CSS custom properties that
                 are generated from those tokens (``inject_theme``).  It also
                 carries the rules for the media stage and the landing, so the
                 dashboard is described by exactly one stylesheet.
3. Components    Small helpers that render themed markup — ``page_header``,
                 ``section``, ``stat_row``, ``panel``, ``sonar_card``,
                 ``badge``, ``meter``.  The landing and the background stage
                 live in ``src/dashboard/cinema.py`` because they own media
                 rather than chrome.
4. Chart theming ``style_figure`` applies the same palette to every Plotly
                 figure so charts and page chrome never drift apart.

Surfaces over the ocean
-----------------------
Every panel in here floats over the background media rendered by
:mod:`src.dashboard.cinema`, so no surface is opaque.  Legibility comes from
``backdrop-filter`` — a blur behind the panel removes the reef's high-frequency
detail, which is what small text actually competes with, while the water keeps
its colour.  Changing a ``glass`` token to an opaque colour would restore the
old flat look and punch a hole in the scene.

To restyle the dashboard, change the tokens below rather than editing pages.
Every page picks the theme up automatically because ``components.set_page``
calls ``inject_theme``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import streamlit as st

# ---------------------------------------------------------------------------
# 1. Design tokens
# ---------------------------------------------------------------------------

# Depth — background layers, darkest first.
VOID = "#020b14"  # deepest plate, painted before the media loads
TRENCH = "#041525"  # abyssal plain
BASIN = "#072038"  # ocean basin
ABYSS = "#04101f"
NAVY = "#0a1628"
DEEP = "#0d1e36"
SURFACE = "#112240"
SURFACE_HI = "#16304f"

# Structure — borders and separators.
BORDER = "#1f4370"
BORDER_SOFT = "#183453"

# Ocean progression — the ordered depth ramp every data visualisation uses.
# Read as a bathymetric scale: OCEAN_DEEP is the trench, FOAM is the surf line.
OCEAN_DEEP = "#053f5c"
OCEAN_MID = "#087ca7"
OCEAN_BRIGHT = "#0ebbd2"
OCEAN_GLOW = "#37e6e6"
FOAM = "#bafcff"

# Highlights — aqua / cyan / turquoise family.
AQUA = "#22d3ee"
TEAL = "#00b4d8"
TURQUOISE = "#2dd4bf"
CYAN = "#90e0ef"
CYAN_SOFT = "#7dd3fc"

# Accent — coral, used sparingly for contrast against the aqua family.
CORAL = "#ff6b6b"
CORAL_SOFT = "#ff8c69"

# Type.
TEXT_BRIGHT = "#eaf4ff"
TEXT = "#c9dcf0"
TEXT_MUTED = "#8ba7c7"
TEXT_DIM = "#5d799d"

# Status.
SUCCESS = "#38e6a2"
WARNING = "#ffb84d"
DANGER = "#ff4d6d"

# ---------------------------------------------------------------------------
# Visualisation ramps
#
# DEPTH_RAMP is the canonical "deep navy → blue → teal → aqua → pale cyan"
# progression named in the design brief.  Mountain layers, stream bands,
# contour isolines and wireframe meshes all sample it, so a chart drawn by any
# builder in src/dashboard/viz/ reads as part of the same system.
# ---------------------------------------------------------------------------

DEPTH_RAMP: tuple[str, ...] = (
    TRENCH,
    OCEAN_DEEP,
    OCEAN_MID,
    OCEAN_BRIGHT,
    OCEAN_GLOW,
    FOAM,
)

# Plotly continuous colourscale form of DEPTH_RAMP (position, colour).
DEPTH_COLORSCALE: tuple[tuple[float, str], ...] = tuple(
    (i / (len(DEPTH_RAMP) - 1), colour) for i, colour in enumerate(DEPTH_RAMP)
)

# Surface used by the sonar wireframe family: almost-black paper, pale mesh.
SONAR_BG = VOID
SONAR_MESH = "#5fd6ea"
SONAR_MESH_SOFT = "rgba(95, 214, 234, 0.32)"

# Geometry.
RADIUS = "16px"
RADIUS_SM = "10px"
SHADOW = "0 18px 40px -24px rgba(0, 0, 0, 0.85)"
SHADOW_SOFT = "0 10px 28px -20px rgba(0, 0, 0, 0.8)"

FONT_STACK = (
    '"Inter", "Segoe UI", -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif'
)
MONO_STACK = '"JetBrains Mono", "SFMono-Regular", "Cascadia Mono", Consolas, monospace'

# Ordered palette handed to Plotly when a chart has no explicit colour map.
COLORWAY: tuple[str, ...] = (
    AQUA,
    CORAL,
    TURQUOISE,
    CYAN_SOFT,
    WARNING,
    "#a78bfa",
    SUCCESS,
    "#f472b6",
)

# Every token exposed to CSS as a custom property (``--cs-<name>``).
_TOKENS: dict[str, str] = {
    "void": VOID,
    "trench": TRENCH,
    "basin": BASIN,
    "ocean-deep": OCEAN_DEEP,
    "ocean-mid": OCEAN_MID,
    "ocean-bright": OCEAN_BRIGHT,
    "ocean-glow": OCEAN_GLOW,
    "foam": FOAM,
    "abyss": ABYSS,
    "navy": NAVY,
    "deep": DEEP,
    "surface": SURFACE,
    "surface-hi": SURFACE_HI,
    "border": BORDER,
    "border-soft": BORDER_SOFT,
    "aqua": AQUA,
    "teal": TEAL,
    "turquoise": TURQUOISE,
    "cyan": CYAN,
    "cyan-soft": CYAN_SOFT,
    "coral": CORAL,
    "coral-soft": CORAL_SOFT,
    "text-bright": TEXT_BRIGHT,
    "text": TEXT,
    "text-muted": TEXT_MUTED,
    "text-dim": TEXT_DIM,
    "success": SUCCESS,
    "warning": WARNING,
    "danger": DANGER,
    "radius": RADIUS,
    "radius-sm": RADIUS_SM,
    "shadow": SHADOW,
    "shadow-soft": SHADOW_SOFT,
    "font": FONT_STACK,
    "mono": MONO_STACK,
    # Reusable composite surfaces.
    #
    # These are DELIBERATELY translucent.  Every panel now floats over the
    # background footage (src/dashboard/cinema.py), so an opaque plate would
    # punch a hole in the media and undo the whole effect.  Readability comes
    # from `backdrop-filter` instead: the blur behind the panel destroys the
    # reef's high-frequency detail, which is what text actually competes with,
    # while the ocean's colour still reads through.  Contrast against
    # --cs-text-bright stays above 12:1 on every layer of the scene.
    "glass": "linear-gradient(158deg, rgba(11, 42, 68, 0.66), rgba(3, 19, 35, 0.78))",
    "glass-hi": "linear-gradient(158deg, rgba(16, 55, 86, 0.76), rgba(4, 25, 44, 0.86))",
    "blur": "blur(18px) saturate(1.25)",
    "blur-sm": "blur(10px) saturate(1.15)",
}


# ---------------------------------------------------------------------------
# 2. Global stylesheet
# ---------------------------------------------------------------------------

_BASE_CSS = """
/* ---------- App shell ----------------------------------------------------
   The app's own background is only a base plate.  The visible background is
   the media stage from src/dashboard/cinema.py, fixed to the viewport and
   behind every page.  This plate is what is on screen for the frame before
   the poster decodes, and what shows if the media is absent entirely. */
.stApp {
  background: linear-gradient(180deg, var(--cs-basin) 0%, var(--cs-trench) 52%, var(--cs-void) 100%);
  background-attachment: fixed;
  font-family: var(--cs-font);
  color: var(--cs-text);
}

/* The header is a strip of glass over the water, not a bar. */
[data-testid="stHeader"] {
  background: linear-gradient(180deg, rgba(2, 14, 26, 0.62), rgba(2, 14, 26, 0.16));
  backdrop-filter: blur(12px) saturate(1.1);
  -webkit-backdrop-filter: blur(12px) saturate(1.1);
}

[data-testid="stMain"] .block-container,
.stMainBlockContainer {
  padding-top: 2.4rem;
  padding-bottom: 4.5rem;
  max-width: 1480px;
}

/* ---------- Typography --------------------------------------------------- */
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
  font-family: var(--cs-font);
  color: var(--cs-text-bright);
  letter-spacing: -0.018em;
}
.stApp h1 { font-size: 2.05rem; font-weight: 700; }
.stApp h2 { font-size: 1.4rem;  font-weight: 650; }
.stApp h3 { font-size: 1.12rem; font-weight: 650; }
.stApp p, .stApp li { color: var(--cs-text); line-height: 1.62; }
.stApp code, .stApp pre { font-family: var(--cs-mono); }
.stApp a { color: var(--cs-cyan-soft); text-decoration: none; }
.stApp a:hover { color: var(--cs-aqua); text-decoration: underline; }

[data-testid="stCaptionContainer"], .stCaption {
  color: var(--cs-text-muted) !important;
}

/* Gradient hairline instead of the default flat rule. */
.stApp hr {
  border: none;
  height: 1px;
  margin: 1.9rem 0;
  background: linear-gradient(90deg, transparent, var(--cs-border), transparent);
}

/* ---------- Sidebar ------------------------------------------------------
   Heavier glass than the content panels: the sidebar carries small type and
   sits over the brightest part of the water column. */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(8, 27, 47, 0.86) 0%, rgba(4, 16, 30, 0.90) 55%,
                                      rgba(2, 11, 21, 0.94) 100%);
  backdrop-filter: blur(22px) saturate(1.2);
  -webkit-backdrop-filter: blur(22px) saturate(1.2);
  border-right: 1px solid var(--cs-border-soft);
  box-shadow: 1px 0 40px -18px rgba(34, 211, 238, 0.5);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { font-size: 0.88rem; }

/* Brand block sits above the auto-generated page list where the DOM allows. */
[data-testid="stSidebarContent"] { display: flex; flex-direction: column; }
/* flex-shrink:0 keeps the sidebar scrollable instead of compressing children. */
[data-testid="stSidebarUserContent"] { order: 1; flex-shrink: 0; }
[data-testid="stSidebarNav"] { order: 2; flex-shrink: 0; }

[data-testid="stSidebarNav"] {
  padding-top: 0.35rem;
  border-top: 1px solid var(--cs-border-soft);
  margin-top: 0.4rem;
}
[data-testid="stSidebarNav"]::before {
  content: "Navigation";
  display: block;
  padding: 0.85rem 1rem 0.5rem;
  font-size: 0.66rem;
  font-weight: 700;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--cs-text-dim);
}
[data-testid="stSidebarNav"] a {
  border-radius: 9px;
  margin: 1px 0.4rem;
  transition: background 0.16s ease, transform 0.16s ease;
}
[data-testid="stSidebarNav"] a:hover {
  background: rgba(34, 211, 238, 0.10);
  transform: translateX(2px);
}
[data-testid="stSidebarNav"] a[aria-current="page"] {
  background: linear-gradient(90deg, rgba(34, 211, 238, 0.20), rgba(34, 211, 238, 0.02));
  box-shadow: inset 3px 0 0 var(--cs-aqua);
}
[data-testid="stSidebarNav"] a[aria-current="page"] span {
  color: var(--cs-cyan-soft) !important;
  font-weight: 600;
}

/* ---------- Brand -------------------------------------------------------- */
.cs-brand {
  display: flex;
  align-items: center;
  gap: 0.7rem;
  padding: 0.5rem 0.15rem 0.9rem;
}
.cs-brand__mark {
  flex: 0 0 auto;
  width: 38px; height: 38px;
  border-radius: 12px;
  display: grid; place-items: center;
  font-size: 1.15rem;
  background: linear-gradient(140deg, rgba(34, 211, 238, 0.30), rgba(45, 212, 191, 0.12));
  border: 1px solid rgba(34, 211, 238, 0.42);
  box-shadow: 0 0 22px -6px rgba(34, 211, 238, 0.6);
}
.cs-brand__name {
  font-size: 1.12rem; font-weight: 700; letter-spacing: -0.01em;
  background: linear-gradient(92deg, var(--cs-cyan-soft), var(--cs-turquoise));
  -webkit-background-clip: text; background-clip: text; color: transparent;
  line-height: 1.15;
}
.cs-brand__tag {
  font-size: 0.68rem; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--cs-text-dim); font-weight: 600;
}

.cs-side-label {
  font-size: 0.66rem; font-weight: 700; letter-spacing: 0.18em;
  text-transform: uppercase; color: var(--cs-text-dim);
  margin: 0.2rem 0 0.45rem;
}

.cs-status {
  display: flex; align-items: center; gap: 0.6rem;
  padding: 0.6rem 0.8rem;
  border-radius: var(--cs-radius-sm);
  background: rgba(255, 255, 255, 0.035);
  border: 1px solid var(--cs-border-soft);
}
.cs-status__dot { flex: 0 0 auto; width: 9px; height: 9px; border-radius: 50%; }
.cs-status__dot--live { animation: cs-pulse 2.4s ease-in-out infinite; }
@keyframes cs-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.42; } }
.cs-status__text { font-size: 0.82rem; font-weight: 650; line-height: 1.3; }
.cs-status__sub { font-size: 0.7rem; color: var(--cs-text-dim); }

.cs-side-note {
  background: linear-gradient(140deg, rgba(255, 107, 107, 0.14), rgba(255, 107, 107, 0.04));
  border: 1px solid rgba(255, 107, 107, 0.30);
  border-left: 3px solid var(--cs-coral);
  border-radius: var(--cs-radius-sm);
  padding: 0.6rem 0.75rem;
  font-size: 0.76rem;
  line-height: 1.5;
  color: #ffd9d2;
}

/* ---------- Page header (lighter than the hero) -------------------------- */
.cs-page {
  display: flex; align-items: flex-start; gap: 1rem;
  padding: 0.2rem 0 1.15rem;
  border-bottom: 1px solid var(--cs-border-soft);
  margin-bottom: 1.4rem;
}
.cs-page__bar {
  flex: 0 0 auto; width: 4px; align-self: stretch; min-height: 46px;
  border-radius: 4px;
  background: linear-gradient(180deg, var(--cs-aqua), var(--cs-turquoise));
  box-shadow: 0 0 18px -2px rgba(34, 211, 238, 0.75);
}
.cs-page__eyebrow {
  font-size: 0.66rem; font-weight: 700; letter-spacing: 0.2em;
  text-transform: uppercase; color: var(--cs-text-dim); margin-bottom: 0.3rem;
}
.cs-page__title {
  margin: 0; font-size: 1.85rem; font-weight: 700;
  letter-spacing: -0.025em; color: var(--cs-text-bright); line-height: 1.18;
}
.cs-page__sub {
  margin: 0.4rem 0 0; max-width: 82ch;
  font-size: 0.92rem; line-height: 1.6; color: var(--cs-text-muted);
}

/* ---------- Section header ----------------------------------------------- */
.cs-section { margin: 2.1rem 0 1rem; }
.cs-section__kicker {
  font-size: 0.64rem; font-weight: 700; letter-spacing: 0.2em;
  text-transform: uppercase; color: var(--cs-aqua); margin-bottom: 0.3rem;
}
.cs-section__title {
  display: flex; align-items: center; gap: 0.7rem;
  margin: 0; font-size: 1.28rem; font-weight: 650;
  letter-spacing: -0.02em; color: var(--cs-text-bright);
}
.cs-section__title::after {
  content: ""; flex: 1; height: 1px;
  background: linear-gradient(90deg, var(--cs-border), transparent);
}
.cs-section__sub {
  margin: 0.38rem 0 0; max-width: 88ch;
  font-size: 0.87rem; line-height: 1.58; color: var(--cs-text-muted);
}

/* ---------- Field group (form subsection) -------------------------------- */
.cs-group {
  display: flex; align-items: center; gap: 0.6rem;
  margin: 1.5rem 0 0.6rem;
  font-size: 0.71rem; font-weight: 700; letter-spacing: 0.16em;
  text-transform: uppercase; color: var(--cs-cyan);
}
.cs-group:first-child { margin-top: 0.2rem; }
.cs-group__num {
  flex: 0 0 auto;
  width: 22px; height: 22px; border-radius: 7px;
  display: grid; place-items: center;
  font-size: 0.72rem; letter-spacing: 0;
  background: rgba(34, 211, 238, 0.14);
  border: 1px solid rgba(34, 211, 238, 0.32);
  color: var(--cs-aqua);
}
.cs-group::after {
  content: ""; flex: 1; height: 1px;
  background: linear-gradient(90deg, var(--cs-border-soft), transparent);
}

/* ---------- Panels and stat cards ---------------------------------------- */
.cs-panel {
  position: relative;
  background: var(--cs-glass);
  backdrop-filter: var(--cs-blur);
  -webkit-backdrop-filter: var(--cs-blur);
  border: 1px solid var(--cs-border-soft);
  border-radius: var(--cs-radius);
  padding: 1.15rem 1.3rem;
  box-shadow: var(--cs-shadow-soft);
  height: 100%;
}
.cs-panel--accent { border-left: 3px solid var(--cs-aqua); }
.cs-panel__label {
  font-size: 0.68rem; font-weight: 700; letter-spacing: 0.16em;
  text-transform: uppercase; color: var(--cs-aqua); margin-bottom: 0.45rem;
}
.cs-panel__title {
  font-size: 1.02rem; font-weight: 650; color: var(--cs-text-bright); margin-bottom: 0.2rem;
}
.cs-panel__body { font-size: 0.87rem; line-height: 1.6; color: var(--cs-text-muted); }
.cs-panel ul { margin: 0.2rem 0 0; padding-left: 1.15rem; }
.cs-panel li { font-size: 0.86rem; line-height: 1.62; color: var(--cs-text); }

.cs-stat {
  position: relative; overflow: hidden; height: 100%;
  background: var(--cs-glass);
  backdrop-filter: var(--cs-blur);
  -webkit-backdrop-filter: var(--cs-blur);
  border: 1px solid var(--cs-border-soft);
  border-radius: var(--cs-radius);
  padding: 1.05rem 1.2rem 1.1rem;
  box-shadow: var(--cs-shadow-soft);
  transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
}
.cs-stat:hover {
  transform: translateY(-2px);
  border-color: rgba(34, 211, 238, 0.42);
  box-shadow: 0 20px 40px -26px rgba(34, 211, 238, 0.75);
}
.cs-stat::before {
  content: ""; position: absolute; top: 0; left: 0; bottom: 0; width: 3px;
  background: var(--cs-stat-accent, var(--cs-aqua));
}
.cs-stat__label {
  font-size: 0.66rem; font-weight: 700; letter-spacing: 0.16em;
  text-transform: uppercase; color: var(--cs-text-muted);
}
.cs-stat__value {
  margin-top: 0.42rem;
  font-size: 1.95rem; font-weight: 720; line-height: 1.05;
  letter-spacing: -0.03em; color: var(--cs-text-bright);
}
.cs-stat__value--sm { font-size: 1.35rem; }
.cs-stat__caption { margin-top: 0.3rem; font-size: 0.74rem; color: var(--cs-text-dim); }

/* ---------- Chips, badges, meters ---------------------------------------- */
.cs-chip {
  display: inline-flex; align-items: center; gap: 0.4rem;
  font-size: 0.75rem; font-weight: 600;
  padding: 0.3rem 0.75rem;
  border-radius: 999px;
  background: rgba(125, 211, 252, 0.08);
  border: 1px solid rgba(125, 211, 252, 0.22);
  color: var(--cs-cyan-soft);
}
.cs-badge {
  display: inline-block;
  font-size: 0.78rem; font-weight: 600;
  padding: 0.18rem 0.7rem;
  border-radius: 999px;
  line-height: 1.5;
}
.cs-meter { margin: 0.42rem 0 0.6rem; }
.cs-meter__head {
  display: flex; justify-content: space-between; align-items: baseline;
  font-size: 0.82rem; margin-bottom: 0.28rem;
}
.cs-meter__name { font-weight: 600; }
.cs-meter__value { color: var(--cs-text-muted); font-variant-numeric: tabular-nums; }
.cs-meter__track {
  position: relative; height: 8px; border-radius: 999px;
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.04);
  overflow: hidden;
}
.cs-meter__fill { height: 100%; border-radius: 999px; }

/* ---------- Prediction result cards -------------------------------------- */
.cs-pred {
  position: relative; overflow: hidden;
  border-radius: var(--cs-radius);
  border: 1px solid var(--cs-border-soft);
  background: var(--cs-glass-hi);
  backdrop-filter: var(--cs-blur);
  -webkit-backdrop-filter: var(--cs-blur);
  padding: 1.35rem 1.45rem 1.45rem;
  box-shadow: var(--cs-shadow);
}
.cs-pred::before {
  content: ""; position: absolute; inset: 0 0 auto 0; height: 3px;
  background: var(--cs-pred-accent, var(--cs-aqua));
}
.cs-pred::after {
  content: ""; position: absolute; right: -60px; top: -60px;
  width: 190px; height: 190px; border-radius: 50%;
  background: var(--cs-pred-accent, var(--cs-aqua));
  opacity: 0.10; filter: blur(14px); pointer-events: none;
}
.cs-pred__inner { position: relative; z-index: 1; }
.cs-pred__task {
  font-size: 0.67rem; font-weight: 700; letter-spacing: 0.18em;
  text-transform: uppercase; color: var(--cs-text-muted);
}
.cs-pred__class {
  margin: 0.5rem 0 0.9rem;
  font-size: 1.9rem; font-weight: 720; line-height: 1.1; letter-spacing: -0.03em;
  color: var(--cs-pred-accent, var(--cs-aqua));
}
.cs-pred__confhead {
  display: flex; justify-content: space-between; align-items: baseline;
  font-size: 0.74rem; letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--cs-text-dim); margin-bottom: 0.35rem;
}
.cs-pred__confval {
  font-size: 1.05rem; font-weight: 700; letter-spacing: 0;
  text-transform: none; color: var(--cs-text-bright);
  font-variant-numeric: tabular-nums;
}
.cs-pred__track {
  height: 10px; border-radius: 999px;
  background: rgba(255, 255, 255, 0.07);
  overflow: hidden;
}
.cs-pred__fill { height: 100%; border-radius: 999px; }
.cs-pred__meta {
  display: flex; flex-wrap: wrap; gap: 0.4rem;
  margin-top: 1.05rem; padding-top: 0.9rem;
  border-top: 1px solid rgba(255, 255, 255, 0.07);
}
.cs-pred__metaitem {
  font-size: 0.71rem;
  padding: 0.2rem 0.6rem;
  border-radius: 7px;
  background: rgba(255, 255, 255, 0.045);
  border: 1px solid rgba(255, 255, 255, 0.06);
  color: var(--cs-text-muted);
}
.cs-pred__metaitem b { color: var(--cs-text); font-weight: 600; }

/* ---------- Status rows (pipeline / milestones) -------------------------- */
.cs-row {
  display: flex; align-items: flex-start; gap: 0.85rem;
  background: var(--cs-glass);
  backdrop-filter: var(--cs-blur-sm);
  -webkit-backdrop-filter: var(--cs-blur-sm);
  border: 1px solid var(--cs-border-soft);
  border-left: 3px solid var(--cs-row-accent, var(--cs-aqua));
  border-radius: var(--cs-radius-sm);
  padding: 0.7rem 1rem;
  margin-bottom: 0.5rem;
  transition: border-color 0.16s ease, transform 0.16s ease;
}
.cs-row:hover { transform: translateX(2px); border-color: rgba(34, 211, 238, 0.35); }
.cs-row__icon {
  flex: 0 0 auto; width: 22px; text-align: center;
  color: var(--cs-row-accent, var(--cs-aqua)); font-weight: 700;
}
.cs-row__body { flex: 1 1 auto; min-width: 0; }
.cs-row__head {
  display: flex; align-items: center; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.2rem;
}
.cs-row__tag {
  font-size: 0.68rem; font-weight: 700; letter-spacing: 0.06em;
  padding: 0.08rem 0.45rem; border-radius: 6px;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.08);
  color: var(--cs-cyan);
}
.cs-row__name { font-weight: 650; color: var(--cs-text-bright); font-size: 0.94rem; }
.cs-row__cat { font-size: 0.79rem; color: var(--cs-text-dim); }
.cs-row__status {
  margin-left: auto; font-size: 0.68rem; font-weight: 700;
  letter-spacing: 0.08em; text-transform: uppercase;
  padding: 0.12rem 0.6rem; border-radius: 999px;
  color: var(--cs-row-accent, var(--cs-aqua));
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid currentColor;
}
.cs-row__desc { font-size: 0.84rem; line-height: 1.55; color: var(--cs-text-muted); }
.cs-row__files {
  font-family: var(--cs-mono); font-size: 0.71rem;
  color: var(--cs-text-dim); margin-top: 0.2rem;
}

/* ---------- Metrics ------------------------------------------------------ */
[data-testid="stMetric"] {
  position: relative; overflow: hidden;
  background: var(--cs-glass);
  backdrop-filter: var(--cs-blur);
  -webkit-backdrop-filter: var(--cs-blur);
  border: 1px solid var(--cs-border-soft);
  border-radius: var(--cs-radius);
  padding: 1rem 1.15rem 1.05rem;
  box-shadow: var(--cs-shadow-soft);
}
[data-testid="stMetric"]::before {
  content: ""; position: absolute; top: 0; left: 0; bottom: 0; width: 3px;
  background: linear-gradient(180deg, var(--cs-aqua), var(--cs-turquoise));
}
[data-testid="stMetricLabel"] p {
  font-size: 0.68rem !important; font-weight: 700 !important;
  letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--cs-text-muted) !important;
}
[data-testid="stMetricValue"] {
  font-size: 1.7rem; font-weight: 720; letter-spacing: -0.03em;
  color: var(--cs-text-bright);
}

/* ---------- Buttons ------------------------------------------------------ */
.stButton > button,
.stDownloadButton > button,
[data-testid="stFormSubmitButton"] > button {
  border-radius: 11px;
  font-weight: 620;
  letter-spacing: 0.01em;
  border: 1px solid var(--cs-border);
  background: rgba(22, 48, 79, 0.6);
  color: var(--cs-text-bright);
  transition: transform 0.16s ease, box-shadow 0.16s ease,
              border-color 0.16s ease, filter 0.16s ease;
}
.stButton > button:hover,
.stDownloadButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {
  border-color: var(--cs-aqua);
  transform: translateY(-1px);
  box-shadow: 0 12px 26px -18px rgba(34, 211, 238, 0.95);
}
button[kind="primary"],
button[kind="primaryFormSubmit"] {
  background: linear-gradient(122deg, var(--cs-aqua), var(--cs-turquoise)) !important;
  border: none !important;
  color: #04202b !important;
  font-weight: 700 !important;
  box-shadow: 0 12px 28px -16px rgba(34, 211, 238, 0.9);
}
button[kind="primary"]:hover,
button[kind="primaryFormSubmit"]:hover {
  filter: brightness(1.08);
  transform: translateY(-1px);
}

/* ---------- Forms and inputs --------------------------------------------- */
[data-testid="stForm"] {
  background: var(--cs-glass);
  backdrop-filter: var(--cs-blur);
  -webkit-backdrop-filter: var(--cs-blur);
  border: 1px solid var(--cs-border-soft);
  border-radius: var(--cs-radius);
  padding: 1.5rem 1.6rem 1.6rem;
  box-shadow: var(--cs-shadow-soft);
}

.stTextInput input,
.stNumberInput input,
.stDateInput input,
[data-baseweb="select"] > div {
  background: rgba(8, 20, 36, 0.72) !important;
  border-color: var(--cs-border-soft) !important;
  border-radius: var(--cs-radius-sm) !important;
  color: var(--cs-text) !important;
}
.stTextInput input:focus,
.stNumberInput input:focus,
[data-baseweb="select"] > div:focus-within {
  border-color: var(--cs-aqua) !important;
  box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.18) !important;
}
[data-baseweb="tag"] {
  background: rgba(34, 211, 238, 0.16) !important;
  border: 1px solid rgba(34, 211, 238, 0.34) !important;
  color: var(--cs-cyan-soft) !important;
  border-radius: 7px !important;
}
[data-testid="stWidgetLabel"] p {
  font-size: 0.83rem; font-weight: 600; color: var(--cs-text-muted);
}

/* ---------- Alerts, expanders, tables, code ------------------------------ */
[data-testid="stAlert"],
[data-testid="stAlertContainer"] {
  border-radius: var(--cs-radius-sm);
  border: 1px solid rgba(255, 255, 255, 0.09);
  backdrop-filter: var(--cs-blur-sm);
  -webkit-backdrop-filter: var(--cs-blur-sm);
}

[data-testid="stExpander"] {
  border: 1px solid var(--cs-border-soft);
  border-radius: var(--cs-radius-sm);
  background: rgba(6, 26, 45, 0.66);
  backdrop-filter: var(--cs-blur-sm);
  -webkit-backdrop-filter: var(--cs-blur-sm);
  overflow: hidden;
}
[data-testid="stExpander"] details { border: none; background: transparent; }
[data-testid="stExpander"] summary { font-weight: 600; color: var(--cs-cyan); }
[data-testid="stExpander"] summary:hover { color: var(--cs-aqua); }

[data-testid="stDataFrame"], [data-testid="stTable"] {
  border: 1px solid var(--cs-border-soft);
  border-radius: var(--cs-radius-sm);
  overflow: hidden;
  background: var(--cs-glass);
  backdrop-filter: var(--cs-blur-sm);
  -webkit-backdrop-filter: var(--cs-blur-sm);
}
.stApp table { border-collapse: collapse; }
.stApp thead th {
  color: var(--cs-cyan) !important;
  font-size: 0.78rem; letter-spacing: 0.04em; text-transform: uppercase;
  border-bottom: 1px solid var(--cs-border) !important;
}
.stApp tbody td { border-color: var(--cs-border-soft) !important; }

[data-testid="stCode"] pre, .stCode pre {
  background: rgba(4, 16, 31, 0.85) !important;
  border: 1px solid var(--cs-border-soft);
  border-radius: var(--cs-radius-sm);
}

[data-testid="stJson"] {
  border: 1px solid var(--cs-border-soft);
  border-radius: var(--cs-radius-sm);
}

/* ---------- Charts ------------------------------------------------------- */
/* No padding here: Plotly measures this element to size itself, and padding
   would push the canvas past the column and create a horizontal scrollbar.
   Figures are rendered with a transparent paper so this surface shows through. */
[data-testid="stPlotlyChart"] {
  background: var(--cs-glass);
  backdrop-filter: var(--cs-blur);
  -webkit-backdrop-filter: var(--cs-blur);
  border: 1px solid var(--cs-border-soft);
  border-radius: var(--cs-radius);
  box-shadow: var(--cs-shadow-soft);
  overflow: hidden;
}

/* ---------- Tabs --------------------------------------------------------- */
[data-baseweb="tab-list"] { gap: 0.35rem; border-bottom: 1px solid var(--cs-border-soft); }
[data-baseweb="tab"] {
  border-radius: 10px 10px 0 0;
  color: var(--cs-text-muted);
  font-weight: 600;
}
[data-baseweb="tab"][aria-selected="true"] { color: var(--cs-aqua); }

/* ---------- Progressive enhancement: themed st.container(border=True) ---- */
/* The anchor is the first element inside the container, so the `>` chain only
   matches that container and never an outer block. Falls back to Streamlit's
   default border if the DOM shape changes. */
[data-testid="stVerticalBlockBorderWrapper"]:has(
  > div > [data-testid="stVerticalBlock"]
  > [data-testid="stElementContainer"]:first-child .cs-card-anchor
) {
  background: var(--cs-glass);
  backdrop-filter: var(--cs-blur);
  -webkit-backdrop-filter: var(--cs-blur);
  border: 1px solid var(--cs-border-soft) !important;
  border-radius: var(--cs-radius) !important;
  box-shadow: var(--cs-shadow-soft);
}
.cs-card-anchor { display: none; }

/* ---------- Stacking ------------------------------------------------------
   Streamlit paints each block in DOM order.  The stage is rendered first by
   set_page(), and every following block is promoted to its own positioned
   layer, so the media stays behind all page content without any page needing
   a z-index of its own. */
[data-testid="stMain"] .block-container > div,
.stMainBlockContainer > div { position: relative; z-index: 1; }
[data-testid="stSidebar"] { z-index: 5; }
[data-testid="stHeader"], [data-testid="stTopNav"] { z-index: 6; }

/* ---------- Scrollbars --------------------------------------------------- */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: rgba(4, 16, 31, 0.6); }
::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, #1f4370, #16304f);
  border-radius: 999px;
  border: 2px solid transparent;
  background-clip: content-box;
}
::-webkit-scrollbar-thumb:hover { background: var(--cs-border); background-clip: content-box; }

/* ---------- The media stage ----------------------------------------------
   One fixed, viewport-filling media plane behind the whole app, plus three
   grading layers.  This replaced a hand-drawn SVG scene of ~4,300 animated
   paths; the entire background is now about a dozen DOM nodes.

   `object-fit: cover` on a fixed, inset-0 element is what makes the media
   fill the viewport at any aspect ratio without letterboxing or stretching. */
.cs-stage {
  position: fixed;
  inset: 0;
  z-index: 0;
  overflow: hidden;
  pointer-events: none;
  background: var(--cs-void);
}
.cs-stage__video,
.cs-stage__plate {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  /* The reef mound sits along the lower half of the frame, so the crop is
     anchored below centre: on a wide viewport it is the top water column that
     gets cropped, never the reef. */
  object-position: center 58%;
  /* Sub-pixel scale kills the 1px edge seam some browsers leave on a
     cover-fitted video at fractional viewport widths. */
  transform: scale(1.02);
}
/* The video is inserted by the component between the plate and the grade, so
   DOM order does the layering.  These only matter for the instant before the
   move completes, or if the stage container cannot be found. */
.cs-stage__video {
  /* Its own fixed layer, sequenced between the plate and the overlays by DOM
     order (see cinema._overlays). */
  position: fixed;
  border: 0;
  background: transparent;
}
/* The overlay block paints nothing itself; only its three children do. */
.cs-stage--overlays { background: none; }
.cs-stage__video,
.cs-stage__plate {
  /* The footage is graded down by the layers above it; this puts back the
     saturation those layers cost, so orange, yellow and green reef tones
     survive to the screen. */
  filter: saturate(1.18) contrast(1.05);
}
.cs-stage__plate {
  background-size: cover;
  background-position: center;
  background-repeat: no-repeat;
}

/* 1. Colour grade — pulls the footage toward the dashboard's ocean palette so
      the media and the UI read as one system rather than a photo behind a UI.

      Deliberately NOT `mix-blend-mode: multiply`.  Multiply scales each channel
      by the overlay's, so a blue wash drives red, orange and yellow toward zero
      — which is precisely the coral colour this footage exists to show.  A plain
      low-opacity wash darkens the frame without collapsing its hue, and the
      shadows are shaped by the scrim and vignette below instead. */
.cs-stage__grade {
  position: absolute; inset: 0;
  background:
    linear-gradient(180deg,
      rgba(6, 40, 60, 0.20) 0%,
      rgba(4, 30, 48, 0.04) 40%,
      rgba(2, 20, 34, 0.30) 100%),
    radial-gradient(120% 80% at 72% 24%, rgba(34, 211, 238, 0.05), transparent 62%);
}

/* 2. Readability scrim — the left-to-right wash the copy sits on.  Heavier at
      the left where the type is, released by the right so the marine subject
      keeps its contrast. */
.cs-stage__scrim {
  position: absolute; inset: 0;
  background:
    linear-gradient(96deg,
      rgba(1, 10, 18, 0.84) 0%,
      rgba(1, 12, 22, 0.66) 26%,
      rgba(2, 16, 28, 0.28) 52%,
      rgba(2, 18, 32, 0.04) 76%,
      rgba(2, 18, 32, 0.00) 100%);
}

/* 3. Vignette — top and bottom only, to seat the nav and the lower content. */
.cs-stage__vignette {
  position: absolute; inset: 0;
  background:
    linear-gradient(180deg, rgba(1, 8, 15, 0.66) 0%, transparent 20%, transparent 64%,
                            rgba(1, 8, 15, 0.62) 100%),
    radial-gradient(140% 100% at 50% 50%, transparent 56%, rgba(1, 8, 15, 0.34) 100%);
}

/* The still stage carries a quieter grade: interior pages have charts to read,
   so the backdrop gives up more contrast than the landing does. */
.cs-stage--still .cs-stage__scrim {
  background: linear-gradient(96deg, rgba(1, 10, 18, 0.90), rgba(1, 12, 21, 0.84));
}
.cs-stage--still .cs-stage__plate { filter: saturate(0.72) brightness(0.72); }

/* The component mount point must not take part in layout: the stage inside it
   is fixed, so the wrapper should occupy no space in the block flow.
   `stBidiComponentRegular` is the Components V2 wrapper for isolate_styles=False
   (the isolated variant is `stBidiComponentIsolated`); both are matched so the
   rule survives a switch. */
[data-testid="stBidiComponentRegular"]:has([data-cs-stage]),
[data-testid="stBidiComponentIsolated"]:has([data-cs-stage]) {
  height: 0 !important;
  min-height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: visible !important;
}

@media (prefers-reduced-motion: reduce) {
  /* The video element keeps its poster attribute, so suppressing playback
     leaves exactly the still frame.  The JS stops decode; this is the belt. */
  .cs-stage__video { animation: none !important; }
}

/* ---------- The landing --------------------------------------------------
   The first viewport IS the hero.  There is deliberately no card, no border
   and no panel around any of this — the media fills the screen and the type
   sits on it. */
.cs-landing {
  /* The first screen's height lives HERE, not on a wrapper.  Streamlit renders
     each element into its own container, so a <div> opened in one st.markdown
     call cannot wrap the blocks that follow it — only a single block can carry
     the height.  This one holds index + title + lede + rule + metadata, which
     is the bulk of the first screen; the CTAs and the bottom row follow it and
     bring the total to roughly one viewport. */
  min-height: 62vh;
  display: flex;
  flex-direction: column;
  justify-content: center;
  max-width: 58ch;
}
/* The bottom band of the first screen. */
.cs-footnote, .cs-feature { margin-top: 2.2rem; }
.cs-landing__index {
  font-family: var(--cs-mono);
  font-size: 0.78rem; font-weight: 600; letter-spacing: 0.32em;
  color: var(--cs-cyan-soft);
  opacity: 0.85;
  margin-bottom: 1.4rem;
}
.cs-landing__title {
  margin: 0 0 1.3rem;
  font-size: clamp(3.2rem, 7vw, 7.5rem);
  font-weight: 700;
  line-height: 0.9;
  letter-spacing: -0.038em;
  color: #f4feff;
  text-shadow: 0 4px 60px rgba(0, 0, 0, 0.72), 0 1px 3px rgba(0, 0, 0, 0.5);
}
.cs-landing__lede {
  margin: 0;
  max-width: 52ch;
  font-size: clamp(0.95rem, 1.15vw, 1.1rem);
  line-height: 1.68;
  color: #dceaf2;
  text-shadow: 0 1px 22px rgba(0, 0, 0, 0.9);
}
.cs-landing__rule {
  width: 100%; max-width: 30rem; height: 1px;
  margin: 2rem 0 1.4rem;
  background: linear-gradient(90deg, rgba(190, 245, 255, 0.55), rgba(190, 245, 255, 0.02));
}
.cs-landing__meta { display: flex; flex-wrap: wrap; gap: 2.2rem; }
.cs-landing__metaitem { display: flex; flex-direction: column; gap: 0.22rem; }
.cs-landing__metakey {
  font-size: 0.62rem; font-weight: 700; letter-spacing: 0.2em;
  text-transform: uppercase; color: rgba(190, 231, 245, 0.62);
}
.cs-landing__metaval {
  font-size: 0.9rem; font-weight: 600; color: #e8f6fb;
  text-shadow: 0 1px 14px rgba(0, 0, 0, 0.8);
}

/* Bottom-left system fact. */
.cs-footnote { max-width: 46ch; }
.cs-footnote__label {
  font-size: 0.62rem; font-weight: 700; letter-spacing: 0.2em;
  text-transform: uppercase; color: var(--cs-cyan-soft);
  margin-bottom: 0.5rem;
}
.cs-footnote__body {
  font-size: 0.82rem; line-height: 1.62; color: rgba(216, 234, 243, 0.86);
  text-shadow: 0 1px 16px rgba(0, 0, 0, 0.85);
}

/* Bottom-right feature card: dark glass, hairline border.  Explicitly not a
   glowing cyan frame — that reads as a UI demo rather than a title card. */
.cs-feature {
  background: rgba(0, 8, 15, 0.62);
  backdrop-filter: blur(20px) saturate(1.15);
  -webkit-backdrop-filter: blur(20px) saturate(1.15);
  border: 1px solid rgba(255, 255, 255, 0.11);
  border-radius: 14px;
  padding: 1.25rem 1.4rem 1.1rem;
  box-shadow: 0 24px 60px -34px rgba(0, 0, 0, 0.95);
}
.cs-feature__label {
  font-size: 0.62rem; font-weight: 700; letter-spacing: 0.2em;
  text-transform: uppercase; color: var(--cs-cyan-soft);
  margin-bottom: 0.9rem;
}
.cs-feature__row {
  display: flex; justify-content: space-between; align-items: baseline; gap: 1rem;
  padding: 0.42rem 0;
  border-top: 1px solid rgba(255, 255, 255, 0.07);
}
.cs-feature__row:first-of-type { border-top: none; }
.cs-feature__k { font-size: 0.8rem; color: rgba(206, 228, 240, 0.78); }
.cs-feature__v { font-size: 0.84rem; font-weight: 650; color: #eaf7fc; }

.cs-scrollcue {
  display: flex; align-items: center; gap: 0.75rem;
  margin-top: 2.2rem;
  font-size: 0.66rem; font-weight: 600; letter-spacing: 0.22em;
  text-transform: uppercase; color: rgba(190, 231, 245, 0.55);
}
.cs-scrollcue__line {
  width: 46px; height: 1px;
  background: linear-gradient(90deg, rgba(190, 231, 245, 0.6), transparent);
}

/* Narrow screens: the landing keeps its shape but gives up the fixed viewport
   height, because a phone's address bar makes 100vh unreliable and the stacked
   Streamlit columns need the room. */
@media (max-width: 820px) {
  .cs-landing { min-height: auto; padding-top: 1rem; }
  .cs-footnote, .cs-feature { margin-top: 1.4rem; }
  .cs-landing__title { font-size: clamp(2.6rem, 12vw, 4rem); }
  .cs-landing__meta { gap: 1.2rem; }
  .cs-landing__rule { margin: 1.4rem 0 1rem; }
  .cs-footnote { max-width: none; }
  /* The horizontal scrim assumes a wide frame; on a phone the copy sits over
     the middle of the shot, so darken evenly instead. */
  .cs-stage__scrim {
    background: linear-gradient(180deg, rgba(1, 10, 18, 0.80), rgba(1, 12, 22, 0.88));
  }
}

/* ---------- Media credits (required by both CC BY licences) --------------- */
.cs-credits {
  margin-top: 1.6rem; padding-top: 0.9rem;
  border-top: 1px solid var(--cs-border-soft);
}
.cs-credit { font-size: 0.7rem; line-height: 1.7; color: var(--cs-text-dim); }
.cs-credit__title { color: var(--cs-text-muted); }
.cs-credit a { color: var(--cs-text-muted); text-decoration: underline; }
.cs-credit a:hover { color: var(--cs-cyan-soft); }
.cs-credit__changes { color: var(--cs-text-dim); opacity: 0.8; }

/* ---------- Top navigation ------------------------------------------------
   st.navigation(position="top") renders its own header bar.  It is restyled
   here as a strip of glass over the media rather than a solid toolbar, which
   is what stops the landing reading as a dashboard with a banner. */
[data-testid="stTopNav"],
[data-testid="stHeader"] {
  background: linear-gradient(180deg, rgba(1, 10, 18, 0.72), rgba(1, 10, 18, 0.10));
  backdrop-filter: blur(14px) saturate(1.1);
  -webkit-backdrop-filter: blur(14px) saturate(1.1);
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}
[data-testid="stTopNav"] a,
[data-testid="stTopNav"] button {
  font-size: 0.82rem;
  font-weight: 600;
  letter-spacing: 0.01em;
  color: rgba(214, 236, 246, 0.78);
  border-radius: 9px;
}
[data-testid="stTopNav"] a:hover,
[data-testid="stTopNav"] button:hover {
  color: #eaf7fc;
  background: rgba(255, 255, 255, 0.07);
}
[data-testid="stTopNav"] a[aria-current="page"] {
  color: var(--cs-foam);
  background: rgba(34, 211, 238, 0.12);
  box-shadow: inset 0 -2px 0 var(--cs-aqua);
}

/* Page links used as calls to action on the landing. */
[data-testid="stPageLink"] a {
  border: 1px solid rgba(255, 255, 255, 0.16);
  border-radius: 999px;
  padding: 0.5rem 1.1rem;
  background: rgba(0, 8, 15, 0.5);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  font-weight: 620;
  transition: background 0.16s ease, border-color 0.16s ease, transform 0.16s ease;
}
[data-testid="stPageLink"] a:hover {
  background: rgba(34, 211, 238, 0.16);
  border-color: rgba(34, 211, 238, 0.5);
  transform: translateY(-1px);
  text-decoration: none;
}

/* ---------- Bathymetric map frame ---------------------------------------- */
/* streamlit-folium renders Leaflet inside its own iframe; only the frame is
   themeable from here — the map's own palette lives in reefmap.py. */
[data-testid="stIFrame"], iframe[title$="st_folium"] {
  border-radius: var(--cs-radius);
  border: 1px solid var(--cs-border-soft);
  box-shadow: var(--cs-shadow-soft);
  background: var(--cs-void);
}

.cs-attrib {
  margin-top: 0.55rem;
  font-size: 0.7rem; line-height: 1.6;
  color: var(--cs-text-dim);
  border-left: 2px solid var(--cs-border-soft);
  padding-left: 0.7rem;
}
.cs-attrib b { color: var(--cs-text-muted); font-weight: 600; }

/* ---------- Sonar metric card (compact readout) -------------------------- */
.cs-sonar {
  display: flex; flex-wrap: wrap; gap: 1.8rem;
  background: linear-gradient(150deg, rgba(5, 63, 92, 0.40), rgba(2, 11, 20, 0.72));
  backdrop-filter: var(--cs-blur-sm);
  -webkit-backdrop-filter: var(--cs-blur-sm);
  border: 1px solid var(--cs-border-soft);
  border-left: 3px solid var(--cs-sonar-accent, var(--cs-ocean-bright));
  border-radius: var(--cs-radius);
  padding: 1rem 1.25rem;
}
.cs-sonar__cell { min-width: 92px; }
.cs-sonar__k {
  font-size: 0.64rem; font-weight: 700; letter-spacing: 0.16em;
  text-transform: uppercase; color: var(--cs-text-dim);
}
.cs-sonar__v {
  margin-top: 0.22rem;
  font-family: var(--cs-mono);
  font-size: 1.15rem; font-weight: 700; letter-spacing: -0.01em;
  font-variant-numeric: tabular-nums;
  color: var(--cs-sonar-value, var(--cs-text-bright));
}

/* ---------- Motion preferences ------------------------------------------- */
@media (prefers-reduced-motion: reduce) {
  .cs-status__dot--live { animation: none; }
  .cs-stat, .cs-row, .stButton > button { transition: none; }
}

/* ---------- Narrow screens ----------------------------------------------- */
@media (max-width: 900px) {
  .cs-page__title { font-size: 1.5rem; }
}
"""


def _stylesheet() -> str:
    """Return the complete ``<style>`` block, tokens first.

    One block, so the cascade order between page chrome and the media behind it
    is fixed here rather than by markdown call order.
    """
    variables = "\n".join(f"  --cs-{name}: {value};" for name, value in _TOKENS.items())
    return f"<style>\n:root {{\n{variables}\n}}\n{_BASE_CSS}</style>"


def inject_theme() -> None:
    """
    Inject the global stylesheet.

    Called once per script run by ``components.set_page`` so every page is
    themed without repeating any CSS.
    """
    st.markdown(_stylesheet(), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 3. Themed components
# ---------------------------------------------------------------------------


def _html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def page_header(
    title: str,
    subtitle: str | None = None,
    *,
    eyebrow: str | None = None,
) -> None:
    """Render the standard header used at the top of every non-home page."""
    parts = ['<div class="cs-page"><div class="cs-page__bar"></div><div>']
    if eyebrow:
        parts.append(f'<div class="cs-page__eyebrow">{eyebrow}</div>')
    parts.append(f'<h1 class="cs-page__title">{title}</h1>')
    if subtitle:
        parts.append(f'<p class="cs-page__sub">{subtitle}</p>')
    parts.append("</div></div>")
    _html("".join(parts))


def section(title: str, subtitle: str | None = None, *, kicker: str | None = None) -> None:
    """Render a section heading with a trailing rule and optional description."""
    parts = ['<div class="cs-section">']
    if kicker:
        parts.append(f'<div class="cs-section__kicker">{kicker}</div>')
    parts.append(f'<h2 class="cs-section__title">{title}</h2>')
    if subtitle:
        parts.append(f'<p class="cs-section__sub">{subtitle}</p>')
    parts.append("</div>")
    _html("".join(parts))


def group(title: str, *, step: int | None = None) -> None:
    """Render a compact subsection label, used to break up long forms."""
    num = f'<span class="cs-group__num">{step}</span>' if step is not None else ""
    _html(f'<div class="cs-group">{num}{title}</div>')


def stat_card_html(
    label: str,
    value: str,
    *,
    caption: str | None = None,
    accent: str = AQUA,
    compact: bool = False,
) -> str:
    """Return the markup for a single statistic card."""
    value_cls = "cs-stat__value cs-stat__value--sm" if compact else "cs-stat__value"
    caption_html = f'<div class="cs-stat__caption">{caption}</div>' if caption else ""
    return (
        f'<div class="cs-stat" style="--cs-stat-accent:{accent}">'
        f'<div class="cs-stat__label">{label}</div>'
        f'<div class="{value_cls}">{value}</div>'
        f"{caption_html}"
        f"</div>"
    )


def stat_row(stats: Sequence[Mapping[str, Any]]) -> None:
    """
    Render a row of statistic cards, one per column.

    Each mapping accepts ``label``, ``value`` and the optional keys
    ``caption``, ``accent`` and ``compact``.
    """
    if not stats:
        return
    for column, stat in zip(st.columns(len(stats)), stats, strict=False):
        with column:
            _html(
                stat_card_html(
                    str(stat.get("label", "")),
                    str(stat.get("value", "—")),
                    caption=stat.get("caption"),
                    accent=str(stat.get("accent", AQUA)),
                    compact=bool(stat.get("compact", False)),
                )
            )


def panel(
    body: str,
    *,
    label: str | None = None,
    title: str | None = None,
    accent: str | None = None,
) -> None:
    """Render a static glass panel. ``body`` may contain inline HTML."""
    style = f' style="border-left:3px solid {accent}"' if accent else ""
    parts = [f'<div class="cs-panel"{style}>']
    if label:
        colour = f' style="color:{accent}"' if accent else ""
        parts.append(f'<div class="cs-panel__label"{colour}>{label}</div>')
    if title:
        parts.append(f'<div class="cs-panel__title">{title}</div>')
    parts.append(f'<div class="cs-panel__body">{body}</div></div>')
    _html("".join(parts))


def badge(text: str, colour: str) -> str:
    """Return a soft-tinted pill badge in *colour*."""
    return (
        f'<span class="cs-badge" style="background:{colour}1f;color:{colour};'
        f'border:1px solid {colour}59">{text}</span>'
    )


def meter_html(label: str, fraction: float, colour: str, *, value_text: str | None = None) -> str:
    """
    Return a labelled horizontal meter.

    ``fraction`` is clamped to ``[0, 1]``; ``value_text`` overrides the
    right-hand readout (which defaults to a percentage).
    """
    pct = max(0.0, min(1.0, float(fraction))) * 100
    readout = value_text if value_text is not None else f"{pct:.1f}%"
    return (
        f'<div class="cs-meter">'
        f'<div class="cs-meter__head">'
        f'<span class="cs-meter__name" style="color:{colour}">{label}</span>'
        f'<span class="cs-meter__value">{readout}</span>'
        f"</div>"
        f'<div class="cs-meter__track">'
        f'<div class="cs-meter__fill" style="width:{pct:.1f}%;'
        f'background:linear-gradient(90deg,{colour}b3,{colour})"></div>'
        f"</div></div>"
    )


def meter(label: str, fraction: float, colour: str, *, value_text: str | None = None) -> None:
    """Render a single meter (see :func:`meter_html`)."""
    _html(meter_html(label, fraction, colour, value_text=value_text))


def status_row(
    *,
    name: str,
    description: str,
    accent: str,
    icon: str = "●",
    tag: str | None = None,
    category: str | None = None,
    status: str | None = None,
    files: str | None = None,
) -> None:
    """Render one row of a status/milestone list."""
    head = []
    if tag:
        head.append(f'<span class="cs-row__tag">{tag}</span>')
    head.append(f'<span class="cs-row__name">{name}</span>')
    if category:
        head.append(f'<span class="cs-row__cat">— {category}</span>')
    if status:
        head.append(f'<span class="cs-row__status">{status}</span>')
    files_html = f'<div class="cs-row__files">{files}</div>' if files else ""
    _html(
        f'<div class="cs-row" style="--cs-row-accent:{accent}">'
        f'<div class="cs-row__icon">{icon}</div>'
        f'<div class="cs-row__body">'
        f'<div class="cs-row__head">{"".join(head)}</div>'
        f'<div class="cs-row__desc">{description}</div>'
        f"{files_html}"
        f"</div></div>"
    )


def sonar_card_html(
    cells: Sequence[tuple[str, str] | tuple[str, str, str]],
    *,
    accent: str = OCEAN_BRIGHT,
) -> str:
    """
    Return a compact instrument readout: a row of ``(label, value)`` pairs in
    tabular mono type.

    An optional third element overrides the value colour for that cell.  This
    is the "compact metric / sonar card" member of the visualisation system —
    used where a chart would add noise but the exact number still has to be
    legible at a glance.
    """
    parts = [f'<div class="cs-sonar" style="--cs-sonar-accent:{accent}">']
    for cell in cells:
        label, value = cell[0], cell[1]
        colour = cell[2] if len(cell) > 2 else TEXT_BRIGHT
        parts.append(
            f'<div class="cs-sonar__cell">'
            f'<div class="cs-sonar__k">{label}</div>'
            f'<div class="cs-sonar__v" style="--cs-sonar-value:{colour}">{value}</div>'
            f"</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def sonar_card(
    cells: Sequence[tuple[str, str] | tuple[str, str, str]],
    *,
    accent: str = OCEAN_BRIGHT,
) -> None:
    """Render a compact instrument readout (see :func:`sonar_card_html`)."""
    _html(sonar_card_html(cells, accent=accent))


def attribution(entries: Sequence[tuple[str, str]]) -> None:
    """
    Render a data-source credit block.

    Each entry is ``(source, description)``.  Used wherever the dashboard
    displays third-party geographic data or an external map service.
    """
    rows = "".join(f"<div><b>{source}</b> — {detail}</div>" for source, detail in entries)
    _html(f'<div class="cs-attrib">{rows}</div>')


def spacer(height: str = "1rem") -> None:
    """Insert vertical space."""
    _html(f'<div style="height:{height}"></div>')


@contextmanager
def card():
    """
    Context manager yielding a themed bordered container.

    Usage::

        with card():
            st.selectbox(...)

    The hidden anchor lets the stylesheet upgrade this specific container to
    the glass surface; if Streamlit's DOM changes the container simply keeps
    its default border.
    """
    container = st.container(border=True)
    with container:
        _html('<span class="cs-card-anchor"></span>')
        yield container


# ---------------------------------------------------------------------------
# 4. Plotly theming
# ---------------------------------------------------------------------------

_AXIS_STYLE: dict[str, Any] = {
    "gridcolor": "rgba(125, 211, 252, 0.10)",
    "zerolinecolor": "rgba(125, 211, 252, 0.16)",
    "linecolor": BORDER_SOFT,
    "tickfont": {"color": TEXT_MUTED, "size": 11},
    "title": {"font": {"color": TEXT_MUTED, "size": 12}},
}


def base_layout() -> dict[str, Any]:
    """Return the shared Plotly layout applied to every figure."""
    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": FONT_STACK, "color": TEXT, "size": 12},
        "title": {
            "font": {"family": FONT_STACK, "color": TEXT_BRIGHT, "size": 15},
            "x": 0.012,
            "xanchor": "left",
        },
        "margin": {"l": 56, "r": 26, "t": 62, "b": 46},
        "colorway": list(COLORWAY),
        "hoverlabel": {
            "bgcolor": SURFACE,
            "bordercolor": BORDER,
            "font": {"family": FONT_STACK, "color": TEXT_BRIGHT, "size": 12},
        },
        "legend": {
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": "rgba(0,0,0,0)",
            "font": {"color": TEXT_MUTED, "size": 11},
            "title": {"font": {"color": CYAN, "size": 11}},
        },
    }


def style_figure(fig: Any, **overrides: Any) -> Any:
    """
    Apply the CoralSense chart theme to a Plotly figure and return it.

    Keyword arguments are forwarded to a final ``update_layout`` call, so
    per-chart settings such as ``showlegend=False`` or ``yaxis={"range": [0, 1]}``
    still win — Plotly merges layout updates rather than replacing them.

    Safe for every figure type: ``update_xaxes``/``update_yaxes`` are no-ops on
    maps and pie charts, which have no Cartesian axes.
    """
    fig.update_layout(**base_layout())
    fig.update_xaxes(**_AXIS_STYLE)
    fig.update_yaxes(**_AXIS_STYLE)
    if overrides:
        fig.update_layout(**overrides)
    return fig


def map_layout_overrides() -> dict[str, Any]:
    """Layout overrides for full-bleed map figures (no axis gutters)."""
    return {"margin": {"l": 0, "r": 0, "t": 46, "b": 0}}


__all__: tuple[str, ...] = (
    "ABYSS",
    "AQUA",
    "BASIN",
    "BORDER",
    "BORDER_SOFT",
    "COLORWAY",
    "CORAL",
    "CORAL_SOFT",
    "CYAN",
    "CYAN_SOFT",
    "DANGER",
    "DEEP",
    "DEPTH_COLORSCALE",
    "DEPTH_RAMP",
    "FOAM",
    "NAVY",
    "OCEAN_BRIGHT",
    "OCEAN_DEEP",
    "OCEAN_GLOW",
    "OCEAN_MID",
    "RADIUS",
    "SONAR_BG",
    "SONAR_MESH",
    "SONAR_MESH_SOFT",
    "SUCCESS",
    "SURFACE",
    "SURFACE_HI",
    "TEAL",
    "TEXT",
    "TEXT_BRIGHT",
    "TEXT_DIM",
    "TEXT_MUTED",
    "TRENCH",
    "TURQUOISE",
    "VOID",
    "WARNING",
    "attribution",
    "badge",
    "base_layout",
    "card",
    "group",
    "inject_theme",
    "map_layout_overrides",
    "meter",
    "meter_html",
    "page_header",
    "panel",
    "section",
    "sonar_card",
    "sonar_card_html",
    "spacer",
    "stat_card_html",
    "stat_row",
    "status_row",
    "style_figure",
)
