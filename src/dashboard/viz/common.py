"""
src/dashboard/viz/common.py — shared primitives for the visualisation system.

Contains the ocean colour ramp helpers, the shared "abyss" Plotly layout, the
histogram used by every distribution mountain, and — the important one — the
ridge profile that guarantees a mountain's summit equals its source value.

Deliberately imports only ``math``, ``plotly.graph_objects`` and the dashboard
theme.  No numpy, no scipy: the dashboard runtime manifest carries neither, and
the arithmetic here does not need them.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import plotly.graph_objects as go

from src.dashboard import theme

# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------

# Depth layers, deepest first — the stacking order for translucent mountain
# layers and for stream bands.
DEPTH_LAYERS: tuple[str, ...] = theme.DEPTH_RAMP


def _hex_to_rgb(colour: str) -> tuple[int, int, int]:
    value = colour.lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def rgba(colour: str, alpha: float) -> str:
    """Return ``colour`` as an ``rgba(...)`` string with the given alpha."""
    r, g, b = _hex_to_rgb(colour)
    return f"rgba({r}, {g}, {b}, {max(0.0, min(1.0, alpha)):.3f})"


def ramp_color(t: float, ramp: Sequence[str] = DEPTH_LAYERS) -> str:
    """
    Sample the ocean depth ramp at ``t`` in ``[0, 1]``.

    ``t = 0`` is the trench, ``t = 1`` is foam.  Interpolation is linear in
    sRGB, which is good enough for a decorative ramp and keeps this module free
    of a colour-science dependency.
    """
    if not ramp:
        return theme.AQUA
    t = max(0.0, min(1.0, float(t)))
    if len(ramp) == 1:
        return ramp[0]
    scaled = t * (len(ramp) - 1)
    i = min(len(ramp) - 2, int(scaled))
    f = scaled - i
    r0, g0, b0 = _hex_to_rgb(ramp[i])
    r1, g1, b1 = _hex_to_rgb(ramp[i + 1])
    return f"#{round(r0 + (r1 - r0) * f):02x}{round(g0 + (g1 - g0) * f):02x}{round(b0 + (b1 - b0) * f):02x}"


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

_AXIS: dict[str, Any] = {
    "showgrid": False,
    "zeroline": False,
    "showline": False,
    "ticks": "",
    "tickfont": {"color": theme.TEXT_MUTED, "size": 11},
    "title": {"font": {"color": theme.TEXT_DIM, "size": 11}},
}


def abyss_layout(**overrides: Any) -> dict[str, Any]:
    """
    Return the layout shared by every chart in this system.

    Transparent paper (the page's glass surface shows through), no chart grid,
    and a hover label that matches the dashboard chrome.  Individual builders
    re-enable exactly the axis furniture they need.
    """
    layout: dict[str, Any] = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": theme.FONT_STACK, "color": theme.TEXT, "size": 12},
        "title": {
            "font": {"family": theme.FONT_STACK, "color": theme.TEXT_BRIGHT, "size": 15},
            "x": 0.012,
            "xanchor": "left",
        },
        "margin": {"l": 56, "r": 26, "t": 62, "b": 48},
        "hoverlabel": {
            "bgcolor": theme.TRENCH,
            "bordercolor": theme.OCEAN_MID,
            "font": {"family": theme.FONT_STACK, "color": theme.FOAM, "size": 12},
        },
        "legend": {
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": "rgba(0,0,0,0)",
            "font": {"color": theme.TEXT_MUTED, "size": 11},
            "title": {"font": {"color": theme.OCEAN_BRIGHT, "size": 11}},
        },
        "xaxis": dict(_AXIS),
        "yaxis": dict(_AXIS),
    }
    layout.update(overrides)
    return layout


def style_viz(fig: go.Figure, **overrides: Any) -> go.Figure:
    """Apply :func:`abyss_layout` to *fig* and return it."""
    fig.update_layout(**abyss_layout(**overrides))
    return fig


# ---------------------------------------------------------------------------
# The ridge profile
# ---------------------------------------------------------------------------

# Number of samples across one ridge. Odd so a sample lands exactly on the
# summit — that sample is what carries the encoded value.
RIDGE_SAMPLES = 81


def ridge_profile(samples: int = RIDGE_SAMPLES, *, texture: float = 0.055) -> list[float]:
    """
    Return a normalised mountain silhouette on ``u ∈ [-1, 1]``.

    Guarantees, all of which the test suite asserts:

    * ``profile[len // 2] == 1.0`` exactly — the centre sample is the summit;
    * ``max(profile) == 1.0`` — nothing rises above the summit;
    * ``profile[0] == profile[-1] == 0.0`` — the ridge meets the baseline.

    The shape is a blend of a raised cosine (the massif) and a cone (the
    shoulders), both of which are exactly 1 at ``u = 0`` and exactly 0 at
    ``|u| = 1``.  ``texture`` adds the topographic asymmetry that stops the
    ridges looking like bell curves; its term carries a ``sin(k·π·u)`` factor
    that is identically zero at the summit and a ``(1 - u²)`` envelope that is
    zero at the feet, so it can move the flanks but never the peak.  The result
    is clamped to 1 as a belt-and-braces guard.

    Because the profile is normalised, a caller multiplies it by the real
    metric and the summit height *is* that metric.
    """
    n = max(3, int(samples) | 1)  # force odd
    profile: list[float] = []
    for i in range(n):
        u = -1.0 + 2.0 * i / (n - 1)
        massif = 0.5 * (1.0 + math.cos(math.pi * u))
        shoulder = (1.0 - abs(u)) ** 1.6
        base = 0.62 * massif + 0.38 * shoulder
        ridge = texture * math.sin(3.0 * math.pi * u) * (1.0 - u * u)
        profile.append(min(1.0, max(0.0, base + ridge)))
    mid = n // 2
    profile[mid] = 1.0  # u == 0 analytically; pin it against float drift
    profile[0] = 0.0
    profile[-1] = 0.0
    return profile


def ridge_xy(
    centre: float,
    peak: float,
    *,
    half_width: float = 0.46,
    samples: int = RIDGE_SAMPLES,
    texture: float = 0.055,
    baseline: float = 0.0,
) -> tuple[list[float], list[float]]:
    """
    Return ``(xs, ys)`` for one ridge centred at *centre* with summit *peak*.

    ``max(ys) - baseline`` equals ``peak`` exactly (for ``peak >= 0``), and the
    summit sits at ``x == centre``.
    """
    profile = ridge_profile(samples, texture=texture)
    n = len(profile)
    xs = [centre + half_width * (-1.0 + 2.0 * i / (n - 1)) for i in range(n)]
    ys = [baseline + peak * p for p in profile]
    return xs, ys


# ---------------------------------------------------------------------------
# Histogram (used by every distribution mountain)
# ---------------------------------------------------------------------------


def histogram(
    values: Iterable[float],
    *,
    bins: int = 26,
    lo: float | None = None,
    hi: float | None = None,
) -> tuple[list[float], list[int], float]:
    """
    Bin *values* and return ``(bin_centres, counts, bin_width)``.

    Plain counting — no kernel, no smoothing, no density estimate.  What comes
    out is what is in the data, so a distribution mountain drawn from this is a
    faithful histogram wearing a different coat.

    Non-finite values are dropped (and therefore excluded from every count).
    """
    data = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    n_bins = max(1, int(bins))
    if not data:
        return [], [], 0.0

    low = float(lo) if lo is not None else min(data)
    high = float(hi) if hi is not None else max(data)
    if high <= low:
        high = low + 1.0
    width = (high - low) / n_bins

    counts = [0] * n_bins
    for value in data:
        idx = int((value - low) / width)
        if idx >= n_bins:  # the maximum lands on the closed right edge
            idx = n_bins - 1
        elif idx < 0:
            idx = 0
        counts[idx] += 1

    centres = [low + width * (i + 0.5) for i in range(n_bins)]
    return centres, counts, width


def quantile(values: Sequence[float], q: float) -> float | None:
    """
    Return the linear-interpolated *q* quantile of *values*.

    Used only to place an exact median marker on a distribution mountain; the
    value is reported verbatim in the annotation, never used to reshape data.
    """
    data = sorted(
        float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))
    )
    if not data:
        return None
    if len(data) == 1:
        return data[0]
    pos = max(0.0, min(1.0, float(q))) * (len(data) - 1)
    low = int(math.floor(pos))
    high = min(len(data) - 1, low + 1)
    frac = pos - low
    return data[low] + (data[high] - data[low]) * frac


def fmt(value: float, digits: int = 4) -> str:
    """Format a metric for display without scientific notation surprises."""
    if value is None:
        return "—"
    if not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}f}"


__all__ = (
    "DEPTH_LAYERS",
    "RIDGE_SAMPLES",
    "abyss_layout",
    "fmt",
    "histogram",
    "quantile",
    "ramp_color",
    "rgba",
    "ridge_profile",
    "ridge_xy",
    "style_viz",
)
