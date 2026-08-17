"""
src/dashboard/viz/mountain.py — layered topographic mountain charts.

Two builders:

``mountain_chart``
    Replaces the bar chart.  Each category becomes a mountain profile whose
    **summit height is exactly the source metric**.  Behind each summit sit two
    or three translucent ridges at fixed fractions of that same metric, which
    is what gives the stacked-paper topographic depth — they are decorative,
    they are mathematically tied to the real value, and they are drawn behind
    and dimmer so they can never be mistaken for the encoded mark.

``ridgeline_chart``
    Replaces the box / violin plot.  One histogram-derived mountain per class,
    stacked as offset ridges.  The outline is a straight polyline through the
    real bin counts, so no smoothing hides a spike.

Both put the exact number in the hover *and* print it beside the mark.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import plotly.graph_objects as go

from src.dashboard import theme
from src.dashboard.viz.common import (
    abyss_layout,
    fmt,
    histogram,
    quantile,
    ramp_color,
    rgba,
    ridge_xy,
)

# Backing layers: (height fraction of the true value, width multiplier, ramp
# position, opacity).  Deepest and widest first so the foreground ridge reads
# as the near range of a topographic diagram.
_BACKDROP_LAYERS: tuple[tuple[float, float, float, float], ...] = (
    (0.58, 1.90, 0.05, 0.30),
    (0.74, 1.48, 0.22, 0.34),
    (0.88, 1.16, 0.40, 0.38),
)


def mountain_chart(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    value_name: str = "Value",
    highlight: str | None = None,
    highlight_color: str = theme.SUCCESS,
    errors: Sequence[float] | None = None,
    error_name: str = "± std",
    digits: int = 4,
    y_range: tuple[float, float] | None = None,
    title: str | None = None,
    height: int = 420,
    show_values: bool = True,
) -> go.Figure:
    """
    Build a layered mountain-ridge comparison chart.

    Parameters
    ----------
    labels, values:
        One mountain per label.  ``values`` are used verbatim: the summit of
        mountain *i* sits at exactly ``values[i]``.
    highlight:
        Label to render in ``highlight_color`` (the champion treatment).
    errors:
        Optional symmetric uncertainties, drawn as a whisker through the
        summit.  Purely additive — the summit itself does not move.
    y_range:
        Axis range, e.g. ``(0, 1)`` for an F1 score.

    Raises
    ------
    ValueError
        If ``labels`` and ``values`` differ in length.
    """
    if len(labels) != len(values):
        raise ValueError(f"labels/values length mismatch: {len(labels)} vs {len(values)}")
    if errors is not None and len(errors) != len(values):
        raise ValueError(f"errors length mismatch: {len(errors)} vs {len(values)}")

    fig = go.Figure()
    n = len(labels)

    for i, (label, raw) in enumerate(zip(labels, values, strict=True)):
        value = float(raw)
        is_champion = highlight is not None and label == highlight
        # Foreground colour walks the ramp so a wide comparison still reads as
        # an ordered ocean gradient rather than a random colour cycle.
        crest = highlight_color if is_champion else ramp_color(0.34 + 0.5 * (i / max(1, n - 1)))

        # --- decorative depth layers, all tied to the same metric -----------
        for frac, width_mult, ramp_pos, alpha in _BACKDROP_LAYERS:
            xs, ys = ridge_xy(
                i,
                value * frac,
                half_width=0.46 * width_mult,
                texture=0.075,
            )
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line={"width": 0},
                    fill="tozeroy",
                    fillcolor=rgba(ramp_color(ramp_pos), alpha),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        # --- the encoded ridge ---------------------------------------------
        xs, ys = ridge_xy(i, value, half_width=0.46)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line={"width": 1.8, "color": crest},
                fill="tozeroy",
                fillcolor=rgba(crest, 0.30 if not is_champion else 0.38),
                name=str(label),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        # --- the summit: the mark that carries the number -------------------
        error_text = ""
        if errors is not None:
            error_text = f"<br>{error_name}: {fmt(float(errors[i]), digits)}"
        fig.add_trace(
            go.Scatter(
                x=[i],
                y=[value],
                mode="markers",
                marker={
                    "size": 9,
                    "color": crest,
                    "line": {"width": 2, "color": theme.VOID},
                    "symbol": "diamond",
                },
                name=str(label),
                showlegend=False,
                hovertemplate=(
                    f"<b>{label}</b><br>{value_name}: "
                    f"<b>{fmt(value, digits)}</b>{error_text}"
                    + ("<br>champion" if is_champion else "")
                    + "<extra></extra>"
                ),
            )
        )

        if errors is not None:
            err = abs(float(errors[i]))
            fig.add_trace(
                go.Scatter(
                    x=[i, i],
                    y=[value - err, value + err],
                    mode="lines",
                    line={"width": 1.4, "color": rgba(crest, 0.75), "dash": "dot"},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        if show_values:
            fig.add_annotation(
                x=i,
                y=value,
                text=f"<b>{fmt(value, digits)}</b>",
                showarrow=False,
                yshift=22,
                font={"color": crest, "size": 13, "family": theme.MONO_STACK},
            )

    # Baseline: the shoreline the whole range sits on.
    fig.add_shape(
        type="line",
        x0=-0.6,
        x1=n - 0.4,
        y0=0,
        y1=0,
        line={"color": rgba(theme.OCEAN_MID, 0.55), "width": 1},
    )

    layout = abyss_layout(
        height=height,
        showlegend=False,
        hovermode="closest",
        xaxis={
            "tickmode": "array",
            "tickvals": list(range(n)),
            "ticktext": [str(label) for label in labels],
            "range": [-0.62, n - 0.38],
            "showgrid": False,
            "zeroline": False,
            "showline": False,
            "ticks": "",
            "tickfont": {"color": theme.TEXT, "size": 12},
        },
        yaxis={
            "title": {"text": value_name, "font": {"color": theme.TEXT_DIM, "size": 11}},
            "showgrid": True,
            "gridcolor": rgba(theme.OCEAN_MID, 0.10),
            "zeroline": False,
            "showline": False,
            "ticks": "",
            "tickfont": {"color": theme.TEXT_MUTED, "size": 11},
        },
    )
    if title:
        layout["title"] = {"text": title}
    if y_range is not None:
        layout["yaxis"]["range"] = list(y_range)
    fig.update_layout(**layout)
    return fig


def ridgeline_chart(
    groups: Sequence[str],
    series: Mapping[str, Sequence[float]],
    *,
    colors: Mapping[str, str] | None = None,
    bins: int = 30,
    axis_title: str = "",
    title: str | None = None,
    height: int = 360,
    overlap: float = 0.62,
    digits: int = 2,
) -> go.Figure:
    """
    Build a layered distribution mountain (one ridge per group).

    Each ridge is the group's histogram over a **shared** bin edge set, so the
    ridges are directly comparable.  The outline connects real bin counts with
    straight segments; nothing is smoothed away.  Hover reports the bin's exact
    interval and its exact row count, and the median of each group is marked
    and labelled with its exact value.

    ``overlap`` controls how far ridges intrude on their neighbour's lane —
    purely visual, it never changes a count.
    """
    present = [g for g in groups if g in series and len(series[g])]
    fig = go.Figure()
    if not present:
        return fig.update_layout(**abyss_layout(height=height))

    # Shared range across every group so ridges line up on the x axis.
    all_values = [float(v) for g in present for v in series[g]]
    lo, hi = min(all_values), max(all_values)

    per_group = {g: histogram(series[g], bins=bins, lo=lo, hi=hi) for g in present}
    peak = max((max(counts) if counts else 0) for _, counts, _ in per_group.values()) or 1
    lane = 1.0
    amplitude = lane * (1.0 + max(0.0, overlap))

    # Draw far ridges first so nearer ones overlap them, like a range of hills.
    for index, group in enumerate(reversed(present)):
        i = len(present) - 1 - index
        centres, counts, width = per_group[group]
        base = i * lane
        colour = (colors or {}).get(group) or ramp_color(0.3 + 0.6 * (i / max(1, len(present) - 1)))

        xs = [centres[0]] + list(centres) + [centres[-1]]
        ys = [base] + [base + amplitude * (c / peak) for c in counts] + [base]
        total = sum(counts)
        customdata = (
            [(0, 0.0, 0.0, 0)]
            + [
                (c, centre - width / 2, centre + width / 2, total)
                for c, centre in zip(counts, centres, strict=True)
            ]
            + [(0, 0.0, 0.0, 0)]
        )

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                customdata=customdata,
                mode="lines",
                line={"width": 1.6, "color": colour, "shape": "linear"},
                fill="toself",
                fillcolor=rgba(colour, 0.34),
                name=str(group).replace("_", " ").title(),
                hovertemplate=(
                    f"<b>{str(group).replace('_', ' ').title()}</b><br>"
                    "%{customdata[1]:.3f} – %{customdata[2]:.3f}<br>"
                    "count: <b>%{customdata[0]:,}</b> of %{customdata[3]:,}"
                    "<extra></extra>"
                ),
            )
        )

        med = quantile(series[group], 0.5)
        if med is not None:
            fig.add_trace(
                go.Scatter(
                    x=[med],
                    y=[base],
                    mode="markers+text",
                    marker={
                        "size": 8,
                        "color": colour,
                        "symbol": "line-ns-open",
                        "line": {"width": 2, "color": colour},
                    },
                    text=[f"median {fmt(med, digits)}"],
                    textposition="top right",
                    textfont={"color": rgba(colour, 0.95), "size": 10},
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{str(group).replace('_', ' ').title()}</b><br>"
                        f"median: <b>{fmt(med, digits)}</b><extra></extra>"
                    ),
                )
            )

    layout = abyss_layout(
        height=height,
        showlegend=False,
        hovermode="closest",
        margin={"l": 132, "r": 26, "t": 58 if title else 24, "b": 46},
        xaxis={
            "title": {"text": axis_title, "font": {"color": theme.TEXT_DIM, "size": 11}},
            "showgrid": True,
            "gridcolor": rgba(theme.OCEAN_MID, 0.09),
            "zeroline": False,
            "showline": False,
            "ticks": "",
            "tickfont": {"color": theme.TEXT_MUTED, "size": 11},
        },
        yaxis={
            "tickmode": "array",
            "tickvals": [i * lane for i in range(len(present))],
            "ticktext": [str(g).replace("_", " ").title() for g in present],
            "showgrid": False,
            "zeroline": False,
            "showline": False,
            "ticks": "",
            "tickfont": {"color": theme.TEXT, "size": 11},
        },
    )
    if title:
        layout["title"] = {"text": title}
    fig.update_layout(**layout)
    return fig


__all__: tuple[str, ...] = ("mountain_chart", "ridgeline_chart")
