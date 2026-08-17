"""
src/dashboard/viz/stream.py — layered stream / mirrored mountain charts.

``stream_chart``
    A streamgraph: stacked non-negative bands over an ordered axis, with the
    stack centred on a common baseline.  Centring moves the *baseline* only —
    every band's thickness is still exactly its source value, and that value is
    what the hover reports.

``mirrored_stream``
    Two stacks sharing one baseline, one growing up and one growing down.  Used
    **only** where the two directions are genuinely different semantic
    dimensions (reference window vs production window, for example).  The
    downward stack is plotted at negative screen coordinates but the axis ticks
    and the hover both show the positive magnitude, because the underlying
    values are positive and inventing a negative would be a lie.

Neither builder ever fabricates a value to make a shape symmetric.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import plotly.graph_objects as go

from src.dashboard import theme
from src.dashboard.viz.common import abyss_layout, ramp_color, rgba

# Cosine-eased interpolation between category stations. This softens the band
# boundaries into organic ridges without moving any station's value: at every
# station x = i the curve passes exactly through the stacked value.
_SUBSTEPS = 14


def _organic_path(values: Sequence[float]) -> tuple[list[float], list[float]]:
    """
    Return ``(xs, ys)`` easing between consecutive station values.

    The curve is pinned: for every station index ``i`` there is a sample at
    ``x == i`` whose ``y`` is exactly ``values[i]``.  Only the space *between*
    stations is interpolated, so a reader hovering a station always sees the
    real number.
    """
    xs: list[float] = []
    ys: list[float] = []
    for i in range(len(values)):
        xs.append(float(i))
        ys.append(float(values[i]))
        if i == len(values) - 1:
            break
        for s in range(1, _SUBSTEPS):
            t = s / _SUBSTEPS
            eased = 0.5 - 0.5 * math.cos(math.pi * t)
            xs.append(i + t)
            ys.append(values[i] + (values[i + 1] - values[i]) * eased)
    return xs, ys


def _stack(
    categories: Sequence[str],
    series: Mapping[str, Sequence[float]],
    keys: Sequence[str],
) -> dict[str, list[float]]:
    """Cumulative upper edge for each key (bands are stacked in ``keys`` order)."""
    running = [0.0] * len(categories)
    edges: dict[str, list[float]] = {}
    for key in keys:
        values = series[key]
        running = [running[i] + float(values[i]) for i in range(len(categories))]
        edges[key] = list(running)
    return edges


def stream_chart(
    categories: Sequence[str],
    series: Mapping[str, Sequence[float]],
    *,
    keys: Sequence[str] | None = None,
    colors: Mapping[str, str] | None = None,
    value_name: str = "Value",
    value_format: str = ",.0f",
    centred: bool = True,
    title: str | None = None,
    height: int = 400,
    show_labels: bool = True,
) -> go.Figure:
    """
    Build a layered streamgraph.

    Parameters
    ----------
    categories:
        Ordered stations along the x axis (regions, windows, time buckets…).
    series:
        ``{band name: value per category}``.  Values must be non-negative —
        a stream band's thickness is a magnitude.
    centred:
        Centre the stack on a shared baseline (the classic streamgraph
        silhouette).  This is a baseline offset only; band thickness and hover
        values are untouched.
    show_labels:
        Print each band's name inside its thickest station.

    Raises
    ------
    ValueError
        If a series length does not match ``categories``, or a value is
        negative (which a stacked stream cannot honestly represent).
    """
    order = list(keys) if keys is not None else list(series.keys())
    order = [k for k in order if k in series]
    n = len(categories)

    for key in order:
        if len(series[key]) != n:
            raise ValueError(f"series '{key}' has {len(series[key])} values, expected {n}")
        for value in series[key]:
            if float(value) < 0:
                raise ValueError(
                    f"series '{key}' contains a negative value ({value}); a stream band "
                    "encodes a magnitude and this builder never flips a sign to create "
                    "visual symmetry"
                )

    fig = go.Figure()
    if not order or n == 0:
        return fig.update_layout(**abyss_layout(height=height))

    edges = _stack(categories, series, order)
    totals = edges[order[-1]]
    offset = [-t / 2.0 for t in totals] if centred else [0.0] * n

    lower = list(offset)
    for index, key in enumerate(order):
        upper = [edges[key][i] + offset[i] for i in range(n)]
        colour = (colors or {}).get(key) or ramp_color(
            0.18 + 0.72 * (index / max(1, len(order) - 1))
        )

        up_x, up_y = _organic_path(upper)
        lo_x, lo_y = _organic_path(lower)
        band_values = list(series[key])

        # Hover carries the raw value at each station via a marker trace, so the
        # eased fill never has to be trusted for a number.
        fig.add_trace(
            go.Scatter(
                x=up_x + lo_x[::-1],
                y=up_y + lo_y[::-1],
                mode="lines",
                line={"width": 1.1, "color": rgba(colour, 0.85)},
                fill="toself",
                fillcolor=rgba(colour, 0.55),
                name=str(key).replace("_", " ").title(),
                hoverinfo="skip",
                legendgroup=str(key),
            )
        )
        mid = [(upper[i] + lower[i]) / 2.0 for i in range(n)]
        fig.add_trace(
            go.Scatter(
                x=list(range(n)),
                y=mid,
                mode="markers",
                marker={"size": 6, "color": rgba(colour, 0.0)},
                customdata=[[categories[i], band_values[i]] for i in range(n)],
                name=str(key).replace("_", " ").title(),
                showlegend=False,
                legendgroup=str(key),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    f"{str(key).replace('_', ' ').title()}<br>"
                    f"{value_name}: <b>%{{customdata[1]:{value_format}}}</b>"
                    "<extra></extra>"
                ),
            )
        )

        if show_labels:
            # Label inside the band's thickest station — the only place the
            # text is guaranteed to fit.
            thicknesses = [upper[i] - lower[i] for i in range(n)]
            thickest = thicknesses.index(max(thicknesses))
            thickness = thicknesses[thickest]
            if thickness > (max(totals) or 1) * 0.10:
                fig.add_annotation(
                    x=thickest,
                    y=(upper[thickest] + lower[thickest]) / 2.0,
                    text=str(key).replace("_", " ").title(),
                    showarrow=False,
                    font={"color": theme.FOAM, "size": 11, "family": theme.FONT_STACK},
                    opacity=0.88,
                )

        lower = upper

    layout = abyss_layout(
        height=height,
        hovermode="closest",
        xaxis={
            "tickmode": "array",
            "tickvals": list(range(n)),
            "ticktext": [str(c) for c in categories],
            "range": [-0.15, n - 0.85],
            "showgrid": False,
            "zeroline": False,
            "showline": False,
            "ticks": "",
            "tickfont": {"color": theme.TEXT, "size": 11},
        },
        yaxis={
            "visible": False,
            "showgrid": False,
            "zeroline": False,
        },
        legend={
            "orientation": "h",
            "y": -0.16,
            "x": 0,
            "font": {"color": theme.TEXT_MUTED, "size": 11},
        },
    )
    if title:
        layout["title"] = {"text": title}
    fig.update_layout(**layout)
    return fig


def mirrored_stream(
    categories: Sequence[str],
    up_series: Mapping[str, Sequence[float]],
    down_series: Mapping[str, Sequence[float]],
    *,
    up_name: str,
    down_name: str,
    keys: Sequence[str] | None = None,
    colors: Mapping[str, str] | None = None,
    value_name: str = "Share",
    value_format: str = ".3f",
    title: str | None = None,
    height: int = 380,
) -> go.Figure:
    """
    Build a mirrored stream: ``up_series`` above the baseline, ``down_series``
    below it.

    Legitimate only because the two directions are two real, different
    measurements of the same categories — ``up_name`` and ``down_name`` name
    them, and both are shown on the chart.  All inputs must be non-negative;
    the sign on the lower half is a drawing convention, and the hover restores
    the true positive magnitude.

    Raises
    ------
    ValueError
        On a length mismatch or a negative input.
    """
    order = (
        list(keys)
        if keys is not None
        else list(dict.fromkeys([*up_series.keys(), *down_series.keys()]))
    )
    n = len(categories)
    fig = go.Figure()
    if not order or n == 0:
        return fig.update_layout(**abyss_layout(height=height))

    for name, block in (("upper", up_series), ("lower", down_series)):
        for key in order:
            values = block.get(key)
            if values is None:
                continue
            if len(values) != n:
                raise ValueError(f"{name} series '{key}' has {len(values)} values, expected {n}")
            if any(float(v) < 0 for v in values):
                raise ValueError(
                    f"{name} series '{key}' contains a negative value; the mirror is a "
                    "drawing convention over two positive measurements, not a sign flip"
                )

    def _half(block: Mapping[str, Sequence[float]], sign: int, window: str) -> None:
        running = [0.0] * n
        for index, key in enumerate(order):
            values = [float(v) for v in block.get(key, [0.0] * n)]
            upper = [running[i] + values[i] for i in range(n)]
            colour = (colors or {}).get(key) or ramp_color(
                0.2 + 0.7 * (index / max(1, len(order) - 1))
            )
            up_x, up_y = _organic_path([sign * v for v in upper])
            lo_x, lo_y = _organic_path([sign * v for v in running])
            fig.add_trace(
                go.Scatter(
                    x=up_x + lo_x[::-1],
                    y=up_y + lo_y[::-1],
                    mode="lines",
                    line={"width": 1.0, "color": rgba(colour, 0.85)},
                    fill="toself",
                    fillcolor=rgba(colour, 0.52 if sign > 0 else 0.34),
                    name=str(key).replace("_", " ").title(),
                    legendgroup=str(key),
                    showlegend=sign > 0,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=list(range(n)),
                    y=[sign * (running[i] + upper[i]) / 2.0 for i in range(n)],
                    mode="markers",
                    marker={"size": 6, "color": rgba(colour, 0.0)},
                    customdata=[[categories[i], values[i]] for i in range(n)],
                    showlegend=False,
                    legendgroup=str(key),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        f"{window} · {str(key).replace('_', ' ').title()}<br>"
                        f"{value_name}: <b>%{{customdata[1]:{value_format}}}</b>"
                        "<extra></extra>"
                    ),
                )
            )
            running = upper

    _half(up_series, +1, up_name)
    _half(down_series, -1, down_name)

    fig.add_shape(
        type="line",
        x0=-0.15,
        x1=n - 0.85,
        y0=0,
        y1=0,
        line={"color": rgba(theme.FOAM, 0.55), "width": 1.4},
    )
    fig.add_annotation(
        x=-0.12,
        y=0,
        xanchor="left",
        yanchor="bottom",
        yshift=6,
        text=f"▲ {up_name}",
        showarrow=False,
        font={"color": theme.OCEAN_GLOW, "size": 10, "family": theme.FONT_STACK},
    )
    fig.add_annotation(
        x=-0.12,
        y=0,
        xanchor="left",
        yanchor="top",
        yshift=-6,
        text=f"▼ {down_name}",
        showarrow=False,
        font={"color": theme.CORAL_SOFT, "size": 10, "family": theme.FONT_STACK},
    )

    layout = abyss_layout(
        height=height,
        hovermode="closest",
        xaxis={
            "tickmode": "array",
            "tickvals": list(range(n)),
            "ticktext": [str(c).replace("_", " ").title() for c in categories],
            "range": [-0.18, n - 0.82],
            "showgrid": False,
            "zeroline": False,
            "showline": False,
            "ticks": "",
            "tickfont": {"color": theme.TEXT, "size": 11},
        },
        # Ticks show the magnitude, not the plotted sign: the lower half is a
        # second positive measurement drawn downward.
        yaxis={
            "showgrid": True,
            "gridcolor": rgba(theme.OCEAN_MID, 0.09),
            "zeroline": False,
            "showline": False,
            "ticks": "",
            "tickfont": {"color": theme.TEXT_MUTED, "size": 10},
            "tickformat": value_format.lstrip(","),
        },
        legend={
            "orientation": "h",
            "y": -0.18,
            "x": 0,
            "font": {"color": theme.TEXT_MUTED, "size": 11},
        },
    )
    if title:
        layout["title"] = {"text": title}
    fig.update_layout(**layout)
    _absolute_value_ticks(fig, up_series, down_series, order, n, value_format)
    return fig


def _absolute_value_ticks(
    fig: go.Figure,
    up_series: Mapping[str, Sequence[float]],
    down_series: Mapping[str, Sequence[float]],
    order: Sequence[str],
    n: int,
    value_format: str,
) -> None:
    """
    Label the y axis with magnitudes rather than plotted sign.

    The lower half is a second positive measurement drawn downward.  Leaving
    Plotly's default ticks there would print negative numbers that exist
    nowhere in the data, so the ticks are replaced with absolute values.
    """

    def _peak(block: Mapping[str, Sequence[float]]) -> float:
        totals = [0.0] * n
        for key in order:
            values = block.get(key)
            if values is None:
                continue
            for i in range(n):
                totals[i] += float(values[i])
        return max(totals) if totals else 0.0

    span = max(_peak(up_series), _peak(down_series))
    if span <= 0:
        return
    step = span / 2.0
    ticks = [-2 * step, -step, 0.0, step, 2 * step]
    fmt_spec = value_format if value_format.startswith((",", ".")) else f".{value_format}"
    fig.update_yaxes(
        tickmode="array",
        tickvals=ticks,
        ticktext=[format(abs(t), fmt_spec.lstrip(",")) for t in ticks],
        range=[-span * 1.12, span * 1.12],
    )


__all__: tuple[str, ...] = ("mirrored_stream", "stream_chart")
