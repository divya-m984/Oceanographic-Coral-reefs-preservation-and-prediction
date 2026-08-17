"""
src/dashboard/viz/contour.py — bathymetric contour charts.

Replaces the spreadsheet heatmap with a depth chart: narrow cyan/teal isolines
over a deep-navy field, faint fill, labelled contours and exact hover.

Honesty rules enforced here
---------------------------
* the matrix is passed to Plotly verbatim — no resampling, no interpolation of
  the *values* (Plotly interpolates between grid nodes to place a line, which
  is what a contour plot is, but every node keeps its own number);
* a missing cell stays missing.  ``None`` is passed straight through and
  Plotly leaves a gap, because a blank cell and a zero are different claims;
* the hover always shows the cell's own value, and the optional ``counts``
  matrix lets a caller show how many rows a binned cell was computed from, so
  a thin cell cannot masquerade as a well-supported one.
"""

from __future__ import annotations

from collections.abc import Sequence

import plotly.graph_objects as go

from src.dashboard import theme
from src.dashboard.viz.common import abyss_layout, rgba

# Deep navy → teal → aqua → pale cyan, as a Plotly colourscale.
CONTOUR_SCALE: list[list[object]] = [
    [0.00, theme.TRENCH],
    [0.22, theme.OCEAN_DEEP],
    [0.48, theme.OCEAN_MID],
    [0.72, theme.OCEAN_BRIGHT],
    [0.90, theme.OCEAN_GLOW],
    [1.00, theme.FOAM],
]


def bathymetric_contour(
    x: Sequence[float | str],
    y: Sequence[float | str],
    z: Sequence[Sequence[float | None]],
    *,
    x_title: str = "",
    y_title: str = "",
    value_name: str = "Value",
    value_format: str = ".2f",
    counts: Sequence[Sequence[int]] | None = None,
    contours: int = 14,
    show_labels: bool = True,
    title: str | None = None,
    height: int = 460,
    colorbar_title: str | None = None,
) -> go.Figure:
    """
    Build a bathymetric contour chart from a 2-D field.

    Parameters
    ----------
    x, y:
        Grid coordinates.  ``z[j][i]`` is the value at ``(x[i], y[j])``.
    z:
        The field.  ``None`` marks an unmeasured cell and is rendered as a gap.
    counts:
        Optional per-cell sample counts, surfaced in the hover.  Use it
        whenever ``z`` is a binned aggregate.
    contours:
        Approximate number of isolines.  Narrow spacing is the point of this
        style, so the default is deliberately high.

    Raises
    ------
    ValueError
        If ``z`` (or ``counts``) does not match the ``x``/``y`` extents.
    """
    if len(z) != len(y):
        raise ValueError(f"z has {len(z)} rows, expected {len(y)}")
    for row in z:
        if len(row) != len(x):
            raise ValueError(f"z row has {len(row)} columns, expected {len(x)}")
    if counts is not None:
        if len(counts) != len(y) or any(len(r) != len(x) for r in counts):
            raise ValueError("counts must have the same shape as z")

    finite = [v for row in z for v in row if v is not None]
    zmin = min(finite) if finite else 0.0
    zmax = max(finite) if finite else 1.0
    if zmax <= zmin:
        zmax = zmin + 1.0

    hover = (
        f"{x_title or 'x'}: %{{x}}<br>"
        f"{y_title or 'y'}: %{{y}}<br>"
        f"{value_name}: <b>%{{z:{value_format}}}</b>"
    )
    customdata = None
    if counts is not None:
        customdata = [[[c] for c in row] for row in counts]
        hover += "<br>observations: %{customdata[0]:,}"
    hover += "<extra></extra>"

    fig = go.Figure(
        go.Contour(
            x=list(x),
            y=list(y),
            z=[list(row) for row in z],
            customdata=customdata,
            colorscale=CONTOUR_SCALE,
            zmin=zmin,
            zmax=zmax,
            # Narrow spacing: sonar depth lines, not a spreadsheet heatmap.
            ncontours=max(4, int(contours)),
            contours={
                "coloring": "heatmap",
                "showlines": True,
                "showlabels": bool(show_labels),
                "labelfont": {"size": 9, "color": theme.FOAM, "family": theme.MONO_STACK},
            },
            line={"width": 1, "color": rgba(theme.OCEAN_GLOW, 0.55), "smoothing": 0.9},
            opacity=0.88,
            connectgaps=False,
            hovertemplate=hover,
            colorbar={
                "title": {
                    "text": colorbar_title or value_name,
                    "font": {"color": theme.TEXT_MUTED, "size": 10},
                },
                "tickfont": {"color": theme.TEXT_MUTED, "size": 10},
                "outlinewidth": 0,
                "thickness": 12,
                "len": 0.82,
            },
        )
    )

    layout = abyss_layout(
        height=height,
        margin={"l": 66, "r": 26, "t": 62 if title else 26, "b": 54},
        xaxis={
            "title": {"text": x_title, "font": {"color": theme.TEXT_DIM, "size": 11}},
            "showgrid": False,
            "zeroline": False,
            "showline": False,
            "ticks": "outside",
            "tickcolor": rgba(theme.OCEAN_MID, 0.4),
            "tickfont": {"color": theme.TEXT_MUTED, "size": 10},
        },
        yaxis={
            "title": {"text": y_title, "font": {"color": theme.TEXT_DIM, "size": 11}},
            "showgrid": False,
            "zeroline": False,
            "showline": False,
            "ticks": "outside",
            "tickcolor": rgba(theme.OCEAN_MID, 0.4),
            "tickfont": {"color": theme.TEXT_MUTED, "size": 10},
        },
        plot_bgcolor=theme.TRENCH,
    )
    if title:
        layout["title"] = {"text": title}
    fig.update_layout(**layout)
    return fig


__all__: tuple[str, ...] = ("CONTOUR_SCALE", "bathymetric_contour")
