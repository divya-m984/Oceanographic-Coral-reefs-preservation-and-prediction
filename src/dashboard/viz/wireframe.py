"""
src/dashboard/viz/wireframe.py — sonar wireframe surfaces.

A dark, almost-black stage with a low-opacity surface and thin pale-cyan mesh
lines running along every row and column — the look of a multibeam sonar
return.  Plotly cannot do bloom, so the glow is faked honestly: the mesh is
drawn twice, once wide and very transparent and once thin and bright, which
reads as a halo without any postprocessing pass.

When to use it
--------------
Only where a real matrix or surface relationship exists — an algorithm ×
metric table, a binned environmental surface, a confusion landscape.  It is
not a decoration for a list of categories, and this module will not invent a
third dimension: it renders exactly the matrix it is handed, and leaves a
``None`` cell as a hole in the mesh rather than filling it in.
"""

from __future__ import annotations

from collections.abc import Sequence

import plotly.graph_objects as go

from src.dashboard import theme
from src.dashboard.viz.common import abyss_layout, rgba

SURFACE_SCALE: list[list[object]] = [
    [0.00, theme.TRENCH],
    [0.30, theme.OCEAN_DEEP],
    [0.60, theme.OCEAN_MID],
    [0.85, theme.OCEAN_BRIGHT],
    [1.00, theme.OCEAN_GLOW],
]


def sonar_wireframe(
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    z: Sequence[Sequence[float | None]],
    *,
    value_name: str = "Value",
    value_format: str = ".4f",
    x_title: str = "",
    y_title: str = "",
    z_title: str = "",
    surface_opacity: float = 0.22,
    title: str | None = None,
    height: int = 520,
    camera: tuple[float, float, float] = (1.55, -1.45, 0.92),
) -> go.Figure:
    """
    Build a sonar-style wireframe surface over a 2-D matrix.

    Parameters
    ----------
    x_labels, y_labels:
        Categorical tick labels.  The surface itself is placed on integer
        indices (a 3-D Plotly surface needs numeric axes); the labels are
        restored as explicit ticks, so nothing about the data is invented.
    z:
        ``z[j][i]`` is the value at ``(x_labels[i], y_labels[j])``.  ``None``
        leaves a hole.

    Raises
    ------
    ValueError
        If ``z`` does not match the label extents.
    """
    if len(z) != len(y_labels):
        raise ValueError(f"z has {len(z)} rows, expected {len(y_labels)}")
    for row in z:
        if len(row) != len(x_labels):
            raise ValueError(f"z row has {len(row)} columns, expected {len(x_labels)}")

    xs = list(range(len(x_labels)))
    ys = list(range(len(y_labels)))
    matrix = [list(row) for row in z]

    fig = go.Figure()

    # Low-opacity surface: the field, not the focus.
    fig.add_trace(
        go.Surface(
            x=xs,
            y=ys,
            z=matrix,
            colorscale=SURFACE_SCALE,
            opacity=max(0.0, min(1.0, surface_opacity)),
            showscale=False,
            hoverinfo="skip",
            contours={
                "z": {
                    "show": True,
                    "usecolormap": False,
                    "color": rgba(theme.OCEAN_GLOW, 0.35),
                    "width": 1,
                    "project": {"z": False},
                }
            },
        )
    )

    # Mesh lines along rows and columns. Drawn twice — wide+faint, then
    # thin+bright — so the mesh glows without a postprocessing pass.
    def _add_line(px: list[float], py: list[float], pz: list[float | None]) -> None:
        for width, alpha in ((5.0, 0.07), (1.4, 0.85)):
            fig.add_trace(
                go.Scatter3d(
                    x=px,
                    y=py,
                    z=pz,
                    mode="lines",
                    line={"width": width, "color": rgba(theme.SONAR_MESH, alpha)},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    for j, row in enumerate(matrix):
        _add_line(list(map(float, xs)), [float(j)] * len(xs), list(row))
    for i in range(len(x_labels)):
        column = [matrix[j][i] for j in range(len(y_labels))]
        _add_line([float(i)] * len(ys), list(map(float, ys)), column)

    # Node markers carry the exact numbers.
    node_x: list[float] = []
    node_y: list[float] = []
    node_z: list[float] = []
    node_text: list[list[str]] = []
    for j, row in enumerate(matrix):
        for i, value in enumerate(row):
            if value is None:
                continue
            node_x.append(float(i))
            node_y.append(float(j))
            node_z.append(float(value))
            node_text.append([str(x_labels[i]), str(y_labels[j])])

    fig.add_trace(
        go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode="markers",
            marker={
                "size": 3.4,
                "color": theme.FOAM,
                "opacity": 0.9,
                "line": {"width": 0},
            },
            customdata=node_text,
            hovertemplate=(
                "<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
                f"{value_name}: <b>%{{z:{value_format}}}</b><extra></extra>"
            ),
            showlegend=False,
        )
    )

    axis_common = {
        "showbackground": True,
        "backgroundcolor": rgba(theme.VOID, 0.86),
        "gridcolor": rgba(theme.OCEAN_MID, 0.20),
        "zerolinecolor": rgba(theme.OCEAN_MID, 0.28),
        "showspikes": False,
        "tickfont": {"color": theme.TEXT_MUTED, "size": 10},
    }

    layout = abyss_layout(
        height=height,
        margin={"l": 4, "r": 4, "t": 56 if title else 12, "b": 4},
        scene={
            "xaxis": {
                **axis_common,
                "title": {"text": x_title, "font": {"color": theme.TEXT_DIM, "size": 11}},
                "tickmode": "array",
                "tickvals": xs,
                "ticktext": [str(v) for v in x_labels],
            },
            "yaxis": {
                **axis_common,
                "title": {"text": y_title, "font": {"color": theme.TEXT_DIM, "size": 11}},
                "tickmode": "array",
                "tickvals": ys,
                "ticktext": [str(v) for v in y_labels],
            },
            "zaxis": {
                **axis_common,
                "title": {
                    "text": z_title or value_name,
                    "font": {"color": theme.TEXT_DIM, "size": 11},
                },
            },
            "camera": {"eye": {"x": camera[0], "y": camera[1], "z": camera[2]}},
            "aspectratio": {"x": 1.35, "y": 1.0, "z": 0.62},
        },
    )
    if title:
        layout["title"] = {"text": title}
    fig.update_layout(**layout)
    return fig


__all__: tuple[str, ...] = ("SURFACE_SCALE", "sonar_wireframe")
