"""
src/dashboard/viz — the CoralSense data-visualisation design language.

Five families, one ocean palette
--------------------------------
=========== ======================================================= ============
Family      Use it when                                             Module
=========== ======================================================= ============
mountain    comparing scalars, or comparing distributions           mountain.py
stream      a composition that is genuinely sequential/distributional stream.py
contour     a 2-D field read as depth — bathymetric isolines        contour.py
wireframe   a 2-D matrix read as a surface — sonar mesh             wireframe.py
sonar card  the number matters more than its shape                  theme.py
=========== ======================================================= ============

Non-negotiable: the encoded value is the value
----------------------------------------------
Every builder here is a *renderer*, never an estimator.  Specifically:

* a mountain peak height is the source metric, exactly — decorative ridge
  variation is a shape term that is identically zero at the summit, and the
  profile is clamped so nothing can rise above it;
* stream band thickness is the source value, exactly — a streamgraph shifts the
  *baseline*, never a band's height, and no builder here emits a negative
  number that was not a real, signed input;
* contour and wireframe builders pass the matrix through untouched, and
  represent a missing cell as a gap rather than an interpolated guess;
* distribution mountains are histograms of the real rows, drawn with straight
  segments between bin centres — no kernel smoothing, no spline, so a spike in
  the data is a spike on screen.

Hover always exposes the exact number, and the scalar builders additionally
print it as text next to the mark, so a reader never has to trust the drawing.

These builders take plain sequences rather than DataFrames: the aggregation is
the page's job, the drawing is theirs.
"""

from __future__ import annotations

from src.dashboard.viz.common import (
    DEPTH_LAYERS,
    abyss_layout,
    histogram,
    ramp_color,
    rgba,
    ridge_profile,
    style_viz,
)
from src.dashboard.viz.contour import bathymetric_contour
from src.dashboard.viz.mountain import mountain_chart, ridgeline_chart
from src.dashboard.viz.stream import mirrored_stream, stream_chart
from src.dashboard.viz.wireframe import sonar_wireframe

__all__ = (
    "DEPTH_LAYERS",
    "abyss_layout",
    "bathymetric_contour",
    "histogram",
    "mirrored_stream",
    "mountain_chart",
    "ramp_color",
    "ridge_profile",
    "ridgeline_chart",
    "rgba",
    "sonar_wireframe",
    "stream_chart",
    "style_viz",
)
