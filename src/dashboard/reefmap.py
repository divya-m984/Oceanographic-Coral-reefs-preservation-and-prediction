"""
src/dashboard/reefmap.py — the bathymetric reef map.

Renderer
--------
Leaflet, via ``folium`` + ``streamlit-folium``.  Leaflet is used rather than a
Plotly map because the primary background is a **WMS** service, which Leaflet
supports natively (``folium.raster_layers.WmsTileLayer``) and Plotly's map
traces do not.

Layer stack (bottom → top)
--------------------------
1. **GEBCO bathymetry** — ``GEBCO_LATEST_2`` (colour-shaded for elevation) as
   the base, with ``GEBCO_LATEST`` (shaded relief) available as a toggleable
   emphasis overlay.  This is what makes ridges and trenches read: shallow
   shelves pale, abyssal plains royal blue.
2. **Graticule** — faint 10° lat/lon lines, the oceanographic-atlas gesture.
3. **Natural Earth Admin-0 land** — parchment fill so land reads as paper and
   the ocean owns the composition.  There is no street layer at any zoom.
4. **Labels** — country names (uppercase, letter-spaced, dark on land) and
   significant cities (smaller, muted).  Both declutter by zoom.
5. **CoralSense observations** — a low-opacity halo plus a small solid core
   per point, coloured by the selected semantic class.

Offline / degraded behaviour
----------------------------
The GEBCO WMS is an external service.  :func:`bathymetry_available` probes it
once per cache window with a short timeout.  When it is unreachable the map is
still built — same projection, same graticule, same land, same labels, same
observations — over a styled abyssal-gradient background, and the caller is
told.  A street map is never substituted, and no failure here can raise out of
the page.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.dashboard import theme

# ---------------------------------------------------------------------------
# External services and local data
# ---------------------------------------------------------------------------

GEBCO_WMS_URL = "https://wms.gebco.net/mapserv"
GEBCO_LAYER_ELEVATION = "GEBCO_LATEST_2"  # colour-shaded for elevation
GEBCO_LAYER_RELIEF = "GEBCO_LATEST"  # grey shaded relief, used as emphasis
GEBCO_ATTRIBUTION = "Imagery reproduced from the GEBCO Grid, GEBCO Compilation Group — gebco.net"
NATURAL_EARTH_ATTRIBUTION = (
    "Boundaries and place names: Natural Earth 1:50m cultural vectors "
    "(public domain, naturalearthdata.com)"
)

GEO_DIR = Path(__file__).resolve().parent / "geo"
COUNTRIES_PATH = GEO_DIR / "ne_admin0_indian_ocean.json"
PLACES_PATH = GEO_DIR / "ne_places_indian_ocean.json"

ATTRIBUTIONS: tuple[tuple[str, str], ...] = (
    ("GEBCO", GEBCO_ATTRIBUTION),
    ("Natural Earth", NATURAL_EARTH_ATTRIBUTION),
    ("Leaflet", "Map rendering by Leaflet (BSD-2-Clause) via folium / streamlit-folium"),
)

# Study window centre and default zoom — the four Indian reef prototype zones.
DEFAULT_CENTER: tuple[float, float] = (13.2, 79.5)
DEFAULT_ZOOM = 5

# Zoom thresholds for label decluttering.
COUNTRY_LABEL_MIN_ZOOM = 4
MINOR_COUNTRY_LABEL_MIN_ZOOM = 6
CITY_LABEL_MIN_ZOOM = 6

# Atlas palette for the vector layers drawn over the bathymetry.
LAND_FILL = "#efe6d3"  # parchment
LAND_LINE = "#8c7f66"
LABEL_COUNTRY = "#1e2b3a"  # dark navy/charcoal on land
LABEL_CITY = "#4a5a6b"
FALLBACK_BACKGROUND = (
    "radial-gradient(120% 90% at 30% -10%, #0b3550 0%, #072038 40%, #041525 72%, #020b14 100%)"
)


# ---------------------------------------------------------------------------
# Local geographic data
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_countries() -> dict[str, Any]:
    """
    Load the packaged Admin-0 boundary subset.

    Read-only.  Returns an empty FeatureCollection if the file is missing, so
    a stripped-down deployment degrades to "bathymetry without borders"
    instead of crashing.
    """
    return _read_geojson(COUNTRIES_PATH)


@st.cache_data(show_spinner=False)
def load_places() -> dict[str, Any]:
    """Load the packaged significant-places subset (capitals + large cities)."""
    return _read_geojson(PLACES_PATH)


def _read_geojson(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and data.get("type") == "FeatureCollection":
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {"type": "FeatureCollection", "features": []}


# ---------------------------------------------------------------------------
# External service probe
# ---------------------------------------------------------------------------


@st.cache_data(ttl=600, show_spinner=False)
def bathymetry_available(url: str = GEBCO_WMS_URL, timeout: float = 4.0) -> bool:
    """
    Return ``True`` when the GEBCO WMS answers a GetCapabilities request.

    Cached for ten minutes so a page rerun does not re-probe, and wrapped so no
    network condition can propagate an exception into the page.
    """
    try:
        import requests

        response = requests.get(
            url,
            params={"service": "WMS", "request": "GetCapabilities", "version": "1.3.0"},
            timeout=timeout,
        )
        return response.status_code == 200 and b"WMS_Capabilities" in response.content[:4096]
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Map construction
# ---------------------------------------------------------------------------


def _country_label_html(name: str, minor: bool) -> str:
    size = 10 if minor else 13
    weight = 600 if minor else 700
    spacing = "0.18em" if minor else "0.26em"
    opacity = 0.62 if minor else 0.82
    return (
        f'<div style="white-space:nowrap;transform:translate(-50%,-50%);'
        f"font-family:{theme.FONT_STACK};font-size:{size}px;font-weight:{weight};"
        f"letter-spacing:{spacing};text-transform:uppercase;color:{LABEL_COUNTRY};"
        f'opacity:{opacity};text-shadow:0 1px 2px rgba(255,255,255,0.55)">{name}</div>'
    )


def _city_label_html(name: str, capital: bool) -> str:
    dot = "◉" if capital else "•"
    weight = 600 if capital else 500
    return (
        f'<div style="white-space:nowrap;transform:translate(-6px,-50%);'
        f"font-family:{theme.FONT_STACK};font-size:9.5px;font-weight:{weight};"
        f'color:{LABEL_CITY};opacity:0.78;text-shadow:0 1px 2px rgba(255,255,255,0.5)">'
        f"{dot}&nbsp;{name}</div>"
    )


def _add_graticule(parent: Any, folium: Any) -> None:
    """Faint 10° lat/lon lines — the atlas grid, not a data layer."""
    style = {"color": "#7fd8e8", "weight": 0.6, "opacity": 0.22, "dashArray": "3,6"}
    for lat in range(-20, 51, 10):
        folium.PolyLine([(lat, 40), (lat, 120)], **style).add_to(parent)
    for lon in range(40, 121, 10):
        folium.PolyLine([(-20, lon), (50, lon)], **style).add_to(parent)


def _zoom_declutter_script(map_name: str, groups: Sequence[tuple[str, int]]) -> str:
    """
    Return the JS that shows/hides label layers by zoom.

    Leaflet has no declarative zoom range for a layer, so each label group is
    added or removed on ``zoomend``.  Written against the folium-generated
    variable names, and guarded so a folium version that renames them simply
    leaves every label visible instead of erroring.
    """
    entries = ", ".join(f"[{name}, {min_zoom}]" for name, min_zoom in groups)
    return f"""
        (function () {{
          try {{
            var map = {map_name};
            var groups = [{entries}];
            function apply() {{
              var z = map.getZoom();
              groups.forEach(function (entry) {{
                var layer = entry[0], minZoom = entry[1];
                if (!layer) return;
                if (z >= minZoom) {{
                  if (!map.hasLayer(layer)) map.addLayer(layer);
                }} else if (map.hasLayer(layer)) {{
                  map.removeLayer(layer);
                }}
              }});
            }}
            map.on('zoomend', apply);
            apply();
          }} catch (err) {{ /* labels stay visible; never break the map */ }}
        }})();
    """


def _attribution_script(map_name: str, credits: Sequence[str]) -> str:
    """
    Push credits into Leaflet's own attribution control.

    WMS tile layers carry their ``attr`` automatically, but a ``GeoJson``
    overlay has no attribution slot, and in the offline fallback there is no
    tile layer at all.  Registering the credits directly keeps the required
    notice on the map itself in every state, not only the happy path.
    """
    payload = json.dumps(list(credits))
    return f"""
        (function () {{
          try {{
            var map = {map_name};
            if (!map.attributionControl) return;
            {payload}.forEach(function (credit) {{
              map.attributionControl.addAttribution(credit);
            }});
          }} catch (err) {{ /* attribution is also printed beneath the map */ }}
        }})();
    """


def build_reef_map(
    df: pd.DataFrame,
    *,
    color_col: str,
    color_map: Mapping[str, str],
    center: tuple[float, float] = DEFAULT_CENTER,
    zoom: int = DEFAULT_ZOOM,
    bathymetry: bool | None = None,
    show_labels: bool = True,
    halo_radius: int = 9,
    core_radius: float = 2.4,
) -> tuple[Any, bool]:
    """
    Build the Leaflet reef map and return ``(folium_map, bathymetry_used)``.

    ``df`` is rendered verbatim — this function performs **no** filtering,
    sampling or aggregation.  Every filter (region, health class, restoration
    class, max points) is applied by the page before the frame arrives here,
    which is what keeps the map's contents identical to the filtered table
    beside it.

    Parameters
    ----------
    color_col:
        ``"reef_health"`` or ``"restoration_suitability"`` — the column whose
        class decides each marker's colour.
    color_map:
        Class → hex colour.  Unknown classes fall back to the aqua token.
    bathymetry:
        Force the WMS on/off.  ``None`` probes the service.

    Raises
    ------
    ImportError
        If folium is not installed — the caller is expected to handle this and
        show a message rather than a traceback.
    """
    import folium

    use_bathymetry = bathymetry_available() if bathymetry is None else bool(bathymetry)

    fmap = folium.Map(
        location=list(center),
        zoom_start=zoom,
        min_zoom=3,
        max_zoom=10,
        tiles=None,  # no street basemap, ever
        prefer_canvas=True,  # canvas renderer: thousands of markers stay smooth
        control_scale=True,
        zoom_control=True,
        attributionControl=True,
    )

    if use_bathymetry:
        folium.raster_layers.WmsTileLayer(
            url=GEBCO_WMS_URL,
            layers=GEBCO_LAYER_ELEVATION,
            name="GEBCO bathymetry (colour-shaded elevation)",
            fmt="image/png",
            transparent=False,
            version="1.3.0",
            attr=GEBCO_ATTRIBUTION,
            overlay=False,
            control=True,
        ).add_to(fmap)
        folium.raster_layers.WmsTileLayer(
            url=GEBCO_WMS_URL,
            layers=GEBCO_LAYER_RELIEF,
            name="Shaded relief emphasis",
            fmt="image/png",
            transparent=True,
            opacity=0.35,
            version="1.3.0",
            attr=GEBCO_ATTRIBUTION,
            overlay=True,
            control=True,
            show=False,
        ).add_to(fmap)
    else:
        # Styled abyssal fallback: the map keeps its projection, graticule,
        # coastlines, labels and data — it just loses the relief imagery.
        fmap.get_root().header.add_child(
            folium.Element(
                f"<style>.folium-map, .leaflet-container "
                f"{{ background: {FALLBACK_BACKGROUND} !important; }}</style>"
            )
        )

    _add_graticule(fmap, folium)

    # ---- Land ------------------------------------------------------------
    countries = load_countries()
    if countries["features"]:
        folium.GeoJson(
            countries,
            name="Coastlines and boundaries",
            style_function=lambda _f: {
                "fillColor": LAND_FILL,
                "color": LAND_LINE,
                "weight": 0.7,
                "fillOpacity": 0.92 if use_bathymetry else 0.86,
                "opacity": 0.75,
            },
            control=False,
            interactive=False,
        ).add_to(fmap)

    # ---- Labels ----------------------------------------------------------
    declutter: list[tuple[str, int]] = []
    if show_labels and countries["features"]:
        major = folium.FeatureGroup(name="Country names", show=True, control=False)
        minor = folium.FeatureGroup(name="Country names (minor)", show=True, control=False)
        for feature in countries["features"]:
            props = feature.get("properties", {})
            lat = props.get("label_lat")
            lon = props.get("label_lon")
            if not lat and not lon:
                continue
            is_minor = int(props.get("rank", 6)) > 3
            folium.Marker(
                location=[lat, lon],
                icon=folium.DivIcon(
                    html=_country_label_html(str(props.get("name", "")), is_minor),
                    icon_size=(0, 0),
                    icon_anchor=(0, 0),
                ),
            ).add_to(minor if is_minor else major)
        major.add_to(fmap)
        minor.add_to(fmap)
        declutter.append((major.get_name(), COUNTRY_LABEL_MIN_ZOOM))
        declutter.append((minor.get_name(), MINOR_COUNTRY_LABEL_MIN_ZOOM))

        places = load_places()
        if places["features"]:
            cities = folium.FeatureGroup(name="Major cities", show=True, control=False)
            for feature in places["features"]:
                props = feature.get("properties", {})
                lon, lat = feature["geometry"]["coordinates"][:2]
                folium.Marker(
                    location=[lat, lon],
                    icon=folium.DivIcon(
                        html=_city_label_html(
                            str(props.get("name", "")), bool(props.get("capital"))
                        ),
                        icon_size=(0, 0),
                        icon_anchor=(0, 0),
                    ),
                ).add_to(cities)
            cities.add_to(fmap)
            declutter.append((cities.get_name(), CITY_LABEL_MIN_ZOOM))

    # ---- Observations ----------------------------------------------------
    observations = folium.FeatureGroup(name="CoralSense observations", show=True, control=True)
    _add_observations(
        observations,
        folium,
        df,
        color_col=color_col,
        color_map=color_map,
        halo_radius=halo_radius,
        core_radius=core_radius,
    )
    observations.add_to(fmap)

    folium.LayerControl(collapsed=True, position="topright").add_to(fmap)

    if declutter:
        fmap.get_root().script.add_child(
            folium.Element(_zoom_declutter_script(fmap.get_name(), declutter))
        )

    credits = [NATURAL_EARTH_ATTRIBUTION]
    if not use_bathymetry:
        credits.append("Bathymetry unavailable — styled abyssal background")
    fmap.get_root().script.add_child(folium.Element(_attribution_script(fmap.get_name(), credits)))

    return fmap, use_bathymetry


def _add_observations(
    parent: Any,
    folium: Any,
    df: pd.DataFrame,
    *,
    color_col: str,
    color_map: Mapping[str, str],
    halo_radius: int,
    core_radius: float,
) -> None:
    """
    Draw every row of *df* as a single scientific marker.

    One ``folium.GeoJson`` layer over one FeatureCollection, not one Python
    object per observation.  That distinction is the whole performance story:
    at 2,000 points, per-row ``CircleMarker`` objects took ~9 s to render and
    produced a 5 MB document, because folium renders each one through its own
    Jinja template.  A single vector layer renders in milliseconds and the
    points become one compact JSON payload.

    The marker itself is a small solid core inside a wide, very transparent
    stroke.  The stroke reads as the halo: it gives the scientific-plot glow
    and lets dense clusters show as density, while the opaque core keeps an
    individual observation locatable.  Combined with the canvas renderer this
    stays smooth into the thousands without clustering or thinning — the page's
    max-points slider remains the only bound on how many are drawn.
    """
    if df is None or df.empty:
        return

    optional = ("depth_m", "water_temperature_c", "coral_cover_percentage")
    columns = ["latitude", "longitude", "region", "reef_health", "restoration_suitability"]
    columns += [c for c in optional if c in df.columns]
    columns = [c for c in columns if c in df.columns]

    features = []
    for row in df[columns].itertuples(index=False):
        record = dict(zip(columns, row, strict=True))
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "colour": color_map.get(str(record.get(color_col, "")), theme.AQUA),
                    "Region": str(record.get("region", "—")),
                    "Reef health": _pretty(record.get("reef_health")),
                    "Restoration": _pretty(record.get("restoration_suitability")),
                    "Depth": _number(record.get("depth_m"), " m"),
                    "Temperature": _number(record.get("water_temperature_c"), " °C"),
                    "Coral cover": _number(record.get("coral_cover_percentage"), " %"),
                },
                "geometry": {
                    "type": "Point",
                    # 4 dp ≈ 11 m: finer than the synthetic scatter and it keeps
                    # the embedded payload small.
                    "coordinates": [
                        round(float(record["longitude"]), 4),
                        round(float(record["latitude"]), 4),
                    ],
                },
            }
        )

    folium.GeoJson(
        {"type": "FeatureCollection", "features": features},
        name="CoralSense observations",
        marker=folium.CircleMarker(),
        style_function=lambda feature: {
            "radius": core_radius,
            "fillColor": feature["properties"]["colour"],
            "fillOpacity": 0.95,
            "color": feature["properties"]["colour"],
            "weight": halo_radius,  # the halo: a wide, faint stroke
            "opacity": 0.16,
        },
        highlight_function=lambda _feature: {"weight": halo_radius + 4, "opacity": 0.34},
        tooltip=folium.GeoJsonTooltip(
            fields=["Region", "Reef health", "Restoration", "Depth", "Temperature", "Coral cover"],
            sticky=True,
            style=(
                f"font-family:{theme.FONT_STACK};font-size:11px;"
                f"background:{theme.TRENCH};color:{theme.TEXT};"
                f"border:1px solid {theme.OCEAN_DEEP};border-radius:8px;"
                "padding:6px 8px;"
            ),
        ),
    ).add_to(parent)


def _pretty(value: Any) -> str:
    """Render a class label for display, or an em dash when absent."""
    if value is None:
        return "—"
    return str(value).replace("_", " ").title()


def _number(value: Any, suffix: str = "") -> str:
    """Render a sensor reading to one decimal, or an em dash when unusable."""
    try:
        return f"{float(value):.1f}{suffix}"
    except (TypeError, ValueError):
        return "—"


__all__: tuple[str, ...] = (
    "ATTRIBUTIONS",
    "CITY_LABEL_MIN_ZOOM",
    "COUNTRIES_PATH",
    "COUNTRY_LABEL_MIN_ZOOM",
    "DEFAULT_CENTER",
    "DEFAULT_ZOOM",
    "GEBCO_ATTRIBUTION",
    "GEBCO_LAYER_ELEVATION",
    "GEBCO_LAYER_RELIEF",
    "GEBCO_WMS_URL",
    "NATURAL_EARTH_ATTRIBUTION",
    "PLACES_PATH",
    "bathymetry_available",
    "build_reef_map",
    "load_countries",
    "load_places",
)
