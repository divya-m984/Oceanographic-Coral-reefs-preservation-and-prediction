# Packaged geographic data — sources and licences

These two files are the **only** geographic data shipped inside the dashboard
image. Everything else the Reef Map draws is either CoralSense's own synthetic
observations or the live GEBCO service.

## `ne_admin0_indian_ocean.json`

| | |
|---|---|
| Source | Natural Earth, `ne_50m_admin_0_countries` (1:50m cultural vectors) |
| Upstream | <https://www.naturalearthdata.com/> · <https://github.com/nvkelso/natural-earth-vector> |
| Licence | **Public domain.** "All versions of Natural Earth raster + vector map data found on this website are in the public domain." |
| Contents | 41 Admin-0 features intersecting the Indian-Ocean study window (48°–112° E, 18° S – 44° N) |
| Used for | Coastlines, national boundaries, parchment land fill, and the uppercase country labels |

## `ne_places_indian_ocean.json`

| | |
|---|---|
| Source | Natural Earth, `ne_50m_populated_places_simple` |
| Upstream | as above |
| Licence | **Public domain** |
| Contents | 62 significant places in the same window — national capitals plus metropolitan areas of 2 million or more |
| Used for | The smaller, muted city labels |

## Derivation

Both files are lossy subsets produced once, offline, from the upstream GeoJSON:

1. features outside the study window are dropped, as are individual polygons of
   a MultiPolygon that fall entirely outside it;
2. rings are simplified with Douglas–Peucker at a 0.022° tolerance;
3. coordinates are rounded to three decimal places (~110 m at the equator);
4. properties are reduced to what the map actually reads — `name`, `iso`,
   `label_lon`, `label_lat`, `rank` for countries; `name`, `capital`, `rank` for
   places.

The result is roughly 180 KB rather than the ~3.9 MB of the two upstream files,
which is what "package only the minimum required data into the dashboard image"
means here. Geometry is *simplified*, never invented: no coastline is added,
moved to a different country, or smoothed into a shape it does not have at
1:50m — it is the same data at lower vertex density.

## Live service (not packaged)

| | |
|---|---|
| Service | GEBCO Web Map Service — `https://wms.gebco.net/mapserv` |
| Layers | `GEBCO_LATEST_2` (colour-shaded for elevation, base) and `GEBCO_LATEST` (shaded relief, optional overlay) |
| Required notice | "Imagery reproduced from the GEBCO Grid, GEBCO Compilation Group" |
| Terms | <https://www.gebco.net/data_and_products/gebco_web_services/web_map_service/> |

The notice is registered with Leaflet's attribution control by
`src/dashboard/reefmap.py` and printed again beneath the map by the page. When
the service is unreachable the map keeps its projection, graticule, coastlines,
labels and observations over a styled abyssal background — a street basemap is
never substituted.

## Renderer

Leaflet (BSD-2-Clause), reached through `folium` (MIT) and `streamlit-folium`
(MIT). Both are pure-Python wrappers that bundle their own JS/CSS; the
dashboard fetches no map library from a CDN.
