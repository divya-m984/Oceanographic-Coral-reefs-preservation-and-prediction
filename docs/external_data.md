# External (Real) Data Layer

**Status:** first real external dataset acquired — GEBCO_2026 bathymetry only.
**Created:** 2026-08-19

This document describes the real-data layer. It is deliberately separate from
the synthetic prototype pipeline described in
[`data_dictionary.md`](data_dictionary.md) and audited in
[`audits/dataset_scientific_audit_2026-08-19.md`](audits/dataset_scientific_audit_2026-08-19.md).

---

## 1. Why this layer is separate

The 2026-08-19 dataset audit established that the synthetic 15,000-row
benchmark suffers from **label-construction leakage / circular supervision**,
and that its labels are algorithmically generated from the same variables fed to
the models. The remedy is not to patch the synthetic generator; it is to build
a genuinely observed track alongside it.

Four rules govern this layer, all inherited from the audit:

1. **The synthetic dataset stays frozen.** `data/raw/observations.csv`
   (SHA-256 `a03cb3e9…1ad8f458`) is unchanged by anything here.
2. **Synthetic and real rows are never silently combined.** There is no code
   path in `src/external/` that reads or writes `data/raw/`.
3. **Real environmental or physical measurements are not reef-condition labels.**
   Thresholding bathymetry, temperature or any other covariate into a reef-health
   class would rebuild the audit's CRITICAL finding using real numbers — which is
   worse, because it looks credible.
4. **Every external observation carries provenance**, including a licence
   verification state, before it may be used.

Rule 3 is enforced in code: `src/external/provenance.py` rejects any product
whose record claims to provide `reef_health` or `restoration_suitability`.

---

## 2. Layout

```
data/external/
    metadata/                       tracked in Git
        gebco_2026.manifest.json    product record + one record per subset
    raw/                            git-ignored (see .gitignore)
        gebco_2026/
            gebco_2026_lakshadweep.nc
            gebco_2026_gulf_of_mannar.nc
            gebco_2026_gulf_of_kutch.nc
            gebco_2026_andaman_nicobar.nc
```

`data/external/raw/` is git-ignored so this layer can later be placed under DVC
without rewriting history. **That is a storage decision, not a licensing one** —
per-product redistribution terms live in each manifest's
`redistribution_allowed` field. GEBCO is public domain and *may* be
redistributed; it is simply not stored in Git.

Code lives in `src/external/` (provenance schema and licence gate) and
`scripts/fetch_gebco_2026.py` (acquisition + validation). Nothing in
`src/data/`, `src/features/`, `src/models/`, `src/api/` or `src/dashboard/` was
modified.

---

## 3. The licence gate

Before an external product may be used inside this repository, its record must
satisfy `validate_source()`:

| Requirement | Rule |
|---|---|
| `licence_verified` | Must be **`True`**. Defaults to `False` — a product is untrusted until someone has read the publisher's own terms. |
| `licence_verified_via` | Must record **where** the terms were read. |
| `redistribution_allowed` | Must be explicitly **`True` or `False`**, never left unstated. |
| `is_synthetic` | Must be `False`. This layer is for real data only. |
| `provides_variables` | Must not claim `reef_health` or `restoration_suitability`. |

`redistribution_allowed = False` does **not** block use. It blocks
*publication*: the metadata may describe the source, but the raw files must not
be committed or redistributed. `assert_publishable()` expresses that check for
callers about to export or commit.

---

## 4. GEBCO_2026 — scientific role

**Product:** The GEBCO_2026 Grid — a continuous terrain model for oceans and land at 15 arc-second intervals
**Provider:** GEBCO / Nippon Foundation–GEBCO Seabed 2030; held at BODC
**DOI:** `10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa`
**Licence:** **Public domain** — verified 2026-08-19 from the official product page *and* from the `license` global attribute inside the delivered NetCDF itself
**Resolution:** 15 arc-seconds (~0.00416667°, ~450 m at the equator)
**CRS:** EPSG:4326 (WGS 84); vertical EPSG:5831, positive up
**Type:** Compiled — fuses measured multibeam/singlebeam soundings with satellite-derived predicted bathymetry (SRTM15+ v2.8 base) and land topography

### What GEBCO_2026 CAN provide

- **Real global bathymetric and terrain context** for the four reef regions
- **Support for a derived `depth_m`**, where `depth_m = -elevation` and the
  derivation applies **only to marine cells** (`elevation < 0`)
- Potentially, **coarse terrain metrics** — slope, regional ruggedness,
  bathymetric position index — subject to the resolution caveat below

### What GEBCO_2026 CANNOT provide

It cannot supply any of the following, and no processing of it may be presented
as doing so:

- `coral_cover_percentage`
- `bleaching_percentage`
- `disease_percentage`
- `reef_health`
- `restoration_suitability`
- `sonar_backscatter`
- `acoustic_complexity_index`

It is **not** a substitute for multibeam or sidescan survey data. GEBCO is a
compiled global product; a large fraction of any given area is interpolated or
satellite-predicted rather than directly sounded. Acoustic seabed
characterisation requires actual survey data with its own calibration path.

### Resolution caveat — `rugosity_index`

**GEBCO's ~15 arc-second (~450 m) grid is far too coarse to reproduce the
project's synthetic fine-scale `rugosity_index` directly.** The synthetic
variable was defined at colony/reef-structure scale (values 1.0–4.8, described
as flat sand to complex three-dimensional reef); GEBCO cells are hundreds of
metres across, two to three orders of magnitude larger.

If a terrain-roughness feature is derived from GEBCO in future, it **must carry
a new, resolution-aware definition** — a different variable name, an explicit
statement of the scale at which it was computed, and no implied equivalence to
the synthetic `rugosity_index`. Reusing the old name would silently misrepresent
what was measured.

### Scientific limitations — what `depth_m` actually is

GEBCO_2026 is a **broad-scale global terrain compilation**, not survey
bathymetry. Four limitations, in GEBCO's own words:

1. **Grid spacing is not survey resolution.** 15 arc-seconds is ~450 m at the
   equator — hundreds of metres of horizontal spacing. GEBCO warns that *"the
   resolution of The GEBCO Grid may be significantly different to that of the
   resolution of the underlying measured data."* The grid is interpolated
   between measurements; cell spacing says nothing about how densely the seabed
   was actually sounded.
2. **Source data are heterogeneous.** The grid *"is based on bathymetric data
   from many different sources of varying quality and coverage"*, generated by
   *"the assimilation of heterogeneous data types."* Quality varies from
   cell to cell, and the TID grid (below) is how that variation is inspected.
3. **Shallow water is the weakest case.** GEBCO notes it is still *"working to
   understand how best to fully assimilate"* shallow-water sources. This is
   exactly where reefs are, so it is exactly where this product is least
   reliable for our purposes.
4. **Vertical datum is not guaranteed to be mean sea level.** GEBCO generally
   assumes MSL, but *"in some shallow water areas, the grids include data from
   sources having a vertical datum other than mean sea level."*

**Therefore `depth_m = -elevation` is a derived, coarse, bathymetric-context
variable for marine cells — and nothing more.** It must never be described as:

- calibrated sonar depth
- fine-scale reef depth
- navigation-quality depth
- ground-truthed local bathymetry

### Bathymetric convention

Elevation is stored as **metres, positive up**: negative values are below mean
sea level (subject to limitation 4 above). The source files are never
transformed. `depth_m = -elevation` is a **derived** field computed in memory
for analysis only, valid for marine cells.

### Publisher disclaimer

GEBCO states the grid **must not be used for navigation or any purpose relating
to safety at sea**.

### Future work — the Type Identifier (TID) grid

**Not acquired in this task. Not a current feature.**

GEBCO publishes a companion **Type Identifier (TID) grid** recording the type of
source information behind each cell, so users can *"assess the 'quality' of the
grid in a particular area, i.e. if it is based on multibeam data, singlebeam
data or on interpolation, etc."* Its classes include:

| Group | Codes | Examples |
|---|---|---|
| Direct measurement | 10–17 | singlebeam (10), **multibeam (11)**, seismic (12), isolated sounding (13), ENC sounding (14), lidar (15), optical (16), combination (17) |
| Indirect measurement | 40–46 | satellite-gravity prediction (40), **computer-algorithm interpolation (41)**, digitised chart contours (42), ENC contours (43) |
| Unknown | 70–72 | pre-generated grid (70), unknown source (71), steering points (72) |
| Land | 0 | — |

**Why this matters: not every GEBCO cell represents a direct depth
measurement.** A cell carrying TID 41 is an algorithmic interpolation; a cell
carrying TID 11 is a real multibeam sounding. Treating the two as equally
trustworthy would overstate what is known about a site.

The TID grid should eventually be acquired for the same four windows and joined
cell-for-cell, so that any GEBCO-derived value can carry a source-quality flag
(and so that cells resting on interpolation can be down-weighted or excluded).
This is deliberately deferred — it is a second acquisition with its own
provenance record.

---

## 5. Acquisition windows

Four regional subsets were acquired. These are **acquisition windows, not
observation-sampling boxes** — GEBCO is a continuous raster, so a rectangle is
the natural request shape. No observation coordinate may ever be drawn uniformly
from one of these; that is precisely the defect the audit found in the synthetic
dataset (§5, coordinates uniform in rectangles over open ocean and land).

| Region | bbox `[lat_min, lat_max, lon_min, lon_max]` | Why |
|---|---|---|
| **Lakshadweep** | `[8.0, 12.6, 71.5, 74.2]` | Includes **Minicoy** (8°15′–8°20′N), which the old synthetic box `[10.0, 12.5, …]` excluded entirely; covers Cherbaniani Reef (12°18′N, 71°53′E) |
| **Gulf of Mannar** | `[8.5, 9.5, 78.0, 79.6]` | The 21-island chain 1–10 km offshore over 160 km, Tuticorin→Dhanushkodi. Held at 9.5°N to exclude Palk Bay (a distinct system) |
| **Gulf of Kutch** | `[22.0, 23.0, 68.9, 70.6]` | Marine National Park (22.467°N, 69.617°E), 42 islands along the Jamnagar coast. Old box reached 24.5°N — ~170 km inland |
| **Andaman & Nicobar** | `[6.5, 14.0, 92.0, 94.3]` | Landfall Island (13°39′N) to Indira Point (6°45′10″N, 93°49′36″E); east bound widened to 94.3 so Indira Point is not clipped |

**These windows must be refined against real reef masks and survey site
locations before any modelling use.** The bathymetric statistics below show why:
in Lakshadweep, 98.8 % of marine cells are deeper than 100 m — the reef occupies
a tiny fraction of the rectangle.

---

## 6. Bathymetric context (not reef labels)

Percentages are of **marine cells only** (`elevation < 0`).

| Region | % marine | 0–10 m | 10–30 m | 30–100 m | >100 m |
|---|---|---|---|---|---|
| Lakshadweep | 99.87 | 0.47 | 0.54 | 0.24 | **98.76** |
| Gulf of Mannar | 71.18 | 23.25 | 27.70 | 8.15 | 40.90 |
| Gulf of Kutch | 37.26 | 37.48 | 37.68 | 24.84 | **0.00** |
| Andaman & Nicobar | 96.58 | 1.42 | 3.69 | 7.78 | 87.11 |

> **Shallow water is not a reef.** These bands are bathymetric context only.
> Depth is one of many controls on reef presence; deriving reef extent from
> depth alone would be a fabricated label.

Observations on plausibility:

- **Lakshadweep** — atolls are small caps on the Chagos–Laccadive Ridge
  surrounded by deep Arabian Sea (median elevation −2000 m). The near-total
  dominance of >100 m water is correct and is the clearest argument for
  reef-mask-based extraction.
- **Gulf of Mannar** — 51 % of marine cells shallower than 30 m, consistent with
  a shallow gulf and island chain. The 40.9 % >100 m comes from the deeper
  south-eastern part of the window.
- **Gulf of Kutch** — **no marine cell deeper than 90 m**, consistent with a
  shallow macrotidal gulf. Land contamination is high (62.7 %) but expected and
  necessary: the reefs fringe the coast, so the coast must be inside the window.
- **Andaman & Nicobar** — deep Andaman Sea with a narrow island ridge; 87 %
  >100 m. Land maxima (679 m) are consistent with the Andaman hills at 450 m
  grid resolution.

---

## 6a. What the SHA-256 values mean

Each subset record carries a `sha256`. Its scope is narrow and worth stating
plainly:

| | |
|---|---|
| **It is** | the hash of the file *this project retrieved*, computed locally at acquisition time |
| **It is not** | a GEBCO-published canonical checksum. GEBCO does not distribute per-subset digests for server-generated extracts, and these values must never be presented as though it does |
| **It guarantees** | the local file has not been altered since acquisition — the integrity check in `tests/test_external_provenance.py` |
| **It does not guarantee** | that re-requesting the same window reproduces identical bytes |

That last point matters for reproducibility claims. A server-side subsetting
service may legitimately return a byte-different NetCDF serialisation — a
different library version, chunking, attribute ordering, or embedded generation
timestamp — while encoding **identical grid values**. So a hash mismatch after a
re-fetch means *"these are different bytes"*, not necessarily *"the data
changed"*.

The current implementation does **not** assume a re-download reproduces the
hash. `fetch_gebco_2026.py` recomputes and rewrites hashes on acquisition;
`--validate-only` re-verifies the files already on disk. Nothing asserts
byte-equality against a remote re-request.

**Scientific identity is pinned by `version = GEBCO_2026` and
`doi = 10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa`, not by these file hashes.**
Establishing value-level equivalence between two extracts would require
array-level comparison; that is deliberately out of scope here.

---

## 7. Reproducing the acquisition

```bash
python scripts/fetch_gebco_2026.py --dry-run        # print the plan, fetch nothing
python scripts/fetch_gebco_2026.py                  # fetch, validate, write manifest
python scripts/fetch_gebco_2026.py --validate-only  # re-validate local files
```

Data is requested from the **THREDDS NetCDF Subset Service** hosted by CEDA on
behalf of BODC/GEBCO, which returns NetCDF-3 classic — readable with
`scipy.io.netcdf_file`, so no new runtime dependency was introduced. The global
~7 GB file is never downloaded; the four subsets total ~3.8 MB.

---

## 8. What has NOT been done

This task acquired and documented GEBCO in isolation. Specifically **not** done:

- No change to `generate_data.py`, `preprocess.py`, `build_features.py`,
  `get_feature_columns()`, or the Pandera schema
- No GEBCO rows appended to `observations.csv`
- No join between GEBCO and any other table
- No labels created of any kind
- No model trained, registered, promoted, or evaluated
- No DVC stage added; the DAG is unchanged
- No MLflow interaction
- No API or dashboard change

---

## 9. Other sources — status

From the acquisition plan, with one **correction**.

| Source | Status |
|---|---|
| **GEBCO_2026** | **ACQUIRED** — public domain, verified |
| **Allen Coral Atlas** | **LICENCE REQUIRES VERIFICATION** — see below |
| NOAA Coral Reef Watch | Not acquired — public domain, verified, next candidate |
| RECIFS | Not acquired — public domain per publisher statement |
| HICORDIS | Not acquired — CC BY 4.0 per article |
| GCRMN | Not acquired — data under Data Sharing Agreements, request required |
| CoralNet / ReefNet | Not acquired — per-source licences; ReefNet is CC BY-NC-SA 4.0 |
| Coral Restoration Database | Not acquired — LICENCE REQUIRES VERIFICATION |
| IUCN Red List | **Excluded permanently** — redistribution prohibited; zero geographic overlap |

### Correction — Allen Coral Atlas

**The acquisition plan of 2026-08-19 stated that Allen Coral Atlas habitat maps
are CC BY 4.0 and promoted them to Tier 1. That claim is withdrawn.**

It was based on a summary of the Atlas FAQ page, not on a licence document
accompanying an actual habitat-map download. The current official Terms of Use
appear more restrictive than that summary implied, and the Atlas serves several
products under different terms — the satellite imagery mosaic is separately
CC BY-NC-SA 4.0 (© Planet Labs), which is easy to conflate with the map layers.

**Current status: `LICENCE REQUIRES VERIFICATION`.**

Before Allen Coral Atlas data may be acquired:

1. The official site Terms of Use and the licence text shipped **inside an
   actual habitat-map download package** must be read and reconciled.
2. The licence applying specifically to the **downloaded habitat-map subset**
   must be established — not the licence of the website, the imagery mosaic, or
   the Zenodo record.
3. `licence_verified` may be set to `True` only if a product-specific licence
   accompanying the real download explicitly establishes it.

Until then it must not be described as CC BY 4.0, must not be downloaded, and
must not be given an ingestion path. This matters beyond bookkeeping: the reef
mask needed to refine the acquisition windows in §5 was expected to come from
Allen, so that refinement is now blocked pending licence resolution. UNEP-WCMC
Global Distribution of Coral Reefs v4.1 is the alternative, and its
"UNEP-WCMC General Data License" also requires verification.
