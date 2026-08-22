# External (Real) Data Layer

**Status:** two real external datasets acquired — GEBCO_2026 bathymetry and NOAA
Coral Reef Watch 5 km v3.1 thermal products.
**Created:** 2026-08-19 · **Updated:** 2026-08-20

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

Rule 3 is enforced in code. `src/external/provenance.py` rejects any product
whose record claims to provide `reef_health` or `restoration_suitability`, and —
since NOAA CRW made the temptation concrete — also `coral_cover_percentage`,
`bleaching_percentage` and `disease_percentage`. Matching is by substring, so
`"bleaching_percentage (derived from DHW)"` is caught too.

---

## 2. Layout

```
data/external/
    metadata/                                tracked in Git
        gebco_2026.manifest.json             product record + one record per subset
        noaa_crw_5km_v3_1.manifest.json      product record + 16 subset records
    raw/                                     git-ignored (see .gitignore)
        gebco_2026/
            gebco_2026_<region>.nc                       4 files
        noaa_crw_5km_v3_1/
            noaa_crw_5km_v3_1_<product>_<region>.nc      16 files
```

One file per (product, region) pair for CRW: four thermal variables across the
same four regions. The split is per-variable because that is how the source
server publishes them — each ERDDAP dataset carries exactly one gridded
variable — so a file is never an editorial recombination of the source.

`data/external/raw/` is git-ignored so this layer can later be placed under DVC
without rewriting history. **That is a storage decision, not a licensing one** —
per-product redistribution terms live in each manifest's
`redistribution_allowed` field. GEBCO is public domain and *may* be
redistributed; it is simply not stored in Git.

Code lives in `src/external/` (provenance schema and licence gate),
`scripts/fetch_gebco_2026.py` and `scripts/fetch_noaa_crw.py` (acquisition +
validation). Nothing in `src/data/`, `src/features/`, `src/models/`, `src/api/`
or `src/dashboard/` was modified.

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
| `provides_variables` | Must not claim `reef_health`, `restoration_suitability`, `coral_cover_percentage`, `bleaching_percentage` or `disease_percentage` — by substring, so a hedged claim is caught too. |

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

## 5. Acquisition windows — shared by both products

Four regional windows. These are **acquisition windows, not
observation-sampling boxes** — both products are continuous rasters, so a
rectangle is the natural request shape. No observation coordinate may ever be
drawn uniformly from one of these; that is precisely the defect the audit found
in the synthetic dataset (§5, coordinates uniform in rectangles over open ocean
and land).

**GEBCO_2026 and NOAA CRW use the identical four windows**, so the two products
describe the same extents and stay comparable.
`tests/test_external_noaa_crw.py` asserts that equality against the GEBCO
manifest, so the two definitions cannot drift apart unnoticed.

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

## 6a. What the SHA-256 values mean — both products

Each subset record carries a `sha256`. Its scope is narrow and worth stating
plainly:

| | |
|---|---|
| **It is** | the hash of the file *this project retrieved*, computed locally at acquisition time |
| **It is not** | a publisher-issued canonical checksum. Neither GEBCO nor NOAA distributes per-subset digests for server-generated extracts, and these values must never be presented as though they do |
| **It guarantees** | the local file has not been altered since acquisition — the integrity checks in `tests/test_external_provenance.py` and `tests/test_external_noaa_crw.py` |
| **It does not guarantee** | that re-requesting the same window reproduces identical bytes |

That last point matters for reproducibility claims. A server-side subsetting
service may legitimately return a byte-different NetCDF serialisation — a
different library version, chunking, attribute ordering, or embedded generation
timestamp — while encoding **identical grid values**. So a hash mismatch after a
re-fetch means *"these are different bytes"*, not necessarily *"the data
changed"*. This is especially true of ERDDAP, which writes a fresh NetCDF on
every request.

Neither implementation assumes a re-download reproduces the hash. Both scripts
recompute and rewrite hashes on acquisition, and `--validate-only` re-verifies
the files already on disk. Nothing asserts byte-equality against a remote
re-request.

**Scientific identity is pinned by version and DOI, not by these file hashes:**

| Product | Version | DOI |
|---|---|---|
| GEBCO | `GEBCO_2026` | `10.5285/4f68d5c7-45eb-f999-e063-7086abc036fa` |
| NOAA CRW | `v3.1` | `10.25921/6jgr-pt28` |

Establishing value-level equivalence between two extracts would require
array-level comparison; that is deliberately out of scope here.

---

## 7. NOAA Coral Reef Watch 5 km v3.1 — scientific role

**Product:** NOAA Coral Reef Watch Daily Global 5 km (0.05°) Satellite Coral Bleaching Heat Stress Monitoring Product Suite
**Provider:** NOAA / NESDIS / STAR Coral Reef Watch Program
**Version:** **v3.1**, released 2018-08-01
**DOI:** `10.25921/6jgr-pt28` · **NCEI id:** `gov.noaa.nodc:CRW-5km-HeatStressProducts`
**Resolution:** 0.05° (~5 km), **daily**
**CRS:** EPSG:4326 (WGS 84), equirectangular, values at grid-cell centres
**Type:** Derived (L4) — SST is a model-assimilated analysis; the stress products are computed from it
**Acquired window:** 2018-01-01 → 2024-12-31 (seven complete calendar years)

### 7.1 The single most important thing about this product

**CRW thermal-stress products are predictors and context. They are not
biological labels.**

```text
DHW      != bleaching_percentage
HotSpot  != bleaching_percentage
BAA      != observed bleaching class
```

They quantify *heat stress* — a well-established driver of bleaching risk.
Whether corals at a site actually bleached is a **biological outcome** that
requires a diver, a quadrat, or a scored photograph. NOAA's own thresholds are
phrased as risk: 4 °C-weeks carries *"a risk of coral bleaching"*, 8 °C-weeks
*"reef-wide coral bleaching with mortality of heat-sensitive corals is likely."*
Likely is not observed.

Thresholding DHW into a `reef_health` class would rebuild exactly the
**label-construction leakage** that the [2026-08-19 audit](audits/dataset_scientific_audit_2026-08-19.md)
found in the synthetic dataset — except with real-looking inputs, which is
*worse*, because the result would survive casual review. This is enforced in
code: `validate_source()` rejects any product whose `provides_variables` names
`reef_health`, `restoration_suitability`, `coral_cover_percentage`,
`bleaching_percentage` or `disease_percentage`, using substring matching so
`"bleaching_percentage (derived from DHW)"` is caught too.

### 7.2 Variables acquired, and what each one means

| Variable (ERDDAP) | Units | Means | Does **not** mean |
|---|---|---|---|
| `analysed_sst` | `degree_C` | Daily night-only gap-free **sea-surface** temperature. An L4 analysis — spatially complete by construction, not by observation | Not an in-situ subsurface probe reading. Not interchangeable with the synthetic `water_temperature_c` |
| `sea_surface_temperature_anomaly` | `degree_C` | Departure of daily SST from the long-term climatological mean for that cell and day | Not a temperature. Its zero point is a *baseline choice*, not a physical origin |
| `hotspot` | `degree_C` | **Instantaneous** thermal stress: SST minus the site's Maximum Monthly Mean (MMM) climatology. ≥ 1 °C indicates bleaching-capable stress, and only those values accumulate into DHW | Not a bleaching observation, not a percentage. **Not restricted to positive values** — see below |
| `degree_heating_week` | `degree_Celsius_weeks` | **Accumulated** stress over the preceding 12 weeks — the running sum of HotSpot values ≥ 1 °C | Not a bleaching percentage. NOAA's 4/8 °C-week thresholds describe *risk*, not measured outcome |

Note the units on DHW: **`degree_Celsius_weeks`, not `degree_C`.** A DHW silently
treated as a temperature is a different physical quantity. The units are pinned
per file in the manifest and asserted in `tests/test_external_noaa_crw.py`.

**HotSpot is signed in the archive.** NOAA's published HotSpot *maps* show only
positive values, but the archived variable carries the full signed SST − MMM
difference (declared valid range −15 … +15 °C). The acquired files contain
negatives down to −11.95 °C in the Gulf of Kutch — that is a genuine winter
observation of water far below the summer MMM, **not missing data and not a
fill value**. Anyone clipping it to zero should do so deliberately.

### 7.3 Mapping onto the project — and where it deliberately stops

| CRW variable | Relationship to the project |
|---|---|
| `analysed_sst` | **Nearest analogue** to the synthetic `water_temperature_c`, but *not* a drop-in replacement — see below |
| `sea_surface_temperature_anomaly` | **New** environmental/context variable. No synthetic counterpart |
| `hotspot` | **New** thermal-stress context variable. No synthetic counterpart |
| `degree_heating_week` | **New** accumulated-stress context variable. No synthetic counterpart |

On SST vs `water_temperature_c`: the synthetic column was specified as a
thermistor/CTD reading at reef depth. CRW SST is a **sea-surface** analysis
averaged over a ~5 km cell. Substituting one for the other would misdescribe
what was measured — a different depth, a different footprint, and a different
measurement process. If a real thermal feature is ever built, it must carry a
new name and an explicit statement of what it is.

**Three of the four variables have no synthetic counterpart at all.** That is
the point. The real-data architecture is allowed to be richer than the legacy
15-column synthetic schema, and forcing these into it would discard information.
**No model schema changed in this task.**

### 7.4 What CRW cannot provide

- `coral_cover_percentage` · `bleaching_percentage` · `disease_percentage`
- `reef_health` · `restoration_suitability`
- `sonar_backscatter` · `acoustic_complexity_index` · `rugosity_index` · `depth_m`
- `ph` · `turbidity_ntu` · `dissolved_oxygen_mg_l` · `salinity_ppt`
- in-situ subsurface `water_temperature_c` at reef depth

### 7.5 Licence and source lineage — two separate questions

These are deliberately kept apart. **7.5.1** settles whether we may use and
redistribute the files we hold. **7.5.2** describes where the numbers came from.
The second does not establish the first.

#### 7.5.1 Licence — the actual basis

NOAA Coral Reef Watch's own disclaimer page states:

> NOAA Coral Reef Watch (CRW) data posted on the internet are freely available to
> the public. **All content on this website is considered to be in the public
> domain and may be distributed freely.** We rely on the ethics and integrity of
> the user to ensure that the source of data and products is appropriately cited
> and credited.

The `license` attribute *inside the delivered NetCDF files* repeats that the data
*"are available for use without restriction"*, with credit requested.

That is the whole basis for `licence_verified = True` and
`redistribution_allowed = True`: **NOAA CRW's published terms for the product it
distributes, plus the licence metadata delivered with the files.** It is **not
inferred from** the acquisition window's position in the source lineage below.

That distinction matters, and not merely pedantically. An earlier draft of this
document argued that the acquired window is unrestricted *because* it falls
outside the OSTIA period. That argument was wrong on its own terms — see 7.5.2 —
and had it been the real basis, the licence determination would have collapsed
along with it.

**Attribution is expected.** The full NCEI citation (Liu et al. 2018) is carried
in the manifest's `citation` field. Raw files are git-ignored — a **storage**
decision, as with GEBCO, not a licensing one.

#### 7.5.2 Source lineage — recorded for provenance

CoralTemp is assembled from more than one analysis. Re-verified 2026-08-22
against NOAA's CoralTemp source-history page,
[`index_5km_sst.php`](https://coralreefwatch.noaa.gov/product/5km/index_5km_sst.php).

| Period | Primary analysis |
|---|---|
| Jan 1985 – Nov 2002 | Met Office **OSTIA** reanalysis contributes directly |
| **Nov 1–29, 2002** | OSTIA and NOAA reprocessed Geo-Polar Blended **linearly merged** |
| Nov 2002 – Oct 2016 | NOAA **reprocessed** Geo-Polar Blended |
| **Oct 1–29, 2016** | reprocessed and near-real-time Geo-Polar **merged** |
| Oct 2016 – present | NOAA **operational** near-real-time Geo-Polar Blended |

These are **not discrete day-level eras.** Apart from the two merge windows,
NOAA describes the handovers at month granularity, and does not publish the
daily blend weights within either merge. The table should be read as NOAA writes
it, not as a set of exact changeover dates.

NOAA states both merges explicitly:

> The OSTIA reanalysis and NOAA's reprocessed 5km Geo-Polar Blended SST dataset
> were linearly merged over a period of 29 days, from November 1-29, 2002.

> The merge point between NOAA's reprocessed and near real-time 5km Geo-Polar
> Blended SST datasets was performed over a 29 day period as well, from
> October 1-29, 2016.

**OSTIA does not disappear after 2002.** NOAA states:

> Bias corrections originally used the NOAA National Centers for Environmental
> Prediction (NCEP) real-time global SST. However, **in 2016, the NOAA 5km
> Geo-Polar Blended SST product** (which CRW's 5km satellite monitoring for coral
> reefs is based on) **switched to using OSTIA as the bias correction.**

So OSTIA is an input to the product this project actually holds — as a
bias-correction reference rather than a directly merged source. **Nothing in our
2018–2024 window may be described as "OSTIA-free."** Tests enforce that wording
in both the script and this document.

Why the OSTIA lineage is worth tracking at all: the OSTIA reanalysis itself
carries restrictive terms — *"pure academic research only, with no commercial or
other application"*, Met Office Standard Terms, a **reproduction licence
application form required before use**, a **five-year** cap, Crown Copyright.
Those terms are a reason to stay out of the period where OSTIA is a *direct*
constituent, which is a provenance judgement we are making, not one NOAA imposes.

#### 7.5.3 The acquisition floor — a project policy

`scripts/fetch_noaa_crw.py` defines
`FIRST_POST_OSTIA_BLEND_REQUEST_DATE = "2002-12-01"` and refuses to build any
request starting earlier:

> Refusing to request CoralTemp data starting …, which is before 2002-12-01.
> This is a conservative project policy, not a NOAA licence boundary: it keeps
> acquisition clear of the period where the Met Office OSTIA reanalysis
> contributes directly to CoralTemp (January 1985 to November 2002) and of the
> November 1-29, 2002 window over which NOAA linearly merged OSTIA into the
> reprocessed Geo-Polar Blended analysis. …

**What the floor is:** a conservative provenance policy — the earliest date this
project chooses to retrieve, set past the documented direct-OSTIA period and past
the 2002 linear-merge window, with a one-day margin rounded to a month boundary
because the daily merge weights are unpublished.
`OSTIA_MERGE_END_DATE = "2002-11-29"` records the documented merge end alongside
it.

**What the floor is not:**

- a NOAA-defined licence boundary — NOAA draws no such line;
- proof that later CoralTemp is free of OSTIA influence — it is not, per 7.5.2;
- a claim that 2002-11-30 is scientifically "OSTIA-free".

The constant is named for the policy it encodes. Earlier names
(`OSTIA_LICENCE_BOUNDARY`, `FIRST_UNRESTRICTED_CORALTEMP_DATE`) asserted a legal
boundary NOAA has not established, and are now banned by test.

The acquired window begins **2018-01-01**, well past the floor.

### 7.6 Why 2018-01-01 → 2024-12-31

Considered: (A) 2018→2024, (B) 2018→latest available, (C) something else. **A was
chosen**, for five reasons — one of which is a provenance reason, not a
convenience:

1. **Exact alignment with the synthetic benchmark.** `observations.csv` spans
   2018-01-01 → 2024-12-31. Matching it exactly is what makes the descriptive
   comparison in §7.9 an honest like-for-like rather than an artefact of
   different windows.
2. **Complete calendar years.** Seven whole years, so annual means and annual
   maxima are not biased by a partial year — which option B would have
   introduced, since 2026 is incomplete.
3. **A single primary SST analysis throughout.** 2018 onward sits wholly inside
   the operational Geo-Polar Blended period (October 2016–present), well clear
   of the October 1–29, 2016 merge. The window therefore contains **no
   intra-window discontinuity** in how SST was produced. It also sits after the
   2016 switch to OSTIA-based bias correction, so that treatment is uniform
   across the window too.
4. **Clear of the direct-OSTIA period.** Satisfies the §7.5.3 policy floor with
   room to spare — 2018-01-01 is over fifteen years past the November 2002
   merge, so it does not rest on exactly where that floor is drawn. Note this is
   a *provenance* reason: it does not make the window "OSTIA-free" (§7.5.2), and
   it is not the basis for the licence determination (§7.5.1).
5. **Finalised, not near-real-time.** The window ends 20 months before
   acquisition, so it is not subject to the revision that recent NRT data can
   still receive.

It also captures real thermal-stress variability — the acquired data shows 2024
as the peak DHW year in three of the four regions and 2020 as a second peak (see
§7.11). The full 1985-present archive was deliberately **not** downloaded.

### 7.7 Geographic extraction — bbox subsets, not virtual stations

Two options were genuinely available.

**NOAA CRW Regional Virtual Stations exist for all four systems** —
`lakshadweep`, `gulf_of_mannar`, `gulf_of_kutch`, `andaman`, `great_nicobar` —
as NOAA-defined polygons with published daily time series. They were
**considered and rejected** for this acquisition:

- They are **already spatially aggregated**, and not by a neutral statistic: the
  series report SST at the pixel holding the **90th-percentile HotSpot**, a
  deliberately warm-biased, management-oriented summary. That cannot be
  un-aggregated.
- They carry **no spatial structure at all** — one series per region — which
  would make the spatial half of validation impossible.
- Their BAA column still uses the **heritage** alert scale; NOAA states the
  Virtual Station pages *"still use the heritage bleaching alert level system"*
  pending an update.

**Gridded bbox subsets were used instead**, at native 0.05° resolution, over the
**same four windows as GEBCO_2026** (§5). Reusing the windows keeps the two
products comparable; `tests/test_external_noaa_crw.py` pins that equality against
the GEBCO manifest so they cannot drift apart silently.

**These remain acquisition windows, not reef masks.** Most cells inside them are
open ocean — GEBCO showed 98.8 % of marine cells in the Lakshadweep window are
deeper than 100 m. No observation coordinate may ever be drawn uniformly from
one of these. Refining them against a real reef mask is still blocked pending the
Allen Coral Atlas / UNEP-WCMC licence question (§10).

### 7.8 The product deliberately omitted — Bleaching Alert Area

BAA was a candidate and was **not acquired**, for three independent reasons:

1. **Redundant.** BAA is a deterministic function of the HotSpot and DHW values
   already acquired. It adds no information.
2. **Temporally incompatible.** The only BAA product on these servers is the
   **7-day maximum composite** — a rolling window dated on its final day. Mixing
   it with daily variables would smear a 7-day maximum across daily rows.
3. **Non-stationary inside our own window.** On **2023-12-15** NOAA revised the
   alert-level system, extending it from Alert Level 2 to **Alert Level 5**,
   after the 2023 heat extremes. The ERDDAP variable metadata **still declares
   the superseded scheme** (`valid_max=4`, flag meanings listing only Alert
   Levels 1–2). A naive ingest of 2018–2024 would silently concatenate two
   different categorical scales into one column.

Reason 3 is the interesting one: it is a live metadata/semantics mismatch on the
authoritative server, and it would not have been visible without reading NOAA's
product page rather than trusting the machine metadata. BAA can be reconstructed
later from HotSpot and DHW under a **single explicit scheme**, which is the
correct way to obtain it.

### 7.9 Limitations

1. **Not biological.** Restated because it is the failure mode that matters:
   these are thermal predictors, not reef-condition observations.
2. **Sea surface, not reef depth.** SST is a surface analysis; reef thermal
   environments at depth can differ substantially.
3. **~5 km cells.** One CRW cell is far larger than a reef. Sub-cell variation —
   lagoon versus fore-reef, shaded versus exposed — is entirely unresolved.
4. **L4 means gap-filled.** SST is *"gap-free"* by construction. Spatial
   completeness is a property of the analysis, not evidence of observation
   density; cloud-obscured days are still filled.
5. **No `mask` variable in the delivered files.** The retrieval endpoint does not
   expose CRW's land/ice/missing classification, so within these files **land and
   genuinely missing data are indistinguishable** — both are `NaN`. High NaN
   fractions in Gulf of Kutch and Gulf of Mannar are land, not data loss, but the
   files themselves cannot prove that. The CoastWatch server carries `mask` if
   this is ever needed.
6. **Missing dates exist.** The daily archive is not perfectly continuous; gaps
   are counted per file in the manifest rather than interpolated away.
7. **Anomaly baseline is a choice.** The acquired anomaly is the suite's own
   (MMM-based) product. NOAA separately publishes an anomaly against a 1991–2020
   standard baseline (2006–present). They are **different quantities** and must
   never be mixed.
8. **Update latency.** The operational products update daily at ~13:30 US
   Eastern; the acquired window is historical and stable.

### 7.10 What was acquired, and what validation found

**16 files, 694 MB (662 MiB), all git-ignored.** Every file opened, and every geometry,
unit, fill value and time axis matched what the source metadata declares.

**Delivered types are not uniform.** SST arrives as `float64`; SST anomaly,
HotSpot and DHW arrive as `float32` — even though the ERDDAP metadata declares
all four as double. This is recorded per file (`variable_dtype`) rather than
assumed, because it sets the precision floor for any later comparison. It is
also why the acquisition is 694 MB rather than the ~1.12 GB upper bound the
dry-run prints.

| Region | Grid | Cells | NaN % | SST mean | SST min–max | SSTA range | HotSpot max | DHW max |
|---|---|---|---|---|---|---|---|---|
| Lakshadweep | 93 × 54 | 5,022 | **0.00** | 29.19 | 24.95 – 32.20 | −1.81 … +2.77 | 2.24 | 9.88 |
| Gulf of Mannar | 21 × 33 | 693 | 27.27 | 28.93 | 25.17 – 32.97 | −1.99 … +3.48 | 2.54 | 9.22 |
| Gulf of Kutch | 21 × 36 | 756 | 54.89 | 26.83 | **17.09** – 31.76 | −5.56 … +3.94 | 2.78 | **20.45** |
| Andaman & Nicobar | 151 × 47 | 7,097 | 2.54 | 29.13 | 25.78 – 32.71 | −1.60 … +3.26 | **3.13** | 13.39 |

Every value sits inside its declared valid range; **no decoded fill value
survived anywhere** (nothing near ±327.68), and DHW is non-negative everywhere as
its definition requires.

**The NaN percentages are land, not data loss** — 54.9 % in the Gulf of Kutch is
the Kathiawar peninsula and Kachchh mainland, exactly as the GEBCO bathymetry
predicted (37.3 % marine). But note limitation 5: **the files cannot prove that**,
because they carry no `mask` variable. The inference is ours, from bathymetry.

**Lakshadweep has 0.00 % NaN** — a striking result. At 5 km resolution the atolls
are too small to occupy a single land cell, so the entire window reads as ocean.
That is the clearest possible illustration of why a 5 km product cannot resolve
these reefs, and why the window is not a reef mask.

**Cross-server verification.** The same subset was fetched from both NOAA ERDDAP
servers and compared cell by cell:

| Variable | Max abs difference | Cells compared | Verdict |
|---|---|---|---|
| `analysed_sst` | 0 | 2,520 | agree |
| `sea_surface_temperature_anomaly` | 2.86 × 10⁻⁸ | 2,520 | agree |
| `hotspot` | 2.29 × 10⁻⁷ | 2,520 | agree |
| `degree_heating_week` | 0 | 2,520 | agree |

The two non-zero differences are **exactly float32 round-trip noise**: 2.29 × 10⁻⁷
against values of order 1 is float32 epsilon (≈1.19 × 10⁻⁷), and both affected
variables are the ones delivered as `float32`. SST and DHW, compared bit-for-bit,
differ by **zero**. There is no data disagreement between the two NOAA servers.

They do encode **missing data** differently — CoastWatch declares
`_FillValue = -327.68`, PIFSC delivers `NaN` — which is a serialisation
difference, reported separately rather than counted as disagreement.

### 7.11 Temporal sanity — the data has real structure

**Missing dates: 31 December 2022, 2023 and 2024 are absent from the archive**
in all four products, plus 30 December 2024 in SST anomaly only. The gaps are
**identical across all four regions** for a given product, which is what
distinguishes an archive gap from a subsetting bug — four independent requests
agreeing on which days exist. `tests/test_external_noaa_crw.py` pins that
agreement.

A consequence worth stating: **ERDDAP's `(time)` selector snaps to the nearest
available step.** Because 2024-12-31 does not exist, requesting it returned
2025-01-01, so every series ends one day past the requested end. The manifest
records the *actual* range rather than the requested one.

**Monthly SST climatology (°C, spatial mean over each window):**

| Region | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec | Amplitude |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Lakshadweep | 28.68 | 28.80 | 29.48 | 30.41 | 30.41 | 29.49 | 28.53 | 28.28 | 28.58 | 29.11 | 29.37 | 29.15 | **2.13** |
| Gulf of Mannar | 27.42 | 27.72 | 28.99 | 30.59 | 30.45 | 29.53 | 28.97 | 28.59 | 28.75 | 29.16 | 28.86 | 28.08 | **3.18** |
| Gulf of Kutch | 21.43 | 22.03 | 24.40 | 27.32 | 29.20 | 30.06 | 29.66 | 28.54 | 28.79 | 29.01 | 27.31 | 23.91 | **8.63** |
| Andaman & Nicobar | 28.38 | 28.44 | 29.11 | 30.22 | 30.35 | 29.45 | 29.06 | 28.88 | 28.78 | 29.05 | 29.15 | 28.69 | **1.97** |

This is physically coherent: a **pre-monsoon April/May peak** in the three
tropical systems, an **August minimum** in Lakshadweep consistent with
south-west monsoon mixing and cloud cover, and a far larger swing in the Gulf of
Kutch — a shallow, high-latitude, macrotidal gulf that genuinely cools to 21 °C
in January.

**Annual maximum DHW (°C-weeks):**

| Region | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|---|---|
| Lakshadweep | 1.07 | 1.31 | 5.85 | 2.56 | 0.98 | 3.00 | **9.88** |
| Gulf of Mannar | 1.29 | 4.20 | 4.83 | 0.95 | 1.79 | 1.40 | **9.22** |
| Gulf of Kutch | 3.14 | 10.18 | **20.45** | 11.38 | 5.92 | 9.00 | 18.33 |
| Andaman & Nicobar | 3.98 | 4.75 | 3.80 | 4.03 | 5.19 | 3.81 | **13.39** |

**2024 is the peak year in three of four regions**, consistent with the global
heat stress of that year, and 2020 stands out in Lakshadweep and the Gulf of
Kutch. This is the interannual variability the synthetic dataset has none of.

> These are **thermal** diagnostics. They say nothing about whether corals at
> these sites bleached, recovered, or died. That would require in-water
> observation.

### 7.12 Cross-region comparison

| Region | SST range | Days DHW ≥ 4 | Days DHW ≥ 8 | Days HotSpot ≥ 1 | NaN % |
|---|---|---|---|---|---|
| Lakshadweep | 7.25 | 172 | 64 | 188 | 0.00 |
| Gulf of Mannar | 7.80 | 222 | 51 | 149 | 27.27 |
| Gulf of Kutch | **14.67** | **686** | **366** | **518** | 54.89 |
| Andaman & Nicobar | 6.93 | 303 | 76 | 277 | 2.54 |

The four regions are clearly distinct, and the ordering is physically sensible —
the shallow northern gulf is the most thermally extreme on every measure.

**No suspicious duplication.** Pairwise comparison of the daily spatial-mean SST
series found no identical pair; correlations range from **+0.169** (Lakshadweep
vs Gulf of Kutch — opposite seasonal phase) to **+0.874** (Gulf of Mannar vs
Andaman & Nicobar — both equatorial). Had a subsetting or indexing bug returned
the same window twice, correlations would have been 1.000.

### 7.13 Real vs synthetic — descriptive only, no join

Compared **descriptively** against the synthetic `water_temperature_c` over the
same 2018–2024 span. **Nothing was joined, and no synthetic value was altered.**
The synthetic file's SHA-256 was verified identical before and after reading.

| Region | Real min | Syn min | Real mean | Syn mean | Real max | Syn max | Real var | Syn var |
|---|---|---|---|---|---|---|---|---|
| Lakshadweep | 24.95 | 27.20 | 29.19 | 29.40 | 32.20 | 31.84 | 0.70 | 0.89 |
| Gulf of Mannar | 25.17 | 26.45 | 28.93 | 30.53 | 32.97 | 33.50 | 1.29 | 2.43 |
| Gulf of Kutch | 17.09 | 20.85 | 26.83 | **31.89** | 31.76 | **37.50** | 9.39 | 19.65 |
| Andaman & Nicobar | 25.78 | 26.24 | 29.13 | 28.83 | 32.71 | 31.50 | 0.58 | 0.94 |

Three defects in the synthetic generator are now quantified against real data:

**1. Seasonality is essentially absent from the synthetic data.**

| Region | Real amplitude | Synthetic amplitude | Real warmest month | Syn warmest month |
|---|---|---|---|---|
| Lakshadweep | **2.13** | 0.19 | May | December |
| Gulf of Mannar | **3.18** | 0.31 | April | April |
| Gulf of Kutch | **8.63** | 1.09 | June | August |
| Andaman & Nicobar | **1.97** | 0.13 | May | November |

The synthetic annual cycle is **10–15× too small** and its warmest month is
essentially arbitrary — unsurprising, since the generator draws temperature from
a uniform distribution independent of the timestamp. The `timestamp` column
carries no seasonal signal at all. This is the §6 finding of the 2026-08-19
audit, now measured against observations rather than argued from the code.

**2. The Gulf of Kutch is drastically mis-specified.** Its synthetic mean
(31.89 °C) is **5.1 °C above** the real mean (26.83 °C), and **52.5 % of its
2,250 synthetic rows exceed the highest SST actually observed there in seven
years** (31.76 °C). The synthetic maximum of 37.50 °C — the audit's lethal-value
finding — is **5.7 °C above** anything real.

**3. Synthetic variance is roughly double reality** in every region (Kutch 19.65
vs 9.39; Andaman 0.94 vs 0.58), so the generator's noise is not merely
mis-centred but too wide.

Rows exceeding the real observed regional maximum:

| Region | Real max | Synthetic rows above it |
|---|---|---|
| Lakshadweep | 32.20 | 0 / 3,450 (0.0 %) |
| Gulf of Mannar | 32.97 | 148 / 3,000 (4.9 %) |
| Gulf of Kutch | 31.76 | **1,181 / 2,250 (52.5 %)** |
| Andaman & Nicobar | 32.71 | 0 / 6,300 (0.0 %) |

One caveat on the comparison itself: real SST is a **sea-surface, 5 km-cell,
daily** quantity and the synthetic column purports to be an **in-situ point
probe**. They are not the same measurement, so this is a plausibility check on
the generator's ranges and seasonality — **not** evidence that CRW SST should
replace the column. **The generator was not recalibrated, and no model was
retrained.**

---

## 8. Reproducing the acquisitions

```bash
python scripts/fetch_gebco_2026.py --dry-run        # print the plan, fetch nothing
python scripts/fetch_gebco_2026.py                  # fetch, validate, write manifest
python scripts/fetch_gebco_2026.py --validate-only  # re-validate local files

python scripts/fetch_noaa_crw.py --dry-run          # print the plan, fetch nothing
python scripts/fetch_noaa_crw.py --cross-check      # fetch, validate, compare servers
python scripts/fetch_noaa_crw.py --validate-only    # re-validate local files
```

GEBCO is requested from the **THREDDS NetCDF Subset Service** hosted by CEDA on
behalf of BODC/GEBCO. CRW is requested from **ERDDAP griddap** on NOAA servers.
Both return NetCDF-3 classic — readable with `scipy.io.netcdf_file`, so **no new
runtime dependency was introduced by either product**. Neither global archive is
ever downloaded.

---

## 9. What has NOT been done

Each acquisition was done in isolation. Specifically **not** done:

- No change to `generate_data.py`, `preprocess.py`, `build_features.py`,
  `get_feature_columns()`, or the Pandera schema
- No external rows appended to `observations.csv`
- **No join between GEBCO and CRW**, and no join between either and any other
  table. They remain two independent products over a shared extent
- No site table, no training table, no new CSVs
- No labels created of any kind
- No model trained, registered, promoted, or evaluated
- No DVC stage added; the DAG is unchanged
- No MLflow interaction
- No API or dashboard change

---

## 10. Other sources — status

From the acquisition plan, with one **correction**.

| Source | Status |
|---|---|
| **GEBCO_2026** | **ACQUIRED** — public domain, verified |
| **NOAA Coral Reef Watch 5 km v3.1** | **ACQUIRED** — public domain, verified (see §7) |
| **Allen Coral Atlas** | **LICENCE REQUIRES VERIFICATION** — see below |
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
