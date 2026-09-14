# External (Real) Data Layer

**Status:** three real external datasets acquired — GEBCO_2026 bathymetry, NOAA
Coral Reef Watch 5 km v3.1 thermal products, and the Seaview Survey tabular
benthic-cover data (the first **biological** source) — across four manifests,
because CRW was acquired twice under separate scopes: the 2018–2024 **India**
windows (§7) and a 2015–2017 **Maldives** historical extension (§8a).
One analysis joins two of them: the Seaview Maldives paired temporal analysis
in §8a. Everything else remains unjoined.
**Created:** 2026-08-19 · **Updated:** 2026-08-30

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

## 8. Seaview Survey — the first biological source

GEBCO says how deep the water is. CRW says how hot it got. Neither says
anything about what is actually *living* on the reef. The Seaview Survey
tabular data is the first product in this repository that does.

It says it at one remove, though, and the whole of §8.4 is about that remove:
the imagery, the dates and the coordinates are real field records, while the
cover values are **image-derived benthic-cover estimates** computed from those
images by a classifier.

| | |
|---|---|
| Title | Seaview Survey Photo-quadrat and Image Classification Dataset |
| Provider | The University of Queensland (UQ eSpace `UQ:734799`) |
| DOI | [10.14264/uql.2019.930](https://doi.org/10.14264/uql.2019.930) |
| Data descriptor | Rodriguez-Ramirez et al. (2020), *Sci Data* **7**, 355 — [10.1038/s41597-020-00698-6](https://doi.org/10.1038/s41597-020-00698-6) |
| Released | 2019 (archive published 2019-12-09) |
| Collected | 2012-09-16 → 2018-05-05 globally |
| Licence | **CC BY 3.0 Unported** (normalized), Open Access — verified from the eSpace record; publisher's own label preserved in §8.10 |
| Acquired | `tabular-data.zip` (341 MB, 18 CSVs) + the publisher's documentation PDF |
| Not acquired | the >1 000 000-image photo-quadrat archive, annotated images, survey previews |

### 8.1 INDIAN OCEAN IS NOT INDIA

**This is the single most important fact about this dataset for this project,
and it is the easiest one to get wrong.**

The dataset's Central Indian Ocean component is **92 surveys**, and every one of
them is in one of two places:

| Territory | Surveys |
|---|---|
| Maldives (`MDV`) | 63 |
| Chagos Archipelago / British Indian Ocean Territory (`CHA`) | 29 |

**No survey in this dataset is in India.** Verified directly: no row of
`seaviewsurvey_surveys.csv` has `country == 'IND'`.

There is a genuine naming trap here. The `ocean` column uses `IND` for the
**Indian Ocean basin**, and `IND` is also the ISO 3166-1 alpha-3 code for
**India**. The human-annotation files are named `annotations_IND_MDV.csv` and
`annotations_IND_CHA.csv`. Read casually, those look like Indian data. They are
not.

Against the project's four target reef systems:

| Project reef system | Surveys here | Nearest Seaview transect |
|---|---|---|
| Lakshadweep | **0** | ~386 km |
| Gulf of Mannar | **0** | ~654 km |
| Gulf of Kutch | **0** | ~1 942 km |
| Andaman and Nicobar Islands | **0** | ~2 046 km |

The Seaview Indian Ocean envelope is latitude −6.699 → **+4.523**, longitude
71.235 → 73.589. Lakshadweep begins near 8°N. The longitude bands overlap; the
latitudes do not come close.

The manifest carries this as a first-class field,
`geographic_transfer_status = "INDIAN_OCEAN_NOT_INDIA"`.

**Consequence.** This dataset does **not** validate any model for Lakshadweep,
the Gulf of Mannar, the Gulf of Kutch, or the Andaman and Nicobar Islands.
Fitting on Maldivian and Chagossian observations and applying the result to
Indian reefs is a **geographic transfer** across a minimum 386 km gap into a
different reef province. That may still be scientifically worthwhile — it is a
far better starting point than nothing — but it must be reported as transfer,
never as Indian validation.

### 8.2 The four observational units

    survey  ──▸  image  ──▸  quadrat  ──▸  annotation point

| Unit | What it is | Indian Ocean count |
|---|---|---|
| Survey | one transect visit, 1.5–2.0 km at ~10 m depth | 92 |
| Image | one raw photograph | 81 773 |
| Quadrat | one standardised ~1 m² crop of an image | 137 698 |
| Annotation point | one classified pixel location (50 per quadrat) | ~6.9 M |

**Multiple quadrats from one image are not independent locations.** They are
neighbouring patches of a single photograph, sharing its position, altitude,
exposure and moment in time. In the Indian Ocean subset, 49 156 images yield 2
quadrats each and 1 306 yield 6 (one yields 15, fourteen yield 16).

> Treating 137 698 quadrats as 137 698 independent sites would overstate the
> sample by roughly an order of magnitude. **The real spatial sample is 92
> transects.**

The aggregation rule from the data descriptor, which the manifest records:
proportional cover per quadrat = points for a label ÷ total points; average
quadrats within an image; average images within a survey; sum detailed labels
into the five functional groups via `seaviewsurvey_labelsets.csv`.

Reproducing that rule from the quadrat table recovers the published survey-level
`pr_hard_coral` to a mean absolute difference of **0.0025** (max 0.040; 1 survey
of 92 above 0.02) — which is the check that the rule above is the rule that was
actually used.

### 8.3 Tables

| File | Rows | Unit |
|---|---|---|
| `seaviewsurvey_surveys.csv` | 860 | survey |
| `seaviewsurvey_quadrats.csv` | 1 082 324 | quadrat (the hierarchy table) |
| `seaviewsurvey_labelsets.csv` | 228 | label definition |
| `seaviewsurvey_annotations.csv` | 55 229 185 | automated annotation point |
| `seaviewsurvey_reefcover_indianocean.csv` | 137 698 | quadrat cover (45 labels) |
| `seaviewsurvey_reefcover_{atlantic,southeastasia,pacificaustralia,pacifichawaii}.csv` | 285 008 / 250 138 / 316 369 / 93 111 | quadrat cover |
| `annotations_{ATL,IND_CHA,IND_MDV,PAC_*}.csv` | 52 450 – 186 420 | **human** annotation point |

### 8.4 The cover values are a classifier output, not a diver's notes

Benthic cover here was produced by **nine VGG-D 16 convolutional neural
networks** (one per country/region), each classifying **50 points per quadrat**.

For the Indian Ocean subset:

| | Quadrats | Share |
|---|---|---|
| Carry human expert annotations | 2 298 | **1.67 %** |
| Classified by CNN only | 135 400 | **98.33 %** |

#### Three layers of evidence, not one

The temptation is to say "real reef survey, therefore real observed coral
cover". That does **not** follow: the survey is real, but the cover number is
two steps away from it.

| Layer | What it is | Status |
|---|---|---|
| **A** — field survey imagery and coordinates | real photographs of real reef, with real transect coordinates, dates and ~10 m depth | **real field record** |
| **B** — human image annotations | expert annotators scoring points on the 1.67 % training/test subset (`annotations_*.csv`) | **human judgement about an image**, not an in-water measurement |
| **C** — ML-classified benthic cover | the cover columns and the survey-level `pr_*` columns, 98.33 % CNN output | **model output** — an image-derived benthic-cover estimate |

**A is real, B is a human reading of A, and C is a model's reading of A.** The
columns this project would actually use are layer C.

The published validation is good — 97 % agreement between human and automated
annotations, errors of <2 %–7 %, R² = 0.97 — and this is a well-built product.
That validation is preserved exactly as published, but it is a property of the
classifier: **a well-validated estimate is still an estimate**, and accuracy
figures do not promote a prediction into a measurement.

So, throughout this repository:

| Use | Do not use |
|---|---|
| image-derived benthic-cover estimate | directly observed coral cover |
| ML-estimated benthic cover | field-measured coral cover |
| image-derived biological response estimate | measured coral cover |
| | biological ground truth |

This matters practically: "we validated against observed coral cover" is **not**
a sentence this dataset supports, because 98 % of those observations are model
output.

### 8.5 Target compatibility — neither target exists here

| Project target | Verdict |
|---|---|
| `reef_health` | **PARTIAL BIOLOGICAL EVIDENCE ONLY — NOT A TARGET** |
| `restoration_suitability` | **NO DIRECT TARGET** |

    hard_coral_cover  !=  reef_health
    algal_cover       !=  reef_health
    benthic class     !=  restoration_suitability

`reef_health` in the synthetic pipeline is a constructed *condition judgement*.
Image-derived hard-coral cover is an *estimate of one component* of reef state.
Thresholding cover into a health class would rebuild exactly the
label-construction leakage the
[2026-08-19 audit](audits/dataset_scientific_audit_2026-08-19.md) found — this
time out of real numbers, which makes it **harder** to spot, not safer.

`restoration_suitability` has no counterpart at all: no intervention, no
restoration outcome, no site-selection judgement, no management variable.

**What is legitimate:** survey-level hard-coral cover is an **image-derived
continuous biological response estimate**, and may be used as the response
variable in a separate real-data analysis provided it is described that way.
That is a different modelling problem from the registered synthetic
classifiers, and it does not change, retrain or supersede them.

**Which field to use, when that analysis happens.** Use the publisher's
survey-level **`pr_hard_coral`** from `seaviewsurvey_surveys.csv`, not a
hard-coral total reconstructed from the quadrat label columns of
`seaviewsurvey_reefcover_indianocean.csv` — those columns carry the export
defect described in §8.8, and the survey is in any case the honest spatial unit
(92 transects, not 137 698 pseudo-independent quadrats). **No such analysis has
been performed here; this is a recommendation for the first one.**

### 8.6 What the Indian Ocean subset actually contains

Survey-level cover, all 92 surveys — image-derived estimates, proportions
summing to 1.000 ± 0.0004:

| | mean | sd | min | median | max |
|---|---|---|---|---|---|
| Hard coral | 0.177 | 0.083 | 0.018 | 0.176 | 0.531 |
| Algae | 0.711 | 0.135 | 0.206 | 0.727 | 0.954 |
| Soft coral | 0.025 | 0.039 | 0.001 | 0.006 | 0.209 |
| Other invertebrates | 0.007 | 0.008 | 0.000 | 0.004 | 0.041 |
| Other / substrate | 0.081 | 0.075 | 0.002 | 0.058 | 0.398 |

Chagos and the Maldives are not interchangeable. Chagos carries more hard coral
(0.211 vs 0.161) and an order of magnitude more soft coral (0.067 vs 0.006);
the Maldives carries more algae (0.765 vs 0.593).

**Missingness: none.** No blank cell in `seaviewsurvey_surveys.csv` or in the
Indian Ocean cover table. **No outliers were removed** — every delivered row is
present as published.

### 8.7 Temporal structure — a paired pre/post design, in one country only

| | Chagos | Maldives |
|---|---|---|
| Date range | 2015-02-12 → 2015-02-24 | 2015-03-29 → 2017-04-01 |
| Distinct dates | 12 | 33 |
| Epochs | **one** (Feb 2015) | **two** (Mar–Apr 2015; Mar–Apr 2017) |
| Repeat transects | **0** | **26** |

Twenty-six Maldivian transects were surveyed twice, **703–722 days apart**,
bracketing the 2016 mass-bleaching period. Of 66 distinct Indian Ocean
transects, 40 were visited once and 26 twice; no transect was visited three
times.

So what those 26 pairs support is a **paired pre/post survey change around the
2016 mass-bleaching period** — for the Maldives only. Chagos has a single
expedition and no repeats, so no paired comparison is possible there at all.

**What that is.** A real temporal contrast: the same transect, twice, roughly
two years apart, with an image-derived benthic-cover estimate at each visit.

**What that is not.** It is **not a "bleaching response"**, and this document
does not call it one. The dataset contains no bleaching observation, no thermal
covariate and no control. A difference in cover between two visits is a
difference in cover; on its own it does not establish that bleaching caused it,
because storm damage, disease, predation, transect re-navigation and
classifier/sampling variation are not excluded by the design.

**What would strengthen it, and how far.** NOAA CRW HotSpot and Degree Heating
Week exposure — already acquired in this repository, §7 — should eventually be
linked to these transects **by site and date**. That linkage has **not** been
performed; no join exists between Seaview and CRW. And even once it does exist,
the design stays observational: it can support an **association** between
thermal exposure and cover change. It does not license automatic causal
attribution.

That is the honest description. This is not a longitudinal monitoring programme
with a regular revisit schedule; it is two expeditions two years apart.

### 8.8 A defect in the published Indian Ocean table — recorded, not repaired

`seaviewsurvey_reefcover_indianocean.csv` has **no `MASE_MEA_L` column**, even
though `MASE_MEA_L` (*Lobophyllia*, functional group **Hard Coral**) is defined
for the Indian Ocean in `seaviewsurvey_labelsets.csv` and does appear in
`seaviewsurvey_annotations.csv`. The same table instead carries `MASE_LRG_I`
(*Isopora*) — a label defined only for **Southeast Asia** — which is zero across
all 137 698 rows.

The evidence is unambiguous in both directions:

- all **135 185** quadrats with no `MASE_MEA_L` annotation sum to **exactly
  1.00** across label columns;
- all **2 513** quadrats that do have `MASE_MEA_L` points sum to **less than
  1.00** (mean 0.963, min 0.44).

**Impact:** quadrat-level hard coral cover is under-reported for 2 513 of
137 698 Indian Ocean quadrats (1.83 %), by 4 669 classified points; the worst
affected quadrat loses 56 % of its cover. The survey-level `pr_hard_coral`
values in `seaviewsurvey_surveys.csv` do *not* show this shortfall and appear to
include the dropped label.

**Handling:** recorded, not repaired. The CSV has not been edited; nothing has
been imputed, dropped or rescaled; no affected row has been removed. The raw
bytes are untouched and their SHA-256 is pinned in the manifest. For
survey-level hard coral cover, prefer `seaviewsurvey_surveys.csv`. For
quadrat-level work, either accept the documented 1.83 % under-count or recover
the points from `seaviewsurvey_annotations.csv`.

**Recommendation for the first biological-response analysis:** use the
publisher-provided survey-level **`pr_hard_coral`** rather than reconstructing
hard-coral cover from these known-defective quadrat label columns. Not
performed here — see §8.5.

### 8.9 The intended real-biological track — designed, not built

```
        Seaview survey record
        (survey date · location · image-derived cover estimate)
                        │
        ┌───────────────┼───────────────┐
        │               │               │
   NOAA CRW        GEBCO             image-derived
   thermal         bathymetric       biological
   context         context           response estimate
```

**None of this exists yet. No join has been performed in this task, and no model
has been trained.** The diagram records intent so the next step is a decision
rather than a drift.

Three things that must be settled *before* any such join:

1. **The spatial unit.** Seaview's real sample is 92 transects; CRW's cell is
   ~5 km. Several transects may fall in one CRW cell.
2. **The temporal unit.** A survey is a single day. Thermal history is a
   trajectory. Which window — and chosen on what grounds, decided before
   looking at the response?
3. **The transfer.** Any such model is fitted on Maldives and Chagos. See §8.1.

### 8.10 Licence — normalized, with the publisher's own wording preserved

`licence_verified = True`, `redistribution_allowed = True`.

Read from the **authoritative dataset record** in UQ eSpace (`UQ:734799`), whose
licence field reads verbatim *"Creative Commons Attribution 3.0 International
(CC BY 3.0)"* with access conditions *"Open Access"*, and which supplies the
licence URI `https://creativecommons.org/licenses/by/3.0/`. Cross-checked
against the peer-reviewed data descriptor, whose Usage Notes state the dataset
is released under a *"Creative Commons Attribution license (CC BY 3.0)"*.
**The two agree.**

**Not** established by the DOI: Crossref registers `10.14264/uql.2019.930` with
title, publisher, year and creators, and **no licence field at all**. A DOI that
resolves is not a licence statement.

**The label is imprecise; the URI is not.** There is no CC BY *"3.0
International"*: version 3.0 was issued as **Unported** plus ported national
forms, and the "International" wording only begins at version 4.0 — which is
not what was granted here. The licence URI in the same record is unambiguous
and resolves to Creative Commons Attribution 3.0 Unported, so **the URI
governs**. Two facts are therefore recorded side by side rather than one
overwriting the other:

| Field | Value |
|---|---|
| `source_reported_licence_label` | "Creative Commons Attribution 3.0 International (CC BY 3.0)" — UQ's own wording, kept verbatim |
| `normalized_licence` | **CC BY 3.0 Unported** |
| `licence_url` | `https://creativecommons.org/licenses/by/3.0/` |

The manifest carries these in a `licence_normalization` block. UQ did write
that label, and the record says so; what changes is our reading of it, not the
publisher's history. The normalization changes wording and not permissions:
Unported and any ported 3.0 form both grant redistribution, adaptation and
commercial use subject to attribution.

`raw_tracked_in_git` remains `False`. The licence would permit committing 341 MB
of CSV; that is a storage decision, and the answer is still no.

**Transport caveat.** The publisher serves this collection over **plain HTTP**
only — `data.qld.edu.au` does not answer on port 443. The transfer was therefore
not authenticated in transit, which is why every acquired file carries a pinned
SHA-256 in the manifest.

---

## 8a. Seaview Maldives paired temporal analysis

**Added:** 2026-08-30. This is the first analysis in the repository that
**joins** two real external products — Seaview survey records and NOAA CRW
thermal history. Everything before it was acquired in isolation, and §10's list
of things not done is amended accordingly: that list still holds for GEBCO, and
for the Indian CRW windows, but no longer for this one pairing.

It answers exactly one question:

> Across the 26 Maldivian transects Seaview surveyed twice, is greater thermal
> exposure between the two visits **associated with** a larger decline in the
> image-derived hard-coral-cover estimate?

`MALDIVES_NOT_INDIA`. Every transect is Maldivian. No Indian reef is involved,
no Indian model is validated, and nothing produced here is Indian data.

### 8a.1 Why exactly 26 pairs

A transect enters the analysis only if it is in the Maldives (`ocean == 'IND'`
**and** `country == 'MDV'`) and has **exactly two** survey rows in
`seaviewsurvey_surveys.csv`. Of 37 Maldivian transects, 26 qualify and 11 were
visited once; no transect anywhere in the Indian Ocean subset was visited three
or more times, so nothing is discarded at the top end.

**Chagos contributes nothing, and is excluded by the data rather than by a
special case.** Its 29 surveys are a single February 2015 expedition with no
repeated transect, so the "exactly two visits" rule removes them without anyone
naming Chagos in the selection code.

**The inferential sample size is 26.** Seaview's Maldives component holds
thousands of photo-quadrats, but quadrats within a transect are
pseudo-replicates. Treating them as independent observations would inflate `n`
by two orders of magnitude and is not done anywhere in this analysis.

### 8a.2 The response, and why it is `pr_hard_coral`

The response is the publisher's **survey-level `pr_hard_coral`**, and the
paired change is

```
delta_hard_coral = second_pr_hard_coral - first_pr_hard_coral
```

with negative meaning decline. `absolute_change` and `relative_change` are also
recorded, descriptively; relative change divides by the first visit's cover and
the smallest denominator among these 26 pairs is 0.0532, so relative values for
the lowest-cover transects are unstable and no test uses that column.

Hard-coral cover is **not** reconstructed from the quadrat label columns. That
is a direct consequence of §8.8: `seaviewsurvey_reefcover_indianocean.csv` omits
the `MASE_MEA_L` (*Lobophyllia*, Hard Coral) column and under-reports hard coral
for 2 513 of 137 698 Indian Ocean quadrats. The survey-level values do not show
that shortfall. The analysis script therefore never opens the quadrat tables at
all, which is pinned by a test.

**The response is an image-derived estimate.** 98.33 % of Indian Ocean cover
values are VGG-D 16 CNN output. It is not measured cover and it is not
biological ground truth — a well-validated estimate is still an estimate, and
the 97 % published classifier accuracy does not convert one into the other. No
`reef_health`, `restoration_suitability`, bleached/not-bleached or
healthy/unhealthy class is derived from it.

### 8a.3 The survey dates — verified, not remembered

Re-derived from `seaviewsurvey_surveys.csv` rather than carried over from
earlier prose:

| | Value |
|---|---|
| Repeat transects | 26 |
| Earliest first survey | **2015-03-29** |
| Latest second survey | **2017-04-01** |
| Interval | 703–722 days (median 719) |
| Epochs | Mar–Apr 2015, then Mar–Apr 2017 |

These dates, and nothing else, determined the CRW acquisition window.

### 8a.4 The CRW historical extension — a separate acquisition

The CRW files already in the repository cover **2018-01-01 → 2024-12-31** for
four **Indian** reef systems. That window does not reach the 2015–2017 surveys
at all, so it was neither joined to them nor substituted for. Climatology was
not substituted either. A separate historical subset was acquired instead.

| | Existing acquisition | This acquisition |
|---|---|---|
| `dataset_id` | `noaa_crw_5km_v3_1` | `noaa_crw_5km_v3_1_maldives_seaview` |
| Purpose | India-region environmental context | Maldives temporal-analysis support |
| Geography | Lakshadweep, Gulf of Mannar, Gulf of Kutch, Andaman and Nicobar | One Maldives window |
| Window | 2018-01-01 → 2024-12-31 | 2015-03-29 → 2017-04-01 |
| Variables | SST, SST anomaly, HotSpot, DHW | HotSpot, DHW |
| Files | 16 | 2 |

The two manifests are separate files and the India one is **unmodified**. The
scientific product, provider, DOI (`10.25921/6jgr-pt28`) and licence
determination are identical; only the acquisition scope differs.

**Availability was verified before any download**, from the NOAA PIFSC
OceanWatch ERDDAP dataset metadata already used by this project:
`CRW_hs_v1_0` declares `time_coverage_start` 1985-01-01 and `CRW_dhw_v1_0`
declares 1985-03-25, both covering 2015–2017.

**Source lineage is unchanged and is not restated more favourably here.** The
requested window sits inside the NOAA reprocessed Geo-Polar Blended era and
crosses the documented October 1–29, 2016 merge into the near-real-time
analysis. It starts long after the project's conservative `2002-12-01` policy
floor, which remains a **project policy** and not a NOAA licence boundary. As in
§7.5.2, **no part of this acquisition may be described as "OSTIA-free"** — NOAA
states the Geo-Polar Blended product switched to OSTIA for bias correction in
2016, which overlaps this window. `scripts/fetch_noaa_crw_maldives.py` imports
the same guard function rather than restating the rule.

Only HotSpot and DHW were requested. SST and SST anomaly are omitted as
unnecessary for the two pre-specified exposure metrics; adding a second,
differently-baselined anomaly would have invited reporting whichever predictor
correlated best.

**What was acquired:**

| Variable | Grid | Days | Size | SHA-256 (first 16) |
|---|---|---|---|---|
| `hotspot` | 43 × 22 cells @ 0.05° | 735 | 2 796 348 B | `8a3eef957ff67d3b` |
| `degree_heating_week` | 43 × 22 cells @ 0.05° | 735 | 2 796 320 B | `9fd8e4bff7431c1c` |

Delivered bbox 2.525–4.625 N, 72.625–73.675 E; zero missing dates; **0.0 % NaN**,
so the whole window is valid ocean and no land mask is in play. Full hashes and
provenance:
`data/external/metadata/noaa_crw_5km_v3_1_maldives_seaview.manifest.json`.

The bounding box was **derived, not chosen**: the minimal box containing every
transect endpoint from both visits, buffered by 0.10° (two native cells) and
snapped outward to the grid. The buffer exists so the nearest-valid-cell search
cannot run off the subset edge. The four Indian windows were deliberately not
reused. Raw files live under the git-ignored
`data/external/raw/noaa_crw_5km_v3_1/maldives_seaview/` and the existing
2018–2024 files were not overwritten.

### 8a.5 Spatial matching

Each transect gets one representative coordinate — the mean of all four
endpoints across both visits — so a single CRW cell serves both visits. The
furthest endpoint from that mean is 1.40 km across all 26 transects, well inside
one cell.

The rule is: **nearest valid CRW ocean grid-cell centre** by great-circle
distance, where "valid" means the cell carries at least one finite value in
**both** products. No interpolation, so no value is ever synthesised across a
coast; and no silent walk to a farther cell, because the distance is recorded on
every row. The maximum permitted match distance is **5 km**, one native cell
width; a match beyond it would be reported unmatched rather than used.

In practice the threshold never bound. All 26 transects matched, distances
0.56–3.23 km (median 1.91), zero displaced by masking and zero unmatched.

### 8a.6 Temporal exposure

Exposure is defined **independently of the response**, before it is looked at:
the closed interval from each pair's own first survey date to its own second
survey date, on daily CRW records. No post-hoc event window was selected to
bracket the observed decline, and no alternative window was tried.

**No CRW observation dated after a pair's second survey enters that pair's
exposure**, so no future thermal information leaks backwards. Survey dates are
day-level integers and CRW is daily, so both sides align at day granularity; no
survey time of day is published, and a survey is treated as occupying its whole
date.

One caveat recorded rather than corrected: DHW is a 12-week backward
accumulation, so a DHW value dated shortly after a first survey partly
accumulates heat from before it. Over 703–722-day intervals that touches only
the first ~84 days and cannot import heat from after the second survey.

Two pre-specified metrics, and no others:

| Metric | Distribution across 26 transects |
|---|---|
| `max_dhw` (°C-weeks) | 5.69 – 7.64, median 6.28, SD 0.51 |
| `max_hotspot` (°C) | 1.49 – 1.67, median 1.58, SD 0.06 |

Zero missing days for either product on any pair. Peak DHW falls between
2016-05-08 and 2016-05-16 for every transect; peak HotSpot between 2016-04-02
and 2016-05-06 — the 2016 event, squarely inside the survey interval.

### 8a.7 Association analysis and results

Spearman rank correlation against `delta_hard_coral`, two-sided, n = 26. No
multivariable model, no train/test split, no cross-validation: this is an
observational association analysis, not an ML benchmark.

**First, the paired change on its own** (before any exposure was loaded):

| | Value |
|---|---|
| Pre `pr_hard_coral` | mean 0.181, median 0.155, range 0.053–0.531 |
| Post `pr_hard_coral` | mean 0.130, median 0.113, range 0.018–0.243 |
| Change | mean −0.051, median −0.023, SD 0.083, range −0.344 to +0.064 |
| Direction | **22 declining, 4 increasing, 0 unchanged** |
| Wilcoxon signed-rank | W = 44.0, **p = 0.00041** |
| Effect size (rank-biserial) | −0.749 |
| Median change, bootstrap 95 % CI | −0.061 to −0.011 |

The paired-difference distribution was strongly non-normal (Shapiro-Wilk
p = 0.00051). Wilcoxon signed-rank is therefore reported as the **primary
rank-based paired analysis**, while the paired t-test (t = −3.16, p = 0.0041) is
retained as a **complementary sensitivity analysis**. That is a reporting
choice, not the claim that a significant Shapiro-Wilk result makes the
signed-rank test the valid one: Wilcoxon carries its own assumptions — symmetry
of the differences about their median — which a normality test does not
establish. Both tests are reported, and both point the same way. No transect was
removed as an outlier.

**Then the association with thermal exposure:**

| Predictor | ρ | p | Bootstrap 95 % CI | Holm-adjusted p |
|---|---|---|---|---|
| `max_dhw` | −0.118 | 0.566 | −0.515 to +0.326 | 1.000 |
| `max_hotspot` | −0.076 | 0.710 | −0.443 to +0.325 | 1.000 |

Both point estimates are negative — the direction the hypothesis predicts — but
both intervals comfortably span zero. Bootstrap intervals use 10 000 resamples
at a fixed seed (20260830). Raw p-values are the values of record; the Holm
column covers the two pre-specified tests only, and no third test was run and
discarded.

The two predictors are correlated with each other (ρ = 0.478, p = 0.014), which
is expected — DHW is by construction the accumulation of HotSpot values at or
above 1 °C — so these are close to one result reported twice, and Holm is
correspondingly conservative.

**Leave-one-pair-out sensitivity.** For `max_dhw`, ρ ranges −0.216 to −0.038
across the 26 refits and the sign never flips. For `max_hotspot`, ρ ranges
−0.139 to +0.009, so dropping a single transect (37015) is enough to flip its
sign — that predictor's near-zero estimate is not stable, and the fact is
reported rather than tidied away.

### 8a.8 Uncertainty, and how to read the null

**The dominant limitation is restriction of range, not sample size alone.** All
26 transects sit inside one Maldivian box roughly 0.9° across and share the same
2016 thermal event. `max_hotspot` spans just 0.18 °C — 18 storage quanta, with
11 tied ranks among 26 transects — so a rank test has very little to rank.
`max_dhw` spans 1.95 °C-weeks with no ties, which is better but still narrow.

**The 26 transects are not 26 independent climatic replicates.** They are
spatially clustered inside that same small box and share the same regional 2016
heat-stress event, so the exposure values attached to them are not independent
draws. Consequently the correlation p-values and bootstrap intervals above are
**interpreted descriptively** and must not be read as inference from 26
independent replicates — the effective number of independent climatic
observations is closer to one event than to 26. No spatial regression, clustered
bootstrap, permutation test or mixed model was fitted to correct for this: the
dependence is disclosed rather than adjusted for, and the quoted intervals
should be treated as descriptive summaries of these 26 rows rather than as
calibrated frequentist coverage.

So the honest reading is: **this design does not resolve an exposure-response
gradient in either direction.** That is *not detected here*, which is a
different claim from *not present*. A null across a narrow predictor range is
evidence about this design, not evidence that thermal exposure and coral-cover
change are unrelated in general. Widening the exposure contrast — more sites,
more thermal regimes, or an event with more spatial variation — is what would
make the question answerable, and the present data cannot substitute for it.

### 8a.9 What this does and does not establish

**Establishes.** Image-derived hard-coral cover on these 26 Maldivian transects
was substantially lower in Mar–Apr 2017 than in Mar–Apr 2015, and the paired
decline is large relative to its uncertainty. All 26 transects experienced
substantial thermal exposure during the interval, peaking in April–May 2016.

**Does not establish. NON-CAUSAL.** This is an association between a
thermal-exposure predictor and a change in an estimate. It does not show that
thermal stress caused mortality, and nothing here proves that bleaching caused
the decline. There is no control, no bleaching observation, and no exclusion of
storm damage, disease, predation, transect re-navigation or classifier
variation. The allowed vocabulary is *associated with*, *correlated with*,
*temporal contrast*, *thermal exposure*, *image-derived coral-cover change*,
*consistent with* / *not consistent with*.

**Does not touch the models.** No `reef_health` label, no
`restoration_suitability` label, no threshold applied to any CRW value to make
one. The registered **synthetic champion** models, `artifacts/mlruns.db` and
`data/raw/observations.csv` are not read, modified or retrained by this
analysis, and their hashes are unchanged by it. No DVC stage was added and the
DAG is unchanged.

**Is not Indian evidence.** `MALDIVES_NOT_INDIA`. The nearest Indian reef
system, Lakshadweep, is roughly 400 km from the nearest transect. This analysis
does not validate any model for Indian reefs.

### 8a.10 Artifacts

| Path | Contents |
|---|---|
| `scripts/fetch_noaa_crw_maldives.py` | Historical CRW acquisition, window derived from the surveys |
| `scripts/analyze_seaview_maldives_pairs.py` | Pair construction, matching, exposure, association |
| `reports/external/seaview_maldives_pairs.csv` | 26 rows, one per transect, with full match provenance |
| `reports/external/seaview_maldives_pairs_summary.json` | Selection checks and paired-change analysis |
| `reports/external/seaview_maldives_crw_association.json` | Matching, exposure, association, sensitivity |
| `reports/figures/seaview_maldives_*.png` | Optional diagnostics (`--figures`; matplotlib not required) |

Daily CRW values are not expanded into a tracked CSV; the pair-level table
carries the CRW cell coordinates and file hashes needed to recompute them.

---

## 9. Reproducing the acquisitions

```bash
python scripts/fetch_gebco_2026.py --dry-run        # print the plan, fetch nothing
python scripts/fetch_gebco_2026.py                  # fetch, validate, write manifest
python scripts/fetch_gebco_2026.py --validate-only  # re-validate local files

python scripts/fetch_noaa_crw.py --dry-run          # print the plan, fetch nothing
python scripts/fetch_noaa_crw.py --cross-check      # fetch, validate, compare servers
python scripts/fetch_noaa_crw.py --validate-only    # re-validate local files

python scripts/fetch_seaview.py --dry-run           # print the plan, fetch nothing
python scripts/fetch_seaview.py                     # fetch tables, extract, write manifest
python scripts/fetch_seaview.py --validate-only     # re-inspect local tables

# §8a — the Maldives historical CRW extension and the paired analysis.
# The fetch script derives its own window from the surveys, so fetch_seaview.py
# must have run first; the analysis reads only local files and never networks.
python scripts/fetch_noaa_crw_maldives.py --dry-run
python scripts/fetch_noaa_crw_maldives.py
python scripts/analyze_seaview_maldives_pairs.py
python scripts/analyze_seaview_maldives_pairs.py --figures   # optional, needs matplotlib
```

GEBCO is requested from the **THREDDS NetCDF Subset Service** hosted by CEDA on
behalf of BODC/GEBCO. CRW is requested from **ERDDAP griddap** on NOAA servers.
Both return NetCDF-3 classic — readable with `scipy.io.netcdf_file`, so **no new
runtime dependency was introduced by either product**. Neither global archive is
ever downloaded.

---

## 10. What has NOT been done

Each acquisition was done in isolation. **Amended 2026-08-30:** §8a joins
Seaview survey records to the Maldives CRW extension, so the "no join" item
below is now scoped rather than absolute. Everything else stands.

Specifically **not** done:

- No change to `generate_data.py`, `preprocess.py`, `build_features.py`,
  `get_feature_columns()`, or the Pandera schema
- No external rows appended to `observations.csv`
- **No join involving GEBCO**, and no join between the 2018–2024 India CRW
  windows and anything else. The one join that exists is Seaview ↔ the Maldives
  CRW extension, for the observational analysis in §8a and nothing else
- No site table, no training table, no new CSVs beyond the 26-row §8a pair table
- No labels created of any kind
- No model trained, registered, promoted, or evaluated
- No DVC stage added; the DAG is unchanged
- No MLflow interaction
- No API or dashboard change

---

## 11. Other sources — status

From the acquisition plan, with one **correction**.

| Source | Status |
|---|---|
| **GEBCO_2026** | **ACQUIRED** — public domain, verified |
| **NOAA Coral Reef Watch 5 km v3.1** | **ACQUIRED** — public domain, verified (see §7) |
| **Seaview Survey (tabular)** | **ACQUIRED** — CC BY 3.0 Unported, verified (see §8.10). Maldives + Chagos only — **not India** |
| NCSCM / CReON (India) | Not acquired — programme use acceptable for academic work, but **no tabular/database export of the underlying observations is available**. Remains an authoritative Indian reference source, not an ML dataset |
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
