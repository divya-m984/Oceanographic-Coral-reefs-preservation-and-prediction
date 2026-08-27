# `data/external/` — real (non-synthetic) external datasets

This directory holds **real** external scientific data and its provenance. It is
structurally separate from the synthetic prototype dataset in `data/raw/`.

```
metadata/    tracked in Git — provenance manifests, one per acquired product
raw/         git-ignored    — the downloaded files themselves
```

## Rules

1. **Nothing here is joined to `data/raw/observations.csv`.** The synthetic
   dataset is a frozen benchmark; real and synthetic rows are never silently
   combined.
2. **No labels.** Real environmental or physical measurements are covariates.
   Thresholding them into a reef-condition class would recreate the
   label-construction leakage documented in the 2026-08-19 dataset audit.
3. **Provenance first.** Every product carries a manifest recording its source,
   version, DOI, licence, verification state, and per-file checksums.
4. **The licence gate.** A product may not be used until `licence_verified` is
   `True` and `redistribution_allowed` is explicitly `True` or `False`. See
   `src/external/provenance.py`.

`raw/` is git-ignored so this layer can later move under DVC without rewriting
history. That is a **storage** decision, not a licensing one — redistribution
terms are recorded per product in its manifest.

## Currently acquired

| Product | Version | Licence | Redistribution | Manifest |
|---|---|---|---|---|
| GEBCO Grid | `GEBCO_2026` | Public domain (verified) | Allowed | `metadata/gebco_2026.manifest.json` |
| NOAA Coral Reef Watch 5 km | `v3.1` | US Government public domain, attribution requested (verified) | Allowed | `metadata/noaa_crw_5km_v3_1.manifest.json` |
| Seaview Survey (tabular) | 2019 release | CC BY 3.0 Unported (verified; publisher's label normalized) | Allowed | `metadata/seaview_survey.manifest.json` |

The first two cover the same four Indian reef systems — Lakshadweep, Gulf of
Mannar, Gulf of Kutch, Andaman and Nicobar Islands — over identical acquisition
windows. **Seaview covers none of them** (see below). **No two of the three are
joined to each other.**

| | GEBCO_2026 | NOAA CRW 5 km v3.1 | Seaview Survey |
|---|---|---|---|
| Quantity | Bathymetry / terrain | Thermal: SST, SST anomaly, HotSpot, DHW | **Biological**: image-derived benthic-cover estimates |
| Resolution | 15″ (~450 m) | 0.05° (~5 km) | ~1 m² photo-quadrat |
| Time | Static compilation | Daily, 2018-01-01 → 2024-12-31 | Survey dates, 2015-02-12 → 2017-04-01 (Indian Ocean) |
| Geography | 4 Indian reef systems | 4 Indian reef systems | **Maldives + Chagos** |
| Files | 4 | 16 (4 variables × 4 regions) | 18 CSV tables |

### Seaview: Indian Ocean is not India

Seaview is the project's first **real biological** source, and the one most
likely to be misdescribed. Its Central Indian Ocean component is **92 surveys**:
**63 in the Maldives** and **29 in the Chagos Archipelago**. No survey is in
India — no row of `seaviewsurvey_surveys.csv` has `country == 'IND'`.

Beware the naming collision: the `ocean` column uses `IND` for the **Indian
Ocean basin**, which is also the ISO code for **India**, and the human
annotation files are named `annotations_IND_MDV.csv` / `annotations_IND_CHA.csv`.

Zero surveys in Lakshadweep, the Gulf of Mannar, the Gulf of Kutch, or the
Andaman and Nicobar Islands; the nearest transect is ~386 km from Lakshadweep.
**This dataset does not validate any model for Indian reefs.** The manifest
records this as `geographic_transfer_status = "INDIAN_OCEAN_NOT_INDIA"`.

### Seaview: cover is estimated from images, not measured in the water

Three distinct things travel together in this product, and they must not be
collapsed into "real observed coral cover":

- **A — field survey imagery and coordinates:** real photographs of real reef,
  with real dates and transect coordinates. A genuine field record.
- **B — human image annotations:** experts scoring points on 1.67 % of Indian
  Ocean quadrats. A human judgement *about an image*.
- **C — ML-classified benthic cover:** the cover columns themselves, 98.33 %
  VGG-D 16 CNN output. **Model output.**

The columns a model would consume are layer C, so call them **image-derived
benthic-cover estimates** or **ML-estimated benthic cover** — never *directly
observed*, *field-measured*, *measured coral cover*, or *biological ground
truth*. The published 97 % classifier validation is real and is preserved, but
a well-validated estimate is still an estimate.

The paired revisits are subject to the same care: 26 Maldivian transects were
surveyed twice, 703–722 days apart. That is a **paired pre/post survey change
around the 2016 mass-bleaching period** — a real temporal contrast, not a
measured "bleaching response", and not on its own evidence that bleaching
caused the change. CRW HotSpot/DHW exposure should eventually be linked by site
and date; that has not been done, and even then the design stays observational.

And the targets: `hard_coral_cover != reef_health`, `benthic class !=
restoration_suitability`. See [`docs/external_data.md`](../../docs/external_data.md) §8.

### CRW licence caveat

**Licence basis.** `licence_verified` and `redistribution_allowed` rest on NOAA
CRW's published terms (posted data are freely available, website content is
public domain and may be distributed freely) plus the licence metadata delivered
inside the files. They are **not inferred from** the acquisition window's
position in CoralTemp's source lineage.

**Source lineage**, tracked separately. CoralTemp is assembled from more than one
analysis: Met Office OSTIA reanalysis contributes directly from January 1985 to
November 2002, then NOAA Geo-Polar Blended reprocessed to October 2016, then
operational Geo-Polar — with 29-day linear merges over November 1–29, 2002 and
October 1–29, 2016. Only the OSTIA reanalysis is restrictively licensed
(academic use only, reproduction licence application required).

OSTIA does **not** drop out later: NOAA states the Geo-Polar Blended product
"switched to using OSTIA as the bias correction" in 2016, which covers the
acquired window. **No part of this dataset is "OSTIA-free."**

`scripts/fetch_noaa_crw.py` refuses to request anything before
`FIRST_POST_OSTIA_BLEND_REQUEST_DATE = "2002-12-01"`. That is a **conservative
project policy** — stay out of the direct-OSTIA period and the 2002 merge window
— not a NOAA licence boundary and not a purity claim. The acquired window starts
2018-01-01, well past it.

### CRW products are not labels

`DHW ≠ bleaching_percentage`. `HotSpot ≠ bleaching_percentage`. These are
thermal-stress **predictors**; thresholding them into a reef-condition class
would recreate the label-construction leakage described in rule 2 above.

Scientific role, limitations and acquisition windows: [`docs/external_data.md`](../../docs/external_data.md).

## Reproducing

```bash
python scripts/fetch_gebco_2026.py --dry-run
python scripts/fetch_gebco_2026.py
python scripts/fetch_gebco_2026.py --validate-only

python scripts/fetch_noaa_crw.py --dry-run
python scripts/fetch_noaa_crw.py --cross-check
python scripts/fetch_noaa_crw.py --validate-only

python scripts/fetch_seaview.py --dry-run
python scripts/fetch_seaview.py
python scripts/fetch_seaview.py --validate-only
```
