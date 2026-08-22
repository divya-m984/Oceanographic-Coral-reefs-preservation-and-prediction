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

Both cover the same four Indian reef systems — Lakshadweep, Gulf of Mannar, Gulf
of Kutch, Andaman and Nicobar Islands — over identical acquisition windows.
**They are not joined to each other.**

| | GEBCO_2026 | NOAA CRW 5 km v3.1 |
|---|---|---|
| Quantity | Bathymetry / terrain | Thermal: SST, SST anomaly, HotSpot, DHW |
| Resolution | 15″ (~450 m) | 0.05° (~5 km) |
| Time | Static compilation | Daily, 2018-01-01 → 2024-12-31 |
| Files | 4 | 16 (4 variables × 4 regions) |

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
```
