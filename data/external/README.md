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

Scientific role, limitations and acquisition windows: [`docs/external_data.md`](../../docs/external_data.md).

## Reproducing

```bash
python scripts/fetch_gebco_2026.py --dry-run
python scripts/fetch_gebco_2026.py
python scripts/fetch_gebco_2026.py --validate-only
```
