"""
src/external — real (non-synthetic) external dataset layer.

This package is deliberately isolated from ``src/data`` (the synthetic
prototype pipeline).  Nothing here reads, writes, or joins
``data/raw/observations.csv``.

Rationale
---------
The dataset audit of 2026-08-19
(``docs/audits/dataset_scientific_audit_2026-08-19.md``) established that the
synthetic 15,000-row benchmark must stay frozen and structurally disconnected
from any real scientific track, and that real observations must never be
silently combined with synthetic rows.  Keeping the external-data code in its
own package makes that separation visible in the import graph rather than
relying on convention.

Contents
--------
``provenance``
    Provenance record schema for externally acquired datasets, plus the
    licence gate that external data must pass before it may be used inside the
    repository.
"""

from __future__ import annotations

from src.external.provenance import (
    LicenceGateError,
    ProvenanceError,
    SourceRecord,
    SubsetRecord,
    assert_publishable,
    load_manifest,
    load_source,
    validate_manifest,
    validate_source,
    validate_subset,
)

__all__ = [
    "LicenceGateError",
    "ProvenanceError",
    "SourceRecord",
    "SubsetRecord",
    "assert_publishable",
    "load_manifest",
    "load_source",
    "validate_manifest",
    "validate_source",
    "validate_subset",
]
