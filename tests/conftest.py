"""
Shared pytest fixtures for the CoralSense test suite.

Isolation policy
----------------
No test may write to, delete, or open as a writable backend any canonical
project artifact:

    artifacts/mlruns.db
    reports/drift_summary.json
    data/raw/, data/reference/, data/production/
    models/*.joblib, data/processed/*.joblib

Tests that need the *contents* of a canonical artifact work on a private copy
inside ``tmp_path``.  Tests that need to render a page which reads a canonical
artifact redirect the relevant configuration at ``tmp_path``.

The fixtures below provide both mechanisms.  Nothing here restores canonical
files after the fact — the point is that they are never touched.
"""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Canonical artifacts that must never be mutated by the test suite.
CANONICAL_MLFLOW_DB = PROJECT_ROOT / "artifacts" / "mlruns.db"
CANONICAL_DRIFT_SUMMARY = PROJECT_ROOT / "reports" / "drift_summary.json"


@pytest.fixture
def canonical_registry_uri(tmp_path: Path) -> str:
    """
    An MLflow tracking URI pointing at a private byte-copy of the canonical
    registry database.

    Opening a SQLite file through MLflow/SQLAlchemy is not guaranteed to be
    byte-preserving even for pure reads: the engine may run schema inspection
    or migration bookkeeping that rewrites pages, so the file checksum can
    change without any logical change.  Tests therefore read a copy.

    The copy is bit-identical at fixture setup, so any assertion about the
    registry's logical contents (aliases, version counts) remains valid.

    Returns an absolute ``sqlite:///`` URI.  ``tmp_path`` is absolute, so the
    f-string yields the correct four-slash absolute form.
    """
    if not CANONICAL_MLFLOW_DB.is_file():
        pytest.skip(f"canonical registry not present: {CANONICAL_MLFLOW_DB}")

    copy = tmp_path / "mlruns.db"
    shutil.copy2(CANONICAL_MLFLOW_DB, copy)
    return f"sqlite:///{copy}"


@pytest.fixture
def isolated_reports_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    Redirect ``Config.paths.reports_dir`` at an empty directory under
    ``tmp_path`` and return it.

    ``Paths`` is a frozen dataclass, so the whole object is replaced via
    ``dataclasses.replace`` and assigned to the (mutable) ``Config.paths``
    field of the cached singleton.  ``monkeypatch`` restores the original
    ``Paths`` at teardown.

    Any code reading ``get_config().drift_summary_path`` or
    ``get_config().paths.reports_dir`` — including dashboard page 8 — will see
    the temporary directory, so the real ``reports/`` tree is never read,
    written, or deleted.
    """
    from src.config import get_config

    cfg = get_config()
    reports = tmp_path / "reports"
    reports.mkdir()
    monkeypatch.setattr(cfg, "paths", dataclasses.replace(cfg.paths, reports_dir=reports))
    return reports
