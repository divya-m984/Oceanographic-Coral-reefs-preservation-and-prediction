"""
Regression tests for canonical-artifact isolation.

Two historical defects motivated this file:

1. ``tests/test_demo.py::TestRegistryInvariants`` pointed MLflow at the real
   ``artifacts/mlruns.db``.  The reads were logically read-only, but opening a
   SQLite file through MLflow/SQLAlchemy can rewrite internal pages, so the
   file's SHA-256 changed on every full-suite run.

2. ``tests/test_dashboard.py::TestAppSmoke::test_drift_page_renders_with_summary``
   overwrote the real, DVC-tracked ``reports/drift_summary.json`` with fixture
   data and then deleted it in a ``finally`` block, destroying the artifact
   instead of restoring it.

These tests fail if either pattern is reintroduced.
"""

from __future__ import annotations

import ast
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import (
    CANONICAL_DRIFT_SUMMARY,
    CANONICAL_MLFLOW_DB,
    PROJECT_ROOT,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Fixture behaviour
# ─────────────────────────────────────────────────────────────────────────────


class TestIsolationFixtures:
    def test_registry_uri_points_into_tmp_not_repo(self, canonical_registry_uri, tmp_path):
        """The MLflow URI must address a copy under tmp_path, never the repo."""
        assert canonical_registry_uri.startswith("sqlite:///")
        db = Path(canonical_registry_uri.removeprefix("sqlite:///"))
        assert db.is_absolute()
        assert db.is_relative_to(tmp_path)
        assert not db.is_relative_to(PROJECT_ROOT)

    def test_registry_copy_matches_canonical_bytes(self, canonical_registry_uri):
        """The copy must be bit-identical, so assertions about it stay valid."""
        db = Path(canonical_registry_uri.removeprefix("sqlite:///"))
        assert _sha256(db) == _sha256(CANONICAL_MLFLOW_DB)

    def test_isolated_reports_dir_redirects_config(self, isolated_reports_dir, tmp_path):
        """get_config() must resolve reports into tmp_path while the fixture is active."""
        from src.config import get_config

        cfg = get_config()
        assert cfg.paths.reports_dir == isolated_reports_dir
        assert isolated_reports_dir.is_relative_to(tmp_path)
        assert cfg.drift_summary_path.is_relative_to(tmp_path)
        assert not cfg.drift_summary_path.is_relative_to(PROJECT_ROOT / "reports")

    def test_isolated_reports_dir_starts_empty(self, isolated_reports_dir):
        assert list(isolated_reports_dir.iterdir()) == []

    def test_config_reports_dir_restored_after_fixture(self):
        """Outside the fixture, config must point back at the real reports/."""
        from src.config import get_config

        assert get_config().paths.reports_dir == PROJECT_ROOT / "reports"


# ─────────────────────────────────────────────────────────────────────────────
# Static guards — fail if the destructive patterns come back
# ─────────────────────────────────────────────────────────────────────────────


class TestNoCanonicalMutationPatterns:
    def test_no_test_sets_mlflow_uri_to_canonical_db(self):
        """No test may bind MLflow to the canonical artifacts/mlruns.db."""
        offenders = []
        for path in sorted((PROJECT_ROOT / "tests").glob("test_*.py")):
            if path.name == Path(__file__).name:
                continue  # this file quotes the patterns it forbids
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                if "set_tracking_uri" not in line:
                    continue
                # Canonical == an artifacts/mlruns.db under the project root.
                if "artifacts" in line and "mlruns.db" in line and "tmp" not in line:
                    offenders.append(f"{path.name}:{lineno}: {line.strip()}")
        assert not offenders, "tests bind MLflow to the canonical registry:\n" + "\n".join(
            offenders
        )

    def test_no_test_unlinks_canonical_drift_summary(self):
        """No test may delete reports/drift_summary.json."""
        offenders = []
        for path in sorted((PROJECT_ROOT / "tests").glob("test_*.py")):
            if path.name == Path(__file__).name:
                continue  # this file quotes the patterns it forbids
            lines = path.read_text().splitlines()
            for lineno, line in enumerate(lines, 1):
                if "unlink" not in line:
                    continue
                # Look at a small window for the path being unlinked.
                window = "\n".join(lines[max(0, lineno - 8) : lineno])
                if "drift_summary" in window and "tmp_path" not in window:
                    offenders.append(f"{path.name}:{lineno}: {line.strip()}")
        assert not offenders, "tests delete the canonical drift summary:\n" + "\n".join(offenders)

    def test_dashboard_drift_tests_use_isolated_reports_fixture(self):
        """Both page-8 smoke tests must request the isolation fixture."""
        tree = ast.parse((PROJECT_ROOT / "tests" / "test_dashboard.py").read_text())
        wanted = {
            "test_drift_page_renders_with_summary",
            "test_drift_page_shows_instructions_when_no_summary",
        }
        seen = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in wanted:
                seen[node.name] = {a.arg for a in node.args.args}
        assert wanted <= set(seen), f"missing page-8 tests: {wanted - set(seen)}"
        for name, args in seen.items():
            assert "isolated_reports_dir" in args, f"{name} does not use isolated_reports_dir"

    def test_registry_invariant_tests_use_registry_copy_fixture(self):
        """The registry-invariant tests must run against the copied database."""
        tree = ast.parse((PROJECT_ROOT / "tests" / "test_demo.py").read_text())
        checked = 0
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ClassDef) and node.name == "TestRegistryInvariants"):
                continue
            for fn in node.body:
                if not (isinstance(fn, ast.FunctionDef) and "champion" in fn.name):
                    continue
                assert "canonical_registry_uri" in {a.arg for a in fn.args.args}, (
                    f"{fn.name} does not use canonical_registry_uri"
                )
                checked += 1
        assert checked >= 2, "expected the champion-alias invariant tests to exist"


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end proof: running the previously destructive tests changes nothing
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestCanonicalArtifactsSurviveFocusedTests:
    """Run the formerly destructive tests in a subprocess and diff checksums."""

    _NODES = [
        "tests/test_demo.py::TestRegistryInvariants::test_champion_remains_v1_health",
        "tests/test_demo.py::TestRegistryInvariants::test_champion_remains_v1_restoration",
        "tests/test_demo.py::TestRegistryInvariants::test_health_has_4_versions",
        "tests/test_demo.py::TestRegistryInvariants::test_restoration_has_4_versions",
        "tests/test_dashboard.py::TestAppSmoke::test_drift_page_renders_with_summary",
        "tests/test_dashboard.py::TestAppSmoke::test_drift_page_shows_instructions_when_no_summary",
    ]

    def test_focused_tests_leave_canonical_artifacts_untouched(self):
        if not CANONICAL_MLFLOW_DB.is_file():
            pytest.skip("canonical registry not present")
        if not CANONICAL_DRIFT_SUMMARY.is_file():
            pytest.skip("canonical drift summary not present")

        before = {
            CANONICAL_MLFLOW_DB: _sha256(CANONICAL_MLFLOW_DB),
            CANONICAL_DRIFT_SUMMARY: _sha256(CANONICAL_DRIFT_SUMMARY),
        }

        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *self._NODES],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"focused tests failed:\n{proc.stdout}\n{proc.stderr}"

        for path, digest in before.items():
            assert path.is_file(), f"{path} was deleted by the focused tests"
            assert _sha256(path) == digest, f"{path} was modified by the focused tests"


# ─────────────────────────────────────────────────────────────────────────────
# Regression: the focused subprocess must not depend on the caller's cwd
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestFocusedSubprocessIsCwdIndependent:
    """
    ``AppTest.from_file`` resolves a relative script path against the calling
    test file, so ``"src/dashboard/app.py"`` became ``tests/src/dashboard/app.py``
    in CI.  The dashboard smoke tests now pass absolute, repository-root-derived
    paths; this proves it by running one from a directory outside the repository.
    """

    _NODE = "tests/test_dashboard.py::TestAppSmoke::test_drift_page_renders_with_summary"

    def test_dashboard_smoke_passes_from_a_foreign_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)  # the parent's cwd is outside the repository
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                str(PROJECT_ROOT / self._NODE),  # absolute node id
            ],
            cwd=tmp_path,  # …and so is the child's
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, (
            f"dashboard smoke test failed from cwd={tmp_path}:\n{proc.stdout}\n{proc.stderr}"
        )
        assert "tests/src/dashboard" not in proc.stdout
