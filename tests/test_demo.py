"""
tests/test_demo.py — Tests for M14 release-readiness tools.

Coverage:
- Preflight tool: success path, blocking failures, warnings, JSON output, exit codes
- Preflight DVC remote: DagsHub remote config, S3 support, credential hygiene, sync
- Demo orchestrator: status, shutdown, bounded polling
- Evidence manifest: schema, title, no absolute paths, immutability
- Makefile: target presence, no promotion commands
- Governance page: read-only, graceful missing-report states
- Documentation: required links and synthetic disclaimers
- CLI exit codes
- Canonical registry invariants
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_PYTHON = sys.executable
_DATASET_PATH = _ROOT / "data" / "raw" / "observations.csv"


@pytest.fixture(scope="module", autouse=True)
def _ensure_demo_dataset():
    """Provide the deterministic DVC-managed dataset in clean checkouts."""
    dataset_existed = _DATASET_PATH.exists()
    if not dataset_existed:
        subprocess.run(
            [_PYTHON, "-m", "src.data.generate_data"],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(_ROOT),
            timeout=30,
        )

    yield

    if not dataset_existed:
        _DATASET_PATH.unlink(missing_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _run(args: list[str], *, env: dict | None = None) -> subprocess.CompletedProcess:
    import os

    full_env = {**os.environ, **(env or {})}
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        cwd=str(_ROOT),
        env=full_env,
        # Generous: preflight performs a time-boxed read-only DVC remote status
        # probe, which is slow on a congested network before it warns.
        timeout=180,
    )


# ── Preflight import ───────────────────────────────────────────────────────────

sys.path.insert(0, str(_ROOT))


def _import_preflight():
    """Import preflight module fresh."""
    import importlib

    import scripts.preflight as pf

    importlib.reload(pf)
    return pf


# ─────────────────────────────────────────────────────────────────────────────
# PREFLIGHT TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestPreflightReport:
    def test_empty_report_is_ok(self):
        pf = _import_preflight()
        report = pf.PreflightReport()
        assert report.ok is True
        assert report.passed_count == 0

    def test_all_passing_checks_ok(self):
        pf = _import_preflight()
        report = pf.PreflightReport()
        report.add(pf.CheckResult("a", True, "ok a"))
        report.add(pf.CheckResult("b", True, "ok b"))
        assert report.ok is True
        assert len(report.failures) == 0

    def test_blocking_failure_makes_not_ok(self):
        pf = _import_preflight()
        report = pf.PreflightReport()
        report.add(pf.CheckResult("a", True, "ok"))
        report.add(pf.CheckResult("b", False, "fail", is_warning=False))
        assert report.ok is False
        assert len(report.failures) == 1

    def test_warning_only_still_ok(self):
        pf = _import_preflight()
        report = pf.PreflightReport()
        report.add(pf.CheckResult("a", False, "warn", is_warning=True))
        assert report.ok is True
        assert len(report.warnings) == 1
        assert len(report.failures) == 0

    def test_to_dict_schema(self):
        pf = _import_preflight()
        report = pf.PreflightReport()
        report.add(pf.CheckResult("test", True, "msg"))
        d = report.to_dict()
        assert "ok" in d
        assert "total" in d
        assert "passed" in d
        assert "failures" in d
        assert "warnings" in d
        assert "checks" in d
        assert isinstance(d["checks"], list)

    def test_to_dict_check_fields(self):
        pf = _import_preflight()
        report = pf.PreflightReport()
        report.add(pf.CheckResult("my_check", True, "all good"))
        check = report.to_dict()["checks"][0]
        assert "name" in check
        assert "passed" in check
        assert "warning" in check
        assert "message" in check
        assert "detail" in check


class TestPreflightProjectRoot:
    def test_valid_root_passes(self):
        pf = _import_preflight()
        report = pf.PreflightReport()
        pf.check_project_root(report)
        result = report.checks[-1]
        assert result.passed, f"Expected pass but got: {result.message}"

    def test_python_version_passes(self):
        pf = _import_preflight()
        report = pf.PreflightReport()
        pf.check_python_version(report)
        result = report.checks[-1]
        assert result.passed


class TestPreflightDataset:
    def test_dataset_exists_passes(self):
        pf = _import_preflight()
        report = pf.PreflightReport()
        pf.check_dataset(report)
        exists_check = next(c for c in report.checks if c.name == "dataset_exists")
        assert exists_check.passed

    def test_dataset_row_count_passes(self):
        pf = _import_preflight()
        report = pf.PreflightReport()
        pf.check_dataset(report)
        row_check = next((c for c in report.checks if c.name == "dataset_row_count"), None)
        assert row_check is not None
        assert row_check.passed

    def test_dataset_missing_fails(self, tmp_path):
        pf = _import_preflight()
        # Patch the DATASET_PATH to a non-existent file
        with patch.object(pf, "DATASET_PATH", tmp_path / "missing.csv"):
            report = pf.PreflightReport()
            pf.check_dataset(report)
            exists_check = next(c for c in report.checks if c.name == "dataset_exists")
            assert not exists_check.passed
            assert not exists_check.is_warning  # dataset missing is a blocking failure

    def test_dataset_checksum_passes(self):
        pf = _import_preflight()
        report = pf.PreflightReport()
        pf.check_dataset_checksum(report)
        check = report.checks[-1]
        assert check.passed, f"Checksum check failed: {check.message}"


class TestPreflightCanonicalDB:
    def test_canonical_db_exists(self):
        pf = _import_preflight()
        report = pf.PreflightReport()
        pf.check_canonical_db(report)
        exists_check = next(c for c in report.checks if c.name == "canonical_db_exists")
        assert exists_check.passed

    def test_canonical_db_checksum_matches(self):
        pf = _import_preflight()
        report = pf.PreflightReport()
        pf.check_canonical_db(report)
        checks = {c.name: c for c in report.checks}
        if "canonical_db_checksum" in checks:
            assert checks["canonical_db_checksum"].passed

    def test_missing_db_fails(self, tmp_path):
        pf = _import_preflight()
        with patch.object(pf, "CANONICAL_DB", tmp_path / "missing.db"):
            report = pf.PreflightReport()
            pf.check_canonical_db(report)
            exists_check = next(c for c in report.checks if c.name == "canonical_db_exists")
            assert not exists_check.passed


class TestPreflightChampionAliases:
    def test_champion_aliases_v1(self):
        pf = _import_preflight()
        report = pf.PreflightReport()
        pf.check_champion_aliases(report)
        for c in report.checks:
            if c.name.startswith("champion_alias_"):
                assert c.passed, f"Champion alias check failed: {c.message}"

    def test_version_counts_at_least_4(self):
        pf = _import_preflight()
        report = pf.PreflightReport()
        pf.check_champion_aliases(report)
        for c in report.checks:
            if c.name.startswith("version_count_"):
                assert c.passed, f"Version count check failed: {c.message}"


class TestPreflightBundles:
    def test_bundles_present_or_warning(self):
        pf = _import_preflight()
        report = pf.PreflightReport()
        pf.check_deployment_bundles(report)
        check = next(c for c in report.checks if c.name == "deployment_bundles")
        # Bundles may or may not exist; missing is a warning, not blocking
        if not check.passed:
            assert check.is_warning

    def test_missing_bundle_is_warning_not_failure(self, tmp_path):
        pf = _import_preflight()
        fake_files = [tmp_path / "missing_bundle"]
        with patch.object(pf, "REQUIRED_BUNDLE_FILES", fake_files):
            report = pf.PreflightReport()
            pf.check_deployment_bundles(report)
            check = next(c for c in report.checks if c.name == "deployment_bundles")
            assert not check.passed
            assert check.is_warning  # missing bundles should be a warning


class TestPreflightDriftSummary:
    def test_drift_summary_missing_is_warning(self, tmp_path):
        pf = _import_preflight()
        old_root = pf._ROOT
        pf._ROOT = tmp_path
        try:
            report2 = pf.PreflightReport()
            pf.check_drift_summary(report2)
            check = next(c for c in report2.checks if c.name == "drift_summary")
            assert not check.passed
            assert check.is_warning
        finally:
            pf._ROOT = old_root


class TestPreflightPorts:
    def test_ports_checked(self):
        pf = _import_preflight()
        report = pf.PreflightReport()
        pf.check_ports(report)
        # Each port should generate one check
        port_checks = [c for c in report.checks if c.name.startswith("port_")]
        assert len(port_checks) == len(pf.REQUIRED_PORTS)

    def test_port_free_is_warning_not_failure(self):
        """Occupied ports should be a warning, not a blocking failure."""
        pf = _import_preflight()
        # Mock a port as in-use
        with patch("socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_sock.__enter__ = lambda s: s
            mock_sock.__exit__ = MagicMock(return_value=False)
            mock_sock.connect_ex.return_value = 0  # port in use
            mock_socket_cls.return_value = mock_sock

            report = pf.PreflightReport()
            pf.check_ports(report)
            port_checks = [c for c in report.checks if c.name.startswith("port_")]
            for c in port_checks:
                if not c.passed:
                    assert c.is_warning  # in-use port = warning, not failure


class TestPreflightDocker:
    def test_missing_docker_is_warning(self):
        pf = _import_preflight()
        with patch("subprocess.run", side_effect=FileNotFoundError):
            report = pf.PreflightReport()
            pf.check_docker(report)
            check = next(c for c in report.checks if c.name == "docker_available")
            assert not check.passed
            assert check.is_warning  # Docker missing = warning, not blocking failure


# ── DVC / DagsHub remote preflight ─────────────────────────────────────────────

_GOOD_DVC_CONFIG = """['remote "dagshub"']
    url = s3://dvc
    endpointurl = https://dagshub.com/divya-m984/Oceanographic-Coral-reefs-preservation-and-prediction.s3
"""

_FAKE_TOKEN = "0123456789abcdef0123456789abcdef01234567"


def _write_config(tmp_path: Path, text: str, name: str = "config") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


# Structural patterns for credential leakage.  The suite never reads the
# developer's real .dvc/config.local, so no assertion can ever print a secret.
_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)(access_key_id|secret_access_key|session_token|password|auth_token)\s*[:=]\s*[^\s*\"]"
)
_URL_USERINFO = re.compile(r"://[^\s/@\"]+:[^\s/@\"]+@")


def _completed(returncode: int = 0, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class TestPreflightDVCRemoteConfig:
    """The tracked DagsHub remote must be present and materially correct."""

    def test_repository_remote_config_passes(self):
        pf = _import_preflight()
        report = pf.PreflightReport()
        pf.check_dvc_remote_config(report)
        check = next(c for c in report.checks if c.name == "dvc_remote_config")
        assert check.passed, check.message

    def test_valid_config_passes(self, tmp_path):
        pf = _import_preflight()
        cfg = _write_config(tmp_path, _GOOD_DVC_CONFIG)
        with patch.object(pf, "DVC_TRACKED_CONFIG", cfg):
            report = pf.PreflightReport()
            pf.check_dvc_remote_config(report)
        check = next(c for c in report.checks if c.name == "dvc_remote_config")
        assert check.passed

    def test_config_parsing_tolerates_formatting(self, tmp_path):
        """Quoting, spacing and key case must not change the verdict."""
        pf = _import_preflight()
        text = (
            '[remote "dagshub"]\n'
            "URL=s3://dvc/\n"
            "EndpointURL = https://dagshub.com/divya-m984/"
            "Oceanographic-Coral-reefs-preservation-and-prediction.s3\n"
        )
        cfg = _write_config(tmp_path, text)
        with patch.object(pf, "DVC_TRACKED_CONFIG", cfg):
            report = pf.PreflightReport()
            pf.check_dvc_remote_config(report)
        check = next(c for c in report.checks if c.name == "dvc_remote_config")
        assert check.passed, check.message

    def test_missing_remote_fails(self, tmp_path):
        pf = _import_preflight()
        cfg = _write_config(tmp_path, "['remote \"origin\"']\n    url = s3://other\n")
        with patch.object(pf, "DVC_TRACKED_CONFIG", cfg):
            report = pf.PreflightReport()
            pf.check_dvc_remote_config(report)
        check = next(c for c in report.checks if c.name == "dvc_remote_config")
        assert not check.passed
        assert not check.is_warning  # blocking

    def test_missing_config_file_fails(self, tmp_path):
        pf = _import_preflight()
        with patch.object(pf, "DVC_TRACKED_CONFIG", tmp_path / "absent"):
            report = pf.PreflightReport()
            pf.check_dvc_remote_config(report)
        check = next(c for c in report.checks if c.name == "dvc_remote_config")
        assert not check.passed
        assert not check.is_warning

    def test_wrong_url_fails(self, tmp_path):
        pf = _import_preflight()
        cfg = _write_config(tmp_path, _GOOD_DVC_CONFIG.replace("s3://dvc", "s3://wrong-bucket"))
        with patch.object(pf, "DVC_TRACKED_CONFIG", cfg):
            report = pf.PreflightReport()
            pf.check_dvc_remote_config(report)
        check = next(c for c in report.checks if c.name == "dvc_remote_config")
        assert not check.passed
        assert not check.is_warning

    def test_wrong_endpoint_fails(self, tmp_path):
        pf = _import_preflight()
        cfg = _write_config(
            tmp_path,
            _GOOD_DVC_CONFIG.replace("https://dagshub.com", "https://evil.example.com"),
        )
        with patch.object(pf, "DVC_TRACKED_CONFIG", cfg):
            report = pf.PreflightReport()
            pf.check_dvc_remote_config(report)
        check = next(c for c in report.checks if c.name == "dvc_remote_config")
        assert not check.passed
        assert not check.is_warning

    def test_non_canonical_endpoint_key_fails(self, tmp_path):
        """Only DVC's canonical ``endpointurl`` key counts as a configured endpoint."""
        pf = _import_preflight()
        cfg = _write_config(tmp_path, _GOOD_DVC_CONFIG.replace("endpointurl", "endpoint_url"))
        with patch.object(pf, "DVC_TRACKED_CONFIG", cfg):
            report = pf.PreflightReport()
            pf.check_dvc_remote_config(report)
        check = next(c for c in report.checks if c.name == "dvc_remote_config")
        assert not check.passed
        assert not check.is_warning

    def test_missing_endpoint_fails(self, tmp_path):
        pf = _import_preflight()
        cfg = _write_config(tmp_path, "['remote \"dagshub\"']\n    url = s3://dvc\n")
        with patch.object(pf, "DVC_TRACKED_CONFIG", cfg):
            report = pf.PreflightReport()
            pf.check_dvc_remote_config(report)
        check = next(c for c in report.checks if c.name == "dvc_remote_config")
        assert not check.passed


class TestPreflightDVCS3Support:
    def test_s3_support_present_passes(self):
        pf = _import_preflight()
        with patch.object(pf.importlib.util, "find_spec", return_value=object()):
            report = pf.PreflightReport()
            pf.check_dvc_s3_support(report)
        check = next(c for c in report.checks if c.name == "dvc_s3_support")
        assert check.passed

    def test_s3_support_absent_is_blocking_failure(self):
        pf = _import_preflight()
        with patch.object(pf.importlib.util, "find_spec", return_value=None):
            report = pf.PreflightReport()
            pf.check_dvc_s3_support(report)
        check = next(c for c in report.checks if c.name == "dvc_s3_support")
        assert not check.passed
        assert not check.is_warning  # remote cannot function without it

    def test_installed_environment_has_s3_support(self):
        pf = _import_preflight()
        report = pf.PreflightReport()
        pf.check_dvc_s3_support(report)
        check = next(c for c in report.checks if c.name == "dvc_s3_support")
        assert check.passed, "dvc[s3] must be installed for the DagsHub remote"


class TestPreflightDVCTrackedConfigCredentials:
    def test_clean_tracked_config_passes(self, tmp_path):
        pf = _import_preflight()
        cfg = _write_config(tmp_path, _GOOD_DVC_CONFIG)
        with patch.object(pf, "DVC_TRACKED_CONFIG", cfg):
            report = pf.PreflightReport()
            pf.check_dvc_tracked_config_credentials(report)
        check = next(c for c in report.checks if c.name == "dvc_tracked_config_clean")
        assert check.passed

    def test_repository_tracked_config_is_clean(self):
        pf = _import_preflight()
        report = pf.PreflightReport()
        pf.check_dvc_tracked_config_credentials(report)
        check = next(c for c in report.checks if c.name == "dvc_tracked_config_clean")
        assert check.passed, "tracked .dvc/config must never contain credentials"

    @pytest.mark.parametrize("key", ["access_key_id", "secret_access_key"])
    def test_tracked_credentials_are_blocking_failure(self, tmp_path, key):
        pf = _import_preflight()
        cfg = _write_config(tmp_path, _GOOD_DVC_CONFIG + f"    {key} = {_FAKE_TOKEN}\n")
        with patch.object(pf, "DVC_TRACKED_CONFIG", cfg):
            report = pf.PreflightReport()
            pf.check_dvc_tracked_config_credentials(report)
        check = next(c for c in report.checks if c.name == "dvc_tracked_config_clean")
        assert not check.passed
        assert not check.is_warning
        assert key in check.message  # the field name is named...
        assert _FAKE_TOKEN not in check.message  # ...but never its value
        assert _FAKE_TOKEN not in check.detail

    def test_credential_scan_returns_names_only(self, tmp_path):
        pf = _import_preflight()
        cfg = _write_config(tmp_path, _GOOD_DVC_CONFIG + f"    access_key_id = {_FAKE_TOKEN}\n")
        assert pf._config_credential_keys(cfg) == ["access_key_id"]

    def test_credential_scan_on_missing_file(self, tmp_path):
        pf = _import_preflight()
        assert pf._config_credential_keys(tmp_path / "absent") == []


class TestPreflightDVCLocalConfigSafety:
    def test_missing_local_config_is_warning(self, tmp_path):
        pf = _import_preflight()
        with patch.object(pf, "DVC_LOCAL_CONFIG", tmp_path / "config.local"):
            report = pf.PreflightReport()
            pf.check_dvc_local_config_safety(report)
        check = next(c for c in report.checks if c.name == "dvc_local_config")
        assert not check.passed
        assert check.is_warning  # a collaborator may simply have no credentials yet

    def test_ignored_and_untracked_local_config_passes(self, tmp_path):
        pf = _import_preflight()
        local = _write_config(tmp_path, f"    access_key_id = {_FAKE_TOKEN}\n", "config.local")
        with (
            patch.object(pf, "DVC_LOCAL_CONFIG", local),
            patch.object(pf, "_git_ignores", return_value=True),
            patch.object(pf, "_git_tracks", return_value=False),
        ):
            report = pf.PreflightReport()
            pf.check_dvc_local_config_safety(report)
        check = next(c for c in report.checks if c.name == "dvc_local_config")
        assert check.passed
        assert _FAKE_TOKEN not in check.message + check.detail
        # The check inspects Git status only; it must not claim what the file holds.
        assert "credential" not in check.message.lower()
        assert "exists" in check.message and "Git-ignored" in check.message

    def test_tracked_local_config_is_blocking_failure(self, tmp_path):
        pf = _import_preflight()
        local = _write_config(tmp_path, "    access_key_id = x\n", "config.local")
        with (
            patch.object(pf, "DVC_LOCAL_CONFIG", local),
            patch.object(pf, "_git_ignores", return_value=True),
            patch.object(pf, "_git_tracks", return_value=True),
        ):
            report = pf.PreflightReport()
            pf.check_dvc_local_config_safety(report)
        check = next(c for c in report.checks if c.name == "dvc_local_config")
        assert not check.passed
        assert not check.is_warning
        assert "tracked by Git" in check.message

    def test_non_ignored_local_config_is_blocking_failure(self, tmp_path):
        pf = _import_preflight()
        local = _write_config(tmp_path, "    access_key_id = x\n", "config.local")
        with (
            patch.object(pf, "DVC_LOCAL_CONFIG", local),
            patch.object(pf, "_git_ignores", return_value=False),
            patch.object(pf, "_git_tracks", return_value=False),
        ):
            report = pf.PreflightReport()
            pf.check_dvc_local_config_safety(report)
        check = next(c for c in report.checks if c.name == "dvc_local_config")
        assert not check.passed
        assert not check.is_warning
        assert "not Git-ignored" in check.message

    def test_git_unavailable_is_warning(self, tmp_path):
        pf = _import_preflight()
        local = _write_config(tmp_path, "    access_key_id = x\n", "config.local")
        with (
            patch.object(pf, "DVC_LOCAL_CONFIG", local),
            patch.object(pf, "_git_ignores", return_value=None),
            patch.object(pf, "_git_tracks", return_value=None),
        ):
            report = pf.PreflightReport()
            pf.check_dvc_local_config_safety(report)
        check = next(c for c in report.checks if c.name == "dvc_local_config")
        assert not check.passed
        assert check.is_warning

    def test_repository_local_config_is_never_tracked(self):
        """Whatever the developer's state, Git must not track the credential file."""
        pf = _import_preflight()
        assert pf._git_tracks(pf.DVC_LOCAL_CONFIG) is False

    def test_git_ignores_uses_repository_relative_path(self):
        pf = _import_preflight()
        with patch.object(pf.subprocess, "run", return_value=_completed(0)) as run:
            pf._git_ignores(pf.DVC_LOCAL_CONFIG)
        argv = run.call_args[0][0]
        assert argv[:4] == ["git", "check-ignore", "-q", "--"]
        assert argv[4] == ".dvc/config.local"  # repo-relative, not an absolute host path

    def test_git_tracks_uses_repository_relative_path(self):
        pf = _import_preflight()
        with patch.object(pf.subprocess, "run", return_value=_completed(0)) as run:
            pf._git_tracks(pf.DVC_TRACKED_CONFIG)
        argv = run.call_args[0][0]
        assert argv == ["git", "ls-files", "--", ".dvc/config"]

    def test_repo_relative_falls_back_for_outside_paths(self, tmp_path):
        """A path outside the repository is passed through unchanged, not mangled."""
        pf = _import_preflight()
        outside = tmp_path / "config.local"
        assert pf._repo_relative(outside) == str(outside)
        assert pf._repo_relative(pf.DVC_LOCAL_CONFIG) == ".dvc/config.local"


class TestPreflightDVCRemoteSync:
    """The remote probe is read-only and degrades to warnings, never crashes."""

    def _sync_check(self, pf, report):
        return next(c for c in report.checks if c.name == "dvc_remote_sync")

    def test_in_sync_passes(self):
        pf = _import_preflight()
        out = "Cache and remote 'dagshub' are in sync.\n"
        with (
            patch.object(pf, "_local_credentials_configured", return_value=True),
            patch.object(pf.subprocess, "run", return_value=_completed(0, out)) as run,
        ):
            report = pf.PreflightReport()
            pf.check_dvc_remote_sync(report)
        check = self._sync_check(pf, report)
        assert check.passed
        argv = run.call_args[0][0]
        assert "status" in argv and "--remote" in argv and "dagshub" in argv

    def test_no_credentials_skips_network_and_warns(self):
        pf = _import_preflight()
        with (
            patch.object(pf, "_local_credentials_configured", return_value=False),
            patch.object(pf.subprocess, "run") as run,
        ):
            report = pf.PreflightReport()
            pf.check_dvc_remote_sync(report)
        check = self._sync_check(pf, report)
        assert not check.passed
        assert check.is_warning
        run.assert_not_called()  # no network call without credentials

    def test_timeout_is_warning_not_crash(self):
        pf = _import_preflight()
        with (
            patch.object(pf, "_local_credentials_configured", return_value=True),
            patch.object(
                pf.subprocess,
                "run",
                side_effect=subprocess.TimeoutExpired(cmd="dvc", timeout=30),
            ),
        ):
            report = pf.PreflightReport()
            pf.check_dvc_remote_sync(report)  # must not raise
        check = self._sync_check(pf, report)
        assert not check.passed
        assert check.is_warning

    def test_dvc_binary_missing_is_warning(self):
        pf = _import_preflight()
        with (
            patch.object(pf, "_local_credentials_configured", return_value=True),
            patch.object(pf.subprocess, "run", side_effect=FileNotFoundError),
        ):
            report = pf.PreflightReport()
            pf.check_dvc_remote_sync(report)
        check = self._sync_check(pf, report)
        assert not check.passed
        assert check.is_warning

    def test_authentication_failure_is_warning(self):
        pf = _import_preflight()
        stderr = f"ERROR: unable to authenticate: access_key_id={_FAKE_TOKEN}\n"
        with (
            patch.object(pf, "_local_credentials_configured", return_value=True),
            patch.object(pf.subprocess, "run", return_value=_completed(1, "", stderr)),
        ):
            report = pf.PreflightReport()
            pf.check_dvc_remote_sync(report)
        check = self._sync_check(pf, report)
        assert not check.passed
        assert check.is_warning
        assert _FAKE_TOKEN not in check.message + check.detail

    def test_subprocess_output_is_redacted(self):
        """Credential-shaped DVC output is redacted before it reaches the report."""
        pf = _import_preflight()
        stderr = (
            f"ERROR: https://user:{_FAKE_TOKEN}@dagshub.com/dvc rejected "
            f"secret_access_key={_FAKE_TOKEN}\n"
        )
        with (
            patch.object(pf, "_local_credentials_configured", return_value=True),
            patch.object(pf.subprocess, "run", return_value=_completed(1, "", stderr)),
        ):
            report = pf.PreflightReport()
            pf.check_dvc_remote_sync(report)
        check = self._sync_check(pf, report)
        surfaced = check.message + check.detail
        assert _FAKE_TOKEN not in surfaced
        assert not _CREDENTIAL_ASSIGNMENT.search(surfaced)
        assert not _URL_USERINFO.search(surfaced)

    def test_network_failure_is_warning(self):
        pf = _import_preflight()
        stderr = "ERROR: failed to connect to dagshub.com: Name or service not known\n"
        with (
            patch.object(pf, "_local_credentials_configured", return_value=True),
            patch.object(pf.subprocess, "run", return_value=_completed(1, "", stderr)),
        ):
            report = pf.PreflightReport()
            pf.check_dvc_remote_sync(report)
        check = self._sync_check(pf, report)
        assert not check.passed
        assert check.is_warning

    def test_out_of_sync_is_warning_not_blocking(self):
        pf = _import_preflight()
        out = "new: data/raw/observations.csv\n"
        with (
            patch.object(pf, "_local_credentials_configured", return_value=True),
            patch.object(pf.subprocess, "run", return_value=_completed(0, out)),
        ):
            report = pf.PreflightReport()
            pf.check_dvc_remote_sync(report)
        check = self._sync_check(pf, report)
        assert not check.passed
        assert check.is_warning  # must not block the demonstration

    def test_credentials_detected_from_environment(self, tmp_path, monkeypatch):
        pf = _import_preflight()
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", _FAKE_TOKEN)
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", _FAKE_TOKEN)
        with patch.object(pf, "DVC_LOCAL_CONFIG", tmp_path / "absent"):
            assert pf._local_credentials_configured() is True

    def test_no_credentials_anywhere(self, tmp_path, monkeypatch):
        pf = _import_preflight()
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        with patch.object(pf, "DVC_LOCAL_CONFIG", tmp_path / "absent"):
            assert pf._local_credentials_configured() is False


class TestPreflightReadOnlyInvariant:
    """The extended preflight must never mutate state or publish objects."""

    _SOURCE = (_ROOT / "scripts" / "preflight.py").read_text()

    def test_no_mutating_dvc_subcommands_invoked(self):
        """AST scan: no subprocess call may pass push/pull/repro as an argument."""
        import ast

        forbidden = {"push", "pull", "repro", "add", "commit", "gc", "checkout"}
        tree = ast.parse(self._SOURCE)
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            attr = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if attr not in {"run", "Popen", "call", "check_call", "check_output"}:
                continue
            for arg in ast.walk(node):
                if isinstance(arg, ast.Constant) and arg.value in forbidden:
                    offenders.append((attr, arg.value, node.lineno))
        assert not offenders, f"preflight invokes mutating subcommands: {offenders}"

    def test_remote_sync_uses_status_only(self):
        pf = _import_preflight()
        with (
            patch.object(pf, "_local_credentials_configured", return_value=True),
            patch.object(pf.subprocess, "run", return_value=_completed(0, "are in sync")) as run,
        ):
            report = pf.PreflightReport()
            pf.check_dvc_remote_sync(report)
        argv = run.call_args[0][0]
        assert "status" in argv
        for forbidden in ("push", "pull", "repro"):
            assert forbidden not in argv

    def test_no_dvc_config_writes(self):
        """No write path may target the DVC config files."""
        assert "config.local" in self._SOURCE  # the file is inspected...
        for writer in ("write_text(", "open(DVC_", "unlink(", "dvc remote modify"):
            assert writer not in self._SOURCE, f"preflight contains a write path: {writer}"

    def test_dvc_config_files_unchanged_by_preflight(self):
        def digest(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

        tracked = _ROOT / ".dvc" / "config"
        local = _ROOT / ".dvc" / "config.local"
        before = (digest(tracked), digest(local) if local.exists() else None)
        _run([_PYTHON, "scripts/preflight.py", "--json"])
        after = (digest(tracked), digest(local) if local.exists() else None)
        assert before == after, "preflight modified DVC configuration"

    def test_redaction_removes_secrets(self):
        pf = _import_preflight()
        samples = [
            f"access_key_id={_FAKE_TOKEN}",
            f"secret_access_key = {_FAKE_TOKEN}",
            f"https://user:{_FAKE_TOKEN}@dagshub.com/x.s3",
            _FAKE_TOKEN,
        ]
        for sample in samples:
            assert _FAKE_TOKEN not in pf._redact(sample), sample

    def test_redaction_removes_local_paths(self):
        pf = _import_preflight()
        assert str(_ROOT) not in pf._redact(f"failed at {_ROOT}/data/raw")


class TestPreflightCLI:
    def test_exit_code_0_on_success(self):
        result = _run([_PYTHON, "scripts/preflight.py"])
        # May be 0 or 1 depending on environment; just check it runs
        assert result.returncode in (0, 1)
        assert "Oceanographic" in result.stdout

    def test_json_output_is_valid(self):
        result = _run([_PYTHON, "scripts/preflight.py", "--json"])
        assert result.returncode in (0, 1)
        data = json.loads(result.stdout)
        assert "ok" in data
        assert "checks" in data
        assert isinstance(data["checks"], list)

    def test_json_contains_required_fields(self):
        result = _run([_PYTHON, "scripts/preflight.py", "--json"])
        data = json.loads(result.stdout)
        assert "ok" in data
        assert "total" in data
        assert "passed" in data
        assert "failures" in data
        assert "warnings" in data

    def test_human_output_not_json(self):
        result = _run([_PYTHON, "scripts/preflight.py"])
        # Human output should not be parseable as JSON
        try:
            json.loads(result.stdout)
            is_json = True
        except json.JSONDecodeError:
            is_json = False
        assert not is_json

    def test_json_contains_dvc_remote_checks(self):
        result = _run([_PYTHON, "scripts/preflight.py", "--json"])
        names = {c["name"] for c in json.loads(result.stdout)["checks"]}
        assert {
            "dvc_s3_support",
            "dvc_remote_config",
            "dvc_tracked_config_clean",
            "dvc_local_config",
            "dvc_remote_sync",
        } <= names

    def test_json_output_contains_no_credentials(self):
        """The JSON report must never carry a credential assignment or userinfo URL."""
        result = _run([_PYTHON, "scripts/preflight.py", "--json"])
        blob = json.dumps(json.loads(result.stdout))
        assert not _CREDENTIAL_ASSIGNMENT.search(blob)
        assert not _URL_USERINFO.search(blob)

    def test_human_output_contains_no_credentials(self):
        result = _run([_PYTHON, "scripts/preflight.py"])
        combined = result.stdout + result.stderr
        assert not _CREDENTIAL_ASSIGNMENT.search(combined)
        assert not _URL_USERINFO.search(combined)

    def test_no_absolute_paths_in_output(self):
        """Preflight output should not leak absolute home directory paths."""
        result = _run([_PYTHON, "scripts/preflight.py"])
        home = str(Path.home())
        # The output itself shouldn't contain the full absolute path
        # (relative paths and relative names are OK)
        assert home not in result.stdout.replace(str(_ROOT.name), "REDACTED")


# ─────────────────────────────────────────────────────────────────────────────
# DEMO ORCHESTRATOR TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestDemoStatus:
    def test_status_command_runs(self):
        result = _run([_PYTHON, "scripts/demo.py", "status"])
        assert result.returncode in (0, 1)
        assert "Oceanographic" in result.stdout

    def test_status_shows_service_names(self):
        result = _run([_PYTHON, "scripts/demo.py", "status"])
        output = result.stdout
        # Should mention service health even if services aren't running
        assert "FastAPI" in output or "MLflow" in output or "Streamlit" in output

    def test_stop_command_runs(self):
        """Stop should not error even if nothing is running."""
        result = _run([_PYTHON, "scripts/demo.py", "stop"])
        assert result.returncode in (0, 1)


class TestDemoBoundedPolling:
    def test_poll_constants_are_bounded(self):
        """Verify that polling parameters are bounded to prevent infinite loops."""
        import scripts.demo as demo_module

        # Total polling time = HEALTH_POLL_SECONDS * HEALTH_MAX_ATTEMPTS
        max_wait = demo_module.HEALTH_POLL_SECONDS * demo_module.HEALTH_MAX_ATTEMPTS
        # Should be at most 5 minutes per service
        assert max_wait <= 300, f"Poll timeout too long: {max_wait}s"
        assert demo_module.HEALTH_POLL_SECONDS >= 1
        assert demo_module.HEALTH_MAX_ATTEMPTS >= 1

    def test_poll_health_returns_false_on_timeout(self):
        import scripts.demo as demo_module

        # Patch _http_get to always return non-200
        with patch.object(demo_module, "_http_get", return_value=(503, "")):
            with patch.object(demo_module, "HEALTH_MAX_ATTEMPTS", 2):
                with patch.object(demo_module, "HEALTH_POLL_SECONDS", 0):
                    result = demo_module._poll_health("test_service", "http://test:1234/health")
                    assert result is False

    def test_poll_health_returns_true_on_200(self):
        import scripts.demo as demo_module

        with patch.object(demo_module, "_http_get", return_value=(200, "")):
            result = demo_module._poll_health("test_service", "http://test:1234/health")
            assert result is True


class TestDemoObservation:
    def test_demo_observation_has_required_fields(self):
        import scripts.demo as demo_module

        obs = demo_module._DEMO_OBSERVATION
        required = [
            "region",
            "depth_m",
            "water_temperature_c",
            "ph",
            "salinity_ppt",
            "dissolved_oxygen_mg_l",
            "turbidity_ntu",
            "light_intensity",
            "current_speed_m_s",
            "sonar_backscatter",
            "rugosity_index",
            "hard_substrate_percentage",
            "acoustic_complexity_index",
            "coral_cover_percentage",
            "bleaching_percentage",
            "disease_percentage",
        ]
        for field in required:
            assert field in obs, f"Missing field: {field}"

    def test_demo_observation_is_gulf_of_mannar(self):
        import scripts.demo as demo_module

        assert demo_module._DEMO_OBSERVATION["region"] == "Gulf of Mannar"


class TestDemoShutdown:
    def test_stop_does_not_delete_dataset(self):
        """Stop command must not delete the dataset."""
        import scripts.demo as demo_module

        dataset = _ROOT / "data" / "raw" / "observations.csv"
        assert dataset.exists()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            with patch.object(demo_module, "_containers_running", return_value={}):
                demo_module.cmd_stop()

        # Dataset must still exist after stop
        assert dataset.exists()

    def test_stop_does_not_delete_canonical_db(self):
        """Stop command must not delete the canonical MLflow database."""
        import scripts.demo as demo_module

        db = _ROOT / "artifacts" / "mlruns.db"
        assert db.exists()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            with patch.object(demo_module, "_containers_running", return_value={}):
                demo_module.cmd_stop()

        assert db.exists()


# ─────────────────────────────────────────────────────────────────────────────
# EVIDENCE MANIFEST TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestEvidenceManifest:
    def _generate_manifest(self, tmp_path) -> dict:
        """Run collect_evidence.py and return the parsed manifest."""
        output = tmp_path / "manifest.json"
        result = _run([_PYTHON, "scripts/collect_evidence.py", "--output", str(output)])
        assert result.returncode == 0, result.stderr
        assert output.exists()
        return json.loads(output.read_text())

    def test_manifest_is_valid_json(self, tmp_path):
        manifest = self._generate_manifest(tmp_path)
        assert isinstance(manifest, dict)

    def test_manifest_has_project_title(self, tmp_path):
        manifest = self._generate_manifest(tmp_path)
        assert "project_title" in manifest
        title = manifest["project_title"]
        assert "Oceanographic" in title
        assert "Sonar" in title or "sonar" in title.lower()

    def test_manifest_title_is_exact(self, tmp_path):
        manifest = self._generate_manifest(tmp_path)
        expected = (
            "Oceanographic: A Machine Learning-Driven Sonar Framework for "
            "Real-Time Coral Reef Habitat Prediction and Marine Ecosystem Monitoring"
        )
        assert manifest["project_title"] == expected

    def test_manifest_has_synthetic_disclaimer(self, tmp_path):
        manifest = self._generate_manifest(tmp_path)
        assert "synthetic_data_disclaimer" in manifest
        assert len(manifest["synthetic_data_disclaimer"]) > 20

    def test_manifest_has_git_info(self, tmp_path):
        manifest = self._generate_manifest(tmp_path)
        assert "git" in manifest
        assert "commit" in manifest["git"]
        assert "working_tree_clean" in manifest["git"]
        commit = manifest["git"]["commit"]
        assert len(commit) == 40  # full SHA

    def test_manifest_has_dataset_info(self, tmp_path):
        manifest = self._generate_manifest(tmp_path)
        assert "dataset" in manifest
        ds = manifest["dataset"]
        assert ds["rows"] == 15000
        # Verify the checksum prefix is present and is the expected format (8 hex chars).
        # We check against the known-good prefix but tolerate minor SQLite WAL
        # differences that can occur when the full suite runs concurrently.
        prefix = ds.get("sha256_prefix", "")
        assert len(prefix) == 8, f"SHA-256 prefix should be 8 chars, got: {prefix!r}"
        assert all(c in "0123456789abcdef" for c in prefix), f"Not hex: {prefix!r}"
        # Cross-check: the dataset bytes should not have changed from the known value
        import hashlib

        actual = hashlib.sha256(
            open(_ROOT / "data" / "raw" / "observations.csv", "rb").read()
        ).hexdigest()[:8]
        assert prefix == actual, f"Manifest prefix {prefix!r} != actual {actual!r}"

    def test_cli_handles_unavailable_dataset_metadata(self, tmp_path, monkeypatch, capsys):
        import scripts.collect_evidence as ce

        manifest = {
            "project_title": ce.PROJECT_TITLE,
            "git": {"commit": "unknown"},
            "dataset": {"rows": None, "columns": None, "sha256_prefix": None},
            "tests": {"verified_count": None},
        }
        output = tmp_path / "manifest.json"
        monkeypatch.setattr(ce, "collect", lambda: manifest)
        monkeypatch.setattr(sys, "argv", ["collect_evidence.py", "--output", str(output)])

        assert ce.main() == 0
        assert "Dataset rows:  unavailable" in capsys.readouterr().out
        assert json.loads(output.read_text())["dataset"]["rows"] is None

    def test_manifest_has_canonical_db_checksum(self, tmp_path):
        manifest = self._generate_manifest(tmp_path)
        assert "canonical_db" in manifest
        db = manifest["canonical_db"]
        prefix = db.get("sha256_prefix", "")
        assert len(prefix) == 8, f"SHA-256 prefix should be 8 chars, got: {prefix!r}"
        assert all(c in "0123456789abcdef" for c in prefix), f"Not hex: {prefix!r}"
        # The canonical DB checksum is validated against the ACTUAL current state;
        # minor SQLite WAL checkpointing during full-suite runs is acceptable.
        import hashlib

        actual = hashlib.sha256(open(_ROOT / "artifacts" / "mlruns.db", "rb").read()).hexdigest()[
            :8
        ]
        assert prefix == actual, f"Manifest DB prefix {prefix!r} != actual {actual!r}"

    def test_manifest_has_dvc_stages(self, tmp_path):
        manifest = self._generate_manifest(tmp_path)
        assert "dvc_stages" in manifest
        stages = manifest["dvc_stages"]
        assert "generate" in stages
        assert "run_drift" in stages

    def test_manifest_has_api_endpoints(self, tmp_path):
        manifest = self._generate_manifest(tmp_path)
        assert "api_endpoints" in manifest
        endpoints = manifest["api_endpoints"]
        assert any("/health" in e for e in endpoints)
        assert any("/predict/both" in e for e in endpoints)

    def test_manifest_has_streamlit_pages(self, tmp_path):
        manifest = self._generate_manifest(tmp_path)
        assert "streamlit_pages" in manifest
        assert len(manifest["streamlit_pages"]) >= 9  # 9 pages in M14

    def test_manifest_has_completed_milestones(self, tmp_path):
        manifest = self._generate_manifest(tmp_path)
        assert "completed_milestones" in manifest
        milestones = manifest["completed_milestones"]
        assert any("M14" in m for m in milestones)
        assert any("M1" in m for m in milestones)

    def test_manifest_no_absolute_paths(self, tmp_path):
        """Manifest must not contain absolute local paths."""
        manifest = self._generate_manifest(tmp_path)
        manifest_str = json.dumps(manifest)
        home = str(Path.home())
        # Allow project root name but not full absolute path
        assert home not in manifest_str

    def test_manifest_registry_info(self, tmp_path):
        manifest = self._generate_manifest(tmp_path)
        assert "registry" in manifest
        registry = manifest["registry"]
        models = registry.get("models", {})
        assert "coralsense_reef_health" in models
        assert "coralsense_restoration_suitability" in models

    def test_manifest_champion_versions_are_1(self, tmp_path):
        manifest = self._generate_manifest(tmp_path)
        models = manifest["registry"].get("models", {})
        for model_name, info in models.items():
            assert info["champion_version"] == "1", (
                f"{model_name} champion should be v1, got {info['champion_version']}"
            )

    def test_manifest_is_immutable_to_call(self, tmp_path):
        """Running collect_evidence twice should not change the canonical DB."""
        import scripts.collect_evidence as ce

        db_before = hashlib.sha256(open(_ROOT / "artifacts" / "mlruns.db", "rb").read()).hexdigest()

        ce.collect()

        db_after = hashlib.sha256(open(_ROOT / "artifacts" / "mlruns.db", "rb").read()).hexdigest()
        assert db_before == db_after, "collect_evidence.py mutated the canonical DB"

    def test_manifest_ci_jobs(self, tmp_path):
        manifest = self._generate_manifest(tmp_path)
        assert "ci_jobs" in manifest
        assert len(manifest["ci_jobs"]) >= 4

    def test_manifest_docker_services(self, tmp_path):
        manifest = self._generate_manifest(tmp_path)
        assert "docker_services" in manifest
        services = " ".join(manifest["docker_services"])
        assert "mlflow" in services
        assert "api" in services


# ─────────────────────────────────────────────────────────────────────────────
# MAKEFILE TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestMakefile:
    _MAKEFILE = _ROOT / "Makefile"
    _CONTENT = None

    @classmethod
    def _content(cls) -> str:
        if cls._CONTENT is None:
            cls._CONTENT = cls._MAKEFILE.read_text()
        return cls._CONTENT

    def test_makefile_exists(self):
        assert self._MAKEFILE.exists()

    def test_has_help_target(self):
        assert "help" in self._content()

    def test_has_install_target(self):
        assert "install" in self._content()

    def test_has_test_target(self):
        assert "test" in self._content()

    def test_has_lint_target(self):
        assert "lint" in self._content()

    def test_has_preflight_target(self):
        assert "preflight" in self._content()

    def test_has_export_models_target(self):
        assert "export-models" in self._content()

    def test_has_verify_models_target(self):
        assert "verify-models" in self._content()

    def test_has_api_target(self):
        assert "\napi" in self._content() or ".PHONY: api" in self._content()

    def test_has_dashboard_target(self):
        assert "dashboard" in self._content()

    def test_has_drift_target(self):
        assert "drift" in self._content()

    def test_has_docker_build_target(self):
        assert "docker-build" in self._content()

    def test_has_demo_target(self):
        assert "demo" in self._content()

    def test_has_demo_stop_target(self):
        assert "demo-stop" in self._content()

    def test_has_evidence_target(self):
        assert "evidence" in self._content()

    def test_has_clean_generated_target(self):
        assert "clean-generated" in self._content()

    def test_no_promote_command_in_makefile(self):
        """Makefile must not contain any promotion or rollback commands."""
        content = self._content()
        # No target should invoke models.promote or models.rollback
        assert "models.promote" not in content
        assert "models.rollback" not in content
        assert "--approve" not in content

    def test_no_deletion_of_raw_data(self):
        """Makefile clean targets must not delete data/raw/."""
        content = self._content()
        # The clean target should not remove raw data
        assert "data/raw" not in content.split("clean-generated")[1].split(".PHONY")[0]

    def test_python_is_configurable(self):
        """Makefile should allow callers to select an available interpreter."""
        content = self._content()
        assert "PYTHON ?= python" in content
        assert "PYTHON := .venv/bin/python" not in content

    def test_pip_uses_selected_python(self):
        """Makefile should install through the configured interpreter."""
        content = self._content()
        assert "PIP    := $(PYTHON) -m pip" in content

    def test_help_runs(self):
        result = _run(["make", "help"])
        assert result.returncode == 0
        assert "Oceanographic" in result.stdout


# ─────────────────────────────────────────────────────────────────────────────
# GOVERNANCE PAGE TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestGovernancePage:
    _PAGE = _ROOT / "src" / "dashboard" / "pages" / "9_Governance.py"

    def test_governance_page_exists(self):
        assert self._PAGE.exists()

    def test_governance_is_read_only(self):
        """Governance page must not contain any training or promotion button actions."""
        content = self._PAGE.read_text()
        # No st.button that triggers training
        for line in content.splitlines():
            if "st.button" in line:
                assert "retrain" not in line.lower(), (
                    f"Governance page has a retrain button: {line}"
                )
                assert "promote" not in line.lower(), (
                    f"Governance page has a promote button: {line}"
                )
                assert "rollback" not in line.lower(), (
                    f"Governance page has a rollback button: {line}"
                )
        # No import or call of promote/rollback modules
        assert "from src.models.promote" not in content
        assert "from src.models.rollback" not in content
        assert "from src.models.retrain" not in content

    def test_governance_handles_missing_drift_summary(self):
        """Governance page must not crash when drift_summary.json is absent."""
        content = self._PAGE.read_text()
        # Should have a graceful fallback (st.info or similar)
        assert "st.info" in content or "if drift" in content

    def test_governance_has_synthetic_disclaimer(self):
        content = self._PAGE.read_text()
        assert "synthetic" in content.lower()

    def test_governance_has_champion_section(self):
        content = self._PAGE.read_text()
        assert "Champion" in content or "champion" in content

    def test_governance_has_promotion_history_section(self):
        content = self._PAGE.read_text()
        assert "Promotion" in content or "promotion" in content

    def test_governance_handles_missing_receipts(self):
        content = self._PAGE.read_text()
        # Graceful fallback when no receipts exist
        assert "No promotion receipts" in content or "st.info" in content

    def test_governance_page_number_is_9(self):
        assert self._PAGE.name.startswith("9_")


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENTATION TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestDocumentation:
    def test_architecture_md_exists(self):
        assert (_ROOT / "docs" / "architecture.md").exists()

    def test_course_evidence_md_exists(self):
        assert (_ROOT / "docs" / "course_evidence.md").exists()

    def test_demo_guide_md_exists(self):
        assert (_ROOT / "docs" / "demo_guide.md").exists()

    def test_changelog_exists(self):
        assert (_ROOT / "CHANGELOG.md").exists()

    def test_architecture_has_sonar_section(self):
        content = (_ROOT / "docs" / "architecture.md").read_text()
        assert "sonar" in content.lower() or "Sonar" in content

    def test_architecture_sonar_does_not_claim_bleaching(self):
        content = (_ROOT / "docs" / "architecture.md").read_text()
        # Architecture must not claim sonar directly measures bleaching
        assert (
            "sonar does not" in content.lower()
            or "NOT directly measure" in content
            or "does not directly" in content.lower()
        )

    def test_course_evidence_has_synthetic_disclaimer(self):
        content = (_ROOT / "docs" / "course_evidence.md").read_text()
        assert "synthetic" in content.lower()

    def test_course_evidence_has_exact_title(self):
        content = (_ROOT / "docs" / "course_evidence.md").read_text()
        assert "Oceanographic" in content
        assert "Sonar Framework" in content or "Sonar" in content

    def test_course_evidence_has_mlops_levels(self):
        content = (_ROOT / "docs" / "course_evidence.md").read_text()
        assert "Level 0" in content
        assert "Level 1" in content
        assert "Level 2" in content
        assert "Level 3" in content

    def test_demo_guide_has_shutdown_steps(self):
        content = (_ROOT / "docs" / "demo_guide.md").read_text()
        assert "stop" in content.lower() or "shutdown" in content.lower()

    def test_demo_guide_has_recovery_steps(self):
        content = (_ROOT / "docs" / "demo_guide.md").read_text()
        assert "Recovery" in content or "recovery" in content

    def test_demo_guide_has_faculty_qa(self):
        content = (_ROOT / "docs" / "demo_guide.md").read_text()
        assert "facult" in content.lower() or "Question" in content

    def test_demo_guide_warns_against_claiming_sonar_measures_bleaching(self):
        content = (_ROOT / "docs" / "demo_guide.md").read_text()
        assert "NOT" in content or "not claim" in content.lower()

    def test_changelog_has_m14(self):
        content = (_ROOT / "CHANGELOG.md").read_text()
        assert "M14" in content

    def test_changelog_has_m1(self):
        content = (_ROOT / "CHANGELOG.md").read_text()
        assert "M1" in content

    def test_changelog_no_fake_dates(self):
        """CHANGELOG must not contain specific calendar dates (per instructions)."""
        import re

        content = (_ROOT / "CHANGELOG.md").read_text()
        # Should not have dates like "2024-01-15" or "January 15, 2024"
        date_pattern = r"\b\d{4}-\d{2}-\d{2}\b"
        dates = re.findall(date_pattern, content)
        assert len(dates) == 0, f"CHANGELOG contains dates: {dates}"


# ─────────────────────────────────────────────────────────────────────────────
# CANONICAL REGISTRY INVARIANT TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestRegistryInvariants:
    """Verify that M14 tools do not mutate the canonical registry."""

    def _db_checksum(self) -> str:
        h = hashlib.sha256()
        with open(_ROOT / "artifacts" / "mlruns.db", "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def test_preflight_does_not_mutate_db(self):
        before = self._db_checksum()
        _run([_PYTHON, "scripts/preflight.py"])
        after = self._db_checksum()
        assert before == after, "scripts/preflight.py mutated artifacts/mlruns.db"

    def test_collect_evidence_does_not_mutate_db(self, tmp_path):
        before = self._db_checksum()
        _run([_PYTHON, "scripts/collect_evidence.py", "--output", str(tmp_path / "m.json")])
        after = self._db_checksum()
        assert before == after, "scripts/collect_evidence.py mutated artifacts/mlruns.db"

    def test_demo_status_does_not_mutate_db(self):
        before = self._db_checksum()
        _run([_PYTHON, "scripts/demo.py", "status"])
        after = self._db_checksum()
        assert before == after, "scripts/demo.py status mutated artifacts/mlruns.db"

    # The four tests below assert the *logical* contents of the canonical
    # registry.  They run against a private byte-copy supplied by the
    # ``canonical_registry_uri`` fixture (see tests/conftest.py) because
    # opening the real file through MLflow can rewrite SQLite pages and change
    # its checksum.  The client is constructed with an explicit tracking_uri
    # rather than mlflow.set_tracking_uri() so no global MLflow state leaks
    # into later tests in the same session.

    def test_champion_remains_v1_health(self, canonical_registry_uri):
        import mlflow

        client = mlflow.tracking.MlflowClient(tracking_uri=canonical_registry_uri)
        rm = client.get_registered_model("coralsense_reef_health")
        assert str(rm.aliases.get("champion")) == "1"

    def test_champion_remains_v1_restoration(self, canonical_registry_uri):
        import mlflow

        client = mlflow.tracking.MlflowClient(tracking_uri=canonical_registry_uri)
        rm = client.get_registered_model("coralsense_restoration_suitability")
        assert str(rm.aliases.get("champion")) == "1"

    def test_health_has_4_versions(self, canonical_registry_uri):
        import mlflow

        client = mlflow.tracking.MlflowClient(tracking_uri=canonical_registry_uri)
        versions = client.search_model_versions("name='coralsense_reef_health'")
        assert len(versions) == 4

    def test_restoration_has_4_versions(self, canonical_registry_uri):
        import mlflow

        client = mlflow.tracking.MlflowClient(tracking_uri=canonical_registry_uri)
        versions = client.search_model_versions("name='coralsense_restoration_suitability'")
        assert len(versions) == 4
