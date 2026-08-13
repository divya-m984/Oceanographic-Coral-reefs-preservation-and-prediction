#!/usr/bin/env python3
"""
scripts/preflight.py — Oceanographic MLOps read-only preflight checker.

Verifies the project environment, data integrity, registry state, and the
DagsHub-backed DVC remote configuration before a classroom demonstration.

Strictly read-only: it never trains, registers, promotes, rolls back, runs
`dvc repro`/`dvc push`/`dvc pull`, writes credentials, or modifies any
project artifact.  Credential values are never read into the report — only
credential *field names* and Git ignore/track status are inspected.

Usage:
    python scripts/preflight.py           # human-readable output
    python scripts/preflight.py --json    # machine-readable JSON to stdout

Exit codes:
    0  All checks passed (warnings may exist)
    1  One or more blocking failures detected
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ── Resolved project root ──────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent

# ── Expected constants ─────────────────────────────────────────────────────────
DATASET_PATH = _ROOT / "data" / "raw" / "observations.csv"
DATASET_SHA256_PREFIX = "a03cb3e9"
DATASET_EXPECTED_ROWS = 15_000

CANONICAL_DB = _ROOT / "artifacts" / "mlruns.db"
CANONICAL_DB_SHA256_PREFIX = "b76a4015"

EXPECTED_PYTHON_MAJOR = 3
EXPECTED_PYTHON_MINOR_MIN = 11

REQUIRED_PORTS = [5000, 8000, 8501]
PROJECT_CONTAINER_NAMES = [
    "coralsense-mlflow",
    "coralsense-api",
    "coralsense-dashboard",
    "coralsense-drift",
]

REQUIRED_IMPORTS = [
    "mlflow",
    "sklearn",
    "xgboost",
    "fastapi",
    "uvicorn",
    "streamlit",
    "evidently",
    "dvc",
    "pandera",
    "pydantic",
    "plotly",
    "requests",
]

REGISTERED_MODELS = {
    "coralsense_reef_health": {"champion_version": "1", "min_versions": 4},
    "coralsense_restoration_suitability": {"champion_version": "1", "min_versions": 4},
}

# ── DagsHub-backed DVC remote (reproducibility + credential hygiene) ───────────
DVC_TRACKED_CONFIG = _ROOT / ".dvc" / "config"
DVC_LOCAL_CONFIG = _ROOT / ".dvc" / "config.local"

DVC_REMOTE_NAME = "dagshub"
DVC_REMOTE_URL = "s3://dvc"
DVC_REMOTE_ENDPOINT = (
    "https://dagshub.com/divya-m984/Oceanographic-Coral-reefs-preservation-and-prediction.s3"
)
# DVC's canonical S3 endpoint key.  Config keys are matched case-insensitively,
# so a hand-written ``EndpointURL`` still normalises to this name.
DVC_ENDPOINT_KEY = "endpointurl"
# The pip extra that supplies the S3-compatible backend DagsHub requires.
DVC_S3_MODULE = "dvc_s3"

# Credential field names that must never appear in the tracked .dvc/config.
DVC_CREDENTIAL_KEYS = frozenset(
    {
        "access_key_id",
        "secret_access_key",
        "session_token",
        "password",
        "token",
        "auth_token",
    }
)

# ``dvc status --remote`` is read-only but talks to the network; the demo
# preflight must stay usable offline, so it is time-boxed and degrades to a
# warning.
DVC_REMOTE_STATUS_TIMEOUT = 30

DVC_EXPECTED_STAGES = {
    "generate",
    "validate",
    "preprocess",
    "train",
    "evaluate",
    "register_candidate",
    "run_drift",
}

REQUIRED_ARTIFACT_FILES = [
    _ROOT / "models" / "best_model_health.joblib",
    _ROOT / "models" / "best_model_restoration.joblib",
    _ROOT / "models" / "evaluation_health.json",
    _ROOT / "models" / "evaluation_restoration.json",
    _ROOT / "data" / "processed" / "preprocessor_health.joblib",
    _ROOT / "data" / "processed" / "preprocessor_restoration.joblib",
]

REQUIRED_CONFIG_FILES = [
    _ROOT / "params.yaml",
    _ROOT / "dvc.yaml",
    _ROOT / "docker-compose.yml",
    _ROOT / ".github" / "workflows" / "ci.yml",
    _ROOT / "pyproject.toml",
]

REQUIRED_BUNDLE_FILES = [
    _ROOT / "deploy" / "bundles" / "manifest.json",
    _ROOT / "deploy" / "bundles" / "health",
    _ROOT / "deploy" / "bundles" / "restoration",
]


# ── Result types ───────────────────────────────────────────────────────────────
@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    is_warning: bool = False
    detail: str = ""


@dataclass
class PreflightReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and not c.is_warning]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.is_warning]

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def ok(self) -> bool:
        return len(self.failures) == 0

    def add(self, result: CheckResult) -> None:
        self.checks.append(result)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "total": len(self.checks),
            "passed": self.passed_count,
            "failures": len(self.failures),
            "warnings": len(self.warnings),
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "warning": c.is_warning,
                    "message": c.message,
                    "detail": c.detail,
                }
                for c in self.checks
            ],
        }


# ── Individual checks ──────────────────────────────────────────────────────────
def _check(
    report: PreflightReport,
    name: str,
    passed: bool,
    ok_msg: str,
    fail_msg: str,
    warning: bool = False,
    detail: str = "",
) -> None:
    report.add(
        CheckResult(
            name=name,
            passed=passed,
            message=ok_msg if passed else fail_msg,
            is_warning=warning and not passed,
            detail=detail,
        )
    )


def check_project_root(report: PreflightReport) -> None:
    markers = [_ROOT / "params.yaml", _ROOT / "dvc.yaml", _ROOT / "src" / "config.py"]
    ok = all(m.exists() for m in markers)
    _check(
        report,
        "project_root",
        ok,
        f"Project root confirmed: {_ROOT.name}",
        f"Project root markers missing in {_ROOT}",
    )


def check_python_version(report: PreflightReport) -> None:
    maj, mn = sys.version_info.major, sys.version_info.minor
    ok = maj == EXPECTED_PYTHON_MAJOR and mn >= EXPECTED_PYTHON_MINOR_MIN
    _check(
        report,
        "python_version",
        ok,
        f"Python {maj}.{mn} (>= 3.{EXPECTED_PYTHON_MINOR_MIN})",
        f"Python {maj}.{mn} is below minimum 3.{EXPECTED_PYTHON_MINOR_MIN}",
    )


def check_venv(report: PreflightReport) -> None:
    in_venv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )
    venv_dir = _ROOT / ".venv"
    _check(
        report,
        "virtual_environment",
        in_venv or venv_dir.exists(),
        "Virtual environment available",
        "No virtual environment detected — run: python3 -m venv .venv",
        warning=True,
    )


def check_imports(report: PreflightReport) -> None:
    missing = []
    for pkg in REQUIRED_IMPORTS:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    ok = len(missing) == 0
    _check(
        report,
        "required_imports",
        ok,
        f"All {len(REQUIRED_IMPORTS)} required packages importable",
        f"Missing packages: {', '.join(missing)} — run: make install",
    )


def check_docker(report: PreflightReport) -> None:
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        ok = result.returncode == 0
        detail = result.stdout.strip() if ok else result.stderr.strip()[:120]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        ok = False
        detail = "docker binary not found"
    _check(
        report,
        "docker_available",
        ok,
        f"Docker available (v{detail})" if ok else "Docker not available",
        "Docker not found or daemon not running — install Docker Desktop or Docker Engine",
        warning=True,
        detail=detail,
    )


def check_docker_compose(report: PreflightReport) -> None:
    try:
        result = subprocess.run(
            ["docker", "compose", "version", "--short"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        ok = result.returncode == 0
        detail = result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        ok = False
        detail = "docker compose not found"
    _check(
        report,
        "docker_compose_available",
        ok,
        f"Docker Compose available (v{detail})" if ok else "Docker Compose not available",
        "Docker Compose plugin not found",
        warning=True,
        detail=detail,
    )


def check_ports(report: PreflightReport) -> None:
    for port in REQUIRED_PORTS:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                in_use = s.connect_ex(("127.0.0.1", port)) == 0
        except OSError:
            in_use = False
        _check(
            report,
            f"port_{port}_free",
            not in_use,
            f"Port {port} is available",
            f"Port {port} is already in use — stop the conflicting service",
            warning=True,
        )


def check_dataset(report: PreflightReport) -> None:
    ok = DATASET_PATH.exists()
    try:
        ds_rel = DATASET_PATH.relative_to(_ROOT)
    except ValueError:
        ds_rel = DATASET_PATH
    _check(
        report,
        "dataset_exists",
        ok,
        "Dataset file found",
        f"Dataset missing: {ds_rel} — run: make preflight after dvc repro",
    )
    if not ok:
        return

    # Row count (header + 15000 data rows = 15001 lines)
    with open(DATASET_PATH, encoding="utf-8") as fh:
        row_count = sum(1 for _ in fh) - 1  # subtract header
    _check(
        report,
        "dataset_row_count",
        row_count == DATASET_EXPECTED_ROWS,
        f"Dataset has {row_count:,} rows",
        f"Dataset has {row_count:,} rows (expected {DATASET_EXPECTED_ROWS:,})",
    )


def check_dataset_checksum(report: PreflightReport) -> None:
    if not DATASET_PATH.exists():
        return
    h = hashlib.sha256()
    with open(DATASET_PATH, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    prefix = h.hexdigest()[:8]
    ok = prefix == DATASET_SHA256_PREFIX
    _check(
        report,
        "dataset_checksum",
        ok,
        f"Dataset SHA-256 prefix matches ({prefix})",
        f"Dataset SHA-256 prefix mismatch: got {prefix}, expected {DATASET_SHA256_PREFIX}",
    )


def check_dvc(report: PreflightReport) -> None:
    dvc_dir = _ROOT / ".dvc"
    ok = dvc_dir.exists()
    _check(
        report,
        "dvc_initialised",
        ok,
        "DVC repository initialised",
        "DVC not initialised — run: dvc init",
    )
    if not ok:
        return

    try:
        result = subprocess.run(
            [sys.executable, "-m", "dvc", "dag", "--md"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(_ROOT),
        )
        stages = {
            line.split("|")[1].strip()
            for line in result.stdout.splitlines()
            if "|" in line and "Stage" not in line and "---" not in line
        }
        # Fallback: parse dvc.yaml directly
        if not stages:
            import re

            dvc_yaml = (_ROOT / "dvc.yaml").read_text()
            stages = set(re.findall(r"^  (\w+):", dvc_yaml, re.MULTILINE))
        missing = DVC_EXPECTED_STAGES - stages
        ok2 = len(missing) == 0
        _check(
            report,
            "dvc_stages",
            ok2,
            f"All {len(DVC_EXPECTED_STAGES)} DVC stages present",
            f"Missing DVC stages: {missing}",
        )
    except subprocess.TimeoutExpired:
        _check(report, "dvc_stages", False, "", "DVC dag timed out", warning=True)


# ── DVC remote helpers (credential-safe) ───────────────────────────────────────
def _redact(text: str) -> str:
    """Strip anything credential-shaped (and local paths) from tool output.

    Applied to every byte of DVC/Git output that reaches a message, a detail
    field, the JSON report, or the terminal.
    """
    if not text:
        return ""
    # scheme://user:secret@host  →  scheme://***@host
    text = re.sub(r"(?i)\b([a-z][a-z0-9+.-]*://)[^\s/@]+:[^\s/@]+@", r"\1***@", text)
    # key = value / key=value for any credential-shaped key
    text = re.sub(
        r"(?i)\b(\w*(?:access_key_id|secret_access_key|session_token|password|token|secret)\w*)"
        r"\s*[:=]\s*\S+",
        r"\1=***",
        text,
    )
    # Bare long opaque tokens (DagsHub tokens are 40-char hex).
    text = re.sub(r"\b[0-9a-fA-F]{20,}\b", "***", text)
    # Never leak absolute local paths into the report.
    text = text.replace(str(_ROOT), ".").replace(str(Path.home()), "~")
    return text


def _summarise(result_stdout: str, result_stderr: str, limit: int = 200) -> str:
    """Build a short, redacted one-line summary of a subprocess result."""
    raw = (result_stderr or result_stdout or "").strip()
    first_lines = " / ".join(line.strip() for line in raw.splitlines()[:2] if line.strip())
    return _redact(first_lines)[:limit]


def _config_credential_keys(path: Path) -> list[str]:
    """Return the *names* of credential fields present in a DVC config file.

    Only the text to the left of the first ``=`` on each line is inspected;
    everything to the right is discarded immediately.  No credential value is
    ever returned, stored, logged, or serialised — this function exists purely
    so the preflight can answer "are credentials present?" without handling
    them.
    """
    if not path.exists():
        return []
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    found: set[str] = set()
    for line in raw.splitlines():
        head, sep, _ = line.partition("=")
        if not sep:
            continue
        key = head.strip().strip("'\"").lower()
        if key in DVC_CREDENTIAL_KEYS:
            found.add(key)
    return sorted(found)


def _remote_section_name(section: str) -> str | None:
    """Extract the remote name from a DVC config section header.

    DVC writes remote sections as ``['remote "dagshub"']``; quoting and
    spacing are normalised here so the check does not depend on exact
    formatting.
    """
    s = section.strip()
    for quote in ("'", '"'):
        if len(s) >= 2 and s.startswith(quote) and s.endswith(quote):
            s = s[1:-1].strip()
            break
    match = re.fullmatch(r"remote\s+(.+)", s, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip().strip("'\"").strip() or None


def _parse_dvc_remotes(path: Path) -> dict[str, dict[str, str]]:
    """Parse the *tracked* DVC config into ``{remote_name: {key: value}}``.

    The tracked file is read directly rather than through ``dvc config
    --list`` because the audit target is exactly what Git records: the merged
    runtime configuration would silently include ``.dvc/config.local``
    credentials.  Parsing is done with :mod:`configparser` so quoting,
    spacing, key case, and section order are all tolerated.
    """
    if not path.exists():
        return {}
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read_string(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, configparser.Error):
        return {}
    remotes: dict[str, dict[str, str]] = {}
    for section in parser.sections():
        name = _remote_section_name(section)
        if name is None:
            continue
        remotes[name] = {key.strip().lower(): value.strip() for key, value in parser.items(section)}
    return remotes


def _normalise_url(value: str) -> str:
    return value.strip().rstrip("/")


def _git_query(args: list[str]) -> subprocess.CompletedProcess | None:
    """Run a read-only Git command, or return ``None`` if Git is unusable."""
    try:
        return subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(_ROOT),
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None


def _repo_relative(path: Path) -> str:
    """Path as Git sees it: relative to the repository root where possible.

    Git commands run with ``cwd`` at the project root, so a repository-relative
    pathspec is both correct and free of absolute host paths.
    """
    try:
        return path.resolve().relative_to(_ROOT).as_posix()
    except ValueError:
        return str(path)


def _git_ignores(path: Path) -> bool | None:
    result = _git_query(["check-ignore", "-q", "--", _repo_relative(path)])
    if result is None or result.returncode not in (0, 1):
        return None
    return result.returncode == 0


def _git_tracks(path: Path) -> bool | None:
    result = _git_query(["ls-files", "--", _repo_relative(path)])
    if result is None or result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def _local_credentials_configured() -> bool:
    """Whether DagsHub credentials appear to be available locally.

    Detection is by credential *field name* or environment-variable presence
    only; no value is ever read into the report.
    """
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        return True
    return bool(_config_credential_keys(DVC_LOCAL_CONFIG))


# ── DVC remote checks ──────────────────────────────────────────────────────────
def check_dvc_s3_support(report: PreflightReport) -> None:
    """The DagsHub remote is S3-compatible; without dvc[s3] it cannot function."""
    try:
        available = importlib.util.find_spec(DVC_S3_MODULE) is not None
    except (ImportError, ValueError):
        available = False
    _check(
        report,
        "dvc_s3_support",
        available,
        "DVC S3 remote support available (dvc[s3])",
        f"DVC S3 support missing — remote '{DVC_REMOTE_NAME}' cannot function; "
        "install: pip install 'dvc[s3]>=3.50'",
    )


def check_dvc_remote_config(report: PreflightReport) -> None:
    """The tracked DVC config must declare the expected DagsHub remote."""
    if not DVC_TRACKED_CONFIG.exists():
        _check(
            report,
            "dvc_remote_config",
            False,
            "",
            "Tracked .dvc/config is missing — the DagsHub remote is not configured",
        )
        return

    remotes = _parse_dvc_remotes(DVC_TRACKED_CONFIG)
    remote = remotes.get(DVC_REMOTE_NAME)
    if remote is None:
        _check(
            report,
            "dvc_remote_config",
            False,
            "",
            f"Tracked .dvc/config declares no remote named '{DVC_REMOTE_NAME}' "
            f"(found: {sorted(remotes) or 'none'})",
        )
        return

    problems: list[str] = []
    url = _normalise_url(remote.get("url", ""))
    if url != _normalise_url(DVC_REMOTE_URL):
        problems.append(f"url is '{url or 'unset'}', expected '{DVC_REMOTE_URL}'")

    endpoint = _normalise_url(remote.get(DVC_ENDPOINT_KEY, ""))
    if endpoint != _normalise_url(DVC_REMOTE_ENDPOINT):
        problems.append(f"endpoint is '{endpoint or 'unset'}', expected the DagsHub S3 endpoint")

    _check(
        report,
        "dvc_remote_config",
        not problems,
        f"DVC remote '{DVC_REMOTE_NAME}' configured for DagsHub ({DVC_REMOTE_URL})",
        f"DVC remote '{DVC_REMOTE_NAME}' is misconfigured: {'; '.join(problems)}",
        detail=DVC_REMOTE_ENDPOINT if problems else "",
    )


def check_dvc_tracked_config_credentials(report: PreflightReport) -> None:
    """Credentials must never reach the Git-tracked DVC config."""
    keys = _config_credential_keys(DVC_TRACKED_CONFIG)
    _check(
        report,
        "dvc_tracked_config_clean",
        not keys,
        "Tracked .dvc/config contains no credential fields",
        f"SECURITY: tracked .dvc/config contains credential field(s): {', '.join(keys)} — "
        "move them to .dvc/config.local (Git-ignored) and rotate the exposed credentials",
    )


def check_dvc_local_config_safety(report: PreflightReport) -> None:
    """``.dvc/config.local`` may hold credentials; Git must never see it."""
    if not DVC_LOCAL_CONFIG.exists():
        _check(
            report,
            "dvc_local_config",
            False,
            "",
            "No .dvc/config.local — DagsHub credentials are not configured locally; "
            "remote synchronisation cannot be verified (see README: DVC remote access)",
            warning=True,
        )
        return

    ignored = _git_ignores(DVC_LOCAL_CONFIG)
    tracked = _git_tracks(DVC_LOCAL_CONFIG)

    if ignored is None or tracked is None:
        _check(
            report,
            "dvc_local_config",
            False,
            "",
            "Could not verify .dvc/config.local against Git — confirm manually that it is "
            "ignored and untracked",
            warning=True,
        )
        return

    problems: list[str] = []
    if tracked:
        problems.append("tracked by Git")
    if not ignored:
        problems.append("not Git-ignored")

    _check(
        report,
        "dvc_local_config",
        not problems,
        "Local DVC config .dvc/config.local exists and is Git-ignored and untracked",
        f"SECURITY: .dvc/config.local is {' and '.join(problems)} — remove it from Git "
        "and rotate the exposed credentials",
    )


def check_dvc_remote_sync(report: PreflightReport) -> None:
    """Read-only ``dvc status --remote`` probe.

    Never pushes, pulls, or reproduces anything.  Offline, unauthenticated,
    and out-of-sync states are warnings so the demo preflight stays usable
    without network access.
    """
    name = "dvc_remote_sync"
    if not _local_credentials_configured():
        _check(
            report,
            name,
            False,
            "",
            f"Remote '{DVC_REMOTE_NAME}' sync check skipped — no local credentials configured",
            warning=True,
        )
        return

    try:
        result = subprocess.run(
            [sys.executable, "-m", "dvc", "status", "--remote", DVC_REMOTE_NAME],
            capture_output=True,
            text=True,
            timeout=DVC_REMOTE_STATUS_TIMEOUT,
            cwd=str(_ROOT),
        )
    except subprocess.TimeoutExpired:
        _check(
            report,
            name,
            False,
            "",
            f"Remote '{DVC_REMOTE_NAME}' status timed out after "
            f"{DVC_REMOTE_STATUS_TIMEOUT}s — continuing offline",
            warning=True,
        )
        return
    except (FileNotFoundError, OSError):
        _check(
            report,
            name,
            False,
            "",
            "DVC command unavailable — remote synchronisation not verified",
            warning=True,
        )
        return

    stdout = result.stdout or ""
    if result.returncode != 0:
        _check(
            report,
            name,
            False,
            "",
            f"Remote '{DVC_REMOTE_NAME}' unreachable or authentication failed — continuing offline",
            warning=True,
            detail=_summarise(stdout, result.stderr or ""),
        )
        return

    if "in sync" in stdout.lower():
        _check(
            report,
            name,
            True,
            f"Cache and remote '{DVC_REMOTE_NAME}' are in sync",
            "",
        )
        return

    _check(
        report,
        name,
        False,
        "",
        f"Remote '{DVC_REMOTE_NAME}' reports unsynchronised objects — "
        f"run 'dvc push -r {DVC_REMOTE_NAME}' when publishing intentionally",
        warning=True,
        detail=_summarise(stdout, ""),
    )


def check_canonical_db(report: PreflightReport) -> None:
    ok = CANONICAL_DB.exists()
    _check(
        report,
        "canonical_db_exists",
        ok,
        f"Canonical MLflow DB found ({CANONICAL_DB.name})",
        f"Canonical DB missing: {CANONICAL_DB}",
    )
    if not ok:
        return

    h = hashlib.sha256()
    with open(CANONICAL_DB, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    prefix = h.hexdigest()[:8]
    ok2 = prefix == CANONICAL_DB_SHA256_PREFIX
    _check(
        report,
        "canonical_db_checksum",
        ok2,
        f"Canonical DB SHA-256 prefix matches ({prefix})",
        f"Canonical DB checksum mismatch: got {prefix}, expected {CANONICAL_DB_SHA256_PREFIX}",
    )


def check_champion_aliases(report: PreflightReport) -> None:
    try:
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{CANONICAL_DB}")
        client = mlflow.tracking.MlflowClient()
        for model_name, spec in REGISTERED_MODELS.items():
            rm = client.get_registered_model(model_name)
            champion_version = rm.aliases.get("champion", "")
            ok = str(champion_version) == spec["champion_version"]
            _check(
                report,
                f"champion_alias_{model_name.split('_')[-1]}",
                ok,
                f"{model_name}: champion=v{champion_version}",
                f"{model_name}: champion alias is v{champion_version}, expected v{spec['champion_version']}",
            )

            versions = client.search_model_versions(f"name='{model_name}'")
            ok2 = len(versions) >= spec["min_versions"]
            _check(
                report,
                f"version_count_{model_name.split('_')[-1]}",
                ok2,
                f"{model_name}: {len(versions)} registered versions",
                f"{model_name}: only {len(versions)} versions, expected >= {spec['min_versions']}",
            )
    except Exception as exc:
        _check(report, "champion_aliases", False, "", f"Failed to query MLflow registry: {exc}")


def check_model_artifacts(report: PreflightReport) -> None:
    missing = [f for f in REQUIRED_ARTIFACT_FILES if not f.exists()]
    ok = len(missing) == 0
    _check(
        report,
        "model_artifacts",
        ok,
        f"All {len(REQUIRED_ARTIFACT_FILES)} model artifacts present",
        f"Missing artifacts: {[str(f.relative_to(_ROOT)) for f in missing]}",
    )


def check_deployment_bundles(report: PreflightReport) -> None:
    missing = [f for f in REQUIRED_BUNDLE_FILES if not f.exists()]
    ok = len(missing) == 0
    missing_labels = []
    for f in missing:
        try:
            missing_labels.append(str(f.relative_to(_ROOT)))
        except ValueError:
            missing_labels.append(str(f))
    _check(
        report,
        "deployment_bundles",
        ok,
        "Deployment bundles present (deploy/bundles/)",
        f"Missing bundle items: {missing_labels} — run: make export-models",
        warning=True,
    )


def check_drift_summary(report: PreflightReport) -> None:
    path = _ROOT / "reports" / "drift_summary.json"
    ok = path.exists() and path.stat().st_size > 0
    _check(
        report,
        "drift_summary",
        ok,
        "Drift summary available (reports/drift_summary.json)",
        "Drift summary missing — run: make drift",
        warning=True,
    )


def check_config_files(report: PreflightReport) -> None:
    missing = [f for f in REQUIRED_CONFIG_FILES if not f.exists()]
    ok = len(missing) == 0
    _check(
        report,
        "config_files",
        ok,
        f"All {len(REQUIRED_CONFIG_FILES)} configuration files present",
        f"Missing config files: {[str(f.relative_to(_ROOT)) for f in missing]}",
    )


def check_running_containers(report: PreflightReport) -> None:
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return
        running = [
            name
            for name in result.stdout.splitlines()
            if any(c in name for c in PROJECT_CONTAINER_NAMES)
        ]
        ok = len(running) == 0
        _check(
            report,
            "no_stale_containers",
            ok,
            "No project containers currently running",
            f"Project containers already running: {running} — run: make demo-stop",
            warning=True,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass  # Docker not available; skip container check


# ── Main ───────────────────────────────────────────────────────────────────────
def run_preflight() -> PreflightReport:
    report = PreflightReport()
    check_project_root(report)
    check_python_version(report)
    check_venv(report)
    check_imports(report)
    check_docker(report)
    check_docker_compose(report)
    check_ports(report)
    check_dataset(report)
    check_dataset_checksum(report)
    check_dvc(report)
    check_dvc_s3_support(report)
    check_dvc_remote_config(report)
    check_dvc_tracked_config_credentials(report)
    check_dvc_local_config_safety(report)
    check_dvc_remote_sync(report)
    check_canonical_db(report)
    check_champion_aliases(report)
    check_model_artifacts(report)
    check_deployment_bundles(report)
    check_drift_summary(report)
    check_config_files(report)
    check_running_containers(report)
    return report


def print_human(report: PreflightReport) -> None:
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"

    print("\nOceanographic MLOps — Preflight Check")
    print("=" * 50)
    for c in report.checks:
        if c.passed:
            status = f"[{PASS}]"
        elif c.is_warning:
            status = f"[{WARN}]"
        else:
            status = f"[{FAIL}]"
        print(f"  {status}  {c.message}")
        if c.detail and not c.passed:
            print(f"           {c.detail}")
    print("=" * 50)
    print(
        f"  {report.passed_count}/{len(report.checks)} checks passed  |  "
        f"{len(report.failures)} failures  |  {len(report.warnings)} warnings"
    )
    if report.ok:
        print("  Result: READY FOR DEMONSTRATION")
    else:
        print("  Result: BLOCKING FAILURES — resolve before demo")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Oceanographic MLOps preflight checker")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args()

    report = run_preflight()

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print_human(report)

    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
