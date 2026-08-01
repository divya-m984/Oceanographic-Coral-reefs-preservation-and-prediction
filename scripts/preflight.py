#!/usr/bin/env python3
"""
scripts/preflight.py — Oceanographic MLOps read-only preflight checker.

Verifies the project environment, data integrity, and registry state before
a classroom demonstration.  Never trains, registers, promotes, or modifies
any project artifact.

Usage:
    python scripts/preflight.py           # human-readable output
    python scripts/preflight.py --json    # machine-readable JSON to stdout

Exit codes:
    0  All checks passed (warnings may exist)
    1  One or more blocking failures detected
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
