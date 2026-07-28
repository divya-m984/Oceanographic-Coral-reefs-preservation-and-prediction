"""
tests/test_docker.py — Structural and safety tests for M12 Docker containerisation.

These tests inspect Docker-related files (Dockerfiles, docker-compose.yml,
init scripts, .dockerignore) without running Docker.  No daemon, no build,
no network calls.

Coverage
--------
- All required Docker files exist.
- Dockerfiles use multi-stage builds with named stages.
- All runtime stages run as a non-root user.
- Health checks are present in all service Dockerfiles.
- Non-root user creation pattern is present.
- .dockerignore excludes secrets and large artefacts.
- .dockerignore does NOT exclude deploy/bundles (needed by Dockerfile.api).
- docker-compose.yml is valid YAML and contains required services.
- docker-compose.yml references correct Dockerfiles.
- Canonical DB (artifacts/mlruns.db) is mounted read-only in the compose file.
- API service uses CORALSENSE_MODEL_MODE=bundle (no MLflow runtime dependency).
- Drift service uses the 'drift' profile (not started by default).
- No --promote flag appears in any Docker file.
- No fit/fit_transform calls in Docker entry scripts.
- init_mlflow.sh never opens the canonical DB for writing.
- init_mlflow.sh rewrites paths before starting the server.
- Named volume declared for mlflow-runtime.
- reports/ is a bind mount (not a named volume) so the host user can write.
- No hardcoded host-absolute paths (/home/, /Users/, /root/) in compose file.
- docker-compose.yml health-checks depend on service_healthy condition.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
_COMPOSE = _ROOT / "docker-compose.yml"
_DOCKERIGNORE = _ROOT / ".dockerignore"
_DOCKERFILE_API = _ROOT / "Dockerfile.api"
_DOCKERFILE_DASHBOARD = _ROOT / "Dockerfile.dashboard"
_DOCKERFILE_MLFLOW = _ROOT / "Dockerfile.mlflow"
_INIT_MLFLOW = _ROOT / "docker" / "init_mlflow.sh"
_INIT_DRIFT = _ROOT / "docker" / "init_drift.sh"

_REQUIRED_DOCKER_FILES = [
    _COMPOSE,
    _DOCKERIGNORE,
    _DOCKERFILE_API,
    _DOCKERFILE_DASHBOARD,
    _DOCKERFILE_MLFLOW,
    _INIT_MLFLOW,
    _INIT_DRIFT,
]

_REQUIRED_SERVICES = {"mlflow", "api", "dashboard"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compose() -> dict:
    """Parsed docker-compose.yml."""
    with _COMPOSE.open() as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def dockerignore_text() -> str:
    return _DOCKERIGNORE.read_text()


@pytest.fixture(scope="module")
def dockerfile_api_text() -> str:
    return _DOCKERFILE_API.read_text()


@pytest.fixture(scope="module")
def dockerfile_dashboard_text() -> str:
    return _DOCKERFILE_DASHBOARD.read_text()


@pytest.fixture(scope="module")
def dockerfile_mlflow_text() -> str:
    return _DOCKERFILE_MLFLOW.read_text()


@pytest.fixture(scope="module")
def init_mlflow_text() -> str:
    return _INIT_MLFLOW.read_text()


@pytest.fixture(scope="module")
def init_drift_text() -> str:
    return _INIT_DRIFT.read_text()


# ---------------------------------------------------------------------------
# File existence
# ---------------------------------------------------------------------------


class TestFileExistence:
    def test_all_docker_files_exist(self):
        missing = [p for p in _REQUIRED_DOCKER_FILES if not p.exists()]
        assert not missing, f"Missing Docker files: {missing}"

    def test_export_champions_script_exists(self):
        assert (_ROOT / "scripts" / "export_champions.py").exists()

    def test_verify_bundle_script_exists(self):
        assert (_ROOT / "scripts" / "verify_deployment_bundle.py").exists()


# ---------------------------------------------------------------------------
# Dockerfile.api
# ---------------------------------------------------------------------------


class TestDockerfileApi:
    def test_multi_stage_build(self, dockerfile_api_text):
        stages = re.findall(r"^FROM\s+\S+\s+AS\s+(\w+)", dockerfile_api_text, re.MULTILINE)
        assert len(stages) >= 2, "Dockerfile.api must use multi-stage builds"

    def test_has_build_and_runtime_stages(self, dockerfile_api_text):
        assert "AS build" in dockerfile_api_text
        assert "AS runtime" in dockerfile_api_text

    def test_non_root_user(self, dockerfile_api_text):
        assert "USER coralsense" in dockerfile_api_text

    def test_non_root_user_created(self, dockerfile_api_text):
        assert "useradd" in dockerfile_api_text
        assert "coralsense" in dockerfile_api_text

    def test_health_check_present(self, dockerfile_api_text):
        assert "HEALTHCHECK" in dockerfile_api_text

    def test_health_check_uses_port_8000(self, dockerfile_api_text):
        assert "8000" in dockerfile_api_text

    def test_bundle_mode_env(self, dockerfile_api_text):
        assert "CORALSENSE_MODEL_MODE=bundle" in dockerfile_api_text

    def test_bundle_path_env(self, dockerfile_api_text):
        assert "CORALSENSE_BUNDLE_PATH" in dockerfile_api_text

    def test_bundles_copied(self, dockerfile_api_text):
        assert "deploy/bundles" in dockerfile_api_text

    def test_no_promote_flag(self, dockerfile_api_text):
        assert "--promote" not in dockerfile_api_text

    def test_uvicorn_command(self, dockerfile_api_text):
        assert "uvicorn" in dockerfile_api_text

    def test_exposes_port_8000(self, dockerfile_api_text):
        assert "EXPOSE 8000" in dockerfile_api_text

    def test_docker_scripts_copied(self, dockerfile_api_text):
        assert "docker/" in dockerfile_api_text


# ---------------------------------------------------------------------------
# Dockerfile.dashboard
# ---------------------------------------------------------------------------


class TestDockerfileDashboard:
    def test_multi_stage_build(self, dockerfile_dashboard_text):
        stages = re.findall(r"^FROM\s+\S+\s+AS\s+(\w+)", dockerfile_dashboard_text, re.MULTILINE)
        assert len(stages) >= 2

    def test_non_root_user(self, dockerfile_dashboard_text):
        assert "USER coralsense" in dockerfile_dashboard_text

    def test_health_check_present(self, dockerfile_dashboard_text):
        assert "HEALTHCHECK" in dockerfile_dashboard_text

    def test_health_check_uses_port_8501(self, dockerfile_dashboard_text):
        assert "8501" in dockerfile_dashboard_text

    def test_streamlit_headless_env(self, dockerfile_dashboard_text):
        assert "STREAMLIT_SERVER_HEADLESS" in dockerfile_dashboard_text

    def test_api_url_env(self, dockerfile_dashboard_text):
        assert "CORALSENSE_API_URL" in dockerfile_dashboard_text

    def test_api_url_points_to_api_service(self, dockerfile_dashboard_text):
        assert "http://api:8000" in dockerfile_dashboard_text

    def test_streamlit_command(self, dockerfile_dashboard_text):
        assert "streamlit" in dockerfile_dashboard_text

    def test_exposes_port_8501(self, dockerfile_dashboard_text):
        assert "EXPOSE 8501" in dockerfile_dashboard_text

    def test_no_promote_flag(self, dockerfile_dashboard_text):
        assert "--promote" not in dockerfile_dashboard_text


# ---------------------------------------------------------------------------
# Dockerfile.mlflow
# ---------------------------------------------------------------------------


class TestDockerfileMlflow:
    def test_file_exists(self, dockerfile_mlflow_text):
        assert len(dockerfile_mlflow_text) > 100

    def test_non_root_user(self, dockerfile_mlflow_text):
        assert "USER coralsense" in dockerfile_mlflow_text

    def test_health_check_present(self, dockerfile_mlflow_text):
        assert "HEALTHCHECK" in dockerfile_mlflow_text

    def test_health_check_uses_port_5000(self, dockerfile_mlflow_text):
        assert "5000" in dockerfile_mlflow_text

    def test_entrypoint_uses_init_script(self, dockerfile_mlflow_text):
        assert "init_mlflow.sh" in dockerfile_mlflow_text

    def test_exposes_port_5000(self, dockerfile_mlflow_text):
        assert "EXPOSE 5000" in dockerfile_mlflow_text

    def test_no_promote_flag(self, dockerfile_mlflow_text):
        assert "--promote" not in dockerfile_mlflow_text

    def test_mlflow_runtime_volume(self, dockerfile_mlflow_text):
        assert "mlflow-runtime" in dockerfile_mlflow_text


# ---------------------------------------------------------------------------
# init_mlflow.sh
# ---------------------------------------------------------------------------


class TestInitMlflow:
    def test_set_e_present(self, init_mlflow_text):
        assert "set -e" in init_mlflow_text

    def test_canonical_db_never_modified(self, init_mlflow_text):
        """The canonical DB should be read-only — only the runtime copy is written."""
        # We look for a RUNTIME_DB variable being used for the server, not CANONICAL_DB
        assert "RUNTIME_DB" in init_mlflow_text or "RUNTIME_DIR" in init_mlflow_text
        # The server must use the runtime path, not the canonical one
        assert "mlflow-runtime" in init_mlflow_text

    def test_path_rewrite_present(self, init_mlflow_text):
        """init_mlflow.sh must rewrite host-absolute paths."""
        assert "REPLACE" in init_mlflow_text or "replace" in init_mlflow_text

    def test_no_hardcoded_home_path(self, init_mlflow_text):
        """No hardcoded /home/ path may appear."""
        assert "/home/" not in init_mlflow_text

    def test_no_hardcoded_users_path(self, init_mlflow_text):
        assert "/Users/" not in init_mlflow_text

    def test_server_starts_on_runtime_db(self, init_mlflow_text):
        assert "mlflow server" in init_mlflow_text
        assert "mlflow-runtime" in init_mlflow_text

    def test_no_promote_flag(self, init_mlflow_text):
        assert "--promote" not in init_mlflow_text

    def test_no_fit_or_fit_transform(self, init_mlflow_text):
        assert ".fit(" not in init_mlflow_text
        assert "fit_transform" not in init_mlflow_text


# ---------------------------------------------------------------------------
# init_drift.sh
# ---------------------------------------------------------------------------


class TestInitDrift:
    def test_set_e_present(self, init_drift_text):
        assert "set -e" in init_drift_text

    def test_calls_run_drift_module(self, init_drift_text):
        assert "src.monitoring.run_drift" in init_drift_text

    def test_no_html_flag(self, init_drift_text):
        """Containers use JSON output only."""
        assert "--no-html" in init_drift_text

    def test_shift_scale_parameterised(self, init_drift_text):
        assert "DRIFT_SHIFT_SCALE" in init_drift_text

    def test_no_promote_flag(self, init_drift_text):
        assert "--promote" not in init_drift_text

    def test_no_fit_or_fit_transform(self, init_drift_text):
        assert ".fit(" not in init_drift_text
        assert "fit_transform" not in init_drift_text

    def test_synthetic_disclaimer_comment(self, init_drift_text):
        assert "SYNTHETIC" in init_drift_text.upper()


# ---------------------------------------------------------------------------
# .dockerignore
# ---------------------------------------------------------------------------


class TestDockIgnore:
    def test_git_excluded(self, dockerignore_text):
        assert ".git/" in dockerignore_text

    def test_venv_excluded(self, dockerignore_text):
        assert ".venv/" in dockerignore_text

    def test_env_secrets_excluded(self, dockerignore_text):
        assert ".env\n" in dockerignore_text or ".env\r" in dockerignore_text

    def test_env_example_not_excluded(self, dockerignore_text):
        assert "!.env.example" in dockerignore_text

    def test_pycache_excluded(self, dockerignore_text):
        assert "__pycache__/" in dockerignore_text

    def test_dvc_cache_excluded(self, dockerignore_text):
        assert ".dvc/cache/" in dockerignore_text

    def test_raw_data_excluded(self, dockerignore_text):
        assert "data/raw/" in dockerignore_text

    def test_processed_data_excluded(self, dockerignore_text):
        assert "data/processed/" in dockerignore_text

    def test_notebooks_excluded(self, dockerignore_text):
        assert "notebooks/" in dockerignore_text

    def test_deploy_bundles_not_excluded(self, dockerignore_text):
        """deploy/bundles/ must NOT be excluded — Dockerfile.api copies it."""
        lines = [ln.strip() for ln in dockerignore_text.splitlines()]
        exclusion_lines = [
            ln for ln in lines if ln in ("deploy/", "deploy/bundles/", "deploy/bundles")
        ]
        assert not exclusion_lines, (
            f"deploy/bundles must not be in .dockerignore, found: {exclusion_lines}"
        )

    def test_mlflow_runtime_db_excluded(self, dockerignore_text):
        assert "mlflow.db" in dockerignore_text


# ---------------------------------------------------------------------------
# docker-compose.yml structure
# ---------------------------------------------------------------------------


class TestComposeStructure:
    def test_compose_is_valid_yaml(self, compose):
        assert isinstance(compose, dict)

    def test_services_key_present(self, compose):
        assert "services" in compose

    def test_required_services_present(self, compose):
        services = set(compose["services"].keys())
        missing = _REQUIRED_SERVICES - services
        assert not missing, f"Missing services: {missing}"

    def test_drift_service_present(self, compose):
        assert "drift" in compose["services"]

    def test_named_volumes_declared(self, compose):
        volumes = compose.get("volumes", {})
        assert "mlflow-runtime" in volumes
        # reports/ is a bind mount so the host user can write drift reports;
        # it is NOT a named volume
        assert "reports" not in volumes

    def test_reports_is_bind_mount_in_dashboard(self, compose):
        """reports must be a bind mount so the host user (running drift) can write."""
        volumes = compose["services"]["dashboard"].get("volumes", [])
        reports_vol = next((v for v in volumes if "reports" in str(v)), None)
        assert reports_vol is not None, "dashboard must mount reports/"
        assert "reports" not in compose.get("volumes", {}), "reports must not be a named volume"

    def test_mlflow_service_references_correct_dockerfile(self, compose):
        build = compose["services"]["mlflow"].get("build", {})
        assert build.get("dockerfile") == "Dockerfile.mlflow"

    def test_api_service_references_correct_dockerfile(self, compose):
        build = compose["services"]["api"].get("build", {})
        assert build.get("dockerfile") == "Dockerfile.api"

    def test_dashboard_service_references_correct_dockerfile(self, compose):
        build = compose["services"]["dashboard"].get("build", {})
        assert build.get("dockerfile") == "Dockerfile.dashboard"

    def test_drift_service_has_drift_profile(self, compose):
        profiles = compose["services"]["drift"].get("profiles", [])
        assert "drift" in profiles, "drift service must have 'drift' profile"

    def test_drift_service_does_not_restart(self, compose):
        restart = compose["services"]["drift"].get("restart", "no")
        assert restart == "no", "drift is one-shot; it must not restart"


# ---------------------------------------------------------------------------
# docker-compose.yml safety
# ---------------------------------------------------------------------------


class TestComposeSafety:
    def test_canonical_db_mounted_read_only(self, compose):
        """artifacts/ volume in the mlflow service must be read-only."""
        mlflow_volumes = compose["services"]["mlflow"].get("volumes", [])
        artifacts_vol = next((v for v in mlflow_volumes if "artifacts" in str(v)), None)
        assert artifacts_vol is not None, "mlflow service must mount artifacts/"
        assert ":ro" in str(artifacts_vol), (
            f"artifacts/ must be mounted read-only (:ro), got: {artifacts_vol}"
        )

    def test_api_bundle_mode_env(self, compose):
        env = compose["services"]["api"].get("environment", {})
        if isinstance(env, list):
            env_dict = {}
            for item in env:
                if "=" in item:
                    k, v = item.split("=", 1)
                    env_dict[k] = v
        else:
            env_dict = env
        assert env_dict.get("CORALSENSE_MODEL_MODE") == "bundle"

    def test_no_promote_flag_in_compose(self):
        compose_text = _COMPOSE.read_text()
        assert "--promote" not in compose_text

    def test_no_hardcoded_home_path_in_compose(self):
        compose_text = _COMPOSE.read_text()
        assert "/home/" not in compose_text

    def test_no_hardcoded_users_path_in_compose(self):
        compose_text = _COMPOSE.read_text()
        assert "/Users/" not in compose_text

    def test_no_hardcoded_root_path_in_compose(self):
        # /root/ as a home dir would be a hardcoded path; mlflow-runtime is OK
        home_root_refs = re.findall(r"/root/[^\s\"']+", _COMPOSE.read_text())
        assert not home_root_refs, f"Hardcoded /root/ paths: {home_root_refs}"

    def test_dashboard_depends_on_api_healthy(self, compose):
        depends = compose["services"]["dashboard"].get("depends_on", {})
        if isinstance(depends, list):
            assert "api" in depends
        else:
            assert "api" in depends
            condition = depends["api"].get("condition", "")
            assert condition == "service_healthy", (
                f"dashboard must wait for api to be service_healthy, got: {condition}"
            )

    def test_api_not_depend_on_mlflow(self, compose):
        """API uses bundle mode — it must NOT depend on the mlflow service at startup."""
        depends = compose["services"]["api"].get("depends_on", {})
        if isinstance(depends, dict):
            deps = set(depends.keys())
        elif isinstance(depends, list):
            deps = set(depends)
        else:
            deps = set()
        assert "mlflow" not in deps, "API must not depend on MLflow at startup in bundle mode"

    def test_data_raw_mounted_read_only_in_dashboard(self, compose):
        volumes = compose["services"]["dashboard"].get("volumes", [])
        raw_vol = next((v for v in volumes if "data/raw" in str(v)), None)
        assert raw_vol is not None, "dashboard must mount data/raw/"
        assert ":ro" in str(raw_vol), f"data/raw must be :ro in dashboard, got: {raw_vol}"


# ---------------------------------------------------------------------------
# Port configuration
# ---------------------------------------------------------------------------


class TestComposeports:
    def test_mlflow_port_configurable(self):
        compose_text = _COMPOSE.read_text()
        # Port should use env var override
        assert "MLFLOW_PORT" in compose_text or "5000:5000" in compose_text

    def test_api_port_exposed(self, compose):
        ports = compose["services"]["api"].get("ports", [])
        port_strs = [str(p) for p in ports]
        assert any("8000" in p for p in port_strs)

    def test_dashboard_port_exposed(self, compose):
        ports = compose["services"]["dashboard"].get("ports", [])
        port_strs = [str(p) for p in ports]
        assert any("8501" in p for p in port_strs)

    def test_mlflow_port_exposed(self, compose):
        ports = compose["services"]["mlflow"].get("ports", [])
        port_strs = [str(p) for p in ports]
        assert any("5000" in p for p in port_strs)

    def test_mlflow_port_localhost_only(self):
        """MLflow must bind to 127.0.0.1 on the host, not 0.0.0.0."""
        compose_text = _COMPOSE.read_text()
        assert "127.0.0.1:" in compose_text, (
            "MLflow port mapping must use 127.0.0.1 to avoid exposing on all interfaces"
        )

    def test_mlflow_port_not_exposed_on_all_interfaces(self, compose):
        """Port mapping must not resolve to 0.0.0.0 (bare port or missing host IP)."""
        ports = compose["services"]["mlflow"].get("ports", [])
        for port_spec in ports:
            spec = str(port_spec)
            if "5000" in spec:
                assert spec.startswith("127.0.0.1:"), (
                    f"MLflow port must be bound to 127.0.0.1, got: {spec}"
                )


# ---------------------------------------------------------------------------
# MLflow launcher script
# ---------------------------------------------------------------------------

_START_MLFLOW = _ROOT / "scripts" / "start_mlflow.sh"


class TestStartMlflowScript:
    @pytest.fixture(autouse=True)
    def _load_script(self):
        self.text = _START_MLFLOW.read_text()

    def test_script_exists(self):
        assert _START_MLFLOW.exists()

    def test_script_is_executable_or_invoked_via_bash(self):
        """Either chmod +x or invoked via 'bash scripts/...'."""
        makefile = (_ROOT / "Makefile").read_text()
        assert "bash scripts/start_mlflow.sh" in makefile

    def test_start_action(self):
        assert "start)" in self.text

    def test_stop_action(self):
        assert "stop)" in self.text

    def test_status_action(self):
        assert "status)" in self.text

    def test_logs_action(self):
        assert "logs)" in self.text

    def test_uses_docker_compose(self):
        assert "docker compose" in self.text

    def test_targets_mlflow_service_only(self):
        """Launcher must only start the mlflow service, not all services."""
        assert "up -d mlflow" in self.text

    def test_health_wait_loop(self):
        assert "127.0.0.1:5000" in self.text

    def test_localhost_url_advertised(self):
        assert "http://127.0.0.1:5000" in self.text

    def test_uses_canonical_db_path(self):
        """Launcher relies on docker-compose.yml which mounts artifacts/mlruns.db."""
        compose_text = _COMPOSE.read_text()
        assert "artifacts" in compose_text
        assert "mlruns.db" in compose_text or "mlruns" in compose_text
