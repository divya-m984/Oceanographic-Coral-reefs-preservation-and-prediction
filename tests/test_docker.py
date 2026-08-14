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


# ---------------------------------------------------------------------------
# Deployment hardening — cloud PORT, /ready health check, standalone assets,
# and per-Dockerfile build-context isolation.
#
# All structural and offline: no daemon, no build, no network, no credentials.
# ---------------------------------------------------------------------------

_IGNORE_API = _ROOT / "Dockerfile.api.dockerignore"
_IGNORE_DASHBOARD = _ROOT / "Dockerfile.dashboard.dockerignore"

# Files that must never reach any serving-image build context.
_FORBIDDEN_CONTEXT_PATHS = (
    ".env",
    ".dvc/",
    ".dvc/config.local",
)


@pytest.fixture(scope="module")
def ignore_api_text() -> str:
    return _IGNORE_API.read_text()


@pytest.fixture(scope="module")
def ignore_dashboard_text() -> str:
    return _IGNORE_DASHBOARD.read_text()


def _instructions(text: str) -> str:
    """Return only the effective lines of a Dockerfile/ignore file.

    Comments are prose about intent — asserting against them tests the wording,
    not the build.  These checks care about what Docker actually executes.
    """
    return "\n".join(
        ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")
    )


def _cmd_line(dockerfile_text: str) -> str:
    """Return the CMD instruction (joined across line continuations)."""
    joined = dockerfile_text.replace("\\\n", " ")
    for line in joined.splitlines():
        if line.startswith("CMD"):
            return line
    raise AssertionError("no CMD instruction found")


def _healthcheck_line(dockerfile_text: str) -> str:
    """Return the HEALTHCHECK instruction (joined across line continuations)."""
    joined = dockerfile_text.replace("\\\n", " ")
    for line in joined.splitlines():
        if line.startswith("HEALTHCHECK"):
            return line
    raise AssertionError("no HEALTHCHECK instruction found")


class TestApiCloudPort:
    """The API image binds 0.0.0.0 on ${PORT}, falling back to 8000."""

    def test_cmd_honours_injected_port(self, dockerfile_api_text):
        cmd = _cmd_line(dockerfile_api_text)
        assert "${PORT:-8000}" in cmd, f"API CMD must honour ${{PORT}}, got: {cmd}"

    def test_cmd_binds_all_interfaces(self, dockerfile_api_text):
        assert "--host 0.0.0.0" in _cmd_line(dockerfile_api_text)

    def test_cmd_is_shell_form_so_port_expands(self, dockerfile_api_text):
        """An exec-form CMD would pass the literal string '${PORT:-8000}'."""
        cmd = _cmd_line(dockerfile_api_text)
        assert "/bin/sh" in cmd and "-c" in cmd

    def test_local_default_port_declared(self, dockerfile_api_text):
        assert re.search(r"^\s*PORT=8000", dockerfile_api_text, re.MULTILINE)

    def test_dead_api_host_port_env_removed(self, dockerfile_api_text):
        """API_HOST/API_PORT were never read by the entrypoint."""
        assert not re.search(r"^\s*API_HOST=", dockerfile_api_text, re.MULTILINE)
        assert not re.search(r"^\s*API_PORT=", dockerfile_api_text, re.MULTILINE)

    def test_main_module_prefers_coralsense_port_then_port(self):
        source = (_ROOT / "src" / "api" / "main.py").read_text()
        assert 'os.getenv("CORALSENSE_PORT")' in source
        assert 'os.getenv("PORT")' in source


class TestDashboardCloudPort:
    """The dashboard image binds 0.0.0.0 on ${PORT}, falling back to 8501."""

    def test_cmd_honours_injected_port(self, dockerfile_dashboard_text):
        cmd = _cmd_line(dockerfile_dashboard_text)
        assert "${PORT:-8501}" in cmd, f"dashboard CMD must honour ${{PORT}}, got: {cmd}"

    def test_cmd_binds_all_interfaces(self, dockerfile_dashboard_text):
        assert "--server.address 0.0.0.0" in _cmd_line(dockerfile_dashboard_text)

    def test_cmd_is_shell_form_so_port_expands(self, dockerfile_dashboard_text):
        cmd = _cmd_line(dockerfile_dashboard_text)
        assert "/bin/sh" in cmd and "-c" in cmd

    def test_local_default_port_declared(self, dockerfile_dashboard_text):
        assert re.search(r"^\s*PORT=8501", dockerfile_dashboard_text, re.MULTILINE)

    def test_no_duplicate_streamlit_port_env(self, dockerfile_dashboard_text):
        """The CMD sets the port; a second STREAMLIT_SERVER_PORT would shadow it."""
        assert not re.search(r"^\s*STREAMLIT_SERVER_PORT=", dockerfile_dashboard_text, re.MULTILINE)


class TestApiReadinessHealthCheck:
    """The container is healthy only when inference is actually available."""

    def test_dockerfile_healthcheck_targets_ready(self, dockerfile_api_text):
        hc = _healthcheck_line(dockerfile_api_text)
        assert "/ready" in hc, f"API HEALTHCHECK must probe /ready, got: {hc}"
        assert "/health" not in hc

    def test_dockerfile_healthcheck_honours_port(self, dockerfile_api_text):
        assert "${PORT:-8000}" in _healthcheck_line(dockerfile_api_text)

    def test_compose_api_healthcheck_targets_ready(self, compose):
        test = " ".join(str(x) for x in compose["services"]["api"]["healthcheck"]["test"])
        assert "/ready" in test
        assert "/health" not in test

    def test_health_endpoint_still_served(self):
        """/health must remain as the process-liveness endpoint."""
        source = (_ROOT / "src" / "api" / "main.py").read_text()
        assert '@app.get("/health"' in source
        assert '"/ready"' in source

    def test_dashboard_healthcheck_unchanged_target(self, dockerfile_dashboard_text):
        """The dashboard has no models; it keeps Streamlit's own health path."""
        assert "_stcore/health" in _healthcheck_line(dockerfile_dashboard_text)


class TestDashboardStandaloneAssets:
    """Required demo assets are baked in, so the image needs no bind mount."""

    #: (path copied into the image, why it is required)
    _REQUIRED = (
        "data/raw/observations.csv",
        "models/evaluation_health.json",
        "models/evaluation_restoration.json",
    )

    @pytest.mark.parametrize("asset", _REQUIRED)
    def test_required_asset_copied(self, asset, dockerfile_dashboard_text):
        copies = [
            ln for ln in dockerfile_dashboard_text.splitlines() if ln.strip().startswith("COPY")
        ]
        assert any(asset in ln for ln in copies), f"{asset} must be COPYed into the image"

    def test_optional_drift_summary_copied(self, dockerfile_dashboard_text):
        assert "reports/drift_summary.json" in dockerfile_dashboard_text

    def test_streamlit_config_copied(self, dockerfile_dashboard_text):
        assert ".streamlit/" in dockerfile_dashboard_text

    def test_params_copied_for_config_module(self, dockerfile_dashboard_text):
        """Page 8 calls get_config(), which reads params.yaml."""
        assert "params.yaml" in dockerfile_dashboard_text

    def test_no_model_artifacts_in_dashboard_image(self, dockerfile_dashboard_text):
        """The dashboard predicts via the API only — never load a model here."""
        assert ".joblib" not in dockerfile_dashboard_text
        assert "deploy/bundles" not in dockerfile_dashboard_text

    def test_no_mlflow_database_in_dashboard_image(self, dockerfile_dashboard_text):
        assert "mlruns.db" not in dockerfile_dashboard_text
        assert "artifacts/" not in dockerfile_dashboard_text

    def test_dashboard_does_not_run_dvc_at_runtime(self, dockerfile_dashboard_text):
        """No DVC command may appear in any executed instruction."""
        instructions = _instructions(dockerfile_dashboard_text).lower()
        for forbidden in ("dvc pull", "dvc fetch", "dvc checkout", "dvc repro", "dagshub"):
            assert forbidden not in instructions

    def test_dashboard_source_never_invokes_dvc(self):
        """No dashboard module may shell out to DVC while serving."""
        offenders = []
        for path in sorted((_ROOT / "src" / "dashboard").rglob("*.py")):
            text = path.read_text()
            if "dvc" in text.lower() and ("subprocess" in text or "os.system" in text):
                offenders.append(path.name)
        assert not offenders, f"dashboard modules invoking DVC: {offenders}"

    def test_dashboard_does_not_generate_or_train(self, dockerfile_dashboard_text):
        instructions = _instructions(dockerfile_dashboard_text)
        for forbidden in ("generate_data", "train", "--promote", "fit_transform"):
            assert forbidden not in instructions

    def test_synthetic_disclaimer_preserved(self, dockerfile_dashboard_text):
        assert "SYNTHETIC-DATA DISCLAIMER" in dockerfile_dashboard_text


class TestApiBundleModeIsMlflowFree:
    """Bundle mode must stay deterministic and independent of MLflow."""

    def test_bundle_loader_does_not_import_mlflow(self):
        source = (_ROOT / "src" / "api" / "bundle_loader.py").read_text()
        assert "import mlflow" not in source
        assert "from mlflow" not in source

    def test_bundle_loader_verifies_checksums(self):
        source = (_ROOT / "src" / "api" / "bundle_loader.py").read_text()
        assert "sha256" in source
        assert "_verify_checksum" in source

    def test_model_loader_has_no_silent_fallback_to_registry(self):
        """A failed bundle load must NOT quietly become a registry load.

        Parsed structurally: the mode branch must be a plain if/else, never a
        try/except that swallows a bundle failure and retries the registry.
        """
        import ast

        source = (_ROOT / "src" / "api" / "model_loader.py").read_text()
        func = next(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef) and node.name == "_load_pipeline"
        )
        assert not [n for n in ast.walk(func) if isinstance(n, ast.Try)], (
            "_load_pipeline must not catch bundle failures and fall back to the registry"
        )
        constructed = {
            n.func.id
            for n in ast.walk(func)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        assert constructed == {"BundleInferencePipeline", "InferencePipeline"}

    def test_api_image_pins_bundle_mode(self, dockerfile_api_text):
        assert "CORALSENSE_MODEL_MODE=bundle" in dockerfile_api_text

    def test_compose_api_does_not_depend_on_mlflow(self, compose):
        assert "depends_on" not in compose["services"]["api"]


class TestServingBuildContexts:
    """Per-Dockerfile ignore files keep each serving context minimal and safe."""

    def test_ignore_files_exist(self):
        assert _IGNORE_API.exists()
        assert _IGNORE_DASHBOARD.exists()

    @pytest.mark.parametrize("name", ["api", "dashboard"])
    def test_context_denies_everything_by_default(
        self, name, ignore_api_text, ignore_dashboard_text
    ):
        text = ignore_api_text if name == "api" else ignore_dashboard_text
        lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
        assert "**" in lines, f"{name} context must start from a deny-all rule"

    @pytest.mark.parametrize("secret", _FORBIDDEN_CONTEXT_PATHS)
    def test_api_context_excludes_secrets(self, secret, ignore_api_text):
        assert secret in ignore_api_text
        assert f"!{secret}" not in ignore_api_text

    @pytest.mark.parametrize("secret", _FORBIDDEN_CONTEXT_PATHS)
    def test_dashboard_context_excludes_secrets(self, secret, ignore_dashboard_text):
        assert secret in ignore_dashboard_text
        assert f"!{secret}" not in ignore_dashboard_text

    @pytest.mark.parametrize(
        "never",
        ["mlruns", "artifacts/", ".venv", "notebooks", "tests", ".git"],
    )
    def test_serving_contexts_never_re_include_bulk_or_vcs(
        self, never, ignore_api_text, ignore_dashboard_text
    ):
        """Deny-all covers these; the point is that nothing negates them back in."""
        for text in (ignore_api_text, ignore_dashboard_text):
            assert f"!{never}" not in text

    def test_api_context_includes_only_expected_paths(self, ignore_api_text):
        included = {
            ln.strip().lstrip("!")
            for ln in ignore_api_text.splitlines()
            if ln.strip().startswith("!")
        }
        assert included == {
            "src/",
            "params.yaml",
            "pyproject.toml",
            "requirements-api.txt",
            "docker/",
            "deploy/bundles/",
        }

    def test_api_context_has_no_dataset_or_model_files(self, ignore_api_text):
        for absent in ("observations.csv", "evaluation_", ".joblib"):
            assert f"!{absent}" not in ignore_api_text

    def test_dashboard_context_re_includes_required_assets(self, ignore_dashboard_text):
        for asset in (
            "!data/raw/observations.csv",
            "!models/evaluation_health.json",
            "!models/evaluation_restoration.json",
            "!reports/drift_summary.json",
        ):
            assert asset in ignore_dashboard_text

    def test_dashboard_context_re_includes_parent_dirs_of_assets(self, ignore_dashboard_text):
        """Docker will not descend into a directory no rule re-includes."""
        for parent in ("!data/", "!data/raw/", "!models/", "!reports/"):
            assert parent in ignore_dashboard_text

    def test_dashboard_context_excludes_model_binaries(self, ignore_dashboard_text):
        assert "models/*.joblib" in ignore_dashboard_text
        assert "!models/best_model" not in ignore_dashboard_text

    def test_dashboard_context_excludes_bundle(self, ignore_dashboard_text):
        assert "!deploy" not in ignore_dashboard_text

    def test_root_dockerignore_serves_mlflow_build(self, dockerignore_text):
        """Dockerfile.mlflow has no specific ignore file, so the root file must
        still admit everything it copies — only docker/init_mlflow.sh.  It pins
        mlflow inline and copies no requirements manifest.  The canonical DB is
        bind-mounted, never baked."""
        mlflow_text = _DOCKERFILE_MLFLOW.read_text()
        copied = [ln for ln in _instructions(mlflow_text).splitlines() if ln.startswith("COPY")]
        assert copied, "Dockerfile.mlflow should copy at least the init script"
        assert all("artifacts" not in ln and "mlruns" not in ln for ln in copied), (
            f"mlflow image must not bake MLflow state, got: {copied}"
        )
        assert all("requirements" not in ln for ln in copied), (
            f"Dockerfile.mlflow pins mlflow inline; it must copy no manifest, got: {copied}"
        )
        rules = {ln.strip() for ln in _instructions(dockerignore_text).splitlines()}
        assert "!docker/" in rules

    def test_root_dockerignore_excludes_mlflow_bulk(self, dockerignore_text):
        """~367 MB of mlruns/ must not be shipped to any build context."""
        rules = {ln.strip() for ln in _instructions(dockerignore_text).splitlines()}
        assert {"artifacts/", "mlruns/"} <= rules
        assert "!artifacts/" not in rules
        assert "!mlruns/" not in rules

    def test_root_dockerignore_is_allow_list(self, dockerignore_text):
        rules = [ln.strip() for ln in _instructions(dockerignore_text).splitlines()]
        assert rules[0] == "**", "root context must start from a deny-all rule"

    def test_root_dockerignore_admits_every_copied_path(self, dockerignore_text):
        """Every path the serving Dockerfiles COPY must be re-included here,
        because the legacy (non-BuildKit) builder only reads this file."""
        rules = {ln.strip() for ln in _instructions(dockerignore_text).splitlines()}
        for required in (
            "!src/",
            "!params.yaml",
            "!pyproject.toml",
            "!requirements-api.txt",
            "!requirements-dashboard.txt",
            "!docker/",
            "!.streamlit/",
            "!deploy/bundles/",
            "!data/raw/observations.csv",
            "!models/evaluation_health.json",
            "!models/evaluation_restoration.json",
            "!reports/drift_summary.json",
        ):
            assert required in rules, f"root .dockerignore must re-include {required}"

    def test_root_dockerignore_excludes_secrets(self, dockerignore_text):
        rules = {ln.strip() for ln in _instructions(dockerignore_text).splitlines()}
        for secret in (".env", ".dvc/", ".dvc/config.local", "*.pem", "*.key"):
            assert secret in rules
            assert f"!{secret}" not in rules

    def test_root_dockerignore_excludes_tests_and_notebooks(self, dockerignore_text):
        rules = {ln.strip() for ln in _instructions(dockerignore_text).splitlines()}
        assert {"tests/", "notebooks/", ".venv/", "models/*.joblib"} <= rules


class TestDashboardApiUrlConfigurable:
    """CORALSENSE_API_URL stays the single, externally settable API address."""

    def test_client_reads_env_var(self):
        source = (_ROOT / "src" / "dashboard" / "api_client.py").read_text()
        assert 'os.getenv("CORALSENSE_API_URL"' in source

    def test_client_default_is_local(self):
        from src.dashboard.api_client import _DEFAULT_BASE_URL

        assert _DEFAULT_BASE_URL == "http://127.0.0.1:8000"

    def test_https_url_is_accepted_without_code_change(self, monkeypatch):
        from src.dashboard.api_client import APIClient

        monkeypatch.setenv("CORALSENSE_API_URL", "https://example-api.invalid")
        assert APIClient().base_url == "https://example-api.invalid"

    def test_trailing_slash_normalised(self, monkeypatch):
        from src.dashboard.api_client import APIClient

        monkeypatch.setenv("CORALSENSE_API_URL", "https://example-api.invalid/")
        assert APIClient().base_url == "https://example-api.invalid"

    def test_compose_sets_service_url(self, compose):
        env = compose["services"]["dashboard"]["environment"]
        assert env["CORALSENSE_API_URL"] == "http://api:8000"

    def test_no_hardcoded_hosting_provider_hostname(self):
        """No Render (or other provider) hostname may be baked into the repo."""
        targets = [
            _ROOT / "src" / "dashboard" / "api_client.py",
            _ROOT / "Dockerfile.dashboard",
            _ROOT / "docker-compose.yml",
        ]
        for path in targets:
            text = path.read_text().lower()
            assert "onrender.com" not in text
            assert "render.com" not in text


class TestComposeWiringStillValid:
    """Local Compose demonstration stack keeps working after hardening."""

    def test_api_and_dashboard_services_present(self, compose):
        assert {"api", "dashboard"} <= set(compose["services"])

    def test_api_publishes_container_port_8000(self, compose):
        assert any("8000" in str(p) for p in compose["services"]["api"]["ports"])

    def test_dashboard_publishes_container_port_8501(self, compose):
        assert any("8501" in str(p) for p in compose["services"]["dashboard"]["ports"])

    def test_dashboard_still_waits_for_healthy_api(self, compose):
        dep = compose["services"]["dashboard"]["depends_on"]
        assert dep["api"]["condition"] == "service_healthy"

    def test_api_keeps_bundle_mode_env(self, compose):
        env = compose["services"]["api"]["environment"]
        assert env["CORALSENSE_MODEL_MODE"] == "bundle"

    def test_no_port_env_var_leaks_into_compose_services(self, compose):
        """A global PORT would be applied to every service at once."""
        for name, svc in compose["services"].items():
            env = svc.get("environment") or {}
            assert "PORT" not in env, f"service {name} must not pin PORT"

    def test_compose_has_no_dvc_or_dagshub_configuration(self):
        text = _COMPOSE.read_text().lower()
        for forbidden in ("dagshub", "dvc pull", "dvc push", "access_key_id", "secret_access_key"):
            assert forbidden not in text

    def test_compose_mounts_no_credential_file(self, compose):
        for name, svc in compose["services"].items():
            for vol in svc.get("volumes", []):
                assert "config.local" not in str(vol), f"service {name} mounts a credential file"
                assert ".env" not in str(vol), f"service {name} mounts an env file"
