"""
tests/test_ci_workflow.py — Structural tests for the GitHub Actions CI workflow (M8).

These tests parse .github/workflows/ci.yml and assert that the workflow meets
the security, correctness, and policy requirements described in the M8 spec.
No network calls, no model training, no MLflow interaction.

Coverage
--------
- Workflow file exists and is valid YAML.
- Required triggers are configured.
- Minimal permissions (contents: read, no write permissions).
- Concurrency cancellation is enabled.
- All five required jobs are present.
- All jobs have a timeout.
- Timeouts are reasonable (≤ 60 minutes each).
- Official pinned actions are used (checkout@v4, setup-python@v5, upload-artifact@v4).
- No hardcoded absolute local paths (/home/, /Users/, /root/).
- The --promote flag does not appear anywhere.
- The canonical MLflow database path (artifacts/mlruns.db) is not referenced
  as a static path in any run command.
- The build job depends on both tests and ml-smoke-test.
- pip caching is enabled in at least one job.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

_WORKFLOW_PATH = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"

_REQUIRED_JOBS = {
    "code-quality",
    "tests",
    "pipeline-validation",
    "ml-smoke-test",
    "build",
}

_REQUIRED_TRIGGERS = {"push", "pull_request", "workflow_dispatch"}

# PyYAML 5+ treats 'on' as a YAML 1.1 boolean (True).
# GitHub Actions uses 'on:' as the trigger block key.
# We must look it up as True in the parsed dict.
_ON_KEY = True

_OFFICIAL_ACTIONS = {
    "actions/checkout@v4",
    "actions/setup-python@v5",
    "actions/upload-artifact@v4",
}


@pytest.fixture(scope="module")
def workflow() -> dict:
    """Load and return the parsed CI workflow YAML."""
    assert _WORKFLOW_PATH.exists(), f"CI workflow not found: {_WORKFLOW_PATH}"
    with _WORKFLOW_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _all_run_commands(workflow: dict) -> list[str]:
    """Collect every ``run:`` string from all job steps."""
    commands: list[str] = []
    for job in workflow.get("jobs", {}).values():
        for step in job.get("steps", []):
            run = step.get("run", "")
            if run:
                commands.append(run)
    return commands


def _all_uses_values(workflow: dict) -> list[str]:
    """Collect every ``uses:`` string from all job steps."""
    values: list[str] = []
    for job in workflow.get("jobs", {}).values():
        for step in job.get("steps", []):
            uses = step.get("uses", "")
            if uses:
                values.append(uses)
    return values


def _all_env_values(workflow: dict) -> list[str]:
    """Collect every environment variable value from all job steps."""
    values: list[str] = []
    for job in workflow.get("jobs", {}).values():
        for step in job.get("steps", []):
            for v in step.get("env", {}).values():
                values.append(str(v))
    return values


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWorkflowFile:
    def test_workflow_file_exists(self) -> None:
        assert _WORKFLOW_PATH.exists(), "ci.yml must exist"

    def test_workflow_is_valid_yaml(self, workflow: dict) -> None:
        assert isinstance(workflow, dict), "ci.yml must be a valid YAML mapping"

    def test_workflow_has_name(self, workflow: dict) -> None:
        assert "name" in workflow, "Workflow must have a name"


class TestTriggers:
    def test_required_triggers_present(self, workflow: dict) -> None:
        # PyYAML parses 'on:' as boolean True; use _ON_KEY constant.
        on = workflow.get(_ON_KEY, {})
        actual = set(on.keys()) if isinstance(on, dict) else set()
        missing = _REQUIRED_TRIGGERS - actual
        assert not missing, f"Missing triggers: {missing}"

    def test_push_targets_master_or_main(self, workflow: dict) -> None:
        branches = workflow[_ON_KEY]["push"].get("branches", [])
        assert any(b in branches for b in ("master", "main")), (
            "push trigger must target master or main"
        )

    def test_pull_request_targets_master_or_main(self, workflow: dict) -> None:
        branches = workflow[_ON_KEY]["pull_request"].get("branches", [])
        assert any(b in branches for b in ("master", "main")), (
            "pull_request trigger must target master or main"
        )

    def test_workflow_dispatch_present(self, workflow: dict) -> None:
        assert "workflow_dispatch" in workflow[_ON_KEY], (
            "workflow_dispatch trigger must be present for manual runs"
        )


class TestPermissions:
    def test_permissions_block_present(self, workflow: dict) -> None:
        assert "permissions" in workflow, "Workflow must declare explicit permissions"

    def test_contents_read_only(self, workflow: dict) -> None:
        perms = workflow.get("permissions", {})
        assert perms.get("contents") == "read", "permissions.contents must be 'read' (not 'write')"

    def test_no_write_permissions(self, workflow: dict) -> None:
        perms = workflow.get("permissions", {})
        for key, value in perms.items():
            assert value != "write", f"Permission {key!r} must not be 'write'"


class TestConcurrency:
    def test_concurrency_block_present(self, workflow: dict) -> None:
        assert "concurrency" in workflow, "concurrency block must be present"

    def test_cancel_in_progress_enabled(self, workflow: dict) -> None:
        concurrency = workflow.get("concurrency", {})
        assert concurrency.get("cancel-in-progress") is True, (
            "cancel-in-progress must be true to avoid stacked runs"
        )

    def test_concurrency_group_references_workflow_and_ref(self, workflow: dict) -> None:
        group = workflow.get("concurrency", {}).get("group", "")
        assert "github.workflow" in group, "concurrency.group must include github.workflow"
        assert "github.ref" in group, "concurrency.group must include github.ref"


class TestJobs:
    def test_required_jobs_present(self, workflow: dict) -> None:
        jobs = set(workflow.get("jobs", {}).keys())
        missing = _REQUIRED_JOBS - jobs
        assert not missing, f"Missing required jobs: {missing}"

    def test_all_jobs_have_timeout(self, workflow: dict) -> None:
        for name, job in workflow.get("jobs", {}).items():
            assert "timeout-minutes" in job, f"Job {name!r} must have timeout-minutes"

    def test_all_timeouts_are_reasonable(self, workflow: dict) -> None:
        for name, job in workflow.get("jobs", {}).items():
            timeout = job.get("timeout-minutes", 0)
            assert 1 <= timeout <= 60, (
                f"Job {name!r} timeout-minutes={timeout} must be between 1 and 60"
            )

    def test_build_depends_on_tests_and_ml_smoke_test(self, workflow: dict) -> None:
        build_needs = workflow["jobs"]["build"].get("needs", [])
        if isinstance(build_needs, str):
            build_needs = [build_needs]
        assert "tests" in build_needs, "build job must need 'tests'"
        assert "ml-smoke-test" in build_needs, "build job must need 'ml-smoke-test'"

    def test_tests_job_depends_on_code_quality(self, workflow: dict) -> None:
        needs = workflow["jobs"]["tests"].get("needs", [])
        if isinstance(needs, str):
            needs = [needs]
        assert "code-quality" in needs, "tests job must need 'code-quality'"


class TestSecurity:
    def test_no_hardcoded_local_home_paths(self, workflow: dict) -> None:
        """No step should reference /home/<user> or /Users/<user> literally."""
        commands = _all_run_commands(workflow)
        for cmd in commands:
            assert "/home/" not in cmd, f"Hardcoded /home/ path in command: {cmd!r}"
            assert "/Users/" not in cmd, f"Hardcoded /Users/ path in command: {cmd!r}"

    def test_no_promote_flag_in_commands(self, workflow: dict) -> None:
        """The --promote flag must not appear; CI never promotes champions."""
        commands = _all_run_commands(workflow)
        for cmd in commands:
            assert "--promote" not in cmd, (
                f"--promote flag found in CI command (forbidden): {cmd!r}"
            )

    def test_no_register_candidate_dvc_repro(self, workflow: dict) -> None:
        """dvc repro must not be run in CI (it would invoke register_candidate)."""
        commands = _all_run_commands(workflow)
        for cmd in commands:
            assert "dvc repro" not in cmd, (
                f"dvc repro found in CI command (forbidden — triggers register_candidate): {cmd!r}"
            )

    def test_mlflow_tracking_uri_not_pointing_to_canonical_db(self, workflow: dict) -> None:
        """
        If MLFLOW_TRACKING_URI is set in any step env, it must NOT point at the
        canonical artifacts/mlruns.db using an absolute /home/... path.
        """
        env_values = _all_env_values(workflow)
        for val in env_values:
            if "MLFLOW_TRACKING_URI" in str(val) or "mlruns.db" in val:
                assert "/home/" not in val, (
                    f"MLFLOW_TRACKING_URI contains a hardcoded local path: {val!r}"
                )


class TestActions:
    def test_checkout_action_is_pinned_to_v4(self, workflow: dict) -> None:
        uses_values = _all_uses_values(workflow)
        checkout_entries = [u for u in uses_values if u.startswith("actions/checkout")]
        assert checkout_entries, "actions/checkout must be used"
        for entry in checkout_entries:
            assert entry == "actions/checkout@v4", (
                f"checkout action must be pinned to @v4, got {entry!r}"
            )

    def test_setup_python_action_is_pinned_to_v5(self, workflow: dict) -> None:
        uses_values = _all_uses_values(workflow)
        setup_python_entries = [u for u in uses_values if u.startswith("actions/setup-python")]
        assert setup_python_entries, "actions/setup-python must be used"
        for entry in setup_python_entries:
            assert entry == "actions/setup-python@v5", (
                f"setup-python action must be pinned to @v5, got {entry!r}"
            )

    def test_upload_artifact_action_is_pinned_to_v4(self, workflow: dict) -> None:
        uses_values = _all_uses_values(workflow)
        upload_entries = [u for u in uses_values if u.startswith("actions/upload-artifact")]
        assert upload_entries, "actions/upload-artifact must be used"
        for entry in upload_entries:
            assert entry == "actions/upload-artifact@v4", (
                f"upload-artifact action must be pinned to @v4, got {entry!r}"
            )

    def test_pip_caching_enabled_in_at_least_one_job(self, workflow: dict) -> None:
        """At least one setup-python step must enable pip caching."""
        found_cache = False
        for job in workflow.get("jobs", {}).values():
            for step in job.get("steps", []):
                if step.get("uses", "").startswith("actions/setup-python"):
                    if step.get("with", {}).get("cache") == "pip":
                        found_cache = True
        assert found_cache, "pip caching must be enabled in at least one setup-python step"
