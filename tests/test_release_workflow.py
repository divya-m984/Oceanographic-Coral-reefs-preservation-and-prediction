"""
tests/test_release_workflow.py — Structural tests for the serving-image release
workflow (.github/workflows/release-images.yml).

These tests are strictly offline.  They parse the workflow YAML and assert the
release contract described in the deployment spec.  Nothing here contacts
GitHub, GHCR or DagsHub, requires DAGSHUB_TOKEN, pushes an image, performs a
live DVC operation, or touches Render.

Coverage
--------
Trigger and guards
  - manual (workflow_dispatch) only; no push/PR/schedule trigger
  - a release from any ref other than refs/heads/main fails explicitly
  - a missing DAGSHUB_TOKEN fails early, naming only that secret
  - single-flight concurrency that does not cancel a running release

Token and secret hygiene
  - least-privilege permissions; contents stays read-only
  - packages: write only where images are published
  - GITHUB_TOKEN publishes to GHCR; no PAT is referenced
  - DAGSHUB_TOKEN is the only externally configured secret
  - the token never becomes a build arg, build secret, label or image env var
  - the token is never echoed; .dvc/config.local is removed via an EXIT trap

Ordering and safety
  - dvc pull precedes champion export, which precedes bundle verification,
    which precedes both image builds
  - no dvc push, dvc repro, training, registration, promotion or rollback

Build
  - Docker Buildx is configured
  - both builds use the PATH context (``context: .``) and an explicit Dockerfile
  - linux/amd64 only
  - lowercase GHCR image paths
  - distinct BuildKit cache scopes per image
  - provenance and SBOM attestations are enabled

Release model
  - only an immutable full-SHA tag is published by the build job
  - the moving ``:main`` tag is advanced in a separate job that depends on the
    build+smoke job, and re-points a tag at an already-tested digest
  - API smoke covers /health, /ready, /model-info (both champions v1) and
    POST /predict/both; dashboard smoke covers /_stcore/health
  - both digests are captured as job outputs and reported

Out of scope for this workflow
  - no MLflow image, no drift container, no Render deployment
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Shared constants and fixtures
# ---------------------------------------------------------------------------

_WORKFLOW_PATH = (
    Path(__file__).resolve().parent.parent / ".github" / "workflows" / "release-images.yml"
)

# PyYAML parses the GitHub Actions trigger key ``on:`` as a YAML 1.1 boolean.
_ON_KEY = True

_API_IMAGE = "ghcr.io/divya-m984/oceanographic-api"
_DASHBOARD_IMAGE = "ghcr.io/divya-m984/oceanographic-dashboard"

# The build job publishes candidates; the promote job advances the moving tag.
_BUILD_JOB = "release"
_PROMOTE_JOB = "promote"
_GUARD_JOB = "guard"

# The only two secrets the release contract allows.  GITHUB_TOKEN is built in;
# DAGSHUB_TOKEN is the single manually configured repository Actions secret.
_ALLOWED_SECRETS = {"GITHUB_TOKEN", "DAGSHUB_TOKEN"}

# Minimum acceptable major version per action.  A minimum (not an exact pin)
# lets future security bumps through while still failing on a downgrade.
_MIN_ACTION_MAJORS = {
    "actions/checkout": 7,
    "actions/setup-python": 7,
    "docker/setup-buildx-action": 4,
    "docker/login-action": 4,
    "docker/metadata-action": 6,
    "docker/build-push-action": 7,
}

# Commands that would mutate tracked state or the canonical registry.  The
# release workflow is read / prepare / build / publish only.
_FORBIDDEN_COMMAND_PATTERNS = [
    r"\bdvc\s+push\b",
    r"\bdvc\s+repro\b",
    r"\bdvc\s+commit\b",
    r"\bdvc\s+add\b",
    r"src\.models\.train\b",
    r"src\.models\.run_register_candidate\b",
    r"src\.models\.promote\b",
    r"src\.models\.rollback\b",
    r"src\.models\.retrain\b",
    r"scripts/run_retraining\.py",
    r"--promote\b",
    r"\bgit\s+push\b",
]

_SECRET_REF = re.compile(r"secrets\.([A-Za-z_][A-Za-z0-9_-]*)")


@pytest.fixture(scope="module")
def workflow() -> dict:
    """Load and return the parsed release workflow YAML."""
    assert _WORKFLOW_PATH.exists(), f"Release workflow not found: {_WORKFLOW_PATH}"
    with _WORKFLOW_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def raw_text() -> str:
    """Raw workflow text, for assertions about literal expressions."""
    return _WORKFLOW_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _steps(workflow: dict, job: str) -> list[dict]:
    return workflow["jobs"][job].get("steps", [])


def _all_steps(workflow: dict) -> list[dict]:
    steps: list[dict] = []
    for job in workflow.get("jobs", {}).values():
        steps.extend(job.get("steps", []))
    return steps


def _all_runs(workflow: dict) -> list[str]:
    return [s["run"] for s in _all_steps(workflow) if s.get("run")]


def _all_uses(workflow: dict) -> list[str]:
    return [s["uses"] for s in _all_steps(workflow) if s.get("uses")]


def _build_steps(workflow: dict) -> dict[str, dict]:
    """Return ``{step id: step}`` for every docker/build-push-action step."""
    found = {}
    for step in _all_steps(workflow):
        if step.get("uses", "").startswith("docker/build-push-action@"):
            found[step.get("id", step.get("name", "?"))] = step
    return found


def _step_index(workflow: dict, job: str, needle: str) -> int:
    """Index of the first step in *job* whose ``run`` or ``uses`` contains *needle*."""
    for i, step in enumerate(_steps(workflow, job)):
        haystack = (
            f"{step.get('run', '')}\n{step.get('uses', '')}\n{yaml.safe_dump(step.get('with', {}))}"
        )
        if needle in haystack:
            return i
    raise AssertionError(f"No step in job '{job}' references {needle!r}")


# ---------------------------------------------------------------------------
# Existence and parsing
# ---------------------------------------------------------------------------


class TestWorkflowFile:
    def test_workflow_file_exists(self) -> None:
        assert _WORKFLOW_PATH.exists(), f"Missing {_WORKFLOW_PATH}"

    def test_workflow_is_valid_yaml(self, workflow: dict) -> None:
        assert isinstance(workflow, dict)

    def test_workflow_has_a_descriptive_name(self, workflow: dict) -> None:
        assert workflow.get("name"), "Workflow must declare a name"
        assert "release" in workflow["name"].lower()

    def test_workflow_does_not_collide_with_ci(self, workflow: dict) -> None:
        ci = _WORKFLOW_PATH.parent / "ci.yml"
        assert ci.exists(), "The existing CI workflow must remain in place"
        with ci.open(encoding="utf-8") as fh:
            assert yaml.safe_load(fh)["name"] != workflow["name"]

    def test_every_job_declares_a_timeout(self, workflow: dict) -> None:
        for name, job in workflow["jobs"].items():
            assert "timeout-minutes" in job, f"Job '{name}' has no timeout-minutes"
            assert 0 < job["timeout-minutes"] <= 120, f"Job '{name}' timeout is unreasonable"

    def test_expected_jobs_exist(self, workflow: dict) -> None:
        assert {_GUARD_JOB, _BUILD_JOB, _PROMOTE_JOB} <= set(workflow["jobs"])


# ---------------------------------------------------------------------------
# Trigger: manual only
# ---------------------------------------------------------------------------


class TestManualTriggerOnly:
    def test_workflow_dispatch_is_configured(self, workflow: dict) -> None:
        assert "workflow_dispatch" in workflow[_ON_KEY]

    def test_no_automatic_trigger_publishes_images(self, workflow: dict) -> None:
        triggers = set(workflow[_ON_KEY])
        assert triggers == {"workflow_dispatch"}, (
            f"Release must be manual only; found extra triggers: {triggers - {'workflow_dispatch'}}"
        )

    @pytest.mark.parametrize(
        "forbidden",
        ["push", "pull_request", "pull_request_target", "schedule", "release", "issue_comment"],
    )
    def test_forbidden_trigger_absent(self, workflow: dict, forbidden: str) -> None:
        assert forbidden not in workflow[_ON_KEY]

    def test_no_untrusted_fork_trigger(self, raw_text: str) -> None:
        # pull_request_target runs with repository secrets in scope; a fork PR
        # could then exfiltrate DAGSHUB_TOKEN.
        assert "pull_request_target" not in raw_text


# ---------------------------------------------------------------------------
# Main-branch guard
# ---------------------------------------------------------------------------


class TestMainBranchGuard:
    def test_guard_job_checks_the_ref(self, workflow: dict) -> None:
        runs = "\n".join(s.get("run", "") for s in _steps(workflow, _GUARD_JOB))
        assert "refs/heads/main" in runs, "No refs/heads/main guard found"

    def test_non_main_ref_fails_explicitly(self, workflow: dict) -> None:
        guard = next(
            s["run"] for s in _steps(workflow, _GUARD_JOB) if "refs/heads/main" in s.get("run", "")
        )
        assert "exit 1" in guard, "The non-main guard must fail the workflow, not warn"
        assert "::error::" in guard, "The non-main guard must emit a workflow error annotation"

    def test_every_other_job_is_gated_behind_the_guard(self, workflow: dict) -> None:
        for name, job in workflow["jobs"].items():
            if name == _GUARD_JOB:
                continue
            needs = job.get("needs", [])
            needs = [needs] if isinstance(needs, str) else needs
            assert _GUARD_JOB in needs, f"Job '{name}' does not depend on '{_GUARD_JOB}'"


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_concurrency_group_is_declared(self, workflow: dict) -> None:
        assert "concurrency" in workflow, "Two releases must not run simultaneously"
        assert workflow["concurrency"].get("group")

    def test_running_release_is_not_cancelled(self, workflow: dict) -> None:
        assert workflow["concurrency"].get("cancel-in-progress") is False, (
            "A release that is already pushing images must be allowed to finish"
        )

    def test_concurrency_group_is_not_per_ref(self, workflow: dict) -> None:
        # Releases only ever run from main, so the group must be global rather
        # than keyed on github.ref, or two dispatches could overlap.
        assert "github.ref" not in str(workflow["concurrency"]["group"])


# ---------------------------------------------------------------------------
# Permissions
# ---------------------------------------------------------------------------


class TestPermissions:
    def test_top_level_permissions_are_read_only(self, workflow: dict) -> None:
        perms = workflow.get("permissions")
        assert perms == {"contents": "read"}, f"Expected minimal top-level permissions, got {perms}"

    def test_contents_is_never_writable(self, workflow: dict) -> None:
        assert workflow["permissions"]["contents"] == "read"
        for name, job in workflow["jobs"].items():
            contents = (job.get("permissions") or {}).get("contents", "read")
            assert contents == "read", f"Job '{name}' requests contents: {contents}"

    def test_packages_write_is_present_where_images_are_published(self, workflow: dict) -> None:
        for job in (_BUILD_JOB, _PROMOTE_JOB):
            perms = workflow["jobs"][job].get("permissions", {})
            assert perms.get("packages") == "write", (
                f"Job '{job}' publishes to GHCR and needs packages: write"
            )

    def test_guard_job_does_not_hold_packages_write(self, workflow: dict) -> None:
        perms = workflow["jobs"][_GUARD_JOB].get("permissions", {})
        assert "packages" not in perms, "The guard job publishes nothing"

    def test_ci_success_gate_adds_only_actions_read(self, workflow: dict) -> None:
        perms = workflow["jobs"][_GUARD_JOB].get("permissions", {})
        # The same-SHA CI gate reads run conclusions through the Actions API.
        assert perms.get("actions") == "read"
        assert set(perms) == {"contents", "actions"}

    def test_no_job_requests_package_deletion_or_admin(self, workflow: dict) -> None:
        for name, job in workflow["jobs"].items():
            for key, value in (job.get("permissions") or {}).items():
                assert value != "admin", f"Job '{name}' requests admin on {key}"

    def test_no_unexpected_write_permission_anywhere(self, workflow: dict) -> None:
        allowed_writes = {"packages"}
        for name, job in workflow["jobs"].items():
            for key, value in (job.get("permissions") or {}).items():
                if value == "write":
                    assert key in allowed_writes, f"Job '{name}' requests write on {key}"


# ---------------------------------------------------------------------------
# Secret contract
# ---------------------------------------------------------------------------


class TestSecretContract:
    def test_only_allowed_secrets_are_referenced(self, raw_text: str) -> None:
        referenced = set(_SECRET_REF.findall(raw_text))
        assert referenced <= _ALLOWED_SECRETS, (
            f"Unexpected secret(s) referenced: {sorted(referenced - _ALLOWED_SECRETS)}"
        )

    def test_dagshub_token_is_the_only_external_secret(self, raw_text: str) -> None:
        referenced = set(_SECRET_REF.findall(raw_text))
        external = referenced - {"GITHUB_TOKEN"}
        assert external == {"DAGSHUB_TOKEN"}, f"Expected only DAGSHUB_TOKEN, got {sorted(external)}"

    def test_no_personal_access_token_is_required(self, raw_text: str) -> None:
        referenced = set(_SECRET_REF.findall(raw_text))
        for name in referenced:
            assert not re.search(r"\b(PAT|GH_PAT|GHCR_PAT|CR_PAT|PERSONAL)\b", name.upper()), (
                f"A personal access token secret is referenced: {name}"
            )

    def test_ghcr_login_uses_the_builtin_token(self, workflow: dict) -> None:
        logins = [
            s for s in _all_steps(workflow) if s.get("uses", "").startswith("docker/login-action@")
        ]
        assert logins, "No GHCR login step found"
        for step in logins:
            assert step["with"]["registry"] == "ghcr.io"
            assert step["with"]["password"] == "${{ secrets.GITHUB_TOKEN }}"

    def test_missing_dagshub_token_fails_early(self, workflow: dict) -> None:
        guard_runs = [s.get("run", "") for s in _steps(workflow, _GUARD_JOB)]
        check = next((r for r in guard_runs if "DAGSHUB_TOKEN" in r and "exit 1" in r), None)
        assert check is not None, "No early DAGSHUB_TOKEN presence check in the guard job"
        assert "::error::" in check

    def test_missing_secret_message_names_only_dagshub_token(self, workflow: dict) -> None:
        check = next(
            r
            for r in (s.get("run", "") for s in _steps(workflow, _GUARD_JOB))
            if "DAGSHUB_TOKEN" in r and "exit 1" in r
        )
        error_lines = [ln for ln in check.splitlines() if "::error::" in ln]
        assert error_lines
        for line in error_lines:
            assert "GITHUB_TOKEN" not in line, (
                "The failure message must name only the missing secret"
            )

    def test_token_is_never_echoed(self, workflow: dict) -> None:
        for run in _all_runs(workflow):
            for line in run.splitlines():
                stripped = line.strip()
                if not stripped.startswith(("echo", "printf", "cat")):
                    continue
                assert "DAGSHUB_TOKEN}" not in stripped and "$DAGSHUB_TOKEN" not in stripped, (
                    f"A command prints the token: {stripped}"
                )

    def test_token_is_not_a_docker_build_arg(self, workflow: dict) -> None:
        for step_id, step in _build_steps(workflow).items():
            with_block = yaml.safe_dump(step.get("with", {}))
            assert "DAGSHUB" not in with_block.upper(), (
                f"Build step '{step_id}' passes DagsHub material into the build"
            )
            assert "build-args" not in step.get("with", {}), (
                f"Build step '{step_id}' declares build-args; none are needed"
            )
            assert "secrets" not in step.get("with", {}), (
                f"Build step '{step_id}' declares build secrets; none are needed"
            )

    def test_token_is_not_baked_into_labels(self, workflow: dict) -> None:
        for step in _all_steps(workflow):
            labels = str(step.get("with", {}).get("labels", ""))
            assert "DAGSHUB" not in labels.upper()
            assert "secrets." not in labels

    def test_build_step_env_carries_no_secret(self, workflow: dict) -> None:
        for step_id, step in _build_steps(workflow).items():
            env_block = yaml.safe_dump(step.get("env", {}) or {})
            assert "secrets." not in env_block, f"Build step '{step_id}' exposes a secret via env"

    def test_workflow_and_job_level_env_carry_no_secret(self, workflow: dict) -> None:
        blocks = [("workflow", workflow.get("env", {}) or {})]
        for name, job in workflow["jobs"].items():
            blocks.append((name, job.get("env", {}) or {}))
        for where, block in blocks:
            assert "secrets." not in yaml.safe_dump(block), f"{where}-level env exposes a secret"

    def test_dagshub_token_is_scoped_to_individual_steps(self, workflow: dict) -> None:
        holders = [
            s.get("name", "?")
            for s in _all_steps(workflow)
            if "DAGSHUB_TOKEN" in yaml.safe_dump(s.get("env", {}) or {})
        ]
        assert holders, "DAGSHUB_TOKEN must be injected through a step env block"
        assert len(holders) <= 2, f"The token is exposed to too many steps: {holders}"

    def test_no_secret_is_uploaded_as_an_artifact(self, workflow: dict) -> None:
        uploads = [u for u in _all_uses(workflow) if "upload-artifact" in u]
        assert not uploads, "The release workflow must not upload artifacts"


# ---------------------------------------------------------------------------
# DVC restore and credential hygiene
# ---------------------------------------------------------------------------


class TestDvcRestore:
    def test_dvc_pull_uses_the_dagshub_remote(self, workflow: dict) -> None:
        runs = "\n".join(_all_runs(workflow))
        assert re.search(r"dvc\s+pull\s+-r\s+dagshub", runs), "No `dvc pull -r dagshub` step"

    def test_credentials_are_configured_locally_only(self, workflow: dict) -> None:
        runs = "\n".join(_all_runs(workflow))
        assert "dvc remote modify --local dagshub access_key_id" in runs
        assert "dvc remote modify --local dagshub secret_access_key" in runs

    def test_tracked_dvc_config_is_never_modified(self, workflow: dict) -> None:
        for run in _all_runs(workflow):
            assert not re.search(r"dvc\s+remote\s+modify(?!\s+--local)", run), (
                "dvc remote modify must always carry --local"
            )
            assert not re.search(r"dvc\s+remote\s+add\b", run)

    def test_credential_file_is_removed_by_a_trap(self, workflow: dict) -> None:
        step = next(s for s in _all_steps(workflow) if "dvc pull -r dagshub" in s.get("run", ""))
        run = step["run"]
        assert "rm -f .dvc/config.local" in run, "The credential file is not removed"
        assert "trap" in run, "Cleanup must run even when dvc pull fails"

    def test_credential_removal_is_asserted_afterwards(self, workflow: dict) -> None:
        runs = "\n".join(_all_runs(workflow))
        assert re.search(r"if\s+\[\s+-f\s+\.dvc/config\.local\s+\]", runs), (
            "No post-restore assertion that .dvc/config.local is gone"
        )

    def test_build_context_is_asserted_free_of_credential_files(self, workflow: dict) -> None:
        runs = "\n".join(_all_runs(workflow))
        assert ".dvc/config.local .env .netrc" in runs or all(
            f in runs for f in (".dvc/config.local", ".env", ".netrc")
        ), "No pre-build assertion that credential files are absent"

    def test_no_dvc_write_operation(self, workflow: dict) -> None:
        runs = "\n".join(_all_runs(workflow))
        for pattern in (r"\bdvc\s+push\b", r"\bdvc\s+repro\b", r"\bdvc\s+commit\b"):
            assert not re.search(pattern, runs), f"Forbidden command matched: {pattern}"


# ---------------------------------------------------------------------------
# Preparation ordering
# ---------------------------------------------------------------------------


class TestPreparationOrdering:
    def test_dvc_pull_precedes_champion_export(self, workflow: dict) -> None:
        pull = _step_index(workflow, _BUILD_JOB, "dvc pull -r dagshub")
        export = _step_index(workflow, _BUILD_JOB, "scripts/export_champions.py")
        assert pull < export, "export_champions.py needs DVC-tracked artifacts restored first"

    def test_export_precedes_bundle_verification(self, workflow: dict) -> None:
        export = _step_index(workflow, _BUILD_JOB, "scripts/export_champions.py")
        verify = _step_index(workflow, _BUILD_JOB, "scripts/verify_deployment_bundle.py")
        assert export < verify

    def test_bundle_verification_precedes_both_builds(self, workflow: dict) -> None:
        verify = _step_index(workflow, _BUILD_JOB, "scripts/verify_deployment_bundle.py")
        api = _step_index(workflow, _BUILD_JOB, "./Dockerfile.api")
        dashboard = _step_index(workflow, _BUILD_JOB, "./Dockerfile.dashboard")
        assert verify < api and verify < dashboard

    def test_preflight_runs_after_dvc_restore(self, workflow: dict) -> None:
        pull = _step_index(workflow, _BUILD_JOB, "dvc pull -r dagshub")
        preflight = _step_index(workflow, _BUILD_JOB, "scripts/preflight.py")
        assert pull < preflight, "preflight checks DVC-tracked artifacts"

    def test_preflight_failure_blocks_the_release(self, workflow: dict) -> None:
        step = next(
            s for s in _steps(workflow, _BUILD_JOB) if "scripts/preflight.py" in s.get("run", "")
        )
        # No `|| true`, no `continue-on-error`: a blocking preflight failure
        # (exit 1) must fail the job.
        assert "|| true" not in step["run"]
        assert step.get("continue-on-error") is not True

    def test_project_tooling_owns_the_invariants(self, raw_text: str) -> None:
        # The protected-artifact checksums live in scripts/preflight.py; the
        # workflow must not duplicate them.
        assert "a03cb3e9" not in raw_text, "Dataset checksum duplicated into the workflow"
        assert "b76a4015" not in raw_text, "Canonical DB checksum duplicated into the workflow"


# ---------------------------------------------------------------------------
# Forbidden state-mutating operations
# ---------------------------------------------------------------------------


class TestNoStateMutation:
    @pytest.mark.parametrize("pattern", _FORBIDDEN_COMMAND_PATTERNS)
    def test_forbidden_command_absent(self, workflow: dict, pattern: str) -> None:
        runs = "\n".join(_all_runs(workflow))
        assert not re.search(pattern, runs), f"Forbidden command present: {pattern}"

    def test_no_champion_alias_is_touched(self, workflow: dict) -> None:
        runs = "\n".join(_all_runs(workflow))
        assert "set_registered_model_alias" not in runs
        assert "--approve" not in runs

    def test_canonical_registry_is_not_written(self, workflow: dict) -> None:
        for run in _all_runs(workflow):
            assert "mlflow server" not in run
            assert "mlflow ui" not in run

    def test_no_hardcoded_local_path(self, raw_text: str) -> None:
        for bad in ("/home/", "/Users/", "/root/"):
            assert bad not in raw_text, f"Hardcoded local path {bad} in the workflow"


# ---------------------------------------------------------------------------
# Docker build configuration
# ---------------------------------------------------------------------------


class TestDockerBuild:
    def test_buildx_is_configured(self, workflow: dict) -> None:
        assert any("docker/setup-buildx-action@" in u for u in _all_uses(workflow))

    def test_two_images_are_built(self, workflow: dict) -> None:
        assert len(_build_steps(workflow)) == 2, "Exactly the API and dashboard images are built"

    def test_both_builds_use_the_path_context(self, workflow: dict) -> None:
        for step_id, step in _build_steps(workflow).items():
            assert step["with"]["context"] == ".", (
                f"Build '{step_id}' must use the checked-out workspace, not the default Git context "
                "(dvc pull and export_champions.py create files that are not in Git)"
            )

    def test_dockerfiles_are_explicit(self, workflow: dict) -> None:
        files = {s["with"]["file"] for s in _build_steps(workflow).values()}
        assert files == {"./Dockerfile.api", "./Dockerfile.dashboard"}

    def test_platform_is_amd64_only(self, workflow: dict) -> None:
        for step_id, step in _build_steps(workflow).items():
            assert step["with"]["platforms"] == "linux/amd64", (
                f"Build '{step_id}' must target linux/amd64 explicitly"
            )

    def test_no_qemu_or_multiarch_complexity(self, workflow: dict) -> None:
        assert not any("setup-qemu" in u for u in _all_uses(workflow))
        assert not any("linux/arm" in str(s.get("with", {})) for s in _all_steps(workflow))

    def test_both_builds_push(self, workflow: dict) -> None:
        for step_id, step in _build_steps(workflow).items():
            assert step["with"]["push"] is True, f"Build '{step_id}' does not push"

    def test_cache_scopes_are_distinct(self, workflow: dict) -> None:
        scopes = []
        for step in _build_steps(workflow).values():
            cache_to = str(step["with"].get("cache-to", ""))
            match = re.search(r"scope=([\w-]+)", cache_to)
            assert match, "Each build must declare an explicit cache scope"
            scopes.append(match.group(1))
        assert len(set(scopes)) == len(scopes), f"Cache scopes collide: {scopes}"

    def test_cache_from_matches_cache_to_scope(self, workflow: dict) -> None:
        for step_id, step in _build_steps(workflow).items():
            to_scope = re.search(r"scope=([\w-]+)", str(step["with"]["cache-to"])).group(1)
            from_scope = re.search(r"scope=([\w-]+)", str(step["with"]["cache-from"])).group(1)
            assert to_scope == from_scope, f"Build '{step_id}' reads a different cache scope"

    def test_provenance_and_sbom_are_enabled(self, workflow: dict) -> None:
        for step_id, step in _build_steps(workflow).items():
            assert step["with"].get("provenance"), (
                f"Build '{step_id}' has no provenance attestation"
            )
            assert step["with"].get("sbom") is True, f"Build '{step_id}' has no SBOM attestation"

    def test_no_external_signing_infrastructure(self, raw_text: str) -> None:
        assert "cosign" not in raw_text.lower(), "Signing is deliberately out of scope for now"

    def test_only_supported_official_actions_are_used(self, workflow: dict) -> None:
        for uses in _all_uses(workflow):
            ref, _, version = uses.partition("@")
            assert ref.startswith(("actions/", "docker/")), f"Non-official action: {uses}"
            minimum = _MIN_ACTION_MAJORS.get(ref)
            if minimum is None:
                continue
            major = int(re.match(r"v(\d+)", version).group(1))
            assert major >= minimum, f"{ref} is pinned below v{minimum}: {uses}"

    def test_python_312_is_used(self, workflow: dict) -> None:
        setups = [
            s for s in _all_steps(workflow) if s.get("uses", "").startswith("actions/setup-python@")
        ]
        assert setups, "No setup-python step"
        for step in setups:
            assert "3.12" in str(step["with"]["python-version"]) or "3.12" in str(
                workflow.get("env", {}).get("PYTHON_VERSION", "")
            )

    def test_runtime_manifests_are_owned_by_the_dockerfiles(self, workflow: dict) -> None:
        # The serving images own their dependency sets.  The runner installs the
        # full development environment (needed for DVC/MLflow/preflight); the
        # runtime manifests are installed inside the images and nowhere else.
        runs = "\n".join(_all_runs(workflow))
        assert "requirements-api" not in runs
        assert "requirements-dashboard" not in runs
        assert "pip install -r requirements.txt" in runs


# ---------------------------------------------------------------------------
# GHCR image names and tags
# ---------------------------------------------------------------------------


class TestGhcrNamingAndTags:
    def test_expected_image_paths(self, workflow: dict) -> None:
        env = workflow.get("env", {})
        assert env.get("API_IMAGE") == _API_IMAGE
        assert env.get("DASHBOARD_IMAGE") == _DASHBOARD_IMAGE

    @pytest.mark.parametrize("image", [_API_IMAGE, _DASHBOARD_IMAGE])
    def test_image_paths_are_lowercase(self, image: str) -> None:
        assert image == image.lower(), f"GHCR image path must be lowercase: {image}"

    def test_images_are_published_to_ghcr(self, raw_text: str) -> None:
        assert _API_IMAGE.startswith("ghcr.io/")
        assert _DASHBOARD_IMAGE.startswith("ghcr.io/")
        assert "ghcr.io" in raw_text

    def test_metadata_action_is_used_for_both_images(self, workflow: dict) -> None:
        metas = [
            s
            for s in _all_steps(workflow)
            if s.get("uses", "").startswith("docker/metadata-action@")
        ]
        assert len(metas) == 2
        images = {s["with"]["images"] for s in metas}
        assert images == {"${{ env.API_IMAGE }}", "${{ env.DASHBOARD_IMAGE }}"}

    def test_immutable_full_sha_tag_is_published(self, workflow: dict) -> None:
        metas = [
            s
            for s in _all_steps(workflow)
            if s.get("uses", "").startswith("docker/metadata-action@")
        ]
        for step in metas:
            tags = str(step["with"]["tags"])
            assert "type=sha" in tags, "No commit-derived tag"
            assert "format=long" in tags, "The commit tag must use the full SHA"

    def test_build_job_publishes_no_moving_tag(self, workflow: dict) -> None:
        metas = [
            s
            for s in _all_steps(workflow)
            if s.get("uses", "").startswith("docker/metadata-action@")
        ]
        for step in metas:
            tags = str(step["with"]["tags"])
            assert "type=raw" not in tags, "The moving tag must not be published before smoke tests"
            assert "type=ref" not in tags

    def test_latest_tag_is_not_published(self, raw_text: str) -> None:
        assert "type=raw,value=latest" not in raw_text
        assert ":latest" not in raw_text

    def test_oci_metadata_labels_are_set(self, workflow: dict) -> None:
        metas = [
            s
            for s in _all_steps(workflow)
            if s.get("uses", "").startswith("docker/metadata-action@")
        ]
        for step in metas:
            labels = str(step["with"]["labels"])
            assert "org.opencontainers.image.title" in labels
            assert "org.opencontainers.image.description" in labels

    def test_source_and_revision_labels_are_provided(self, workflow: dict) -> None:
        # docker/metadata-action emits org.opencontainers.image.source and
        # .revision automatically from the GitHub context, so they must not be
        # hand-written (which is how local paths leak into labels).
        metas = [
            s
            for s in _all_steps(workflow)
            if s.get("uses", "").startswith("docker/metadata-action@")
        ]
        assert metas
        for step in metas:
            labels = str(step["with"]["labels"])
            assert "org.opencontainers.image.source=" not in labels
            assert "org.opencontainers.image.revision=" not in labels

    def test_labels_leak_no_infrastructure_detail(self, workflow: dict) -> None:
        for step in _all_steps(workflow):
            labels = str(step.get("with", {}).get("labels", ""))
            for bad in ("sqlite:///", "mlruns.db", "dagshub.com", "/home/", "s3://"):
                assert bad not in labels, f"Label leaks {bad}"


# ---------------------------------------------------------------------------
# Smoke tests on the published candidates
# ---------------------------------------------------------------------------


class TestCandidateSmokeTests:
    def test_api_smoke_runs_the_published_digest(self, workflow: dict) -> None:
        runs = "\n".join(_all_runs(workflow))
        assert "${API_REF}" in runs or "$API_REF" in runs
        step = next(
            s for s in _all_steps(workflow) if "API_REF" in yaml.safe_dump(s.get("env", {}) or {})
        )
        assert "outputs.digest" in str(step["env"]["API_REF"]), (
            "The API smoke test must run the exact published digest"
        )

    def test_dashboard_smoke_runs_the_published_digest(self, workflow: dict) -> None:
        step = next(
            s
            for s in _all_steps(workflow)
            if "DASHBOARD_REF" in yaml.safe_dump(s.get("env", {}) or {})
        )
        assert "outputs.digest" in str(step["env"]["DASHBOARD_REF"])

    def test_api_smoke_covers_readiness(self, workflow: dict) -> None:
        runs = "\n".join(_all_runs(workflow))
        assert "/ready" in runs, "The API smoke test must check GET /ready"
        assert "/health" in runs, "The API smoke test must check GET /health"

    def test_api_smoke_covers_model_info_and_champion_versions(self, workflow: dict) -> None:
        runs = "\n".join(_all_runs(workflow))
        assert "/model-info" in runs
        assert "coralsense_reef_health" in runs
        assert "coralsense_restoration_suitability" in runs
        assert 'entry["version"] == "1"' in runs, "Both champions must be asserted at v1"

    def test_api_smoke_makes_a_prediction(self, workflow: dict) -> None:
        runs = "\n".join(_all_runs(workflow))
        assert "/predict/both" in runs
        assert '"region": "Gulf of Mannar"' in runs, "A schema-valid observation must be posted"
        assert '"bleaching_percentage"' in runs

    def test_api_smoke_scans_logs_for_loading_failures(self, workflow: dict) -> None:
        runs = "\n".join(_all_runs(workflow))
        for needle in (
            "ImportError",
            "ModuleNotFoundError",
            "UnpicklingError",
            "model unavailable",
        ):
            assert needle in runs, f"Log scan does not look for {needle}"

    def test_dashboard_smoke_checks_streamlit_health(self, workflow: dict) -> None:
        runs = "\n".join(_all_runs(workflow))
        assert "/_stcore/health" in runs

    def test_dashboard_reaches_the_api_over_the_env_var(self, workflow: dict) -> None:
        runs = "\n".join(_all_runs(workflow))
        assert "CORALSENSE_API_URL=http://coralsense-release-api:8000" in runs

    def test_containers_share_an_isolated_network(self, workflow: dict) -> None:
        runs = "\n".join(_all_runs(workflow))
        assert "docker network create coralsense-release" in runs
        assert runs.count("--network coralsense-release") >= 2

    def test_dashboard_smoke_renders_pages_headlessly(self, workflow: dict) -> None:
        runs = "\n".join(_all_runs(workflow))
        assert "AppTest" in runs, "The dashboard smoke test must render pages without a browser"
        assert "len(at.exception) == 0" in runs

    def test_smoke_failure_fails_the_job(self, workflow: dict) -> None:
        for step in _all_steps(workflow):
            name = step.get("name", "")
            if name.startswith("Smoke-test"):
                assert step.get("continue-on-error") is not True, f"'{name}' swallows failures"
                assert "set -euo pipefail" in step["run"]

    def test_temporary_containers_are_cleaned_up(self, workflow: dict) -> None:
        teardown = next(
            (s for s in _all_steps(workflow) if "docker rm -f" in s.get("run", "")), None
        )
        assert teardown is not None, "No teardown step for smoke-test containers"
        assert teardown.get("if") == "always()", "Teardown must run even when a smoke test fails"
        assert "docker network rm coralsense-release" in teardown["run"]


# ---------------------------------------------------------------------------
# Moving-tag promotion
# ---------------------------------------------------------------------------


class TestMovingTagPromotion:
    def test_promotion_is_a_separate_job(self, workflow: dict) -> None:
        assert _PROMOTE_JOB in workflow["jobs"]

    def test_promotion_depends_on_the_smoke_tested_build(self, workflow: dict) -> None:
        needs = workflow["jobs"][_PROMOTE_JOB]["needs"]
        needs = [needs] if isinstance(needs, str) else needs
        assert _BUILD_JOB in needs, "The moving tag must only advance after build + smoke succeed"

    def test_promotion_has_no_success_override(self, workflow: dict) -> None:
        job = workflow["jobs"][_PROMOTE_JOB]
        # `if: always()` would advance :main even for a failed candidate.
        assert "always()" not in str(job.get("if", "")), "A failed candidate must not move :main"
        for step in _steps(workflow, _PROMOTE_JOB):
            assert "always()" not in str(step.get("if", ""))
            assert step.get("continue-on-error") is not True

    def test_promotion_retags_a_digest_without_rebuilding(self, workflow: dict) -> None:
        runs = "\n".join(s.get("run", "") for s in _steps(workflow, _PROMOTE_JOB))
        assert "imagetools create" in runs, "The moving tag must re-point at an existing manifest"
        rebuilds = [
            s
            for s in _steps(workflow, _PROMOTE_JOB)
            if s.get("uses", "").startswith("docker/build-push-action@")
            or re.search(r"\bdocker\s+build\b", s.get("run", ""))
        ]
        assert not rebuilds, "The promote job must not rebuild anything"

    def test_promotion_targets_the_tested_digests(self, workflow: dict) -> None:
        runs = "\n".join(s.get("run", "") for s in _steps(workflow, _PROMOTE_JOB))
        assert "${API_IMAGE}@${DIGEST}" in runs
        assert "${DASHBOARD_IMAGE}@${DIGEST}" in runs
        envs = [str(s.get("env", {}).get("DIGEST", "")) for s in _steps(workflow, _PROMOTE_JOB)]
        assert any("needs.release.outputs.api-digest" in e for e in envs)
        assert any("needs.release.outputs.dashboard-digest" in e for e in envs)

    def test_moving_tag_resolution_is_verified(self, workflow: dict) -> None:
        runs = "\n".join(s.get("run", "") for s in _steps(workflow, _PROMOTE_JOB))
        assert "imagetools inspect" in runs, "The advanced tag must be read back"
        assert '!= "${DIGEST}"' in runs, (
            "The promote job must fail if :main does not resolve to the tested digest"
        )

    def test_moving_tag_is_main_not_latest(self, workflow: dict) -> None:
        runs = "\n".join(s.get("run", "") for s in _steps(workflow, _PROMOTE_JOB))
        assert "${API_IMAGE}:main" in runs
        assert "${DASHBOARD_IMAGE}:main" in runs
        assert ":latest" not in runs

    def test_no_destructive_package_operation(self, workflow: dict) -> None:
        runs = "\n".join(_all_runs(workflow))
        assert "imagetools rm" not in runs
        assert "delete-package" not in runs
        assert "packages/container" not in runs, "Package administration is a manual step"


# ---------------------------------------------------------------------------
# Digest reporting
# ---------------------------------------------------------------------------


class TestDigestReporting:
    def test_build_job_exports_both_digests(self, workflow: dict) -> None:
        outputs = workflow["jobs"][_BUILD_JOB].get("outputs", {})
        assert "api-digest" in outputs
        assert "dashboard-digest" in outputs
        assert "outputs.digest" in outputs["api-digest"]
        assert "outputs.digest" in outputs["dashboard-digest"]

    def test_build_job_exports_both_sha_tags(self, workflow: dict) -> None:
        outputs = workflow["jobs"][_BUILD_JOB].get("outputs", {})
        assert "api-tag" in outputs
        assert "dashboard-tag" in outputs

    def test_summary_reports_the_release_facts(self, workflow: dict) -> None:
        runs = "\n".join(_all_runs(workflow))
        assert "GITHUB_STEP_SUMMARY" in runs
        summary = "\n".join(r for r in _all_runs(workflow) if "GITHUB_STEP_SUMMARY" in r)
        for field in ("Release Git SHA", "API digest", "Dashboard digest", "SHA tag"):
            assert field in summary, f"The summary does not report: {field}"

    def test_summary_reports_whether_main_advanced(self, workflow: dict) -> None:
        summary = "\n".join(r for r in _all_runs(workflow) if "GITHUB_STEP_SUMMARY" in r)
        assert ":main" in summary and "tags advanced" in summary

    def test_summary_prints_no_secret(self, workflow: dict) -> None:
        summary = "\n".join(r for r in _all_runs(workflow) if "GITHUB_STEP_SUMMARY" in r)
        assert "DAGSHUB" not in summary.upper()
        assert "secrets." not in summary

    def test_ghcr_visibility_is_reported_as_a_manual_step(self, workflow: dict) -> None:
        summary = "\n".join(r for r in _all_runs(workflow) if "GITHUB_STEP_SUMMARY" in r)
        assert "visibility" in summary.lower()


# ---------------------------------------------------------------------------
# Explicit non-scope
# ---------------------------------------------------------------------------


class TestOutOfScope:
    def test_no_render_deployment(self, workflow: dict) -> None:
        runs = "\n".join(_all_runs(workflow))
        for bad in ("api.render.com", "render deploy", "RENDER_API_KEY", "RENDER_SERVICE_ID"):
            assert bad not in runs, f"Render deployment must not exist yet: {bad}"
        for uses in _all_uses(workflow):
            assert "render" not in uses.lower(), f"Render deployment action present: {uses}"

    def test_no_render_secret_is_referenced(self, raw_text: str) -> None:
        assert not any("RENDER" in s.upper() for s in _SECRET_REF.findall(raw_text))

    def test_mlflow_image_is_not_built(self, workflow: dict) -> None:
        files = {s["with"]["file"] for s in _build_steps(workflow).values()}
        assert "./Dockerfile.mlflow" not in files
        runs = "\n".join(_all_runs(workflow))
        assert "Dockerfile.mlflow" not in runs

    def test_drift_container_is_not_built_or_published(self, workflow: dict) -> None:
        runs = "\n".join(_all_runs(workflow))
        assert "--profile drift" not in runs
        assert "init_drift" not in runs
        assert "oceanographic-drift" not in runs

    def test_no_mlflow_service_is_deployed(self, workflow: dict) -> None:
        runs = "\n".join(_all_runs(workflow))
        assert "coralsense-mlflow" not in runs
        assert "MLFLOW_TRACKING_URI" not in runs

    def test_docker_compose_is_not_used_to_publish(self, workflow: dict) -> None:
        runs = "\n".join(_all_runs(workflow))
        assert "docker compose push" not in runs
        assert "docker compose up" not in runs


# ---------------------------------------------------------------------------
# The existing CI workflow is untouched by this feature
# ---------------------------------------------------------------------------


class TestExistingCiUnchanged:
    def test_ci_workflow_still_exists(self) -> None:
        assert (_WORKFLOW_PATH.parent / "ci.yml").exists()

    def test_ci_workflow_does_not_publish_images(self) -> None:
        ci = (_WORKFLOW_PATH.parent / "ci.yml").read_text(encoding="utf-8")
        assert "ghcr.io" not in ci, "Image publication belongs to the release workflow only"
        assert "build-push-action" not in ci

    def test_ci_workflow_requires_no_release_secret(self) -> None:
        ci = (_WORKFLOW_PATH.parent / "ci.yml").read_text(encoding="utf-8")
        assert "DAGSHUB_TOKEN" not in ci
