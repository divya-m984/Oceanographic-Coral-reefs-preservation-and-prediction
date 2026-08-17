"""
tests/test_runtime_requirements.py — Serving-image dependency separation.

The repository keeps three dependency manifests:

    requirements.txt            full MLOps/development environment — training,
                                MLflow, DVC, Evidently, shap, pandera, the
                                dashboard, pytest and ruff.  Used by CI and the
                                Makefile on the host.
    requirements-api.txt        public inference runtime (Dockerfile.api).
    requirements-dashboard.txt  Streamlit runtime (Dockerfile.dashboard).

These tests prove the separation structurally, without Docker and without a
network call:

- Each serving manifest contains exactly the distributions its image needs.
- Neither serving manifest contains a training / MLOps / development package.
- The static import closure of each serving entrypoint is fully covered by its
  manifest, so nothing needed is missing either.
- The one manifest entry NOT explained by a source import — xgboost, needed to
  unpickle the restoration champion — is justified against the actual bundle.
- No manifest carries an editable path, credential, VCS URL, alternate index,
  or host-absolute path.
- Each serving Dockerfile installs its own manifest and no other.

SYNTHETIC-DATA DISCLAIMER
All bundled champions are trained on synthetic data only.
"""

from __future__ import annotations

import ast
import importlib.util
import pickletools
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent

_REQ_FULL = _ROOT / "requirements.txt"
_REQ_API = _ROOT / "requirements-api.txt"
_REQ_DASHBOARD = _ROOT / "requirements-dashboard.txt"

_DOCKERFILE_API = _ROOT / "Dockerfile.api"
_DOCKERFILE_DASHBOARD = _ROOT / "Dockerfile.dashboard"

_BUNDLE_DIR = _ROOT / "deploy" / "bundles"


# ---------------------------------------------------------------------------
# Manifest parsing
# ---------------------------------------------------------------------------


def _normalise(name: str) -> str:
    """PEP 503 normalisation: lowercase, runs of -_. collapse to a single '-'."""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def _parse_requirements(path: Path) -> set[str]:
    """Return the set of normalised distribution names declared in *path*.

    Extras, version specifiers, environment markers and inline comments are
    stripped, so ``uvicorn[standard]>=0.30`` yields ``uvicorn``.
    """
    names: set[str] = set()
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        line = line.split(";", 1)[0]  # environment marker
        line = re.split(r"[<>=!~\[]", line, maxsplit=1)[0]
        if line.strip():
            names.add(_normalise(line))
    return names


@pytest.fixture(scope="module")
def api_reqs() -> set[str]:
    return _parse_requirements(_REQ_API)


@pytest.fixture(scope="module")
def dashboard_reqs() -> set[str]:
    return _parse_requirements(_REQ_DASHBOARD)


@pytest.fixture(scope="module")
def full_reqs() -> set[str]:
    return _parse_requirements(_REQ_FULL)


# ---------------------------------------------------------------------------
# Static import-closure analysis
# ---------------------------------------------------------------------------


def _imported_top_level_modules(path: Path) -> set[str]:
    """Every top-level module name imported by *path*, at any nesting depth.

    Function-body and ``try:``-guarded imports count: ``src/config.py`` imports
    ``dotenv`` inside a function, and it still has to be installed for that
    branch to work.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import — first-party, resolved elsewhere
                continue
            if node.module:
                modules.add(node.module.split(".")[0])
    return modules


def _module_to_path(dotted: str) -> Path | None:
    candidate = _ROOT / Path(*dotted.split("."))
    if candidate.with_suffix(".py").is_file():
        return candidate.with_suffix(".py")
    if (candidate / "__init__.py").is_file():
        return candidate / "__init__.py"
    return None


def _first_party_imports(path: Path) -> set[str]:
    """Dotted ``src.*`` module names imported by *path*.

    ``from src.dashboard import seascape, theme`` imports two *modules*, not two
    attributes of one.  Recording only ``src.dashboard`` would stop the walk at
    the package ``__init__`` and leave both submodules outside the closure —
    which is exactly how ``theme.py`` escaped the forbidden-import and manifest
    guards until the seascape was added.  So each imported name is also offered
    as a submodule, and kept when it resolves to a real file.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "src":
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if not node.level and node.module and node.module.split(".")[0] == "src":
                found.add(node.module)
                for alias in node.names:
                    submodule = f"{node.module}.{alias.name}"
                    if _module_to_path(submodule) is not None:
                        found.add(submodule)
    return found


def _closure(entrypoints: list[Path], skip: set[str]) -> tuple[set[str], set[Path]]:
    """Walk first-party imports from *entrypoints*, collecting third-party modules.

    *skip* names ``src.*`` modules that are deliberately excluded from the
    closure because they are unreachable in the deployed configuration.  Every
    entry in *skip* is separately proven unreachable by its own test.
    """
    seen: set[Path] = set()
    queue = list(entrypoints)
    third_party: set[str] = set()

    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        for mod in _imported_top_level_modules(path):
            if mod != "src":
                third_party.add(mod)
        for dotted in _first_party_imports(path):
            if dotted in skip:
                continue
            resolved = _module_to_path(dotted)
            if resolved is not None:
                queue.append(resolved)
    return third_party, seen


def _pickle_top_level_modules(path: Path) -> set[str]:
    """Top-level modules a pickle file will import when unpickled.

    Handles both class-reference forms: the protocol-2 ``GLOBAL`` opcode, which
    carries ``"module name"`` inline, and the protocol-4 ``STACK_GLOBAL``
    opcode, which pops two strings previously pushed onto the stack (possibly
    via the memo).  joblib writes protocol 4+, so ``STACK_GLOBAL`` is the form
    that actually appears in these bundles.
    """
    _STR_OPS = frozenset(
        {
            "SHORT_BINUNICODE",
            "BINUNICODE",
            "BINUNICODE8",
            "UNICODE",
            "STRING",
            "SHORT_BINSTRING",
            "BINSTRING",
        }
    )
    _PUT_OPS = frozenset({"PUT", "BINPUT", "LONG_BINPUT"})
    _GET_OPS = frozenset({"GET", "BINGET", "LONG_BINGET"})

    # STACK_GLOBAL consumes the two most recently pushed strings — module first,
    # then qualname — so only string pushes need tracking, not the full stack.
    recent: list[str] = []
    memo: dict[object, str] = {}
    last_string: str | None = None
    modules: set[str] = set()

    for op, arg, _pos in pickletools.genops(path.read_bytes()):
        name = op.name
        if name in _STR_OPS and isinstance(arg, str):
            recent.append(arg)
            last_string = arg
        elif name == "MEMOIZE":
            if last_string is not None:
                memo[len(memo)] = last_string
        elif name in _PUT_OPS:
            if last_string is not None:
                memo[arg] = last_string
        elif name in _GET_OPS:
            value = memo.get(arg)
            if value is not None:
                recent.append(value)
                last_string = value
        elif name == "STACK_GLOBAL":
            if len(recent) >= 2:
                modules.add(recent[-2].split(".")[0])
            del recent[-2:]
            last_string = None
        elif name == "GLOBAL" and arg:
            modules.add(arg.split(" ")[0].split(".")[0])
            last_string = None
        elif name in ("REDUCE", "BUILD", "NEWOBJ", "NEWOBJ_EX"):
            # A constructed object is not a string; stop attributing the last
            # literal to the memo so a later PUT cannot record a stale value.
            last_string = None

    # Tracking only string pushes occasionally mistakes a nearby dict key (e.g.
    # "penalty", "n_estimators") for the module slot.  Keep only names that
    # actually resolve to an importable top-level module, which removes that
    # noise without hiding a genuine dependency: a module the unpickler really
    # needs is, by definition, importable in the development environment.
    return {m for m in modules if _is_real_module(m)}


def _is_real_module(name: str) -> bool:
    """True if *name* resolves to an importable top-level module."""
    if not name.isidentifier():
        return False
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


# Python 3.12 standard library plus __future__ — never installed from PyPI.
_STDLIB = {
    "__future__",
    "ast",
    # collections.abc — the typing-only ABCs used across src/dashboard/viz/.
    "collections",
    "contextlib",
    "dataclasses",
    "datetime",
    "hashlib",
    # html.escape — hero copy flows into unsafe_allow_html.
    "html",
    "json",
    "logging",
    "math",
    "os",
    "pathlib",
    # random.Random — the seascape's seeded, deterministic reef.
    "random",
    "re",
    "sys",
    "typing",
    "urllib",
}

# Import name → distribution name, for the cases where they differ.
_MODULE_TO_DIST = {
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "dotenv": "python-dotenv",
    "plotly": "plotly",
}

# ``src.models.predict`` is imported by ``src/api/model_loader.py`` only in the
# registry-mode branch.  The serving image pins CORALSENSE_MODEL_MODE=bundle, so
# it is never imported there — proven by TestBundleModeExcludesRegistryPath.
_REGISTRY_ONLY = {"src.models.predict"}

_API_ENTRYPOINTS = [_ROOT / "src" / "api" / "main.py"]
_DASHBOARD_ENTRYPOINTS = [_ROOT / "src" / "dashboard" / "app.py"] + sorted(
    (_ROOT / "src" / "dashboard" / "views").glob("*.py")
)


# ---------------------------------------------------------------------------
# Manifests exist and stay distinct
# ---------------------------------------------------------------------------


class TestManifestsExist:
    def test_full_requirements_still_present(self):
        """requirements.txt remains the training/MLOps environment used by CI."""
        assert _REQ_FULL.is_file()

    @pytest.mark.parametrize("pkg", ["mlflow", "dvc", "evidently", "pandera", "shap", "pytest"])
    def test_full_requirements_still_carries_mlops_stack(self, pkg, full_reqs):
        assert pkg in full_reqs, f"requirements.txt must still provide {pkg} for the DVC pipeline"

    def test_api_manifest_exists(self):
        assert _REQ_API.is_file()

    def test_dashboard_manifest_exists(self):
        assert _REQ_DASHBOARD.is_file()

    def test_serving_manifests_are_smaller_than_full(self, api_reqs, dashboard_reqs, full_reqs):
        assert len(api_reqs) < len(full_reqs)
        assert len(dashboard_reqs) < len(full_reqs)

    def test_manifests_are_not_empty(self, api_reqs, dashboard_reqs):
        assert api_reqs
        assert dashboard_reqs


# ---------------------------------------------------------------------------
# API runtime manifest
# ---------------------------------------------------------------------------


class TestApiRuntimeManifest:
    @pytest.mark.parametrize(
        "forbidden",
        [
            "mlflow",
            "dvc",
            "dvc-s3",
            "evidently",
            "streamlit",
            "plotly",
            "shap",
            "pandera",
            "httpx",
            "pytest",
            "pytest-cov",
            "ruff",
            "jupyter",
            "notebook",
            "ipykernel",
            "matplotlib",
            "faker",
        ],
    )
    def test_excludes_non_serving_package(self, forbidden, api_reqs):
        assert forbidden not in api_reqs, (
            f"{forbidden} is not reachable from a bundle-mode prediction request "
            f"and must not be installed into the inference image"
        )

    def test_excludes_every_dvc_extra_spelling(self):
        text = _REQ_API.read_text()
        instructions = "\n".join(
            ln.split("#", 1)[0] for ln in text.splitlines() if not ln.strip().startswith("#")
        )
        for spelling in ("dvc[", "dvc-s3", "dvc_s3", "dagshub"):
            assert spelling not in instructions

    @pytest.mark.parametrize(
        "required,reason",
        [
            ("fastapi", "src/api/main.py builds the FastAPI app"),
            ("uvicorn", "Dockerfile.api CMD runs uvicorn"),
            ("pydantic", "src/api/schemas.py request/response models"),
            ("pandas", "src/api/main.py and bundle_loader.py build DataFrames"),
            ("numpy", "src/api/bundle_loader.py; pickled arrays in both payloads"),
            ("joblib", "src/api/bundle_loader.py loads payload/preprocessor"),
            ("scikit-learn", "check_is_fitted; both ColumnTransformers; health champion"),
            ("pyyaml", "src/config.py reads params.yaml at startup"),
            ("python-dotenv", "src/config._load_dotenv optional import"),
        ],
    )
    def test_includes_required_package(self, required, reason, api_reqs):
        assert required in api_reqs, f"{required} is required: {reason}"

    def test_provides_the_xgboost_module(self, api_reqs):
        """Either distribution installs the ``xgboost`` import module.

        ``xgboost-cpu`` is upstream's CPU-only build of the same package; it
        omits the nvidia-nccl-cu12 dependency (~400 MB of CUDA libraries) that
        the default wheel pulls in purely for multi-GPU training.
        """
        assert api_reqs & {"xgboost", "xgboost-cpu"}, (
            "the restoration champion payload unpickles xgboost classes"
        )

    def test_version_constraints_match_the_full_environment(self, api_reqs):
        """No serving package may float to a different major than requirements.txt."""
        full_pins = {}
        for raw in _REQ_FULL.read_text().splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line or line.startswith("-"):
                continue
            name = _normalise(re.split(r"[<>=!~\[]", line, maxsplit=1)[0])
            full_pins[name] = line[len(re.split(r"[<>=!~\[]", line, maxsplit=1)[0]) :].strip()

        for raw in _REQ_API.read_text().splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line or line.startswith("-"):
                continue
            name = _normalise(re.split(r"[<>=!~\[]", line, maxsplit=1)[0])
            spec = line[len(re.split(r"[<>=!~\[]", line, maxsplit=1)[0]) :].strip()
            if name in full_pins:
                assert spec == full_pins[name], (
                    f"{name} constraint '{spec}' differs from requirements.txt "
                    f"'{full_pins[name]}' — serving and development must resolve alike"
                )


# ---------------------------------------------------------------------------
# Dashboard runtime manifest
# ---------------------------------------------------------------------------


class TestDashboardRuntimeManifest:
    @pytest.mark.parametrize(
        "forbidden",
        [
            "mlflow",
            "dvc",
            "dvc-s3",
            "evidently",
            "xgboost",
            "xgboost-cpu",
            "scikit-learn",
            "joblib",
            "shap",
            "pandera",
            "scipy",
            "fastapi",
            "uvicorn",
            "pytest",
            "ruff",
        ],
    )
    def test_excludes_non_dashboard_package(self, forbidden, dashboard_reqs):
        assert forbidden not in dashboard_reqs, (
            f"{forbidden} is imported nowhere under src/dashboard/; the dashboard "
            f"loads no model and calls the API over CORALSENSE_API_URL"
        )

    def test_excludes_every_dvc_extra_spelling(self):
        text = _REQ_DASHBOARD.read_text()
        instructions = "\n".join(
            ln.split("#", 1)[0] for ln in text.splitlines() if not ln.strip().startswith("#")
        )
        for spelling in ("dvc[", "dvc-s3", "dvc_s3", "dagshub"):
            assert spelling not in instructions

    @pytest.mark.parametrize(
        "required,reason",
        [
            ("streamlit", "every dashboard module imports streamlit"),
            ("plotly", "pages 2, 3, 4, 6 and 8 import plotly.express"),
            ("pandas", "src/dashboard/data_loader.py reads observations.csv"),
            ("requests", "src/dashboard/api_client.py talks to the API"),
            ("pyyaml", "page 8 imports src.config, which parses params.yaml"),
            ("python-dotenv", "src/config._load_dotenv optional import"),
        ],
    )
    def test_includes_required_package(self, required, reason, dashboard_reqs):
        assert required in dashboard_reqs, f"{required} is required: {reason}"

    def test_plotly_stays_on_the_five_series(self):
        """The viz builders use plotly.graph_objects; plotly 6 is a major bump."""
        line = next(
            ln for ln in _REQ_DASHBOARD.read_text().splitlines() if ln.strip().startswith("plotly")
        )
        assert ">=5.24" in line
        assert "<6" in line

    def test_includes_the_leaflet_map_renderer(self, dashboard_reqs):
        """The Reef Map is Leaflet: folium is now a hard dashboard dependency.

        This replaces the former ``test_excludes_folium_which_is_imported_nowhere``.
        That test recorded a fact about the code — nothing imported folium — and
        that fact changed when ``src/dashboard/reefmap.py`` was written: the
        ocean-floor background is a GEBCO **WMS** layer, which folium consumes
        natively and no plotly map trace can.  The assertions below are pinned to
        the imports so the manifest cannot drift away from the code again.
        """
        assert "folium" in dashboard_reqs
        assert "streamlit-folium" in dashboard_reqs

        reefmap = (_ROOT / "src" / "dashboard" / "reefmap.py").read_text()
        assert "import folium" in reefmap
        page = (_ROOT / "src" / "dashboard" / "views" / "2_Reef_Map.py").read_text()
        assert "from streamlit_folium import st_folium" in page

    def test_reef_map_uses_no_street_basemap(self):
        """The bathymetry is the point; a street tile layer would undo it."""
        reefmap = (_ROOT / "src" / "dashboard" / "reefmap.py").read_text()
        assert "wms.gebco.net" in reefmap
        assert "tiles=None" in reefmap
        for street in ("open-street-map", "openstreetmap", "cartodb", "stamen"):
            assert street not in reefmap.lower(), f"{street} basemap must not be used"


# ---------------------------------------------------------------------------
# The manifests actually cover the code that runs
# ---------------------------------------------------------------------------


class TestManifestsCoverImportClosure:
    def test_api_closure_is_fully_covered(self, api_reqs):
        third_party, _ = _closure(_API_ENTRYPOINTS, skip=_REGISTRY_ONLY)
        missing = set()
        for mod in third_party - _STDLIB:
            dist = _MODULE_TO_DIST.get(mod, _normalise(mod))
            if dist not in api_reqs:
                missing.add(f"{mod} -> {dist}")
        assert not missing, f"requirements-api.txt is missing: {sorted(missing)}"

    def test_dashboard_closure_is_fully_covered(self, dashboard_reqs):
        third_party, _ = _closure(_DASHBOARD_ENTRYPOINTS, skip=set())
        missing = set()
        for mod in third_party - _STDLIB:
            dist = _MODULE_TO_DIST.get(mod, _normalise(mod))
            if dist not in dashboard_reqs:
                missing.add(f"{mod} -> {dist}")
        assert not missing, f"requirements-dashboard.txt is missing: {sorted(missing)}"

    def test_api_closure_reaches_the_bundle_loader(self):
        """Guards the skip-list: the closure must still cover the serving path."""
        _, files = _closure(_API_ENTRYPOINTS, skip=_REGISTRY_ONLY)
        names = {p.name for p in files}
        assert {"main.py", "model_loader.py", "schemas.py", "config.py"} <= names

    @pytest.mark.parametrize("forbidden", ["mlflow", "dvc", "evidently", "streamlit", "plotly"])
    def test_api_closure_imports_no_forbidden_module(self, forbidden):
        third_party, _ = _closure(_API_ENTRYPOINTS, skip=_REGISTRY_ONLY)
        assert forbidden not in third_party

    @pytest.mark.parametrize(
        "forbidden", ["mlflow", "dvc", "evidently", "xgboost", "sklearn", "joblib", "numpy"]
    )
    def test_dashboard_closure_imports_no_forbidden_module(self, forbidden):
        third_party, _ = _closure(_DASHBOARD_ENTRYPOINTS, skip=set())
        assert forbidden not in third_party


class TestBundleModeExcludesRegistryPath:
    """Justifies the one module skipped by the API import closure."""

    def test_registry_pipeline_import_is_guarded_by_mode(self):
        source = (_ROOT / "src" / "api" / "model_loader.py").read_text()
        assert 'if mode == "bundle":' in source
        bundle_branch, registry_branch = source.split('if mode == "bundle":', 1)[1].split(
            "else:", 1
        )
        assert "from src.api.bundle_loader import BundleInferencePipeline" in bundle_branch
        assert "from src.models.predict import InferencePipeline" in registry_branch
        assert "from src.models.predict" not in bundle_branch

    def test_registry_pipeline_is_the_module_that_needs_mlflow(self):
        """Confirms the skip is load-bearing rather than hiding a real dependency.

        ``src/models/predict.py`` reaches mlflow through ``src.models.registry``,
        so importing it in bundle mode would drag the registry stack into the
        serving image.
        """
        third_party, _ = _closure([_ROOT / "src" / "models" / "predict.py"], skip=set())
        assert "mlflow" in third_party

    def test_dockerfile_pins_bundle_mode(self):
        assert "CORALSENSE_MODEL_MODE=bundle" in _DOCKERFILE_API.read_text()


class TestXgboostJustification:
    """xgboost appears in no API import — prove the bundle needs it anyway."""

    def test_restoration_payload_references_xgboost(self):
        payload = _BUNDLE_DIR / "restoration" / "payload.joblib"
        if not payload.is_file():
            pytest.skip("deployment bundle not exported; run scripts/export_champions.py")
        assert "xgboost" in _pickle_top_level_modules(payload), (
            "restoration champion should unpickle xgboost classes"
        )

    def test_health_payload_needs_only_the_sklearn_stack(self):
        payload = _BUNDLE_DIR / "health" / "payload.joblib"
        if not payload.is_file():
            pytest.skip("deployment bundle not exported")
        assert "sklearn" in _pickle_top_level_modules(payload)

    @pytest.mark.parametrize("absent", ["mlflow", "dvc", "evidently", "shap", "pandera"])
    def test_no_bundle_artefact_references_a_trimmed_package(self, absent):
        if not _BUNDLE_DIR.is_dir():
            pytest.skip("deployment bundle not exported")
        for artefact in sorted(_BUNDLE_DIR.rglob("*.joblib")):
            assert absent not in _pickle_top_level_modules(artefact), (
                f"{artefact.name} references {absent} — the API image would need it to unpickle"
            )

    def test_every_module_the_bundle_unpickles_is_in_the_api_manifest(self, api_reqs):
        """The decisive completeness check for the trim."""
        if not _BUNDLE_DIR.is_dir():
            pytest.skip("deployment bundle not exported")
        needed: set[str] = set()
        for artefact in sorted(_BUNDLE_DIR.rglob("*.joblib")):
            needed |= _pickle_top_level_modules(artefact)
        missing = set()
        for mod in needed - _STDLIB - {"builtins", "copyreg", "collections", "functools"}:
            dist = _MODULE_TO_DIST.get(mod, _normalise(mod))
            if dist not in api_reqs and not (
                dist == "xgboost" and api_reqs & {"xgboost", "xgboost-cpu"}
            ):
                missing.add(f"{mod} -> {dist}")
        assert not missing, f"bundle unpickling needs packages absent from the manifest: {missing}"

    def test_no_bundle_artefact_is_needed_by_the_dashboard(self):
        """The dashboard image copies no bundle, so it needs no unpickling stack."""
        ignore = (_ROOT / "Dockerfile.dashboard.dockerignore").read_text()
        assert "!deploy" not in ignore
        assert "deploy/bundles" not in _DOCKERFILE_DASHBOARD.read_text()


# ---------------------------------------------------------------------------
# Manifest hygiene
# ---------------------------------------------------------------------------


class TestManifestHygiene:
    @pytest.mark.parametrize("path", [_REQ_API, _REQ_DASHBOARD], ids=["api", "dashboard"])
    def test_no_editable_or_local_path_install(self, path):
        for raw in path.read_text().splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            assert not line.startswith(("-e", "--editable")), f"editable install: {line}"
            assert not line.startswith("."), f"local path install: {line}"
            assert not line.startswith("/"), f"absolute path install: {line}"
            assert "file://" not in line, f"file:// install: {line}"

    @pytest.mark.parametrize("path", [_REQ_API, _REQ_DASHBOARD], ids=["api", "dashboard"])
    def test_no_vcs_url(self, path):
        for raw in path.read_text().splitlines():
            line = raw.split("#", 1)[0].strip()
            for scheme in ("git+", "hg+", "svn+", "bzr+"):
                assert scheme not in line, f"VCS install: {line}"

    @pytest.mark.parametrize("path", [_REQ_API, _REQ_DASHBOARD], ids=["api", "dashboard"])
    def test_no_alternate_index_or_credentials(self, path):
        for raw in path.read_text().splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            assert "--index-url" not in line
            assert "--extra-index-url" not in line
            assert "://" not in line, f"URL in requirement line: {line}"

    @pytest.mark.parametrize("path", [_REQ_API, _REQ_DASHBOARD], ids=["api", "dashboard"])
    def test_no_host_absolute_or_developer_path(self, path):
        text = path.read_text()
        for marker in ("/home/", "/Users/", "/root/", "C:\\", "BAAHbun"):
            assert marker not in text, f"host path leaked into {path.name}: {marker}"

    @pytest.mark.parametrize("path", [_REQ_API, _REQ_DASHBOARD], ids=["api", "dashboard"])
    def test_no_credential_material(self, path):
        text = path.read_text().lower()
        for marker in ("password", "token", "secret", "api_key", "apikey", "dagshub"):
            assert marker not in text, f"credential-looking string in {path.name}: {marker}"

    @pytest.mark.parametrize("path", [_REQ_API, _REQ_DASHBOARD], ids=["api", "dashboard"])
    def test_every_requirement_is_version_constrained(self, path):
        for raw in path.read_text().splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            assert re.search(r"[<>=~!]", line), f"unconstrained requirement: {line}"


# ---------------------------------------------------------------------------
# Dockerfiles install the right manifest — and only that one
# ---------------------------------------------------------------------------


def _instructions(text: str) -> str:
    """Dockerfile text with comment-only lines removed."""
    return "\n".join(ln for ln in text.splitlines() if not ln.strip().startswith("#"))


class TestDockerfilesUseRuntimeManifests:
    def test_api_installs_the_api_manifest(self):
        body = _instructions(_DOCKERFILE_API.read_text())
        assert "requirements-api.txt" in body
        assert "pip install -r requirements-api.txt" in body

    def test_api_does_not_install_the_full_environment(self):
        body = _instructions(_DOCKERFILE_API.read_text())
        assert not re.search(r"(?<!-)\brequirements\.txt\b", body), (
            "Dockerfile.api must not reference the full requirements.txt"
        )

    def test_api_does_not_install_the_dashboard_manifest(self):
        assert "requirements-dashboard.txt" not in _instructions(_DOCKERFILE_API.read_text())

    def test_dashboard_installs_the_dashboard_manifest(self):
        body = _instructions(_DOCKERFILE_DASHBOARD.read_text())
        assert "requirements-dashboard.txt" in body
        assert "pip install -r requirements-dashboard.txt" in body

    def test_dashboard_does_not_install_the_full_environment(self):
        body = _instructions(_DOCKERFILE_DASHBOARD.read_text())
        assert not re.search(r"(?<!-)\brequirements\.txt\b", body), (
            "Dockerfile.dashboard must not reference the full requirements.txt"
        )

    def test_dashboard_does_not_install_the_api_manifest(self):
        assert "requirements-api.txt" not in _instructions(_DOCKERFILE_DASHBOARD.read_text())

    def test_dashboard_no_longer_filters_the_full_manifest(self):
        """The old image installed requirements.txt minus a grep -v of evidently."""
        body = _instructions(_DOCKERFILE_DASHBOARD.read_text())
        assert "grep -v" not in body

    @pytest.mark.parametrize(
        "dockerfile,manifest",
        [
            (_DOCKERFILE_API, "requirements-api.txt"),
            (_DOCKERFILE_DASHBOARD, "requirements-dashboard.txt"),
        ],
        ids=["api", "dashboard"],
    )
    def test_build_context_admits_the_manifest(self, dockerfile, manifest):
        ignore = Path(str(dockerfile) + ".dockerignore")
        assert ignore.is_file()
        rules = {ln.strip() for ln in ignore.read_text().splitlines()}
        assert f"!{manifest}" in rules

    @pytest.mark.parametrize(
        "dockerfile,foreign",
        [
            (_DOCKERFILE_API, "requirements-dashboard.txt"),
            (_DOCKERFILE_DASHBOARD, "requirements-api.txt"),
        ],
        ids=["api", "dashboard"],
    )
    def test_build_context_excludes_the_other_manifest(self, dockerfile, foreign):
        rules = {
            ln.strip() for ln in Path(str(dockerfile) + ".dockerignore").read_text().splitlines()
        }
        assert f"!{foreign}" not in rules

    @pytest.mark.parametrize(
        "dockerfile",
        [_DOCKERFILE_API, _DOCKERFILE_DASHBOARD],
        ids=["api", "dashboard"],
    )
    def test_build_context_excludes_the_full_environment(self, dockerfile):
        rules = {
            ln.strip() for ln in Path(str(dockerfile) + ".dockerignore").read_text().splitlines()
        }
        assert "!requirements.txt" not in rules, (
            "the full development manifest must not reach a serving build context"
        )

    def test_root_dockerignore_admits_both_runtime_manifests(self):
        rules = {ln.strip() for ln in (_ROOT / ".dockerignore").read_text().splitlines()}
        assert "!requirements-api.txt" in rules
        assert "!requirements-dashboard.txt" in rules


class TestServingHardeningPreserved:
    """The trim must not undo anything from the deployment-hardening pass."""

    @pytest.mark.parametrize(
        "dockerfile", [_DOCKERFILE_API, _DOCKERFILE_DASHBOARD], ids=["api", "dashboard"]
    )
    def test_python_312_base(self, dockerfile):
        assert "python:3.12-slim" in dockerfile.read_text()

    @pytest.mark.parametrize(
        "dockerfile", [_DOCKERFILE_API, _DOCKERFILE_DASHBOARD], ids=["api", "dashboard"]
    )
    def test_runs_as_non_root(self, dockerfile):
        assert "USER coralsense" in dockerfile.read_text()

    @pytest.mark.parametrize(
        "dockerfile", [_DOCKERFILE_API, _DOCKERFILE_DASHBOARD], ids=["api", "dashboard"]
    )
    def test_honours_provider_port(self, dockerfile):
        assert "${PORT:-" in dockerfile.read_text()

    def test_api_healthcheck_targets_ready(self):
        assert "/ready" in _DOCKERFILE_API.read_text()

    def test_dashboard_keeps_api_url_env(self):
        assert "CORALSENSE_API_URL" in _DOCKERFILE_DASHBOARD.read_text()
