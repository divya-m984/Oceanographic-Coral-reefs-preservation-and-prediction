"""
tests/test_dashboard_visuals.py — the dashboard's visual systems.

Three systems are covered:

1. **Cinematic media shell** (``src/dashboard/cinema.py`` +
   ``src/dashboard/media.py`` + ``src/dashboard/app.py``) — the background is
   licensed underwater footage, not generated artwork; the landing fills the
   first viewport rather than sitting in a card; navigation is a top bar with
   the original URLs preserved; only the landing loads video; playback stops
   when hidden, off-screen or when reduced motion is requested; every shipped
   file is licensed for redistribution and credited on screen; and content
   stays readable over all of it.
2. **Bathymetric reef map** (``src/dashboard/reefmap.py``) — GEBCO as the
   primary background with no street basemap, packaged Natural Earth labels,
   attribution in every state, and a map whose contents are exactly the frame
   the page filtered.
3. **Visualisation builders** (``src/dashboard/viz/``) — the exact-value
   guarantee for every family: mountain summits equal source metrics, stream
   bands never invent a negative, contour and wireframe matrices survive the
   round trip unchanged.

Plus a guard that this presentation work touched no backend, model, pipeline or
deployment file.

Nothing here needs a browser, a network connection or a running API.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tokenize
from pathlib import Path

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DASHBOARD = _PROJECT_ROOT / "src" / "dashboard"
_GEO = _DASHBOARD / "geo"
# ===========================================================================
# 1. The cinematic media shell
# ===========================================================================


def _code_only(path: Path) -> str:
    """Return *path*'s source with comments and string literals removed.

    Several rules below assert that something is *not done* by a module.  Run
    against raw text they also fire on the docstring that explains why it is not
    done, which would push the modules toward saying less about their own
    history — the opposite of what is wanted here.
    """
    out: list[str] = []
    with path.open("rb") as handle:
        for token in tokenize.tokenize(handle.readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(token.string)
    return " ".join(out)


class TestGeneratedArtworkIsGone:
    """The hand-drawn SVG ocean and the procedural 3-D scene must not return.

    Both were rejected: the WebGL pass read as low-poly, and the SVG pass put
    thousands of animated nodes on every page.  The background is now real
    footage, and these rules are what stop a future edit regressing to either.
    """

    def test_seascape_module_is_gone(self):
        assert not (_DASHBOARD / "seascape.py").exists(), (
            "the generated SVG ocean must not come back"
        )

    def test_hero_module_is_gone(self):
        assert not (_DASHBOARD / "hero.py").exists(), (
            "the bordered hero card was replaced by the full-viewport landing"
        )

    def test_three_js_component_is_gone(self):
        assert not (_DASHBOARD / "frontend").exists()

    def test_no_webgl_or_canvas_anywhere(self):
        for path in _DASHBOARD.rglob("*.py"):
            code = _code_only(path).lower()
            for banned in ("three.js", "three.module", "webgl", "getcontext", "<canvas"):
                assert banned not in code, f"{path.name} reaches for {banned}"

    def test_no_generated_marine_life_or_coral(self):
        """No module may generate whale, coral or fish geometry any more."""
        for path in _DASHBOARD.rglob("*.py"):
            code = _code_only(path)
            for banned in (
                "_branching_coral",
                "_sea_fan",
                "_boulder_coral",
                "_tube_sponge",
                "_kelp_blade",
                "_animal_svg",
                "_reef_band",
                "_swimmer",
            ):
                assert banned not in code, f"{path.name} still generates reef/marine art"

    def test_stylesheet_has_no_generated_scene_rules(self):
        from src.dashboard import theme

        css = theme._stylesheet()
        for banned in (".cs-sea__reef", ".cs-whale", ".cs-sea__swim", "feTurbulence"):
            assert banned not in css, f"the stylesheet still carries {banned}"

    def test_background_dom_is_tiny(self):
        """The whole point of the rewrite: a background of a dozen nodes.

        The previous pass emitted ~4,300 SVG paths per page.
        """
        from src.dashboard import cinema

        html = cinema._STAGE_HTML
        assert html.count("<path") == 0
        assert html.count("<") < 30, f"{html.count('<')} tags is not a lightweight backdrop"

    def test_no_per_creature_javascript_animation(self):
        """Marine movement must come from the footage, not from script."""
        from src.dashboard import cinema

        js = cinema._STAGE_JS
        for banned in ("requestAnimationFrame", "setInterval", "@keyframes"):
            assert banned not in js, f"the stage animates with {banned}"


class TestMediaStage:
    """One fixed, viewport-filling media plane behind the whole app."""

    def test_module_exists(self):
        assert (_DASHBOARD / "cinema.py").is_file()
        assert (_DASHBOARD / "media.py").is_file()

    def test_stage_is_fixed_and_covers_the_viewport(self):
        from src.dashboard import theme

        css = theme._stylesheet()
        block = css[css.index(".cs-stage {") : css.index(".cs-stage__grade")]
        assert "position: fixed" in block
        assert "inset: 0" in block
        assert "object-fit: cover" in block
        assert "height: 100%" in block

    def test_stage_carries_the_three_overlay_layers(self):
        """Grade, readability scrim and vignette — the brief names all three.

        They live in the Python-rendered plate, not in the component, so they
        survive a client-side component failure along with the photograph.
        """
        import inspect

        from src.dashboard import cinema, theme

        css = theme._stylesheet()
        overlays = inspect.getsource(cinema._overlays)
        for layer in ("cs-stage__grade", "cs-stage__scrim", "cs-stage__vignette"):
            assert f".{layer}" in css, f"the stage has no {layer} rule"
            assert layer in overlays, f"the overlay markup has no {layer}"

    def test_readability_scrim_runs_left_to_right(self):
        from src.dashboard import theme

        css = theme._stylesheet()
        block = css[css.index(".cs-stage__scrim") :][:600]
        assert "linear-gradient(96deg" in block, "the scrim must be a horizontal wash"

    def test_stage_is_not_interactive(self):
        from src.dashboard import theme

        block = theme._stylesheet()
        assert "pointer-events: none" in block

    def test_stage_is_hidden_from_assistive_technology(self):
        from src.dashboard import cinema

        assert 'aria-hidden="true"' in cinema._STAGE_HTML

    def test_content_is_promoted_above_the_stage(self):
        from src.dashboard import theme

        css = theme._stylesheet()
        assert "z-index: 1" in css
        assert 'data-testid="stMain"' in css

    def test_video_element_has_every_required_attribute(self):
        """muted+playsinline+autoplay is what makes inline autoplay legal at all."""
        from src.dashboard import cinema

        html = cinema._STAGE_HTML
        for attribute in ("autoplay", "muted", "loop", "playsinline", "poster="):
            assert attribute in html, f"the video is missing {attribute}"

    def test_video_offers_webm_before_mp4(self):
        """Both encodings ship; the browser should get the smaller one first."""
        from src.dashboard import cinema

        html = cinema._STAGE_HTML
        assert html.index("video/webm") < html.index("video/mp4")

    def test_component_is_v2_and_not_iframed(self):
        """V1 would iframe the shell, which cannot sit behind the app's content."""
        source = (_DASHBOARD / "cinema.py").read_text()
        assert "components.v2" in source
        assert "components.v1" not in source
        assert "declare_component" not in source
        assert "isolate_styles=False" in source, (
            "the stage must mount into the app DOM, not a shadow root"
        )

    def test_stage_reports_the_mode_it_rendered(self):
        from src.dashboard import cinema

        assert "video" in cinema.stage.__doc__
        assert "still" in cinema.stage.__doc__


_NODE = shutil.which("node")
_needs_node = pytest.mark.skipif(_NODE is None, reason="node is not installed")

#: Minimal DOM good enough to run the stage component's lifecycle.  Not a
#: browser — just enough surface for the module to wire itself up and hand back
#: a cleanup function.
_DOM_STUB = """
const listeners = {};
const el = (extra = {}) => Object.assign({
  attributes: {}, style: {}, parentElement: null, children: [],
  setAttribute(k, v) { this.attributes[k] = v; },
  removeAttribute(k) { delete this.attributes[k]; },
  querySelector() { return null; },
  insertBefore(node) { node.parentElement = this; this.children.push(node); return node; },
  removeChild(node) { node.parentElement = null; },
  pause() { this.paused = true; },
  play() { this.paused = false; return Promise.resolve(); },
}, extra);

const video = el({ tagName: 'VIDEO' });
globalThis.IntersectionObserver = class {
  observe() {}
  disconnect() { globalThis.__disconnected = true; }
};
// In a browser `window` IS the global, so anything on globalThis is `in window`.
// The stub has to reproduce that or the component's feature detection silently
// skips the observer.
globalThis.window = {
  matchMedia: () => ({ matches: false, addEventListener() {}, removeEventListener() {} }),
  IntersectionObserver: globalThis.IntersectionObserver,
};
globalThis.document = {
  hidden: false,
  addEventListener(type, fn) { (listeners[type] ||= []).push(fn); },
  removeEventListener(type, fn) {
    listeners[type] = (listeners[type] || []).filter((f) => f !== fn);
  },
  querySelector: () => null,
};

const mod = await import(MODULE_URL);
const parentElement = el({ querySelector: (sel) => (sel.includes('cs-video') ? video : null) });
const cleanup = mod.default({ parentElement, data: null, key: 'k', name: 'n' });

const out = {
  defaultIsFunction: typeof mod.default === 'function',
  cleanupIsFunction: typeof cleanup === 'function',
  videoMuted: video.muted === true,
  visibilityListener: (listeners['visibilitychange'] || []).length === 1,
};
cleanup();
out.listenerRemoved = (listeners['visibilitychange'] || []).length === 0;
out.observerDisconnected = globalThis.__disconnected === true;
console.log(JSON.stringify(out));
"""


@_needs_node
class TestComponentJavaScriptIsAValidModule:
    """The stage JS must be a real ES module, executed in a real JS engine.

    This class exists because a text-only check missed a fatal bug: the JS was
    written as a bare function body with `if (...) return;` and a trailing
    `return () => {...}`.  Streamlit turns the string into a Blob and
    ``import()``s it, so those are *module-level* returns — "SyntaxError:
    Illegal return statement" — and the whole component died at runtime while
    every string assertion still passed.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def module_path(cls, tmp_path_factory) -> Path:
        from src.dashboard import cinema

        path = tmp_path_factory.mktemp("stage") / "stage.mjs"
        path.write_text(cinema._STAGE_JS)
        return path

    def test_javascript_parses_as_a_module(self, module_path):
        result = subprocess.run(
            [_NODE, "--check", str(module_path)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, f"component JS does not parse:\n{result.stderr}"

    def test_no_module_level_return(self, module_path):
        """The exact failure mode, asserted directly on the source."""
        source = module_path.read_text()
        assert "export default" in source, "Streamlit requires a default export"
        # A `return` in column 0 is outside every function body by definition.
        offenders = [
            (number, line)
            for number, line in enumerate(source.splitlines(), start=1)
            if line.startswith("return")
        ]
        assert not offenders, f"module-level return at {offenders}"

    def test_module_runs_and_returns_a_cleanup(self, module_path, tmp_path):
        """Import it, call it against a DOM stub, and check the lifecycle."""
        harness = tmp_path / "harness.mjs"
        harness.write_text(f"const MODULE_URL = {json.dumps(module_path.as_uri())};\n" + _DOM_STUB)
        result = subprocess.run(
            [_NODE, str(harness)], capture_output=True, text=True, timeout=90, check=False
        )
        assert result.returncode == 0, f"component JS threw:\n{result.stderr[-2000:]}"
        report = json.loads(result.stdout.strip().splitlines()[-1])
        assert report["defaultIsFunction"], "Streamlit needs module.default to be callable"
        assert report["cleanupIsFunction"], "the cleanup contract returns a function"
        assert report["videoMuted"], "autoplay requires the muted property, not just the attribute"
        assert report["visibilityListener"], "playback is not wired to visibilitychange"
        assert report["listenerRemoved"], "cleanup leaked a document listener"
        assert report["observerDisconnected"], "cleanup leaked an IntersectionObserver"

    def test_a_bare_function_body_would_fail_this_suite(self, tmp_path):
        """Proves the check has teeth: the old shape must still be rejected."""
        bad = tmp_path / "bad.mjs"
        bad.write_text(
            "const v = document.querySelector('x');\nif (!v) return;\nreturn () => {};\n"
        )
        result = subprocess.run(
            [_NODE, "--check", str(bad)], capture_output=True, text=True, timeout=60, check=False
        )
        assert result.returncode != 0
        assert "return" in result.stderr.lower()


class TestReducedMotionAndPlaybackBudget:
    """Video must stop when it is not wanted, not merely be quiet."""

    @pytest.fixture(scope="class")
    @classmethod
    def js(cls) -> str:
        from src.dashboard import cinema

        return cinema._STAGE_JS

    def test_reduced_motion_is_honoured(self, js):
        assert "prefers-reduced-motion" in js
        assert "video.pause()" in js

    def test_reduced_motion_leaves_the_poster(self, js):
        """The poster attribute is the fallback frame; it must not be cleared."""
        from src.dashboard import cinema

        assert "removeAttribute('poster')" not in js
        assert "poster=" in cinema._STAGE_HTML

    def test_pauses_when_the_document_is_hidden(self, js):
        assert "visibilitychange" in js
        assert "document.hidden" in js

    def test_pauses_when_scrolled_out_of_view(self, js):
        assert "IntersectionObserver" in js

    def test_listeners_are_cleaned_up(self, js):
        assert "removeEventListener" in js
        assert "disconnect()" in js

    def test_stylesheet_also_guards_reduced_motion(self):
        from src.dashboard import theme

        assert "@media (prefers-reduced-motion: reduce)" in theme._stylesheet()


class TestOnlyTheLandingLoadsVideo:
    """Analytical pages must not pay for video decode."""

    def test_home_is_the_only_view_asking_for_motion(self):
        asking = [
            p.name for p in (_DASHBOARD / "views").glob("*.py") if "motion=True" in p.read_text()
        ]
        assert asking == ["0_Home.py"], f"these views request video: {asking}"

    def test_set_page_defaults_to_no_motion(self):
        import inspect

        from src.dashboard.components import set_page

        assert inspect.signature(set_page).parameters["motion"].default is False

    def test_interior_pages_get_the_still_stage(self):
        from src.dashboard import cinema

        assert cinema.stage(motion=False) in {"still", "gradient"}

    def test_interior_still_is_pre_graded_not_filtered_live(self):
        """Blur/darken/desaturate are baked into the JPEG, not done per frame."""
        from src.dashboard import media

        credit = media.credit_for(media.INTERIOR_STILL)
        assert credit is not None
        assert "blur" in credit.changes.lower()

    def test_plate_uses_no_script(self):
        """The photographic layer must be pure markup — it is the fallback."""
        import inspect

        from src.dashboard import cinema

        plate = inspect.getsource(cinema._plate)
        assert "components" not in plate
        assert "<script" not in plate
        assert "<video" not in plate


class TestBackgroundNeverFallsBackToFlatColour:
    """A client-side failure must still leave a real underwater photograph.

    The regression this guards: the component crashed on a JS syntax error, the
    Python side had already returned "video", nothing else had been rendered,
    and the page showed flat navy.
    """

    def test_the_plate_is_rendered_before_the_component(self):
        """Order matters: the photograph must not depend on the component."""
        import inspect

        from src.dashboard import cinema

        source = inspect.getsource(cinema.stage)
        assert source.index("_plate(") < source.index("_video_stage()"), (
            "the photographic plate must be rendered before the video is attempted"
        )

    def test_landing_plate_is_the_video_poster(self):
        """So the still and the first video frame are the same image."""
        import inspect

        from src.dashboard import cinema

        assert "HOME_POSTER" in inspect.getsource(cinema.stage)

    def test_a_dead_component_still_leaves_a_photograph(self, monkeypatch):
        """Simulate the exact failure: mounting raises."""
        from src.dashboard import cinema

        monkeypatch.setattr(cinema, "_stage_component", None)
        monkeypatch.setattr(cinema, "_stage_failed", False)
        monkeypatch.setattr(cinema, "_video_stage", lambda: (_ for _ in ()).throw(RuntimeError))
        rendered: list[str] = []
        monkeypatch.setattr(cinema.st, "markdown", lambda html, **kw: rendered.append(str(html)))
        try:
            cinema.stage(motion=True)
        except RuntimeError:
            pytest.fail("a failing component must not propagate")
        blob = " ".join(rendered)
        assert "cs-stage__plate" in blob
        assert "background-image:url(" in blob.replace(" ", ""), "no photograph in the fallback"

    def test_the_video_is_an_enhancement_not_the_background(self):
        """The <video> alone must never be the only thing painting the stage."""
        import inspect

        from src.dashboard import cinema

        assert "cs-stage__plate" not in cinema._STAGE_HTML
        assert "cs-stage__plate" in inspect.getsource(cinema._plate)


class TestLandingComposition:
    """The first viewport is the hero — not a card inside a dashboard."""

    @pytest.fixture(scope="class")
    @classmethod
    def home(cls) -> str:
        return (_DASHBOARD / "views" / "0_Home.py").read_text()

    def test_landing_is_about_one_viewport_tall(self):
        """The height must live on .cs-landing, which is a single markdown block.

        A wrapper <div> cannot carry it: Streamlit renders each element into its
        own container, so an opening tag from one st.markdown call is closed by
        the parser before the next block and wraps nothing.
        """
        from src.dashboard import theme

        css = theme._stylesheet()
        block = css[css.index(".cs-landing {") : css.index(".cs-landing__index")]
        assert "min-height" in block and "vh" in block

    def test_no_cross_block_wrapper_div_is_emitted(self):
        """Guards the mistake above from being reintroduced."""
        home = (_DASHBOARD / "views" / "0_Home.py").read_text()
        assert "st.markdown('<div" not in home, (
            "an unclosed wrapper div cannot span Streamlit blocks"
        )

    def test_landing_is_not_wrapped_in_a_card(self):
        from src.dashboard import theme

        css = theme._stylesheet()
        block = css[css.index(".cs-landing {") :][:300]
        assert "border:" not in block, "the landing must not be a bordered rectangle"

    def test_title_uses_the_cinematic_scale(self):
        from src.dashboard import theme

        css = theme._stylesheet()
        block = css[css.index(".cs-landing__title") :][:500]
        assert "clamp(" in block and "7vw" in block
        assert "line-height: 0.9" in block
        assert "font-weight: 700" in block

    def test_every_briefed_landing_element_is_present(self, home):
        for element in ("cinema.landing", "cinema.footnote", "cinema.feature_card"):
            assert element in home, f"the landing is missing {element}"

    def test_landing_carries_index_title_lede_and_meta(self, home):
        assert "index=" in home
        assert "title_lines=" in home
        assert "lede=" in home
        assert "meta=" in home

    def test_title_does_not_copy_the_reference_wording(self, home):
        assert "ONE OCEAN" not in home.upper().replace("\n", " ")

    def test_feature_card_is_dark_glass_not_a_glowing_frame(self):
        from src.dashboard import theme

        css = theme._stylesheet()
        block = css[css.index(".cs-feature {") : css.index(".cs-feature__label")]
        assert "backdrop-filter" in block
        assert "rgba(0, 8, 15" in block, "the card must be dark translucent glass"
        assert "box-shadow: 0 0" not in block, "no glow ring"
        assert "var(--cs-aqua)" not in block, "no cyan border"

    def test_calls_to_action_use_real_routing(self, home):
        """page_link keeps navigation Streamlit's job rather than a fake anchor."""
        assert "st.page_link" in home

    def test_lower_dashboard_survived(self, home):
        for section in ("Dataset at a Glance", "Project Objective", "Champion Models"):
            assert section in home, f"{section} was lost in the redesign"


class TestTopNavigation:
    """The sidebar nav is gone; the router declares pages explicitly."""

    @pytest.fixture(scope="class")
    @classmethod
    def router(cls) -> str:
        return (_DASHBOARD / "app.py").read_text()

    def test_router_uses_top_navigation(self, router):
        assert "st.navigation" in router
        assert 'position="top"' in router

    def test_router_declares_pages_explicitly(self, router):
        assert "st.Page" in router

    def test_auto_discovery_is_disabled(self):
        """A pages/ folder beside the entrypoint would add a competing nav."""
        assert not (_DASHBOARD / "pages").exists()
        assert (_DASHBOARD / "views").is_dir()

    def test_every_view_is_registered(self, router):
        views = sorted(p.name for p in (_DASHBOARD / "views").glob("*.py"))
        for name in views:
            assert name in router, f"{name} is not registered with the router"

    @pytest.mark.parametrize(
        "url_path",
        [
            "Overview",
            "Reef_Map",
            "Habitat_Health",
            "Restoration_Planning",
            "Predict",
            "Model_Performance",
            "MLOps_Status",
            "Drift_Monitoring",
            "Governance",
        ],
    )
    def test_original_url_is_preserved(self, router, url_path):
        """Auto-discovery derived these from filenames; they must not change."""
        assert f'"{url_path}"' in router, f"/{url_path} would 404 after the migration"

    def test_primary_labels_match_the_brief(self, router):
        for label in ("Overview", "Reef Map", "Habitat", "Restoration", "Predict", "Models"):
            assert f'"{label}"' in router, f"missing top-level label {label}"

    def test_operational_pages_are_in_a_secondary_group(self, router):
        assert "OPERATIONS" in router or "Operations" in router
        for label in ("MLOps Status", "Drift Monitoring", "Governance"):
            assert f'"{label}"' in router

    def test_only_the_router_sets_page_config(self):
        """Under st.navigation a second call raises."""
        assert "set_page_config" in (_DASHBOARD / "app.py").read_text()
        for view in (_DASHBOARD / "views").glob("*.py"):
            assert "set_page_config" not in view.read_text(), f"{view.name} sets page config"

    def test_sidebar_is_filters_only(self):
        """Brand and status left the rail; seven of ten pages now have none."""
        source = (_DASHBOARD / "components.py").read_text()
        block = source[source.index("def render_sidebar") :]
        assert "cs-brand" not in block, "branding still lives in the sidebar"


class TestMediaLicensing:
    """Shipped footage must be licensed for redistribution and credited."""

    def test_registry_exists(self):
        from src.dashboard import media

        assert media.MEDIA_CREDITS

    @pytest.mark.parametrize("field", ["title", "creator", "source_url", "licence", "credit"])
    def test_every_credit_is_complete(self, field):
        from src.dashboard import media

        for entry in media.MEDIA_CREDITS:
            value = getattr(entry, field)
            assert value and value.strip(), f"{entry.files}: empty {field}"

    def test_every_licence_permits_redistribution(self):
        from src.dashboard import media

        allowed = ("CC0", "CC BY", "Public domain")
        for entry in media.MEDIA_CREDITS:
            assert entry.licence.startswith(allowed), f"{entry.licence} is not redistributable"
            assert "NC" not in entry.licence, "non-commercial media cannot ship here"
            assert "ND" not in entry.licence, "no-derivatives media cannot be re-encoded"

    def test_changes_are_stated(self):
        """CC BY requires indicating that the work was modified."""
        from src.dashboard import media

        for entry in media.MEDIA_CREDITS:
            assert entry.changes and len(entry.changes) > 20, f"{entry.files}: no change statement"

    def test_licence_urls_are_creative_commons(self):
        from src.dashboard import media

        for entry in media.MEDIA_CREDITS:
            assert entry.licence_url.startswith("https://creativecommons.org/"), entry.licence_url

    def test_no_broadcast_footage(self):
        """The brief rules out BBC / Blue Planet material explicitly."""
        from src.dashboard import media

        blob = " ".join(
            f"{e.title} {e.creator} {e.source_url} {e.credit}" for e in media.MEDIA_CREDITS
        ).lower()
        for banned in ("bbc", "blue planet", "netflix", "disney", "national geographic"):
            assert banned not in blob, f"{banned} material must not be used"

    def test_every_shipped_file_is_registered(self):
        """A file on disk with no credit would be an unattributed redistribution."""
        from src.dashboard import media

        registered = {name for entry in media.MEDIA_CREDITS for name in entry.files}
        on_disk = {
            p.name
            for p in media.MEDIA_DIR.glob("*")
            if p.suffix.lower() in {".mp4", ".webm", ".jpg", ".jpeg", ".png", ".webp"}
        }
        unregistered = on_disk - registered - {"cs-wordmark.svg"}
        assert not unregistered, f"unattributed media: {sorted(unregistered)}"

    def test_attribution_document_exists_and_names_every_source(self):
        doc = _DASHBOARD / "static" / "media" / "ATTRIBUTION.md"
        assert doc.is_file()
        text = doc.read_text()
        from src.dashboard import media

        for entry in media.MEDIA_CREDITS:
            assert entry.licence in text
            assert entry.source_url in text

    def test_credit_is_rendered_on_screen(self):
        """Recording the licence is not enough; CC BY needs a visible credit."""
        from src.dashboard import media

        html = media.credits_html()
        if media.shipped_files():
            assert html
            for entry in media.MEDIA_CREDITS:
                if any(media.available(f) for f in entry.files):
                    assert entry.title in html
                    assert entry.licence in html

    def test_home_renders_the_credits(self):
        assert "media_credits" in (_DASHBOARD / "views" / "0_Home.py").read_text()


class TestMediaBudget:
    """The media has to stay small enough to be a background."""

    def test_video_is_not_4k(self):
        from src.dashboard import media

        for name in (media.HOME_VIDEO_MP4, media.HOME_VIDEO_WEBM):
            if not media.available(name):
                pytest.skip(f"{name} not present in this checkout")
            # 1920x1080 H.264/VP9 at a few seconds cannot exceed this.
            assert media.size_bytes(name) < 12_000_000, f"{name} is too heavy for a backdrop"

    def test_total_media_weight_is_bounded(self):
        from src.dashboard import media

        if not media.shipped_files():
            pytest.skip("no media in this checkout")
        assert media.total_bytes() < 16_000_000, (
            f"shipped media is {media.total_bytes() / 1e6:.1f} MB"
        )

    def test_a_poster_exists_for_the_video(self):
        from src.dashboard import media

        if media.available(media.HOME_VIDEO_MP4):
            assert media.available(media.HOME_POSTER), "the video has no poster frame"

    def test_media_is_served_from_the_apps_own_origin(self):
        from src.dashboard import media

        assert media.url(media.HOME_VIDEO_MP4).startswith("app/static/")
        assert "http" not in media.url(media.HOME_VIDEO_MP4)

    def test_static_serving_is_enabled(self):
        config = (_PROJECT_ROOT / ".streamlit" / "config.toml").read_text()
        assert "enableStaticServing" in config
        assert "true" in config.split("enableStaticServing")[1][:20]

    def test_static_folder_sits_beside_the_entrypoint(self):
        """Streamlit resolves ./static relative to the main script, not the CWD."""
        from src.dashboard import media

        assert media.MEDIA_DIR.parent.parent == _DASHBOARD


class TestMediaDegradesWithoutFiles:
    """A checkout without the binaries must still render."""

    def test_missing_file_reports_unavailable(self, monkeypatch, tmp_path):
        from src.dashboard import media

        monkeypatch.setattr(media, "MEDIA_DIR", tmp_path / "absent")
        assert not media.available(media.HOME_VIDEO_MP4)
        assert media.shipped_files() == ()
        assert media.total_bytes() == 0

    def test_stage_falls_back_to_a_gradient(self, monkeypatch, tmp_path):
        from src.dashboard import cinema, media

        monkeypatch.setattr(media, "MEDIA_DIR", tmp_path / "absent")
        assert cinema.stage(motion=True) == "gradient"

    def test_credits_are_empty_when_nothing_ships(self, monkeypatch, tmp_path):
        from src.dashboard import media

        monkeypatch.setattr(media, "MEDIA_DIR", tmp_path / "absent")
        assert media.credits_html() == ""

    def test_a_fallback_gradient_is_defined(self):
        from src.dashboard import media

        assert "gradient" in media.FALLBACK_GRADIENT


class TestPanelsStayReadableOverMedia:
    """Full-bleed media is only acceptable if the content still reads."""

    @pytest.fixture(scope="class")
    @classmethod
    def css(cls) -> str:
        from src.dashboard import theme

        return theme._stylesheet()

    @pytest.mark.parametrize(
        "surface",
        [".cs-panel", ".cs-stat", ".cs-pred", '[data-testid="stMetric"]', '[data-testid="stForm"]'],
    )
    def test_surface_blurs_what_is_behind_it(self, css, surface):
        block = css[css.index(surface + " {") :][:900]
        assert "backdrop-filter" in block, f"{surface} has no backdrop blur"

    def test_chart_surface_is_blurred_too(self, css):
        block = css[css.index('[data-testid="stPlotlyChart"]') :][:600]
        assert "backdrop-filter" in block

    def test_surfaces_are_translucent_not_opaque(self, css):
        glass = css[css.index("--cs-glass:") : css.index("--cs-glass-hi:")]
        alphas = [float(a) for a in re.findall(r"rgba\([^)]*?,\s*([\d.]+)\)", glass)]
        assert alphas and max(alphas) < 1.0, "the glass surfaces are opaque"

    def test_landing_type_carries_a_shadow_over_footage(self, css):
        block = css[css.index(".cs-landing__title") :][:500]
        assert "text-shadow" in block


# ===========================================================================
# 2. Bathymetric reef map
# ===========================================================================


def _sample_observations(n: int = 40) -> pd.DataFrame:
    regions = ["Lakshadweep", "Gulf of Mannar", "Gulf of Kutch", "Andaman and Nicobar Islands"]
    health = ["healthy", "stressed", "bleached", "severely_degraded"]
    restoration = ["suitable", "moderately_suitable", "unsuitable"]
    return pd.DataFrame(
        {
            "latitude": [8.0 + (i % 15) for i in range(n)],
            "longitude": [70.0 + (i % 20) for i in range(n)],
            "region": [regions[i % 4] for i in range(n)],
            "reef_health": [health[i % 4] for i in range(n)],
            "restoration_suitability": [restoration[i % 3] for i in range(n)],
            "depth_m": [1.0 + i * 0.5 for i in range(n)],
            "water_temperature_c": [26.0 + (i % 6) for i in range(n)],
            "coral_cover_percentage": [10.0 + (i % 50) for i in range(n)],
        }
    )


class TestPackagedGeographicData:
    def test_country_boundary_file_exists(self):
        assert (_GEO / "ne_admin0_indian_ocean.json").is_file()

    def test_place_label_file_exists(self):
        assert (_GEO / "ne_places_indian_ocean.json").is_file()

    def test_attribution_document_names_every_source(self):
        text = (_GEO / "ATTRIBUTION.md").read_text()
        assert "Natural Earth" in text
        assert "public domain" in text.lower()
        assert "GEBCO" in text
        assert "Leaflet" in text

    def test_countries_load_as_a_feature_collection(self):
        from src.dashboard.reefmap import load_countries

        data = load_countries()
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) >= 10

    def test_india_and_its_neighbours_are_present(self):
        from src.dashboard.reefmap import load_countries

        names = {f["properties"]["name"] for f in load_countries()["features"]}
        for expected in ("INDIA", "SRI LANKA", "MALDIVES"):
            assert expected in names, f"{expected} missing from the boundary subset"

    def test_country_labels_are_uppercase_with_a_label_anchor(self):
        from src.dashboard.reefmap import load_countries

        for feature in load_countries()["features"]:
            props = feature["properties"]
            assert props["name"] == props["name"].upper()
            assert "label_lat" in props and "label_lon" in props

    def test_places_are_significant_cities_only(self):
        from src.dashboard.reefmap import load_places

        features = load_places()["features"]
        assert 10 <= len(features) <= 400, "the subset must stay minimal"
        names = {f["properties"]["name"] for f in features}
        assert "New Delhi" in names
        assert "Chennai" in names

    def test_packaged_data_stays_small(self):
        """'Package only the minimum required data into the dashboard image.'"""
        total = sum(p.stat().st_size for p in _GEO.glob("*.json"))
        assert total < 600_000, f"packaged geo data is {total} bytes"

    def test_missing_file_degrades_to_an_empty_collection(self, tmp_path, monkeypatch):
        from src.dashboard import reefmap

        monkeypatch.setattr(reefmap, "COUNTRIES_PATH", tmp_path / "absent.json")
        reefmap.load_countries.clear()
        try:
            assert reefmap.load_countries()["features"] == []
        finally:
            reefmap.load_countries.clear()


class TestBathymetricMapRendering:
    @pytest.fixture(scope="class")
    @classmethod
    def rendered(cls) -> str:
        from src.dashboard.components import HEALTH_COLORS
        from src.dashboard.reefmap import build_reef_map

        fmap, used = build_reef_map(
            _sample_observations(),
            color_col="reef_health",
            color_map=HEALTH_COLORS,
            bathymetry=True,
        )
        assert used is True
        return fmap.get_root().render()

    def test_gebco_wms_is_the_primary_background(self, rendered):
        assert "wms.gebco.net" in rendered
        assert "GEBCO_LATEST_2" in rendered

    def test_shaded_relief_overlay_is_offered(self, rendered):
        assert "GEBCO_LATEST" in rendered

    def test_gebco_attribution_is_present(self, rendered):
        assert "GEBCO Compilation Group" in rendered

    def test_natural_earth_attribution_is_present(self, rendered):
        assert "naturalearthdata.com" in rendered

    def test_no_street_basemap_is_loaded(self, rendered):
        lowered = rendered.lower()
        for street in ("tile.openstreetmap.org", "basemaps.cartocdn.com", "stamen"):
            assert street not in lowered, f"{street} must not appear"

    def test_country_labels_are_rendered(self, rendered):
        assert "INDIA" in rendered
        assert "text-transform:uppercase" in rendered

    def test_city_labels_are_rendered(self, rendered):
        assert "Chennai" in rendered

    def test_labels_declutter_by_zoom(self, rendered):
        assert "zoomend" in rendered
        assert "getZoom()" in rendered

    def test_observations_carry_hover_information(self, rendered):
        assert "Reef health" in rendered
        assert "Restoration" in rendered
        assert "Coral cover" in rendered

    def test_graticule_is_drawn(self, rendered):
        assert "poly_line" in rendered.lower() or "polyline" in rendered.lower()


class TestBathymetryFallback:
    @pytest.fixture(scope="class")
    @classmethod
    def rendered(cls) -> str:
        from src.dashboard.components import HEALTH_COLORS
        from src.dashboard.reefmap import build_reef_map

        fmap, used = build_reef_map(
            _sample_observations(),
            color_col="reef_health",
            color_map=HEALTH_COLORS,
            bathymetry=False,
        )
        assert used is False
        return fmap.get_root().render()

    def test_no_wms_request_is_made(self, rendered):
        assert "wms.gebco.net" not in rendered

    def test_a_styled_background_replaces_it(self, rendered):
        assert "radial-gradient" in rendered

    def test_no_street_map_is_substituted(self, rendered):
        lowered = rendered.lower()
        for street in ("tile.openstreetmap.org", "basemaps.cartocdn.com", "stamen"):
            assert street not in lowered

    def test_data_layers_survive(self, rendered):
        assert "INDIA" in rendered
        assert "Reef health" in rendered

    def test_attribution_survives(self, rendered):
        assert "naturalearthdata.com" in rendered

    def test_probe_never_raises(self, monkeypatch):
        from src.dashboard import reefmap

        def _explode(*args, **kwargs):
            raise OSError("network down")

        monkeypatch.setattr("requests.get", _explode)
        reefmap.bathymetry_available.clear()
        try:
            assert reefmap.bathymetry_available() is False
        finally:
            reefmap.bathymetry_available.clear()


class TestMapReceivesTheFilteredFrame:
    """The map must show exactly the rows the page filtered — no more, no less."""

    @staticmethod
    def _observation_features(df, color_col="reef_health"):
        """Return the point features the map would draw for *df*.

        Observations are one vector layer over one FeatureCollection rather
        than a folium object per row, so the frame that reaches the map is read
        straight off that collection.
        """
        import folium

        from src.dashboard.components import HEALTH_COLORS, RESTORATION_COLORS
        from src.dashboard.reefmap import build_reef_map

        colours = HEALTH_COLORS if color_col == "reef_health" else RESTORATION_COLORS
        fmap, _ = build_reef_map(df, color_col=color_col, color_map=colours, bathymetry=False)
        for child in _walk(fmap):
            if isinstance(child, folium.GeoJson) and child.layer_name == (
                "CoralSense observations"
            ):
                return child.data["features"]
        return []

    def test_every_filtered_row_is_drawn(self):
        df = _sample_observations(12)
        assert len(self._observation_features(df)) == len(df)

    def test_coordinates_match_the_source_rows(self):
        df = _sample_observations(6)
        features = self._observation_features(df)
        drawn = sorted(tuple(f["geometry"]["coordinates"]) for f in features)
        expected = sorted(
            (round(float(lon), 4), round(float(lat), 4))
            for lon, lat in zip(df["longitude"], df["latitude"], strict=True)
        )
        assert drawn == expected

    def test_a_narrower_filter_draws_fewer_markers(self):
        full = _sample_observations(24)
        filtered = full[full["reef_health"] == "healthy"]
        assert 0 < len(filtered) < len(full)
        assert len(self._observation_features(filtered)) == len(filtered)
        assert len(self._observation_features(filtered)) < len(self._observation_features(full))

    def test_class_filter_changes_which_rows_reach_the_map(self):
        full = _sample_observations(24)
        bleached = full[full["reef_health"] == "bleached"]
        labels = {f["properties"]["Reef health"] for f in self._observation_features(bleached)}
        assert labels == {"Bleached"}

    def test_each_marker_carries_a_halo_and_a_core(self):
        """Small solid core inside a wide, low-opacity halo stroke."""
        import folium

        from src.dashboard.components import HEALTH_COLORS
        from src.dashboard.reefmap import build_reef_map

        fmap, _ = build_reef_map(
            _sample_observations(4),
            color_col="reef_health",
            color_map=HEALTH_COLORS,
            bathymetry=False,
        )
        layer = next(
            c
            for c in _walk(fmap)
            if isinstance(c, folium.GeoJson) and c.layer_name == "CoralSense observations"
        )
        style = layer.style_function(layer.data["features"][0])
        assert style["fillOpacity"] > 0.8, "the core must be solid"
        assert style["opacity"] < 0.3, "the halo must be low-opacity"
        assert style["weight"] > style["radius"], "the halo must be wider than the core"

    def test_builder_does_not_resample_or_reorder(self):
        """No sampling happens inside the builder; that is the page's job."""
        source = (_DASHBOARD / "reefmap.py").read_text()
        assert "sample(" not in source
        assert "head(" not in source

    def test_colour_column_selects_the_semantic_class(self):
        from src.dashboard.components import RESTORATION_COLORS
        from src.dashboard.reefmap import build_reef_map

        df = _sample_observations(8)
        fmap, _ = build_reef_map(
            df,
            color_col="restoration_suitability",
            color_map=RESTORATION_COLORS,
            bathymetry=False,
        )
        html = fmap.get_root().render()
        assert RESTORATION_COLORS["suitable"] in html

    def test_empty_frame_renders_a_bare_map(self):
        from src.dashboard.components import HEALTH_COLORS
        from src.dashboard.reefmap import build_reef_map

        empty = _sample_observations(0)
        fmap, _ = build_reef_map(
            empty, color_col="reef_health", color_map=HEALTH_COLORS, bathymetry=False
        )
        assert "INDIA" in fmap.get_root().render()


def _walk(node):
    """Yield every child of a folium/branca element tree, depth first."""
    for child in getattr(node, "_children", {}).values():
        yield child
        yield from _walk(child)


class TestReefMapPageContract:
    """Page 2 must keep every control it had, with unchanged semantics."""

    @pytest.fixture(scope="class")
    @classmethod
    def source(cls) -> str:
        return (_DASHBOARD / "views" / "2_Reef_Map.py").read_text()

    def test_uses_the_leaflet_renderer(self, source):
        assert "from streamlit_folium import st_folium" in source
        assert "build_reef_map" in source

    def test_plotly_map_is_gone(self, source):
        assert "scatter_map" not in source
        assert "open-street-map" not in source

    @pytest.mark.parametrize(
        "control",
        [
            "Colour observations by",
            "Health class filter",
            "Restoration filter",
            "Max points",
        ],
    )
    def test_control_is_preserved(self, control, source):
        assert control in source

    def test_max_points_options_are_unchanged(self, source):
        assert "[500, 1000, 2000, 5000]" in source
        assert "value=2000" in source

    def test_region_filter_is_still_applied(self, source):
        assert 'df_all["region"].isin(selected_regions)' in source

    def test_class_filters_are_still_applied(self, source):
        assert 'df["reef_health"].isin(health_filter)' in source
        assert 'df["restoration_suitability"].isin(rest_filter)' in source

    def test_sampling_still_bounds_the_render(self, source):
        assert "sample_for_display(df, max_n=max_pts)" in source

    def test_region_table_is_preserved(self, source):
        assert "Observations by Region" in source
        assert "Healthy (%)" in source
        assert "Suitable (%)" in source

    def test_attribution_is_shown_on_the_page(self, source):
        assert "theme.attribution(reefmap.ATTRIBUTIONS)" in source

    def test_unreachable_service_is_reported_not_raised(self, source):
        assert "bathymetry_used" in source
        assert "st.warning" in source


# ===========================================================================
# 3. Visualisation builders — the exact-value guarantee
# ===========================================================================


class TestRidgeProfile:
    def test_summit_is_exactly_one(self):
        from src.dashboard.viz import ridge_profile

        profile = ridge_profile()
        assert profile[len(profile) // 2] == 1.0

    def test_nothing_rises_above_the_summit(self):
        from src.dashboard.viz import ridge_profile

        assert max(ridge_profile()) == 1.0

    def test_feet_touch_the_baseline(self):
        from src.dashboard.viz import ridge_profile

        profile = ridge_profile()
        assert profile[0] == 0.0
        assert profile[-1] == 0.0

    @pytest.mark.parametrize("texture", [0.0, 0.03, 0.055, 0.12, 0.3])
    def test_texture_never_moves_the_peak(self, texture):
        """Decorative ridge variation must not change the encoded height."""
        from src.dashboard.viz import ridge_profile

        profile = ridge_profile(texture=texture)
        assert max(profile) == 1.0
        assert profile[len(profile) // 2] == 1.0

    def test_scaled_ridge_peaks_at_the_source_value(self):
        from src.dashboard.viz.common import ridge_xy

        for value in (0.0, 0.7612, 1.0, 42.0, 15_000.0):
            _, ys = ridge_xy(0.0, value)
            assert max(ys) == pytest.approx(value, abs=0.0)


class TestMountainChartPreservesValues:
    METRICS = [0.7612, 0.7455, 0.7301]
    LABELS = ["Logistic Regression", "Random Forest", "XGBoost"]

    def _fig(self, **kwargs):
        from src.dashboard.viz import mountain_chart

        return mountain_chart(self.LABELS, self.METRICS, value_name="CV Macro F1", **kwargs)

    def test_summit_marker_equals_the_source_metric(self):
        fig = self._fig()
        summits = sorted(
            float(trace.y[0])
            for trace in fig.data
            if trace.mode == "markers" and trace.y is not None and len(trace.y) == 1
        )
        assert summits == sorted(self.METRICS)

    def test_ridge_maximum_equals_the_source_metric(self):
        fig = self._fig()
        maxima = {round(max(t.y), 12) for t in fig.data if t.fill == "tozeroy"}
        for value in self.METRICS:
            assert round(value, 12) in maxima

    def test_no_ridge_exceeds_its_metric(self):
        """Backing layers are fractions of the value, never above it."""
        fig = self._fig()
        assert max(max(t.y) for t in fig.data if t.fill == "tozeroy") <= max(self.METRICS)

    def test_hover_prints_the_exact_value(self):
        fig = self._fig()
        templates = " ".join(t.hovertemplate or "" for t in fig.data)
        for value in self.METRICS:
            assert f"{value:.4f}" in templates

    def test_exact_value_is_printed_beside_the_peak(self):
        fig = self._fig()
        annotations = " ".join(a.text for a in fig.layout.annotations)
        for value in self.METRICS:
            assert f"{value:.4f}" in annotations

    def test_champion_highlight_does_not_change_the_height(self):
        plain = self._fig()
        highlighted = self._fig(highlight="Logistic Regression")

        def summits(fig):
            return sorted(
                float(t.y[0])
                for t in fig.data
                if t.mode == "markers" and t.y is not None and len(t.y) == 1
            )

        assert summits(plain) == summits(highlighted)

    def test_error_whisker_brackets_the_summit(self):
        fig = self._fig(errors=[0.011, 0.009, 0.014])
        whiskers = [t for t in fig.data if t.line and t.line.dash == "dot"]
        assert len(whiskers) == len(self.METRICS)
        for trace, value, err in zip(whiskers, self.METRICS, [0.011, 0.009, 0.014], strict=True):
            assert min(trace.y) == pytest.approx(value - err)
            assert max(trace.y) == pytest.approx(value + err)

    def test_zero_is_drawn_flat_not_hidden(self):
        from src.dashboard.viz import mountain_chart

        fig = mountain_chart(["a", "b"], [0.0, 1.0])
        maxima = sorted(round(max(t.y), 12) for t in fig.data if t.fill == "tozeroy")
        assert 0.0 in maxima

    def test_length_mismatch_is_rejected(self):
        from src.dashboard.viz import mountain_chart

        with pytest.raises(ValueError, match="length mismatch"):
            mountain_chart(["a", "b"], [1.0])


class TestRidgelineIsAFaithfulHistogram:
    def test_counts_are_the_real_row_counts(self):
        from src.dashboard.viz.common import histogram

        values = [1, 1, 1, 2, 2, 3]
        _, counts, _ = histogram(values, bins=3, lo=1, hi=4)
        assert sum(counts) == len(values)

    def test_no_value_is_dropped_at_the_upper_edge(self):
        from src.dashboard.viz.common import histogram

        _, counts, _ = histogram([0.0, 5.0, 10.0], bins=5, lo=0.0, hi=10.0)
        assert sum(counts) == 3

    def test_every_group_becomes_a_ridge(self):
        from src.dashboard.viz import ridgeline_chart

        fig = ridgeline_chart(
            ["healthy", "stressed"],
            {"healthy": [1, 2, 3], "stressed": [2, 3, 4]},
        )
        filled = [t for t in fig.data if t.fill == "toself"]
        assert len(filled) == 2

    def test_hover_carries_exact_counts(self):
        from src.dashboard.viz import ridgeline_chart

        fig = ridgeline_chart(["a"], {"a": [1, 1, 2, 3]})
        template = next(t.hovertemplate for t in fig.data if t.fill == "toself")
        assert "customdata[0]" in template
        assert "count" in template

    def test_median_marker_is_the_real_median(self):
        from src.dashboard.viz import ridgeline_chart

        fig = ridgeline_chart(["a"], {"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        markers = [t for t in fig.data if t.mode == "markers+text"]
        assert markers and float(markers[0].x[0]) == pytest.approx(3.0)

    def test_outline_is_linear_not_smoothed(self):
        """A spline would smooth away a spike; the shape must stay honest."""
        from src.dashboard.viz import ridgeline_chart

        fig = ridgeline_chart(["a"], {"a": [1, 2, 3]})
        for trace in fig.data:
            if trace.fill == "toself":
                assert trace.line.shape == "linear"

    def test_empty_group_is_skipped_not_faked(self):
        from src.dashboard.viz import ridgeline_chart

        fig = ridgeline_chart(["a", "b"], {"a": [1, 2, 3], "b": []})
        assert len([t for t in fig.data if t.fill == "toself"]) == 1


class TestStreamNeverInventsNegatives:
    def test_band_values_reach_the_hover_unchanged(self):
        from src.dashboard.viz import stream_chart

        series = {"healthy": [10, 20, 30], "stressed": [5, 15, 25]}
        fig = stream_chart(["A", "B", "C"], series)
        seen = {
            tuple(int(point[1]) for point in trace.customdata)
            for trace in fig.data
            if trace.customdata is not None
        }
        assert tuple(series["healthy"]) in seen
        assert tuple(series["stressed"]) in seen

    def test_band_thickness_equals_the_value(self):
        """Centring shifts the baseline; it must not change any thickness."""
        from src.dashboard.viz.stream import _stack

        categories = ["A", "B"]
        series = {"x": [3.0, 5.0], "y": [1.0, 2.0]}
        edges = _stack(categories, series, ["x", "y"])
        assert edges["x"] == [3.0, 5.0]
        assert [edges["y"][i] - edges["x"][i] for i in range(2)] == [1.0, 2.0]

    def test_negative_input_is_refused(self):
        from src.dashboard.viz import stream_chart

        with pytest.raises(ValueError, match="negative"):
            stream_chart(["A"], {"x": [-3.0]})

    def test_no_plotted_point_is_negative_when_not_centred(self):
        from src.dashboard.viz import stream_chart

        fig = stream_chart(["A", "B"], {"x": [1.0, 2.0]}, centred=False)
        for trace in fig.data:
            if trace.fill == "toself":
                assert min(trace.y) >= 0

    def test_length_mismatch_is_rejected(self):
        from src.dashboard.viz import stream_chart

        with pytest.raises(ValueError, match="expected"):
            stream_chart(["A", "B"], {"x": [1.0]})

    def test_stations_are_pinned_to_their_exact_values(self):
        """Easing may bend the space between stations, never a station itself."""
        from src.dashboard.viz.stream import _organic_path

        values = [1.0, 7.0, 3.0]
        xs, ys = _organic_path(values)
        for i, value in enumerate(values):
            assert ys[xs.index(float(i))] == value


class TestMirroredStreamIsSemantic:
    def test_both_halves_are_named(self):
        from src.dashboard.viz import mirrored_stream

        fig = mirrored_stream(
            ["healthy", "stressed"],
            {"share": [0.6, 0.4]},
            {"share": [0.3, 0.7]},
            up_name="Reference window",
            down_name="Production window",
        )
        annotations = " ".join(a.text for a in fig.layout.annotations)
        assert "Reference window" in annotations
        assert "Production window" in annotations

    def test_hover_reports_positive_magnitudes(self):
        from src.dashboard.viz import mirrored_stream

        fig = mirrored_stream(
            ["healthy"],
            {"share": [0.6]},
            {"share": [0.3]},
            up_name="Reference",
            down_name="Production",
        )
        values = [
            point[1]
            for trace in fig.data
            if trace.customdata is not None
            for point in trace.customdata
        ]
        assert all(value >= 0 for value in values)
        assert 0.6 in values and 0.3 in values

    def test_axis_ticks_show_magnitudes_not_negatives(self):
        from src.dashboard.viz import mirrored_stream

        fig = mirrored_stream(["a"], {"s": [0.8]}, {"s": [0.4]}, up_name="Ref", down_name="Prod")
        assert fig.layout.yaxis.ticktext is not None
        assert all(not str(t).startswith("-") for t in fig.layout.yaxis.ticktext)

    def test_negative_input_is_refused(self):
        from src.dashboard.viz import mirrored_stream

        with pytest.raises(ValueError, match="negative"):
            mirrored_stream(["a"], {"s": [0.5]}, {"s": [-0.5]}, up_name="Ref", down_name="Prod")


class TestContourPreservesTheMatrix:
    def test_values_survive_the_round_trip(self):
        from src.dashboard.viz import bathymetric_contour

        z = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        fig = bathymetric_contour([10, 20, 30], [1, 2], z)
        assert [list(row) for row in fig.data[0].z] == z

    def test_gaps_are_preserved_not_interpolated(self):
        from src.dashboard.viz import bathymetric_contour

        fig = bathymetric_contour([1, 2], [1, 2], [[1.0, None], [None, 4.0]])
        assert fig.data[0].connectgaps is False
        assert list(fig.data[0].z[0])[1] is None

    def test_counts_reach_the_hover(self):
        from src.dashboard.viz import bathymetric_contour

        fig = bathymetric_contour([1, 2], [1], [[0.5, 0.6]], counts=[[12, 34]])
        assert "observations" in fig.data[0].hovertemplate
        assert fig.data[0].customdata is not None

    def test_isolines_are_drawn(self):
        from src.dashboard.viz import bathymetric_contour

        fig = bathymetric_contour([1, 2], [1, 2], [[1.0, 2.0], [3.0, 4.0]])
        assert fig.data[0].contours.showlines is True

    def test_shape_mismatch_is_rejected(self):
        from src.dashboard.viz import bathymetric_contour

        with pytest.raises(ValueError):
            bathymetric_contour([1, 2, 3], [1], [[1.0, 2.0]])

    def test_count_shape_mismatch_is_rejected(self):
        from src.dashboard.viz import bathymetric_contour

        with pytest.raises(ValueError, match="counts"):
            bathymetric_contour([1, 2], [1], [[1.0, 2.0]], counts=[[1]])


class TestWireframePreservesTheMatrix:
    def test_surface_values_survive_the_round_trip(self):
        from src.dashboard.viz import sonar_wireframe

        z = [[0.1, 0.2], [0.3, 0.4]]
        fig = sonar_wireframe(["a", "b"], ["p", "q"], z)
        assert [list(row) for row in fig.data[0].z] == z

    def test_every_node_is_reachable_with_its_exact_value(self):
        from src.dashboard.viz import sonar_wireframe

        z = [[0.11, 0.22], [0.33, 0.44]]
        fig = sonar_wireframe(["a", "b"], ["p", "q"], z)
        nodes = next(t for t in fig.data if t.type == "scatter3d" and t.mode == "markers")
        assert sorted(float(v) for v in nodes.z) == sorted(v for row in z for v in row)

    def test_missing_cells_are_holes_not_zeros(self):
        from src.dashboard.viz import sonar_wireframe

        fig = sonar_wireframe(["a", "b"], ["p"], [[1.0, None]])
        nodes = next(t for t in fig.data if t.type == "scatter3d" and t.mode == "markers")
        assert list(nodes.z) == [1.0]

    def test_mesh_lines_run_along_rows_and_columns(self):
        from src.dashboard.viz import sonar_wireframe

        fig = sonar_wireframe(["a", "b", "c"], ["p", "q"], [[1, 2, 3], [4, 5, 6]])
        line_traces = [t for t in fig.data if t.type == "scatter3d" and t.mode == "lines"]
        # Two passes (glow + core) over 2 rows + 3 columns.
        assert len(line_traces) == (2 + 3) * 2

    def test_surface_stays_low_opacity(self):
        from src.dashboard.viz import sonar_wireframe

        fig = sonar_wireframe(["a"], ["p"], [[1.0]])
        assert fig.data[0].opacity <= 0.4

    def test_shape_mismatch_is_rejected(self):
        from src.dashboard.viz import sonar_wireframe

        with pytest.raises(ValueError):
            sonar_wireframe(["a", "b"], ["p"], [[1.0]])


class TestVisualSystemIsCentralised:
    """The builders live in one place and every chart page uses them."""

    @pytest.mark.parametrize(
        "module",
        ["__init__", "common", "mountain", "stream", "contour", "wireframe"],
    )
    def test_module_exists(self, module):
        assert (_DASHBOARD / "viz" / f"{module}.py").is_file()

    @pytest.mark.parametrize(
        "page",
        [
            "1_Overview.py",
            "3_Habitat_Health.py",
            "4_Restoration_Planning.py",
            "6_Model_Performance.py",
            "8_Drift_Monitoring.py",
        ],
    )
    def test_chart_page_uses_the_system(self, page):
        source = (_DASHBOARD / "views" / page).read_text()
        assert "from src.dashboard.viz import" in source

    @pytest.mark.parametrize(
        "page",
        [
            "1_Overview.py",
            "2_Reef_Map.py",
            "3_Habitat_Health.py",
            "4_Restoration_Planning.py",
            "6_Model_Performance.py",
            "8_Drift_Monitoring.py",
        ],
    )
    def test_no_default_plotly_express_chart_remains(self, page):
        """px.bar / px.pie / px.box and friends are gone from every page."""
        source = (_DASHBOARD / "views" / page).read_text()
        assert "plotly.express" not in source
        for builder in ("px.bar", "px.pie", "px.box", "px.violin", "px.histogram", "px.scatter"):
            assert builder not in source, f"{page} still calls {builder}"

    def test_viz_package_avoids_numpy(self):
        """The dashboard runtime manifest carries no numpy; keep it that way."""
        for path in (_DASHBOARD / "viz").glob("*.py"):
            source = path.read_text()
            assert "import numpy" not in source
            assert "from numpy" not in source

    def test_theme_exposes_the_ocean_tokens(self):
        from src.dashboard import theme

        for token in (
            "VOID",
            "TRENCH",
            "BASIN",
            "OCEAN_DEEP",
            "OCEAN_MID",
            "OCEAN_BRIGHT",
            "OCEAN_GLOW",
            "FOAM",
        ):
            assert hasattr(theme, token), f"theme is missing {token}"
            assert getattr(theme, token).startswith("#")

    def test_depth_ramp_runs_deep_to_pale(self):
        from src.dashboard import theme

        assert theme.DEPTH_RAMP[0] == theme.TRENCH
        assert theme.DEPTH_RAMP[-1] == theme.FOAM

    def test_ramp_endpoints_are_exact(self):
        from src.dashboard.viz import ramp_color
        from src.dashboard.viz.common import DEPTH_LAYERS

        assert ramp_color(0.0) == DEPTH_LAYERS[0]
        assert ramp_color(1.0) == DEPTH_LAYERS[-1]


# ===========================================================================
# 4. Every page still renders
# ===========================================================================


# app.py is the st.navigation router; running it renders the default page
# (0_Home).  0_Home is therefore covered through the router and not listed
# separately: its st.page_link calls can only resolve under navigation.
_ALL_PAGES = ["src/dashboard/app.py"] + sorted(
    str(p.relative_to(_PROJECT_ROOT))
    for p in (_DASHBOARD / "views").glob("*.py")
    if p.name != "0_Home.py"
)


class TestEveryPageRenders:
    """AppTest over all ten scripts, with the dataset redirected at tmp_path."""

    @pytest.fixture
    def prepared(self, tmp_path, monkeypatch):
        from src.dashboard import data_loader

        regions = [
            "Lakshadweep",
            "Gulf of Mannar",
            "Gulf of Kutch",
            "Andaman and Nicobar Islands",
        ]
        health = ["healthy", "stressed", "bleached", "severely_degraded"]
        restoration = ["suitable", "moderately_suitable", "unsuitable"]
        n = 120
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
                "latitude": [8.0 + (i % 15) * 0.9 for i in range(n)],
                "longitude": [70.0 + (i % 22) * 1.1 for i in range(n)],
                "region": [regions[i % 4] for i in range(n)],
                "depth_m": [1.0 + (i % 40) for i in range(n)],
                "water_temperature_c": [24.0 + (i % 9) * 0.8 for i in range(n)],
                "ph": [7.9 + (i % 5) * 0.05 for i in range(n)],
                "salinity_ppt": [30.0 + (i % 7) for i in range(n)],
                "dissolved_oxygen_mg_l": [5.0 + (i % 6) * 0.7 for i in range(n)],
                "turbidity_ntu": [0.5 + (i % 18) for i in range(n)],
                "light_intensity": [200.0 + (i % 30) * 55 for i in range(n)],
                "current_speed_m_s": [0.05 * (i % 12) for i in range(n)],
                "sonar_backscatter": [-45.0 + (i % 35) for i in range(n)],
                "rugosity_index": [1.0 + (i % 7) * 0.8 for i in range(n)],
                "hard_substrate_percentage": [10.0 + (i % 40) * 2 for i in range(n)],
                "acoustic_complexity_index": [0.1 + (i % 8) * 0.1 for i in range(n)],
                "coral_cover_percentage": [5.0 + (i % 35) * 2 for i in range(n)],
                "bleaching_percentage": [(i % 28) * 1.0 for i in range(n)],
                "disease_percentage": [(i % 14) * 1.0 for i in range(n)],
                "reef_health": [health[i % 4] for i in range(n)],
                "restoration_suitability": [restoration[i % 3] for i in range(n)],
            }
        )
        csv = tmp_path / "observations.csv"
        frame.to_csv(csv, index=False)
        monkeypatch.setattr(data_loader, "_RAW_CSV", csv)
        data_loader.load_observations.clear()

        labels = {"health": health, "restoration": restoration}
        for task, names in labels.items():
            entry = {
                "cv_macro_f1_mean": 0.7612,
                "cv_macro_f1_std": 0.0113,
                "cv_balanced_accuracy_mean": 0.7702,
                "cv_balanced_accuracy_std": 0.0104,
                "test_accuracy": 0.8011,
                "test_balanced_accuracy": 0.7803,
                "test_macro_precision": 0.7402,
                "test_macro_recall": 0.7604,
                "test_macro_f1": 0.7871,
                "test_weighted_f1": 0.7902,
                "test_per_class": {
                    name: {"precision": 0.75, "recall": 0.75, "f1": 0.75, "support": 100}
                    for name in names
                },
            }
            (tmp_path / f"evaluation_{task}.json").write_text(
                json.dumps(
                    {
                        "task": task,
                        "best_model_name": (
                            "logistic_regression" if task == "health" else "xgboost"
                        ),
                        "best_cv_macro_f1": 0.7612,
                        "label_names": names,
                        "training_duration_s": 5.0,
                        "models": {
                            "logistic_regression": entry,
                            "random_forest": dict(entry, cv_macro_f1_mean=0.7455),
                            "xgboost": dict(entry, cv_macro_f1_mean=0.7301),
                        },
                    }
                )
            )
        monkeypatch.setattr(data_loader, "_EVAL_DIR", tmp_path)
        data_loader.load_evaluation.clear()

        # The reef map must not touch the network from a test.
        from src.dashboard import reefmap

        reefmap.bathymetry_available.clear()
        monkeypatch.setattr(reefmap, "bathymetry_available", lambda *a, **k: False)
        return tmp_path

    @pytest.mark.parametrize("page", _ALL_PAGES, ids=lambda p: Path(p).stem)
    def test_page_runs_without_exception(self, page, prepared, isolated_reports_dir):
        from unittest import mock

        from streamlit.testing.v1 import AppTest

        from src.dashboard.api_client import APIError

        with mock.patch("src.dashboard.api_client.APIClient") as MockClient:
            MockClient.return_value.health.return_value = {"status": "ok"}
            MockClient.return_value.model_info.side_effect = APIError("offline")
            at = AppTest.from_file(str(_PROJECT_ROOT / page), default_timeout=180)
            at.run()
            assert len(at.exception) == 0, [str(e.value) for e in at.exception]

    def test_prediction_result_renders_two_probability_mountains(self):
        """Submitting the Predict form draws a mountain for each champion.

        The two probability sets are deliberately given identical *values*
        here: that is the case that collides on Streamlit's auto-generated
        element ID unless both charts carry an explicit key.
        """
        from unittest import mock

        from streamlit.testing.v1 import AppTest

        response = {
            "health": {
                "predicted_class": "healthy",
                "probabilities": {
                    "healthy": 0.80,
                    "stressed": 0.15,
                    "bleached": 0.03,
                    "severely_degraded": 0.02,
                },
                "confidence": 0.80,
                "task": "health",
                "registered_model_name": "coralsense_reef_health",
                "model_version": "1",
                "model_alias": "champion",
                "run_id": "a",
                "prediction_timestamp": "2026-07-14T10:00:00+00:00",
                "synthetic_data_disclaimer": "Synthetic data only.",
            },
            "restoration": {
                "predicted_class": "suitable",
                "probabilities": {
                    "suitable": 0.80,
                    "moderately_suitable": 0.15,
                    "unsuitable": 0.05,
                },
                "confidence": 0.80,
                "task": "restoration",
                "registered_model_name": "coralsense_restoration_suitability",
                "model_version": "1",
                "model_alias": "champion",
                "run_id": "b",
                "prediction_timestamp": "2026-07-14T10:00:00+00:00",
                "synthetic_data_disclaimer": "Synthetic data only.",
            },
        }

        with mock.patch("src.dashboard.api_client.APIClient") as MockClient:
            MockClient.return_value.health.return_value = {"status": "ok"}
            MockClient.return_value.predict_both.return_value = response
            at = AppTest.from_file(str(_DASHBOARD / "views" / "5_Predict.py"), default_timeout=180)
            at.run()
            assert len(at.exception) == 0
            at.button[0].click().run()
            assert len(at.exception) == 0, [str(e.value) for e in at.exception]
            assert len(at.get("plotly_chart")) == 2


# ===========================================================================
# 5. Scope guard — presentation only
# ===========================================================================

# Paths this visual pass is forbidden from touching. Anything the working tree
# reports as modified must fall outside every one of these.
_PROTECTED_PREFIXES = (
    "src/api/",
    "src/models/",
    "src/data/",
    "src/features/",
    "src/monitoring/",
    "src/config.py",
    "models/",
    "data/",
    "artifacts/",
    "mlruns/",
    "dvc.yaml",
    "dvc.lock",
    "params.yaml",
    ".github/",
    "docker-compose.yml",
    "Dockerfile.api",
    "Dockerfile.mlflow",
    "render.yaml",
)


def _changed_paths() -> list[str] | None:
    """Return the working tree's changed paths, or None outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    paths = []
    for line in result.stdout.splitlines():
        if len(line) > 3:
            paths.append(line[3:].strip().strip('"'))
    return paths


#: The one protected file the navigation migration is allowed to touch, and why.
#:
#: The release workflow renders every page script by globbing the page
#: directory.  Moving ``pages/`` to ``views/`` (required, because a ``pages/``
#: folder beside the entrypoint is auto-discovered and would fight
#: ``st.navigation``) makes that glob match nothing, so the release job would
#: silently verify zero pages.  The exception is deliberately a single named
#: file, and ``test_the_release_workflow_exception_is_only_the_page_glob``
#: pins what the change was allowed to be.
_MIGRATION_EXCEPTIONS: dict[str, str] = {
    ".github/workflows/release-images.yml": ("page-render glob follows the pages/ -> views/ move"),
}

#: Backend files the P0 scientific-disclosure hardening is allowed to touch.
#:
#: The 2026-08-19 dataset audit found label-construction leakage / circular
#: supervision, and the disclosure of that finding has to reach every surface
#: that presents metrics or predictions — which necessarily includes the API
#: payload disclaimer and the generated model cards.  A dashboard-only edit
#: cannot carry that disclosure.
#:
#: Each entry is a single named file with a stated reason, and
#: ``test_the_disclosure_exceptions_are_only_disclosure_text`` pins that the
#: change to each one was in fact disclosure wording, so this cannot become a
#: general-purpose hole in the backend guard.
_DISCLOSURE_EXCEPTIONS: dict[str, str] = {
    "src/api/schemas.py": "_DISCLAIMER shipped in every API response payload",
    "src/api/main.py": "module docstring + OpenAPI description disclosure",
    "src/models/model_card.py": "_SYNTHETIC_DISCLAIMER and Limitations section",
}


class TestNoBackendFileWasTouched:
    """This is a dashboard-presentation change and nothing else."""

    def test_working_tree_touches_no_protected_path(self):
        changed = _changed_paths()
        if changed is None:
            pytest.skip("not a git checkout")
        allowed = {**_MIGRATION_EXCEPTIONS, **_DISCLOSURE_EXCEPTIONS}
        offenders = [
            path
            for path in changed
            if any(path.startswith(prefix) for prefix in _PROTECTED_PREFIXES)
            and path not in allowed
        ]
        assert not offenders, f"protected files modified: {sorted(offenders)}"

    def test_the_disclosure_exceptions_are_only_disclosure_text(self):
        """The backend files exempted above must carry the disclosure, and only that.

        Without this, ``_DISCLOSURE_EXCEPTIONS`` would be a standing licence to
        change the API and model-card modules unnoticed.  Each exempted file has
        to actually contain the scientific disclosure it was exempted for.
        """
        required = ("algorithmically generated", "circular supervision")
        for path in _DISCLOSURE_EXCEPTIONS:
            text = (_PROJECT_ROOT / path).read_text(encoding="utf-8").lower()
            for phrase in required:
                assert phrase in text, (
                    f"{path} is exempted for scientific disclosure but does not contain {phrase!r}"
                )

    def test_the_release_workflow_exception_is_only_the_page_glob(self):
        """The one allowed deployment edit must be exactly what it claims.

        Without this, the exception above would be a hole big enough to change
        anything in the release pipeline unnoticed.
        """
        workflow = (_PROJECT_ROOT / ".github/workflows/release-images.yml").read_text()
        assert '(root / "views").glob' in workflow, "the workflow lost the page-render sweep"
        assert '(root / "pages").glob' not in workflow, "the workflow still globs the old folder"
        # Nothing about how the images are built, tagged, pushed or scanned moved.
        for unchanged in ("docker build", "docker run", "dvc pull", "ghcr.io", "buildx"):
            assert unchanged in workflow, f"the release workflow lost {unchanged!r}"

    def test_dashboard_still_imports_no_model_or_registry_code(self):
        for path in _DASHBOARD.rglob("*.py"):
            source = path.read_text()
            assert "src.models.train" not in source
            assert "src.models.registry" not in source
            assert "src.models.predict" not in source

    def test_dashboard_still_loads_no_model(self):
        for path in _DASHBOARD.rglob("*.py"):
            source = path.read_text()
            assert "joblib.load" not in source
            assert "import mlflow" not in source

    def test_prediction_still_goes_through_the_api(self):
        source = (_DASHBOARD / "views" / "5_Predict.py").read_text()
        assert "client.predict_both(payload)" in source
        assert "build_predict_payload(form_values)" in source

    def test_map_module_writes_nothing(self):
        source = (_DASHBOARD / "reefmap.py").read_text()
        for writer in ("open(", ".write(", ".to_csv(", "mkdir("):
            if writer == "open(":
                # Read-only opens are fine; a write mode is not.
                assert '"w"' not in source and "'w'" not in source
            else:
                assert writer not in source
