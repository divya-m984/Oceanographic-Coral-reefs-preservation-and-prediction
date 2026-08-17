"""
src/dashboard/cinema.py — the cinematic shell: full-bleed media stage and landing.

What this is
------------
Two things, and deliberately only two:

1. **The stage** — a fixed, viewport-filling media plane behind the whole page,
   plus the three grading layers that make text readable over it.  On the home
   page it is one ``<video>``; everywhere else it is one still image and no
   video at all.
2. **The landing** — the first viewport of the home page: page index, the
   cinematic title, a lede, a rule, run metadata, and calls to action, laid out
   over the stage rather than inside a card.

Everything else — charts, maps, forms, model output — stays ordinary Streamlit.

Why Custom Components V2
------------------------
The stage needs a real ``<video>`` element and a little JavaScript to pause it,
neither of which survives ``st.markdown(unsafe_allow_html=True)``.  Streamlit
1.60 ships ``st.components.v2``, which mounts markup **directly into the app's
DOM** (``isolate_styles=False``) and runs script with normal page privileges.
That is the whole reason to use it here: the V1 API would have put the stage in
an iframe, which cannot sit behind the app's own content and would have meant
iframing the shell.  The component owns the media only; it does not wrap the
dashboard.

Cost
----
The background is now one video element, one poster, and three empty overlay
divs — roughly a dozen DOM nodes, replacing the ~4,300 animated SVG paths of the
previous pass.  Interior pages drop the video entirely and paint a single
pre-blurred JPEG, so Plotly and Leaflet never contend with video decode.

Degradation, in order
---------------------
* ``prefers-reduced-motion: reduce`` → poster frame only, video never plays.
* Media files absent from the checkout → graded CSS gradient, no ``<video>``.
* Video fails to load or autoplay is blocked → the poster stays visible, since
  it is the element's own ``poster`` attribute.
* Component API unavailable → the still backdrop is used instead, which is
  plain CSS and cannot fail.
"""

from __future__ import annotations

from collections.abc import Sequence
from html import escape
from typing import Any

import streamlit as st

from src.dashboard import media

# ---------------------------------------------------------------------------
# 1. The media stage
# ---------------------------------------------------------------------------

#: The component contributes ONLY the video element.
#:
#: The stage container, the poster plate and the three grading overlays are
#: rendered from Python on every page (see ``_plate``), so the photographic
#: background exists whether or not this component ever mounts.  The JS moves
#: this element into that container, which is what layers it correctly:
#: plate -> video -> grade -> scrim -> vignette.
_STAGE_HTML = """
<video class="cs-stage__video" data-cs-video aria-hidden="true"
       autoplay muted loop playsinline preload="auto"
       poster="{poster}" tabindex="-1">
  <source src="{webm}" type="video/webm">
  <source src="{mp4}" type="video/mp4">
</video>
"""

#: Playback policy.
#:
#: **This must be an ES module with a default-exported function.**  Streamlit
#: turns the string into a Blob and `import()`s it
#: (``BidiComponent.*.js`` -> ``getOrCreateUrlForJs`` -> ``import(s)``), then
#: requires ``module.default`` to be callable and invokes it with
#: ``{name, data, key, parentElement, setStateValue, setTriggerValue}``.  The
#: value it returns is retained and later called as the cleanup
#: (``Promise.resolve(v).then(f => f?.())``).
#:
#: A bare function body — `if (...) return;` and a trailing `return () => {}`
#: at top level — is therefore a *module-level* return, which is a hard
#: SyntaxError ("return not in function") and kills the whole component.  Every
#: return below is inside the exported function for exactly that reason.
#:
#: The element is declaratively autoplay/loop/muted; this only *stops* it — when
#: the tab is hidden, when the stage is scrolled away, and when the visitor has
#: asked for reduced motion.  No animation loop, nothing per-frame.
_STAGE_JS = """
export default function ({ parentElement }) {
  const host = parentElement || document;
  const video = host.querySelector('[data-cs-video]');
  if (!video) {
    return undefined;
  }

  // Autoplay is only permitted for muted media, and some browsers ignore the
  // bare attribute — set the property too.
  video.muted = true;
  video.defaultMuted = true;

  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');

  // Reduced motion keeps the real photograph: the poster attribute and the
  // plate underneath both stay painted, only the motion stops.
  const applyReduced = () => {
    if (reduced.matches) {
      video.pause();
      video.removeAttribute('autoplay');
      video.setAttribute('data-cs-paused', 'reduced-motion');
    } else {
      video.removeAttribute('data-cs-paused');
      void video.play().catch(() => {});
    }
  };

  let onScreen = true;
  const sync = () => {
    if (reduced.matches) {
      return;
    }
    if (document.hidden || !onScreen) {
      video.pause();
    } else {
      void video.play().catch(() => {});
    }
  };

  const onVisibility = () => sync();
  document.addEventListener('visibilitychange', onVisibility);
  reduced.addEventListener('change', applyReduced);

  // Stop decoding once the stage has been scrolled past.
  let observer = null;
  if ('IntersectionObserver' in window) {
    observer = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        onScreen = entry.isIntersecting;
      }
      sync();
    }, { threshold: 0 });
    observer.observe(video);
  }

  applyReduced();

  return () => {
    document.removeEventListener('visibilitychange', onVisibility);
    reduced.removeEventListener('change', applyReduced);
    if (observer) {
      observer.disconnect();
    }
    video.pause();
    if (video.parentElement) {
      video.parentElement.removeChild(video);
    }
  };
}
"""

_stage_component: Any = None
_stage_failed = False


def _video_stage() -> Any:
    """Register the video stage component once per process, or give up quietly."""
    global _stage_component, _stage_failed
    if _stage_component is not None or _stage_failed:
        return _stage_component
    try:
        _stage_component = st.components.v2.component(
            "coralsense_stage",
            html=_STAGE_HTML.format(
                poster=media.url(media.HOME_POSTER),
                webm=media.url(media.HOME_VIDEO_WEBM),
                mp4=media.url(media.HOME_VIDEO_MP4),
            ),
            js=_STAGE_JS,
            # The stage must be styleable by the app's own stylesheet and must
            # sit behind app content; a shadow root would isolate it from both.
            isolate_styles=False,
        )
    except Exception:
        _stage_failed = True
        _stage_component = None
    return _stage_component


def _plate(image: str | None, *, still: bool) -> None:
    """Paint the photographic plate — the bottom layer of the stage.

    Rendered from Python on **every** page and in every mode, with no component
    and no script, so the cinematic background cannot be lost to a client-side
    failure.  This is what makes a flat navy page impossible: the photograph is
    already on screen before the video is even attempted.
    """
    layer = (
        f"url('{media.url(image)}')"
        if image and media.available(image)
        else media.FALLBACK_GRADIENT
    )
    variant = " cs-stage--still" if still else ""
    st.markdown(
        f'<div class="cs-stage{variant}" data-cs-stage aria-hidden="true">'
        f'<div class="cs-stage__plate" style="background-image:{layer}"></div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def _overlays(*, still: bool) -> None:
    """Paint the three grading layers — the top of the stage.

    Emitted as a **separate block after** the video so the stack is built by DOM
    order alone: plate -> video -> grade -> scrim -> vignette.  Three sibling
    blocks in the main container each form a layer at the same z-index, and the
    later one paints over the earlier, which is exactly the order wanted.

    Ordering it this way rather than moving the video into the plate with
    JavaScript matters: the plate is a React-managed node, so anything script
    inserted into it could be wiped by a rerun re-applying the markup.
    """
    variant = " cs-stage--still" if still else ""
    st.markdown(
        f'<div class="cs-stage cs-stage--overlays{variant}" aria-hidden="true">'
        f'<div class="cs-stage__grade"></div>'
        f'<div class="cs-stage__scrim"></div>'
        f'<div class="cs-stage__vignette"></div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def stage(*, motion: bool = False) -> str:
    """
    Render the background stage and report which mode was used.

    ``motion=True`` asks for the video; it is granted only on the home page and
    only when the files are actually present.  Every other page gets the still,
    which is the whole reason charts and maps stay responsive.

    The photographic plate is rendered **first and unconditionally**.  The video
    is an enhancement layered on top of it, never a replacement for it, so the
    worst case is a still frame of the same footage rather than an empty page.

    Returns ``"video"``, ``"still"`` or ``"gradient"`` so the caller (and the
    tests) can assert what a page actually loaded.
    """
    playable = media.available(media.HOME_VIDEO_MP4) or media.available(media.HOME_VIDEO_WEBM)
    wants_video = motion and playable

    # On the landing the plate is the video's own poster frame, so the still and
    # the first frame of the footage are the same image and the hand-off between
    # them is invisible.
    if wants_video and media.available(media.HOME_POSTER):
        plate = media.HOME_POSTER
    elif media.available(media.INTERIOR_STILL):
        plate = media.INTERIOR_STILL
    else:
        plate = None
    _plate(plate, still=not wants_video)

    mode = "still" if plate else "gradient"
    if wants_video:
        # Registration and mounting are both inside the guard.  Nothing about
        # the video is allowed to take down a page whose background is already
        # painted — the plate is the product, the video is the polish.
        try:
            component = _video_stage()
            if component is not None:
                component(height="content", width="stretch")
                mode = "video"
        except Exception:
            pass  # the plate is already on screen; nothing more to do

    # Always last, so the grade sits over whichever of the two rendered.
    _overlays(still=not wants_video)
    return mode


# ---------------------------------------------------------------------------
# 2. The landing
# ---------------------------------------------------------------------------


def landing(
    *,
    index: str,
    title_lines: Sequence[str],
    lede: str,
    meta: Sequence[tuple[str, str]] = (),
) -> None:
    """
    Render the first viewport: index, title, lede, rule, metadata.

    Deliberately *not* wrapped in a bordered card.  The brief's whole point is
    that the viewport is the hero — a panel here would reintroduce the "small
    ocean rectangle inside a dark dashboard" the design is trying to escape.
    """
    heading = "<br>".join(escape(line) for line in title_lines)
    meta_html = "".join(
        f'<div class="cs-landing__metaitem">'
        f'<span class="cs-landing__metakey">{escape(key)}</span>'
        f'<span class="cs-landing__metaval">{escape(value)}</span></div>'
        for key, value in meta
    )
    st.markdown(
        f'<div class="cs-landing">'
        f'<div class="cs-landing__index">{escape(index)}</div>'
        f'<h1 class="cs-landing__title">{heading}</h1>'
        f'<p class="cs-landing__lede">{escape(lede)}</p>'
        f'<div class="cs-landing__rule"></div>'
        f'<div class="cs-landing__meta">{meta_html}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def footnote(label: str, body: str) -> None:
    """The small educational/system fact in the bottom-left of the landing."""
    st.markdown(
        f'<div class="cs-footnote">'
        f'<div class="cs-footnote__label">{escape(label)}</div>'
        f'<div class="cs-footnote__body">{escape(body)}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def feature_card(
    *,
    label: str,
    rows: Sequence[tuple[str, str]],
) -> None:
    """
    The translucent card in the bottom-right of the landing.

    Dark glass with a backdrop blur and a hairline border — explicitly not a
    glowing cyan frame, which reads as a UI demo rather than a film title card.
    The card is markup only; the link into the dashboard is a real
    ``st.page_link`` rendered by the caller underneath it, so routing stays
    Streamlit's job.
    """
    row_html = "".join(
        f'<div class="cs-feature__row">'
        f'<span class="cs-feature__k">{escape(key)}</span>'
        f'<span class="cs-feature__v">{escape(value)}</span></div>'
        for key, value in rows
    )
    st.markdown(
        f'<div class="cs-feature">'
        f'<div class="cs-feature__label">{escape(label)}</div>'
        f"{row_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


def scroll_cue(text: str = "Scroll for the analysis") -> None:
    """A quiet marker that there is a dashboard below the first viewport."""
    st.markdown(
        f'<div class="cs-scrollcue"><span class="cs-scrollcue__line"></span>{escape(text)}</div>',
        unsafe_allow_html=True,
    )


def media_credits() -> None:
    """Render the on-screen attribution the CC BY licences require."""
    html = media.credits_html()
    if html:
        st.markdown(html, unsafe_allow_html=True)


__all__ = (
    "feature_card",
    "footnote",
    "landing",
    "media_credits",
    "scroll_cue",
    "stage",
)
