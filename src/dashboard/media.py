"""
src/dashboard/media.py — the dashboard's licensed media registry.

What this is
------------
One place that knows every piece of third-party footage the dashboard ships:
where the file is, how to reach it from the browser, who made it, under what
licence, and what credit has to appear on screen.  Nothing else in the
dashboard hard-codes a media path.

Why a registry rather than a URL in a template
----------------------------------------------
Both clips are Creative Commons **Attribution** licences.  Redistribution is
permitted, but only with credit and an indication that the file was changed.
Keeping the credit next to the path means a future edit cannot quietly ship an
asset with no attribution: ``tests/test_dashboard_media.py`` walks this registry
and fails if any shipped file lacks a creator, licence, source URL or credit
line, and ``static/media/ATTRIBUTION.md`` carries the long form.

How the browser reaches the files
---------------------------------
Streamlit's static file server (``server.enableStaticServing`` in
``.streamlit/config.toml``) publishes a ``static`` folder at ``/app/static/``.
That folder is resolved **next to the main script**, not next to the working
directory — ``streamlit/file_util.py`` computes
``Path(main_script_path).parent / "static"`` — so for an entrypoint of
``src/dashboard/app.py`` the assets must live in ``src/dashboard/static/``.
A repository-root ``static/`` silently 404s.

Serving them this way means a first-party path on the app's own origin: no CDN
request, no hot-linking of someone else's bandwidth, and the files travel with
the Docker image under the existing ``COPY src/ ./src/``.

Degradation
-----------
The media is deliberately *optional*.  ``available()`` reports whether a file is
actually on disk, and every consumer falls back to a graded CSS gradient when it
is not.  A checkout without the binaries — or an image built before they were
added — still renders a coherent dashboard rather than a broken one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------

#: The dashboard package directory — Streamlit resolves the static folder
#: relative to the main script, which lives here.
_PKG = Path(__file__).resolve().parent

#: Where the assets live on disk, and the URL prefix Streamlit serves them at.
MEDIA_DIR = _PKG / "static" / "media"
MEDIA_URL = "app/static/media"


@dataclass(frozen=True)
class MediaCredit:
    """The attribution contract for one shipped asset.

    Every field is required by the licences in use: CC BY needs the title, the
    creator, the licence, a link to the licence, and a statement of changes.
    """

    #: Files this credit covers, relative to ``MEDIA_DIR``.
    files: tuple[str, ...]
    title: str
    creator: str
    source_url: str
    licence: str
    licence_url: str
    #: The one-line credit rendered in the dashboard UI.
    credit: str
    #: What was done to the original — required by CC BY.
    changes: str
    #: Free-text note about why this asset was chosen.
    note: str = ""
    extra_creators: tuple[str, ...] = field(default_factory=tuple)


MEDIA_CREDITS: tuple[MediaCredit, ...] = (
    MediaCredit(
        files=(
            "cs-reef.mp4",
            "cs-reef.webm",
            "cs-reef-poster.jpg",
            "cs-reef-still.jpg",
        ),
        title=(
            "First records of the species Hemitaurichthys polylepis at the "
            "Chesterfield-Bellona reef complex in the Coral Sea Marine Park"
        ),
        creator="D. Pelletier, A. Powell, W. Roman, L. Carpentier / IFREMER",
        source_url=(
            "https://commons.wikimedia.org/wiki/"
            "File:First_records_of_the_species_Hemitaurichthys_polylepis_at_the_"
            "Chesterfield-Bellona_reef_complex_in_the_Coral_Sea_Marine_Park_"
            "(Ifremer_00675-78693).webm"
        ),
        licence="CC BY 4.0",
        licence_url="https://creativecommons.org/licenses/by/4.0/",
        credit=(
            "Chesterfield-Bellona reef survey — D. Pelletier, A. Powell, W. Roman, "
            "L. Carpentier / IFREMER (CC BY 4.0), via Wikimedia Commons"
        ),
        changes=(
            "Trimmed to a 9-second loop, mirrored horizontally, denoised, resampled "
            "to 24 fps and scaled to 1600x900, re-encoded to VP9 and H.264, audio "
            "removed. The interior still is one frame from the same clip, blurred and "
            "graded down."
        ),
        note=(
            "An unbaited underwater camera survey of a real coral reef in the Coral "
            "Sea Marine Park, New Caledonia — the same task this dashboard models. "
            "One source now supplies the landing video, its poster and the still "
            "behind the analytical pages."
        ),
    ),
)

# ---------------------------------------------------------------------------
# Named assets
#
# Consumers ask for a role ("the landing backdrop"), never a filename.
# ---------------------------------------------------------------------------

HOME_VIDEO_MP4 = "cs-reef.mp4"
HOME_VIDEO_WEBM = "cs-reef.webm"
HOME_POSTER = "cs-reef-poster.jpg"
INTERIOR_STILL = "cs-reef-still.jpg"

#: Shown whenever a file is missing, and as the paint before the poster loads.
#: Graded from the same footage's own colour, so the fallback is not a different
#: design — it is the same frame with the detail removed.
FALLBACK_GRADIENT = (
    "radial-gradient(120% 90% at 68% 26%, #2c7f86 0%, #185f6d 34%, transparent 72%), "
    "linear-gradient(168deg, #1d7f88 0%, #12586b 26%, #0a3a50 52%, #05243a 76%, #021320 100%)"
)


def path(name: str) -> Path:
    """Absolute path to a media file (whether or not it exists)."""
    return MEDIA_DIR / name


def available(name: str) -> bool:
    """True when *name* is actually on disk and non-empty.

    Consumers branch on this rather than assuming: the binaries are large and a
    checkout may legitimately not have them.
    """
    file = path(name)
    try:
        return file.is_file() and file.stat().st_size > 0
    except OSError:
        return False


def url(name: str) -> str:
    """Browser-facing URL for a media file, served from the app's own origin."""
    return f"{MEDIA_URL}/{name}"


def size_bytes(name: str) -> int:
    """On-disk size, or 0 when absent.  Used by the media-weight report."""
    try:
        return path(name).stat().st_size
    except OSError:
        return 0


def credit_for(name: str) -> MediaCredit | None:
    """Return the credit covering *name*, or ``None`` if it is unregistered."""
    for entry in MEDIA_CREDITS:
        if name in entry.files:
            return entry
    return None


def shipped_files() -> tuple[str, ...]:
    """Every registered file that is actually present on disk."""
    return tuple(name for entry in MEDIA_CREDITS for name in entry.files if available(name))


def total_bytes() -> int:
    """Combined weight of the shipped media, for the performance report."""
    return sum(size_bytes(name) for name in shipped_files())


def credits_html(*, compact: bool = True) -> str:
    """Render the on-screen attribution required by both CC BY licences.

    Only credits for assets that are actually shipped are rendered — crediting a
    file the build does not contain would be inaccurate.
    """
    present = [e for e in MEDIA_CREDITS if any(available(f) for f in e.files)]
    if not present:
        return ""
    rows = []
    for entry in present:
        who = entry.creator
        if entry.extra_creators:
            who += ", " + ", ".join(entry.extra_creators)
        rows.append(
            f'<div class="cs-credit">'
            f'<span class="cs-credit__title">{entry.title}</span> — {who} · '
            f'<a href="{entry.licence_url}" target="_blank" rel="noopener noreferrer">'
            f"{entry.licence}</a> · "
            f'<a href="{entry.source_url}" target="_blank" rel="noopener noreferrer">source</a>'
            + ("" if compact else f'<div class="cs-credit__changes">{entry.changes}</div>')
            + "</div>"
        )
    return f'<div class="cs-credits">{"".join(rows)}</div>'


__all__ = (
    "FALLBACK_GRADIENT",
    "HOME_POSTER",
    "HOME_VIDEO_MP4",
    "HOME_VIDEO_WEBM",
    "INTERIOR_STILL",
    "MEDIA_CREDITS",
    "MEDIA_DIR",
    "MEDIA_URL",
    "MediaCredit",
    "available",
    "credit_for",
    "credits_html",
    "path",
    "shipped_files",
    "size_bytes",
    "total_bytes",
    "url",
)
