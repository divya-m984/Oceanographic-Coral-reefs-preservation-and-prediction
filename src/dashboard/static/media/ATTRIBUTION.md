# Media attribution and licences

Every media file in this directory is third-party footage used under a Creative
Commons licence that permits redistribution, including in this public
repository and in the deployed dashboard image. The licence requires
attribution, which the dashboard renders on screen (see
`src/dashboard/media.py` → `MEDIA_CREDITS`, surfaced in the landing footer).

No BBC, *Blue Planet*, or other commercially licensed broadcast footage is used
anywhere in this project.

---

## `cs-reef.webm` / `cs-reef.mp4` / `cs-reef-poster.jpg` / `cs-reef-still.jpg`

One source supplies the whole dashboard: the moving background of the landing
viewport, its poster frame, and the quiet still behind the analytical pages.

| | |
|---|---|
| **Title** | *First records of the species Hemitaurichthys polylepis at the Chesterfield-Bellona reef complex in the Coral Sea Marine Park* (Ifremer 00675-78693) |
| **Creators** | Dominique Pelletier, Abigail Powell, William Roman, Liliane Carpentier |
| **Institution** | IFREMER — Institut français de recherche pour l'exploitation de la mer |
| **Provenance** | <https://image.ifremer.fr/data/00675/78693> |
| **Source file** | <https://commons.wikimedia.org/wiki/File:First_records_of_the_species_Hemitaurichthys_polylepis_at_the_Chesterfield-Bellona_reef_complex_in_the_Coral_Sea_Marine_Park_(Ifremer_00675-78693).webm> |
| **Licence** | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| **Licence text** | <https://creativecommons.org/licenses/by/4.0/> |
| **Attribution required** | Yes |
| **Redistribution permitted** | Yes — CC BY 4.0 §2(a) grants the right to reproduce and share the material in any medium |
| **Modifications permitted** | Yes — CC BY 4.0 grants adaptation rights, provided changes are indicated |
| **Restrictions** | None beyond attribution and indicating changes |
| **Subject** | Unbaited underwater camera survey of a coral reef, Chesterfield-Bellona atolls, Coral Sea Marine Park, New Caledonia (2013) |

**Required credit line** (rendered by the dashboard):

> Chesterfield-Bellona reef survey — D. Pelletier, A. Powell, W. Roman,
> L. Carpentier / IFREMER (CC BY 4.0), via Wikimedia Commons.

**Changes made** (CC BY 4.0 §3(a)(1)(B) requires these be indicated):

- **Cropped in time** — trimmed to a 9-second loop starting at 00:00.2.
- **Mirrored horizontally** — so the reef mound falls on the centre/right of the
  frame and the open water on the left, where the landing copy sits. The clip is
  used as decorative page background only, never as data or evidence.
- **Denoised** — light temporal/spatial denoise (`hqdn3d`) to remove suspended
  particle noise, which otherwise consumes a large share of the bitrate.
- **Resampled** — to 24 fps.
- **Rescaled** — to 1600x900 from the 1920x1080 original.
- **Transcoded** — to VP9 (WebM, primary) and H.264 (MP4, fallback); audio
  removed.
- **Poster frame extracted** — `cs-reef-poster.jpg` is the first frame of the
  loop, so the still and the video's opening frame are the same image.
- **Interior still graded** — `cs-reef-still.jpg` is a single frame from 00:04 of
  the same clip, blurred and reduced in brightness, contrast and saturation so it
  can sit behind charts without competing with them.

**Why this footage:** it is a real reef-monitoring survey from a marine research
institute, which is the same task this dashboard models. It has a dense coral
mound with visible branching structure, schooling reef fish, natural sunlight
and water haze, real front-to-back depth, and open blue water for the landing
copy to sit on.

---

## `cs-wordmark.svg`

Self-authored. No third-party rights.

---

## Sources reviewed and rejected

Recorded so the next media change does not repeat the research.

- **NOAA Ocean Exploration** — genuinely public domain (17 U.S.C. §105), but the
  ROV material is deep-sea and artificially lit, which does not match a sunlit
  reef look.
- ***Tropical Fish Banner Fish on Coral Reef*** (Wikimedia Commons, uploader
  `underwatercam`) — visually the strongest candidate found, but **rejected on
  licence grounds**. The Commons page carries a CC BY 3.0 tag that "has not yet
  been reviewed by an administrator or reviewer", and the stated origin is
  Videvo, which operates three different licence tiers (Videvo Standard, Videvo
  Attribution, and CC BY 3.0) and states that *redistribution of raw, unedited
  clips is prohibited under all tiers*. Since bundling the file in a public
  repository is exactly that kind of redistribution, and the specific tier for
  this clip could not be confirmed at source, it was not used.
- ***Killer whales swimming in the wild*** (Fair Projects / Steve Hathaway, CC BY
  3.0) — correctly licensed and previously shipped as the landing background,
  but replaced by reef footage on the product brief. Its files have been deleted
  from this directory rather than left as dead weight in the image.
- **Pexels / Coverr / Pixabay** — permissive but bespoke licences, harder to
  audit for open-source redistribution than a standard Creative Commons licence.

## If you replace this media

Keep `src/dashboard/media.py` in step: `MEDIA_CREDITS` is what the dashboard
renders on screen, and `tests/test_dashboard_visuals.py::TestMediaLicensing`
fails if a shipped asset has no licence, creator, source URL, credit line and
statement of changes recorded here.
