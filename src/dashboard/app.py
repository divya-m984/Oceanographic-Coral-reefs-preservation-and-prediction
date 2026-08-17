"""
src/dashboard/app.py — CoralSense router.

Launch
------
    python -m streamlit run src/dashboard/app.py

This file is the entrypoint and does one job: declare the pages and run the
navigation.  All page content lives in ``src/dashboard/views/``.

Navigation
----------
The dashboard used Streamlit's automatic ``pages/`` discovery, which can only
render navigation in the sidebar.  It now declares pages explicitly with
``st.Page`` and runs ``st.navigation(..., position="top")``, so the primary
navigation is a horizontal bar and the sidebar is free for page filters.

**URLs are preserved.**  Automatic discovery derived a URL from each filename
(``1_Overview.py`` -> ``/Overview``).  Every ``st.Page`` below pins the same
value with ``url_path``, so existing links keep working even though the scripts
moved from ``pages/`` to ``views/``.  The directory had to be renamed: a
``pages/`` folder next to the entrypoint is auto-discovered, and that would have
produced a second, competing set of navigation entries.

SYNTHETIC-DATA DISCLAIMER
--------------------------
All observations and predictions shown in this dashboard are based on
synthetically generated data. They must NOT be used to guide real-world
coral-reef conservation decisions.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.dashboard import media

_VIEWS = Path(__file__).resolve().parent / "views"

# ---------------------------------------------------------------------------
# Page configuration
#
# Under st.navigation the entrypoint owns the page config; individual views must
# not call st.set_page_config again.  The sidebar starts collapsed because the
# landing has nothing to put in it — only the three filtered pages do.
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CoralSense",
    page_icon=":ocean:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_wordmark = media.path("cs-wordmark.svg")
if _wordmark.is_file():
    st.logo(str(_wordmark), size="large", link=None)


def _page(script: str, title: str, url_path: str, **kwargs) -> st.Page:
    """Declare one page, pinning the URL its filename used to produce."""
    return st.Page(_VIEWS / script, title=title, url_path=url_path, **kwargs)


# The six labels the brief names as primary, then the operational pages.
# Streamlit's top navigation moves whatever does not fit into an overflow menu,
# so ordering here is what decides visibility on a narrow window.
PRIMARY = [
    _page("0_Home.py", "Home", "home", default=True),
    _page("1_Overview.py", "Overview", "Overview"),
    _page("2_Reef_Map.py", "Reef Map", "Reef_Map"),
    _page("3_Habitat_Health.py", "Habitat", "Habitat_Health"),
    _page("4_Restoration_Planning.py", "Restoration", "Restoration_Planning"),
    _page("5_Predict.py", "Predict", "Predict"),
    _page("6_Model_Performance.py", "Models", "Model_Performance"),
]

OPERATIONS = [
    _page("7_MLOps_Status.py", "MLOps Status", "MLOps_Status"),
    _page("8_Drift_Monitoring.py", "Drift Monitoring", "Drift_Monitoring"),
    _page("9_Governance.py", "Governance", "Governance"),
]

st.navigation({"": PRIMARY, "Operations": OPERATIONS}, position="top").run()
