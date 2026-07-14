"""
Page 5 — Predict.

Provides a sensor-input form that calls POST /predict/both on the FastAPI
service and displays both reef-health and restoration-suitability results.

Design rules enforced here
--------------------------
- No model is loaded, trained, registered or promoted by this page.
- The payload sent to the API never includes 'reef_health' or
  'restoration_suitability' (target labels excluded by build_predict_payload).
- Predictions are only triggered by an explicit form submission button.
- No automatic prediction on widget change.
- Connection errors are shown as a user-friendly message, never as a traceback.
"""

from __future__ import annotations

import streamlit as st

from src.dashboard.api_client import APIClient, APIError
from src.dashboard.components import (
    DISCLAIMER_SHORT,
    build_predict_payload,
    render_prediction,
    render_sidebar,
    set_page,
)

set_page("Predict")

render_sidebar(show_region_filter=False)

st.title("Live Prediction Interface")
st.caption(
    "Enter raw sonar and environmental sensor values to request predictions "
    "from the FastAPI champion models. All predictions are on synthetic data."
)

st.info(
    "Predictions are computed by the FastAPI service (POST /predict/both). "
    "This page never loads or calls any model directly."
)

# ---------------------------------------------------------------------------
# Example defaults (Gulf of Mannar, moderate healthy reef conditions)
# ---------------------------------------------------------------------------

DEFAULTS = {
    "region": "Gulf of Mannar",
    "depth_m": 5.0,
    "water_temperature_c": 27.5,
    "ph": 8.1,
    "salinity_ppt": 35.0,
    "dissolved_oxygen_mg_l": 7.0,
    "turbidity_ntu": 2.0,
    "light_intensity": 800.0,
    "current_speed_m_s": 0.2,
    "sonar_backscatter": -15.0,
    "rugosity_index": 3.5,
    "hard_substrate_percentage": 60.0,
    "acoustic_complexity_index": 0.7,
    "coral_cover_percentage": 45.0,
    "bleaching_percentage": 5.0,
    "disease_percentage": 2.0,
}

REGION_OPTIONS = [
    "Lakshadweep",
    "Gulf of Mannar",
    "Gulf of Kutch",
    "Andaman and Nicobar Islands",
]

# ---------------------------------------------------------------------------
# Prediction form
# ---------------------------------------------------------------------------

with st.form("predict_form"):
    st.subheader("Sensor Input")
    st.caption(
        "All 16 inference features are required. Numeric bounds match the API validation schema."
    )

    region = st.selectbox(
        "Reef Region",
        options=REGION_OPTIONS,
        index=REGION_OPTIONS.index(DEFAULTS["region"]),
        help="One of the four supported Indian prototype reef regions.",
    )

    st.markdown("**Depth and Water Column**")
    c1, c2, c3 = st.columns(3)
    with c1:
        depth_m = st.number_input(
            "Depth (m)",
            min_value=0.0,
            max_value=50.0,
            value=DEFAULTS["depth_m"],
            step=0.5,
            help="Sensor depth in metres (0–50 m).",
        )
    with c2:
        water_temperature_c = st.number_input(
            "Water Temperature (°C)",
            min_value=10.0,
            max_value=42.0,
            value=DEFAULTS["water_temperature_c"],
            step=0.1,
            help="In-situ water temperature (10–42 °C).",
        )
    with c3:
        salinity_ppt = st.number_input(
            "Salinity (ppt)",
            min_value=20.0,
            max_value=50.0,
            value=DEFAULTS["salinity_ppt"],
            step=0.5,
            help="Practical salinity in ppt (20–50).",
        )

    st.markdown("**Water Chemistry**")
    c4, c5, c6 = st.columns(3)
    with c4:
        ph = st.number_input(
            "pH",
            min_value=7.0,
            max_value=9.0,
            value=DEFAULTS["ph"],
            step=0.01,
            help="Seawater pH on the total scale (7.0–9.0).",
        )
    with c5:
        dissolved_oxygen_mg_l = st.number_input(
            "Dissolved Oxygen (mg/L)",
            min_value=0.0,
            max_value=15.0,
            value=DEFAULTS["dissolved_oxygen_mg_l"],
            step=0.1,
            help="Dissolved oxygen concentration (0–15 mg/L).",
        )
    with c6:
        turbidity_ntu = st.number_input(
            "Turbidity (NTU)",
            min_value=0.0,
            max_value=100.0,
            value=DEFAULTS["turbidity_ntu"],
            step=0.5,
            help="Water turbidity in NTU (0–100).",
        )

    st.markdown("**Hydrodynamics and Light**")
    c7, c8 = st.columns(2)
    with c7:
        light_intensity = st.number_input(
            "Light Intensity (µmol m⁻² s⁻¹)",
            min_value=0.0,
            max_value=3000.0,
            value=DEFAULTS["light_intensity"],
            step=10.0,
            help="PAR light intensity (0–3000 µmol m⁻² s⁻¹).",
        )
    with c8:
        current_speed_m_s = st.number_input(
            "Current Speed (m/s)",
            min_value=0.0,
            max_value=5.0,
            value=DEFAULTS["current_speed_m_s"],
            step=0.05,
            help="Near-bottom current speed (0–5 m/s).",
        )

    st.markdown("**Sonar / Structural Features**")
    c9, c10, c11, c12 = st.columns(4)
    with c9:
        sonar_backscatter = st.number_input(
            "Sonar Backscatter (dB)",
            min_value=-60.0,
            max_value=0.0,
            value=DEFAULTS["sonar_backscatter"],
            step=0.5,
            help="Acoustic backscatter in dB (−60 to 0). Hard reef ≈ −10 to −3 dB.",
        )
    with c10:
        rugosity_index = st.number_input(
            "Rugosity Index",
            min_value=1.0,
            max_value=10.0,
            value=DEFAULTS["rugosity_index"],
            step=0.1,
            help="Benthic rugosity (1 = flat, ≥3.5 = complex reef structure).",
        )
    with c11:
        hard_substrate_percentage = st.number_input(
            "Hard Substrate (%)",
            min_value=0.0,
            max_value=100.0,
            value=DEFAULTS["hard_substrate_percentage"],
            step=1.0,
            help="Hard substrate coverage percentage (0–100).",
        )
    with c12:
        acoustic_complexity_index = st.number_input(
            "Acoustic Complexity Index",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULTS["acoustic_complexity_index"],
            step=0.01,
            help="Normalised acoustic complexity [0, 1].",
        )

    st.markdown("**Biological Condition**")
    c13, c14, c15 = st.columns(3)
    with c13:
        coral_cover_percentage = st.number_input(
            "Coral Cover (%)",
            min_value=0.0,
            max_value=100.0,
            value=DEFAULTS["coral_cover_percentage"],
            step=1.0,
            help="Live coral cover percentage (0–100).",
        )
    with c14:
        bleaching_percentage = st.number_input(
            "Bleaching (%)",
            min_value=0.0,
            max_value=100.0,
            value=DEFAULTS["bleaching_percentage"],
            step=0.5,
            help="Bleached coral tissue fraction (0–100).",
        )
    with c15:
        disease_percentage = st.number_input(
            "Disease (%)",
            min_value=0.0,
            max_value=100.0,
            value=DEFAULTS["disease_percentage"],
            step=0.5,
            help="Disease-affected colony fraction (0–100).",
        )

    submitted = st.form_submit_button(
        "Request Prediction",
        type="primary",
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# Prediction result (only after explicit submit)
# ---------------------------------------------------------------------------

if submitted:
    form_values = {
        "region": region,
        "depth_m": depth_m,
        "water_temperature_c": water_temperature_c,
        "ph": ph,
        "salinity_ppt": salinity_ppt,
        "dissolved_oxygen_mg_l": dissolved_oxygen_mg_l,
        "turbidity_ntu": turbidity_ntu,
        "light_intensity": light_intensity,
        "current_speed_m_s": current_speed_m_s,
        "sonar_backscatter": sonar_backscatter,
        "rugosity_index": rugosity_index,
        "hard_substrate_percentage": hard_substrate_percentage,
        "acoustic_complexity_index": acoustic_complexity_index,
        "coral_cover_percentage": coral_cover_percentage,
        "bleaching_percentage": bleaching_percentage,
        "disease_percentage": disease_percentage,
    }

    # Explicitly exclude target labels before sending to API
    payload = build_predict_payload(form_values)

    with st.spinner("Requesting prediction from CoralSense API…"):
        try:
            client = APIClient()
            result = client.predict_both(payload)
            predict_ok = True
        except APIError as exc:
            st.error(f"Prediction failed: {exc}")
            predict_ok = False

    if predict_ok:
        st.success("Prediction received from the FastAPI service.")
        st.divider()

        res_col, rest_col = st.columns(2)
        with res_col:
            render_prediction(result["health"], "Reef Health")
        with rest_col:
            render_prediction(result["restoration"], "Restoration Suitability")

        st.divider()
        st.caption(
            f"Disclaimer: {DISCLAIMER_SHORT} "
            "Results displayed are from a model trained on synthetic data only."
        )

        with st.expander("Raw API response"):
            st.json(result)
