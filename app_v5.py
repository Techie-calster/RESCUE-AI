"""
RESCUE-AI - Phase 1-3 frontend
Incident form + live location + map + image upload.

Install:
    pip install streamlit folium streamlit-folium streamlit-js-eval

Run:
    streamlit run app.py
"""

import json
from datetime import datetime

import folium
import streamlit as st
from streamlit_folium import st_folium
from streamlit_js_eval import get_geolocation

st.set_page_config(page_title="RESCUE-AI", page_icon="🚑", layout="centered")

# ---------------------------------------------------------------- styling
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; max-width: 820px; }
    .hero {
        background: linear-gradient(135deg, #b71c1c 0%, #e53935 55%, #ff7043 100%);
        color: white; padding: 1.6rem 1.8rem; border-radius: 16px;
        margin-bottom: 1.2rem; box-shadow: 0 6px 18px rgba(183,28,28,.25);
    }
    .hero h1 { margin: 0; font-size: 2rem; color: white; }
    .hero p  { margin: .3rem 0 0 0; opacity: .92; font-size: .95rem; }
    .section-title {
        font-size: 1.15rem; font-weight: 700; margin: 0 0 .4rem 0;
        color: #b71c1c;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px; }
    .stButton > button {
        border-radius: 10px; font-weight: 600; padding: .55rem 1.1rem;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #c62828, #ef5350); border: none;
        width: 100%; padding: .75rem; font-size: 1.05rem;
    }
    .coord-chip {
        display: inline-block; background: #ffebee; color: #b71c1c;
        padding: .3rem .8rem; border-radius: 999px; font-weight: 600;
        font-size: .9rem; margin: .3rem 0;
    }
    </style>
    <div class="hero">
        <h1>🚑 RESCUE-AI</h1>
        <p>Report a road accident. This is a simulation / decision-support system,
        not a real emergency dispatch service.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

DEFAULT_CENTER = (28.6139, 77.2090)  # Delhi, used only as the initial map view

INCIDENT_TYPES = {
    "Vehicle Collision": "vehicle_collision",
    "Vehicle–Pedestrian Accident": "vehicle_pedestrian_accident",
    "Multiple Vehicle Collision": "multiple_vehicle_collision",
    "Vehicle Rollover": "vehicle_rollover",
    "Unknown": "unknown",
}
CASUALTY_OPTIONS = ["Unknown", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10+"]

# ---------------------------------------------------------------- session state
ss = st.session_state
ss.setdefault("lat", None)
ss.setdefault("lon", None)
ss.setdefault("request_gps", False)      # True only after the button is clicked
ss.setdefault("last_click", None)        # last map click already processed
ss.setdefault("map_center", DEFAULT_CENTER)
ss.setdefault("map_zoom", 5)

# ---------------------------------------------------------------- incident details
with st.container(border=True):
    st.markdown('<div class="section-title">📋 Incident Details</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        incident_label = st.selectbox("Incident type", list(INCIDENT_TYPES))
    with col2:
        casualties_label = st.selectbox(
            "Reported casualties",
            CASUALTY_OPTIONS,
            help="The number reported by you. This is NOT detected by the Vision Agent.",
        )
    incident_time = st.time_input("Approximate incident time", value=datetime.now().time())
    additional_info = st.text_area(
        "Additional information (optional)",
        placeholder="e.g. Truck collided with two cars near the intersection. Traffic is blocked.",
    )

# ---------------------------------------------------------------- location
with st.container(border=True):
    st.markdown('<div class="section-title">📍 Accident Location</div>', unsafe_allow_html=True)

    # Browser geolocation: permission is requested ONLY after the button click.
    if st.button("📍 Use My Current Location"):
        ss.request_gps = True

    if ss.request_gps:
        with st.spinner("Waiting for browser location permission..."):
            loc = get_geolocation(component_key="gps")  # None until the browser answers
        if loc and "coords" in loc:
            ss.lat = loc["coords"]["latitude"]
            ss.lon = loc["coords"]["longitude"]
            ss.map_center = (ss.lat, ss.lon)
            ss.map_zoom = 17
            ss.request_gps = False
            st.rerun()
        elif loc and "error" in loc:
            ss.request_gps = False
            st.error("Could not get your location (permission denied or unavailable). "
                     "You can click on the map to place the pin manually.")

    if ss.lat is not None:
        st.markdown(
            f'<span class="coord-chip">Current pin: {ss.lat:.6f}, {ss.lon:.6f}</span>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Click the button above, or click directly on the map to place the pin.")

    m = folium.Map(location=ss.map_center, zoom_start=ss.map_zoom, control_scale=True)
    if ss.lat is not None:
        folium.Marker(
            [ss.lat, ss.lon],
            tooltip="Accident location",
            icon=folium.Icon(color="red", icon="warning-sign"),
        ).add_to(m)

    map_state = st_folium(
        m,
        height=380,
        use_container_width=True,
        key="location_map",
        returned_objects=["last_clicked"],
    )

    # Click on the map -> move the pin to the exact accident location
    click = map_state.get("last_clicked") if map_state else None
    if click and click != ss.last_click:
        ss.last_click = click
        ss.lat, ss.lon = click["lat"], click["lng"]
        ss.map_center = (ss.lat, ss.lon)
        ss.map_zoom = 17
        st.rerun()

    st.caption("Click on the map to move the pin to the exact accident location.")

    location_description = st.text_input(
        "Location description",
        placeholder="Near India Gate, New Delhi",
    )

# ---------------------------------------------------------------- image
with st.container(border=True):
    st.markdown('<div class="section-title">📷 Accident Image</div>', unsafe_allow_html=True)
    image_file = st.file_uploader("Upload an image of the scene", type=["jpg", "jpeg", "png"])
    if image_file:
        st.image(image_file, caption="Uploaded image", use_container_width=True)

# ---------------------------------------------------------------- submit
if st.button("🚨 Submit Report", type="primary"):
    errors = []
    if ss.lat is None:
        errors.append("Please set the accident location on the map.")
    if not location_description.strip():
        errors.append("Please enter a location description.")
    if image_file is None:
        errors.append("Please upload an accident image.")

    if errors:
        for e in errors:
            st.error(e)
    else:
        if casualties_label == "Unknown":
            reported_casualties = None
        else:
            reported_casualties = int(casualties_label.rstrip("+"))  # "10+" -> 10

        incident_data = {
            "incident_type": INCIDENT_TYPES[incident_label],
            "reported_casualties": reported_casualties,
            "incident_time": incident_time.strftime("%H:%M"),
            "location": {
                "latitude": round(ss.lat, 6),
                "longitude": round(ss.lon, 6),
                "description": location_description.strip(),
            },
            "additional_information": additional_info.strip(),
            "image_filename": image_file.name,
            "submitted_at": datetime.now().isoformat(timespec="seconds"),
        }
        image_bytes = image_file.getvalue()  # passed separately to the Vision Agent

        report_json = json.dumps(incident_data, indent=2, ensure_ascii=False)

        st.success("✅ Report submitted. Emergency analysis will now run in the background.")
        st.markdown("**Incident report (JSON)**")
        st.code(report_json, language="json")
        st.download_button(
            "⬇️ Download report.json",
            data=report_json,
            file_name="incident_report.json",
            mime="application/json",
        )

        # ---- Backend hand-off (user-facing part ends here) ----
        # incident_data + image_bytes -> Vision Agent -> Incident State -> ...
        # from vision_agent import analyze_image
        # vision_result = analyze_image(image_bytes)