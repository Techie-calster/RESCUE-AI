"""
accident form + live location + map + image upload version 3
"""
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
ss.setdefault("gps", None)               # (lat, lon, accuracy_m) as reported by the browser
ss.setdefault("request_gps", False)      # True only after the button is clicked
ss.setdefault("last_click", None)        # last map click already processed
ss.setdefault("map_center", DEFAULT_CENTER)
ss.setdefault("map_zoom", 5)

st.title("🚑 RESCUE-AI")
st.caption("Simulation / decision-support system for road accident response. "
           "Not a real emergency dispatch system.")

# ---------------------------------------------------------------- incident details
st.subheader("Incident Details")
incident_label = st.selectbox("Incident type", list(INCIDENT_TYPES))
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
st.subheader("📍 Accident Location")

# 1) Browser geolocation: permission is requested ONLY after the button click.
if st.button("📍 Use My Current Location"):
    ss.request_gps = True

if ss.request_gps:
    with st.spinner("Waiting for browser location permission..."):
        loc = get_geolocation(component_key="gps")  # None until the browser answers
    if loc and "coords" in loc:
        ss.lat = loc["coords"]["latitude"]
        ss.lon = loc["coords"]["longitude"]
        ss.gps = (ss.lat, ss.lon, loc["coords"].get("accuracy"))
        ss.map_center = (ss.lat, ss.lon)
        ss.map_zoom = 17
        ss.request_gps = False
        st.rerun()
    elif loc and "error" in loc:
        ss.request_gps = False
        st.error("Could not get your location (permission denied or unavailable). "
                 "You can click on the map to place the pin manually.")

if ss.lat is not None:
    st.write(f"**Current pin:** {ss.lat:.6f}, {ss.lon:.6f}")
    if ss.gps and ss.gps[2]:
        st.caption(f"Device location accuracy: about ±{ss.gps[2]:.0f} m (blue circle). "
                   "Click the map to move the pin to the exact spot.")
else:
    st.info("Click the button above, or click directly on the map to place the pin.")

# 2) Map with pin
m = folium.Map(location=ss.map_center, zoom_start=ss.map_zoom, control_scale=True)
if ss.gps and ss.gps[2]:
    # Blue circle = area where the device thinks you are; the pin can be anywhere.
    folium.Circle(
        [ss.gps[0], ss.gps[1]],
        radius=ss.gps[2],
        color="#1a73e8",
        fill=True,
        fill_opacity=0.12,
        tooltip=f"Your device location (±{ss.gps[2]:.0f} m)",
    ).add_to(m)
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

# 3) Click on the map -> move the pin to the exact accident location
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
st.subheader("Accident Image")
image_file = st.file_uploader("Upload an image of the scene", type=["jpg", "jpeg", "png"])
if image_file:
    st.image(image_file, caption="Uploaded image", use_container_width=True)

# ---------------------------------------------------------------- submit
st.divider()
if st.button("Analyze Incident", type="primary"):
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
        }
        image_bytes = image_file.getvalue()  # passed separately to the Vision Agent

        st.success("Incident data collected.")
        st.json(incident_data)

        # ---- Phase 3 hook: connect the Vision Agent here ----
        # from vision_agent import analyze_image
        # vision_result = analyze_image(image_bytes)
        # -> {"accident_detected": True, "confidence": 0.91}