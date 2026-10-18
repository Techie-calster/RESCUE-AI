import streamlit as st
import folium

from streamlit_folium import st_folium

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="RESCUE-AI",
    page_icon="🚨",
    layout="wide"
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🚨 RESCUE-AI")
st.subheader("Emergency Incident Report")

st.write(
    "Provide basic incident information and upload an accident image "
)


# --------------------------------------------------
# INCIDENT INFORMATION
# --------------------------------------------------

st.header("Incident Information")

col1, col2 = st.columns(2)


with col1:

    incident_type = st.selectbox(
        "Incident Type ",
        [
            "Vehicle Collision",
            "Vehicle–Pedestrian Accident",
            "Multiple Vehicle Collision",
            "Vehicle Rollover",
            "Unknown"
        ]
    )
    additional_information = st.text_area(
            "Additional Information",
            placeholder="Example: Two people may be trapped inside the vehicle."
        )



with col2:

    reported_casualties = st.selectbox(
        "Reported Casualties",
        [
            "Unknown",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10+"
        ]
    )

    incident_time = st.time_input(
        "Incident Time"
    )


# --------------------------------------------------
# LOCATION
# --------------------------------------------------

st.header("Accident Location")

st.write(
    "Search for the accident area or click directly on the map."
)

# --------------------------------------------------
# DEFAULT MAP LOCATION
# --------------------------------------------------

DEFAULT_LAT = 28.6139
DEFAULT_LON = 77.2090

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "selected_location" not in st.session_state:

    st.session_state.selected_location = {
        "latitude": DEFAULT_LAT,
        "longitude": DEFAULT_LON
    }

if "selected_location" not in st.session_state:

    st.session_state.selected_location = {
        "latitude": DEFAULT_LAT,
        "longitude": DEFAULT_LON
    }


# --------------------------------------------------
# MAP
# --------------------------------------------------

latitude = st.session_state.selected_location["latitude"]
longitude = st.session_state.selected_location["longitude"]


m = folium.Map(
    location=[latitude, longitude],
    zoom_start=12
)


# Existing selected location marker

folium.Marker(
    [latitude, longitude],
    tooltip="Selected Accident Location",
    popup="Accident Location"
).add_to(m)


# --------------------------------------------------
# DISPLAY MAP
# --------------------------------------------------

map_data = st_folium(
    m,
    width=1200,
    height=450
)


# --------------------------------------------------
# HANDLE MAP CLICK
# --------------------------------------------------

if map_data and map_data["last_clicked"]:

    clicked_lat = map_data["last_clicked"]["lat"]
    clicked_lon = map_data["last_clicked"]["lng"]

    st.session_state.selected_location = {
        "latitude": clicked_lat,
        "longitude": clicked_lon
    }

    st.rerun()


# --------------------------------------------------
# SELECTED LOCATION
# --------------------------------------------------

latitude = st.session_state.selected_location["latitude"]
longitude = st.session_state.selected_location["longitude"]


st.success(
    f"Selected Location: "
    f"{latitude:.6f}, {longitude:.6f}"
)


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

st.header("Accident Image")

uploaded_image = st.file_uploader(
    "Upload an accident image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# ANALYZE BUTTON
# --------------------------------------------------

st.divider()

submit_report = st.button(
    " Submit Report",
    type="primary",
    use_container_width=True
)


# --------------------------------------------------
# PROCESS FORM
# --------------------------------------------------

if submit_report:

    if uploaded_image is None:

        st.error(
            "Please upload an accident image."
        )

    else:

        st.success(
            "Incident information received."
        )

        # ------------------------------------------
        # INCIDENT DATA
        # ------------------------------------------

        incident_data = {

            "incident_type": incident_type,

            "reported_casualties": reported_casualties,

            "location": {
                "latitude": latitude,
                "longitude": longitude
            },

            "incident_time": str(incident_time),

            "additional_information":
                additional_information
        }


        # ------------------------------------------
        # DISPLAY INCIDENT DATA
        # ------------------------------------------

        st.header("📋 Submitted Incident")

        st.json(incident_data)


        # ------------------------------------------
        # DISPLAY IMAGE
        # ------------------------------------------

        st.image(
            uploaded_image,
            caption="Uploaded Accident Image",
            use_container_width=True
        )