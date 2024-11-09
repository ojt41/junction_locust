import streamlit as st
#import folium
#from streamlit_folium import folium_static
import streamlit.components.v1 as components

# Configure the page
st.set_page_config(
    page_title="Interactive Map",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve the layout and prevent overlap
st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-bottom: 1rem;
    }
    /* Ensure columns don't overlap */
    [data-testid="stHorizontalBlock"] {
        gap: 2rem;
    }
    /* Adjust map container */
    [data-testid="stDecoration"] {
        overflow: hidden;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.title("Map Controls")

    if st.button("Reset View"):
        pass

    if st.button("Toggle Labels"):
        pass

    zoom_level = st.slider("Zoom Level", 1, 18, 10)

    st.subheader("Layer Controls")
    show_markers = st.checkbox("Show Markers", True)
    map_style = st.selectbox(
        "Map Style",
        ["OpenStreetMap", "Stamen Terrain", "Cartodb Positron"]
    )

st.title("Interactive Map")

# Create container for the main content
main_container = st.container()

# Use columns with specific ratios and add padding
with main_container:
    # Adjust the ratio to prevent overlap (changed from 3:1 to 2:1)
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        try:
            with open('../maps/locust_observations_20241109_183935.html', 'r', encoding='utf-8') as f:
                html_content = f.read()
            components.html(html_content, height=600, width=600)
        except FileNotFoundError:
            st.error("Please place your HTML map file in the same directory as this script")

        try:
            with open('../maps/locust_predictions_2024_1_20241109_184300.html', 'r', encoding='utf-8') as f:
                html_content = f.read()
            components.html(html_content, height=600, width=600)
        except FileNotFoundError:
            st.error("Please place your HTML map file in the same directory as this script")

    with col2:
        # Add a container to ensure proper spacing
        info_container = st.container()

        with info_container:
            st.subheader("Information")
            st.write("Current zoom level:", zoom_level)
            st.write("Markers enabled:", show_markers)
            st.write("Selected style:", map_style)

            st.subheader("Statistics")
            st.metric(label="Total Markers", value="1")
            st.metric(label="Active Layers", value="1")

# Footer with proper spacing
st.markdown("---")
st.markdown("Made with Streamlit")
