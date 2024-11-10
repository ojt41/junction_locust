import streamlit as st
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

    map_style = st.selectbox(
        "Choose map",
        ["Observations", "Predictions", "All"]
    )

# Track sidebar state
if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = True

st.title("Locust Maps")

# Create container for the main content
main_container = st.container()

# Use columns with specific ratios and add padding
with main_container:
    # Adjust the ratio to prevent overlap (changed from 3:1 to 2:1)
    col1, col2 = st.columns([3, 1], gap="large")

    with col1:
        map_width = 750 if st.session_state.sidebar_state else 1000

        if map_style == "Observations" or map_style == "All":
            try:
                with open('./maps/locust_observations_20241109_183935.html', 'r', encoding='utf-8') as f:
                    html_content = f.read()
                components.html(html_content, height=600, width=map_width)
            except FileNotFoundError:
                st.error("Please place your HTML map file in the same directory as this script")
        if map_style == "Predictions" or map_style == "All":
            try:
                with open('./maps/locust_predictions_2024_1_20241109_184300.html', 'r', encoding='utf-8') as f:
                    html_content = f.read()
                components.html(html_content, height=600, width=map_width)
            except FileNotFoundError:
                st.error("Please place your HTML map file in the same directory as this script")

    with col2:
        # Add a container to ensure proper spacing
        info_container = st.container()

        with info_container:
            st.subheader("Information")
            st.write("Map 1: Locust Observations - 2018-2021")
            st.write("Map 2: Predictions Heat Map of Locust Swarms - 2024")

            # Update sidebar state
            if st.button('Expand Map'):
                st.session_state.sidebar_state = not(st.session_state.sidebar_state)

# Footer with proper spacing
st.markdown("---")
st.markdown("Made with Streamlit")
##