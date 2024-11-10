import streamlit as st
import streamlit.components.v1 as components
from big_hopper_main import LocustDataAnalyzer
import datetime
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Third-party imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import folium
from folium import plugins
from branca.colormap import LinearColormap
import matplotlib.pyplot as plt
import seaborn as sns

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

    model_button = st.selectbox("Choose model",
                                ["Advanced Model", "Regular model"]
                                )



    if "last_selected_date" not in st.session_state:
        st.session_state.last_selected_date = None

    if model_button == "Regular model":
        selected_date = st.date_input("Select a month and year to predict", datetime.today())

    CURRENT_DIR = Path(__file__).parent
    DATA_PATH = CURRENT_DIR / "locust_data_2018_onwards.csv"


    # Add a submit button
    submit_button = st.button("Generate Analysis")

    # Only run analysis when submit button is clicked
    if submit_button:
        st.session_state.last_selected_date = selected_date
        try:
            # Initialize analyzer
            analyzer = LocustDataAnalyzer(str(DATA_PATH))

            # Load and preprocess data
            data = analyzer.load_and_preprocess_data()

            # Generate features and train model
            X, y = analyzer.generate_features()
            model, metrics = analyzer.train_and_evaluate_model(X, y)

            # Create visualizations
            analyzer.create_observation_map()
            analyzer.create_prediction_map(model, selected_date.year, selected_date.month)
            analyzer.plot_temporal_distribution()

            # Analyze spatial patterns
            spatial_analysis = analyzer.analyze_spatial_patterns()

            # Save model
            analyzer.save_model(model)

            # Add a success message
            st.success("Analysis completed successfully!")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


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
        map_width = 750

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

# Footer with proper spacing
st.markdown("---")
st.markdown("Made with Streamlit")
##