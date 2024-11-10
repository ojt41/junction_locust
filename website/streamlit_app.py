import streamlit as st
import streamlit.components.v1 as components
from big_hopper_main import LocustDataAnalyzer
from datetime import datetime, date
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

# At the top of your script, add state management
if 'predictions_path' not in st.session_state:
    st.session_state.predictions_path = './maps/advanced model prediction.html'
if 'new_map' not in st.session_state:
    st.session_state.new_map = False


def create_prediction_map(self, model: RandomForestClassifier,
                        year: int, month: int,
                        save_path: str = "maps") -> str:
    """
    Create a heat map of model predictions for a specific time period.
    Returns the path to the saved map.
    """
    # Create prediction grid
    lat_range = np.linspace(self.lat_min, self.lat_max, self.grid_size)
    lon_range = np.linspace(self.lon_min, self.lon_max, self.grid_size)

    grid_points = []
    predictions = []

    # Generate predictions for each grid point
    for lat in lat_range:
        for lon in lon_range:
            features = [year, month, lat, lon,
                       np.sin(2 * np.pi * month / 12),
                       np.cos(2 * np.pi * month / 12)]

            # Scale features (excluding year and month)
            features_scaled = features.copy()
            features_scaled[2:] = self.scaler.transform([features[2:]])[0]

            # Get prediction probability
            prob = model.predict_proba([features_scaled])[0][1]

            grid_points.append([lat, lon])
            predictions.append(prob)

    # Create prediction map
    center_lat = (self.lat_min + self.lat_max) / 2
    center_lon = (self.lon_min + self.lon_max) / 2

    prediction_map = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles='CartoDB positron'
    )

    # Add heatmap layer
    heatmap_data = [[lat, lon, prob] for (lat, lon), prob in zip(grid_points, predictions)]
    plugins.HeatMap(
        heatmap_data,
        min_opacity=0.3,
        max_opacity=0.8,
        radius=15,
        blur=10,
        gradient={0.4: 'blue', 0.6: 'yellow', 0.8: 'orange', 1: 'red'}
    ).add_to(prediction_map)

    # Add colormap
    colormap = LinearColormap(
        colors=['blue', 'yellow', 'orange', 'red'],
        vmin=0,
        vmax=1,
        caption='Predicted Probability of Locust Presence'
    )
    colormap.add_to(prediction_map)

    # Save map
    Path(save_path).mkdir(exist_ok=True)
    map_path = Path(save_path) / f"locust_predictions_{year}_{month}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    prediction_map.save(str(map_path))
    logging.info(f"Prediction map saved to {map_path}")

    return str(map_path)



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
        selected_date = st.date_input("Select a month and year to predict", date(2024, 1, 1))

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

            # Save model
            model_path = './new_model.joblib'
            analyzer.save_model(model)

            # Generate new map and save its path in session state
            new_map_path = create_prediction_map(analyzer, model, selected_date.year, selected_date.month)
            st.session_state.predictions_path = new_map_path
            st.session_state.new_map = True

            st.success("Analysis completed successfully!")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.session_state.new_map = False


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
                st.error("Observation map file not found")

        if map_style == "Predictions" or map_style == "All":
            try:
                with open(st.session_state.predictions_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                components.html(html_content, height=600, width=map_width)
            except FileNotFoundError:
                st.error(f"Prediction map not found at {st.session_state.predictions_path}")

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
