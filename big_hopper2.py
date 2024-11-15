# Standard library imports
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


def setup_logging() -> None:
    """
    Set up logging configuration with both file and console handlers.
    Creates a logs directory if it doesn't exist and generates timestamped log files.
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"locust_analyzer_{timestamp}.log"

    # Configure logging format and handlers
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logging.info(f"Logging initialized. Log file: {log_file}")


class LocustDataAnalyzer:
    """Analyzer for desert locust occurrence data using machine learning and mapping."""

    def __init__(self, filepath: str, grid_size: int = 100, random_state: int = 42):
        """
        Initialize the LocustDataAnalyzer.

        Args:
            filepath: Path to the CSV data file
            grid_size: Size of the grid for spatial analysis
            random_state: Random seed for reproducibility
        """
        self.grid_size = grid_size
        self.random_state = random_state
        self.data = None
        self.lat_min = None
        self.lat_max = None
        self.lon_min = None
        self.lon_max = None
        self.filepath = Path(filepath)
        self.scaler = StandardScaler()
        self.base_map = None
        self.prediction_map = None
        self.lon_increase = None
        self.lat_increase = None

        if not self.filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

    def load_and_preprocess_data(self) -> pd.DataFrame:
        """
        Load and preprocess the locust data.

        Returns:
            Preprocessed DataFrame
        """
        try:
            self.data = pd.read_csv(self.filepath)
            required_columns = [
                "lat", "lon", "Category", "Observation Date", "Country",
                "Soil_Moisture", "Temperature_2m"  # New required columns
            ]
            missing_columns = [col for col in required_columns if col not in self.data.columns]

            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Filter relevant columns and countries
            self.data = self.data[required_columns].copy()
            target_countries = self.data["Country"].unique()
            target_categories = ["Swarm", "Band"]

            self.data = self.data[
                self.data["Country"].isin(target_countries) &
                self.data["Category"].isin(target_categories)
                ]

            if self.data.empty:
                raise ValueError("No data remaining after filtering")

            # Convert dates and add temporal features
            self.data['Observation Date'] = pd.to_datetime(self.data['Observation Date'])
            self.data['Year'] = self.data['Observation Date'].dt.year
            self.data['Month'] = self.data['Observation Date'].dt.month
            self.data['Season'] = np.where(self.data['Month'].isin([3, 4, 5]), 'Spring',
                                           np.where(self.data['Month'].isin([6, 7, 8]), 'Summer',
                                                    np.where(self.data['Month'].isin([9, 10, 11]), 'Fall', 'Winter')))

            # Handle missing environmental data
            self.data['Soil_Moisture'].fillna(self.data['Soil_Moisture'].mean(), inplace=True)
            self.data['Temperature_2m'].fillna(self.data['Temperature_2m'].mean(), inplace=True)

            self.data.drop(columns=['Observation Date'], inplace=True)

            # Calculate grid boundaries
            self._calculate_grid_boundaries()

            logging.info(f"Data loaded successfully. Shape: {self.data.shape}")
            return self.data

        except Exception as e:
            logging.error(f"Error in data loading: {str(e)}")
            raise

    def generate_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate features and targets for the model.

        Returns:
            Tuple of (features array, targets array)
        """
        if self.data is None:
            raise ValueError("Data must be loaded before generating features")

        features = []
        targets = []

        years_range = range(self.data["Year"].min(), self.data["Year"].max() + 1)
        months_range = range(1, 13)

        total_iterations = len(years_range) * len(months_range) * self.grid_size * self.grid_size
        processed = 0

        for year in years_range:
            for month in months_range:
                # Filter data for current time period
                monthly_data = self.data[
                    (self.data["Year"] == year) &
                    (self.data["Month"] == month)
                    ]

                if not monthly_data.empty:
                    # Prepare location and environmental data
                    locations = list(zip(
                        monthly_data["lat"].values,
                        monthly_data["lon"].values,
                        monthly_data["Soil_Moisture"].values,
                        monthly_data["Temperature_2m"].values
                    ))

                    # Vectorized grid generation
                    lat_grid = np.linspace(self.lat_min, self.lat_max, self.grid_size + 1)[:-1]
                    lon_grid = np.linspace(self.lon_min, self.lon_max, self.grid_size + 1)[:-1]

                    for lat in lat_grid:
                        for lon in lon_grid:
                            # Find nearest environmental conditions
                            nearest_conditions = self._find_nearest_conditions(
                                lat, lon, locations
                            )

                            features.append([
                                year,
                                month,
                                lat,
                                lon,
                                np.sin(2 * np.pi * month / 12),  # Seasonal features
                                np.cos(2 * np.pi * month / 12),
                                nearest_conditions['soil_moisture'],
                                nearest_conditions['temperature']
                            ])

                            # Check for locust presence in grid cell
                            presence = any(
                                (lat <= obs_lat < lat + self.lat_increase) and
                                (lon <= obs_lon < lon + self.lon_increase)
                                for obs_lat, obs_lon, _, _ in locations
                            )
                            targets.append(float(presence))

                            processed += 1
                            if processed % 10000 == 0:
                                logging.info(f"Processing progress: {processed / total_iterations * 100:.2f}%")

        # Convert to numpy arrays and scale features
        X = np.array(features)
        y = np.array(targets)

        # Scale features except year and month
        X_scaled = X.copy()
        X_scaled[:, 2:] = self.scaler.fit_transform(X[:, 2:])

        return X_scaled, y

    def _find_nearest_conditions(self, lat: float, lon: float,
                                 locations: List[Tuple[float, float, float, float]]) -> Dict[str, float]:
        """
        Find the nearest environmental conditions for a given location.

        Args:
            lat: Latitude of the point
            lon: Longitude of the point
            locations: List of tuples containing (lat, lon, soil_moisture, temperature)

        Returns:
            Dictionary containing nearest environmental conditions
        """
        if not locations:
            return {
                'soil_moisture': self.data['Soil_Moisture'].mean(),
                'temperature': self.data['Temperature_2m'].mean()
            }

        # Calculate distances to all points
        distances = np.sqrt(
            (lat - np.array([loc[0] for loc in locations])) ** 2 +
            (lon - np.array([loc[1] for loc in locations])) ** 2
        )

        # Find index of nearest point
        nearest_idx = np.argmin(distances)

        return {
            'soil_moisture': locations[nearest_idx][2],
            'temperature': locations[nearest_idx][3]
        }

    def create_prediction_map(self, model: RandomForestClassifier,
                              year: int, month: int,
                              save_path: str = "maps") -> folium.Map:
        """
        Create an improved heat map of model predictions with better spatial resolution
        and more focused predictions.

        Args:
            model: Trained RandomForestClassifier
            year: Year to predict for
            month: Month to predict for
            save_path: Directory to save the HTML map

        Returns:
            Folium map object with prediction heatmap
        """
        # Get environmental conditions specific to the region and time period
        monthly_conditions = self.data[
            self.data['Month'] == month
            ].agg({
            'Soil_Moisture': ['mean', 'std'],
            'Temperature_2m': ['mean', 'std']
        })

        # Create a finer prediction grid for better resolution
        grid_size = self.grid_size * 2  # Double the resolution
        lat_range = np.linspace(self.lat_min, self.lat_max, grid_size)
        lon_range = np.linspace(self.lon_min, self.lon_max, grid_size)

        grid_points = []
        predictions = []

        # Calculate seasonal features
        sin_month = np.sin(2 * np.pi * month / 12)
        cos_month = np.cos(2 * np.pi * month / 12)

        # Generate predictions for each grid point
        for lat in lat_range:
            for lon in lon_range:
                # Find nearest actual environmental conditions
                nearby_data = self.data[
                    (abs(self.data['lat'] - lat) < 2) &  # Within 2 degrees
                    (abs(self.data['lon'] - lon) < 2) &  # Within 2 degrees
                    (self.data['Month'] == month)
                    ]

                if len(nearby_data) > 0:
                    soil_moisture = nearby_data['Soil_Moisture'].mean()
                    temperature = nearby_data['Temperature_2m'].mean()
                else:
                    soil_moisture = monthly_conditions['Soil_Moisture']['mean']
                    temperature = monthly_conditions['Temperature_2m']['mean']

                features = [
                    year, month, lat, lon,
                    sin_month, cos_month,
                    soil_moisture, temperature
                ]

                # Scale features (excluding year and month)
                features_scaled = features.copy()
                features_scaled[2:] = self.scaler.transform([features[2:]])[0]

                # Get prediction probability
                prob = model.predict_proba([features_scaled])[0][1]

                # Apply geographic weighting based on historical presence
                historical_presence = self.data[
                    (abs(self.data['lat'] - lat) < 0.5) &
                    (abs(self.data['lon'] - lon) < 0.5)
                    ].shape[0]

                if historical_presence > 0:
                    # Increase probability for areas with historical presence
                    prob = min(1.0, prob * 1.2)

                # Only include points with significant probability
                if prob > 0.2:  # Threshold to reduce noise
                    grid_points.append([lat, lon])
                    predictions.append(prob)

        # Create prediction map
        center_lat = (self.lat_min + self.lat_max) / 2
        center_lon = (self.lon_min + self.lon_max) / 2

        self.prediction_map = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=5,
            tiles='CartoDB positron'
        )

        # Add heatmap layer with adjusted parameters
        if grid_points:  # Only add heatmap if we have predictions
            plugins.HeatMap(
                [[lat, lon, prob] for (lat, lon), prob in zip(grid_points, predictions)],
                min_opacity=0.4,
                max_opacity=0.9,
                radius=12,  # Reduced radius for better definition
                blur=8,  # Reduced blur for sharper boundaries
                gradient={
                    0.2: 'blue',
                    0.4: 'cyan',
                    0.6: 'yellow',
                    0.8: 'orange',
                    1.0: 'red'
                }
            ).add_to(self.prediction_map)

        # Add colormap legend
        colormap = LinearColormap(
            colors=['blue', 'cyan', 'yellow', 'orange', 'red'],
            vmin=0.2,
            vmax=1.0,
            caption='Predicted Probability of Locust Presence'
        )
        colormap.add_to(self.prediction_map)

        # Add historical observations as markers for reference
        historical_data = self.data[
            (self.data['Year'] == year) &
            (self.data['Month'] == month)
            ]

        if not historical_data.empty:
            marker_cluster = plugins.MarkerCluster(name='Historical Observations')
            for _, row in historical_data.iterrows():
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=5,
                    color='black',
                    fill=True,
                    popup=f"Historical {row['Category']} observation"
                ).add_to(marker_cluster)
            marker_cluster.add_to(self.prediction_map)

        # Add layer control
        folium.LayerControl().add_to(self.prediction_map)

        # Save map
        Path(save_path).mkdir(exist_ok=True)
        map_path = Path(
            save_path) / f"locust_predictions_{year}_{month}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        self.prediction_map.save(str(map_path))
        logging.info(f"Prediction map saved to {map_path}")

        return self.prediction_map

    def _calculate_grid_boundaries(self) -> None:
        """Calculate grid boundaries for the spatial analysis."""
        if self.data is None:
            raise ValueError("Data must be loaded before calculating grid boundaries")

        # Add small buffer to boundaries
        buffer = 0.1  # degrees
        self.lat_max = self.data["lat"].max() + buffer
        self.lat_min = self.data["lat"].min() - buffer
        self.lon_max = self.data["lon"].max() + buffer
        self.lon_min = self.data["lon"].min() - buffer

        self.lon_increase = (self.lon_max - self.lon_min) / self.grid_size
        self.lat_increase = (self.lat_max - self.lat_min) / self.grid_size

    def train_and_evaluate_model(self, X: np.ndarray, y: np.ndarray,
                                 test_size: float = 0.2) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
        """
        Train and evaluate the Random Forest model.

        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data to use for testing

        Returns:
            Tuple of (trained model, performance metrics)
        """
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )

            # Train model with better parameters
            rf_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight='balanced'  # Handle class imbalance
            )

            rf_model.fit(X_train, y_train)

            # Make predictions
            y_pred = rf_model.predict(X_test)
            y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'classification_report': classification_report(y_test, y_pred),
                'feature_importance': dict(zip(
                    ['year', 'month', 'latitude', 'longitude', 'sin_month', 'cos_month'],
                    rf_model.feature_importances_
                ))
            }

            logging.info(f"Model training completed. Accuracy: {metrics['accuracy']:.4f}")
            return rf_model, metrics

        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise

    def create_observation_map(self, save_path: str = "maps") -> folium.Map:
        """
        Create an interactive map of locust observations.

        Args:
            save_path: Directory to save the HTML map

        Returns:
            Folium map object
        """
        if self.data is None:
            raise ValueError("Data must be loaded before creating map")

        # Calculate center point
        center_lat = (self.lat_min + self.lat_max) / 2
        center_lon = (self.lon_min + self.lon_max) / 2

        # Create base map
        self.base_map = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            tiles='CartoDB positron'
        )

        # Create marker clusters for swarms and bands
        swarm_cluster = plugins.MarkerCluster(name='Swarms')
        band_cluster = plugins.MarkerCluster(name='Bands')

        # Add markers for each observation
        for _, row in self.data.iterrows():
            popup_text = f"""
            Date: {row['Year']}-{row['Month']}
            Type: {row['Category']}
            Country: {row['Country']}
            """

            # Different icons for swarms and bands
            if row['Category'] == 'Swarm':
                cluster = swarm_cluster
                icon_color = 'red'
            else:  # Band
                cluster = band_cluster
                icon_color = 'orange'

            folium.Marker(
                location=[row['lat'], row['lon']],
                popup=popup_text,
                icon=folium.Icon(color=icon_color, icon='info-sign')
            ).add_to(cluster)

        # Add clusters to map
        swarm_cluster.add_to(self.base_map)
        band_cluster.add_to(self.base_map)

        # Add layer control
        folium.LayerControl().add_to(self.base_map)

        # Save map
        Path(save_path).mkdir(exist_ok=True)
        map_path = Path(save_path) / f"locust_observations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        self.base_map.save(str(map_path))
        logging.info(f"Observation map saved to {map_path}")

        return self.base_map

    def plot_temporal_distribution(self, save_path: str = "plots") -> None:
        """
        Create plots showing temporal distribution of locust occurrences.

        Args:
            save_path: Directory to save the plots
        """
        if self.data is None:
            raise ValueError("Data must be loaded before creating plots")

        try:
            # Create output directory if it doesn't exist
            Path(save_path).mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Set up the plotting style
            plt.style.use('seaborn')

            # Create figure with multiple subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

            # 1. Monthly distribution
            monthly_counts = self.data.groupby(['Month', 'Category']).size().unstack(fill_value=0)
            monthly_counts.plot(kind='bar', ax=ax1)
            ax1.set_title('Monthly Distribution of Locust Occurrences')
            ax1.set_xlabel('Month')
            ax1.set_ylabel('Count')
            ax1.legend(title='Category')

            # 2. Yearly trend
            yearly_counts = self.data.groupby(['Year', 'Category']).size().unstack(fill_value=0)
            yearly_counts.plot(kind='line', marker='o', ax=ax2)
            ax2.set_title('Yearly Trend of Locust Occurrences')
            ax2.set_xlabel('Year')
            ax2.set_ylabel('Count')
            ax2.legend(title='Category')

            # 3. Seasonal patterns by country
            seasonal_counts = self.data.groupby(['Season', 'Country', 'Category']).size().unstack(fill_value=0)
            seasonal_counts.plot(kind='bar', ax=ax3)
            ax3.set_title('Seasonal Patterns by Country')
            ax3.set_xlabel('Season')
            ax3.set_ylabel('Count')
            ax3.legend(title='Category')

            # Adjust layout and save
            plt.tight_layout()
            plot_path = Path(save_path) / f"temporal_distribution_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            logging.info(f"Temporal distribution plots saved to {plot_path}")

        except Exception as e:
            logging.error(f"Error creating temporal distribution plots: {str(e)}")
            raise

    def plot_environmental_analysis(self, save_path: str = "plots") -> None:
        """
        Create plots showing relationships between environmental factors and locust occurrences.

        Args:
            save_path: Directory to save the plots
        """
        if self.data is None:
            raise ValueError("Data must be loaded before creating plots")

        try:
            # Create output directory if it doesn't exist
            Path(save_path).mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Set up the plotting style
            plt.style.use('seaborn')

            # Create figure with multiple subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))

            # 1. Soil Moisture Distribution by Category
            sns.boxplot(x='Category', y='Soil_Moisture', data=self.data, ax=ax1)
            ax1.set_title('Soil Moisture Distribution by Locust Category')
            ax1.set_ylabel('Soil Moisture')

            # 2. Temperature Distribution by Category
            sns.boxplot(x='Category', y='Temperature_2m', data=self.data, ax=ax2)
            ax2.set_title('Temperature Distribution by Locust Category')
            ax2.set_ylabel('Temperature (°C)')

            # 3. Soil Moisture vs Temperature scatter plot
            sns.scatterplot(
                data=self.data,
                x='Soil_Moisture',
                y='Temperature_2m',
                hue='Category',
                alpha=0.6,
                ax=ax3
            )
            ax3.set_title('Soil Moisture vs Temperature')
            ax3.set_xlabel('Soil Moisture')
            ax3.set_ylabel('Temperature (°C)')

            # 4. Environmental conditions by season
            seasonal_data = self.data.melt(
                id_vars=['Season', 'Category'],
                value_vars=['Soil_Moisture', 'Temperature_2m'],
                var_name='Environmental Factor',
                value_name='Value'
            )
            sns.boxplot(
                x='Season',
                y='Value',
                hue='Environmental Factor',
                data=seasonal_data,
                ax=ax4
            )
            ax4.set_title('Environmental Conditions by Season')
            ax4.set_ylabel('Value')

            # Adjust layout and save
            plt.tight_layout()
            plot_path = Path(save_path) / f"environmental_analysis_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            logging.info(f"Environmental analysis plots saved to {plot_path}")

        except Exception as e:
            logging.error(f"Error creating environmental analysis plots: {str(e)}")
            raise

    @staticmethod
    def get_version() -> str:
        """Return the version of the LocustDataAnalyzer."""


if __name__ == "__main__":
    # Set up logging
    setup_logging()

    try:
        # Initialize analyzer
        analyzer = LocustDataAnalyzer("final_data.csv")

        # Load and preprocess data
        data = analyzer.load_and_preprocess_data()

        # Generate features and train model
        X, y = analyzer.generate_features()
        model, metrics = analyzer.train_and_evaluate_model(X, y)

        # Create visualizations
        analyzer.create_observation_map()
        analyzer.create_prediction_map(model, 2024, 10)
        analyzer.plot_temporal_distribution()

        # Analyze spatial patterns
        spatial_analysis = analyzer.analyze_spatial_patterns()

        # Save model
        analyzer.save_model(model)

        logging.info("Analysis completed successfully")

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

