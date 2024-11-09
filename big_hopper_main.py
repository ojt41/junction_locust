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
            required_columns = ["lat", "lon", "Category", "Observation Date", "Country"]
            missing_columns = [col for col in required_columns if col not in self.data.columns]

            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Filter relevant columns and countries
            self.data = self.data[required_columns].copy()
            target_countries = ["Ethiopia", "Kenya", "Somalia"]
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

            self.data.drop(columns=['Observation Date'], inplace=True)

            # Calculate grid boundaries
            self._calculate_grid_boundaries()

            logging.info(f"Data loaded successfully. Shape: {self.data.shape}")
            return self.data

        except Exception as e:
            logging.error(f"Error in data loading: {str(e)}")
            raise

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
                    latlon = list(zip(monthly_data["lat"].values, monthly_data["lon"].values))

                    # Vectorized grid generation
                    lat_grid = np.linspace(self.lat_min, self.lat_max, self.grid_size + 1)[:-1]
                    lon_grid = np.linspace(self.lon_min, self.lon_max, self.grid_size + 1)[:-1]

                    for lat in lat_grid:
                        for lon in lon_grid:
                            # Check for locust presence in grid cell
                            presence = any(
                                (lat <= obs_lat < lat + self.lat_increase) and
                                (lon <= obs_lon < lon + self.lon_increase)
                                for obs_lat, obs_lon in latlon
                            )

                            features.append([
                                year,
                                month,
                                lat,
                                lon,
                                np.sin(2 * np.pi * month / 12),  # Seasonal features
                                np.cos(2 * np.pi * month / 12)
                            ])
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

    def create_prediction_map(self, model: RandomForestClassifier,
                              year: int, month: int,
                              save_path: str = "maps") -> folium.Map:
        """
        Create a heat map of model predictions for a specific time period.

        Args:
            model: Trained RandomForestClassifier
            year: Year to predict for
            month: Month to predict for
            save_path: Directory to save the HTML map

        Returns:
            Folium map object with prediction heatmap
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

        self.prediction_map = folium.Map(
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
        ).add_to(self.prediction_map)

        # Add colormap
        colormap = LinearColormap(
            colors=['blue', 'yellow', 'orange', 'red'],
            vmin=0,
            vmax=1,
            caption='Predicted Probability of Locust Presence'
        )
        colormap.add_to(self.prediction_map)

        # Save map
        Path(save_path).mkdir(exist_ok=True)
        map_path = Path(
            save_path) / f"locust_predictions_{year}_{month}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        self.prediction_map.save(str(map_path))
        logging.info(f"Prediction map saved to {map_path}")

        return self.prediction_map

    def save_model(self, model: RandomForestClassifier, output_dir: str = "models") -> None:
            """
            Save the trained model and scaler to disk.

            Args:
                model: Trained RandomForestClassifier model
                output_dir: Directory to save model files
            """
            try:
                # Create output directory if it doesn't exist
                output_path = Path(output_dir)
                output_path.mkdir(exist_ok=True)

                # Generate timestamped filenames
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = output_path / f"locust_model_{timestamp}.joblib"
                scaler_path = output_path / f"scaler_{timestamp}.joblib"

                # Save model and scaler
                joblib.dump(model, model_path)
                joblib.dump(self.scaler, scaler_path)

                logging.info(f"Model saved to {model_path}")
                logging.info(f"Scaler saved to {scaler_path}")

            except Exception as e:
                logging.error(f"Error saving model: {str(e)}")
                raise

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

    def analyze_spatial_patterns(self) -> Dict[str, Any]:
            """
            Analyze spatial patterns in locust occurrences.

            Returns:
                Dictionary containing spatial analysis results
            """
            if self.data is None:
                raise ValueError("Data must be loaded before analyzing spatial patterns")

            try:
                # Calculate spatial statistics
                country_stats = {
                    'occurrence_counts': self.data.groupby('Country')['Category'].value_counts().to_dict(),
                    'spatial_extent': {
                        country: {
                            'lat_range': (
                                country_data['lat'].min(),
                                country_data['lat'].max()
                            ),
                            'lon_range': (
                                country_data['lon'].min(),
                                country_data['lon'].max()
                            ),
                            'area_coverage': (
                                    (country_data['lat'].max() - country_data['lat'].min()) *
                                    (country_data['lon'].max() - country_data['lon'].min())
                            )
                        }
                        for country, country_data in self.data.groupby('Country')
                    }
                }

                # Calculate density metrics
                for country in country_stats['spatial_extent']:
                    area = country_stats['spatial_extent'][country]['area_coverage']
                    total_occurrences = sum(
                        count for (c, _), count in country_stats['occurrence_counts'].items()
                        if c == country
                    )
                    country_stats['spatial_extent'][country]['density'] = total_occurrences / area

                logging.info("Spatial pattern analysis completed")
                return country_stats

            except Exception as e:
                logging.error(f"Error in spatial pattern analysis: {str(e)}")
                raise

    @staticmethod
    def get_version() -> str:
            """Return the version of the LocustDataAnalyzer."""
            return "1.0.0"

if __name__ == "__main__":
        # Set up logging
        setup_logging()

        try:
            # Initialize analyzer
            analyzer = LocustDataAnalyzer("locust_data.csv")

            # Load and preprocess data
            data = analyzer.load_and_preprocess_data()

            # Generate features and train model
            X, y = analyzer.generate_features()
            model, metrics = analyzer.train_and_evaluate_model(X, y)

            # Create visualizations
            analyzer.create_observation_map()
            analyzer.create_prediction_map(model, 2024, 1)
            analyzer.plot_temporal_distribution()

            # Analyze spatial patterns
            spatial_analysis = analyzer.analyze_spatial_patterns()

            # Save model
            analyzer.save_model(model)

            logging.info("Analysis completed successfully")

        except Exception as e:
            logging.error(f"Error in main execution: {str(e)}")
            raise