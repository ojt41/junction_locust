import requests
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
import numpy as np



class LocustDataCollector:
    def __init__(self):
        self.fao_base_url = "https://locust-hub-hqfao.hub.arcgis.com/api/v3"
        self.nasa_base_url = "https://power.larc.nasa.gov/api/temporal/"

    def get_fao_locust_data(self, start_date, end_date):
        """
        Fetch locust presence data from FAO's API
        """
        try:
            # Format dates for API request
            start = start_date.strftime("%Y-%m-%d")
            end = end_date.strftime("%Y-%m-%d")

            # Example endpoint for locust data
            endpoint = f"{self.fao_base_url}/feature-layers/locust_presence"
            params = {
                'where': f"observation_date BETWEEN '{start}' AND '{end}'",
                'outFields': '*',
                'f': 'json'
            }

            response = requests.get(endpoint, params=params)
            if response.status_code == 200:
                return pd.DataFrame(response.json()['features'])
            else:
                print(f"Error fetching FAO data: {response.status_code}")
                return None

        except Exception as e:
            print(f"Error in FAO data collection: {str(e)}")
            return None

    def get_nasa_weather_data(self, latitude, longitude, start_date, end_date):
        """
        Fetch weather data from NASA POWER API
        """
        try:
            params = {
                'parameters': 'T2M,PRECTOT,WS2M',  # Temperature, Precipitation, Wind Speed
                'community': 'AG',  # Agricultural community
                'longitude': longitude,
                'latitude': latitude,
                'start': start_date.strftime("%Y%m%d"),
                'end': end_date.strftime("%Y%m%d"),
                'format': 'JSON'
            }

            response = requests.get(f"{self.nasa_base_url}/point", params=params)
            if response.status_code == 200:
                return pd.DataFrame(response.json()['features'])
            else:
                print(f"Error fetching NASA data: {response.status_code}")
                return None

        except Exception as e:
            print(f"Error in NASA data collection: {str(e)}")
            return None

    def process_vegetation_indices(self, sentinel_data):
        """
        Process vegetation indices from Sentinel-2 data
        """
        try:
            # Calculate NDVI (Normalized Difference Vegetation Index)
            ndvi = (sentinel_data['B8'] - sentinel_data['B4']) / (sentinel_data['B8'] + sentinel_data['B4'])

            # Calculate EVI (Enhanced Vegetation Index)
            evi = 2.5 * ((sentinel_data['B8'] - sentinel_data['B4']) /
                         (sentinel_data['B8'] + 6 * sentinel_data['B4'] -
                          7.5 * sentinel_data['B2'] + 1))

            return pd.DataFrame({
                'ndvi': ndvi,
                'evi': evi
            })

        except Exception as e:
            print(f"Error in vegetation indices processing: {str(e)}")
            return None

    def combine_data_sources(self, locust_data, weather_data, vegetation_data):
        """
        Combine different data sources and prepare for the dashboard
        """
        try:
            # Merge datasets based on location and time
            combined_data = pd.merge(
                locust_data,
                weather_data,
                on=['latitude', 'longitude', 'date'],
                how='left'
            )

            combined_data = pd.merge(
                combined_data,
                vegetation_data,
                on=['latitude', 'longitude', 'date'],
                how='left'
            )

            # Calculate risk levels based on combined factors
            combined_data['risk_level'] = self.calculate_risk_level(
                combined_data['swarm_size'],
                combined_data['temperature'],
                combined_data['wind_speed'],
                combined_data['ndvi']
            )

            return combined_data

        except Exception as e:
            print(f"Error in data combination: {str(e)}")
            return None

    @staticmethod
    def calculate_risk_level(swarm_size, temperature, wind_speed, ndvi):
        """
        Calculate risk level based on multiple factors
        """
        # Example risk calculation logic
        risk_score = (
                0.4 * swarm_size +
                0.3 * temperature +
                0.2 * wind_speed +
                0.1 * ndvi
        )

        # Convert score to risk level
        risk_levels = pd.cut(
            risk_score,
            bins=[-np.inf, 0.3, 0.6, np.inf],
            labels=['Low', 'Medium', 'High']
        )

        return risk_levels


def main():
    # Initialize collector
    collector = LocustDataCollector()

    # Set date range for data collection
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    # Example coordinates (Horn of Africa)
    latitude = 7.946527
    longitude = 38.799319

    # Collect data from different sources
    locust_data = collector.get_fao_locust_data(start_date, end_date)
    weather_data = collector.get_nasa_weather_data(latitude, longitude, start_date, end_date)

    # Process and combine data
    if locust_data is not None and weather_data is not None:
        combined_data = collector.combine_data_sources(
            locust_data,
            weather_data,
            None  # Vegetation data would come from Sentinel-2
        )

        # Save to CSV for dashboard
        combined_data.to_csv('locust_dashboard_data.csv', index=False)
        print("Data collection and processing completed successfully")
    else:
        print("Error in data collection process")


if __name__ == "__main__":
    main()