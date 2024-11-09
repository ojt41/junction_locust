import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import math
#Pre-Processing

locust_data = pd.read_csv("locust_data.csv")

locust_data = locust_data[["lat", "lon", "Category", "Observation Date", "Country"]]
locust_data = locust_data.loc[((locust_data["Country"] == "Ethiopia") | (locust_data["Country"] == "Kenya") | (locust_data["Country"] == "Somalia")) & ((locust_data["Category"] == "Swarm") | (locust_data["Category"] == "Band"))]
## 1. Split locust data into months/quarters/semesters

dates = locust_data["Observation Date"].values

years = []
month = []

for d in dates:
    spl = d.split("-")
    years.append(int(spl[0]))
    month.append(int(spl[1]))

locust_data = locust_data.drop(columns=["Observation Date"])
locust_data["Year"] = years
locust_data["Month"] = month

#Gather max and min values of longitude and latitude which we will use for the grid

lat_max = np.max(locust_data["lat"].values)
lat_min = np.min(locust_data["lat"].values)

lon_max = np.max(locust_data["lon"].values)
lon_min = np.min(locust_data["lon"].values)


grid_size = 100
lon_increase = (lon_max - lon_min)/grid_size
lat_increase = (lat_max - lat_min)/grid_size


#Get lowest year and latest year

old_year = np.min(locust_data["Year"].values)
last_year = np.max(locust_data["Year"].values)

final_features = []
final_targets = []

for year in range(old_year, last_year+1):
    for month in range(1, 13):
        # Gather the data from that year and month

        data = locust_data.loc[(locust_data["Year"] == year) & (locust_data["Month"] == month)]
        latlon = list(zip(data["lat"].values, data["lon"].values))

        previous_data = locust_data.loc[(locust_data["Year"] == year) & (locust_data["Month"] == month)]

        if len(latlon) != 0:
            for i in range(grid_size):
                cur_lat = lat_min + lat_increase*i
                for j in range(grid_size):
                    cur_lon = lon_min + lon_increase*j
                    bool_val = 0.0
                    for tup in latlon:
                        if (tup[0] > cur_lat) and (tup[0] < (cur_lat + lat_increase)) and (tup[1] > cur_lon) and (tup[1] < (cur_lon + lon_increase)):
                            bool_val = 1.0
                            break
                    

                    #Features to put: NVDI, Soil Moisture, Year, Month, 
                    final_features.append([year, month, cur_lat, cur_lon])
                    final_targets.append(bool_val)


# Random forest that shit

X_train, X_test, y_train, y_test = train_test_split(final_features, final_targets, test_size=0.4, shuffle=False)

rf_model = RandomForestClassifier(n_estimators=100)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)


# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Detailed classification report
print(classification_report(y_test, y_pred))

# ROC AUC score
roc_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
print(f'ROC AUC: {roc_auc:.2f}')


print(np.sum(y_pred)-np.sum(y_test))
import folium

# Example: Create a map centered on Yemen (use actual coordinates for Yemen's center)
m = folium.Map(location=[15.0, 48.0], zoom_start=7)

# Plot each grid point and its probability on the map
for i in range(5, len(X_test)):
    folium.CircleMarker(
        location=[X_test[i][2], X_test[i][3]],
        radius=5,
        color='red' if y_pred[i] > 0.5 else 'blue',  # Color based on probability
        fill=True,
        fill_color='red' if y_pred[i] > 0.5 else 'blue',
        fill_opacity=0.6
    ).add_to(m)

# Save the map to an HTML file
m.save("locust_predictions_map.html")
