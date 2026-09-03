
import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium

from sklearn.ensemble import RandomForestClassifier
from streamlit_folium import st_folium


# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="HydroSafe",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 HydroSafe")
st.subheader("AI-Powered Flood Risk & Emergency Response System")


# =========================================
# AI TRAINING DATA
# =========================================

np.random.seed(42)

n = 500

ai_data = pd.DataFrame({
    "rainfall_mm": np.random.uniform(20, 400, n),
    "water_level_m": np.random.uniform(1, 12, n),
    "elevation_m": np.random.uniform(1, 20, n),
    "river_distance_m": np.random.uniform(50, 2000, n)
})

score = (
    0.40 * (ai_data["rainfall_mm"] / 400) +
    0.35 * (ai_data["water_level_m"] / 12) +
    0.15 * (1 - ai_data["elevation_m"] / 20) +
    0.10 * (1 - ai_data["river_distance_m"] / 2000)
) * 100

ai_data["risk"] = pd.cut(
    score,
    bins=[-1, 35, 70, 101],
    labels=["LOW", "MEDIUM", "HIGH"]
)

features = [
    "rainfall_mm",
    "water_level_m",
    "elevation_m",
    "river_distance_m"
]

X = ai_data[features]
y = ai_data["risk"]

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)


# =========================================
# SIDEBAR INPUT
# =========================================

st.sidebar.header("🌧️ Flood Scenario")

rainfall = st.sidebar.slider(
    "Rainfall (mm)",
    0,
    400,
    250
)

water_level = st.sidebar.slider(
    "Water Level (m)",
    0.0,
    12.0,
    7.0
)

elevation = st.sidebar.slider(
    "Elevation (m)",
    1.0,
    20.0,
    5.0
)

river_distance = st.sidebar.slider(
    "Distance from River (m)",
    50,
    2000,
    300
)


# =========================================
# AI PREDICTION
# =========================================

input_data = pd.DataFrame({
    "rainfall_mm": [rainfall],
    "water_level_m": [water_level],
    "elevation_m": [elevation],
    "river_distance_m": [river_distance]
})

prediction = model.predict(input_data)[0]


# =========================================
# RISK SCORE
# =========================================

risk_score = (
    0.40 * (rainfall / 400) +
    0.35 * (water_level / 12) +
    0.15 * (1 - elevation / 20) +
    0.10 * (1 - river_distance / 2000)
) * 100


# =========================================
# DASHBOARD METRICS
# =========================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "🌧️ Rainfall",
    f"{rainfall} mm"
)

col2.metric(
    "🌊 Water Level",
    f"{water_level:.1f} m"
)

col3.metric(
    "📊 Risk Score",
    f"{risk_score:.1f}"
)

col4.metric(
    "🤖 AI Risk",
    prediction
)


# =========================================
# ALERT
# =========================================

if prediction == "HIGH":

    st.error(
        "🚨 HIGH FLOOD RISK: Immediate emergency response recommended."
    )

elif prediction == "MEDIUM":

    st.warning(
        "⚠️ MEDIUM FLOOD RISK: Monitor water levels and prepare evacuation."
    )

else:

    st.success(
        "✅ LOW FLOOD RISK: Continue monitoring."
    )


# =========================================
# DEMO BUILDINGS
# =========================================

# =========================================
# DEMO BUILDINGS
# =========================================

buildings = pd.DataFrame({
    "id": ["B001", "B002", "B003", "B004", "B005"],
    "type": [
        "Hospital",
        "School",
        "House",
        "House",
        "Hospital"
    ],
    "latitude": [
        11.025,
        11.026,
        11.020,
        11.030,
        11.028
    ],
    "longitude": [
        76.965,
        76.968,
        76.960,
        76.970,
        76.958
    ],
    "population": [
        250,
        500,
        5,
        4,
        180
    ]
})



# =========================================
# DEMO ROADS
# =========================================

roads = pd.DataFrame({
    "road": [
        "Road A",
        "Road B",
        "Road C",
        "Road D"
    ],
    "latitude": [
        11.021,
        11.026,
        11.029,
        11.018
    ],
    "longitude": [
        76.961,
        76.967,
        76.969,
        76.957
    ],
    "importance": [
        "HIGH",
        "MEDIUM",
        "HIGH",
        "LOW"
    ]
})


# =========================================
# MAP
# =========================================

m = folium.Map(
    location=[11.025, 76.965],
    zoom_start=14
)


# Flood risk zone

if prediction == "HIGH":
    zone_color = "red"

elif prediction == "MEDIUM":
    zone_color = "orange"

else:
    zone_color = "green"


folium.Circle(
    location=[11.025, 76.965],
    radius=1000,
    color=zone_color,
    fill=True,
    fill_color=zone_color,
    fill_opacity=0.25,
    popup=f"Predicted Flood Risk: {prediction}"
).add_to(m)


# =========================================
# BUILDING MARKERS
# =========================================

for _, row in buildings.iterrows():

    if row["type"] == "Hospital":
        icon_color = "purple"

    elif row["type"] == "School":
        icon_color = "blue"

    else:
        icon_color = "black"

    folium.Marker(
        location=[
            row["latitude"],
            row["longitude"]
        ],
        popup=(
            f"Building: {row['id']}<br>"
            f"Type: {row['type']}<br>"
            f"Population: {row['population']}<br>"
            f"Flood Risk: {prediction}"
        ),
        icon=folium.Icon(
            color=icon_color,
            icon="info-sign"
        )
    ).add_to(m)


# =========================================
# ROAD MARKERS
# =========================================

for _, row in roads.iterrows():

    folium.CircleMarker(
        location=[
            row["latitude"],
            row["longitude"]
        ],
        radius=8,
        popup=(
            f"Road: {row['road']}<br>"
            f"Importance: {row['importance']}<br>"
            f"Flood Risk: {prediction}"
        ),
        color=zone_color,
        fill=True,
        fill_color=zone_color
    ).add_to(m)


# =========================================
# DISPLAY MAP
# =========================================

st.subheader("🗺️ Live Flood Risk Map")

st_folium(
    m,
    width=1400,
    height=550
)


# =========================================
# IMPACT ANALYSIS
# =========================================

st.subheader("🚨 Impact Analysis")

if prediction == "HIGH":

    affected_buildings = len(buildings)
    affected_roads = len(roads)

elif prediction == "MEDIUM":

    affected_buildings = 3
    affected_roads = 2

else:

    affected_buildings = 1
    affected_roads = 0


c1, c2, c3 = st.columns(3)

c1.metric(
    "🏢 Affected Buildings",
    affected_buildings
)

c2.metric(
    "🛣️ Affected Roads",
    affected_roads
)

c3.metric(
    "👥 Population at Risk",
    int(buildings["population"].sum())
)


# =========================================
# EMERGENCY RESPONSE
# =========================================

st.subheader("🚑 Recommended Response")

if prediction == "HIGH":

    st.write("""
    🔴 **Priority Actions**

    1. Activate emergency response teams.
    2. Prepare evacuation of high-risk areas.
    3. Protect hospitals and schools.
    4. Restrict access to high-risk roads.
    5. Continuously monitor rainfall and water levels.
    """)

elif prediction == "MEDIUM":

    st.write("""
    🟠 **Priority Actions**

    1. Monitor river/water level.
    2. Alert vulnerable populations.
    3. Prepare evacuation resources.
    4. Monitor critical roads and buildings.
    """)

else:

    st.write("""
    🟢 **Priority Actions**

    1. Continue monitoring.
    2. Keep emergency teams ready.
    3. Update rainfall and water-level data.
    """)
