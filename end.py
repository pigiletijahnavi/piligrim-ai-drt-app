import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from geopy.distance import geodesic
import random
import time
import os
import branca.colormap as cm

# Geolocation package
from streamlit_geolocation import geolocation

# ---------------- SYSTEM FIX ----------------
os.environ["OMP_NUM_THREADS"] = "1"

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI-Powered DRT – Tirupati",
    page_icon="🚌",
    layout="wide"
)

# ---------------- WELCOME MESSAGE ----------------
st.info("""
👋 Welcome to **AI-Powered DRT – Tirupati!**

**Steps to Use:**
1. Select the demand scenario (Normal, Peak, Festival).  
2. Select maximum AI pickup stops.  
3. Choose your final destination.  
4. Click **Run AI Optimization**.  
5. View the optimized route and AI bus stops on the map.  
6. Find your nearest AI pickup stop using your current location (browser geolocation) or manual input.
""")

# ---------------- SESSION STATE ----------------
for key in ["df","centers","optimized_route","naive_route","naive_dist","opt_dist","last_refresh"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ---------------- LOCATIONS ----------------
PILGRIM_SPOTS = {
    "Tirumala Temple": (13.6833, 79.3470),
    "Alipiri": (13.6288, 79.4192),
    "Srivari Mettu": (13.6516, 79.4025),
    "Kapila Theertham": (13.6284, 79.4190),
    "ISKCON Temple": (13.6341, 79.4184),
    "Chandragiri Fort": (13.5846, 79.3175),
}

DESTINATIONS = {
    "Tirupati Railway Station": (13.6288, 79.4192),
    "Tirupati Bus Stand": (13.6283, 79.4197)
}

MIN_CAPACITY = 10

# ---------------- SIDEBAR ----------------
st.sidebar.header("🔧 Control Panel")
scenario = st.sidebar.selectbox("Demand Scenario", ["Normal", "Peak", "Festival"])
max_stops = st.sidebar.slider("Maximum AI Pickup Stops", 2, 6, 4)
destination_name = st.sidebar.selectbox("Final Destination", list(DESTINATIONS.keys()))
auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh (10 sec)")
run = st.sidebar.button("🚀 Run AI Optimization")

# ---------------- DATA GENERATION ----------------
def generate_pilgrim_data(scenario):
    rows = []
    for place, (lat, lon) in PILGRIM_SPOTS.items():
        count = random.randint(20, 35) if scenario=="Normal" else random.randint(40,70) if scenario=="Peak" else random.randint(80,130)
        for _ in range(count):
            rows.append([place, lat + np.random.normal(0,0.002), lon + np.random.normal(0,0.002)])
    return pd.DataFrame(rows, columns=["location","lat","lon"])

# ---------------- DISTANCE ----------------
def total_distance(route):
    return sum(geodesic(route[i], route[i+1]).km for i in range(len(route)-1))

# ---------------- 2-OPT OPTIMIZATION ----------------
def two_opt(route):
    best = route
    best_dist = total_distance(route)
    improved = True
    while improved:
        improved = False
        for i in range(1,len(route)-2):
            for j in range(i+1,len(route)-1):
                new_route = route[:i]+route[i:j][::-1]+route[j:]
                new_dist = total_distance(new_route)
                if new_dist < best_dist:
                    best = new_route
                    best_dist = new_dist
                    improved = True
        route = best
    return best

# ---------------- RUN OPTIMIZATION ----------------
if run or st.session_state.df is not None:
    if run or (auto_refresh and (st.session_state.last_refresh is None or time.time() - st.session_state.last_refresh > 10)):
        st.session_state.last_refresh = time.time()
        if run:
            with st.spinner("Running AI Optimization..."):
                time.sleep(1)
                df = generate_pilgrim_data(scenario)
                st.session_state.df = df
                X = df[["lat","lon"]].values
                kmeans = KMeans(n_clusters=max_stops, random_state=42)
                df["cluster"] = kmeans.fit_predict(X)
                centers=[]
                for c in range(max_stops):
                    cluster_df = df[df["cluster"]==c]
                    if len(cluster_df) >= MIN_CAPACITY:
                        centers.append([cluster_df["lat"].mean(), cluster_df["lon"].mean()])
                st.session_state.centers = centers
                dest = DESTINATIONS[destination_name]
                naive_route = [dest]+centers+[dest]
                optimized_route = two_opt(naive_route)
                st.session_state.naive_route = naive_route
                st.session_state.optimized_route = optimized_route
                st.session_state.naive_dist = total_distance(naive_route)
                st.session_state.opt_dist = total_distance(optimized_route)

    # ---------------- LOAD SESSION DATA ----------------
    df = st.session_state.df
    centers = st.session_state.centers
    optimized_route = st.session_state.optimized_route
    naive_route = st.session_state.naive_route
    naive_dist = st.session_state.naive_dist
    opt_dist = st.session_state.opt_dist
    dest = DESTINATIONS[destination_name]

    # ---------------- MAP ----------------
    m = folium.Map(location=dest, zoom_start=13)
    cluster_colors = cm.linear.Set1_06.scale(0,max_stops).to_step(max_stops)

    # Passengers
    for _, r in df.iterrows():
        cluster_id = r["cluster"]
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=3,
            color=cluster_colors(cluster_id),
            fill=True,
            fill_opacity=0.5,
            popup=f"{r['location']} | Cluster {cluster_id}"
        ).add_to(m)

    # Cluster centers
    for idx, c in enumerate(centers):
        cluster_count = len(df[df["cluster"]==idx])
        folium.Marker(
            location=c,
            icon=folium.Icon(color="green", icon="bus"),
            tooltip=f"AI Pickup Stop {idx+1} | Passengers: {cluster_count}"
        ).add_to(m)

    # Destination
    folium.Marker(
        location=dest,
        icon=folium.Icon(color="red", icon="flag"),
        tooltip=destination_name
    ).add_to(m)

    # Optimized route with arrows
    for i in range(len(optimized_route)-1):
        start = optimized_route[i]
        end = optimized_route[i+1]
        folium.PolyLine(locations=[start,end], color="purple", weight=4, opacity=0.8).add_to(m)
        folium.RegularPolygonMarker(
            location=[(start[0]+end[0])/2, (start[1]+end[1])/2],
            fill_color='purple',
            number_of_sides=3,
            radius=6,
            rotation=0
        ).add_to(m)

    # Legend
    legend_html = "<div style='position: fixed; bottom: 50px; left: 50px; width: 150px; height: auto; background-color: white; z-index:9999; font-size:14px; padding: 10px; border:2px solid grey;'><b>Cluster Colors</b><br>"
    for i in range(max_stops):
        legend_html += f"<i style='background:{cluster_colors(i)};width:12px;height:12px;display:inline-block;margin-right:5px'></i> Cluster {i}<br>"
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))

    # ---------------- UI ----------------
    col1,col2 = st.columns([3,1])
    with col1:
        st_folium(m, height=600, width=950)

    with col2:
        st.metric("Naive Distance (km)", f"{naive_dist:.2f}")
        st.metric("Optimized Distance (km)", f"{opt_dist:.2f}")
        st.metric("Distance Saved (km)", f"{naive_dist-opt_dist:.2f}")
        st.metric("Passengers Served", len(df))
        st.metric("AI Pickup Stops", len(centers))
        st.write("📍 Demand Distribution")
        st.dataframe(df["location"].value_counts())

    # ---------------- NEAREST STOP WITH GEOLOCATION ----------------
    st.subheader("🧭 Find Nearest AI Bus Stop")

    location = geolocation()
    if location:
        user_lat = location['lat']
        user_lon = location['lon']
        st.success(f"Your Current Location: ({user_lat:.6f}, {user_lon:.6f})")
    else:
        user_lat = st.number_input("Your Latitude", value=13.63, format="%.6f")
        user_lon = st.number_input("Your Longitude", value=79.42, format="%.6f")
        st.warning("Geolocation not detected. Please enter your location manually.")

    if centers:
        nearest = min(
            centers,
            key=lambda c: geodesic((user_lat,user_lon),(c[0],c[1])).km
        )
        dist = geodesic((user_lat,user_lon), (nearest[0], nearest[1])).km
        st.success(f"Nearest AI Pickup Stop: {nearest} | Distance: {dist:.2f} km")

else:
    st.info("Configure settings and click **Run AI Optimization**")
