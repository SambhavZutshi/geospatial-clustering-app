
import streamlit as st
import geopandas as gpd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Geospatial Clustering App", layout="wide")

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("🌍 Attribute-Based Geospatial Clustering")
st.subheader("San Diego Census Tracts using K-Means & Folium")

# --------------------------------------------------
# Load Data
# --------------------------------------------------
@st.cache_data
def load_data():
    gdf = gpd.read_file("sandiego_tracts.gpkg")
    return gdf.to_crs(epsg=4326)  # Required for Folium

gdf = load_data()

# --------------------------------------------------
# Sidebar Controls
# --------------------------------------------------
st.sidebar.header("⚙️ Clustering Settings")

numeric_cols = gdf.select_dtypes(include=[np.number]).columns.tolist()

selected_features = st.sidebar.multiselect(
    "Select attributes for clustering",
    numeric_cols,
    default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
)

k = st.sidebar.slider("Number of Clusters (K)", 2, 8, 4)

# --------------------------------------------------
# Data Preview
# --------------------------------------------------
st.subheader("📊 Dataset Preview")
st.dataframe(gdf.head())

# --------------------------------------------------
# Run Clustering
# --------------------------------------------------
if selected_features:
    features = gdf[selected_features].dropna()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)

    gdf.loc[features.index, "Cluster"] = clusters

    # --------------------------------------------------
    # Folium Map
    # --------------------------------------------------
    st.subheader("🗺️ Interactive Cluster Map")

    center = [
        gdf.geometry.centroid.y.mean(),
        gdf.geometry.centroid.x.mean()
    ]

    m = folium.Map(location=center, zoom_start=10, tiles="cartodbpositron")

    folium.Choropleth(
        geo_data=gdf,
        data=gdf,
        columns=[gdf.index, "Cluster"],
        key_on="feature.id",
        fill_color="Set1",
        fill_opacity=0.7,
        line_opacity=0.3,
        legend_name="Cluster ID"
    ).add_to(m)

    folium.GeoJson(
        gdf,
        tooltip=folium.GeoJsonTooltip(
            fields=["Cluster"],
            aliases=["Cluster:"],
            localize=True
        )
    ).add_to(m)

    st_folium(m, width=1200, height=600)

    # --------------------------------------------------
    # Download Clustered GeoJSON
    # --------------------------------------------------
    st.subheader("⬇️ Download Clustered GeoJSON")

    clustered_gdf = gdf.dropna(subset=["Cluster"]).copy()
    geojson_data = clustered_gdf.to_json()

    st.download_button(
        label="Download Clustered GeoJSON",
        data=geojson_data,
        file_name="san_diego_clustered_tracts.geojson",
        mime="application/geo+json"
    )

else:
    st.warning("Please select at least one attribute for clustering.")
