
import streamlit as st
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="Geospatial Clustering Project–1", layout="wide")

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("📍 PROJECT–1")
st.subheader("Attribute-Based Geospatial Clustering of San Diego Census Tracts")
st.markdown("**Using GeoPandas and K-Means (Non-Folium Version)**")

# --------------------------------------------------
# Load Data
# --------------------------------------------------
@st.cache_data
def load_data():
    return gpd.read_file("sandiego_tracts.gpkg")

gdf = load_data()

# --------------------------------------------------
# Sidebar Controls
# --------------------------------------------------
st.sidebar.header("⚙️ Clustering Parameters")

numeric_cols = gdf.select_dtypes(include=[np.number]).columns.tolist()

selected_features = st.sidebar.multiselect(
    "Select attributes for clustering",
    numeric_cols,
    default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
)

k = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=8, value=4)

# --------------------------------------------------
# Dataset Preview
# --------------------------------------------------
st.subheader("📊 Dataset Preview")
st.dataframe(gdf.head())

# --------------------------------------------------
# Clustering Logic
# --------------------------------------------------
if selected_features:
    features = gdf[selected_features].dropna()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)

    gdf.loc[features.index, "Cluster"] = clusters

    # --------------------------------------------------
    # Map Visualization (Matplotlib)
    # --------------------------------------------------
    st.subheader("🗺️ Clustered Census Tracts Map")

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    gdf.plot(
        column="Cluster",
        categorical=True,
        cmap="tab10",
        legend=True,
        edgecolor="black",
        linewidth=0.2,
        ax=ax
    )
    ax.set_axis_off()
    ax.set_title("K-Means Clustering of San Diego Census Tracts")

    st.pyplot(fig)

    # --------------------------------------------------
    # Cluster Summary
    # --------------------------------------------------
    st.subheader("📈 Cluster Summary Statistics")
    summary = gdf.groupby("Cluster")[selected_features].mean().round(2)
    st.dataframe(summary)

else:
    st.warning("Please select at least one numerical attribute for clustering.")
