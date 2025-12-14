import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(page_title="Amazon Music Clustering", layout="wide")

# Title and Description
st.title("🎵 Amazon Music Clustering Analysis")
st.markdown("""
This application groups songs into clusters based on their audio characteristics using K-Means clustering.
Explore the clusters to understand different musical patterns.
""")

# Load Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('src/data/raw/single_genre_artists.csv')
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'src/data/raw/single_genre_artists.csv' exists.")
        return None

df = load_data()

if df is not None:
    # Sidebar Configuration
    st.sidebar.header("Configuration")
    
    # Feature Selection
    default_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
    selected_features = st.sidebar.multiselect("Select Features for Clustering", default_features, default=default_features)
    
    k = st.sidebar.slider("Number of Clusters (k)", min_value=2, max_value=10, value=4)
    
    if len(selected_features) > 0:
        # Preprocessing
        X = df[selected_features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Clustering
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        df['cluster'] = clusters
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Cluster Visualization", "📈 Cluster Profiles", "📄 Data Overview"])
        
        with tab1:
            st.subheader("PCA Visualization")
            st.markdown("Dimensionality reduction to 2D for visualization purposes.")
            
            pca = PCA(n_components=2)
            components = pca.fit_transform(X_scaled)
            df['pca1'] = components[:, 0]
            df['pca2'] = components[:, 1]
            
            # Convert cluster to string for categorical coloring
            df['cluster_str'] = df['cluster'].astype(str)
            
            fig = px.scatter(
                df, 
                x='pca1', 
                y='pca2', 
                color='cluster_str',
                hover_data=['artist_name', 'track_name'] if 'artist_name' in df.columns else None,
                title=f"K-Means Clustering (k={k})",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            st.subheader("Cluster Profiles")
            st.markdown("Average feature values for each cluster.")
            
            profile = df.groupby('cluster')[selected_features].mean()
            
            # Display as a styled dataframe
            st.dataframe(profile.style.background_gradient(cmap='viridis', axis=0))
            
            # Heatmap
            st.subheader("Feature Heatmap")
            fig_heatmap, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(profile.T, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig_heatmap)
            
        with tab3:
            st.subheader("Clustered Data")
            st.dataframe(df)
            
            st.subheader("Download Results")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name='clustered_music_data.csv',
                mime='text/csv',
            )
            
    else:
        st.warning("Please select at least one feature for clustering.")
