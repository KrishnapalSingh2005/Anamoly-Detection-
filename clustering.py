import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

def render(df):
    st.header("K-Means Clustering")
    st.markdown("Automatically group your data into distinct clusters based on similarity.")
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("You need at least 2 numeric columns for 2D clustering visualization.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            cluster_features = st.multiselect("Select 2 Features for Clustering:", numeric_cols, default=numeric_cols[:2])
        with col2:
            num_clusters = st.slider("Select Number of Clusters (K):", min_value=2, max_value=10, value=3)
            
        if len(cluster_features) == 2:
            cluster_df = df[cluster_features].dropna()
            
            # K-Means Training
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(cluster_df)
            cluster_df['Cluster'] = cluster_labels
            
            st.markdown("### Clustering Visualization")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            sns.scatterplot(
                data=cluster_df, 
                x=cluster_features[0], 
                y=cluster_features[1], 
                hue='Cluster', 
                palette='tab10', 
                ax=ax,
                alpha=0.7
            )
            
            # Plot centroids
            centroids = kmeans.cluster_centers_
            ax.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', label='Centroids')
            
            ax.set_title(f"K-Means Clustering (K={num_clusters})")
            ax.legend()
            st.pyplot(fig)
            
            # Insights
            st.markdown(f"""
            <div class="insight-box">
                <strong>💡 AI Insight:</strong> The algorithm has partitioned the data into {num_clusters} distinct groups based on {cluster_features[0]} and {cluster_features[1]}. The red 'X' markers represent the mathematical center (centroid) of each cluster.
            </div>
            """, unsafe_allow_html=True)
            
        elif len(cluster_features) > 2:
            st.warning("Please select exactly 2 features for 2D visualization.")
        else:
            st.warning("Please select at least 2 features.")
