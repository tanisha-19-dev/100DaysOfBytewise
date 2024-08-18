import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = iris.target

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.title("Clustering Analysis on Iris Dataset")

# Task 1: Implementing K-Means Clustering
st.header("Task 1: K-Means Clustering")
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['kmeans_clusters'] = clusters

fig1, ax1 = plt.subplots()
scatter = ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', marker='o')
ax1.set_title('K-Means Clustering of Iris Dataset')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
plt.colorbar(scatter, ax=ax1)
st.pyplot(fig1)

# Task 2: Choosing the Optimal Number of Clusters
st.header("Task 2: Choosing the Optimal Number of Clusters")

# Elbow Method
st.subheader("Elbow Method")
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

fig2, ax2 = plt.subplots()
ax2.plot(range(1, 11), wcss, marker='o')
ax2.set_title('Elbow Method for Optimal Number of Clusters')
ax2.set_xlabel('Number of clusters')
ax2.set_ylabel('WCSS')
st.pyplot(fig2)

# Silhouette Score
st.subheader("Silhouette Score")
silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

fig3, ax3 = plt.subplots()
ax3.plot(range(2, 11), silhouette_scores, marker='o')
ax3.set_title('Silhouette Score for Optimal Number of Clusters')
ax3.set_xlabel('Number of clusters')
ax3.set_ylabel('Silhouette Score')
st.pyplot(fig3)

# Task 3: Cluster Visualization with PCA
st.header("Task 3: Cluster Visualization with PCA")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters_pca = kmeans.fit_predict(X_pca)
df['kmeans_clusters_pca'] = clusters_pca

fig4, ax4 = plt.subplots()
scatter_pca = ax4.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_pca, cmap='viridis', marker='o')
ax4.set_title('K-Means Clustering of Iris Dataset (PCA-reduced)')
ax4.set_xlabel('Principal Component 1')
ax4.set_ylabel('Principal Component 2')
plt.colorbar(scatter_pca, ax=ax4)
st.pyplot(fig4)

# Task 4: Hierarchical Clustering - Dendrogram
st.header("Task 4: Hierarchical Clustering - Dendrogram")
Z = linkage(X_scaled, method='ward')

fig5, ax5 = plt.subplots(figsize=(12, 8))
dendrogram(Z, truncate_mode='level', p=3, ax=ax5)
ax5.set_title('Dendrogram for Hierarchical Clustering')
ax5.set_xlabel('Sample index')
ax5.set_ylabel('Distance')
st.pyplot(fig5)

# Task 5: Comparing Clustering Algorithms
st.header("Task 5: Comparing Clustering Algorithms")
agg_clustering = AgglomerativeClustering(n_clusters=3)
clusters_agg = agg_clustering.fit_predict(X_scaled)
df['agg_clusters'] = clusters_agg

fig6, (ax6_1, ax6_2) = plt.subplots(1, 2, figsize=(14, 6))

scatter_kmeans = ax6_1.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_pca, cmap='viridis', marker='o')
ax6_1.set_title('K-Means Clustering (PCA-reduced)')
ax6_1.set_xlabel('Principal Component 1')
ax6_1.set_ylabel('Principal Component 2')

scatter_agg = ax6_2.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_agg, cmap='viridis', marker='o')
ax6_2.set_title('Agglomerative Hierarchical Clustering (PCA-reduced)')
ax6_2.set_xlabel('Principal Component 1')
ax6_2.set_ylabel('Principal Component 2')

plt.colorbar(scatter_kmeans, ax=ax6_1)
plt.colorbar(scatter_agg, ax=ax6_2)
st.pyplot(fig6)

# Discussion on strengths and weaknesses
st.header("Discussion on Strengths and Weaknesses")

st.subheader("K-Means Clustering:")
st.write("""
- **Strengths:**
  - Computationally efficient for large datasets.
  - Easy to implement and interpret.
  - Works well when clusters are spherical and evenly sized.
- **Weaknesses:**
  - Sensitive to the initial choice of centroids.
  - May struggle with clusters of varying sizes and densities.
  - Assumes clusters are convex shapes.
""")

st.subheader("Agglomerative Hierarchical Clustering:")
st.write("""
- **Strengths:**
  - Does not require specifying the number of clusters in advance.
  - Can capture complex hierarchical relationships between data points.
  - Provides a dendrogram for visualizing the clustering process.
- **Weaknesses:**
  - Computationally intensive, especially for large datasets.
  - Difficult to interpret for very large datasets.
  - Less efficient for high-dimensional data.
""")
