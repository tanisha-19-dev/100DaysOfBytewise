import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv('Mall_Customers.csv')

# Selecting the features (Annual Income and Spending Score)
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Visualize the K-Means clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=df, palette='viridis')
plt.title('K-Means Customer Segments')
plt.show()

# Finding the Optimal Number of Clusters using Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Finding the Optimal Number of Clusters using Silhouette Score
silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    preds = kmeans.fit_predict(X)
    silhouette_scores.append(silhouette_score(X, preds))

plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Score')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

# Cluster Profiling
cluster_profile = df.groupby('Cluster').mean()

# Display Cluster Profile Insights
print("Cluster Profiles:")
print(cluster_profile)

# Visualize Cluster Profiles
plt.figure(figsize=(10, 6))
sns.heatmap(cluster_profile, annot=True, cmap='coolwarm')
plt.title('Cluster Profiles')
plt.show()

# Apply Hierarchical Clustering
linked = linkage(X, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked)
plt.title('Dendrogram')
plt.show()

# Agglomerative Clustering
hierarchical = AgglomerativeClustering(n_clusters=5)
df['Cluster_HC'] = hierarchical.fit_predict(X)

# Visualize Hierarchical Clustering Segments
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster_HC', data=df, palette='coolwarm')
plt.title('Hierarchical Clustering Segments')
plt.show()

# Apply PCA to reduce dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualize K-Means Clusters in PCA-reduced Space
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Cluster'], palette='viridis')
plt.title('K-Means Clusters in PCA-reduced Space')
plt.show()

# Visualize Hierarchical Clusters in PCA-reduced Space
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Cluster_HC'], palette='coolwarm')
plt.title('Hierarchical Clusters in PCA-reduced Space')
plt.show()
