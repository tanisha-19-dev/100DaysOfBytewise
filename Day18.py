import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

# Load the Wholesale Customers dataset
df = pd.read_csv('Wholesale_customers_data.csv')

# Selecting the features (Annual spending in different categories)
X = df.iloc[:, 2:]  # Assuming the first two columns are not used for clustering

# Task 1: K-Means Clustering for Customer Segmentation
# Apply K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster_KMeans'] = kmeans.fit_predict(X)

# Visualize the K-Means clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X['Grocery'], y=X['Milk'], hue='Cluster_KMeans', data=df, palette='viridis')
plt.title('K-Means Customer Segments')
plt.xlabel('Grocery Spending')
plt.ylabel('Milk Spending')
plt.show()

# Task 2: Evaluating the Optimal Number of Clusters
# Elbow Method
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

# Silhouette Score
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

# Task 3: Cluster Analysis and Interpretation
# Cluster Profiling
cluster_profile = df.groupby('Cluster_KMeans').mean()

# Display Cluster Profile Insights
print("Cluster Profiles:")
print(cluster_profile)

# Visualize Cluster Profiles
plt.figure(figsize=(10, 6))
sns.heatmap(cluster_profile, annot=True, cmap='coolwarm')
plt.title('Cluster Profiles')
plt.show()

# Task 4: Hierarchical Clustering: Dendrogram and Cluster Formation
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
sns.scatterplot(x=X['Grocery'], y=X['Milk'], hue='Cluster_HC', data=df, palette='coolwarm')
plt.title('Hierarchical Clustering Segments')
plt.xlabel('Grocery Spending')
plt.ylabel('Milk Spending')
plt.show()

# Task 5: Comparison of Clustering Results
# Apply PCA to reduce dimensions for comparison
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualize K-Means Clusters in PCA-reduced Space
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Cluster_KMeans'], palette='viridis')
plt.title('K-Means Clusters in PCA-reduced Space')
plt.show()

# Visualize Hierarchical Clusters in PCA-reduced Space
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Cluster_HC'], palette='coolwarm')
plt.title('Hierarchical Clusters in PCA-reduced Space')
plt.show()

# Discussion
print("Comparison of Clustering Methods:")
print("K-Means tends to form more compact clusters, while Hierarchical Clustering can show more flexibility in cluster formation.")
