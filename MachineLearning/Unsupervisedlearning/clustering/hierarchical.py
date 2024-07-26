import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as shc
from scipy.cluster.hierarchy import linkage, fcluster

# Load the Wine dataset
wine = load_wine()
df = pd.DataFrame(data=wine.data, columns=wine.feature_names)

# Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Plot the Dendrogram for Agglomerative Clustering
plt.figure(figsize=(10, 7))
plt.title("Wine Dendrogram")
dendrogram = shc.dendrogram(shc.linkage(df_scaled, method='ward'))
plt.savefig('wine_dendrogram.png')
plt.close()

# Apply Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
agg_clustering.fit(df_scaled)
df['Agglomerative_Cluster'] = agg_clustering.labels_

# Plot the Agglomerative Clustering results
plt.figure(figsize=(10, 7))
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df['Agglomerative_Cluster'], cmap='viridis', edgecolor='k', s=150)
plt.xlabel('Alcohol (standardized)')
plt.ylabel('Malic Acid (standardized)')
plt.title('Agglomerative Clustering of Wine Data')
plt.savefig('agglomerative_clustering_wine.png')
plt.close()

# Divisive clustering using hierarchical clustering
# Perform hierarchical clustering
Z = linkage(df_scaled, method='ward')

# Define a function to split the clusters
def divisive_clustering(Z, num_clusters):
    clusters = fcluster(Z, t=num_clusters, criterion='maxclust')
    return clusters

# Get clusters for Divisive Clustering
num_clusters = 3
clusters = divisive_clustering(Z, num_clusters)
df['Divisive_Cluster'] = clusters

# Plot the Divisive Clustering results
plt.figure(figsize=(10, 7))
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df['Divisive_Cluster'], cmap='viridis', edgecolor='k', s=150)
plt.xlabel('Alcohol (standardized)')
plt.ylabel('Malic Acid (standardized)')
plt.title('Divisive Clustering of Wine Data')
plt.savefig('divisive_clustering_wine.png')
plt.close()

