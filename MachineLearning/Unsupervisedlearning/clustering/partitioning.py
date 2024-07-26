import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils import read_sample, distance_metric, type_metric

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(df_scaled)
df['Cluster'] = kmeans.labels_

# Plot K-Means clustering results
plt.figure(figsize=(10, 6))
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df['Cluster'], cmap='viridis', edgecolor='k', s=150)
plt.xlabel('Sepal Length (standardized)')
plt.ylabel('Sepal Width (standardized)')
plt.title('K-Means Clustering of Iris Data')
plt.savefig('kmeans_clustering_iris.png')
plt.close()

# K-Medoids (PAM) Clustering
# Initial medoids (randomly choose initial points)
initial_medoids = [0, 50, 100]  # These are index values of data points

# Create instance of K-Medoids algorithm
kmedoids_instance = kmedoids(df_scaled, initial_medoids, metric=distance_metric(type_metric.EUCLIDEAN))

# Run cluster analysis and obtain the result
kmedoids_instance.process()
clusters = kmedoids_instance.get_clusters()
medoids = kmedoids_instance.get_medoids()

# Convert clustering result into cluster labels for visualization
labels = np.zeros(df_scaled.shape[0])
for cluster_id, cluster in enumerate(clusters):
    for data_index in cluster:
        labels[data_index] = cluster_id

df['Cluster_PAM'] = labels

# Plot K-Medoids clustering results
plt.figure(figsize=(10, 6))
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df['Cluster_PAM'], cmap='viridis', edgecolor='k', s=150)
plt.xlabel('Sepal Length (standardized)')
plt.ylabel('Sepal Width (standardized)')
plt.title('K-Medoids Clustering of Iris Data')
plt.savefig('kmedoids_clustering_iris.png')
plt.close()
