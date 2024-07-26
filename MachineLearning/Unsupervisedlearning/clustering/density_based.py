import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

# Load the California Housing dataset
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)

# Select features for clustering
features = ['AveRooms', 'MedInc']  # Average number of rooms and median income
df_selected = df[features]

# Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_selected)

# Apply DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(df_scaled)
df['DBSCAN_Cluster'] = dbscan.labels_

# Plot the DBSCAN clustering results
plt.figure(figsize=(10, 7))
scatter = plt.scatter(df['AveRooms'], df['MedInc'], c=df['DBSCAN_Cluster'], cmap='viridis', edgecolor='k', s=50)
plt.xlabel('Average Number of Rooms')
plt.ylabel('Median Income')
plt.title('DBSCAN Clustering of California Housing Data')
plt.colorbar(scatter)
plt.savefig('dbscan_clustering_california.png')
plt.close()

# Apply OPTICS Clustering
optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.1)
optics.fit(df_scaled)
df['OPTICS_Cluster'] = optics.labels_

# Plot the OPTICS clustering results
plt.figure(figsize=(10, 7))
scatter = plt.scatter(df['AveRooms'], df['MedInc'], c=df['OPTICS_Cluster'], cmap='viridis', edgecolor='k', s=50)
plt.xlabel('Average Number of Rooms')
plt.ylabel('Median Income')
plt.title('OPTICS Clustering of California Housing Data')
plt.colorbar(scatter)
plt.savefig('optics_clustering_california.png')
plt.close()

# Print the number of clusters found by each algorithm
print(f"Number of clusters found by DBSCAN: {len(set(df['DBSCAN_Cluster'])) - (1 if -1 in df['DBSCAN_Cluster'] else 0)}")
print(f"Number of clusters found by OPTICS: {len(set(df['OPTICS_Cluster'])) - (1 if -1 in df['OPTICS_Cluster'] else 0)}")

# Basic statistics of the clusters
print("\nDBSCAN Cluster Statistics:")
print(df.groupby('DBSCAN_Cluster')[features].mean())

print("\nOPTICS Cluster Statistics:")
print(df.groupby('OPTICS_Cluster')[features].mean())
