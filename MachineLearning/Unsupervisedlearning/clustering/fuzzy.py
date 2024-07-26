import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler
from palmerpenguins import load_penguins

# Load the Palmer Penguins dataset
penguins = load_penguins()
df = penguins.dropna().reset_index(drop=True)  # Remove any rows with missing values and reset index

# Select features for clustering
features = ['bill_length_mm', 'bill_depth_mm']
df_selected = df[features]

# Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_selected)

# Apply Fuzzy C-Means Clustering
n_clusters = 3
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    df_scaled.T, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None
)

# Assign each point to the cluster with the highest membership degree
cluster_labels = np.argmax(u, axis=0)
df = df.copy()  # Create a copy to avoid the SettingWithCopyWarning
df['FCM_Cluster'] = cluster_labels

# Plot the Fuzzy C-Means clustering results
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['bill_length_mm'], df['bill_depth_mm'], 
                      c=df['FCM_Cluster'], cmap='viridis', edgecolor='k', s=50)
plt.xlabel('Bill Length (mm)')
plt.ylabel('Bill Depth (mm)')
plt.title('Fuzzy C-Means Clustering of Palmer Penguins')
plt.colorbar(scatter)

# Add cluster centers to the plot
cluster_centers = scaler.inverse_transform(cntr)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=200, linewidths=3)

plt.savefig('fcm_clustering_penguins.png')
plt.close()

# Print cluster centers
print("Cluster Centers:")
for i, center in enumerate(cluster_centers):
    print(f"Cluster {i}: Bill Length = {center[0]:.2f} mm, Bill Depth = {center[1]:.2f} mm")

# Print average membership degree for each cluster
print("\nAverage Membership Degrees:")
for i in range(n_clusters):
    print(f"Cluster {i}: {u[i].mean():.2f}")

# Print basic statistics of the clusters
print("\nCluster Statistics:")
print(df.groupby('FCM_Cluster')[features].mean())

# Compare clusters with actual species
print("\nCluster composition by species:")
print(pd.crosstab(df['FCM_Cluster'], df['species'], normalize='index'))

# Visualize the distribution of species within each cluster
plt.figure(figsize=(12, 6))
sns.countplot(x='FCM_Cluster', hue='species', data=df)
plt.title('Distribution of Penguin Species in Each Cluster')
plt.savefig('fcm_clustering_penguins_species_distribution.png')
plt.close()