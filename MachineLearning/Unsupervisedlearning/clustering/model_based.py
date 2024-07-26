import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

# Load the Wine dataset
wine = load_wine()
df = pd.DataFrame(data=wine.data, columns=wine.feature_names)

# Select features for clustering
df_selected = df[['alcohol', 'malic_acid']]  # Using two features for simplicity

# Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_selected)

# Apply Gaussian Mixture Model Clustering
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(df_scaled)
df['GMM_Cluster'] = gmm.predict(df_scaled)

# Plot the GMM clustering results
plt.figure(figsize=(10, 7))
plt.scatter(df['alcohol'], df['malic_acid'], c=df['GMM_Cluster'], cmap='viridis', edgecolor='k', s=150)
plt.xlabel('Alcohol')
plt.ylabel('Malic Acid')
plt.title('GMM Clustering of Wine Data')
plt.savefig('gmm_clustering_wine.png')
plt.close()

