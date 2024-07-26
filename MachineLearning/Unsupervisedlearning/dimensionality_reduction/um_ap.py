import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Apply UMAP
umap_model = umap.UMAP(n_components=2, random_state=42)
umap_results = umap_model.fit_transform(df_scaled)
df_umap = pd.DataFrame(data=umap_results, columns=['Dim1', 'Dim2'])

# Plot the UMAP results
plt.figure(figsize=(10, 7))
plt.scatter(df_umap['Dim1'], df_umap['Dim2'], c=iris.target, cmap='viridis', edgecolor='k', s=150)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('UMAP of Iris Dataset')
plt.savefig('umap_iris_dataset.png')
plt.close()

