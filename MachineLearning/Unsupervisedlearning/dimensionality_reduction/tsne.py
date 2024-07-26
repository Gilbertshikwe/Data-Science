import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Load the Digits dataset
digits = load_digits()
df = pd.DataFrame(data=digits.data)

# Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(df_scaled)
df_tsne = pd.DataFrame(data=tsne_results, columns=['Dim1', 'Dim2'])

# Plot the t-SNE results
plt.figure(figsize=(10, 7))
plt.scatter(df_tsne['Dim1'], df_tsne['Dim2'], c=digits.target, cmap='viridis', edgecolor='k', s=150)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('t-SNE of Digits Dataset')
plt.savefig('tsne_digits_dataset.png')
plt.close()
