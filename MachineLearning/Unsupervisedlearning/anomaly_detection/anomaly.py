import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Create a synthetic dataset
X, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.60, random_state=0)
X_anomalies = np.array([[3, 3], [-3, -3], [3, -3], [-3, 3]])  # Adding anomalies
X_with_anomalies = np.vstack([X, X_anomalies])

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_with_anomalies)

# Apply Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_forest.fit(X_scaled)
iso_labels = iso_forest.predict(X_scaled)

# Apply One-Class SVM
oc_svm = OneClassSVM(gamma='auto', nu=0.1)
oc_svm.fit(X_scaled)
oc_labels = oc_svm.predict(X_scaled)

# Plot Isolation Forest results
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=iso_labels, cmap='coolwarm', edgecolor='k', s=50)
plt.title('Isolation Forest Anomaly Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot One-Class SVM results
plt.subplot(1, 2, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=oc_labels, cmap='coolwarm', edgecolor='k', s=50)
plt.title('One-Class SVM Anomaly Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.savefig('anomaly_detection_comparison.png')
plt.close()
