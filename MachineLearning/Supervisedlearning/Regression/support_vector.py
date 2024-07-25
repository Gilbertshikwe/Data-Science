import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Visualize the distribution of the classes
plt.figure()
sns.countplot(x=y)
plt.title('Distribution of Classes')
plt.savefig('class_distribution.png')
plt.close()

# We will use only two features for easier visualization
X = X[['mean radius', 'mean texture']]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVR model
svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr.fit(X_train_scaled, y_train)

# Predict
y_pred = svr.predict(X_test_scaled)

# Visualize the scatter plot of the input features
plt.figure()
plt.scatter(X['mean radius'], X['mean texture'], c=y, cmap='coolwarm')
plt.xlabel('Mean Radius')
plt.ylabel('Mean Texture')
plt.title('Input Features Scatter Plot')
plt.savefig('input_features.png')
plt.close()

# Plot the decision boundary
def plot_decision_boundary(X, y, model, scaler):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    X_plot = np.c_[xx.ravel(), yy.ravel()]
    X_plot_scaled = scaler.transform(X_plot)
    Z = model.predict(X_plot_scaled)
    Z = Z.reshape(xx.shape)
    
    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap='coolwarm')
    plt.xlabel('Mean Radius')
    plt.ylabel('Mean Texture')
    plt.title('Decision Boundary')
    plt.savefig('decision_boundary.png')
    plt.close()

# Plot decision boundary
plot_decision_boundary(X_train.values, y_train, svr, scaler)
