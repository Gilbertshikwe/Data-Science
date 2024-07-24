import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Prostate Cancer dataset
url = 'https://web.stanford.edu/~hastie/ElemStatLearn/datasets/prostate.data'
data = pd.read_csv(url, sep='\t', index_col=0)

# Drop the 'train' column
data = data.drop(columns=['train'])

# Display the first few rows of the dataset
print(data.head())

# Split the data into features and target variable
X = data.drop(columns=['lpsa'])
y = data['lpsa']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the K-Nearest Neighbors Regressor
k = 5  # You can choose different values of k
model = KNeighborsRegressor(n_neighbors=k)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Plot the actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual lpsa')
plt.ylabel('Predicted lpsa')
plt.title('Actual vs Predicted lpsa')
plt.savefig('K-nn.png')
