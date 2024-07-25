import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Create a synthetic dataset
np.random.seed(42)
n_samples = 1000

years_experience = np.random.normal(5, 2, n_samples)  # Mean of 5 years, standard deviation of 2 years
education_level = np.random.randint(1, 5, n_samples)  # 1 to 4 (e.g., 1: Bachelor's, 2: Master's, etc.)
job_level = np.random.randint(1, 4, n_samples)        # 1 to 3 (e.g., 1: Junior, 2: Mid, 3: Senior)

# Assume a simple model for salary
salary = (years_experience * 5000) + (education_level * 7000) + (job_level * 10000) + np.random.normal(0, 10000, n_samples)

# Create a DataFrame
data = pd.DataFrame({
    'years_experience': years_experience,
    'education_level': education_level,
    'job_level': job_level,
    'salary': salary
})

# Display basic statistics
print(data.head())
print(data.describe())

# Visualize the relationships
sns.pairplot(data)
plt.show()

# Split the data
X = data.drop('salary', axis=1)
y = data['salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Decision Tree Regressor
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Plot the actual vs predicted salaries
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Salaries')
plt.ylabel('Predicted Salaries')
plt.title('Actual vs Predicted Salaries')
plt.savefig('salaries.png')

# Visualize the decision tree (optional)
from sklearn.tree import plot_tree

plt.figure(figsize=(20,10))
plot_tree(model, filled=True, feature_names=data.columns[:-1], rounded=True)
plt.savefig('decison_tree.png')
