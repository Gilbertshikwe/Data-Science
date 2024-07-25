import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import csv

# Data to be written to the CSV file
data = [
    ["Hours", "Scores"],
    [2.5, 21],
    [5.1, 47],
    [3.2, 27],
    [8.5, 75],
    [3.5, 30],
    [1.5, 20],
    [9.2, 88],
    [5.5, 60],
    [8.3, 81],
    [2.7, 25],
    [7.7, 85],
    [5.9, 62],
    [4.5, 41],
    [3.3, 42],
    [1.1, 17],
    [8.9, 95],
    [2.5, 30],
    [1.9, 24],
    [6.1, 67],
    [7.4, 69],
    [2.7, 30],
    [4.8, 54],
    [3.8, 35],
    [6.9, 76],
    [7.8, 86]
]

# Creating the CSV file
with open('student_scores.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

print("CSV file created successfully.")




# Load the data
df = pd.read_csv('student_scores.csv')

# Inspect the first few rows of the dataframe
print(df.head())

# Splitting the data into features (X) and target (y)
X = df[['Hours']]
y = df['Scores']

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')

# Plotting the regression line
plt.scatter(X, y, color='blue')  # Plotting the actual data
plt.plot(X, model.predict(X), color='red')  # Plotting the regression line
plt.title('Hours vs Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Scores Obtained')
plt.savefig('regression_plot.png')
