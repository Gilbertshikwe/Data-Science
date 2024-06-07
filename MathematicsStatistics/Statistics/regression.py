# Regression analysis estimates the relationships among variables.

import statsmodels.api as sm #type:ignore # For statistical analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt #type:ignore # For plotting

# Define data
# Define the data
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
y = np.array([55, 60, 61, 62, 65, 66, 68, 69, 70, 72, 75, 78, 80, 82, 85, 87, 89, 92, 95, 98])

# Add a constant term for the intercept
X = sm.add_constant(X)

# Fit the model
model = sm.OLS(y, X).fit()
predictions = model.predict(X)

print(model.summary())

# Scatter plot of the data
plt.scatter(X[:, 1], y, label='Data Points')

# Plot the regression line
plt.plot(X[:, 1], predictions, color='red', label='Fitted Line')

plt.xlabel('Hours Studied')
plt.ylabel('Test Scores')
plt.legend()
plt.title('Regression Analysis: Hours Studied vs. Test Scores')
plt.show()

