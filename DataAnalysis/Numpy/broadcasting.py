# Broadcasting
# Broadcasting is a powerful feature that allows NumPy 
# to perform arithmetic operations on arrays of different shapes.
# This feature makes it possible to vectorize operations, leading to more efficient code.


import numpy as np

# Create a 1D array
a = np.array([1, 2, 3])

# Create a 2D array
b = np.array([[4], [5], [6]])

# Broadcasting allows us to add these arrays directly
result = a + b

print("Array a:\n", a)
print("Array b:\n", b)
print("Result of broadcasting:\n", result)

# Monthly sales data (rows: products, columns: months)
sales_data = np.array([[100, 150, 200], [120, 180, 240]])

# Monthly growth factors
growth_factors = np.array([1.05, 1.07, 1.08])

# Adjust sales data with growth factors
adjusted_sales = sales_data * growth_factors
print("Adjusted Sales Data:\n", adjusted_sales)
# Output:
# [[105.  160.5 216. ]
# [126.  192.6 259.2]]