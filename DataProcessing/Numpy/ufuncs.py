# Universal Functions (ufuncs)
# Universal functions (ufuncs) are functions that operate element-wise on ndarrays.
# NumPy provides a variety of built-in ufuncs, including mathematical, 
# logical, bitwise, and comparison operations

import numpy as np

# Create an array
arr = np.array([1, 2, 3, 4])

# Apply ufuncs
squared = np.square(arr)
sqrt = np.sqrt(arr)
exp = np.exp(arr)
log = np.log(arr)

print("Squared:", squared)
print("Square Root:", sqrt)
print("Exponential:", exp)
print("Logarithm:", log)
