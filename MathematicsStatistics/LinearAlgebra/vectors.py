# A vector is an ordered array of numbers, 
# which can represent points in space, directions, or other quantities.

import numpy as np


# Creating a vector
v = np.array([1, 2, 3])
print(v)  

# Vector operations
u = np.array([4, 5, 6])
scalar = 2

# Addition
v_plus_u = v + u
print(v_plus_u) 

# Scalar multiplication
scalar_mult = scalar * v
print(scalar_mult)  