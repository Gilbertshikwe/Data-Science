# A matrix is a two-dimensional array of numbers.

import numpy as np



# Creating matrices
A = np.array([[1, 2], [3, 4]])
print(A)
# Output:
# [[1 2]
#  [3 4]]

B = np.array([[1, 0], [0, 1]])
print(B)
# Output:
# [[1 0]
#  [0 1]]

# Matrix operations
C = A + B
print(C)
# Output:
# [[2 2]
#  [3 5]]

# Matrix multiplication
D = np.dot(A, B)
print(D)
# Output:
# [[1 2]
#  [3 4]]
