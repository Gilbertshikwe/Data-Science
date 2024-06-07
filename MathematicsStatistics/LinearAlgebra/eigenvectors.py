# Eigenvalues and eigenvectors are properties of a square matrix. For a matrix 
# 𝐴A, if 𝑣v is a vector and 𝜆λ is a scalar such that 𝐴𝑣=𝜆𝑣Av=λv, 
# then 𝑣v is an eigenvector and 𝜆λ is an eigenvalue.

import numpy as np


# Define the matrix
matrix = np.array([[1, 2], [3, 4]])

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(matrix)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
