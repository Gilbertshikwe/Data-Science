import numpy as np

def main():
    # Matrix Multiplication
    print("Matrix Multiplication:")
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    C = np.dot(A, B)  # Using np.dot
    D = A @ B  # Using @ operator
    print("Matrix A:\n", A)
    print("Matrix B:\n", B)
    print("Matrix C (A dot B):\n", C)
    print("Matrix D (A @ B):\n", D)
    print("\n")

    # Matrix Transpose
    print("Matrix Transpose:")
    A = np.array([[1, 2, 3], [4, 5, 6]])
    A_T = A.T
    print("Matrix A:\n", A)
    print("Transpose of A:\n", A_T)
    print("\n")

    # Determinant
    print("Determinant:")
    A = np.array([[1, 2], [3, 4]])
    det_A = np.linalg.det(A)
    print("Matrix A:\n", A)
    print("Determinant of A:", det_A)
    print("\n")

    # Inverse
    print("Inverse:")
    if np.linalg.det(A) != 0:
        inv_A = np.linalg.inv(A)
        print("Matrix A:\n", A)
        print("Inverse of A:\n", inv_A)
    else:
        print("Matrix A is not invertible.")
    print("\n")

    # Eigenvalues and Eigenvectors
    print("Eigenvalues and Eigenvectors:")
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print("Matrix A:\n", A)
    print("Eigenvalues:\n", eigenvalues)
    print("Eigenvectors:\n", eigenvectors)
    print("\n")

    # Singular Value Decomposition (SVD)
    print("Singular Value Decomposition (SVD):")
    A = np.array([[1, 2], [3, 4], [5, 6]])
    U, S, V = np.linalg.svd(A)
    print("Matrix A:\n", A)
    print("U matrix:\n", U)
    print("Singular values:\n", S)
    print("V matrix:\n", V)
    print("\n")

    # Solving Linear Systems
    print("Solving Linear Systems:")
    A = np.array([[3, 1], [1, 2]])
    B = np.array([9, 8])
    det_A = np.linalg.det(A)
    print("Determinant of A:", det_A)
    if det_A != 0:
        inv_A = np.linalg.inv(A)
        print("Inverse of A:\n", inv_A)
        X = np.linalg.solve(A, B)
        print("Solution of the system Ax = B:\n", X)
    else:
        print("Matrix A is not invertible.")
    print("\n")

if __name__ == "__main__":
    main()
