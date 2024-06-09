# NumPy

NumPy is a powerful Python library for numerical computing that provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays. It is widely used in scientific computing, data analysis, and machine learning.

## Installation

NumPy can be easily installed using pip, the Python package installer. Make sure you have Python installed on your system before proceeding with the following steps.

```bash
pip install numpy
```

## Usage

### Importing NumPy

Before using NumPy in your Python script or Jupyter Notebook, you need to import it:

```python
import numpy as np
```

### Creating Arrays

NumPy's main object is the ndarray, a multi-dimensional array. Here are some common ways to create arrays:

```python
# Create an array from a Python list
my_array = np.array([1, 2, 3, 4, 5])

# Create an array with a range of values
my_range_array = np.arange(0, 10, 2)  # Start from 0, end at 10 (exclusive), step by 2

# Create arrays of zeros or ones
zeros_array = np.zeros((3, 3))  # 3x3 array of zeros
ones_array = np.ones((2, 4))     # 2x4 array of ones
```

### Array Operations

You can perform element-wise operations on arrays:

```python
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])

# Element-wise addition
result_addition = array1 + array2  # Output: [5, 7, 9]

# Element-wise multiplication
result_multiplication = array1 * array2  # Output: [4, 10, 18]
```

### Indexing and Slicing

Access elements or slices of arrays using indexing and slicing:

```python
my_array = np.array([1, 2, 3, 4, 5])

# Accessing elements
print(my_array[0])    # Output: 1
print(my_array[-1])   # Output: 5

# Slicing
print(my_array[1:4])  # Output: [2, 3, 4]
```

### Array Shape and Reshaping

Get array shape and reshape arrays:

```python
my_array = np.array([[1, 2, 3], [4, 5, 6]])

# Get array shape
print(my_array.shape)  # Output: (2, 3) - 2 rows, 3 columns

# Reshape array
reshaped_array = my_array.reshape(3, 2)
```

## Documentation

For more information and advanced usage, refer to the [NumPy documentation](https://numpy.org/doc/).


# NumPy: Random Data, Sorting, and Searching

NumPy is a powerful Python library for scientific computing that provides support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays. This README covers the topics of random data generation, sorting, and searching in NumPy.

## Random Data in NumPy

NumPy provides a sub-module called `numpy.random` that allows you to generate random numbers from various probability distributions. Here are some common functions:

- `numpy.random.rand(size)`: Generates an array of random floats between 0 and 1.
- `numpy.random.randn(size)`: Generates an array of random numbers from a standard normal distribution (Gaussian, with mean=0 and std=1).
- `numpy.random.randint(low, high, size)`: Generates an array of random integers within a specified range.
- `numpy.random.choice(array, size, replace=True)`: Generates an array by randomly sampling from the given array.

```python
import numpy as np

# Generate a 3x3 array of random floats
random_array = np.random.rand(3, 3)

# Generate 5 random integers between 1 and 10
random_integers = np.random.randint(1, 11, size=5)
```

## Sorting in NumPy

NumPy provides various sorting functions to sort arrays along one or more axes. The most common function is `numpy.sort()`:

```python
arr = np.array([5, 2, 8, 1, 9])

# Sort the array in ascending order
sorted_arr = np.sort(arr)
print(sorted_arr)  # Output: [1 2 5 8 9]
```

You can also sort arrays along a specific axis for multi-dimensional arrays:

```python
arr = np.array([[3, 1, 4], [2, 6, 5]])

# Sort along the column axis (axis=0)
sorted_arr = np.sort(arr, axis=0)
print(sorted_arr)
# Output: [[2 1 4]
#          [3 6 5]]
```

## Searching in NumPy

NumPy provides functions to search for specific elements or values in an array. Here are some common functions:

- `numpy.where(condition)`: Returns the indices of elements that satisfy the given condition.
- `numpy.argmax(arr)`: Returns the index of the maximum value in the array.
- `numpy.argmin(arr)`: Returns the index of the minimum value in the array.
- `numpy.searchsorted(arr, value)`: Finds the indices where the given value should be inserted to maintain the sorted order.

```python
arr = np.array([1, 4, 2, 7, 3])

# Find indices of elements greater than 3
indices = np.where(arr > 3)
print(indices)  # Output: (array([1, 3]),)

# Find the index of the maximum value
max_index = np.argmax(arr)
print(max_index)  # Output: 3

# Find the indices where 5 should be inserted to maintain sorted order
insert_index = np.searchsorted(arr, 5)
print(insert_index)  # Output: 3
```

These are just a few examples of the functionalities provided by NumPy for random data generation, sorting, and searching. NumPy offers many more advanced functions and capabilities for efficient numerical computing with arrays and matrices in Python.


## Advanced Numpy Features

This repository contains examples and explanations of advanced NumPy features, including Universal Functions (ufuncs) and Broadcasting. These features enable efficient and powerful operations on arrays, leading to more concise and faster code.

## Prerequisites

To run the examples in this repository, you need to have Python installed along with the `numpy` library. You can install `numpy` using `pip`:

```sh
pip install numpy
```

## Universal Functions (ufuncs)

Universal functions, or ufuncs, are functions that operate element-wise on `ndarray` objects. NumPy provides a wide variety of built-in ufuncs, including mathematical, logical, bitwise, and comparison operations. These functions are highly optimized and can perform operations much faster than traditional loops in Python.

### Example with ufuncs

Here's an example demonstrating the use of some common ufuncs:

```python
import numpy as np

# Initialize arrays
arr1 = np.array([0, 1, 2, 3])
arr2 = np.array([10, 11, 12, 13])

# Element-wise addition
sum_arr = np.add(arr1, arr2)
print("Sum:", sum_arr)  # Output: [10 12 14 16]

# Element-wise multiplication
product_arr = np.multiply(arr1, arr2)
print("Product:", product_arr)  # Output: [ 0 11 24 39]

# Element-wise comparison
comparison_arr = np.greater(arr2, arr1)
print("Comparison (arr2 > arr1):", comparison_arr)  # Output: [ True  True  True  True]
```


### Broadcasting

Broadcasting is a powerful feature in NumPy that allows arithmetic operations on arrays of different shapes. It enables vectorized operations, which lead to more efficient and concise code. Understanding the rules of broadcasting is essential for leveraging this feature.

#### Rules of Broadcasting

1. **Rank Adjustment**: If the arrays do not have the same rank, prepend the shape of the smaller array with ones until both shapes have the same length.
2. **Output Shape**: The size of the output shape is the maximum size along each dimension of the input shapes.
3. **Shape Compatibility**: An input array can be implicitly broadcast to the output shape if its shape is either equal to the output shape or equal to 1 in that dimension.

### Example with Broadcasting

Here's an example demonstrating broadcasting:

```python
import numpy as np

# Initialize arrays
arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([1, 2, 3])

# Broadcasting addition
sum_arr = arr1 + arr2
print("Sum with broadcasting:\n", sum_arr)
# Output:
# [[2 4 6]
# [5 7 9]]

# Broadcasting rules applied:
# arr1 shape: (2, 3)
# arr2 shape: (3,) -> (1, 3) after prepending ones
# Result shape: (2, 3)
```

### Real-life Application

**Sales Data Analysis**:
Imagine you have sales data for different products over several months. You want to adjust these sales figures by applying a monthly growth factor. Broadcasting allows you to efficiently apply this factor across the entire dataset.

```python
import numpy as np

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
```

In this example, broadcasting makes it straightforward to apply the growth factors to each month's sales data.

## Conclusion

This repository covers key advanced features of NumPy, including Universal Functions (ufuncs) and Broadcasting. These features are essential for efficient numerical computations and data analysis. By understanding and utilizing these tools, you can write more efficient and powerful code for a variety of applications.


# Linear Algebra with NumPy

NumPy provides a powerful set of functions for performing linear algebra operations on arrays. This README covers some of the most common linear algebra operations in NumPy.

## 1. Matrix Multiplication

Matrix multiplication can be performed using the `np.dot` function or the `@` operator.

```python
import numpy as np

# Creating two matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication
C = np.dot(A, B)  # Using np.dot
D = A @ B  # Using @ operator

print("Matrix A:\n", A)
print("Matrix B:\n", B)
print("Matrix C (A dot B):\n", C)
print("Matrix D (A @ B):\n", D)
```

## 2. Matrix Transpose

The transpose of a matrix can be obtained using the `T` attribute.

```python
A = np.array([[1, 2, 3], [4, 5, 6]])
A_T = A.T

print("Matrix A:\n", A)
print("Transpose of A:\n", A_T)
```

## 3. Determinant

The determinant of a matrix can be calculated using `np.linalg.det`.

```python
A = np.array([[1, 2], [3, 4]])
det_A = np.linalg.det(A)

print("Matrix A:\n", A)
print("Determinant of A:", det_A)
```

## 4. Inverse

The inverse of a matrix can be calculated using `np.linalg.inv`.

```python
A = np.array([[1, 2], [3, 4]])
inv_A = np.linalg.inv(A)

print("Matrix A:\n", A)
print("Inverse of A:\n", inv_A)
```

## 5. Eigenvalues and Eigenvectors

Eigenvalues and eigenvectors can be computed using `np.linalg.eig`.

```python
A = np.array([[1, 2], [3, 4]])
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Matrix A:\n", A)
print("Eigenvalues:\n", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
```

## 6. Singular Value Decomposition (SVD)

SVD can be performed using `np.linalg.svd`.

```python
A = np.array([[1, 2], [3, 4], [5, 6]])
U, S, V = np.linalg.svd(A)

print("Matrix A:\n", A)
print("U matrix:\n", U)
print("Singular values:\n", S)
print("V matrix:\n", V)
```

## 7. Solving Linear Systems

You can solve a system of linear equations `Ax = B` using `np.linalg.solve`.

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([5, 6])
X = np.linalg.solve(A, B)

print("Matrix A:\n", A)
print("Vector B:\n", B)
print("Solution X:\n", X)
```

These examples demonstrate some of the most commonly used linear algebra operations in NumPy. NumPy's linear algebra module provides a comprehensive set of functions for working with matrices, vectors, and linear systems efficiently.


