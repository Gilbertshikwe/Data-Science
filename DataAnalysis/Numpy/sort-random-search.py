import numpy as np

# Seed for reproducibility
np.random.seed(42)

# Generate random data
data = np.random.randint(1, 100, size=10)
print("Original Data:", data)

# Sort the data
sorted_data = np.sort(data)
print("Sorted Data:", sorted_data)

# Get the indices that would sort the data
sorted_indices = np.argsort(data)
print("Sorted Indices:", sorted_indices)

# Find elements greater than 50
indices_gt_50 = np.where(data > 50)
print("Indices of elements greater than 50:", indices_gt_50)

# Insert a value to maintain order
new_data = np.insert(sorted_data, np.searchsorted(sorted_data, 55), 55)
print("Data after inserting 55:", new_data)

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