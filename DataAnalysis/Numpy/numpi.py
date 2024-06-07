import numpy as np

# Simulating some data
data = np.random.randn(1000)  # 1000 random numbers from a normal distribution

# Basic statistics
mean = np.mean(data)
std_dev = np.std(data)

# Histogram of the data
hist, bins = np.histogram(data, bins=30)

# Reshape data for matrix operations
reshaped_data = data.reshape(100, 10)

# Calculate covariance matrix
cov_matrix = np.cov(reshaped_data, rowvar=False)

print("Mean:", mean)
print("Standard Deviation:", std_dev)
print("Covariance Matrix:\n", cov_matrix)
