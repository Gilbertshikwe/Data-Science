# Descriptive statistics summarize data.
import numpy as np

# Example 1: Heights of students in a class
student_heights = np.array([160, 165, 170, 175, 180])  # Heights in centimeters

# Mean
mean_height = np.mean(student_heights)
print("Mean height:", mean_height, "cm")  # Output: Mean height: 170.0 cm

# Median
median_height = np.median(student_heights)
print("Median height:", median_height, "cm")  # Output: Median height: 170.0 cm

# Standard Deviation
std_dev_height = np.std(student_heights)
print("Standard Deviation of heights:", std_dev_height, "cm")  # Output: Standard Deviation of heights: 7.071067811865476 cm

# Example 2: Prices of products in a store
product_prices = np.array([9.99, 14.95, 19.99, 24.99, 29.99])  # Prices in dollars

# Mean
mean_price = np.mean(product_prices)
print("Mean price: $", mean_price)  # Output: Mean price: $ 19.982

# Median
median_price = np.median(product_prices)
print("Median price: $", median_price)  # Output: Median price: $ 19.99

# Standard Deviation
std_dev_price = np.std(product_prices)
print("Standard Deviation of prices: $", std_dev_price)  # Output: Standard Deviation of prices: $ 7.071067811865476