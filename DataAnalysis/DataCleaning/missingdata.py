import pandas as pd
import numpy as np
import seaborn as sns #type:ignore
import matplotlib.pyplot as plt  #type:ignore


# Generate a more realistic but simpler DataFrame
data = {
    'Date': pd.date_range(start='2023-01-01', periods=7, freq='D'),
    'Sales': [200, np.nan, 150, 300, np.nan, 500, 400],
    'Customers': [20, 25, np.nan, 30, 28, np.nan, 35],
    'Category': ['Electronics', 'Furniture', 'Electronics', np.nan, 'Furniture', 'Electronics', np.nan]
}
df = pd.DataFrame(data)

# 1. Detecting Missing Data

# Detecting missing values
print("Detecting missing values:")
print(df.isnull())

# Detecting non-missing values
print("\nDetecting non-missing values:")
print(df.notnull())

# 2. Handling Missing Data

# Dropping Missing Values

# Drop rows with any missing values
df_dropped_rows = df.dropna()
print("\nDrop rows with any missing values:")
print(df_dropped_rows)

# Drop columns with any missing values
df_dropped_cols = df.dropna(axis=1)
print("\nDrop columns with any missing values:")
print(df_dropped_cols)

# Drop rows where all elements are missing
df_dropped_all = df.dropna(how='all')
print("\nDrop rows where all elements are missing:")
print(df_dropped_all)

# Drop rows where less than a certain number of non-NA values are present
df_dropped_thresh = df.dropna(thresh=3)
print("\nDrop rows where less than a certain number of non-NA values are present:")
print(df_dropped_thresh)

# Filling Missing Values

# Fill missing values with a specified value
df_filled_value = df.fillna({
    'Sales': 0,
    'Customers': df['Customers'].mean(),
    'Category': 'Unknown'
})
print("\nFill missing values with specified values:")
print(df_filled_value)

# Fill missing values with the mean of the column
df_filled_mean = df.fillna(df.mean(numeric_only=True))
print("\nFill missing values with the mean of the column:")
print(df_filled_mean)

# Fill missing values using forward fill (propagate last valid observation forward)
df_filled_ffill = df.fillna(method='ffill')
print("\nFill missing values using forward fill:")
print(df_filled_ffill)

# Fill missing values using backward fill (propagate next valid observation backward)
df_filled_bfill = df.fillna(method='bfill')
print("\nFill missing values using backward fill:")
print(df_filled_bfill)

# Interpolating Missing Values

# Interpolate missing values
df_interpolated = df.interpolate()
print("\nInterpolate missing values:")
print(df_interpolated)

# 3. Analyzing Missing Data Patterns

# Checking the number of missing values per column
missing_per_column = df.isnull().sum()
print("\nNumber of missing values per column:")
print(missing_per_column)

# Checking the percentage of missing values per column
missing_percentage = df.isnull().mean() * 100
print("\nPercentage of missing values per column:")
print(missing_percentage)

# Visualizing missing data pattern using a heatmap
print("\nVisualizing missing data pattern using a heatmap:")
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.show()
