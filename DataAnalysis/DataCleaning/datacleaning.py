import pandas as pd
import numpy as np

# Create a sample DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', None],
    'Age': [24, 27, np.nan, 32, 28, 22],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', None, 'Phoenix'],
    'Income': ['50000', '60000', '55000', '70000', '80000', '85000']
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)
print()

# Handling Missing Values
print("Missing values per column:")
print(df.isnull().sum())
print()

df['Name'].fillna('Unknown', inplace=True)
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['City'].fillna('Unknown', inplace=True)
print("DataFrame after filling missing values:")
print(df)
print()

# Removing Duplicates
# Adding a duplicate row for demonstration
df_with_duplicate = pd.concat([df, df.iloc[[1]]], ignore_index=True)
print("DataFrame with duplicate row:")
print(df_with_duplicate)
print()

df.drop_duplicates(inplace=True)
print("DataFrame after removing duplicates:")
print(df)
print()

# Correcting Data Types
print("Data types before correction:")
print(df.dtypes)
print()

df['Income'] = pd.to_numeric(df['Income'])
print("Data types after correction:")
print(df.dtypes)
print()

# Handling Outliers
Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1
print("Interquartile Range (IQR) for Age:", IQR)
print()

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Age'] < lower_bound) | (df['Age'] > upper_bound)]
print("Outliers in Age:")
print(outliers)
print()

df['Age'] = np.where(df['Age'] < lower_bound, lower_bound, df['Age'])
df['Age'] = np.where(df['Age'] > upper_bound, upper_bound, df['Age'])
print("DataFrame after handling outliers:")
print(df)
print()

# Renaming Columns
df.rename(columns={'Name': 'Full Name', 'Income': 'Annual Income'}, inplace=True)
print("DataFrame after renaming columns:")
print(df)
print()

# Dropping Unnecessary Columns
df.drop(columns=['City'], inplace=True)
print("DataFrame after dropping 'City' column:")
print(df)
print()

# Data Standardization
df['Annual Income'] = (df['Annual Income'] - df['Annual Income'].mean()) / df['Annual Income'].std()
print("DataFrame after standardizing 'Annual Income':")
print(df)
print()

# Save DataFrame to a CSV file
df.to_csv('cleaned_data.csv', index=False)

# Read DataFrame from the CSV file
new_df = pd.read_csv('cleaned_data.csv')

print("DataFrame read from CSV file:")
print(new_df)

