# Data Cleaning with Python and Pandas

This tutorial demonstrates how to perform data cleaning using Python and the Pandas library. Data cleaning is essential in data analysis and preprocessing, where you identify and correct (or remove) errors and inconsistencies in data to improve its quality.

## Prerequisites

Ensure you have Pandas installed. You can install it using pip:

```sh
pip install pandas
```

### Data Cleaning Steps

Data cleaning is a crucial step in the data preprocessing pipeline, ensuring that the data is accurate, consistent, and ready for analysis. Here's a breakdown of the key data cleaning steps and why they are performed:

1. **Importing Libraries:**
   - Reason: Import necessary libraries such as Pandas and NumPy to facilitate data manipulation and analysis.

2. **Loading the Data:**
   - Reason: Load the raw data into a DataFrame to start the cleaning process.

3. **Handling Missing Values:**
   - Reason: Missing values can affect the accuracy of statistical analysis and machine learning models. Handling them by filling in missing values or removing rows/columns with missing data ensures the integrity of the dataset.

4. **Removing Duplicates:**
   - Reason: Duplicate records may skew analysis results and introduce bias. Removing duplicates ensures that each observation is unique, preventing overcounting or redundant information.

5. **Correcting Data Types:**
   - Reason: Ensuring that data types are correct is essential for accurate analysis. Converting data to the appropriate types (e.g., numeric, categorical) improves computational efficiency and prevents errors in calculations.

6. **Handling Outliers:**
   - Reason: Outliers can significantly impact statistical analysis and model performance. Handling outliers by capping/extending or removing them helps prevent them from disproportionately influencing results.

7. **Renaming Columns:**
   - Reason: Renaming columns to more descriptive or standardized names improves readability and clarity of the dataset, making it easier to understand and work with.

8. **Dropping Unnecessary Columns:**
   - Reason: Columns that are irrelevant or redundant for the analysis should be removed to streamline the dataset and reduce computational overhead.

9. **Data Standardization:**
   - Reason: Standardizing data ensures that different scales or units across variables do not bias analysis results. By scaling data to a common range or distribution, comparisons between variables become more meaningful and accurate.

By following these data cleaning steps, we ensure that the dataset is prepared for further analysis, leading to more reliable insights and decisions.

## Example Script

Here is a complete example script demonstrating various data cleaning tasks:

```python
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
df = pd.concat([df, df.iloc[[1]]], ignore_index=True)
print("DataFrame with duplicate row:")
print(df)
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
```

## Explanation

1. **Creating the DataFrame**: A sample DataFrame is created with some initial data.
2. **Handling Missing Values**: Missing values are identified and filled with appropriate values.
3. **Removing Duplicates**: Duplicate rows are identified and removed.
4. **Correcting Data Types**: The 'Income' column is converted to a numeric type.
5. **Handling Outliers**: Outliers in the 'Age' column are detected and handled using the Interquartile Range (IQR) method.
6. **Renaming Columns**: Columns are renamed for better readability.
7. **Dropping Unnecessary Columns**: Unnecessary columns are dropped from the DataFrame.
8. **Data Standardization**: The 'Annual Income' column is standardized.
9. **Writing to CSV**: The cleaned DataFrame is saved to a CSV file named `cleaned_data.csv`.
10. **Reading from CSV**: The CSV file is read back into a new DataFrame named `new_df`.

This script demonstrates common data cleaning tasks and provides a template you can adjust to fit your specific dataset and requirements.

# Creating and Reading CSV Files with Pandas

This example demonstrates how to create a CSV file from a Pandas DataFrame and then read the data from the CSV file back into a new DataFrame.

## Creating a Sample DataFrame

```python
import pandas as pd

# Create a sample DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 32, 28],
        'City': ['New York', 'Los Angeles', 'Chicago']}
df = pd.DataFrame(data)
```

## Writing the DataFrame to a CSV File

```python
# Write the DataFrame to a CSV file
df.to_csv('cleaned_data.csv', index=False)
```

This will create a new CSV file named `sample_data.csv` in the same directory as your Python script, containing the data from the `df` DataFrame.

## Reading Data from the CSV File

```python
# Read the data from the CSV file into a new DataFrame
new_df = pd.read_csv('cleaned_data.csv')

# Print the new DataFrame
print(new_df)
```

This will read the data from the `sample_data.csv` file and create a new DataFrame `new_df` with the same data as the original `df` DataFrame.

Make sure you have the `pandas` library installed before running this example.

This README provides a simple example of how to create a DataFrame, write it to a CSV file using `df.to_csv('sample_data.csv', index=False)`, and then read the data from the CSV file back into a new DataFrame using `pd.read_csv('sample_data.csv')`. It also includes some brief explanations and instructions for running the example.
---

This README provides an overview of the data cleaning process using Python and Pandas, along with a complete example script. Feel free to adjust the script and explanations to better fit your project's needs.