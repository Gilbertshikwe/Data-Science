import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #type:ignore

# Function to print a section header
def print_header(title):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title) + "\n")

# 1. Data Collection
print_header("1. Data Collection")
# Create sample data
data = {
    'Name': ['John', 'Anna', 'Peter', 'Linda', 'John'],
    'Age': [28, 22, 35, 32, np.nan],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
    'Salary': [50000, 54000, np.nan, 58000, 52000]
}
df = pd.DataFrame(data)
print("Original DataFrame:\n", df)

# 2. Data Cleaning
print_header("2. Data Cleaning")
# Handling missing values
print("Handling Missing Values:")
print("Missing values:\n", df.isnull().sum())
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Salary'].fillna(df['Salary'].mean(), inplace=True)
print("DataFrame after filling missing values:\n", df)

# Removing duplicates
print("\nRemoving Duplicates:")
print("Number of duplicates:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("DataFrame after removing duplicates:\n", df)

# Correcting errors
print("\nCorrecting Errors:")
df['Name'].replace('John', 'Jonathan', inplace=True)
print("DataFrame after correcting errors:\n", df)

# 3. Data Transformation
print_header("3. Data Transformation")
# Renaming columns
print("Renaming Columns:")
df.rename(columns={'Salary': 'Annual Salary'}, inplace=True)
print("DataFrame after renaming columns:\n", df)

# Converting data types
print("\nConverting Data Types:")
df['Annual Salary'] = df['Annual Salary'].astype(int)
print("DataFrame after converting data types:\n", df)

# Creating new columns
print("\nCreating New Columns:")
df['Monthly Salary'] = df['Annual Salary'] / 12
print("DataFrame after creating new columns:\n", df)

# 4. Data Integration
print_header("4. Data Integration")
# Merging DataFrames (creating another sample dataframe)
data2 = {
    'Name': ['Jonathan', 'Anna', 'Peter', 'Linda'],
    'Department': ['HR', 'Finance', 'IT', 'Marketing']
}
df2 = pd.DataFrame(data2)
print("Another DataFrame for merging:\n", df2)
merged_df = pd.merge(df, df2, on='Name')
print("Merged DataFrame:\n", merged_df)

# Concatenating DataFrames
print("\nConcatenating DataFrames:")
concatenated_df = pd.concat([df, df2], axis=0)
print("Concatenated DataFrame:\n", concatenated_df)

# 5. Data Reduction
print_header("5. Data Reduction")
# Dropping columns
print("Dropping Columns:")
df.drop(columns=['Monthly Salary'], inplace=True)
print("DataFrame after dropping a column:\n", df)

# Filtering rows
print("\nFiltering Rows:")
filtered_df = df[df['Age'] > 30]
print("Filtered DataFrame (Age > 30):\n", filtered_df)

# 6. Exploratory Data Analysis (EDA)
print_header("6. Exploratory Data Analysis (EDA)")
# Descriptive statistics
print("Descriptive Statistics:")
print(df.describe())

# Visualizing data
print("\nVisualizing Data:")
df['Age'].hist()
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

plt.scatter(df['Age'], df['Annual Salary'])
plt.title('Age vs. Annual Salary')
plt.xlabel('Age')
plt.ylabel('Annual Salary')
plt.show()

# 7. Saving the Cleaned Data
print_header("7. Saving the Cleaned Data")
# Saving to a CSV file
df.to_csv('cleaned_data.csv', index=False)
print("Cleaned data saved to 'cleaned_data.csv'")
