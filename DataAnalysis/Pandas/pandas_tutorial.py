import pandas as pd
import matplotlib.pyplot as plt #type:ignore

# Creating a Series
s = pd.Series([1, 3, 5, 7, 9])
print("Series:")
print(s)
print()

# Creating a DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [24, 27, 22, 32],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']
}
df = pd.DataFrame(data)
print("DataFrame:")
print(df)
print()

# Viewing Data
print("First few rows of the DataFrame:")
print(df.head())
print()

print("Last few rows of the DataFrame:")
print(df.tail())
print()

# Data Information
print("DataFrame Info:")
print(df.info())
print()

print("Summary Statistics:")
print(df.describe())
print()

# Selecting Data
print("Select a single column (Name):")
print(df['Name'])
print()

print("Select multiple columns (Name, City):")
print(df[['Name', 'City']])
print()

print("Select first row by index:")
print(df.iloc[0])  # First row
print()

print("Select first two rows by index:")
print(df.iloc[0:2])  # First two rows
print()

print("Select first row by label:")
print(df.loc[0])  # First row
print()

# Filtering Data
print("Filter rows where Age > 25:")
print(df[df['Age'] > 25])
print()

# Modifying Data
df['Country'] = 'USA'
print("DataFrame after adding Country column:")
print(df)
print()

df.loc[0, 'Age'] = 25
print("DataFrame after updating Age for the first row:")
print(df)
print()

# Dropping Data
df = df.drop('Country', axis=1)
print("DataFrame after dropping Country column:")
print(df)
print()

df = df.drop(0)  # Drop the first row
print("DataFrame after dropping the first row:")
print(df)
print()

# Handling Missing Data
print("Check for missing data:")
print(df.isnull())
print()

print("Sum of missing data per column:")
print(df.isnull().sum())
print()

# Adding missing value for demonstration
df.loc[1, 'Age'] = None
print("DataFrame with a missing value:")
print(df)
print()

df['Age'].fillna(df['Age'].mean(), inplace=True)
print("DataFrame after filling missing values:")
print(df)
print()

df.dropna(inplace=True)
print("DataFrame after dropping rows with missing values:")
print(df)
print()

# Grouping Data
grouped = df.groupby('City')
print("Grouped data by City (mean of numeric columns):")
print(grouped['Age'].mean())  # Select only numeric column
print()

# Merging and Joining Data
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['B', 'C', 'D'], 'value': [4, 5, 6]})

merged_df = pd.merge(df1, df2, on='key')
print("Merged DataFrame:")
print(merged_df)
print()

df1 = df1.set_index('key')
df2 = df2.set_index('key')

joined_df = df1.join(df2, lsuffix='_left', rsuffix='_right')
print("Joined DataFrame:")
print(joined_df)
print()

# Plotting Data
df.plot(kind='bar', x='Name', y='Age')
plt.show()


