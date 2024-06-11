# Data Wrangling with Python

Data wrangling, also known as data munging, is the process of transforming and mapping raw data into a more usable format for analysis. This is a critical step in data science as it ensures that data is clean, organized, and ready for analysis. In Python, several libraries are commonly used for data wrangling, including Pandas, NumPy, and more. This README provides a step-by-step guide to data wrangling using Python, covering all the essential topics.

## Table of Contents

1. [Introduction to Data Wrangling](#introduction-to-data-wrangling)
2. [Setting Up the Environment](#setting-up-the-environment)
3. [Data Collection](#data-collection)
4. [Data Cleaning](#data-cleaning)
   - [Handling Missing Values](#handling-missing-values)
   - [Removing Duplicates](#removing-duplicates)
   - [Correcting Errors](#correcting-errors)
5. [Data Transformation](#data-transformation)
   - [Renaming Columns](#renaming-columns)
   - [Converting Data Types](#converting-data-types)
   - [Creating New Columns](#creating-new-columns)
6. [Data Integration](#data-integration)
   - [Merging DataFrames](#merging-dataframes)
   - [Concatenating DataFrames](#concatenating-dataframes)
7. [Data Reduction](#data-reduction)
   - [Dropping Columns](#dropping-columns)
   - [Filtering Rows](#filtering-rows)
8. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
   - [Descriptive Statistics](#descriptive-statistics)
   - [Visualizing Data](#visualizing-data)
9. [Saving the Cleaned Data](#saving-the-cleaned-data)

## Introduction to Data Wrangling

Data wrangling involves several steps:
1. **Data Collection**: Gathering raw data from various sources.
2. **Data Cleaning**: Removing or fixing errors, handling missing values.
3. **Data Transformation**: Converting data into a suitable format or structure.
4. **Data Integration**: Combining data from multiple sources.
5. **Data Reduction**: Reducing the volume of data without losing important information.

## Setting Up the Environment

First, you'll need to install the necessary libraries if you haven't already:

```bash
pip install pandas numpy
```

Then, import the libraries in your Python script or Jupyter Notebook:

```python
import pandas as pd
import numpy as np
```

## Data Collection

You can collect data from various sources like CSV files, Excel files, databases, or APIs. Here, we'll use a CSV file as an example.

```python
# Reading data from a CSV file
df = pd.read_csv('data.csv')
print(df.head())
```

## Data Cleaning

Data cleaning involves handling missing values, correcting errors, and ensuring consistency.

### Handling Missing Values

```python
# Checking for missing values
print(df.isnull().sum())

# Dropping rows with missing values
df.dropna(inplace=True)

# Filling missing values with a specific value
df.fillna(0, inplace=True)

# Filling missing values with the mean of the column
df.fillna(df.mean(), inplace=True)
```

### Removing Duplicates

```python
# Checking for duplicates
print(df.duplicated().sum())

# Removing duplicates
df.drop_duplicates(inplace=True)
```

### Correcting Errors

```python
# Correcting a specific error (e.g., replacing incorrect values)
df['column_name'].replace('wrong_value', 'correct_value', inplace=True)
```

## Data Transformation

Data transformation involves changing the format or structure of the data.

### Renaming Columns

```python
# Renaming columns
df.rename(columns={'old_name': 'new_name'}, inplace=True)
```

### Converting Data Types

```python
# Converting data types
df['column_name'] = df['column_name'].astype(float)
```

### Creating New Columns

```python
# Creating new columns based on existing data
df['new_column'] = df['column1'] + df['column2']
```

## Data Integration

Combining data from multiple sources can be done using various techniques.

### Merging DataFrames

```python
# Merging two DataFrames on a common column
df1 = pd.read_csv('data1.csv')
df2 = pd.read_csv('data2.csv')
merged_df = pd.merge(df1, df2, on='common_column')
```

### Concatenating DataFrames

```python
# Concatenating DataFrames
concatenated_df = pd.concat([df1, df2], axis=0)  # axis=0 for rows, axis=1 for columns
```

## Data Reduction

Reducing the volume of data can be important for efficiency and performance.

### Dropping Columns

```python
# Dropping unnecessary columns
df.drop(columns=['column_to_drop'], inplace=True)
```

### Filtering Rows

```python
# Filtering rows based on a condition
filtered_df = df[df['column_name'] > threshold]
```

## Exploratory Data Analysis (EDA)

After wrangling the data, it's ready for analysis. EDA helps in understanding the data better.

### Descriptive Statistics

```python
# Getting summary statistics
print(df.describe())
```

### Visualizing Data

```python
import matplotlib.pyplot as plt

# Plotting histograms
df['column_name'].hist()
plt.show()

# Plotting scatter plots
plt.scatter(df['column1'], df['column2'])
plt.show()
```

## Saving the Cleaned Data

After cleaning and transforming the data, you can save it for future use.

```python
# Saving the DataFrame to a new CSV file
df.to_csv('cleaned_data.csv', index=False)
```

## Summary

Data wrangling is an essential step in the data science workflow. It ensures that data is clean, structured, and ready for analysis. Using Python and libraries like Pandas and NumPy, you can efficiently handle tasks like data collection, cleaning, transformation, integration, and reduction. This foundational step sets the stage for accurate and meaningful data analysis.

By practicing these steps with different datasets, you'll become proficient in data wrangling and be better prepared for more advanced data science tasks.