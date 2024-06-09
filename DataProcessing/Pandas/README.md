# Pandas Tutorial

This tutorial covers the basics of using the Pandas library in Python for data manipulation and analysis. It includes instructions for installation, creating and working with Series and DataFrames, and performing common data operations.

## Installation

First, ensure you have Pandas installed. You can install it using pip:

```sh
pip install pandas
```

Additionally, for plotting, you will need `matplotlib`:

```sh
pip install matplotlib
```

## Running the Tutorial Script

Save the provided script to a file named `pandas_tutorial.py`. You can then run the script using Python:

```sh
python pandas_tutorial.py
```

This script demonstrates the following:

1. **Creating a Series**:
    ```python
    s = pd.Series([1, 3, 5, 7, 9])
    ```

2. **Creating a DataFrame**:
    ```python
    data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [24, 27, 22, 32],
        'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']
    }
    df = pd.DataFrame(data)
    ```

3. **Viewing Data**:
    ```python
    print(df.head())
    print(df.tail())
    ```

4. **Data Information**:
    ```python
    print(df.info())
    print(df.describe())
    ```

5. **Selecting Data**:
    ```python
    print(df['Name'])
    print(df[['Name', 'City']])
    print(df.iloc[0])
    print(df.iloc[0:2])
    print(df.loc[0])
    ```

6. **Filtering Data**:
    ```python
    print(df[df['Age'] > 25])
    ```

7. **Modifying Data**:
    ```python
    df['Country'] = 'USA'
    df.loc[0, 'Age'] = 25
    ```

8. **Dropping Data**:
    ```python
    df = df.drop('Country', axis=1)
    df = df.drop(0)
    ```

9. **Handling Missing Data**:
    ```python
    print(df.isnull())
    print(df.isnull().sum())
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df.dropna(inplace=True)
    ```

10. **Grouping Data**:
    ```python
    grouped = df.groupby('City')
    print(grouped['Age'].mean())
    ```

11. **Merging and Joining Data**:
    ```python
    df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value': [1, 2, 3]})
    df2 = pd.DataFrame({'key': ['B', 'C', 'D'], 'value': [4, 5, 6]})
    merged_df = pd.merge(df1, df2, on='key')
    df1 = df1.set_index('key')
    df2 = df2.set_index('key')
    joined_df = df1.join(df2, lsuffix='_left', rsuffix='_right')
    ```

12. **Plotting Data**:
    ```python
    df.plot(kind='bar', x='Name', y='Age')
    plt.show()
    ```

## Summary

This tutorial provides a basic overview of using Pandas for data manipulation and analysis. By following the steps in the script, you will learn how to create and modify Series and DataFrames, select and filter data, handle missing data, group data, merge and join datasets, and create plots.

Feel free to modify the script and experiment with different data and operations to get more comfortable with Pandas.

This `README.md` file provides a summary of the tutorial steps and instructions on how to run the script. Save it in the same directory as your `pandas_tutorial.py` script for easy reference.