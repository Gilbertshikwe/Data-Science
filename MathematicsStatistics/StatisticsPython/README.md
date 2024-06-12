# Central Tendency Measures in Statistics

Central tendency is a statistical measure that identifies a single value as representative of an entire dataset. The most common measures of central tendency are the mean, median, and mode. This README covers different types of mean and central tendency calculations for both ungrouped and grouped data.

## Types of Mean in Python

The mean is a measure of central tendency that represents the central or typical value of a dataset. There are different types of mean, each with its own formula and characteristics. Here, we will explore three types of mean: arithmetic mean, geometric mean, and harmonic mean, along with their Python implementations.

### 1. Arithmetic Mean

The arithmetic mean (or simply mean) is the sum of all values in a dataset divided by the number of values.

**Formula**:

```
Arithmetic Mean = (Σ x_i) / n
```

Where `Σ x_i` is the sum of all values, and `n` is the number of values.

**Python Code**:

```python
import numpy as np

data = [2, 4, 6, 8, 10]
arithmetic_mean = np.mean(data)
print(f"Arithmetic Mean: {arithmetic_mean}")
```

**Explanation**: For the given dataset `[2, 4, 6, 8, 10]`, the arithmetic mean is calculated as follows:
1. Calculate the sum of all values: `2 + 4 + 6 + 8 + 10 = 30`
2. Count the number of values: `n = 5`
3. Divide the sum by the number of values: `Arithmetic Mean = 30 / 5 = 6`

In the Python code, we use the `np.mean()` function from NumPy to calculate the arithmetic mean of the provided dataset.

### 2. Geometric Mean

The geometric mean is the nth root of the product of all values, where `n` is the number of values. It is often used when dealing with percentages, ratios, or growth rates.

**Formula**:

```
Geometric Mean = (∏ x_i)^(1/n)
```

Where `∏ x_i` is the product of all values, and `n` is the number of values.

**Python Code**:

```python
from scipy.stats import gmean

data = [2, 4, 6, 8, 10]
geometric_mean = gmean(data)
print(f"Geometric Mean: {geometric_mean}")
```

**Explanation**: For the given dataset `[2, 4, 6, 8, 10]`, the geometric mean is calculated as follows:
1. Calculate the product of all values: `2 × 4 × 6 × 8 × 10 = 3840`
2. Count the number of values: `n = 5`
3. Take the nth root of the product: `Geometric Mean = (3840)^(1/5) ≈ 5.52`

In the Python code, we use the `gmean()` function from the `scipy.stats` module to calculate the geometric mean of the provided dataset.

### 3. Harmonic Mean

The harmonic mean is the reciprocal of the arithmetic mean of the reciprocals of the values. It is often used when dealing with rates or ratios, and it is particularly useful when averaging quantities whose reciprocals are more meaningful than the quantities themselves (e.g., speeds).

**Formula**:

```
Harmonic Mean = n / (Σ (1/x_i))
```

Where `n` is the number of values, and `x_i` are the values.

**Python Code**:

```python
from scipy.stats import hmean

data = [2, 4, 6, 8, 10]
harmonic_mean = hmean(data)
print(f"Harmonic Mean: {harmonic_mean}")
```

**Explanation**: For the given dataset `[2, 4, 6, 8, 10]`, the harmonic mean is calculated as follows:
1. Calculate the reciprocals of the values: `1/2, 1/4, 1/6, 1/8, 1/10`
2. Calculate the arithmetic mean of the reciprocals: `(1/2 + 1/4 + 1/6 + 1/8 + 1/10) / 5 = 0.3`
3. Take the reciprocal of the arithmetic mean of the reciprocals: `Harmonic Mean = 1 / 0.3 ≈ 3.33`

In the Python code, we use the `hmean()` function from the `scipy.stats` module to calculate the harmonic mean of the provided dataset.

These different types of mean are useful in various situations, depending on the characteristics of the data and the specific problem being addressed. The arithmetic mean is the most commonly used, while the geometric and harmonic means are often employed in specific scenarios where they provide more meaningful or appropriate representations of the central tendency.

## Ungrouped Data

Ungrouped data is raw data that has not been organized into groups. Measures of central tendency for ungrouped data can be directly calculated using the data values.

**Example for Mean, Median, and Mode**:

```python
import numpy as np
from scipy import stats

data = [2, 4, 6, 8, 10, 6]
mean = np.mean(data)
median = np.median(data)
mode = stats.mode(data).mode[0]

print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Mode: {mode}")
```

## Grouped Data

Grouped data is data that has been organized into frequency distribution tables. Measures of central tendency for grouped data are calculated using the class intervals and their frequencies.

**Example for Grouped Data**:

Suppose we have the following frequency distribution table:

| Class Interval | Frequency |
| -------------- | --------- |
| 0-10           | 2         |
| 10-20          | 3         |
| 20-30          | 5         |
| 30-40          | 4         |
| 40-50          | 1         |

We can calculate the mean, median, and mode for grouped data.

**Mean (Grouped Data)**

**Formula**:

```
Mean = (∑ (f_i · x_i)) / (∑ f_i)
```

Where `f_i` is the frequency of the class interval, and `x_i` is the midpoint of the class interval.

**Median (Grouped Data)**

**Formula**:

```
Median = L + ((N/2 - CF) / f) × h
```

Where:
- `L` is the lower boundary of the median class
- `N` is the total number of observations
- `CF` is the cumulative frequency of the class before the median class
- `f` is the frequency of the median class
- `h` is the class interval size

**Mode (Grouped Data)**

**Formula**:

```
Mode = L + ((f_1 - f_0) / (2f_1 - f_0 - f_2)) × h
```

Where:
- `L` is the lower boundary of the modal class
- `f_1` is the frequency of the modal class
- `f_0` is the frequency of the class before the modal class
- `f_2` is the frequency of the class after the modal class
- `h` is the class interval size

```python
# Example data
data = [2, 4, 6, 8, 10, 6]
mean, median, mode = ungrouped_data_central_tendency(data)
print(f"Ungrouped Data - Mean: {mean}, Median: {median}, Mode: {mode}")

class_intervals = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
frequencies = [2, 3, 5, 4, 1]
mean_grouped, median_grouped, mode_grouped = grouped_data_central_tendency(class_intervals, frequencies)
print(f"Grouped Data - Mean: {mean_grouped}, Median: {median_grouped}, Mode: {mode_grouped}")
```

The functions `ungrouped_data_central_tendency` and `grouped_data_central_tendency` are not provided in the given code, but they should implement the formulas and calculations for the respective types of data.
# Measures of Variability in Statistics

Measures of variability (or dispersion) in statistics quantify the extent to which data points in a dataset differ from the central tendency (mean, median, or mode). They provide insights into the spread, distribution, and consistency of the data. This README explores common measures of variability and shows how to calculate them using Python.

## 1. Range

The range is the simplest measure of variability and is the difference between the maximum and minimum values in a dataset.

**Formula:**
```
Range = Maximum Value - Minimum Value
```

**Python Code:**

```python
import numpy as np

data = [2, 4, 6, 8, 10]
range_value = np.ptp(data)
print(f"Range: {range_value}")
```

**Explanation:**
For the given dataset `[2, 4, 6, 8, 10]`, the range is calculated as follows:
1. Find the maximum value: `max(data) = 10`
2. Find the minimum value: `min(data) = 2`
3. Calculate the range: `Range = max(data) - min(data) = 10 - 2 = 8`

In the Python code, we use the `np.ptp()` function from NumPy, which calculates the range (peak-to-peak) of the provided dataset. It internally finds the maximum and minimum values and returns their difference.

## 2. Interquartile Range (IQR)

The interquartile range (IQR) measures the range within which the central 50% of values fall. It is the difference between the third quartile (Q3) and the first quartile (Q1).

**Formula:**
```
IQR = Q3 - Q1
```

**Python Code:**

```python
import numpy as np

data = [2, 4, 6, 8, 10]
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
iqr = q3 - q1
print(f"Interquartile Range (IQR): {iqr}")
```

**Explanation:**
For the given dataset `[2, 4, 6, 8, 10]`, the IQR is calculated as follows:
1. Sort the data in ascending order: `[2, 4, 6, 8, 10]`
2. Find the first quartile (Q1), which is the 25th percentile: `Q1 = np.percentile(data, 25) = 4`
3. Find the third quartile (Q3), which is the 75th percentile: `Q3 = np.percentile(data, 75) = 8`
4. Calculate the IQR: `IQR = Q3 - Q1 = 8 - 4 = 4`

In the Python code, we use the `np.percentile()` function from NumPy to calculate the 25th and 75th percentiles, which correspond to the first quartile (Q1) and third quartile (Q3), respectively. Then, we calculate the IQR by subtracting Q1 from Q3.

## 3. Variance

Variance measures the average squared deviation of each data point from the mean. It provides a measure of the spread of the data points. The variance is calculated differently for populations and samples.

**Formula for Population Variance:**
```
Variance(σ²) = Σ(x_i - μ)² / N
```

**Formula for Sample Variance:**
```
Variance(s²) = Σ(x_i - x̄)² / (n - 1)
```

where `x_i` are the data points, `μ` is the population mean, `x̄` is the sample mean, `N` is the population size, and `n` is the sample size.

The sample variance is an unbiased estimator of the population variance, and it is commonly used when working with samples.

**Python Code:**

```python
import numpy as np

data = [2, 4, 6, 8, 10]
variance = np.var(data, ddof=1)  # ddof=1 for sample variance
print(f"Variance: {variance}")
```

**Explanation:**
For the given dataset `[2, 4, 6, 8, 10]`, the sample variance is calculated as follows:
1. Calculate the mean: `mean = (2 + 4 + 6 + 8 + 10) / 5 = 6`
2. Calculate the deviations of each data point from the mean: `[2 - 6, 4 - 6, 6 - 6, 8 - 6, 10 - 6] = [-4, -2, 0, 2, 4]`
3. Square each deviation: `[-4², -2², 0², 2², 4²] = [16, 4, 0, 4, 16]`
4. Sum the squared deviations: `16 + 4 + 0 + 4 + 16 = 40`
5. Divide the sum of squared deviations by (n - 1), where n is the sample size: `Variance = 40 / (5 - 1) = 40 / 4 = 10`

In the Python code, we use the `np.var()` function from NumPy to calculate the sample variance. The `ddof=1` parameter specifies that we want to calculate the sample variance by dividing the sum of squared deviations by `(n - 1)`.

## 4. Standard Deviation

The standard deviation is the square root of the variance and provides a measure of the average deviation from the mean. It is expressed in the same units as the data points, making it easier to interpret than the variance.

**Formula for Population Standard Deviation:**
```
Standard Deviation(σ) = √(Σ(x_i - μ)² / N)
```

**Formula for Sample Standard Deviation:**
```
Standard Deviation(s) = √(Σ(x_i - x̄)² / (n - 1))
```

**Python Code:**

```python
import numpy as np

data = [2, 4, 6, 8, 10]
std_deviation = np.std(data, ddof=1)  # ddof=1 for sample standard deviation
print(f"Standard Deviation: {std_deviation}")
```

**Explanation:**
For the given dataset `[2, 4, 6, 8, 10]`, the sample standard deviation is calculated as follows:
1. Calculate the variance using the steps mentioned earlier: `Variance = 10`
2. Take the square root of the variance: `Standard Deviation = √10 ≈ 3.16`

In the Python code, we use the `np.std()` function from NumPy to calculate the sample standard deviation. The `ddof=1` parameter specifies that we want to calculate the sample standard deviation by taking the square root of the sample variance.

## Summary

Measures of variability are crucial for understanding the spread and distribution of data. They complement measures of central tendency by providing insights into the data's consistency and spread. The range and IQR provide simple measures of the total spread and the spread of the central values, respectively. The variance and standard deviation quantify the average deviation from the mean, with the standard deviation being more interpretable due to its use of the same units as the data.

Using Python and libraries like NumPy, these calculations are straightforward and efficient. By combining measures of central tendency and variability, you can gain a comprehensive understanding of your dataset and make informed decisions based on its characteristics.
```


