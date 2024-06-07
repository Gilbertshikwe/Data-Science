# Regression Analysis: Hours Studied vs. Test Scores

This project aims to analyze the relationship between the number of hours studied and the corresponding test scores using linear regression analysis.

## Data

The data consists of two variables:

1. `X`: The independent variable representing the number of hours studied (ranging from 1 to 20).
2. `y`: The dependent variable representing the test scores (ranging from 55 to 98).

## Methodology

The linear regression analysis is performed using the `statsmodels` library in Python. The steps involved are as follows:

1. Import the necessary libraries: `statsmodels.api`, `numpy`, and `matplotlib.pyplot`.
2. Define the data: `X` (hours studied) and `y` (test scores) as NumPy arrays.
3. Add a constant term to `X` for the intercept in the linear regression model.
4. Fit the linear regression model using `sm.OLS(y, X).fit()`.
5. Generate predictions using `model.predict(X)`.
6. Print the summary of the regression model using `print(model.summary())`.
7. Create a scatter plot of the data points and the fitted regression line using `matplotlib.pyplot`.

## Results

The output of the regression analysis provides the following information:

- **Coefficients**:
  - Intercept (const): 53.1789
  - Slope (x1): 2.1211
- **Interpretation**: The estimated linear regression model is `Test Scores = 53.1789 + 2.1211 * Hours Studied`. For every additional hour studied, the test scores are expected to increase by approximately 2.1211 points, on average.
- **R-squared**: 0.988, indicating that the model explains 98.8% of the variance in the test scores.
- **Statistical Significance**: Both the intercept and slope coefficients are statistically significant, suggesting a strong relationship between hours studied and test scores.

The scatter plot visualizes the data points and the fitted regression line, allowing for a visual understanding of the relationship between the variables.

## Assumptions and Limitations

The results of this linear regression analysis are based on the assumptions of a linear relationship between the variables and the validity of the underlying assumptions of linear regression, such as normality of residuals, homoscedasticity (constant variance of errors), and the absence of multicollinearity (if there were multiple independent variables). It is important to consider these assumptions when interpreting the results.

Additionally, the analysis is limited to the provided dataset and may not generalize to other populations or scenarios. Further analysis and validation may be required for broader applicability.

# Probability distributions
# Understanding Normal Distribution with Student Heights Example

**Prerequisites**

Before diving into the code, make sure you have the following Python libraries installed:

- NumPy: A library for numerical computing, including support for large, multi-dimensional arrays and matrices, and a vast collection of mathematical functions.
- Matplotlib: A plotting library for creating static, animated, and interactive visualizations in Python.
- Seaborn: A data visualization library based on Matplotlib, providing a high-level interface for creating attractive and informative statistical graphics.

**Code Explanation**

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate data representing the heights of 1000 students
student_heights = np.random.normal(loc=170, scale=10, size=1000)

# Plot the distribution of student heights
sns.histplot(student_heights, kde=True)
plt.title("Distribution of Student Heights")
plt.xlabel("Height (cm)")
plt.ylabel("Frequency")

# Display the plot
plt.show()
```

1. **Importing Libraries**: We start by importing the necessary libraries: NumPy for numerical computations, Matplotlib for plotting, and Seaborn for data visualization.

2. **Generating Data**: The line `student_heights = np.random.normal(loc=170, scale=10, size=1000)` generates 1000 random heights from a normal distribution. The `loc` parameter specifies the mean (average) height, which is set to 170 cm, and the `scale` parameter sets the standard deviation to 10 cm. These values are chosen based on the assumption that the average height of students in the school is around 170 cm, and the heights are distributed with a standard deviation of 10 cm.

3. **Plotting the Distribution**: The line `sns.histplot(student_heights, kde=True)` creates a histogram plot of the generated student heights using Seaborn's `histplot` function. A histogram is a graphical representation that displays the distribution of data by grouping the values into bins or intervals. The `kde=True` parameter adds a kernel density estimate (KDE) curve, which provides a smooth approximation of the underlying probability density function.

4. **Adding Plot Details**: The lines `plt.title("Distribution of Student Heights")`, `plt.xlabel("Height (cm)")`, and `plt.ylabel("Frequency")` add a descriptive title and labels to the plot for better understanding.

5. **Displaying the Plot**: The line `plt.show()` displays the plot.

**Interpreting the Output**

The resulting plot will show a histogram of the generated student heights, with the bars representing the frequency or count of students falling within each height interval or bin. The KDE curve superimposed on the histogram provides a smooth approximation of the underlying normal distribution of student heights.

By visualizing the distribution of student heights, school administrators or researchers can gain insights into the typical range of heights, identify any potential outliers or deviations from the expected distribution, and make informed decisions related to student health, facilities planning, or other relevant areas.

**Significance of Normal Distribution**

The normal distribution is a commonly encountered probability distribution in various fields, including statistics, physics, and engineering. It is widely used to model real-world phenomena, such as human characteristics like height, which are often influenced by various factors (genetic, environmental, etc.) that collectively result in a normal distribution.

Understanding and visualizing normal distributions can help in analyzing data, making predictions, and drawing meaningful conclusions in a wide range of applications.

# Hypothesis Testing: Comparing Teaching Methods using Student Test Scores

**Prerequisites**

Before running the code, make sure you have the following Python libraries installed:

- NumPy: A library for numerical computing, including support for large, multi-dimensional arrays and matrices, and a vast collection of mathematical functions.
- SciPy: A library for scientific and technical computing in Python, providing many user-friendly and efficient numerical routines, such as routines for numerical integration, interpolation, optimization, linear algebra, and statistics.

**Code Explanation**

```python
from scipy import stats
import numpy as np

# Generate test scores for students taught with Method A
scores_method_a = np.random.normal(loc=75, scale=10, size=100)

# Generate test scores for students taught with Method B
scores_method_b = np.random.normal(loc=80, scale=10, size=100)

# Perform t-test to compare the means of the two groups
t_stat, p_value = stats.ttest_ind(scores_method_a, scores_method_b)
print("T-statistic:", t_stat)
print("P-value:", p_value)

# Set significance level
alpha = 0.05

# Decision based on p-value
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference in test scores between the two teaching methods.")
else:
    print("Fail to reject the null hypothesis: There is no significant difference in test scores between the two teaching methods.")
```

1. **Importing Libraries**: We start by importing the necessary libraries: NumPy for numerical computations and SciPy for statistical functions.

2. **Generating Data**: The lines `scores_method_a = np.random.normal(loc=75, scale=10, size=100)` and `scores_method_b = np.random.normal(loc=80, scale=10, size=100)` generate test scores for 100 students taught with two different teaching methods (Method A and Method B, respectively). The `loc` parameter specifies the mean (average) test score, and the `scale` parameter sets the standard deviation. In this example, we assume that the average test score for students taught with Method A is 75, and for Method B, it's 80, with a standard deviation of 10 for both groups.

3. **Performing the t-test**: The line `t_stat, p_value = stats.ttest_ind(scores_method_a, scores_method_b)` performs an independent two-sample t-test to compare the means of the two groups (Method A and Method B). The `t_stat` variable stores the calculated t-statistic, and `p_value` stores the associated p-value.

4. **Printing Results**: The lines `print("T-statistic:", t_stat)` and `print("P-value:", p_value)` print the calculated t-statistic and p-value for reference.

5. **Setting Significance Level**: The line `alpha = 0.05` sets the significance level (alpha) to 0.05, which is a commonly used value in hypothesis testing. The significance level represents the probability of rejecting the null hypothesis when it is actually true (Type I error).

6. **Decision based on p-value**: The conditional statement `if p_value < alpha: ...` compares the calculated p-value with the significance level to make a decision. If the p-value is less than the significance level, we reject the null hypothesis, which means there is a significant difference in test scores between the two teaching methods. Otherwise, we fail to reject the null hypothesis, indicating no significant difference in test scores between the two methods.

**Interpreting the Output**

By running this code, you'll get the t-statistic, p-value, and a decision based on the hypothesis test. If the p-value is small (less than the significance level), it suggests that the difference in test scores between the two teaching methods is statistically significant, and we reject the null hypothesis. If the p-value is large (greater than the significance level), we fail to reject the null hypothesis, indicating that there is no significant difference in test scores between the two methods.

**Real-life Application**

This analysis can help educators or researchers evaluate the effectiveness of different teaching methods and make informed decisions about which method to adopt or recommend based on the statistical evidence from student test scores.

For example, if the results show a significant difference in test scores between the two methods, with Method B having higher average scores, it may suggest that Method B is more effective in improving student performance. On the other hand, if there is no significant difference, it may indicate that both methods are equally effective, and other factors, such as cost or implementation feasibility, could be considered in selecting the teaching method.

Hypothesis testing is a powerful statistical tool that allows researchers and decision-makers to draw conclusions from data while accounting for the inherent variability and uncertainty in real-world scenarios.

# Understanding Mean, Median, and Standard Deviation with Real-Life Examples

## Introduction

In statistics, mean, median, and standard deviation are essential measures used to describe the central tendency and spread of a dataset. These measures provide valuable insights into the distribution of data and are widely used in various fields, including finance, economics, and scientific research.

## Prerequisites

Before diving into the examples, make sure you have the following Python library installed:

- NumPy: A library for numerical computing, including support for large, multi-dimensional arrays and matrices, and a vast collection of mathematical functions.

## Code Explanation

```python
import numpy as np

data = np.array([5, 10, 15, 20, 25])

# Mean
mean = np.mean(data)
print("Mean:", mean)  # Output: Mean: 15.0

# Median
median = np.median(data)
print("Median:", median)  # Output: Median: 15.0

# Standard Deviation
std_dev = np.std(data)
print("Standard Deviation:", std_dev)  # Output: Standard Deviation: 7.071067811865476
```

1. **Importing NumPy**: We start by importing the NumPy library, which provides efficient numerical operations and data structures.

2. **Creating a Dataset**: The line `data = np.array([5, 10, 15, 20, 25])` creates a NumPy array with the given values, representing a sample dataset.

3. **Calculating the Mean**: The line `mean = np.mean(data)` calculates the mean (average) value of the dataset using NumPy's `mean()` function. The mean is the sum of all values divided by the number of values.

4. **Calculating the Median**: The line `median = np.median(data)` calculates the median value of the dataset using NumPy's `median()` function. The median is the middle value when the data is sorted in ascending or descending order.

5. **Calculating the Standard Deviation**: The line `std_dev = np.std(data)` calculates the standard deviation of the dataset using NumPy's `std()` function. The standard deviation is a measure of how spread out the data is from the mean.

## Real-Life Examples

### Mean
- The mean is commonly used to calculate the average performance of students in a class, the average income of a population, or the average temperature over a period of time.
- For example, if the test scores of five students are 75, 80, 85, 90, and 95, the mean score would be 85, calculated as (75 + 80 + 85 + 90 + 95) / 5.

### Median
- The median is useful when dealing with skewed distributions or datasets with outliers, as it is not influenced by extreme values.
- For example, in a company with 20 employees, if the annual salaries (in thousands of dollars) are: 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 130, 140, 150, the median salary would be 75,000, which provides a more representative measure of the "typical" salary compared to the mean, which would be skewed by the highest and lowest salaries.

### Standard Deviation
- The standard deviation is a measure of the spread or dispersion of a dataset around the mean.
- For example, in a manufacturing process, the standard deviation of product dimensions can indicate how much variation exists in the production process. A smaller standard deviation implies that most products are closer to the target dimensions, while a larger standard deviation suggests greater variability and potential quality control issues.

By understanding these statistical measures and their real-life applications, you can gain insights into the central tendency and spread of data, enabling better decision-making and analysis in various domains.