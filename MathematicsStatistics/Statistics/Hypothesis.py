# Hypothesis testing is a method of making decisions using data.

from scipy import stats #type:ignore
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







