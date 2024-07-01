import numpy as np
from scipy.stats import ttest_ind, ttest_rel

# Example 1: Medical Treatment Effectiveness
print("Example 1: Medical Treatment Effectiveness")
np.random.seed(0)
control_group = np.random.normal(loc=50, scale=10, size=100)
treatment_group = np.random.normal(loc=55, scale=10, size=100)

t_statistic, p_value = ttest_ind(control_group, treatment_group)

print("Null Hypothesis (H₀): The drug has no effect.")
print("Alternative Hypothesis (H₁): The drug has a significant effect.")
print(f"p-value: {p_value}")

alpha = 0.05
if p_value < alpha:
    print("Conclusion: Reject the null hypothesis. The drug has a significant effect.")
else:
    print("Conclusion: Fail to reject the null hypothesis. The drug may not have a significant effect.")

# Example 2: Marketing Campaign Success
print("\nExample 2: Marketing Campaign Success")
np.random.seed(1)
sales_before = np.random.normal(loc=1000, scale=100, size=50)
sales_after = np.random.normal(loc=1100, scale=100, size=50)

t_statistic, p_value = ttest_rel(sales_before, sales_after)

print("Null Hypothesis (H₀): The campaign has no impact on sales.")
print("Alternative Hypothesis (H₁): The campaign significantly increases sales.")
print(f"p-value: {p_value}")

if p_value < alpha:
    print("Conclusion: Reject the null hypothesis. The campaign significantly increases sales.")
else:
    print("Conclusion: Fail to reject the null hypothesis. The campaign may not have a significant impact on sales.")

# Example 3: Educational Intervention
print("\nExample 3: Educational Intervention")
np.random.seed(2)
control_scores = np.random.normal(loc=70, scale=10, size=100)
treatment_scores = np.random.normal(loc=75, scale=10, size=100)

t_statistic, p_value = ttest_ind(control_scores, treatment_scores)

print("Null Hypothesis (H₀): The intervention has no effect on test scores.")
print("Alternative Hypothesis (H₁): The intervention improves test scores significantly.")
print(f"p-value: {p_value}")

if p_value < alpha:
    print("Conclusion: Reject the null hypothesis. The intervention improves test scores significantly.")
else:
    print("Conclusion: Fail to reject the null hypothesis. The intervention may not have a significant effect on test scores.")

# Example 4: Website Redesign Impact
print("\nExample 4: Website Redesign Impact")
np.random.seed(3)
engagement_before = np.random.normal(loc=50, scale=5, size=30)
engagement_after = np.random.normal(loc=55, scale=5, size=30)

t_statistic, p_value = ttest_rel(engagement_before, engagement_after)

print("Null Hypothesis (H₀): The website redesign has no impact on user engagement.")
print("Alternative Hypothesis (H₁): The website redesign improves user engagement significantly.")
print(f"p-value: {p_value}")

if p_value < alpha:
    print("Conclusion: Reject the null hypothesis. The website redesign improves user engagement significantly.")
else:
    print("Conclusion: Fail to reject the null hypothesis. The website redesign may not have a significant impact on user engagement.")