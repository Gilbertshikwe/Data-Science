import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt

# Creating example data
data = {
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'Satisfaction': ['Satisfied', 'Satisfied', 'Neutral', 'Dissatisfied', 'Dissatisfied', 'Neutral', 'Satisfied', 'Dissatisfied']
}

df = pd.DataFrame(data)

# Creating a contingency table
contingency_table = pd.crosstab(df['Gender'], df['Satisfaction'])
print("Contingency Table:")
print(contingency_table)

# Performing the Chi-Square Test
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi-Square Statistic: {chi2}")
print(f"P-Value: {p}")
print(f"Degrees of Freedom: {dof}")
print(f"Expected Frequencies: \n{expected}")

# Interpreting the results
if p < 0.05:
    print("\nThere is a significant association between gender and satisfaction level.")
else:
    print("\nThere is no significant association between gender and satisfaction level.")

# Visualizing the data
sns.countplot(data=df, x='Satisfaction', hue='Gender')
plt.title('Customer Satisfaction by Gender')
plt.xlabel('Satisfaction Level')
plt.ylabel('Count')
plt.savefig('satisfaction_by_gender.png')
plt.close()

