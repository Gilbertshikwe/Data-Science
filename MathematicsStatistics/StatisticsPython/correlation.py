import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# 1. Creating example data
n = 50
study_hours = np.random.uniform(1, 5, n)  # Hours spent studying per day
exam_scores = 60 + 8 * study_hours + np.random.normal(0, 5, n)  # Exam scores (0-100)
social_media_hours = 4 - 0.5 * study_hours + np.random.normal(0, 0.5, n)  # Hours spent on social media
coffee_consumption = np.random.uniform(0, 5, n)  # Cups of coffee per day

# 2. Calculating Pearson correlation
corr_study_exam = stats.pearsonr(study_hours, exam_scores)
corr_study_social = stats.pearsonr(study_hours, social_media_hours)
corr_coffee_exam = stats.pearsonr(coffee_consumption, exam_scores)

print("Correlation between study hours and exam scores:", corr_study_exam)
print("Correlation between study hours and social media usage:", corr_study_social)
print("Correlation between coffee consumption and exam scores:", corr_coffee_exam)

# 3. Visualizing correlations
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.scatter(study_hours, exam_scores)
ax1.set_title(f'Study Hours vs Exam Scores (r={corr_study_exam[0]:.2f})')
ax1.set_xlabel('Study Hours')
ax1.set_ylabel('Exam Scores')

ax2.scatter(study_hours, social_media_hours)
ax2.set_title(f'Study Hours vs Social Media Usage (r={corr_study_social[0]:.2f})')
ax2.set_xlabel('Study Hours')
ax2.set_ylabel('Social Media Hours')

ax3.scatter(coffee_consumption, exam_scores)
ax3.set_title(f'Coffee Consumption vs Exam Scores (r={corr_coffee_exam[0]:.2f})')
ax3.set_xlabel('Coffee Cups per Day')
ax3.set_ylabel('Exam Scores')

plt.tight_layout()
plt.savefig('correlation_plots.png')
plt.close()

# 4. Using pandas for correlation
df = pd.DataFrame({
    'Study Hours': study_hours,
    'Exam Scores': exam_scores,
    'Social Media Hours': social_media_hours,
    'Coffee Consumption': coffee_consumption
})
correlation_matrix = df.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

# 5. Visualizing correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()