import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# 1. Creating example data
n = 50
x = np.random.rand(n)
y1 = x + np.random.normal(0, 0.1, n)  # Strong positive correlation
y2 = -x + np.random.normal(0, 0.1, n)  # Strong negative correlation
y3 = np.random.rand(n)  # No correlation

# 2. Calculating Pearson correlation
corr_xy1 = stats.pearsonr(x, y1)
corr_xy2 = stats.pearsonr(x, y2)
corr_xy3 = stats.pearsonr(x, y3)

print("Correlation between x and y1:", corr_xy1)
print("Correlation between x and y2:", corr_xy2)
print("Correlation between x and y3:", corr_xy3)

# 3. Visualizing correlations
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.scatter(x, y1)
ax1.set_title(f'Strong Positive Correlation (r={corr_xy1[0]:.2f})')
ax1.set_xlabel('x')
ax1.set_ylabel('y1')

ax2.scatter(x, y2)
ax2.set_title(f'Strong Negative Correlation (r={corr_xy2[0]:.2f})')
ax2.set_xlabel('x')
ax2.set_ylabel('y2')

ax3.scatter(x, y3)
ax3.set_title(f'No Correlation (r={corr_xy3[0]:.2f})')
ax3.set_xlabel('x')
ax3.set_ylabel('y3')

plt.tight_layout()
plt.show()

# 4. Using pandas for correlation
df = pd.DataFrame({'x': x, 'y1': y1, 'y2': y2, 'y3': y3})
correlation_matrix = df.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

# 5. Visualizing correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Heatmap')
plt.show()