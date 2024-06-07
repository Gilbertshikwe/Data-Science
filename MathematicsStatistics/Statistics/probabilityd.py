# Probability distributions describe how probabilities are distributed 
# over the values of a random variable.

import numpy as np  # Correctly import numpy with alias 'np'
import matplotlib.pyplot as plt  # type:ignore
import seaborn as sns  # type:ignore

# Generate data representing the heights of 1000 students
student_heights = np.random.normal(loc=170, scale=10, size=1000)

# Plot the distribution of student heights
sns.histplot(student_heights, kde=True)
plt.title("Distribution of Student Heights")
plt.xlabel("Height (cm)")
plt.ylabel("Frequency")

# Display the plot
plt.show()