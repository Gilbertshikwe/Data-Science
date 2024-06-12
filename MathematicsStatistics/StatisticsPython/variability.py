import numpy as np

def measures_of_variability(data):
    # Range
    range_value = np.ptp(data)
    
    # Interquartile Range (IQR)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    # Variance (Sample)
    variance = np.var(data, ddof=1)
    
    # Standard Deviation (Sample)
    std_deviation = np.std(data, ddof=1)
    
    return range_value, iqr, variance, std_deviation

# Example data
data = [2, 4, 6, 8, 10]
range_value, iqr, variance, std_deviation = measures_of_variability(data)

print(f"Range: {range_value}")
print(f"Interquartile Range (IQR): {iqr}")
print(f"Variance: {variance}")
print(f"Standard Deviation: {std_deviation}")
