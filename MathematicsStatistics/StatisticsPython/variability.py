import numpy as np
import statistics

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

# 1. List of integers
data_integers = [1, 2, 3, 4, 5]
range_integers, iqr_integers, variance_integers, std_deviation_integers = measures_of_variability(data_integers)
print("For integers:")
print(f"Range: {range_integers}")
print(f"Interquartile Range (IQR): {iqr_integers}")
print(f"Variance: {variance_integers}")
print(f"Standard Deviation: {std_deviation_integers}\n")

# 2. List of floats
data_floats = [1.1, 2.2, 3.3, 4.4, 5.5]
range_floats, iqr_floats, variance_floats, std_deviation_floats = measures_of_variability(data_floats)
print("For floats:")
print(f"Range: {range_floats}")
print(f"Interquartile Range (IQR): {iqr_floats}")
print(f"Variance: {variance_floats}")
print(f"Standard Deviation: {std_deviation_floats}\n")

# 3. List of tuples
# For tuples, we need to extract the specific element for which we want to calculate the measures of variability
data_tuples = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]
# Extracting the first element of each tuple
data_tuples_first_element = [x[0] for x in data_tuples]
range_tuples, iqr_tuples, variance_tuples, std_deviation_tuples = measures_of_variability(data_tuples_first_element)
print("For first elements of tuples:")
print(f"Range: {range_tuples}")
print(f"Interquartile Range (IQR): {iqr_tuples}")
print(f"Variance: {variance_tuples}")
print(f"Standard Deviation: {std_deviation_tuples}\n")

# 4. List of objects (custom class)
class DataPoint:
    def __init__(self, value):
        self.value = value

data_objects = [DataPoint(1), DataPoint(2), DataPoint(3), DataPoint(4), DataPoint(5)]
# Extracting the 'value' attribute from each object
data_objects_values = [obj.value for obj in data_objects]
range_objects, iqr_objects, variance_objects, std_deviation_objects = measures_of_variability(data_objects_values)
print("For object values:")
print(f"Range: {range_objects}")
print(f"Interquartile Range (IQR): {iqr_objects}")
print(f"Variance: {variance_objects}")
print(f"Standard Deviation: {std_deviation_objects}")
