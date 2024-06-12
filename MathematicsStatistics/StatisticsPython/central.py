import numpy as np
from scipy import stats
from scipy.stats import gmean, hmean

def ungrouped_data_central_tendency(data):
    mean = np.mean(data)
    median = np.median(data)
    mode = stats.mode(data, keepdims=True)[0][0]  # Use keepdims=True to get a scalar value
    return mean, median, mode

def grouped_data_central_tendency(class_intervals, frequencies):
    midpoints = [(interval[0] + interval[1]) / 2 for interval in class_intervals]

    # Mean for grouped data
    mean_grouped = sum(f * m for f, m in zip(frequencies, midpoints)) / sum(frequencies)

    # Median for grouped data
    total_frequency = sum(frequencies)
    cumulative_frequencies = np.cumsum(frequencies)
    median_class_index = np.where(cumulative_frequencies >= total_frequency / 2)[0][0]
    L = class_intervals[median_class_index][0]
    CF = cumulative_frequencies[median_class_index - 1] if median_class_index > 0 else 0
    f = frequencies[median_class_index]
    h = class_intervals[median_class_index][1] - class_intervals[median_class_index][0]
    median_grouped = L + ((total_frequency / 2 - CF) / f) * h

    # Mode for grouped data
    modal_class_index = np.argmax(frequencies)
    L = class_intervals[modal_class_index][0]
    f1 = frequencies[modal_class_index]
    f0 = frequencies[modal_class_index - 1] if modal_class_index > 0 else 0
    f2 = frequencies[modal_class_index + 1] if modal_class_index < len(frequencies) - 1 else 0
    mode_grouped = L + ((f1 - f0) / (2 * f1 - f0 - f2)) * h

    return mean_grouped, median_grouped, mode_grouped

# Example data
data = [2, 4, 6, 8, 10, 6]
mean, median, mode = ungrouped_data_central_tendency(data)
print(f"Ungrouped Data - Mean: {mean}, Median: {median}, Mode: {mode}")

class_intervals = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
frequencies = [2, 3, 5, 4, 1]
mean_grouped, median_grouped, mode_grouped = grouped_data_central_tendency(class_intervals, frequencies)
print(f"Grouped Data - Mean: {mean_grouped}, Median: {median_grouped}, Mode: {mode_grouped}")
