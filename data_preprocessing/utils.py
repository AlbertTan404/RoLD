import copy
import numpy as np


def uniform_normalization(raw_data: np.array, min_values: np.array, max_values: np.array):
    lower_bound = -1
    upper_bound = 1
    std = max_values - min_values

    normalized_data = np.zeros_like(raw_data)
    # normalize the raw_data to [-1, 1] uniformly
    for i in range(7):
        if std[i] == 0:  # ignore
            continue
        normalized_data[:, i] = lower_bound + (upper_bound - lower_bound) * (raw_data[:, i] - min_values[i]) / std[i]

    return normalized_data

def scale_only_normalization(raw_data, min_values, max_values):
    normalized_data = copy.deepcopy(raw_data)
    for i in range(7):
        if min_values[i] == 0 or max_values[i] == 0:
            continue
        larger = max(abs(min_values[i]), abs(max_values[i]))
        normalized_data[i] /= larger
    return normalized_data

def scale_only_unnormalization(raw_data, min_values, max_values):
    unnormalized_data = copy.deepcopy(raw_data)
    for i in range(7):
        if min_values[i] == 0 or max_values[i] == 0:
            continue
        larger = max(abs(min_values[i]), abs(max_values[i]))
        unnormalized_data[i] *= larger
    return unnormalized_data

def uniform_unnormalization(normalized_data: np.array, min_values: np.array, max_values: np.array):
    # unnormalize the data in [-1, 1] back to the raw_data
    raw_data = min_values + 0.5 * (normalized_data + 1) * (max_values - min_values)
    return raw_data
