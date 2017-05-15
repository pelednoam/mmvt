import numpy as np


def filter_traces(data, t_range, top_k=0, threshold=0):
    if data.shape[2] == 2:
        diff_data = np.diff(data[:, t_range, :], axis=2)
    else:
        diff_data = data[:, t_range, :]
    max_diff = np.max(np.abs(np.squeeze(diff_data)), axis=1)
    if top_k == 0:
        top_k = sum(max_diff > 0)
    indices = np.where(max_diff > threshold)[0]
    filtered_objects = sorted(indices, key=lambda i: max_diff[i])[::-1][:top_k]
    return filtered_objects, max_diff
