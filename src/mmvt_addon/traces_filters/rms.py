import numpy as np


def filter_traces(data, t_range, top_k=0, threshold=0):
    data_diff = np.diff(data[:, t_range, :], axis=2)
    if data_diff.shape[2] > 1:
        data_diff = np.sum(data_diff, axis=2)
    data_diff = np.squeeze(data_diff)
    rms = np.sqrt(np.sum(np.power(data_diff, 2), 1))
    if top_k == 0:
        top_k = sum(rms > 0)
    filtered_objects = np.argsort(rms)[::-1][:top_k]
    return filtered_objects, rms