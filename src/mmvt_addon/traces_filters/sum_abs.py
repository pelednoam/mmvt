import numpy as np


def filter_traces(data, t_range, top_k=0, threshold=0):
    sub_abs = np.sum(abs(data[:, t_range, :]), (1, 2))
    if top_k == 0:
        top_k = sum(sub_abs > 0)
    objects = np.argsort(sub_abs)[::-1][:top_k]
    return objects, sub_abs
