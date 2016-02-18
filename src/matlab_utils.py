import numpy as np
import scipy.io as sio


def matlab_cell_arrays_to_dict(mat_fname):
    res_dict = {}
    d = dict(**sio.loadmat(mat_fname))
    keys = [key for key in d.keys() if key not in ['__header__', '__version__', '__globals__']]
    for key in keys:
        data_type = d[key][0][0].dtype
        #todo: check more types than only numbers and strings
        if data_type.kind in np.typecodes['AllInteger'] + 'u':
            res_dict[key] = matlab_cell_array_to_list(d[key])
        else:
            res_dict[key] = matlab_cell_str_to_list(d[key])
    return res_dict


def matlab_cell_array_to_list(cell_arr):
    ret_list = []
    for ind in range(len(cell_arr[0])):
        arr = cell_arr[0][ind]
        ret_list.append([arr[k][0] for k in range(len(arr))])
    return ret_list


def matlab_cell_str_to_list(cell_arr):
    ret_list = []
    for ind in range(len(cell_arr[0])):
        ret_list.append(cell_arr[0][ind][0].astype(str))
    return ret_list