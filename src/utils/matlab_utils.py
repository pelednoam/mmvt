import numpy as np
import scipy.io as sio
from pprint import pprint

from src.utils import utils


def load_mat_to_bag(mat_fname):
    return utils.Bag(dict(**sio.loadmat(mat_fname)))


def matlab_cell_arrays_to_dict(mat_fname):
    res_dict = {}
    d = dict(**sio.loadmat(mat_fname))
    keys = [key for key in d.keys() if key not in ['__header__', '__version__', '__globals__']]
    for key in keys:
        data_type = d[key][0][0].dtype
        #todo: check more types than only numbers and strings
        # check if the data type is numeric ('u' for unsigned numeric)
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
    if len(cell_arr[0]) > 1:
        ret_list = [cell_arr[0][ind][0].astype(str) for ind in range(len(cell_arr[0]))]
    else:
        ret_list = [cell_arr[ind][0][0].astype(str) for ind in range(len(cell_arr))]
    return ret_list


if __name__ == '__main__':
    mat_fanme = '/cluster/neuromind/npeled/Documents/darpa_electrodes_csvs/ChannelPairNames2.mat'
    vars_dic = matlab_cell_arrays_to_dict(mat_fanme)
    pprint(vars_dic)