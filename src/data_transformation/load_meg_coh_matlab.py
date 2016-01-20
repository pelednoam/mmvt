import scipy.io as sio
import os.path as op
import numpy as np
import matplotlib.pyplot as plt

from src import utils
# from src import preproc_for_blender

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
BLENDER_ROOT_DIR = op.join(LINKS_DIR, 'mmvt')

TATIANA_ROOT = '/autofs/space/franklin_003/users/npeled/glassy_brain'
HEMIS_WITHIN, HEMIS_BETWEEN = range(2)

def load_tatiana_data():
    d = sio.loadmat(op.join(TATIANA_ROOT, 'coh.m'))
    # labels is a cell array, so convert in to a list
    d['labels'] = [str(l[0][0]) for l in d['labels']]
    d['hemis'] = [str(l[0][0]) for l in d['hemis']]
    d['conditions'] = [str(l[0][0]) for l in d['conditions']]
    np.savez(op.join(TATIANA_ROOT, 'coh'), **d)


def load_meg_coh_python_data():
    d = utils.Bag(np.load(op.join(TATIANA_ROOT, 'coh.npz')))
    # 'coh_colors', 'locations', 'labels', 'conditions', 'data', 'hemis'
    print(d)


def load_electrodes_coh(subject, threshold=0.8, bipolar=False, color_map='jet'):
    d = {}
    # todo: read the laels from the matlab file!!!
    d['labels'], d['locations'] = get_electrodes_info(subject, bipolar)
    d['hemis'] = ['rh' if elc[0] == 'R' else 'lh' for elc in d['labels']]
    d['con_colors'], d['con_indices'], d['con_names'],  d['con_values'], d['con_types'] = \
        calc_electrodes_coh_colors(subject, d['labels'], d['hemis'], threshold, color_map)
    d['conditions'] = ['interference', 'neutral']
    np.savez(op.join(BLENDER_ROOT_DIR, subject, 'electrodes_coh'), **d)


def get_electrodes_info(subject, bipolar=False):
    positions_file_name = 'electrodes{}_positions.npz'.format('_bipolar' if bipolar else '')
    positions_file_name = op.join(SUBJECTS_DIR, subject, 'electrodes', positions_file_name)
    d = np.load(positions_file_name)
    names = [l.astype(str) for l in d['names']]
    return names, d['pos']


def calc_electrodes_coh_colors(subject, labels, hemis, threshold=0.5, color_map='jet'):
    input_file = op.join(BLENDER_ROOT_DIR, subject, 'electrodes_coh.npy')
    coh = np.load(input_file)
    M = coh.shape[0]
    W = coh.shape[2]
    L = int((M*M+M)/2-M)
    # coh_colors = np.zeros((L, 6, W, 2))
    con_colors = np.zeros((L, W, 2, 3))
    con_indices = np.zeros((L, 2))
    con_values = np.zeros((L, W, 2))
    con_names = [None] * L
    con_type = np.zeros((L))
    for cond in range(2):
        x = coh[:, :, :, cond].ravel()
        min_x, max_x = np.percentile(x, 1), np.percentile(x, 99) # np.min(x), np.max(x)
        if np.max(x) < threshold:
            continue
        # sm = utils.get_scalar_map(threshold, max_x, color_map=color_map)
        sm = utils.get_scalar_map(0.8, 0.9, color_map=color_map)
        for w in range(W):
            # win_colors = utils.mat_to_colors(coh[:, :, w, cond], threshold, max_x, color_map, sm)
            coh_arr = utils.lower_rec_to_arr(coh[:, :, w, cond])
            win_colors = utils.arr_to_colors(coh_arr, 0.8, 0.9, color_map, sm)
            for ind, (i, j) in enumerate(utils.lower_rec_indices(M)):
                con_colors[ind, w, cond, :] = win_colors[ind][:3]
                con_values[ind, w, cond] = coh[i, j, w, cond]
    for ind, (i, j) in enumerate(utils.lower_rec_indices(M)):
        con_indices[ind, :] = [i, j]
        con_names[ind] = '{}-{}'.format(labels[i].astype(str), labels[j].astype(str))
        con_type[ind] = HEMIS_WITHIN if hemis[i] == hemis[j] else HEMIS_BETWEEN

    print(L, ind)
    con_indices = con_indices.astype(np.int)
    return con_colors, con_indices, con_names, con_values, con_type


if __name__ == '__main__':
    subject = 'mg78'
    # load_tatiana_data()
    # load_meg_coh_python_data()
    load_electrodes_coh(subject)
    print('finish!')