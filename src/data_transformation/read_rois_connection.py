import scipy.io as sio
import os.path as op
import numpy as np

from src.utils import utils
from src.utils import labels_utils as lu

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
BLENDER_ROOT_DIR = op.join(LINKS_DIR, 'mmvt')


STAT_AVG, STAT_DIFF = range(2)
HEMIS_WITHIN, HEMIS_BETWEEN = range(2)


def save_connectivity_to_blender(subject, atlas, data, conditions, stat, w=0, threshold=0, threshold_percentile=0):
    d = {}
    d['labels'] = lu.read_labels(subject, SUBJECTS_DIR, atlas, exclude=('unknown', 'corpuscallosum'),
            sorted_according_to_annot_file=True)
    d['locations'] = lu.calc_center_of_mass(d['labels'], ret_mat=True)
    d['hemis'] = ['rh' if l.hemi == 'rh' else 'lh' for l in d['labels']]
    d['labels'] = [l.name for l in d['labels']]
    d['con_colors'], d['con_indices'], d['con_names'],  d['con_values'], d['con_types'] = \
        calc_connections_colors(data, d['labels'], d['hemis'], stat, w, threshold_percentile=threshold_percentile)
    d['conditions'] = conditions
    np.savez(op.join(BLENDER_ROOT_DIR, subject, 'rois_con'), **d)


def calc_connections_colors(data, labels, hemis, stat, w, threshold=0, threshold_percentile=0, color_map='jet',
                            norm_by_percentile=True, norm_percs=(1, 99)):
    M = data.shape[0]
    W = data.shape[2] if w == 0 else w
    L = int((M * M + M) / 2 - M)
    con_indices = np.zeros((L, 2))
    con_values = np.zeros((L, W, 2))
    con_names = [None] * L
    con_type = np.zeros((L))
    axis = data.ndim - 1
    coh_stat = utils.calc_stat_data(data, stat, axis=axis)
    x = coh_stat.ravel()
    data_max, data_min = utils.get_data_max_min(x, norm_by_percentile, norm_percs)
    data_minmax = max(map(abs, [data_max, data_min]))
    for cond in range(2):
        for w in range(W):
            for ind, (i, j) in enumerate(utils.lower_rec_indices(M)):
                if W > 1:
                    con_values[ind, w, cond] = data[i, j, w, cond]
                else:
                    con_values[ind, w, cond] = data[i, j, cond]
    stat_data = utils.calc_stat_data(con_values, stat)
    con_colors = utils.mat_to_colors(stat_data, -data_minmax, data_minmax, color_map)

    for ind, (i, j) in enumerate(utils.lower_rec_indices(M)):
        con_indices[ind, :] = [i, j]
        con_names[ind] = '{}-{}'.format(labels[i], labels[j])
        con_type[ind] = HEMIS_WITHIN if hemis[i] == hemis[j] else HEMIS_BETWEEN

    con_indices = con_indices.astype(np.int)
    con_names = np.array(con_names)
    if threshold_percentile > 0:
        threshold = np.percentile(np.abs(stat_data), threshold_percentile)
    if threshold > 0:
        indices = np.where(np.abs(stat_data) >= threshold)[0]
        con_colors = con_colors[indices]
        con_indices = con_indices[indices]
        con_names = con_names[indices]
        con_values = con_values[indices]
        con_type  = con_type[indices]
    print(len(con_names))
    return con_colors, con_indices, con_names, con_values, con_type


def flatten_data(data, w=0):
    M = data.shape[0]
    L = int((M*M+M)/2-M)
    W = data.shape[2] if w == 0 else w
    values = np.zeros((L, W, 2))
    for cond in range(2):
        for w in range(W):
            for ind, (i, j) in enumerate(utils.lower_rec_indices(M)):
                values[ind, 0, cond] = data[i, j, cond]
    return values


if __name__ == '__main__':
    subject = 'fsaverage5c'
    fsaverage = 'fsaverage5c'
    atlas = 'laus125'

    root = '/cluster/neuromind/npeled/linda'
    flexibility_mat = sio.loadmat(op.join(root, 'figure_file1.mat'))['figure_file']
    conditions = ['task', 'rest']
    save_connectivity_to_blender(subject, atlas, flexibility_mat, conditions, STAT_DIFF, 1,
                                 threshold_percentile=99)

