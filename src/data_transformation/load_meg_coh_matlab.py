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
STAT_AVG, STAT_DIFF = range(2)

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


def load_tatiana_meg_coh(subject, atlas, d):
    d['labels'] = [l.strip()[:-len('label')-1] for l in d['ROIs']]
    d['hemis'] = [l[-2:] for l in d['labels']]
    locations = utils.load(op.join(SUBJECTS_DIR, subject, 'label', '{}_center_of_mass.pkl'.format(atlas)))
    d['locations'] = np.array([locations[l] for l in d['labels']]) # * 1000.0
    d['conditions'] = ['high_vs_low'] # ['high', 'low']
    low_high = np.zeros((d['high'].shape[0], d['high'].shape[1], 2))
    low_high[:, :, 0] = np.mean(d['high'], 2)
    low_high[:, :, 1] = np.mean(d['low'], 2)
    N = d['pp_mat'].shape[0]
    d['data'] = np.zeros((N, N, len(d['conditions'])))
    d['data'][:, :, 0] = d['pval_mat'] # d['pp_mat']
    d['data'][np.where(np.isnan(d['data']))] = 1
    d['con_colors'], d['con_indices'], d['con_names'], d['con_values'], d['con_types'] = \
        create_meg_coh_data(d['data'], d['labels'], d['hemis'], d['conditions'], low_high)
    # d['con_colors'], d['con_indices'], d['con_names'], d['con_values'], d['con_types'] =  \
    #     calc_coherence.calc_connections_colors(subject, d['labels'], d['hemis'], stat)
    np.savez(op.join(BLENDER_ROOT_DIR, subject, 'meg_coh_bev.npz'), **d)


def create_meg_coh_data(data, labels, hemis, conds, high_low):
    # todo: this is copy paste from matlab, should be re-written to real python...
    N = data.shape[0]
    M = N * N
    con_names = []
    conds_num = len(conds)
    # con_colors = np.zeros((M, conds_num, 3))
    con_values = np.zeros((M, conds_num))
    high_low_diff = np.zeros((M))
    con_indices = np.zeros((M, 2))
    con_types = np.zeros((M))
    triu_indices = np.triu_indices(N)
    for cond in range(conds_num):
        data[triu_indices[0], triu_indices[1], cond] = data[:, :, cond][np.tril_indices(N)]
    for cond in range(conds_num):
        ind = 0
        for i in range(N):
            for j in range(N):
                # con_colors[num, cond, :] = [i, j, data[i, j, cond]]
                con_values[ind, cond] = data[i, j, cond]
                if cond == 0:
                    con_indices[ind, :] = [i, j]
                    con_names.append('{}-{}'.format(labels[i], labels[j]))
                    con_types[ind] = HEMIS_WITHIN if hemis[i] == hemis[j] else HEMIS_BETWEEN
                    high_low_diff[ind] = high_low[i, j, 0] - high_low[i, j, 1]
                ind += 1

                    # min_val = np.min(con_colors[:, 2, cond])
    # max_val = np.max(con_colors[:, 2, cond])
    # vals_range = max_val - min_val
    stat_data = utils.calc_stat_data(con_values, STAT_AVG, axis=1)
    # con_colors = utils.arr_to_colors(stat_data, 0, 1)[:, :3]
    # con_colors = utils.arr_to_colors_two_colors_maps(stat_data, 0, 1, 'RdPu', 'hot', 0.05, flip_cm_big=True)[:, :3]
    from src.mmvt_addon import colors_utils
    red = np.array(colors_utils.name_to_rgb('red')) / 255.0
    blue = np.array(colors_utils.name_to_rgb('blue')) / 255.0
    magenta = np.array(colors_utils.name_to_rgb('magenta')) / 255.0
    green = np.array(colors_utils.name_to_rgb('green')) / 255.0
    con_colors = np.zeros((M, 3))
    con_colors[(stat_data <= 0.05) & (high_low_diff >= 0)] = red
    con_colors[(stat_data <= 0.05) & (high_low_diff < 0)] = blue
    con_colors[(stat_data > 0.05) & (high_low_diff >= 0)] = magenta
    con_colors[(stat_data > 0.05) & (high_low_diff < 0)] = green
    return con_colors, con_indices, con_names, con_values, con_types

if __name__ == '__main__':
    subject = 'fscopy'
    atlas = 'arc_april2016'
    # load_tatiana_data()
    # load_meg_coh_python_data()

    root = '/autofs/space/sophia_002/users/DARPA-MEG/arc/noam'
    # root = '/home/noam/MEG/ARC/pp009/misc/'
    coh_d = sio.loadmat(op.join(root, 'alpha_risk.mat'))
    meta_d = sio.loadmat(op.join(root, 'ARC12_30all_bw8wpli.mat'))
    meta_d['conty_data'] = []
    d = utils.merge_two_dics(coh_d, meta_d)
    load_tatiana_meg_coh(subject, atlas, d)
    print('finish!')