import scipy.io as sio
import os.path as op
import numpy as np
import matplotlib.pyplot as plt

from src.utils import utils
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


def load_tatiana_meg_coh(subject, fsaverage, atlas, ds_all, ds_subject, hc_indices, subject_index, conditions, labels):
    for sub, ds, indices in zip([subject, fsaverage], [ds_subject, ds_all], [hc_indices, subject_index]):
        if sub != 'fscopy':
            continue
        print(sub)
        d = {}
        N = len(labels)
        d['labels'] = labels
        d['hemis'] = [l[-2:] for l in d['labels']]
        locations = utils.load(op.join(SUBJECTS_DIR, sub, 'label', '{}_center_of_mass.pkl'.format(atlas)))
        d['locations'] = np.array([locations[l] for l in d['labels']]) * 1000.0
        # todo: find why...
        # if sub == subject:
        #     d['locations'] *= 1000.0
        d['conditions'] = conditions
        d['con_indices'], d['con_names'], d['con_types'] = calc_meta_data(d['labels'], d['hemis'], 1, N)
        high_low_diff = np.zeros((len(np.tril_indices(N, -1)[0]), len(ds_all)))
        for ind in range(len(ds)):
            high_low = np.zeros((ds_all[ind]['high'].shape[0], ds_all[ind]['high'].shape[1], 2))
            high_low[:, :, 0] = np.mean(ds_all[ind]['high'][:, :, indices], 2)
            high_low[:, :, 1] = np.mean(ds_all[ind]['low'][:, :, indices], 2)
            ds[ind]['data'] = np.zeros((N, N, 1))
            ds[ind]['data'][:, :, 0] = ds[ind]['pp_mat']
            ds[ind]['data'][np.where(np.isnan(ds[ind]['data']))] = 1
            ds[ind]['con_values'], high_low_diff[:, ind] = create_meg_coh_data(
                ds[ind]['data'], len(d['conditions']), high_low)
        d['con_values'] = np.zeros((ds[0]['con_values'].shape[0], len(ds), 1))
        for ind in range(len(ds)):
            d['con_values'][:, ind, :] = ds[ind]['con_values']
        d['con_colors'] = calc_con_colors(d['con_values'], high_low_diff)
        np.savez(op.join(BLENDER_ROOT_DIR, sub, 'rois_con.npz'), **d)


def calc_meta_data(labels, hemis, conds_num, N):
    # todo: this is copy paste from matlab, should be re-written to real python...
    M = len(np.tril_indices(N, -1)[0]) #N * N - N
    con_names = []
    con_indices = np.zeros((M, 2))
    con_types = np.zeros((M))
    ind = 0
    for i in range(N):
        for j in range(N):
            if i > j:
                con_indices[ind, :] = [i, j]
                con_names.append('{}-{}'.format(labels[i], labels[j]))
                con_types[ind] = HEMIS_WITHIN if hemis[i] == hemis[j] else HEMIS_BETWEEN
                ind += 1
    return con_indices, con_names, con_types


def create_meg_coh_data(data, conds_num, high_low):
    # todo: this is copy paste from matlab, should be re-written to real python...
    N = data.shape[0]
    M = len(np.tril_indices(N, -1)[0]) #N * N - N
    con_values = np.zeros((M, conds_num))
    high_low_diff = np.zeros((M))
    triu_indices = np.triu_indices(N)
    for cond in range(conds_num):
        data[triu_indices[0], triu_indices[1], cond] = data[:, :, cond][np.tril_indices(N)]
    for cond in range(conds_num):
        ind = 0
        for i in range(N):
            for j in range(N):
                if i > j:
                    if cond == 0:
                        high_low_diff[ind] = high_low[i, j, 0] - high_low[i, j, 1]
                    con_values[ind, cond] = -np.log10(data[i, j, cond]) * np.sign(high_low_diff[ind])
                    ind += 1
    return con_values, high_low_diff


def calc_con_colors(con_values, high_low_diff):
    M = con_values.shape[0]
    stat_data = utils.calc_stat_data(con_values, STAT_AVG, axis=con_values.ndim-1)
    # con_colors = utils.arr_to_colors(stat_data, 0, 1)[:, :3]
    # con_colors = utils.arr_to_colors_two_colors_maps(stat_data, 0, 1, 'RdPu', 'hot', 0.05, flip_cm_big=True)[:, :3]
    from src.mmvt_addon import colors_utils
    red = np.array(colors_utils.name_to_rgb('red')) / 255.0
    blue = np.array(colors_utils.name_to_rgb('blue')) / 255.0
    magenta = np.array(colors_utils.name_to_rgb('magenta')) / 255.0
    green = np.array(colors_utils.name_to_rgb('green')) / 255.0
    if con_values.ndim == 2:
        con_colors = np.zeros((M, 3))
        con_colors[(stat_data <= 0.05) & (high_low_diff >= 0)] = red
        con_colors[(stat_data <= 0.05) & (high_low_diff < 0)] = blue
        con_colors[(stat_data > 0.05) & (high_low_diff >= 0)] = magenta
        con_colors[(stat_data > 0.05) & (high_low_diff < 0)] = green
    elif con_values.ndim == 3:
        W = con_values.shape[1]
        con_colors = np.zeros((M, W, 3))
        for w in range(W):
            stat_w = stat_data[:, w]
            high_low_diff_w = high_low_diff[:, w]
            sig_high = (abs(stat_w) >= -np.log10(0.05)) & (high_low_diff_w >= 0)
            sig_low =  (abs(stat_w) >= -np.log10(0.05)) & (high_low_diff_w < 0)
            print(w, sig_high, sig_low)
            con_colors[sig_high, w] = red
            con_colors[sig_low, w] = blue
            con_colors[(abs(stat_w) < -np.log10(0.05)) & (high_low_diff_w >= 0), w] = (1, 1, 1)
            con_colors[(abs(stat_w) < -np.log10(0.05)) & (high_low_diff_w < 0), w] = (1, 1, 1)
    # con_colors = con_colors[:, :, :, np.newaxis]
    return con_colors

if __name__ == '__main__':
    subject = 'pp009'
    fsaverage = 'fscopy'
    atlas = 'arc_april2016'
    # load_tatiana_data()
    # load_meg_coh_python_data()

    root = '/autofs/space/sophia_002/users/DARPA-MEG/project_slides_160420/noam'
    if not op.isdir(root):
        root = '/home/noam/MEG/ARC/pp009/misc/'
    # coh_d = sio.loadmat(op.join(root, 'alpha_risk.mat'))
    meta_d = sio.loadmat(op.join(root, 'ARC12_30all_bw8wpli.mat'))
    meta_d['conty_data'] = []
    freq = 'beta'
    con_all = [sio.loadmat(op.join(root, '{}_risk_all.mat'.format(freq))),
                sio.loadmat(op.join(root, '{}_reward_all.mat'.format(freq))),
                sio.loadmat(op.join(root, '{}_approshock_all.mat'.format(freq)))]
    con_subject = [sio.loadmat(op.join(root, '{}_risk_{}.mat'.format(freq, subject))),
                    sio.loadmat(op.join(root, '{}_reward_{}.mat'.format(freq, subject))),
                    sio.loadmat(op.join(root, '{}_approshock_{}.mat'.format(freq, subject)))]
    # d = utils.merge_two_dics(coh_d, meta_d)
    hc_indices = np.arange(12)
    subject_index = 12
    labels = ['vlpfc-rh', 'vlpfc-lh', 'ofc-rh', 'ofc-lh']
    conditions = ['high_vs_low']
    rois = [l.strip()[:-len('label')-1] for l in meta_d['ROIs']]
    N = con_all[0]['pp_mat'].shape[0]
    labels = [l for l in rois if l in labels] if len(labels) > 0 else rois[:N]
    load_tatiana_meg_coh('pp009', 'fscopy', atlas, con_all, con_subject, hc_indices, hc_indices, conditions, labels)
    print('finish!')