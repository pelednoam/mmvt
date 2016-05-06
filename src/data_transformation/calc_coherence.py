import numpy as np
import os.path as op
import scipy.io as sio
import matplotlib.pyplot as plt
from mne.connectivity import spectral_connectivity
import time
from src.utils import utils
# from src import preproc_for_blender

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
BLENDER_ROOT_DIR = op.join(LINKS_DIR, 'mmvt')

HEMIS_WITHIN, HEMIS_BETWEEN = range(2)
STAT_AVG, STAT_DIFF = range(2)

def calc_electrodes_coh(subject, from_t_ind, to_t_ind, sfreq=1000, fmin=55, fmax=110, bw=15,
        dt=0.1, window_len=0.1, n_jobs=6):
    input_file = op.join(SUBJECTS_DIR, subject, 'electrodes', 'electrodes_data_trials.mat')
    d = sio.loadmat(input_file)
    output_file = op.join(BLENDER_ROOT_DIR, subject, 'electrodes_coh.npy')
    windows = np.linspace(0, 2.5-dt, 2.5 / dt)
    for cond, data in enumerate([d['interference'], d['noninterference']]):
        if cond == 0:
            coh_mat = np.zeros((data.shape[1], data.shape[1], len(windows), 2))
            # coh_mat = np.load(output_file)
            # continue
        ds_data = downsample_data(data)
        ds_data = ds_data[:, :, from_t_ind:to_t_ind]
        now = time.time()
        for win, tmin in enumerate(windows):
            print('cond {}, tmin {}'.format(cond, tmin))
            utils.time_to_go(now, win+1, len(windows))
            con_cnd, _, _, _, _ = spectral_connectivity(
                ds_data, method='coh', mode='multitaper', sfreq=sfreq,
                fmin=fmin, fmax=fmax, mt_adaptive=True, n_jobs=n_jobs, mt_bandwidth=bw, mt_low_bias=True,
                tmin=tmin, tmax=tmin+window_len)
            con_cnd = np.mean(con_cnd, axis=2)
            coh_mat[:, :, win, cond] = con_cnd
            # plt.matshow(con_cnd)
            # plt.show()
        np.save(output_file[:-4], coh_mat)


def downsample_data(data):
    C, E, T = data.shape
    new_data = np.zeros((C, E, int(T/2)))
    for epoch in range(C):
        new_data[epoch, :, :] = utils.downsample_2d(data[epoch, :, :], 2)
    return new_data


def save_electrodes_coh_to_blender(subject,  stat, threshold=0.8, bipolar=False):
    d = {}
    d['labels'], d['locations'] = get_electrodes_info(subject, bipolar)
    d['hemis'] = ['rh' if elc[0] == 'R' else 'lh' for elc in d['labels']]
    coh = get_electrodes_coh(subject, d['labels'])
    d['con_colors'], d['con_indices'], d['con_names'],  d['con_values'], d['con_types'] = \
        calc_connections_colors(subject, coh, d['labels'], d['hemis'], stat)
    d['conditions'] = ['interference', 'neutral']
    np.savez(op.join(BLENDER_ROOT_DIR, subject, 'electrodes_coh'), **d)


def load_coherence_meta_data_from_matlab(subject, matlab_electrodes_data_file):
    input_file = op.join(SUBJECTS_DIR, subject, 'electrodes', matlab_electrodes_data_file)
    d = utils.Bag(sio.loadmat(input_file))
    d['electrodes'] = [e[0][0].astype(str) for e in d['electrodes']]
    for f in ['Tdurr', 'Toffset', 'dt']:
        d[f] = d[f][0][0]
    meta_data = {f:d[f] for f in d.keys() if f in ['Tdurr', 'Toffset', 'dt', 'electrodes']}
    utils.save(meta_data, op.join(SUBJECTS_DIR, subject, 'electrodes_coh_meta_data.pkl'))


def get_electrodes_info(subject, bipolar=False):
    positions_file_name = 'electrodes{}_positions.npz'.format('_bipolar' if bipolar else '')
    positions_file_name = op.join(SUBJECTS_DIR, subject, 'electrodes', positions_file_name)
    d = np.load(positions_file_name)
    names = [l.astype(str) for l in d['names']]
    return names, d['pos']


def calc_sorting_indices(subject, labels):
    meta_data = utils.Bag(utils.load(op.join(SUBJECTS_DIR, subject, 'electrodes_coh_meta_data.pkl')))
    sorting_indices = np.array(utils.find_list_items_in_list(meta_data.electrodes, labels))
    if -1 in sorting_indices:
        raise Exception('You should check your lalbels...')
    return sorting_indices


def get_electrodes_coh(subject, labels):
    sorting_indices = calc_sorting_indices(subject, labels)
    input_file = op.join(BLENDER_ROOT_DIR, subject, 'electrodes_coh.npy')
    coh = np.load(input_file)
    sorted_coh = np.zeros(coh.shape)
    M = coh.shape[0]
    for i in range(M):
        new_i = sorting_indices[i]
        for j in range(M):
            new_j = sorting_indices[j]
            # print('{},{} -> {},{}'.format(i, j, new_i, new_j))
            sorted_coh[new_i, new_j] = coh[i, j]
    return sorted_coh


def calc_connections_colors(subject, data, labels, hemis, stat, threshold=0, color_map='jet',
                            norm_by_percentile=True, norm_percs=(1,99)):
        # cm_big='YlOrRd', cm_small='PuBu', flip_cm_big=True, flip_cm_small=False):
    M = data.shape[0]
    W = data.shape[2]
    L = int((M*M+M)/2-M)
    # con_colors = np.zeros((L, W, 3))
    con_indices = np.zeros((L, 2))
    con_values = np.zeros((L, W, 2))
    con_names = [None] * L
    con_type = np.zeros((L))
    coh_stat = utils.calc_stat_data(data, stat, axis=3)
    x = coh_stat.ravel()
    data_max, data_min = utils.get_data_max_min(x, norm_by_percentile, norm_percs)
    data_minmax = max(map(abs, [data_max, data_min]))
    # sm = utils.get_scalar_map(threshold, data_max, color_map=color_map)
    for cond in range(2):
        for w in range(W):
            # win_colors = utils.mat_to_colors(coh[:, :, w, cond], threshold, max_x, color_map, sm)
            # coh_arr = utils.lower_rec_to_arr(coh[:, :, w, cond])
            # win_colors = utils.arr_to_colors(coh_arr, threshold, max_x, color_map, sm)
            for ind, (i, j) in enumerate(utils.lower_rec_indices(M)):
                # con_colors[ind, w, cond, :] = win_colors[ind][:3]
                con_values[ind, w, cond] = data[i, j, w, cond]
    stat_data = utils.calc_stat_data(con_values, stat)
    con_colors = utils.mat_to_colors(stat_data, -data_minmax, data_minmax, color_map)

    for ind, (i, j) in enumerate(utils.lower_rec_indices(M)):
        con_indices[ind, :] = [i, j]
        con_names[ind] = '{}-{}'.format(labels[i].astype(str), labels[j].astype(str))
        con_type[ind] = HEMIS_WITHIN if hemis[i] == hemis[j] else HEMIS_BETWEEN

    print(L, ind)
    con_indices = con_indices.astype(np.int)
    return con_colors, con_indices, con_names, con_values, con_type


def main(subject, matlab_electrodes_data_file):
    from_t_ind, to_t_ind = 500, 3000
    stat = STAT_DIFF
    # calc_electrodes_coh(subject, from_t_ind, to_t_ind, n_jobs=1)
    # load_coherence_meta_data_from_matlab(subject, matlab_electrodes_data_file)
    save_electrodes_coh_to_blender(subject, stat)


if __name__ == '__main__':
    matlab_electrodes_data_file = 'electrodes_data_trials.mat'
    main('mg78', matlab_electrodes_data_file)
    print('finish!')