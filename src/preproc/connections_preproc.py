import os
import os.path as op
import numpy as np
import scipy.io as sio

from src.utils import utils
from src.utils import labels_utils as lu

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
FREE_SURFER_HOME = utils.get_link_dir(LINKS_DIR, 'freesurfer', 'FREESURFER_HOME')
BLENDER_ROOT_DIR = op.join(LINKS_DIR, 'mmvt')
os.environ['SUBJECTS_DIR'] = SUBJECTS_DIR

STAT_AVG, STAT_DIFF = range(2)
HEMIS_WITHIN, HEMIS_BETWEEN = range(2)


#todo: Add the necessary parameters
def save_electrodes_coh(subject,  stat, threshold=0.8, bipolar=False):
    d = dict()
    d['labels'], d['locations'] = get_electrodes_info(subject, bipolar)
    d['hemis'] = ['rh' if elc[0] == 'R' else 'lh' for elc in d['labels']]
    coh = calc_electrodes_coh(subject, from_t_ind=0, to_t_ind=-1, sfreq=1000, fmin=55, fmax=110, bw=15,
                              dt=0.1, window_len=0.1, n_jobs=6)
    d['con_colors'], d['con_indices'], d['con_names'],  d['con_values'], d['con_types'] = \
        calc_connections_colors(subject, coh, d['labels'], d['hemis'], stat)
    d['conditions'] = ['interference', 'neutral']
    np.savez(op.join(BLENDER_ROOT_DIR, subject, 'electrodes_coh'), **d)


def get_electrodes_info(subject, bipolar=False):
    positions_file_name = 'electrodes{}_positions.npz'.format('_bipolar' if bipolar else '')
    positions_file_name = op.join(SUBJECTS_DIR, subject, 'electrodes', positions_file_name)
    d = np.load(positions_file_name)
    names = [l.astype(str) for l in d['names']]
    return names, d['pos']


def calc_electrodes_coh(subject, from_t_ind, to_t_ind, sfreq=1000, fmin=55, fmax=110, bw=15,
                        dt=0.1, window_len=0.1, n_jobs=6):

    from mne.connectivity import spectral_connectivity
    import time

    input_file = op.join(SUBJECTS_DIR, subject, 'electrodes', 'electrodes_data_trials.mat')
    d = sio.loadmat(input_file)
    output_file = op.join(BLENDER_ROOT_DIR, subject, 'electrodes_coh.npy')
    windows = np.linspace(0, 2.5 - dt, 2.5 / dt)
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
            utils.time_to_go(now, win + 1, len(windows))
            con_cnd, _, _, _, _ = spectral_connectivity(
                ds_data, method='coh', mode='multitaper', sfreq=sfreq,
                fmin=fmin, fmax=fmax, mt_adaptive=True, n_jobs=n_jobs, mt_bandwidth=bw, mt_low_bias=True,
                tmin=tmin, tmax=tmin + window_len)
            con_cnd = np.mean(con_cnd, axis=2)
            coh_mat[:, :, win, cond] = con_cnd
            # plt.matshow(con_cnd)
            # plt.show()
        np.save(output_file[:-4], coh_mat)
    return coh_mat


def downsample_data(data):
    C, E, T = data.shape
    new_data = np.zeros((C, E, int(T/2)))
    for epoch in range(C):
        new_data[epoch, :, :] = utils.downsample_2d(data[epoch, :, :], 2)
    return new_data


def save_rois_connectivity(subject, atlas, rois_con_fname, mat_field, conditions, stat=STAT_DIFF, w=0,
                           exclude=['unknown', 'corpuscallosum'], threshold=0, threshold_percentile=0,
                           color_map='jet', norm_by_percentile=True, norm_percs=(1, 99)):
    d = dict()
    data = sio.loadmat(rois_con_fname)[mat_field]
    d['labels'] = lu.read_labels(subject, SUBJECTS_DIR, atlas, exclude=exclude, sorted_according_to_annot_file=True)
    d['locations'] = lu.calc_center_of_mass(d['labels'], ret_mat=True)
    d['hemis'] = ['rh' if l.hemi == 'rh' else 'lh' for l in d['labels']]
    d['labels'] = [l.name for l in d['labels']]
    d['con_colors'], d['con_indices'], d['con_names'],  d['con_values'], d['con_types'] = \
        calc_connections_colors(data, d['labels'], d['hemis'], stat, w, threshold, threshold_percentile, color_map,
                                norm_by_percentile, norm_percs)
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


def main(subject, args):
    if utils.should_run('save_rois_connectivity'):
        save_rois_connectivity(subject, args.atlas, args.rois_con_fname, args.mat_field, args.conditions, args.stat,
                               args.w, args.labels_exclude, args.threshold, args.threshold_percentile,
                               args.color_map, args.norm_by_percentile, args.norm_percs)

    if utils.should_run('save_electrodes_coh'):
        # todo: Add the necessary parameters
        save_electrodes_coh(subject, args.stat, args.threshold)


def example(args):
    subject = 'fsaverage5c'
    args.atlas = 'laus125'
    args.rois_con_fname = '/cluster/neuromind/npeled/linda/figure_file1.mat'
    args.mat_field = 'figure_file'
    args.w = 1
    args.stat = STAT_DIFF
    args.threshold_percentile = 99
    args.conditions = ['task', 'rest']
    save_rois_connectivity(subject, args.atlas, args.rois_con_fname, args.mat_field, args.conditions, args.stat,
                           w=args.w, threshold_percentile=args.threshold_percentile)


if __name__ == '__main__':
    import argparse
    from src.utils import args_utils as au
    parser = argparse.ArgumentParser(description='MMVT stim preprocessing')
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-c', '--conditions', help='conditions names', required=True, type=au.str_arr_type)
    parser.add_argument('-a', '--atlas', help='atlas name', required=False, default='aparc.DKTatlas40')
    parser.add_argument('-f', '--function', help='function name', required=False, default='all', type=au.str_arr_type)
    parser.add_argument('--rois_con_fname', help='rois connection file name', required=False, default='')
    parser.add_argument('--mat_field', help='rois connection matlab field name', required=False, default='')
    parser.add_argument('--labels_exclude', help='rois to exclude', required=False, default='unknown,corpuscallosum',
                        type=au.str_arr_type)
    parser.add_argument('--norm_by_percentile', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--norm_percs', help='', required=False, default='1,99', type=au.int_arr_type)
    parser.add_argument('--n_jobs', help='cpu num', required=False, default=-1)
    args = utils.Bag(au.parse_parser(parser))
    print(args)
    args.n_jobs = utils.get_n_jobs(args.n_jobs)

    for subject in args.subject:
        main(subject, args)

    print('finish!')