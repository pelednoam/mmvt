import os.path as op
import numpy as np
import scipy.io as sio
import traceback

from src.utils import utils
from src.utils import preproc_utils as pu
from src.utils import labels_utils as lu

SUBJECTS_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()
LINKS_DIR = utils.get_links_dir()
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
STAT_AVG, STAT_DIFF = range(2)
HEMIS_WITHIN, HEMIS_BETWEEN = range(2)


#todo: Add the necessary parameters
# args.conditions, args.mat_fname, args.t_max, args.stat, args.threshold)
def save_electrodes_coh(subject, args):
    output_fname = op.join(MMVT_DIR, subject, 'electrodes', 'electrodes_con.npz')
    utils.remove_file(output_fname)
    try:
        d = dict()
        d['labels'], d['locations'] = get_electrodes_info(subject, args.bipolar)
        d['hemis'] = ['rh' if elc[0] == 'R' else 'lh' for elc in d['labels']]
        coh_fname = op.join(MMVT_DIR, subject, 'electrodes', 'electrodes_coh.npy')
        if not op.isfile(coh_fname):
            coh = calc_electrodes_coh(
                subject, args.conditions, args.mat_fname, args.t_max, from_t_ind=0, to_t_ind=-1, sfreq=1000, fmin=55, fmax=110, bw=15,
                dt=0.1, window_len=0.1, n_jobs=6)
        else:
            coh = np.load(coh_fname)
        (d['con_colors'], d['con_indices'], d['con_names'],  d['con_values'], d['con_types'],
         d['data_max'], d['data_min']) = calc_connectivity(coh, d['labels'], d['hemis'], args)
        d['conditions'] = args.conditions # ['interference', 'neutral']
        np.savez(output_fname, **d)
        print('Electodes coh was saved to {}'.format(output_fname))
    except:
        print(traceback.format_exc())
        return False
    return op.isfile(output_fname)


def get_electrodes_info(subject, bipolar=False):
    positions_fname= 'electrodes{}_positions.npz'.format('_bipolar' if bipolar else '')
    positions_full_fname = op.join(SUBJECTS_DIR, subject, 'electrodes', positions_fname)
    if not op.isfile(positions_full_fname):
        positions_full_fname = op.join(MMVT_DIR, subject, 'electrodes', positions_fname)
    d = np.load(positions_full_fname)
    names = [l.astype(str) for l in d['names']]
    return names, d['pos']


def calc_electrodes_coh(subject, conditions, mat_fname, t_max, from_t_ind, to_t_ind, sfreq=1000, fmin=55, fmax=110, bw=15,
                        dt=0.1, window_len=0.1, n_jobs=6):

    from mne.connectivity import spectral_connectivity
    import time

    input_file = op.join(SUBJECTS_DIR, subject, 'electrodes', mat_fname)
    d = sio.loadmat(input_file)
    output_file = op.join(MMVT_DIR, subject, 'electrodes_coh.npy')
    windows = np.linspace(0, t_max - dt, t_max / dt)
    for cond, data in enumerate([d[cond] for cond in conditions]):
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


def save_rois_connectivity(subject, args):
    # atlas, mat_fname, mat_field, conditions, stat=STAT_DIFF, windows=0,
    #                        labels_exclude=['unknown', 'corpuscallosum'], threshold=0, threshold_percentile=0,
    #                        color_map='jet', norm_by_percentile=True, norm_percs=(1, 99), symetric_colors=True):

    # args.atlas, args.mat_fname, args.mat_field, args.conditions, args.stat,
    # args.windows, args.labels_exclude, args.threshold, args.threshold_percentile,
    # args.color_map, args.norm_by_percentile, args.norm_percs)


    d = dict()
    data = sio.loadmat(args.mat_fname)[args.mat_field]
    d['labels'], d['locations'], d['hemis'] = calc_lables_info(subject)
    (d['con_colors'], d['con_indices'], d['con_names'],  d['con_values'], d['con_types'],
     d['data_max'], d['data_min']) = calc_connectivity(data, d['labels'], d['hemis'], args)
    # args.stat, args.conditions, args.windows, args.threshold,
    #     args.threshold_percentile, args.color_map, args.norm_by_percentile, args.norm_percs, args.symetric_colors)
    d['conditions'] = args.conditions
    np.savez(op.join(MMVT_DIR, subject, 'rois_con'), **d)


def calc_lables_meg_connectivity(subject, args):
    data, names = {}, {}
    LBL = op.join(MMVT_DIR, subject, 'labels_data_{}.npz')
    for hemi in utils.HEMIS:
        labels_output_fname = LBL.format(hemi)
        f = np.load(labels_output_fname)
        data[hemi] = np.squeeze(f['data'])
        names[hemi] = f['names']
    data = np.concatenate((data['lh'], data['rh']))
    names = np.concatenate((names['lh'], names['rh']))
    corr = np.zeros((data.shape[0], data.shape[0], data.shape[2]))
    for w in range(data.shape[2]):
        corr[:, :, w] = np.corrcoef(data[:, :, w])
        np.fill_diagonal(corr[:, :, w], 0)
    # np.sum(abs(np.mean(corr, 2)) > 0.9)
    args.threshold = 0.9
    args.threshold_percentile = 0
    args.symetric_colors = True
    corr = corr[:, :, :, np.newaxis]
    d = dict()
    args.labels_exclude = []
    d['labels'], d['locations'], d['hemis'] = calc_lables_info(subject, args, False, names)
    (_, d['con_indices'], d['con_names'], d['con_values'], d['con_types'],
     d['data_max'], d['data_min']) = calc_connectivity(corr, d['labels'], d['hemis'], args)
    output_fname = op.join(MMVT_DIR, subject, 'rois_con.npz')
    print('Saving results to {}'.format(output_fname))
    np.savez(output_fname, **d)
    return True


def calc_lables_info(subject, args, sorted_according_to_annot_file=True, sorted_labels_names=None):
    labels = lu.read_labels(
        subject, SUBJECTS_DIR, args.atlas, exclude=args.labels_exclude,
        sorted_according_to_annot_file=sorted_according_to_annot_file)
    if not sorted_labels_names is None:
        labels.sort(key=lambda x: np.where(sorted_labels_names == x.name)[0])
    locations = lu.calc_center_of_mass(labels, ret_mat=True) * 1000
    hemis = ['rh' if l.hemi == 'rh' else 'lh' for l in labels]
    return labels, locations, hemis


def calc_connectivity(data, labels, hemis, args):
    # stat, conditions, w, threshold=0, threshold_percentile=0, color_map='jet',
    #                         norm_by_percentile=True, norm_percs=(1, 99), symetric_colors=True):
    # import time
    M = data.shape[0]
    W = data.shape[2] if not 'windows' in args or args.windows == 0 else args.windows
    L = int((M * M + M) / 2 - M)
    con_indices = np.zeros((L, 2))
    con_values = np.zeros((L, W, len(args.conditions)))
    con_names = [None] * L
    con_type = np.zeros((L))
    lower_rec_indices = list(utils.lower_rec_indices(M))
    LRI = len(lower_rec_indices)
    for cond in range(len(args.conditions)):
        for w in range(W):
            # now = time.time()
            for ind, (i, j) in enumerate(lower_rec_indices):
                # utils.time_to_go(now, ind, LRI, LRI/10)
                if W > 1 and data.ndim == 4:
                    con_values[ind, w, cond] = data[i, j, w, cond]
                elif data.ndim > 2:
                    con_values[ind, w, cond] = data[i, j, cond]
                else:
                    con_values[ind, w, cond] = data[i, j]
    if len(args.conditions) > 1:
        stat_data = utils.calc_stat_data(con_values, args.stat)
    else:
        stat_data = np.squeeze(con_values)

    for ind, (i, j) in enumerate(utils.lower_rec_indices(M)):
        con_indices[ind, :] = [i, j]
        con_names[ind] = '{}-{}'.format(labels[i], labels[j])
        con_type[ind] = HEMIS_WITHIN if hemis[i] == hemis[j] else HEMIS_BETWEEN

    con_indices = con_indices.astype(np.int)
    con_names = np.array(con_names)
    data_max, data_min = utils.get_data_max_min(stat_data, args.norm_by_percentile, args.norm_percs)
    data_minmax = max(map(abs, [data_max, data_min]))
    if 'threshold_percentile' in args and args.threshold_percentile > 0:
        args.threshold = np.percentile(np.abs(stat_data), args.threshold_percentile)
    if args.threshold > data_minmax:
        raise Exception('threshold > abs(max(data)) ({})'.format(data_minmax))
    if args.threshold >= 0:
        indices = np.where(np.abs(stat_data) > args.threshold)[0]
        # con_colors = con_colors[indices]
        con_indices = con_indices[indices]
        con_names = con_names[indices]
        con_values = con_values[indices]
        con_type  = con_type[indices]
        stat_data = stat_data[indices]

    con_values = np.squeeze(con_values)
    if 'data_max' not in args and 'data_min' not in args or args.data_max == 0 and args.data_min == 0:
        if args.symetric_colors and np.sign(data_max) != np.sign(data_min):
            data_max, data_min = data_minmax, -data_minmax
    else:
        data_max, data_min = args.data_max, args.data_min
    print('data_max: {}, data_min: {}'.format(data_max, data_min))
    # con_colors = utils.mat_to_colors(stat_data, data_min, data_max, args.color_map)

    print(len(con_names))
    return None, con_indices, con_names, con_values, con_type, data_max, data_min


def main(subject, remote_subject_dir, args, flags):
    if utils.should_run(args, 'save_rois_connectivity'):
        flags['save_rois_connectivity'] = save_rois_connectivity(subject, args)

    if utils.should_run(args, 'save_electrodes_coh'):
        # todo: Add the necessary parameters
        flags['save_electrodes_coh'] = save_electrodes_coh(subject, args)

    return flags


def read_cmd_args(argv=None):
    import argparse
    from src.utils import args_utils as au
    parser = argparse.ArgumentParser(description='MMVT stim preprocessing')
    parser.add_argument('-c', '--conditions', help='conditions names', required=False, default='contrast', type=au.str_arr_type)
    parser.add_argument('-t', '--task', help='task name', required=False, default='')
    parser.add_argument('--mat_fname', help='matlab connection file name', required=False, default='')
    parser.add_argument('--mat_field', help='matlab connection field name', required=False, default='')
    parser.add_argument('--labels_exclude', help='rois to exclude', required=False, default='unknown,corpuscallosum',
                        type=au.str_arr_type)
    parser.add_argument('--bipolar', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--norm_by_percentile', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--norm_percs', help='', required=False, default='1,99', type=au.int_arr_type)
    parser.add_argument('--stat', help='', required=False, default=STAT_DIFF, type=int)
    parser.add_argument('--windows', help='', required=False, default=0, type=int)
    parser.add_argument('--t_max', help='', required=False, default=0, type=int)
    parser.add_argument('--threshold_percentile', help='', required=False, default=0, type=int)
    parser.add_argument('--threshold', help='', required=False, default=0, type=float)
    parser.add_argument('--color_map', help='', required=False, default='jet')
    parser.add_argument('--symetric_colors', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--data_max', help='', required=False, default=0, type=float)
    parser.add_argument('--data_min', help='', required=False, default=0, type=float)
    pu.add_common_args(parser)

    args = utils.Bag(au.parse_parser(parser, argv))
    if len(args.conditions) == 1:
        args.stat = STAT_AVG
    print(args)
    return args


if __name__ == '__main__':
    args = read_cmd_args()
    pu.run_on_subjects(args, main)
    print('finish!')