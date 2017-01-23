import os.path as op
import numpy as np
import scipy.io as sio
import glob
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
    output_fname = op.join(MMVT_DIR, subject, 'connectivity', 'electrodes.npz')
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


def save_rois_matlab_connectivity(subject, args):
    if not op.isfile(args.mat_fname):
        print("Can't find the input file {}!".format(args.mat_fname))
        return False
    d = dict()
    data = sio.loadmat(args.mat_fname)[args.mat_field]
    d['labels'], d['locations'], d['hemis'] = calc_lables_info(subject)
    (d['con_colors'], d['con_indices'], d['con_names'],  d['con_values'], d['con_types'],
     d['data_max'], d['data_min']) = calc_connectivity(data, d['labels'], d['hemis'], args)
    # args.stat, args.conditions, args.windows, args.threshold,
    #     args.threshold_percentile, args.color_map, args.norm_by_percentile, args.norm_percs, args.symetric_colors)
    d['conditions'] = args.conditions
    np.savez(op.join(MMVT_DIR, subject, 'rois_con'), **d)
    return op.isfile(op.join(MMVT_DIR, subject, 'rois_con'))


def calc_lables_connectivity(subject, args):
    import mne.connectivity

    data, names = {}, {}
    output_fname = op.join(MMVT_DIR, subject, 'connectivity', '{}_rois_con.npz'.format(args.connectivity_modality))
    output_fname_no_wins = op.join(MMVT_DIR, subject, 'connectivity',
                                   '{}_rois_con_static.npz'.format(args.connectivity_modality))
    con_vertices_fname = op.join(
        MMVT_DIR, subject, 'connectivity', '{}_rois_con_vertices.pkl'.format(args.connectivity_modality))
    utils.make_dir(op.join(MMVT_DIR, subject, 'connectivity'))
    conn_fol = op.join(MMVT_DIR, subject, args.connectivity_modality)
    labels_data_fnames = glob.glob(op.join(conn_fol, '*labels_data*.npz'))
    if len(labels_data_fnames) == 0:
        print("You don't have any connectivity data in {}, create it using the {} preproc".format(
            conn_fol, args.connectivity_modality))
        return False
    if len(labels_data_fnames) != 2:
        print("You have more than one type of {} connectivity data in {}, please pick one".format(
            args.connectivity_modality, conn_fol))
        print('For now, just move the other files somewhere else...')
        #todo: Write code that lets the user pick one
        return False
    labels_data_fname_template = labels_data_fnames[0].replace('rh', '{hemi}').replace('lh', '{hemi}')
    if not utils.both_hemi_files_exist(labels_data_fname_template):
        print("Can't find the labels data for both hemi in {}".format(conn_fol))
        return False
    for hemi in utils.HEMIS:
        labels_input_fname = labels_data_fname_template.format(hemi=hemi)
        f = np.load(labels_input_fname)
        data[hemi] = np.squeeze(f['data'])
        names[hemi] = f['names']
    data = np.concatenate((data['lh'], data['rh']))
    labels_names = np.concatenate((names['lh'], names['rh']))
    conditions = f['conditions'] if 'conditions' in f else ['rest']

    if data.ndim == 2:
        # No windows yet
        import math
        T = data.shape[1] # If this is fMRI data, the real T is T*tr
        windows_nun = math.floor((T - args.windows_length) / args.windows_shift + 1)
        windows = np.zeros((windows_nun, 2))
        for win_ind in range(windows_nun):
            windows[win_ind] = [win_ind * args.windows_shift, win_ind * args.windows_shift + args.windows_length]
    elif data.ndim == 3:
        windows_nun = data.shape[2]
    else:
        print('Wronge number of dims in data! Can be 2 or 3, not {}.'.format(data.ndim))
        return False

    conn = np.zeros((data.shape[0], data.shape[0], windows_nun))
    conn_no_wins = None
    if 'corr' in args.connectivity_method:
        for w in range(windows_nun):
            if data.ndim == 3:
                conn[:, :, w] = np.corrcoef(data[:, :, w])
            else:
                conn[:, :, w] = np.corrcoef(data[:, windows[w, 0]:windows[w, 1]])
            np.fill_diagonal(conn[:, :, w], 0)
            connectivity_method = 'Pearson corr'
    elif 'wpli2_debiased' in args.connectivity_method:
        conn_data = np.transpose(data, [2, 0, 1])
        conn = mne.connectivity.spectral_connectivity(conn_data, 'wpli2_debiased', sfreq=1000.0, fmin=5, fmax=100)
        connectivity_method = 'PLI'

    if 'cv' in args.connectivity_method:
        no_wins_connectivity_method = '{} CV'.format(connectivity_method)
        conn_no_wins = np.mean(np.abs(conn), 2) / np.nanstd(np.abs(conn), 2)
        dFC = np.nanmean(conn_no_wins, 1)
        lu.create_labels_coloring(subject, labels_names, dFC, 'pearson_corr_cv', norm_percs=(3, 99),
                           norm_by_percentile=True, colors_map='YlOrRd')
    conn = conn[:, :, :, np.newaxis]
    d = save_connectivity(subject, conn, connectivity_method, labels_names, conditions, output_fname, con_vertices_fname)
    if not conn_no_wins is None:
        conn_no_wins = conn_no_wins[:, :, np.newaxis]
        save_connectivity(subject, conn_no_wins, no_wins_connectivity_method, labels_names, conditions,
                          output_fname_no_wins, '', d['labels'], d['locations'], d['hemis'])


def save_connectivity(subject, conn, connectivity_method, labels_names, conditions, output_fname, con_vertices_fname='',
                      labels=None, locations=None, hemis=None):
    d = dict()
    d['conditions'] = conditions
    # args.labels_exclude = []
    if labels is None or locations is None or hemis is None:
        d['labels'], d['locations'], d['hemis'] = calc_lables_info(subject, args, False, labels_names)
    else:
        d['labels'], d['locations'], d['hemis'] = labels, locations, hemis
    (_, d['con_indices'], d['con_names'], d['con_values'], d['con_types'],
     d['data_max'], d['data_min']) = calc_connectivity(conn, d['labels'], d['hemis'], args)
    d['connectivity_method'] = connectivity_method
    print('Saving results to {}'.format(output_fname))
    np.savez(output_fname, **d)
    if con_vertices_fname != '':
        vertices, vertices_lookup = create_vertices_lookup(d['con_indices'], d['con_names'], d['labels'])
        utils.save((vertices, vertices_lookup), con_vertices_fname)
    return d


def create_vertices_lookup(con_indices, con_names, labels):
    from collections import defaultdict
    vertices, vertices_lookup = set(), defaultdict(list)
    for (i, j), conn_name in zip(con_indices, con_names):
        vertices.add(i)
        vertices.add(j)
        vertices_lookup[labels[i]].append(conn_name)
        vertices_lookup[labels[j]].append(conn_name)
    return np.array(list(vertices)), vertices_lookup


def calc_lables_info(subject, args, sorted_according_to_annot_file=True, sorted_labels_names=None):
    labels = lu.read_labels(
        subject, SUBJECTS_DIR, args.atlas, exclude=args.labels_exclude,
        sorted_according_to_annot_file=sorted_according_to_annot_file)
    if not sorted_labels_names is None:
        labels.sort(key=lambda x: np.where(sorted_labels_names == x.name)[0])
        # Remove labels that are not in sorted_labels_names
        labels = [l for l in labels if l.name in sorted_labels_names]
    locations = lu.calc_center_of_mass(labels, ret_mat=True) * 1000
    hemis = ['rh' if l.hemi == 'rh' else 'lh' for l in labels]
    labels_names = [l.name for l in labels]
    return labels_names, locations, hemis


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
        if stat_data.ndim >= 2:
            indices = np.where(np.max(abs(stat_data), axis=1) > args.threshold)[0]
        else:
            indices = np.where(abs(stat_data) > args.threshold)[0]
        # indices = np.where(np.abs(stat_data) > args.threshold)[0]
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
        flags['save_rois_connectivity'] = save_rois_matlab_connectivity(subject, args)

    if utils.should_run(args, 'save_electrodes_coh'):
        # todo: Add the necessary parameters
        flags['save_electrodes_coh'] = save_electrodes_coh(subject, args)

    if utils.should_run(args, 'calc_lables_connectivity'):
        flags['calc_lables_connectivity'] = calc_lables_connectivity(subject, args)

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
    parser.add_argument('--connectivity_method', help='', required=False, default='corr,cv', type=au.str_arr_type)
    parser.add_argument('--connectivity_modality', help='', required=False, default='fmri')
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
    parser.add_argument('--windows_length', help='', required=False, default=2000, type=int)
    parser.add_argument('--windows_shift', help='', required=False, default=500, type=int)
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