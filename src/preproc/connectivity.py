import os.path as op
import numpy as np
import scipy.io as sio
import mne.connectivity
import glob
import traceback
import shutil
import itertools
from scipy.spatial.distance import cdist
import fnmatch
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils import utils
from src.utils import preproc_utils as pu
from src.utils import labels_utils as lu
from src.preproc import fMRI as fmri

SUBJECTS_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()
LINKS_DIR = utils.get_links_dir()
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
FMRI_DIR = utils.get_link_dir(LINKS_DIR, 'fMRI')
ELECTRODES_DIR = utils.get_link_dir(LINKS_DIR, 'electrodes')

STAT_AVG, STAT_DIFF = range(2)
STAT_NAME = {STAT_DIFF: 'diff', STAT_AVG: 'avg'}
HEMIS_WITHIN, HEMIS_BETWEEN = range(2)
ROIS_TYPE, ELECTRODES_TYPE = range(2)

#todo: Add the necessary parameters
# args.conditions, args.mat_fname, args.t_max, args.stat, args.threshold)
def calc_electrodes_coh(subject, args):
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


def calc_rois_matlab_connectivity(subject, args):
    from src.utils import matlab_utils as matu
    if not op.isfile(args.mat_fname):
        print("Can't find the input file {}!".format(args.mat_fname))
        return False
    d = dict()
    mat_file = sio.loadmat(args.mat_fname)
    data = mat_file[args.mat_field]
    sorted_labels_names = None
    if args.sorted_labels_names_field != '':
        sorted_labels_names = matu.matlab_cell_str_to_list(mat_file[args.sorted_labels_names_field])
    d['labels'], d['locations'], d['hemis'] = calc_lables_info(subject, args, sorted_labels_names=sorted_labels_names)
    (d['con_colors'], d['con_indices'], d['con_names'],  d['con_values'], d['con_types'],
     d['data_max'], d['data_min']) = calc_connectivity(data, d['labels'], d['hemis'], args)
    # args.stat, args.conditions, args.windows, args.threshold,
    #     args.threshold_percentile, args.color_map, args.norm_by_percentile, args.norm_percs, args.symetric_colors)
    d['conditions'] = args.conditions
    output_fname = op.join(MMVT_DIR, subject, 'connectivity', 'rois_con.npy')
    np.savez(output_fname, **d)
    vertices, vertices_lookup = create_vertices_lookup(d['con_indices'], d['con_names'], d['labels'])
    con_vertices_fname = op.join(
        MMVT_DIR, subject, 'connectivity', '{}_vertices.pkl'.format('rois'))
    utils.save((vertices, vertices_lookup), con_vertices_fname)
    return op.isfile(output_fname) and con_vertices_fname


def calc_lables_connectivity(subject, labels_extract_mode, args):

    def get_output_fname(connectivity_method, labels_extract_mode='', identifier=''):
        comps_num = '_{}'.format(labels_extract_mode.split('_')[1]) if labels_extract_mode.startswith('pca_') else ''
        return op.join(MMVT_DIR, subject, 'connectivity', '{}_{}{}{}.npz'.format(
            args.connectivity_modality, identifier, connectivity_method, comps_num))

    def get_output_mat_fname(connectivity_method, labels_extract_mode='', identifier=''):
        comps_num = '_{}'.format(labels_extract_mode.split('_')[1]) if labels_extract_mode.startswith('pca_') else ''
        return op.join(MMVT_DIR, subject, 'connectivity', '{}_{}{}{}.npy'.format(
            args.connectivity_modality, identifier, connectivity_method, comps_num))

    def backup(fname):
        if args.backup_existing_files and op.isfile(fname):
            backup_fname = utils.add_str_to_file_name(fname, '_backup')
            shutil.copy(fname, backup_fname)

    data, names = {}, {}
    identifier = '{}_'.format(args.identifier) if args.identifier != '' else ''
    if 'cv' in args.connectivity_method:
        static_output_fname = op.join(MMVT_DIR, subject, 'connectivity', '{}_{}{}_{}_cv_{}.npz'.format(
            args.connectivity_modality, identifier, args.atlas, args.connectivity_method[0], labels_extract_mode))
        static_output_mat_fname = op.join(MMVT_DIR, subject, 'connectivity', '{}_{}{}_{}_cv_{}.npz'.format(
            args.connectivity_modality, identifier, args.atlas, args.connectivity_method[0], labels_extract_mode))
        static_mean_output_mat_fname = op.join(MMVT_DIR, subject, 'connectivity', '{}_{}{}_{}_cv_mean_{}.npz'.format(
            args.connectivity_modality, identifier, args.atlas, args.connectivity_method[0], labels_extract_mode))
    conn_mean_mat_fname = op.join(MMVT_DIR, subject, 'connectivity', '{}_{}{}_{}_{}_mean.npy'.format(
            args.connectivity_modality, identifier, args.atlas, args.connectivity_method[0], labels_extract_mode))
    labels_avg_output_fname = op.join(MMVT_DIR, subject, 'connectivity', '{}_{}{}_{}_{}_labels_avg.npz'.format(
        args.connectivity_modality, identifier, args.connectivity_method[0], args.atlas, '{hemi}'))
    con_vertices_fname = op.join(
        MMVT_DIR, subject, 'connectivity', '{}_vertices.pkl'.format(args.connectivity_modality))
    utils.make_dir(op.join(MMVT_DIR, subject, 'connectivity'))
    conn_fol = op.join(MMVT_DIR, subject, args.connectivity_modality)
    if args.labels_data_name != '':
        labels_data_fname = op.join(conn_fol, args.labels_data_name)
    else:
        labels_data_fname = utils.select_one_file(glob.glob(op.join(
            conn_fol, '{}*labels_data*_{}_*{}_rh.npz'.format(args.identifier, args.atlas, labels_extract_mode))))
    if len(labels_data_fname) == 0:
        modalities_fols_dic = dict(meg=MEG_DIR, fmri=FMRI_DIR, electrodes=ELECTRODES_DIR)
        conn_fol = op.join(modalities_fols_dic[args.connectivity_modality], subject)
        labels_data_fname = [f for f in glob.glob(op.join(conn_fol, '*labels_data*.npz')) if 'norm' not in utils.namebase(f)]
    if len(labels_data_fname) == 0:
        print("You don't have any connectivity data ({}) in {}, create it using the {} preproc".format(
            '*labels_data_{}_{}_?h.npz'.format(args.atlas, labels_extract_mode), conn_fol, args.connectivity_modality))
        return False
    # if len(labels_data_fname) != 2:
    #     print("You have more than one type of {} connectivity data in {}, please pick one".format(
    #         args.connectivity_modality, conn_fol))
    #     print(labels_data_fname)
    #     print('For now, just move the other files somewhere else...')
    #     #todo: Write code that lets the user pick one
    #     return False
    if '{hemi}' not in labels_data_fname:
        labels_data_fname_template = labels_data_fname.replace('rh', '{hemi}').replace('lh', '{hemi}')
    else:
        labels_data_fname_template = labels_data_fname
    if not utils.both_hemi_files_exist(labels_data_fname_template):
        print("Can't find the labels data for both hemi in {}".format(conn_fol))
        return False
    for hemi in utils.HEMIS:
        labels_input_fname = labels_data_fname_template.format(hemi=hemi)
        f = np.load(labels_input_fname)
        print('Loading {} ({})'.format(labels_input_fname, utils.file_modification_time(labels_input_fname)))
        data[hemi] = np.squeeze(f['data'])
        zeros_num = len(np.where(np.sum(data[hemi], 1) == 0)[0])
        if zeros_num > 0:
            print('{} has {}/{} flat time series!'.format(hemi, zeros_num, data[hemi].shape[0]))
        names[hemi] = f['names']

    data = np.concatenate((data['lh'], data['rh']))
    tmin = args.tmin if args.tmin is not None else 0
    tmax = args.tmax if args.tmax is not None else data.shape[1]
    data = data[:, tmin:tmax]
    labels_names = np.concatenate((names['lh'], names['rh']))
    labels_fname = op.join(MMVT_DIR, subject, 'connectivity', 'labels_names.npy')
    np.save(labels_fname, labels_names)
    labels_indices = np.array([ind for ind,l in enumerate(labels_names) if not np.any(
        [e in l for e in args.labels_exclude])])
    if len(labels_indices) < len(labels_names):
        labels_names = labels_names[labels_indices]
        data = data[labels_indices]

    conditions = f['conditions'] if 'conditions' in f else ['rest']
    labels_hemi_indices = {}
    for hemi in utils.HEMIS:
        labels_hemi_indices[hemi] = np.array([ind for ind,l in enumerate(labels_names) if l in names[hemi]])

    subs_fname = op.join(
        MMVT_DIR, subject, 'fmri', 'subcorticals_{}.npz'.format(labels_extract_mode))
    if args.calc_subs_connectivity and op.isfile(subs_fname):
        print('Loading subs data from {}'.format(subs_fname))
        f = np.load(subs_fname)
        subs_data = np.squeeze(f['data'])
        subs_names = f['names']
        labels_subs_indices = np.arange(len(labels_names), len(labels_names) + len(subs_names))
        data = np.concatenate((data, subs_data))
        labels_names = np.concatenate((labels_names, subs_names))
        subs_avg_output_fname = op.join(MMVT_DIR, subject, 'connectivity', '{}_{}_subs_avg.npz'.format(
            args.connectivity_modality, args.connectivity_method[0]))
    else:
        labels_subs_indices = []

    if args.windows_num == 1 and data.ndim == 3:
        data = np.mean(data, axis=2)

    # Check this code!!!
    if data.ndim == 3 and data.shape[2] == len(conditions):
        data = np.diff(data, axis=2).squeeze()
    if data.ndim == 2 or labels_extract_mode.startswith('pca_') and data.ndim == 3:
        # No windows yet
        import math
        T = data.shape[1] # If this is fMRI data, the real T is T*tr
        if args.windows_length == 0:
            args.windows_length = T
            windows_num = 1
        else:
            windows_num = math.floor((T - args.windows_length) / args.windows_shift + 1)
        windows = np.zeros((windows_num, 2))
        for win_ind in range(windows_num):
            windows[win_ind] = [win_ind * args.windows_shift, win_ind * args.windows_shift + args.windows_length]
        windows = windows.astype(np.int)
    elif data.ndim == 3:
        windows_num = data.shape[2]
    else:
        print('Wronge number of dims in data! Can be 2 or 3, not {}.'.format(data.ndim))
        return False

    if windows_num == 1:
        identifier = '{}static_'.format(identifier) if identifier != '' else 'static_'
    if args.max_windows_num is not None:
        windows_num = min(args.max_windows_num, windows_num)
    output_fname = get_output_fname(args.connectivity_method[0], labels_extract_mode, identifier)
    output_mat_fname = get_output_mat_fname(args.connectivity_method[0], labels_extract_mode, identifier)
    static_conn = None
    if op.isfile(output_mat_fname) and not args.recalc_connectivity:
        conn = np.load(output_mat_fname)
        if conn.shape[0] != data.shape[0]:
            args.recalc_connectivity = True
        if 'corr' in args.connectivity_method:
            connectivity_method = 'Pearson corr'
        elif 'pli' in args.connectivity_method:
            connectivity_method = 'PLI'
        elif 'mi' in args.connectivity_method:
            connectivity_method = 'MI'
        elif 'coherence' in args.connectivity_method:
            connectivity_method = 'COH'
    if not op.isfile(output_mat_fname) or args.recalc_connectivity:
        if 'corr' in args.connectivity_method:
            conn = np.zeros((data.shape[0], data.shape[0], windows_num))
            if labels_extract_mode.startswith('pca_'):
                comps_num = int(labels_extract_mode.split('_')[1])
                dims = (data.shape[0], data.shape[0], windows_num, comps_num * 2, comps_num * 2)
                conn = np.zeros(dims)
            if data.ndim == 3 and not labels_extract_mode.startswith('pca_') or data.ndim == 4:
                for w in range(windows_num):
                    conn[:, :, w] = np.corrcoef(data[:, :, w])
            else:
                if not labels_extract_mode.startswith('pca_'):
                    for w in range(windows_num):
                        conn[:, :, w] = np.corrcoef(data[:, windows[w, 0]:windows[w, 1]])
                        np.fill_diagonal(conn[:, :, w], 0)
                else:
                    params = []
                    for w in range(windows_num):
                        w1, w2 = int(windows[w, 0]), int(windows[w, 1])
                        data_w = data[:, w1:w2]
                        params.append((data_w, comps_num))
                    # params = [(data, w, windows, comps_num) for w in range(windows_num)]
                    chunks = utils.chunks(list(enumerate(params)), windows_num / args.n_jobs)
                    results = utils.run_parallel(_corr_matrix_parallel, chunks, args.n_jobs)
                    for chunk in results:
                        for w, con in chunk.items():
                            conn[:, :, w] = con
            if conn.shape[2] == 1:
                conn = conn.squeeze()
            backup(output_mat_fname)
            print('Saving {}, {}'.format(output_mat_fname, conn.shape))
            np.save(output_mat_fname, conn)
            connectivity_method = 'Pearson corr'

        elif 'pli' in args.connectivity_method:
            conn = np.zeros((data.shape[0], data.shape[0], windows_num))
            if data.ndim == 3:
                conn_data = np.transpose(data, [2, 1, 0])
            elif data.ndim == 2:
                conn_data = np.zeros((windows_num, data.shape[0], args.windows_length))
                for w in range(windows_num):
                    conn_data[w] = data[:, windows[w, 0]:windows[w, 1]]
            indices = np.array_split(np.arange(windows_num), args.n_jobs)
            # chunks = utils.chunks(list(enumerate(conn_data)), windows_num / args.n_jobs)
            chunks = [(conn_data[chunk_indices], chunk_indices, len(labels_names), args.windows_length)
                  for chunk_indices in indices]
            results = utils.run_parallel(_pli_parallel, chunks, args.n_jobs)
            for chunk in results:
                for w, con in chunk.items():
                    conn[:, :, w] = con
            backup(output_mat_fname)
            np.save(output_mat_fname, conn)
            connectivity_method = 'PLI'

        elif 'coherence' in args.connectivity_method:
            if args.bands == '':
                args.bands = dict(theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55], high_gamma=[65, 200])
            conn = np.zeros((data.shape[0], data.shape[0], len(args.bands))) # What about windows_num?
            epochs = mne.read_epochs(args.epochs_fname)
            indices = np.array_split(np.arange(len(args.bands)), args.n_jobs)
            # for iband, (band, (fmin, fmax)) in enumerate(args.bands.items()):
            chunks = [(epochs, chunk_indices, args.sfreq, args.bands)
                      for chunk_indices in indices]
            results = utils.run_parallel(_coh_parallel, chunks, args.n_jobs)
            for chunk in results:
                for iband, con in chunk.items():
                    # todo: fix this
                    conn[:, :, iband] = con

        elif 'mi' in args.connectivity_method or 'mi_vec' in args.connectivity_method:
            conn = np.zeros((data.shape[0], data.shape[0], windows_num))
            corr_fname = get_output_mat_fname('corr', labels_extract_mode)
            if op.isfile(corr_fname):
                corr = np.load(get_output_mat_fname('corr', labels_extract_mode))
            if not op.isfile(corr_fname) or corr.shape[0] != data.shape[0]:
                new_args = utils.Bag(args.copy())
                new_args.connectivity_method = ['corr']
                calc_lables_connectivity(subject, labels_extract_mode, new_args)
                corr = np.load(get_output_mat_fname('corr', labels_extract_mode))
            if 'mi' in args.connectivity_method or 'mi_vec' in args.connectivity_method and corr.ndim == 3:
                conn_fname = get_output_mat_fname('mi', labels_extract_mode)
                if op.isfile(conn_fname):
                    conn = np.load(conn_fname)
                if not op.isfile(conn_fname) or conn.shape[0] != data.shape[0]:
                    conn = np.zeros(corr.shape)
                    params = [(corr[:, :, w]) for w in range(windows_num)]
                    chunks = utils.chunks(list(enumerate(params)), windows_num / args.n_jobs)
                    results = utils.run_parallel(_mi_parallel, chunks, args.n_jobs)
                    for chunk in results:
                        for w, con in chunk.items():
                            conn[:, :, w] = con
                    backup(conn_fname)
                    np.save(conn_fname, conn)
            if 'mi_vec' in args.connectivity_method and corr.ndim == 5:
                conn_fname = get_output_mat_fname('mi_vec', labels_extract_mode)
                if op.isfile(conn_fname):
                    conn = np.load(conn_fname)
                if not op.isfile(conn_fname) or conn.shape[0] != data.shape[0]:
                    # comps_num = int(labels_extract_mode.split('_')[1])
                    dims = (data.shape[0], data.shape[0], windows_num)
                    conn = np.zeros(dims)
                    params = [(corr[:, :, w]) for w in range(windows_num)]
                    chunks = utils.chunks(list(enumerate(params)), windows_num / args.n_jobs)
                    results = utils.run_parallel(_mi_vec_parallel, chunks, args.n_jobs)
                    for chunk in results:
                        for w, con in chunk.items():
                            conn[:, :, w] = con
                    backup(conn_fname)
                    np.save(conn_fname, conn)
            connectivity_method = 'MI'

    if 'corr' in args.connectivity_method or 'pli' in args.connectivity_method and \
            not utils.both_hemi_files_exist(labels_avg_output_fname):
        avg_per_label = np.mean(conn, 0)
        abs_minmax = utils.calc_abs_minmax(conn)
        for hemi in utils.HEMIS:
            inds = labels_hemi_indices[hemi]
            backup(labels_avg_output_fname.format(hemi=hemi))
            np.savez(labels_avg_output_fname.format(hemi=hemi), data=avg_per_label[inds], names=labels_names[inds],
                     conditions=conditions, minmax=[-abs_minmax, abs_minmax])
        if len(labels_subs_indices) > 0:
            inds = labels_subs_indices
            backup(subs_avg_output_fname)
            np.savez(subs_avg_output_fname, data=avg_per_label[inds], names=labels_names[inds],
                     conditions=conditions, minmax=[-abs_minmax, abs_minmax])
    if 'cv' in args.connectivity_method:
        no_wins_connectivity_method = '{} CV'.format(args.connectivity_method)
        # todo: check why if it's not always True, the else fails
        if True:#not op.isfile(static_output_mat_fname):
            conn_std = np.nanstd(conn, 2)
            static_conn = conn_std / np.mean(np.abs(conn), 2)
            if np.ndim(static_conn) == 2:
                np.fill_diagonal(static_conn, 0)
            elif np.ndim(static_conn) == 4:
                L, M = static_conn.shape[2:]
                for i, j in itertools.product(range(), range(4)):
                    np.fill_diagonal(static_conn[:, :, i, j], 0)
            backup(static_output_mat_fname)
            print('Saving {}, {}'.format(static_output_mat_fname, static_conn.shape))
            np.savez(static_output_mat_fname, static_conn=static_conn, conn_std=conn_std)
            # static_conn[np.isnan(static_conn)] = 0
        else:
            d = np.load(static_output_mat_fname)
            static_conn = d['static_conn']
            conn_std = d['conn_std']
        static_con_fig_fname = utils.change_fname_extension(static_output_mat_fname, 'png')
        if not op.isfile(static_con_fig_fname) and args.do_plot_static_conn:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cax = ax.matshow(static_conn)
            fig.colorbar(cax)
            plt.title('{} Cv'.format(connectivity_method))
            plt.savefig(static_con_fig_fname)
            plt.close()
        if not op.isfile(static_mean_output_mat_fname):
            dFC = np.nanmean(static_conn, 1)
            std_mean = np.nanmean(conn_std, 1)
            stat_conn = np.nanmean(np.abs(conn), 1)
            backup(static_mean_output_mat_fname)
            print('Saving {}, {}'.format(static_mean_output_mat_fname, std_mean.shape))
            np.savez(static_mean_output_mat_fname, dFC=dFC, std_mean=std_mean, stat_conn=stat_conn)
            lu.create_labels_coloring(subject, labels_names, dFC, '{}_{}_cv_mean'.format(
                args.connectivity_modality, args.connectivity_method[0]), norm_percs=(1, 99), norm_by_percentile=True,
                colors_map='YlOrRd')
    if windows_num > 1 and conn.ndim == 3 and not op.isfile(conn_mean_mat_fname):
        mean_conn = np.mean(conn, 2)
        np.save(conn_mean_mat_fname, mean_conn)
    if not args.save_mmvt_connectivity:
        return True
    if conn.ndim == 3:
        conn = conn[:, :, :, np.newaxis]
    elif conn.ndim == 2:
        conn = conn[:, :, np.newaxis]
    else:
        raise Exception('Wrong number of dims!')
    d = save_connectivity(subject, conn, args.connectivity_method, ROIS_TYPE, labels_names, conditions, output_fname, args,
                          con_vertices_fname)
    ret = op.isfile(output_fname)
    if not static_conn is None:
        static_conn = static_conn[:, :, np.newaxis]
        save_connectivity(subject, static_conn, no_wins_connectivity_method, ROIS_TYPE, labels_names, conditions,
                          static_output_fname, args, '', d['labels'], d['locations'], d['hemis'])
        ret = ret and op.isfile(static_output_fname)

    return ret


def pli(data, channels_num, window_length):
    try:
        from scipy.signal import hilbert
        if data.shape[0] != channels_num:
            data = data.T
        data_hil = hilbert(data)
        # if data_hil.shape != (channels_num, window_length):
        #     raise Exception('PLI: Wrong dimentions!')
        m = np.zeros((channels_num, channels_num))
        for i in range(channels_num):
            for j in range(channels_num):
                if i < j:
                    m[i, j] = abs(np.mean(np.sign(np.imag(data_hil[i] / data_hil[j]))))
                    # m[i, j] = abs(np.mean(np.sign(np.imag(data_hil[:, i] / data_hil[:, j]))))
        return m + m.T
    except:
        print(traceback.format_exc())
        return None


def _pli_parallel(p):
    res = {}
    conn_data, indices, channels_num, window_length = p
    # for window_ind, window in windows_chunk:
    for window_ind, window in zip(indices, conn_data):
        print('PLI: Window ind {}'.format(window_ind))
        pli_val = pli(window, channels_num, window_length)
        if not pli_val is None:
            res[window_ind] = pli_val
        else:
            print('Error in PLI! windowsw ind {}'.format(window_ind))
    return res


def _coh_parallel(p):
    res = {}
    conn_data, indices, sfreq, bands = p
    # for window_ind, window in zip(indices, conn_data):
    for iband, (fmin, fmax) in enumerate(bands.values()):
        print('COH: Window ind {}'.format(fmin, fmax))
        coh_val = coherence(conn_data, sfreq, fmin, fmax)
        if not coh_val is None:
            res[iband] = coh_val
        else:
            print('Error in COH! window_ind: {}'.format(iband))
    return res


def coherence(data, sfreq, fmin, fmax):
    from mne.connectivity import spectral_connectivity
    # Shouldn't be epochs, should be sources
    # https://martinos.org/mne/stable/auto_examples/connectivity/plot_mne_inverse_label_connectivity.html#sphx-glr-auto-examples-connectivity-plot-mne-inverse-label-connectivity-py
    con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
        data, method='coh', mode='multitaper', sfreq=sfreq, fmin=fmin,
        fmax=fmax, faverage=True, mt_adaptive=True, n_jobs=1)
    return con


def corr_matrix(data, comps_num):
    corr = np.zeros((data.shape[0], data.shape[0], comps_num * 2, comps_num * 2))
    # w1, w2 = int(windows[w, 0]), int(windows[w, 1])
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            if i < j:
                corr[i, j] = np.corrcoef(data[i].T, data[j].T)
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            if i > j:
                corr[i, j] = corr[j, i]
    return corr


def _corr_matrix_parallel(windows_chunk):
    res = {}
    for window_ind, window in windows_chunk:
        print('_corr_matrix_parallel: window ind {}'.format(window_ind))
        data_w, comps_num = window
        res[window_ind] = corr_matrix(data_w, comps_num)
    return res


def mi(conn_w):
    nch = conn_w.shape[0]
    conn = np.zeros((nch, nch))
    for i in range(nch):
        for j in range(nch):
            if i < j:
                conn[i, j] = -0.5 * np.log(1 - conn_w[i, j] ** 2)
    conn = conn + conn.T
    return conn


def _mi_parallel(windows_chunk):
    res = {}
    for window_ind, corr_w in windows_chunk:
        print('_mi_parallel: window ind {}'.format(window_ind))
        res[window_ind] = mi(corr_w)
    return res


def mi_vec(corr_w):
    nch = corr_w.shape[0]
    conn = np.zeros((nch, nch))
    for i in range(nch):
        for j in range(nch):
            if i < j:
                conn[i, j] = -0.5 * np.log(np.linalg.norm(
                    np.eye(corr_w.shape[3]) - corr_w[i, j] * corr_w[i, j].T))
    conn = conn + conn.T
    return conn


def _mi_vec_parallel(windows_chunk):
    res = {}
    for window_ind, corr_w in windows_chunk:
        print('_mi_vec_parallel: window ind {}'.format(window_ind))
        res[window_ind] = mi_vec(corr_w)
    return res


@utils.tryit()
def save_connectivity(subject, conn, connectivity_method, obj_type, labels_names, conditions, output_fname, args,
                      con_vertices_fname='', labels=None, locations=None, hemis=None):
    d = dict()
    d['conditions'] = conditions
    # args.labels_exclude = []
    if labels is None or locations is None or hemis is None:
        if obj_type == ROIS_TYPE:
            d['labels'], d['locations'], d['hemis'] = calc_lables_info(subject, args, False, labels_names)
        elif obj_type == ELECTRODES_TYPE:
            bipolar = '-' in labels_names[0]
            d['labels'], d['locations'] = get_electrodes_info(subject, bipolar)
            assert(np.all(np.array(d['labels']) == labels_names))
            d['hemis'] = []
            groups_hemis = utils.load(op.join(MMVT_DIR, subject, 'electrodes', 'sorted_groups.pkl'))
            for elc_name in d['labels']:
                group_name = utils.elec_group(elc_name, bipolar)
                d['hemis'].append('rh' if group_name in groups_hemis['rh'] else 'lh')
            wrong_assigments = [(name, hemi) for name, hemi in zip(labels_names, d['hemis']) if name[0].lower()!=hemi[0]]
            if len(wrong_assigments) > 0:
                print('hemis wrong assigments:')
                print(wrong_assigments)
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
        subject, SUBJECTS_DIR, args.atlas, exclude=tuple(args.labels_exclude),
        sorted_according_to_annot_file=sorted_according_to_annot_file)
    if not sorted_labels_names is None:
        sorted_labels_names_fix = []
        org_delim, org_pos, label, label_hemi = lu.get_hemi_delim_and_pos(labels[0].name)
        for label_name in sorted_labels_names:
            delim, pos, label, label_hemi = lu.get_hemi_delim_and_pos(label_name)
            label_fix = lu.build_label_name(org_delim, org_pos, label, label_hemi)
            sorted_labels_names_fix.append(label_fix)
        sorted_labels_names = sorted_labels_names_fix
        labels.sort(key=lambda x: np.where(sorted_labels_names == x.name)[0])
        # Remove labels that are not in sorted_labels_names
        labels = [l for l in labels if l.name in sorted_labels_names]
    locations = lu.calc_center_of_mass(labels, ret_mat=True, find_vertice=True) * 1000
    hemis = ['rh' if l.hemi == 'rh' else 'lh' for l in labels]
    labels_names = [l.name for l in labels]
    return labels_names, locations, hemis


def calc_connectivity(data, labels, hemis, args):
    # stat, conditions, w, threshold=0, threshold_percentile=0, color_map='jet',
    #                         norm_by_percentile=True, norm_percs=(1, 99), symetric_colors=True):
    # import time
    M = data.shape[0]
    if data.ndim == 2:
        data = data[:, :, np.newaxis]
    W = data.shape[2] if 'windows' not in args or args.windows == 0 else args.windows
    L = int((M * M + M) / 2 - M)
    con_indices = np.zeros((L, 2))
    con_values = np.zeros((L, W, len(args.conditions)))
    con_names = [None] * L
    con_type = np.zeros((L))
    lower_rec_indices = list(utils.lower_rec_indices(M))
    # LRI = len(lower_rec_indices)
    for cond in range(len(args.conditions)):
        for w in range(W):
            # now = time.time()
            if W > 1 and data.ndim == 4:
                con_values[:, w, cond] = [data[i, j, w, cond] for i, j in lower_rec_indices]
            elif data.ndim > 2:
                con_values[:, w, cond] = [data[i, j, cond] for i, j in lower_rec_indices]
            else:
                con_values[:, w, cond] = [data[i, j] for i, j in lower_rec_indices]
            # for ind, (i, j) in enumerate(lower_rec_indices):
            #     # utils.time_to_go(now, ind, LRI, LRI/10)
            #     if W > 1 and data.ndim == 4:
            #         con_values[ind, w, cond] = data[i, j, w, cond]
            #     elif data.ndim > 2:
            #         con_values[ind, w, cond] = data[i, j, cond]
            #     else:
            #         con_values[ind, w, cond] = data[i, j]
    if len(args.conditions) > 1:
        stat_data = utils.calc_stat_data(con_values, args.stat)
    else:
        stat_data = np.squeeze(con_values)

    con_indices = np.array(lower_rec_indices)
    for ind, (i, j) in enumerate(utils.lower_rec_indices(M)):
        try:
            # con_indices[ind, :] = [i, j]
            con_names[ind] = '{}-{}'.format(labels[i], labels[j])
            con_type[ind] = HEMIS_WITHIN if hemis[i] == hemis[j] else HEMIS_BETWEEN
        except:
            print('error in calc_connectivity!')
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
        if args.symetric_colors and np.sign(data_max) != np.sign(data_min) and data_min != 0 and data_max != 0:
            data_max, data_min = data_minmax, -data_minmax
    else:
        data_max, data_min = args.data_max, args.data_min
    print('data_max: {}, data_min: {}'.format(data_max, data_min))
    # con_colors = utils.mat_to_colors(stat_data, data_min, data_max, args.color_map)

    print(len(con_names))
    return None, con_indices, con_names, con_values, con_type, data_max, data_min


def calc_electrodes_rest_connectivity(subject, args):

    def get_electrode_conn_data():
        data_fnames, meta_data_fnames = get_fnames()
        if len(meta_data_fnames) == 1 and len(data_fnames) == 1:
            conn_data = np.load(data_fnames[0])
            conn_data = np.transpose(conn_data, [2, 0, 1])
        else:
            electrodes_names_fname = op.join(ELECTRODES_DIR, subject, 'electrodes.npy')
            data_fname = op.join(ELECTRODES_DIR, subject, 'data.npy')
            if op.isfile(electrodes_names_fname) and op.isfile(data_fname):
                conn_data = np.load(data_fname)
            else:
                raise Exception("Electrodes data can't be found!")
        return conn_data

    def get_electrodes_names():
        data_fnames, meta_data_fnames = get_fnames()
        if len(meta_data_fnames) == 1 and len(data_fnames) == 1:
            d = np.load(meta_data_fnames[0])
            electrodes_names = d['names'] # conditions=conditions, times=times)
        else:
            electrodes_names_fname = op.join(ELECTRODES_DIR, subject, 'electrodes.npy')
            if op.isfile(electrodes_names_fname):
                electrodes_names = np.load(electrodes_names_fname)
            else:
                raise Exception("Electrodes names can't be found!")
        return electrodes_names

    def get_fnames():
        fol = op.join(MMVT_DIR, subject, 'electrodes')
        meta_data_fnames = glob.glob(op.join(fol, 'electrodes{}_meta_data*.npz'.format(
            '_bipolar' if args.bipolar else '', STAT_NAME[args.stat])))
        data_fnames = glob.glob(op.join(fol, 'electrodes{}_data*.npy'.format(
            '_bipolar' if args.bipolar else '', STAT_NAME[args.stat])))
        return data_fnames, meta_data_fnames


    args.connectivity_modality = 'electrodes'
    output_mat_fname = op.join(MMVT_DIR, subject, 'connectivity', '{}.npy'.format(args.connectivity_modality))
    utils.make_dir(op.join(MMVT_DIR, subject, 'connectivity'))
    output_fname = op.join(MMVT_DIR, subject, 'connectivity', '{}.npz'.format(args.connectivity_modality))
    output_fname_static = op.join(MMVT_DIR, subject, 'connectivity',
                                   '{}_static.npz'.format(args.connectivity_modality))
    con_vertices_fname = op.join(
        MMVT_DIR, subject, 'connectivity', '{}_vertices.pkl'.format(args.connectivity_modality))

    if not op.isfile(output_mat_fname):
        conn_data = get_electrode_conn_data()
        windows_num, E, windows_length = conn_data.shape

        if windows_num == 1:
            from mne import filter
            conn_data = conn_data[0]
            conn_data = filter.filter_data(conn_data, args.sfreq, args.fmin, args.fmax)
            # plt.figure()
            # plt.psd(conn_data, Fs=args.sfreq)
            conn_data = conn_data[np.newaxis, :, :]

            import math
            T = windows_length
            windows_num = math.floor((T - args.windows_length) / args.windows_shift + 1)
            data_winodws = np.zeros((windows_num, E, args.windows_length))
            for w in range(windows_num):
                data_winodws[w] = conn_data[0, :, w * args.windows_shift:w * args.windows_shift + args.windows_length]
            conn_data = data_winodws
        if args.max_windows_num != 0:
            windows_num = min(args.max_windows_num, windows_num)

        # pli_wins = 1
        conn = np.zeros((E, E, windows_num))
        conn_data = conn_data[:windows_num]
        chunks = utils.chunks(list(enumerate(conn_data)), windows_num / args.n_jobs)
        results = utils.run_parallel(_pli_parallel, chunks, args.n_jobs)
        for chunk in results:
            for w, con in chunk.items():
                conn[:, :, w] = con

        # five_cycle_freq = 5. * args.sfreq / float(conn_data.shape[2])
        # for w in range(windows_num - pli_wins):
        #     window_conn_data = conn_data[w:w+pli_wins, :, :]
        #     con, _, _, _, _ = mne.connectivity.spectral_connectivity(
        #         window_conn_data, 'pli2_unbiased', sfreq=args.sfreq, fmin=args.fmin, fmax=args.fmax,
        #         n_jobs=args.n_jobs)
        #     con = np.mean(con, 2) # Over freqs
        #     conn[:, :, w] = con + con.T

        np.save(output_mat_fname, conn)
    else:
        conn = np.load(output_mat_fname)

    connectivity_method = 'PLI'
    no_wins_connectivity_method = '{} CV'.format(connectivity_method)
    static_conn = np.nanstd(conn, 2) / np.mean(np.abs(conn), 2)
    np.fill_diagonal(static_conn, 0)
    conn = conn[:, :, :, np.newaxis]
    conditions = ['rest']
    electrodes_names = get_electrodes_names()
    d = save_connectivity(subject, conn, connectivity_method, ELECTRODES_TYPE, electrodes_names, conditions, output_fname, args,
                          con_vertices_fname)
    ret = op.isfile(output_fname)
    if not static_conn is None:
        static_conn = static_conn[:, :, np.newaxis]
        save_connectivity(subject, static_conn, no_wins_connectivity_method, ELECTRODES_TYPE, electrodes_names, conditions,
                          output_fname_static, args, '', d['labels'], d['locations'], d['hemis'])
        ret = ret and op.isfile(output_fname_static)
    return ret


def calc_seed_corr(subject, atlas, identifier, labels_regex, new_label_name, new_label_r=5, overwrite=False,
                        n_jobs=6):
    new_label, hemi = get_new_label(
        subject, atlas, labels_regex, new_label_name, new_label_r, overwrite, n_jobs)
    x = fmri.load_fmri_data_for_both_hemis(subject, identifier)
    return calc_label_corr(subject, x, new_label, hemi, new_label_name, identifier, overwrite, n_jobs)


def get_new_label(subject, atlas, regex, new_label_name, new_label_r=5, overwrite=False, n_jobs=6):
    new_label_fname = op.join(SUBJECTS_DIR, subject, 'label', '{}.label'.format(new_label_name))
    if op.isfile(new_label_fname) and not overwrite:
        new_label = mne.read_label(new_label_fname)
        return new_label, new_label.hemi

    labels = lu.read_labels(subject, SUBJECTS_DIR, atlas)
    selected_labels = [l for l in labels if fnmatch.fnmatch(l.name, '*{}*'.format(regex))]
    hemis = set([lu.get_label_hemi(l.name) for l in selected_labels])
    if len(hemis) > 1:
        raise Exception('The selected labels belong to more than one hemi!')
    selected_hemi = list(hemis)[0]
    # centers_of_mass = lu.calc_center_of_mass(selected_labels)
    selected_labels_pos = np.array(utils.flat_list([l.pos for l in selected_labels]))
    center_of_mass = np.mean(selected_labels_pos, 0) * 1000
    hemi_verts = np.load(op.join(MMVT_DIR, subject, 'surf', '{}.pial.npz'.format(selected_hemi)))['verts']
    dists = cdist(hemi_verts, [center_of_mass])
    vertice_indice = np.argmin(dists)
    new_label = lu.grow_label(subject, vertice_indice, selected_hemi, new_label_name, new_label_r, n_jobs)
    return new_label, selected_hemi


def calc_label_corr(subject, x, label, hemi, label_name, identifier, overwrite=False, n_jobs=6):
    identifier = identifier if identifier != '' else '{}_'.format(identifier)
    output_fname_template = op.join(MMVT_DIR, subject, 'fmri', 'fmri_seed_{}_{}_{}.npy'.format(
        identifier, label_name, '{hemi}'))
    minmax_fname = op.join(MMVT_DIR, subject, 'fmri', 'seed_{}_{}_minmax.pkl'.format(identifier, label_name))
    if utils.both_hemi_files_exist(output_fname_template) and op.isfile(minmax_fname) and not overwrite:
        print('All files already exist ({}, {})'.format(output_fname_template, minmax_fname))
        return True
    label_ts = np.mean(x[hemi][label.vertices, :], 0)
    corr_min, corr_max = 0 , 0
    for hemi in utils.HEMIS:
        verts_num = x[hemi].shape[0]
        corr_vals = np.zeros((verts_num))
        indices = np.array_split(np.arange(verts_num), n_jobs)
        chunks = [(x[hemi][indices_chunk], indices_chunk, label_ts, thread) for thread, indices_chunk in enumerate(indices)]
        for hemi_corr_chunk, indices_chunk in utils.run_parallel(_calc_label_corr, chunks, n_jobs):
            corr_vals[indices_chunk] = hemi_corr_chunk
        corr_vals[np.isnan(corr_vals)] = 0
        np.save(output_fname_template.format(hemi=hemi), corr_vals)
        corr_min = min(corr_min, np.min(corr_vals))
        corr_max = max(corr_max, np.max(corr_vals))
    corr_minmax = utils.get_max_abs(corr_max, corr_min)
    corr_min, corr_max = -corr_minmax, corr_minmax
    utils.save((corr_min, corr_max), minmax_fname)
    return utils.both_hemi_files_exist(output_fname_template) and op.isfile(minmax_fname)


@utils.ignore_warnings
def _calc_label_corr(p):
    hemi_data, indices_chunk, label_ts, thread = p
    hemi_corr = np.zeros((len(indices_chunk)))
    for ind, v_ts in tqdm(enumerate(hemi_data), total=len(indices_chunk)):
        hemi_corr[ind] = np.corrcoef(label_ts, v_ts)[0, 1]
    return hemi_corr, indices_chunk


def calc_fmri_corr_degree(subject, identifier='', threshold=0.7, connectivity_method='corr'):
    if isinstance(connectivity_method, str):
        connectivity_method = [connectivity_method]
    connectivity_method = connectivity_method[0]
    identifier = '{}_static_'.format(identifier) if identifier != '' else 'static_'
    corr_fname = op.join(MMVT_DIR, subject, 'connectivity', 'fmri_{}{}.npy'.format(identifier, connectivity_method))
    if not op.isfile(corr_fname):
        print("Can't find the connectivity fname ({})!".format(corr_fname))
        print("You should call calc_lables_connectivity first, like in " +
              "src.preproc.examples.connectivity.calc_fmri_static_connectivity with windows_length=0")
        return False
    corr = np.load(corr_fname)
    np.fill_diagonal(corr, 0)
    degree_mat = np.sum(corr > threshold, 0)
    output_fname = op.join(MMVT_DIR, subject, 'connectivity',  '{}{}_{}_degree.npy'.format(
        identifier.replace('_static', ''), connectivity_method, str(threshold)))
    np.save(output_fname, degree_mat)
    return op.isfile(output_fname)


def call_main(args):
    pu.run_on_subjects(args, main)


def main(subject, remote_subject_dir, args, flags):
    if utils.should_run(args, 'calc_rois_matlab_connectivity'):
        flags['calc_rois_matlab_connectivity'] = calc_rois_matlab_connectivity(subject, args)

    if utils.should_run(args, 'calc_electrodes_coh'):
        # todo: Add the necessary parameters
        flags['calc_electrodes_coh'] = calc_electrodes_coh(subject, args)

    if utils.should_run(args, 'calc_electrodes_rest_connectivity'):
        # todo: Add the necessary parameters
        flags['calc_electrodes_coh'] = calc_electrodes_rest_connectivity(subject, args)

    if utils.should_run(args, 'calc_lables_connectivity'):
        for labels_extract_mode in args.labels_extract_mode:
            flags['calc_lables_connectivity'] = calc_lables_connectivity(subject, labels_extract_mode, args)

    if utils.should_run(args, 'calc_seed_corr'):
        flags['calc_seed_corr'] = calc_seed_corr(
            subject, args.atlas, args.identifier, args.labels_regex, args.seed_label_name, args.seed_label_r,
            args.overwrite_seed_data, args.n_jobs)

    if utils.should_run(args, 'calc_fmri_corr_degree'):
        flags['calc_fmri_corr_degree'] = calc_fmri_corr_degree(
            subject, args.identifier, args.connectivity_threshold, args.connectivity_method)


    return flags


def read_cmd_args(argv=None):
    import argparse
    from src.utils import args_utils as au
    parser = argparse.ArgumentParser(description='MMVT stim preprocessing')
    parser.add_argument('-c', '--conditions', help='conditions names', required=False, default='contrast', type=au.str_arr_type)
    parser.add_argument('-t', '--task', help='task name', required=False, default='')
    parser.add_argument('--mat_fname', help='matlab connection file name', required=False, default='')
    parser.add_argument('--mat_field', help='matlab connection field name', required=False, default='')
    parser.add_argument('--sorted_labels_names_field', help='matlab connection labels name', required=False, default='')
    parser.add_argument('--labels_exclude', help='rois to exclude', required=False, default='unknown,corpuscallosum',
                        type=au.str_arr_type)
    parser.add_argument('--bipolar', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--connectivity_method', help='', required=False, default='corr,cv', type=au.str_arr_type)
    parser.add_argument('--labels_extract_mode', help='', required=False, default='mean_flip', type=au.str_arr_type)
    parser.add_argument('--connectivity_modality', help='', required=False, default='fmri')
    parser.add_argument('--labels_data_name', help='', required=False, default='')
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
    parser.add_argument('--windows_length', help='', required=False, default=0, type=int)
    parser.add_argument('--windows_shift', help='', required=False, default=500, type=int)
    parser.add_argument('--windows_num', help='', required=False, default=1, type=int)
    parser.add_argument('--max_windows_num', help='', required=False, default=None, type=au.int_or_none)
    parser.add_argument('--tmin', help='', required=False, default=None, type=au.int_or_none)
    parser.add_argument('--tmax', help='', required=False, default=None, type=au.int_or_none)

    parser.add_argument('--sfreq', help='', required=False, default=1000, type=float)
    parser.add_argument('--fmin', help='', required=False, default=5, type=float)
    parser.add_argument('--fmax', help='', required=False, default=100, type=float)
    parser.add_argument('--bands', required=False, default='')
    parser.add_argument('--epochs_fname', required=False, default='')

    parser.add_argument('--save_mmvt_connectivity', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--calc_subs_connectivity', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--recalc_connectivity', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--do_plot_static_conn', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--backup_existing_files', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--identifier', help='', required=False, default='')
    parser.add_argument('--connectivity_threshold', help='', required=False, default=0.7, type=float)

    parser.add_argument('--labels_regex', help='labels regex', required=False, default='post*cingulate*rh')
    parser.add_argument('--seed_label_name', help='', required=False, default='posterior_cingulate_rh')
    parser.add_argument('--seed_label_r', help='', required=False, default=5, type=int)
    parser.add_argument('--overwrite_seed_data', help='', required=False, default=0, type=au.is_true)

    pu.add_common_args(parser)
    args = utils.Bag(au.parse_parser(parser, argv))
    if args.max_windows_num is None:
        args.max_windows_num = np.inf
    if len(args.conditions) == 1:
        args.stat = STAT_AVG
    # print(args)
    return args


if __name__ == '__main__':
    args = read_cmd_args()
    call_main(args)
    print('finish!')
