import copy
import csv
import gc
import glob
import itertools
import logging
import os
import random
import time
import traceback
import warnings
from collections import defaultdict
from datetime import datetime
from functools import partial
from itertools import product, groupby

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas
from mne.beamformer import lcmv
from mne.time_frequency import compute_epochs_csd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import leastsq, minimize
from sklearn import mixture
from sklearn.datasets.base import Bunch
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, ElasticNetCV

from src import dtw
from src import utils
from src.beamformers import tf_dics as tf
from src.preproc import meg_preproc
from src.preproc.meg_preproc import (calc_cov, calc_csd, get_cond_fname, get_file_name, make_forward_solution_to_specific_points, TASKS)

warnings.filterwarnings("ignore")

try:
    from tqdm import tqdm
except:
    print('no tqdm!')

LINKS_DIR = utils.get_links_dir()
SUBJECTS_MEG_DIR = os.path.join(LINKS_DIR, 'meg')
SUBJECTS_MRI_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
FREE_SURFER_HOME = utils.get_link_dir(LINKS_DIR, 'freesurfer', 'FREESURFER_HOME')
BLENDER_ROOT_DIR = os.path.join(LINKS_DIR, 'mmvt_addon')

# Setting frequency bins as in Dalal et al. 2008
# BAD_ELECS = {'noninterference':['RAF4-RAF3', 'RAF5-RAF4', 'RAF8-RAF7', 'RAT8-RAT7'], 'interference': ['RAF8-RAF7', 'RAF4-RAF3']}
# BAD_ELECS = ['RAF4-RAF3', 'RAF5-RAF4', 'RAF8-RAF7', 'RAT8-RAT7']
BAD_ELECS = []
OUTLIERS = {'neutral': ['RPT8-RPT7', 'RMT3-RMT2', 'RPT2-RPT1', 'LPT2-LPT1'], 'interference': []}

ERROR_RECONSTRUCT_METHODS = ['RMS', 'RMSN', 'rol_corr'] # 'maxabs', 'dtw', 'diff_rms', 'diff'

def load_all_subcorticals(subject_meg_fol, sub_corticals_codes_file, cond, from_t, to_t, normalize=True, all_vertices=False, inverse_method='lcmv'):
    sub_corticals = utils.read_sub_corticals_code_file(sub_corticals_codes_file)
    meg_data = {}
    for sub_cortical_index in sub_corticals:
        sub_cortical, _ = utils.get_numeric_index_to_label(sub_cortical_index, None, FREE_SURFER_HOME)
        meg_data_file_name = '{}-{}-{}{}.npy'.format(cond, sub_cortical, inverse_method, '-all-vertices' if all_vertices else '')
        data = np.load(os.path.join(subject_meg_fol, 'subcorticals', meg_data_file_name))
        data = data.T
        data = data[from_t: to_t]
        if normalize:
            data = data * 1/max(data)

        meg_data[sub_cortical] = data
    return meg_data


def read_vars(events_id, region, read_csd=True, read_cov=True):
    for event in events_id.keys():
        forward, data_cov, noise_cov, noise_csd, data_csd = None, None, None, None, None
        if not region is None:
            forward = mne.read_forward_solution(get_cond_fname(FWD_X, event, region=region)) #, surf_ori=True)
        epochs = mne.read_epochs(get_cond_fname(EPO, event))
        evoked = mne.read_evokeds(get_cond_fname(EVO, event), baseline=(None, 0))[0]
        if read_cov:
            noise_cov = calc_cov(get_cond_fname(DATA_COV, event), event, epochs, None, 0)
            data_cov = calc_cov(get_cond_fname(NOISE_COV, event), event, epochs, 0.0, 1.0)
        if read_csd:
            noise_csd = calc_csd(NOISE_CSD, event, epochs, -0.5, 0., mode='multitaper', fmin=6, fmax=10, overwrite=False)
            data_csd = calc_csd(DATA_CSD, event, epochs, 0.0, 1.0, mode='multitaper', fmin=6, fmax=10, overwrite=False)
        yield event, forward, evoked, epochs, data_cov, noise_cov, data_csd, noise_csd


def get_fif_name(subject, fname_format, raw_cleaning_method, constrast, task):
    root_dir = os.path.join(SUBJECTS_MEG_DIR, task, subject)
    return partial(get_file_name, root_dir=root_dir, fname_format=fname_format, subject=subject,
        file_type='fif', raw_cleaning_method=raw_cleaning_method, constrast=constrast)


def init_msit(subject, constrast, raw_cleaning_method, region):
    events_id = dict(interference=1) # dict(interference=1, neutral=2)
    fname_format = '{subject}_msit_{raw_cleaning_method}_{constrast}_{cond}_1-15-{ana_type}.{file_type}'
    _get_fif_name = get_fif_name(subject, fname_format, raw_cleaning_method, constrast, 'MSIT')
    fwd_fname = _get_fif_name('{region}-fwd')#.format(region))
    evo_fname = _get_fif_name('ave')
    epo_fname = _get_fif_name('epo')
    data_cov_fname = _get_fif_name('data-cov')
    noise_cov_fnmae = _get_fif_name('noise-cov')
    return events_id, fwd_fname, evo_fname, epo_fname, data_cov_fname, noise_cov_fnmae


def calc_electrode_fwd(subject, electrode, events_id, bipolar=False, overwrite_fwd=False, read_if_exist=False, n_jobs=4):
    fwd_elec_fnames = [get_cond_fname(FWD_X, cond, region=electrode) for cond in events_id.keys()]
    if not np.all([os.path.isfile(fname) for fname in fwd_elec_fnames]) or overwrite_fwd:
        names, pos, org_pos = get_electrodes_positions(subject, bipolar)
        if bipolar and org_pos is None:
            raise Exception('bipolar and org_pos is None!')
        index = np.where(names==electrode)[0][0]
        elec_pos = org_pos[index] if bipolar else np.array([pos[index]])
        fwds = make_forward_solution_to_specific_points(events_id, elec_pos, electrode, EPO, FWD_X,
            n_jobs=n_jobs, usingEEG=True)
    else:
        if read_if_exist:
            fwds = []
            for fwd_fname in fwd_elec_fnames:
                fwds.append(mne.read_forward_solution(fwd_fname))
        else:
            fwds = [None] * len(events_id)
    return fwds[0] if len(events_id) == 1 else fwds


def calc_electrodes_fwd(subject, electrodes, events_id, bipolar=False, overwrite_fwd=False, read_if_exist=False, n_jobs=4):
    region = 'bipolar_electrodes' if bipolar else 'regular_electrodes'
    fwd_elec_fnames = [get_cond_fname(FWD_X, cond, region=region) for cond in events_id.keys()]
    if not np.all([os.path.isfile(fname) for fname in fwd_elec_fnames]) or overwrite_fwd:
        names, pos, org_pos = get_electrodes_positions(subject, bipolar)
        if bipolar and org_pos is None:
            raise Exception('bipolar and org_pos is None!')
        elec_pos = org_pos if bipolar else np.array(pos)
        fwds = make_forward_solution_to_specific_points(events_id, elec_pos, region, EPO, FWD_X,
            n_jobs=n_jobs, usingEEG=True)
    else:
        if read_if_exist:
            fwds = []
            for fwd_fname in fwd_elec_fnames:
                fwds.append(mne.read_forward_solution(fwd_fname))
        else:
            fwds = [None] * len(events_id)
    return fwds[0] if len(events_id) == 1 else fwds


# def calc_bipolar_electrode_fwd(subject, electrode, n_jobs=4):
#     electrodes_biplor, elecs_pos_biplor, elecs_pos_org_biplor = get_electrodes_positions(subject, bipolar=True)
#     for elec, elec_pos, elec_pos_org in zip(electrodes_biplor, elecs_pos_biplor, elecs_pos_org_biplor):
#         if elec==electrode:
#             fwd = make_forward_solution_to_specific_points(events_id, elec_pos_org, elec, EPO, FWD_X,
#                 n_jobs=n_jobs, usingEEG=True)
#             return fwd


def calc_all_electrodes_fwd(subject, events_id, overwrite_fwd=False, n_jobs=6):
    electrodes, elecs_pos, _ = get_electrodes_positions(subject, bipolar=False)
    electrodes_biplor, elecs_pos_biplor, elecs_pos_org_biplor = get_electrodes_positions(subject, bipolar=True)

    for elec, elec_pos in zip(electrodes, elecs_pos):
        if not check_if_fwd_exist(elec, events_id) or overwrite_fwd:
            print('make forward solution for {}'.format(elec))
            try:
                make_forward_solution_to_specific_points(events_id, [np.array(elec_pos)], elec, EPO, FWD_X,
                    n_jobs=n_jobs, usingEEG=True)
            except:
                print(traceback.format_exc())

    for elec, elec_pos, elec_pos_org in zip(electrodes_biplor, elecs_pos_biplor, elecs_pos_org_biplor):
        if not check_if_fwd_exist(elec, events_id) or overwrite_fwd:
            print('make forward solution for {}'.format(elec))
            try:
                make_forward_solution_to_specific_points(events_id, elec_pos_org, elec, EPO, FWD_X,
                    n_jobs=n_jobs, usingEEG=True)
            except:
                print(traceback.format_exc())


def check_if_fwd_exist(elec, events_id):
    fwd_elec_fnames = [get_cond_fname(FWD_X, cond, region=elec) for cond in events_id.keys()]
    return np.all([os.path.isfile(fname) for fname in fwd_elec_fnames])


def get_electrodes_positions(subject, bipolar):
    subject_mri_dir = os.path.join(SUBJECTS_MRI_DIR, subject)
    positions_file_name = 'electrodes{}_positions.npz'.format('_bipolar' if bipolar else '')
    positions_file_name = os.path.join(subject_mri_dir, 'electrodes', positions_file_name)
    d = np.load(positions_file_name)
    return d['names'], d['pos'], d['pos_org'] if 'pos_org' in d else None


def call_lcmv(electrode, forward, data_cov, noise_cov, evoked, epochs, cond, data_fname='', all_verts=False, pick_ori=None, whitened_data_cov_reg=0.01):
    if data_fname == '':
        data_fname = os.path.join(SUBJECT_MEG_FOL, 'subcorticals', 'lcmv_{}-{}-{}.npy'.format(cond, electrode, pick_ori))
    if not os.path.isfile(data_fname):
        stc = lcmv(evoked, forward, noise_cov, data_cov, reg=whitened_data_cov_reg, pick_ori=pick_ori, rank=None)
        data = stc.data[:len(stc.vertices)]
        data = data.T if all_verts else data.mean(0).T
        np.save(data_fname[:-4], data)
    else:
        data = np.load(data_fname)
    return data


def call_dics(forward, evoked, bipolar,  noise_csd, data_csd, cond='', data_fname='', all_verts=False, overwrite=False, electrode=''):
    from mne.beamformer import dics
    if data_fname == '':
        fmin, fmax = map(int, np.round(data_csd.frequencies)[[0, -1]])
        data_fol = os.path.join(SUBJECT_MEG_FOL, 'subcorticals', 'dics', 'bipolar' if bipolar else 'regular')
        data_fname = os.path.join(data_fol, 'dics_{}-{}-{}-{}.npy'.format(cond, electrode, fmin, fmax))
    if not os.path.isfile(data_fname) or overwrite:
        stc = dics(evoked, forward, noise_csd, data_csd)
        data = stc.data[:len(stc.vertices)]
        data = data.T if all_verts else data.mean(0).T
        np.save(data_fname[:-4], data)
    else:
        data = np.load(data_fname)
    return data


def plot_activation(events_id, meg_data, elec_data, electrode, from_t, to_t, method):
    xaxis = range(from_t, to_t)
    T = len(xaxis)
    f, axs = plt.subplots(2, sharex=True)#, sharey=True)
    for cond, ax in zip(events_id.keys(), axs):
        ax.plot(xaxis, meg_data[cond][:T], label='MEG', color='r')
        if not elec_data is None:
            ax.plot(xaxis, elec_data[cond][:T], label='electrode', color='b')
        ax.axvline(x=0, linestyle='--', color='k')
        ax.legend()
        ax.set_title('{}-{}'.format(cond, method))
    plt.xlabel('Time(ms)')
    plt.show()


def plot_activation_cond(cond, meg_data, elec_data, electrode, T,  method='', do_plot=True, plt_fname=''):
    xaxis = range(T)
    plt.figure()

    if type(meg_data) is dict:
        colors = utils.get_spaced_colors(len(meg_data.keys()) + 1)
        colors = set(colors) - set(['b'])
        for (key, data), color in zip(meg_data.items(), colors):
            plt.plot(xaxis, meg_data[key], label=key, color=color)
    else:
        plt.plot(xaxis, meg_data, label='MEG', color='r')
    if not elec_data is None:
        plt.plot(xaxis, elec_data, label=electrode, color='b')
    plt.axvline(x=0, linestyle='--', color='k')
    plt.legend()
    plt.title('{} {}'.format(cond, method))
    plt.xlabel('Time(ms)')
    if do_plot:
        plt.show()
    if plt_fname != '':
        plt.savefig(plt_fname)


def plot_activation_options(meg_data_all, elec_data, electrode, T, elec_opts=False):
    xaxis = range(T)
    f, axs = plt.subplots(len(meg_data_all.keys()), sharex=True)#, sharey=True)
    if len(meg_data_all.keys())==1:
        axs = [axs]
    for ind, ((params_option), ax) in enumerate(zip(meg_data_all.keys(), axs)):
        ax.plot(xaxis, meg_data_all[params_option], label='MEG', color='r')
        elec = elec_data if not elec_opts else elec_data[params_option]
        label = electrode if not elec_opts else params_option
        ax.plot(xaxis, elec, label=label, color='b')
        ax.axvline(x=0, linestyle='--', color='k')
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.legend()
        plt.setp(ax.get_legend().get_texts(), fontsize='20')
        plt.xlabel('Time(ms)', fontsize=20)
    plt.show()


def plot_activation_one_fig(cond, meg_data_all, elec_data, electrode, T):
    xaxis = range(T)
    colors = utils.get_spaced_colors(len(meg_data_all.keys()))
    plt.figure()
    for (label, meg_data), color in zip(meg_data_all.items(), colors):
        plt.plot(xaxis, meg_data, label=label, color=color)
    plt.plot(xaxis, elec_data, label=electrode, color='k')
    plt.axvline(x=0, linestyle='--', color='k')
    plt.legend()
    # plt.set_title('{}-{}'.format(cond, params_option), fontsize=20)
    plt.xlabel('Time(ms)', fontsize=20)
    plt.show()


def plot_all_vertices(cond, electrode, meg_data, elec_data, from_t, to_t, from_i, to_i, params_option):
    xaxis = np.arange(from_i, to_i) - 500
    plt.plot(xaxis, meg_data[from_i: to_i], label=params_option, color='r')
    plt.plot(xaxis, elec_data[cond][from_i: to_i], label=electrode, color='b')
    plt.axvline(x=0, linestyle='--', color='k')
    plt.legend()
    plt.title('{}-{}'.format(cond, params_option), fontsize=20)
    plt.xlabel('Time(ms)', fontsize=20)
    plt.xlim((from_t, to_t))
    plt.show()



def test_pick_ori(forward, data_cov, noise_cov, evoked, epochs):
    meg_data = {}
    for pick_ori, key in zip([None, 'max-power'], ['None', 'max-power']):
        meg_data[key] = call_lcmv(forward, data_cov, noise_cov, evoked, epochs, pick_ori=pick_ori)
    return meg_data


def test_whitened_data_cov_reg(forward, data_cov, noise_cov, evoked, epochs):
    meg_data = {}
    for reg in [0.001, 0.01, 0.1]:
        meg_data[reg] = call_lcmv(forward, data_cov, noise_cov, evoked, epochs, whitened_data_cov_reg=reg)
    return meg_data


def test_all_verts(forward, data_cov, noise_cov, evoked, epochs):
    return dict(all=call_lcmv(forward, data_cov, noise_cov, evoked, epochs, all_verts=True))


def normalize_meg_data(meg_data, elec_data, from_t, to_t, sigma=0, norm_max=True):
    if sigma != 0:
        meg_data = gaussian_filter1d(meg_data, sigma)
    meg_data = meg_data[from_t:to_t]
    if norm_max:
        meg_data *= 1/max(meg_data)
    if not elec_data is None:
        meg_data -= meg_data[0] - elec_data[0]
    return meg_data


def normalize_elec_data(elec_data, from_t, to_t):
    elec_data = elec_data[from_t:to_t]
    elec_data = elec_data - min(elec_data)
    elec_data *= 1/max(elec_data)
    return elec_data

# def smooth_meg_data(meg_data):
#     meg_data_all = {}
#     for sigma in [8, 10, 12]:
#         meg_data_all[sigma] = gaussian_filter1d(meg_data, sigma)
#     return meg_data_all


# def check_electrodes():
#     meg_data_all, elec_data_all = {}, {}
#     electrodes = ['LAT1', 'LAT2', 'LAT3', 'LAT4']
#     vars = read_vars(events_id, None)
#     for cond, forward, evoked, epochs, data_cov, noise_cov, data_csd, noise_csd in vars:
#         for electrode in electrodes:
#             calc_electrode_fwd(MRI_SUBJECT, electrode, events_id, bipolar, overwrite_fwd=False)
#             forward = mne.read_forward_solution(get_cond_fname(FWD_X, cond, region=electrode)) #, surf_ori=True)
#             elec_data = load_electrode_msit_data(bipolar, electrode, BLENDER_SUB_FOL, positive=True, normalize_data=True)
#             meg_data = call_dics(forward, evoked, bipolar, noise_csd, data_csd, cond)
#             elec_data_norm, meg_data_norm = normalize_data(elec_data[cond], meg_data, from_t, to_t)
#             meg_data_norm = gaussian_filter1d(meg_data_norm, 10)
#             meg_data_all[electrode] = meg_data_norm
#             elec_data_all[electrode] = elec_data_norm
#         plot_activation_options(meg_data_all, elec_data_all, electrodes, 500, elec_opts=True)


def get_dics_fname(cond, bipolar, electrode, fmin, fmax):
    dics_fol = os.path.join(SUBJECT_MEG_FOL, 'subcorticals', 'dics', 'bipolar' if bipolar else 'regular')
    return os.path.join(dics_fol, 'dics_{}-{}-{}-{}.npy'.format(cond, electrode, fmin, fmax))


def calc_dics_freqs_csd(events_id, electrodes, bipolar, from_t, to_t, time_split, freqs_bands,
        overwrite_csds=False, overwrite_dics=False, gk_sigma=3, njobs=6):
    vars = list(read_vars(events_id, None, read_csd=False, read_cov=False))
    # electrodes = get_all_electrodes_names(bipolar)
    dics_fol = os.path.join(SUBJECT_MEG_FOL, 'subcorticals', 'dics', 'bipolar' if bipolar else 'regular')
    utils.make_dir(dics_fol)
    for cond, _, evoked, epochs, _, _, _, _ in vars:
        event_id = {cond: events_id[cond]}
        all_electrodes_freqs = list(product(electrodes, freqs_bands))
        electrodes_freqs = [(el, (fmin, fmax)) for el, (fmin, fmax) in all_electrodes_freqs \
            if not os.path.isfile(get_dics_fname(cond, bipolar, el, fmin, fmax)) \
            or overwrite_csds or overwrite_dics]
        np.random.shuffle(electrodes_freqs)
        chunks = utils.chunks(electrodes_freqs, len(electrodes_freqs) / njobs)
        params = [(event_id, chunk, evoked, epochs, bipolar,
                   overwrite_csds, overwrite_dics, gk_sigma) for chunk in chunks]
        utils.run_parallel(_par_calc_dics_chunk_electrodes, params, njobs)


def calc_all_fwds(events_id, electrodes, bipolar, from_t, to_t, time_split, overwrite_fwd=False, njobs=6):
    vars = list(read_vars(events_id, None, read_csd=False))
    electrodes = get_all_electrodes_names(bipolar)
    for cond, _, evoked, epochs, data_cov, noise_cov, _, _ in vars:
        event_id = {cond: events_id[cond]}
        for electrode in electrodes:
            calc_electrode_fwd(MRI_SUBJECT, electrode, event_id, bipolar,
                overwrite_fwd=overwrite_fwd, read_if_exist=(not overwrite_fwd), n_jobs=njobs)


def check_bipolar_fwd(forward, event_id, electrode, bipolar):
    vertno = utils.fwd_vertno(forward)
    fwd_was_wrong = True
    if bipolar and vertno != 2:
        fwd_was_wrong = True
        forward = calc_electrode_fwd(MRI_SUBJECT, electrode, event_id, bipolar, overwrite_fwd=True)
        vertno = utils.fwd_vertno(forward)
        if vertno != 2:
            raise Exception('vertno != 2')
    return forward, fwd_was_wrong


def load_all_dics(freqs_bins, event_id, bipolar, electrodes, from_t, to_t, gk_sigma, dont_calc_new_csd=False, njobs=2):
    meg_data_dic = {}
    cond = utils.first_key(event_id)
    freqs_bins_sorted = sorted(freqs_bins)
    for electrode in electrodes:
        dics_files = glob.glob(os.path.join(SUBJECT_MEG_FOL, 'subcorticals', 'dics',
            'bipolar' if bipolar else 'regular', 'dics_{}-{}-*.npy'.format(cond, electrode)))
        if len(dics_files) < len(freqs_bins_sorted) and dont_calc_new_csd:
            print('{} does not have all the csd files'.format(electrode))
            continue
        params = [(event_id, None, None, None, None, None, electrode, bipolar,
                   from_t, to_t, False, False, gk_sigma, True, dont_calc_new_csd, ifreq, fmin, fmax)
                  for ifreq, (fmin, fmax) in enumerate(freqs_bins_sorted)]
        results = utils.run_parallel(_par_calc_dics_frqs, params, njobs)
        meg_data_arr = np.zeros((len(results), to_t-from_t))
        data_is_none = False
        for data, ifreq, fmin, fmax in results:
            if data is None:
                print('{}, {}-{}: data is None!'.format(electrode, fmin, fmax))
                data_is_none = True
                break
            meg_data_arr[ifreq, :] = data
        if not data_is_none:
            meg_data_dic[electrode] = meg_data_arr

    return meg_data_dic


def reconstruct_meg(events_id, freqs_bins, electrodes, from_t, to_t, time_split, gk_sigma=3, bipolar=True, plot_elecs=False, title=None,
        predicted_electrodes=[], plot_results=False, dont_calc_new_csd=True, vars=None, all_meg_data=None, res_ind=0,
        elec_data=None, optimization_method='leastsq', error_calc_method='RMS', optimization_params={},
        save_plots_in_pred_electrode_fol=False, do_save_plots=False, uuid='', njobs=6):
    root_fol = os.path.join(SUBJECT_MEG_FOL, 'subcorticals', 'dics', 'bipolar' if bipolar else 'regular')
    opt_ps, errors, opt_cv_params = {}, {}, {}
    time_diff = np.diff(time_split)[0] if len(time_split) > 1 else 500

    if not all_dics_are_already_computed(events_id, freqs_bins, electrodes, predicted_electrodes, dont_calc_new_csd, root_fol):
        return None, None, None
    if elec_data is None:
        elec_data = load_electrodes_data(events_id, bipolar, electrodes, from_t, to_t,
            subtract_min=True, normalize_data=True)
    if plot_elecs:
        plot_electrodes(events_id, electrodes, elec_data)

    vars = prepare_vars(events_id, vars, dont_calc_new_csd)
    for cond, _, evoked, epochs, data_cov, noise_cov, _, _ in vars:
        event_id = {cond: events_id[cond]}
        meg_data_dic = calc_meg_data_dic(event_id, evoked, epochs, data_cov, noise_cov, all_meg_data, freqs_bins, electrodes,
            predicted_electrodes, bipolar, from_t, to_t, gk_sigma, dont_calc_new_csd, root_fol, njobs)
        if not meg_data_dic:
            continue

        ps, cvs_parameters = [], []
        for from_t, to_t in zip(time_split, time_split + time_diff):
            p, cv_parameters = calc_optimization_features(optimization_method, freqs_bins, cond,
                meg_data_dic, elec_data, electrodes, from_t, to_t, optimization_params)
            ps = p if len(ps) == 0 else np.vstack((ps, p))
            cvs_parameters.append(cv_parameters)

        opt_ps[cond] = ps
        opt_cv_params[cond] = cvs_parameters
        errors[cond] = calc_reconstruction_errors(ps, cond, electrodes + predicted_electrodes,
                                                  elec_data, meg_data_dic, time_split, time_diff, error_calc_method)
        if plot_results:
            plot_leastsq_results(meg_data_dic, cond, elec_data, electrodes, opt_ps[cond], time_split,
                optimization_method, predicted_electrodes, same_ps=True, do_plot=True, title=title, res_ind=res_ind,
                save_in_pred_electrode_fol=save_plots_in_pred_electrode_fol, do_save=do_save_plots, uuid=uuid)
    return errors, opt_ps, opt_cv_params


def prepare_vars(events_id, vars, dont_calc_new_csd):
    if dont_calc_new_csd:
        vars = [(event, None, None, None, None, None, None, None) for event in events_id.keys()]
    else:
        if vars is None:
            vars = list(read_vars(events_id, None, read_csd=False))
    return vars


def all_dics_are_already_computed(events_id, freqs_bins, electrodes, predicted_electrodes, dont_calc_new_csd, root_fol):
    # Check if all the dics files are already computed
    for electrode in electrodes + predicted_electrodes:
        dics_files_num = np.array([len(list(glob.glob(os.path.join(root_fol, 'dics_{}-{}-*.npy'.format(cond, electrode))))) for cond in events_id])
        if np.any(dics_files_num < len(freqs_bins)) and dont_calc_new_csd:
            mes = '{}: not all the csds are calculated {}/{} for {}'.format(electrode, dics_files_num, len(freqs_bins), events_id.keys())
            print(mes)
            logging.error(mes)
            return False
    return True


def plot_electrodes(events_id, electrodes, elec_data):
    plt.figure()
    for electrode in electrodes:
        for cond in events_id:
            plt.plot(elec_data[electrode][cond], label='{} {}'.format(cond, electrode))
    plt.legend()
    plt.show()


def calc_meg_data_dic(event_id, evoked, epochs, data_cov, noise_cov, all_meg_data, freqs_bins, electrodes, predicted_electrodes,
        bipolar, from_t, to_t, gk_sigma, dont_calc_new_csd, root_fol, njobs):
    cond = utils.first_key(event_id)
    meg_data_dic = {}
    for electrode in electrodes + predicted_electrodes:
        if dont_calc_new_csd:
            if not all_meg_data is None:
                dics_files_num = len(list(glob.glob(os.path.join(root_fol, 'dics_{}-{}-*.npy'.format(cond, electrode)))))
                if dics_files_num < len(freqs_bins):
                    logging.error('dics_files_num ({}) < len(CSD_FREQS) ({})!'.format(dics_files_num, len(freqs_bins)))
                    continue
            else:
                params = [(event_id, None, None, None, None, None, electrode, bipolar,
                    from_t, to_t, False, False, gk_sigma, True, dont_calc_new_csd, ifreq, fmin, fmax)
                    for ifreq, (fmin, fmax) in enumerate(freqs_bins)]
        else:
            params = [(event_id, None, evoked, epochs, data_cov, noise_cov, electrode, bipolar,
                from_t, to_t, False, False, gk_sigma, True, dont_calc_new_csd, ifreq, fmin, fmax)
                for ifreq, (fmin, fmax) in enumerate(freqs_bins)]
        try:
            if all_meg_data is None:
                results = utils.run_parallel(_par_calc_dics_frqs,  params, njobs)
                meg_data_arr = []
                for data, fmin, fmax in results:
                    meg_data_arr = data if len(meg_data_arr)==0 else np.vstack((meg_data_arr, data))
                meg_data_dic[electrode] = meg_data_arr
            else:
                meg_data_dic[electrode] = all_meg_data[electrode]
        except:
            print('check_freqs: Error in gathering the csd files!')
            print(traceback.format_exc())
            logging.error(traceback.format_exc())
            if electrode in meg_data_dic:
                meg_data_dic.pop(electrode)
    return meg_data_dic


def calc_optimization_features(optimization_method, freqs_bins, cond, meg_data_dic, elec_data, electrodes, from_t, to_t, optimization_params={}):
    # scorer = make_scorer(rol_corr, False)
    cv_parameters = []
    if optimization_method in ['Ridge', 'RidgeCV', 'Lasso', 'LassoCV', 'ElasticNet', 'ElasticNetCV']:
        # vstack all meg data, such that X.shape = T*n X F, where n is the electrodes num
        # Y is T*n * 1
        X = np.hstack((meg_data_dic[electrode][:, from_t:to_t] for electrode in electrodes))
        Y = np.hstack((elec_data[electrode][cond][from_t:to_t] for electrode in electrodes))
        funcs_dic = {'Ridge': Ridge(alpha=0.1), 'RidgeCV':RidgeCV(np.logspace(0, -10, 11)), # scoring=scorer
            'Lasso': Lasso(alpha=1.0/X.shape[0]), 'LassoCV':LassoCV(alphas=np.logspace(0, -10, 11), max_iter=1000),
            'ElasticNetCV': ElasticNetCV(alphas= np.logspace(0, -10, 11), l1_ratio=np.linspace(0, 1, 11))}
        clf = funcs_dic[optimization_method]
        clf.fit(X.T, Y)
        p = clf.coef_
        if len(p) != len(freqs_bins):
            raise Exception('{} (len(clf.coef)) != {} (len(freqs_bin))!!!'.format(len(p), len(freqs_bins)))
        if optimization_method in ['RidgeCV', 'LassoCV']:
            cv_parameters = clf.alpha_
        elif optimization_method == 'ElasticNetCV':
            cv_parameters = [clf.alpha_, clf.l1_ratio_]
        args = [(meg_pred(p, meg_data_dic[electrode][:, from_t:to_t]), elec_data[electrode][cond][from_t:to_t]) for electrode in electrodes]
        p0 = leastsq(post_ridge_err_func, [1], args=args, maxfev=0)[0]
        p = np.hstack((p0, p))
    elif optimization_method in ['leastsq', 'dtw', 'minmax', 'diff_rms', 'rol_corr']:
        args = ([(meg_data_dic[electrode][:, from_t:to_t], elec_data[electrode][cond][from_t:to_t]) for electrode in electrodes], optimization_params)
        p0 = np.ones((1, len(freqs_bins)+1))
        funcs_dic = {'leastsq': partial(leastsq, func=err_func, x0=p0, args=args),
                     'dtw': partial(minimize, fun=dtw_err_func, x0=p0, args=args),
                     'minmax': partial(minimize, fun=minmax_err_func, x0=p0, args=args),
                     'diff_rms': partial(minimize, fun=min_diff_rms_err_func, x0=p0, args=args),
                     'rol_corr': partial(minimize, fun=max_rol_corr, x0=p0, args=args)}
        res = funcs_dic[optimization_method]()
        p = res[0] if optimization_method=='leastsq' else res.x
        cv_parameters = optimization_params
    else:
        raise Exception('Unknown optimization_method! {}'.format(optimization_method))
    return p, cv_parameters


def calc_reconstruction_errors(electrode_ps, cond, electrodes, elec_data, meg_data_dic, time_split, time_diff,
        error_calc_method='RMS', dtw_window=10):
    errors = {}
    for electrode in electrodes:
        err = electrode_reconstruction_error(electrode, elec_data[electrode][cond], electrode_ps, meg_data_dic,
            error_calc_method, time_split, time_diff, dtw_window=dtw_window)
        errors[electrode] = err
    return errors


def electrode_reconstruction_error(electrode, electrode_data, electrode_ps, meg_data_dic,
        error_calc_method, time_split, time_diff, dtw_window=10, rol_corr_window=30, meg=None):
    if meg is None:
        meg = combine_meg_chunks(meg_data_dic[electrode], electrode_ps, time_split, time_diff)
    if error_calc_method == 'RMS':
        err = sum((electrode_data - meg)**2)
    elif error_calc_method == 'RMSN':
         err = sum((electrode_data - meg)**2) * 1.0/utils.max_min_diff(electrode_data)
    elif error_calc_method == 'maxabs':
        err = maxabs(electrode_data, meg)
    elif error_calc_method == 'dtw':
        err = dtw.distance_w(electrode_data, meg, dtw_window)
    elif error_calc_method == 'diff':
        err = sum(abs(np.diff(electrode_data) - np.diff(meg)))
    elif error_calc_method == 'diff_rms':
        err = diff_rms(electrode_data, meg)
    elif error_calc_method == 'rol_corr':
        err = rol_corr(electrode_data, meg, window=rol_corr_window)
    else:
        raise Exception('Unreconize error_calc_method! {}'.format(error_calc_method))
    return err


def meg_pred(p, X):
    if len(p) == X.shape[0]:
        return np.dot(p, X)
    else:
        return p[0] + np.dot(p[1:], X)


def electrode_err_func(p, X, y):
    return y - meg_pred(p, X)


def err_func(p, XY, params):
    return sum([electrode_err_func(p, X, y) for X, y in XY])


def post_ridge_err_func(p, XY):
    return sum([y - (p + X) for X, y in XY])


def dtw_err_func(p, XY, params):
    dists = sum([dtw.distance_w(y, meg_pred(p, X), 10) for X, y in XY])
    return dists + np.mean(p**2)


def minmax_err_func(p, XY, params):
    return sum([max(abs(y - meg_pred(p, X))) for X, y in XY])


def min_diff_rms_err_func(p, XY, params):
    err = 0
    for X, y in XY:
        meg = meg_pred(p, X)
        err += diff_rms(y, meg)
    return err


def max_rol_corr(p, XY, params):
    err = 0
    window = params.get('window', 30)
    alpha = params.get('alpha', 1)
    for X, y in XY:
        meg = meg_pred(p, X)
        err += rol_corr(y, meg, window=window, alpha=alpha) #+ 0.0001 * np.sum((p)**2)
    return err


def maxabs(y, meg):
    return max(abs(y - meg)) * 1/utils.max_min_diff(y)


def diff_rms(y, meg):
    # diffs_sum = sum(abs(np.diff(y) - np.diff(meg)))
    # diffs_sum = sum(abs(utils.diff_4pc(y) - utils.diff_4pc(meg)))
    # y = gaussian_filter1d(y, 3)
    diffs_sum = sum(abs(np.gradient(y) - np.gradient(meg)))
    rms = np.sum((y-meg)**2)
    max_abs = max(abs(y-meg))
    if max_abs > 0.3:
        max_abs = np.inf
    if rms * 1/utils.max_min_diff(y) > 10:
        rms = np.inf
    return (diffs_sum + rms + max_abs) * 1/utils.max_min_diff(y)


def rol_corr(y, meg, window=30, alpha=5):
    rol_corr = pandas.rolling_corr(y, meg, window)
    return (1 - np.nanmean(rol_corr)) * alpha + np.sum((y-meg)**2) * 1/utils.max_min_diff(y)


# def rol_corr(y, meg, window=30):
#     rol_corr = pandas.rolling_corr(y, meg, window)
#     max_cor = len(y) - window + 1  # sum(~np.isnan(rol_corr))
#     alpha = 30 * len(y) / 500
#     corr_term = (max_cor - np.nansum(rol_corr)) * alpha / max_cor
#     rms_term = np.sum((y-meg)**2) * 1/utils.max_min_diff(y)
#     err = corr_term + rms_term
#     return err

def _par_calc_dics_chunk_electrodes(params_chunck):
    (event_id, elecs_freqs_chunck, evoked, epochs, bipolar,
        overwrite_csd, overwrite_dics, gk_sigma) = params_chunck
    cond = utils.first_key(event_id)
    data_fol = os.path.join(SUBJECT_MEG_FOL, 'subcorticals', 'dics', 'bipolar' if bipolar else 'regular')
    forwards = {}
    for electrode, (fmin, fmax) in elecs_freqs_chunck:
        data_fname = os.path.join(data_fol, 'dics_{}-{}-{}-{}.npy'.format(cond, electrode, fmin, fmax))
        csd_fname = os.path.join(data_fol, 'csd_{}-{}-{}-{}.npy'.format(cond, electrode, fmin, fmax))
        if not os.path.isfile(data_fname) or overwrite_csd or overwrite_dics:
            print('compute csd and dics for {} {}-{}'.format(electrode, fmin, fmax))
            if electrode not in forwards:
                fwd_fname = get_cond_fname(FWD_X, cond, region=electrode)
                forwards[electrode] = mne.read_forward_solution(fwd_fname)
            if bipolar:
                forwards[electrode], fwd_was_wrong = check_bipolar_fwd(forwards[electrode], event_id, electrode, bipolar)
            if not os.path.isfile(csd_fname) or overwrite_csd:
                noise_csd = compute_epochs_csd(epochs, 'multitaper', tmin=-0.5, tmax=0.0, fmin=fmin, fmax=fmax)
                data_csd = compute_epochs_csd(epochs, 'multitaper', tmin=0.0, tmax=1.0, fmin=fmin, fmax=fmax)
                utils.save((data_csd, noise_csd), csd_fname)
            else:
                data_csd, noise_csd = utils.load(csd_fname)
            data = call_dics(forwards[electrode], evoked, bipolar, noise_csd, data_csd, data_fname=data_fname, all_verts=bipolar,
                 overwrite=overwrite_dics, electrode=electrode)
            if bipolar:
                if data.shape[1] != 2:
                    raise Exception('Should be 2 sources in the bipolar fwd!')
                data = np.diff(data).squeeze()
            np.save(data_fname, data)
            del noise_csd, data_csd, data
            gc.collect()
        else:
            print('{} already exists'.format(utils.namebase(data_fname)))


def _par_calc_dics_frqs(p):
    event_id, forward, evoked, epochs, data_cov, noise_cov, electrode, bipolar, from_t, to_t,\
        overwrite_csd, overwrite_dics, gk_sigma, load_data, dont_calc_new_csd, ifreq, fmin, fmax = p
    cond = utils.first_key(event_id)
    data_fname = os.path.join(SUBJECT_MEG_FOL, 'subcorticals', 'dics', 'bipolar' if bipolar else 'regular',
        'dics_{}-{}-{}-{}.npy'.format(cond, electrode, fmin, fmax))
    data = None
    if not os.path.isfile(data_fname) or overwrite_csd:
        if dont_calc_new_csd:
            raise Exception('dont_calc_new_csd flag and not all csds are computed! ({} {} {}-{})'.format(cond, electrode, fmin, fmax))
        print('compute csd and dics for {}-{}'.format(fmin, fmax))
        noise_csd = compute_epochs_csd(epochs, 'multitaper', tmin=-0.5, tmax=0.0, fmin=fmin, fmax=fmax)
        data_csd = compute_epochs_csd(epochs, 'multitaper', tmin=0.0, tmax=1.0, fmin=fmin, fmax=fmax)
        data = call_dics(forward, evoked, bipolar, noise_csd, data_csd, data_fname=data_fname, all_verts=bipolar,
             overwrite=overwrite_dics, electrode=electrode)
        if bipolar:
            if data.shape[1] != 2:
                raise Exception('Should be 2 sources in the bipolar fwd!')
            data = np.diff(data).squeeze()
        np.save(data_fname, data)
    else:
        if load_data:
            data = np.load(data_fname)
    if not data is None and not from_t is None:
        data = normalize_meg_data(data, None, from_t, to_t, gk_sigma, norm_max=False)
    if not data is None:
        data = data.squeeze()
    return data, ifreq, fmin, fmax


# def comp_lcmv_dics_electrode(events_id, electrode, bipolar):
#     elec_data = load_electrode_msit_data(bipolar, electrode, BLENDER_SUB_FOL, positive=True, normalize_data=True)
#     calc_electrode_fwd(MRI_SUBJECT, electrode, events_id, bipolar, overwrite_fwd=False)
#     vars = read_vars(events_id, electrode)
#     for cond, forward, evoked, epochs, data_cov, noise_cov, data_csd, noise_csd in vars:
#         meg_data_lcmv = call_lcmv(forward, data_cov, noise_cov, evoked, epochs, cond, all_verts=True)
#         meg_data_dics = call_dics(forward, evoked, bipolar, noise_csd, data_csd, cond, all_verts=True)
#         if bipolar:
#             meg_data_lcmv = np.diff(meg_data_lcmv).squeeze()
#             meg_data_dics = np.diff(meg_data_dics).squeeze()
#         elec_data_norm = normalize_elec_data(elec_data[cond], from_t, to_t)
#         meg_data_lcmv_norm = normalize_meg_data(meg_data_lcmv, elec_data_norm, from_t, to_t, 3)
#         meg_data_dics_norm = normalize_meg_data(meg_data_dics, elec_data_norm, from_t, to_t, 3)
#         plot_activation_cond(cond, {'lcmv': meg_data_lcmv_norm, 'dics': meg_data_dics_norm}, elec_data_norm, electrode, 500)
#

def check_bipolar_meg(events_id, electrode, bipolar, from_t, to_t):
    elec_name2, elec_name1 = electrode.split('-')
    # Warning: load_electrodes_data parameters have been changed!
    elec_bip_data = load_electrodes_data (True, electrode, BLENDER_SUB_FOL, positive=True, normalize_data=False)
    elec2_data = load_electrodes_data(False, elec_name2, BLENDER_SUB_FOL, positive=True, normalize_data=False)
    elec1_data = load_electrodes_data(False, elec_name1, BLENDER_SUB_FOL, positive=True, normalize_data=False)

    vars = read_vars(events_id, electrode)
    for cond, forward, evoked, epochs, data_cov, noise_cov, data_csd, noise_csd in vars:
        max_electrode = max([max(data) for data in [elec_bip_data[cond][from_t:to_t], elec2_data[cond][from_t:to_t], elec1_data[cond][from_t:to_t]]])
        elec_bip_data_norm = (elec_bip_data[cond] * 1.0/max_electrode)[from_t:to_t]
        elec2_data_norm = (elec2_data[cond] * 1.0/max_electrode)[from_t:to_t]
        elec1_data_norm = (elec1_data[cond] * 1.0/max_electrode)[from_t:to_t]

        meg_data_lcmv = call_lcmv(forward, data_cov, noise_cov, evoked, epochs, cond, all_verts=True)
        meg_data_dics = call_dics(forward, evoked, bipolar, noise_csd, data_csd, cond, all_verts=True)
        for data, method in zip([meg_data_lcmv, meg_data_dics], ['lcmv', 'dics']):
            data_1 = normalize_meg_data(data[:, 0], elec_bip_data_norm, from_t, to_t, 3)
            data_2 = normalize_meg_data(data[:, 1], elec_bip_data_norm, from_t, to_t, 3)
            plt.figure()
            plt.plot(data_1-data_2, label='{} diff'.format(method))
            plt.plot(data_1, label='{} 1'.format(method))
            plt.plot(data_2, label='{} 2'.format(method))
            plt.plot(elec_bip_data_norm, label=electrode)
            plt.plot(elec1_data_norm, label=elec_name1)
            plt.plot(elec2_data_norm, label=elec_name2)
            plt.plot()
            plt.legend()
            plt.title(cond)
            plt.show()


def get_electrodes_parcellation(electrodes, bipolar, include_white_matter=True):
    parc = defaultdict(dict)
    parc_fname = os.path.join(electrode_parc_fol(),
        '{}_laus250_electrodes_all_rois_cigar_r_3_l_4{}.csv'.format(
        MRI_SUBJECT, '_bipolar_stretch' if bipolar else ''))
    if os.path.isfile(parc_fname):
        electrodes_probs = np.genfromtxt(parc_fname, dtype=np.str, delimiter=',')
        rois = electrodes_probs[0, 1:]
        for electrode_probs in electrodes_probs[1:, :]:
            elec_name = electrode_probs[0]
            for elec_prob, roi in zip(electrode_probs[1:], rois):
                if float(elec_prob) > 0:
                    if include_white_matter or not 'White-Matter' in roi:
                        parc[elec_name][roi] = elec_prob
    return parc

def get_figs_fol():
    return os.path.join(utils.get_figs_fol(), 'meg_electrodes')


def electrode_parc_fol():
    return os.path.join(utils.get_parent_fol(), 'electrodes_parcellation')


# def load_all_electrodes_data(root_fol, bipolar):
#     d = np.load(os.path.join(root_fol, 'electrodes{}_data.npz'.format('_bipolar' if bipolar else '')))
#     data, names, elecs_conditions = d['data'], d['names'], d['conditions']
#     return data, names, elecs_conditions


def find_significant_electrodes(events_id, bipolar, from_t, to_t, do_plot=False, do_save=False, plot_only_sig=False):
    elec_data = load_electrodes_data(events_id, bipolar, subtract_min=False, normalize_data=False)
    T = to_t - from_t
    sig_electrodes = defaultdict(list)
    if do_save and do_plot:
        fol = os.path.join(SUBJECT_MRI_FOL, 'electrodes', 'figs', 'bipolar' if bipolar else 'regular')
        utils.delete_folder_files(fol)

    for cond in events_id.keys():
        # cond_id = MEG_ELEC_CONDS_TRANS[cond]
        # cond = EVENTS_TRANS_INV[cond]
        for electrode in elec_data.keys():
            if electrode in BAD_ELECS:
                continue
            org_data = elec_data[electrode][cond]
            data_std = np.std(org_data[:from_t])
            data_mean = np.mean(org_data[:from_t])
            data = org_data[from_t:to_t] - data_mean
            sig = False
            for stds_num, sig_len in [(3, 30), (4, 20), (5, 10)]:
                sig_indices = np.where((data > data_mean + stds_num * data_std) | (data < data_mean - stds_num * data_std))[0]
                diff = np.diff(sig_indices)
                sig = sig or max(map(len, ''.join([str(x==y)[0] for (x,y) in zip(diff[:-1], diff[1:])]).split('F'))) > sig_len
            if sig:
                dics_files = glob.glob(os.path.join(SUBJECT_MEG_FOL, 'subcorticals', 'dics',
                    'bipolar' if bipolar else 'regular', 'dics_{}-{}-*.npy'.format(cond, electrode)))
                if len(dics_files) > 0:
                    sig_electrodes[cond].append(electrode)
                    # print(cond, name)
                #     print('dic file exist for {}-{} sig'.format(cond, name))
            if do_plot or (plot_only_sig and sig):
                plt.figure()
                plt.plot(data - data_mean, 'b')
                plt.plot((0, T), (3 * data_std, 3 * data_std), 'r--')
                plt.plot((0, T), (-3 * data_std, -3 * data_std), 'r--')
                plt.plot((0, T), (2 * data_std, 2 * data_std), 'y--')
                plt.plot((0, T), (-2 * data_std, -2 * data_std), 'y--')
                plt.plot((0, T), (2.5 * data_std, 2.5 * data_std), 'c--')
                plt.plot((0, T), (-2.5 * data_std, -2.5 * data_std), 'c--')
                title = '{}-{}{}'.format(cond, electrode, '-sig' if sig else '')
                plt.title(title)
                if do_save:
                    plt.savefig(os.path.join(fol, '{}.png'.format(title)))
                    plt.close()
                else:
                    plt.show()
                print(title)
    # fname = 'sig_{}electrodes.pkl'.format('bipolar_' if bipolar else '')
    # utils.save(sig_electrodes, os.path.join(SUBJECT_MRI_FOL, 'electrodes', fname))
    return sig_electrodes


def check_freqs_for_all_electrodes(events_id, from_t, to_t, time_split, njobs=4):
    for bipolar in [True, False]:
        # electrodes, _, _ = get_electrodes_positions(MRI_SUBJECT, bipolar)
        electrodes = get_all_electrodes_names(bipolar)
        for electrode in electrodes:
            try:
                reconstruct_meg(events_id, [electrode], from_t, to_t, time_split, gk_sigma=3, bipolar=bipolar,
                    plot_elecs=False, plot_results=False, predicted_electrodes=[], njobs=njobs)
            except:
                pass

def learn_and_pred(events_id, bipolar, from_t, to_t, time_split):
    sig = find_significant_electrodes(events_id, bipolar, from_t, to_t)
    electrodes, predicted_electrodes = [],[]
    for k, (cond, name) in enumerate(sig.items()):
        if EVENTS_TRANS[cond] in events_id.keys():
            if k%3 == 0:
                predicted_electrodes.append(name)
            else:
                electrodes.append(name)
    reconstruct_meg(events_id, electrodes, from_t, to_t, time_split, predicted_electrodes=predicted_electrodes, gk_sigma=3, bipolar=bipolar, njobs=1)


def find_fit(events_id, bipolar, from_t, to_t, time_split, gk_sigma=3, err_threshold=np.inf, plot_results=False, njobs=3):
    sig_electrodes = find_significant_electrodes(events_id, bipolar, from_t, to_t, do_plot=False, do_save=False, plot_only_sig=False)
    bad_channels = {}
    for cond in events_id:
        event_id = {cond: events_id[cond]}
        if not cond in bad_channels:
            bad_channels[cond] = []
        print('calc leastsq for {}'.format(cond))
        electrodes = list(set(sig_electrodes[cond]) - set(bad_channels[cond]))
        elec_data = load_electrodes_data(event_id, bipolar, electrodes, from_t, to_t,
            subtract_min=False, normalize_data=False)
        meg_data_dic = load_all_dics(freqs_bin, event_id, bipolar, electrodes, from_t, to_t, gk_sigma, njobs=njobs)
        errors, ps, cv_params = {}, {}, {}
        for electrode in electrodes:
            elec_errs, elec_ps, opt_cv_params = reconstruct_meg(event_id, [electrode], from_t, to_t, time_split,
                plot_results=plot_results, bipolar=bipolar, dont_calc_new_csd=True, all_meg_data=meg_data_dic,
                elec_data=elec_data, njobs=njobs)
            if elec_errs:
                print(electrode, elec_errs[cond][electrode])
                errors[electrode] = elec_errs[cond][electrode]
                ps[electrode] = elec_ps[cond]
                cv_params[electrode] = opt_cv_params[cond]

        utils.save((errors, ps, cv_params), get_pkl_file('{}_leastsq_time_split_{}.pkl'.format(cond, len(time_split))))


def analyze_leastsq_results(events_id, time_split):
    for cond in events_id:
        print(cond)
        errors, ps = np.load(get_pkl_file('{}_leastsq_time_split_{}.pkl'.format(cond, len(time_split))))
        electrodes = errors.keys()
        print(sorted(ps.keys()))
        print(cond, len(errors))
        X = []
        for elec, p in ps.items():
            X = p if len(X)==0 else np.vstack((X, p))

        utils.plot_3d_PCA(X, electrodes)
        x_pca = utils.calc_PCA(X, n_components=3)
        res, best_gmm, bic = utils.calc_clusters_bic(X, 10)
        # means = res['spherical'][6].means_
        best_gmm = res['spherical'][6]
        utils.plot_3d_scatter(x_pca, classifier=best_gmm)
        print('sdf')


def find_best_groups(event_id, bipolar, from_t, to_t, time_split, err_threshold=7, groups_panelty=5, only_sig_electrodes=False,
        electrodes_positive=True, electrodes_normalize=True, gk_sigma=3, njobs=4):

    cond = utils.first_key(event_id)
    if only_sig_electrodes:
        sig_electrodes = find_significant_electrodes(event_id, bipolar, from_t, to_t, do_plot=False, do_save=False, plot_only_sig=False)
        all_electrodes = sig_electrodes[cond]
    else:
        all_electrodes = get_all_electrodes_names(bipolar)

    elec_data = load_electrodes_data(event_id, bipolar, all_electrodes, from_t, to_t,
        subtract_min=electrodes_positive, normalize_data=electrodes_normalize)
    meg_data_dic = load_all_dics(freqs_bin, event_id, bipolar, all_electrodes, from_t, to_t, gk_sigma, njobs=njobs)

    run_min_score = np.inf
    run_num = 1
    uuid = utils.rand_letters(5)
    print('find_best_groups', cond, err_threshold)
    while True:
        electrodes = set(list(all_electrodes))
        new_electrode, elec_ps, elec_err = pick_new_electrode(event_id, bipolar, from_t, to_t, time_split,
            electrodes, meg_data_dic, elec_data, njobs)
        electrodes.remove(new_electrode)
        groups_errs = [[elec_err]]
        groups_ps = [elec_ps]
        groups = [[new_electrode]]
        while len(electrodes) > 0:
            new_electrode, new_err, new_ps = find_best_new_electrode(groups[-1], electrodes,
                event_id, meg_data_dic, elec_data, from_t, to_t, time_split, bipolar, njobs)
            if new_err < err_threshold:
                groups[-1].append(new_electrode)
                groups_errs[-1].append(new_err)
                groups_ps[-1] = new_ps
            else:
                print('new group!')
                # for debug:
                plot_leastsq_results(meg_data_dic, cond, elec_data, groups[-1], groups_ps[-1], time_split,
                    same_ps=True, do_plot=True, do_save=True, uuid=uuid)
                new_electrode, elec_ps, elec_err = pick_new_electrode(event_id, bipolar, from_t, to_t, time_split,
                    electrodes, meg_data_dic, elec_data, njobs)
                groups.append([new_electrode])
                groups_errs.append([elec_err])
                groups_ps.append(elec_ps)

            print(groups[-1], groups_errs[-1])
            electrodes.remove(new_electrode)

        run_score = sum(map(np.mean, groups_errs)) + len(groups) * groups_panelty
        if run_score < run_min_score:
            print('new min err!')
            run_min_score = run_score
            utils.save((groups, groups_ps, groups_errs), get_pkl_file(
                '{}_find_best_groups_{}_{}_{}_gp{}.pkl'.format(cond, len(time_split), err_threshold,
                'only_sig' if only_sig_electrodes else 'all', groups_panelty)))
        print('{} run: {}, run score: {}, run min score: {}, groups err: {}, threshold: {}'.format(
                cond, run_num, run_score, run_min_score, map(np.mean, groups_errs), err_threshold))
        utils.save((groups, groups_ps, groups_errs), get_pkl_file(
            '{}_find_best_groups_run_{}_{}_{}_{}_gp{}_{}.pkl'.format(cond, run_num, len(time_split), err_threshold,
            'only_sig' if only_sig_electrodes else 'all', groups_panelty, uuid)))
        run_num += 1


def pick_new_electrode(event_id, bipolar, from_t, to_t, time_split, electrodes, meg_data_dic, elec_data, njobs):
    cond = utils.first_key(event_id)
    new_electrode = random.sample(electrodes, 1)[0]
    elec_errors, elec_ps, opt_cv_params = reconstruct_meg(event_id, [new_electrode], from_t, to_t, time_split,
        plot_results=False, bipolar=bipolar, dont_calc_new_csd=True, all_meg_data=meg_data_dic,
        elec_data=elec_data, njobs=njobs)
    print(new_electrode, elec_errors[cond][new_electrode])
    return new_electrode, elec_ps[cond], elec_errors[cond][new_electrode], opt_cv_params[cond]


def find_best_new_electrode(group_electrodes, other_electrodes, event_id, freqs_bins, meg_data_dic, elec_data,
        from_t, to_t, time_split, bipolar, njobs):
    cond = utils.first_key(event_id)
    errors, ps = {}, {}
    for electrode in other_electrodes:
        elec_errors, elec_ps, opt_cv_params = reconstruct_meg(event_id, freqs_bins, [electrode] + group_electrodes, from_t, to_t, time_split,
            plot_results=False, bipolar=bipolar, dont_calc_new_csd=True, all_meg_data=meg_data_dic,
            elec_data=elec_data, njobs=njobs)
        #  for debug
        # plot_leastsq_results(meg_data_dic, cond, elec_data, [electrode] + group_electrodes, elec_ps[cond], time_split, same_ps=True, do_plot=True)
        errors[electrode] = max([err for elc, err in elec_errors[cond].items()])
        ps[electrode] = elec_ps[cond]
    best_electrode = min(errors, key=errors.get)
    min_err = errors[best_electrode]
    best_ps = ps[best_electrode]
    return best_electrode, min_err, best_ps


def find_best_predictive_subset(event_id, bipolar, freqs_bins, from_t, to_t, time_split, k=4,
        only_sig_electrodes=False, check_only_pred_score=True, only_from_same_lead=False,
        electrodes_positive=False, electrodes_normalize=False, electrodes_subtract_mean=False,
        gk_sigma=3, error_threshold=20, uuid_len=5, optimization_method='Ridge', error_calc_method='RMS',
        do_plot_results=False, do_plot_all_results=False, do_save_partial_results=True,
        combs=None, optimization_params={}, vebrose=True, meg_data_dic=None, elec_data=None,
        all_electrodes=None, save_results=True, njobs=4):
    if vebrose:
        print('find_best_predictive_subset:\nk={}, optimization_method={} '.format(k, optimization_method) +
              'error_threshold={}, electrodes_positive={} '.format(error_threshold, str(electrodes_positive)[0]) +
              'electrodes_normalize={}'.format(str(electrodes_normalize)[0]))
    if only_from_same_lead and only_sig_electrodes:
        raise Exception("Can't handle only_from_same_lead and only_sig_electrodes!")

    cond = utils.first_key(event_id)
    uuid = results_fol = ''
    if save_results:
        results_fol = get_results_fol(optimization_method, electrodes_normalize, electrodes_positive)
        utils.make_dir(results_fol)
        uuid, output_file = bps_find_unique_results_name(cond, k, results_fol, optimization_params, uuid_len)
        if do_save_partial_results:
            utils.make_dir(os.path.join(results_fol, uuid))

    if all_electrodes is None:
        if only_sig_electrodes:
            sig_electrodes = find_significant_electrodes(event_id, bipolar, from_t, to_t, do_plot=False,
                do_save=False, plot_only_sig=False)
            all_electrodes = sig_electrodes[cond]
        else:
            all_electrodes = get_all_electrodes_names(bipolar)

    if elec_data is None:
        elec_data = load_electrodes_data(event_id, bipolar, all_electrodes, from_t, to_t,
            subtract_min=electrodes_positive, normalize_data=electrodes_normalize,
            subtract_mean=electrodes_subtract_mean)
        if len(set(all_electrodes) - set(elec_data.keys())) > 0:
            print('data_electrodes_set - all_electrodes_set:')
            print(set(all_electrodes) - set(elec_data.keys()))
            raise Exception('Not the same electrodes in all_electrodes and electodes_data!')

    if meg_data_dic is None:
        meg_data_dic = load_all_dics(freqs_bins, event_id, bipolar, all_electrodes, from_t, to_t, gk_sigma,
            dont_calc_new_csd=True, njobs=njobs)
    take_only_first_prediction = not combs is None
    if combs is None:
        if only_from_same_lead:
            combs = get_lead_groups(k, bipolar)
        else:
            electrodes = set(meg_data_dic.keys())
            combs = list(itertools.combinations(electrodes, k))
    N = len(combs)
    # np.random.shuffle(combs)
    combs_chuncked = utils.chunks(combs, int(N / njobs))
    params = [(comb_chuncked, event_id, elec_data, meg_data_dic, freqs_bins, from_t, to_t, time_split, bipolar, k, check_only_pred_score, vebrose,
               optimization_method, error_calc_method, optimization_params, error_threshold, run, int(N / njobs), uuid, do_plot_results, do_plot_all_results,
               do_save_partial_results, results_fol, take_only_first_prediction) for run, comb_chuncked in enumerate(combs_chuncked)]
    runs_results = utils.run_parallel(_find_best_predictive_subset_parallel, params, njobs)
    all_results = []
    for run_results in runs_results:
        all_results.extend(run_results)
    if save_results:
        if vebrose:
            print('saving results in {}'.format(output_file))
        utils.save(all_results, output_file)
    return all_results


def bps_find_unique_results_name(cond, k, results_fol, optimization_params, uuid_len=10):
    uuid = utils.rand_letters(uuid_len)
    # find unique file name
    params_suffix = utils.params_suffix(optimization_params)
    output_file = os.path.join(results_fol, 'bps_{}_{}_{}{}.pkl'.format(cond, k, uuid, params_suffix))
    while os.path.isfile(output_file):
        uuid = utils.rand_letters(uuid_len)
        output_file = os.path.join(results_fol, 'bps_{}_{}_{}{}.pkl'.format(cond, k, uuid, params_suffix))
    return uuid, output_file


def bps_find_unique_thread_name(thread_fol, uuid_len=5):
    thread_uuid = utils.rand_letters(uuid_len)
    thred_output = os.path.join(thread_fol, '{}.pkl'.format(thread_uuid))
    while os.path.isfile(thred_output):
        thread_uuid = utils.rand_letters(uuid_len)
        thred_output = os.path.join(thread_fol, '{}.pkl'.format(thread_uuid))
    return thred_output


def _find_best_predictive_subset_parallel(params_chunks):
    (comb_chuncked, event_id, elec_data, meg_data_dic, freqs_bins, from_t, to_t, time_split, bipolar, k, check_only_pred_score, vebrose,
     optimization_method, error_calc_method, optimization_params, error_threshold, run, N, uuid, do_plot_results,
     do_plot_all_results, do_save_partial_results, results_fol, take_only_first_prediction) = params_chunks
    results = []
    if do_save_partial_results:
        thread_fol = os.path.join(results_fol, uuid)
        thred_output = bps_find_unique_thread_name(uuid, thread_fol)
    # todo
    for run, comb in enumerate(comb_chuncked):
        cond = utils.first_key(event_id)
        if run % 1000 == 0 and vebrose:
            print('{}/{}'.format(run, N))
        for predicted_electrode in comb:
            train_electrodes = [e for e in comb if e != predicted_electrode]
            elec_errors, elec_ps, opt_cv_params = reconstruct_meg(event_id, freqs_bins, train_electrodes, from_t, to_t, time_split,
                optimization_method = optimization_method, error_calc_method=error_calc_method, optimization_params=optimization_params,
                predicted_electrodes=[predicted_electrode], plot_results=do_plot_all_results, bipolar=bipolar,
                dont_calc_new_csd=True, all_meg_data=meg_data_dic, elec_data=elec_data, njobs=1)
            if elec_ps is None or elec_errors is None:
                print('No meg reconstruction for {}!'.format(train_electrodes + [predicted_electrode]))
                continue
            if check_only_pred_score:
                good_result = elec_errors[cond][predicted_electrode] < error_threshold
            else:
                good_result = np.all(np.array(list(elec_errors[cond].values())) < error_threshold)
            if good_result:
                if vebrose:
                    errors_str = ','.join(['{:.2f}'.format(elec_errors[cond][elec]) for elec in train_electrodes + [predicted_electrode]])
                    print('{}->{}: {}'.format(train_electrodes, predicted_electrode, errors_str))#, opt_cv_params[cond]))
                if do_plot_results:
                    plot_leastsq_results(meg_data_dic, cond, elec_data, train_electrodes, elec_ps[cond],
                        time_split, optimization_method, [predicted_electrode], do_plot=True, do_save=False,
                        uuid=uuid, res_ind=utils.rand_letters(3))
                # errors = copy.deepcopy(elec_errors[cond])
                # ps = copy.deepcopy(elec_ps[cond])
                # train = copy.deepcopy(train_electrodes)
                results.append((predicted_electrode, train_electrodes, elec_errors[cond], elec_ps[cond], opt_cv_params[cond]))
                if do_save_partial_results:
                    utils.save(results, thred_output)
            if take_only_first_prediction:
                break
    return results


def best_predictive_subset_collect_results(event_id, freqs_bin, bipolar, from_t, to_t, time_split, uuid, k=3, gk_sigma=3,
        electrodes_positive=False, electrodes_normalize=False, electrodes_subtract_mean=False,
        error_threshold=10, optimization_method='', elec_data=None,
        error_calc_method='RMS', sort_only_accoring_to_pred=True, calc_all_errors=False, dtw_window=10,
        do_save=False, do_plot=True, save_in_pred_electrode_fol=False, write_errors_csv=False, optimization_params={},
        do_plot_electrodes=False, error_functions=(), check_only_pred_score=True, do_plot_together=False, njobs=4):
    print('best_predictive_subset_collect_results:\nk={}, optimization_method={} '.format(k, optimization_method) +
          'error_calc_method={} error_threshold={} '.format(error_calc_method, error_threshold) +
          'electrodes_positive={} electrodes_normalize={}'.format(str(electrodes_positive)[0], str(electrodes_normalize)[0]))
    cond = utils.first_key(event_id)
    electrodes = get_all_electrodes_names(bipolar)
    if elec_data is None:
        elec_data = load_electrodes_data(event_id, bipolar, electrodes, from_t, to_t,
            subtract_min=electrodes_positive, normalize_data=electrodes_normalize, subtract_mean=electrodes_subtract_mean)
    meg_data_dic = load_all_dics(freqs_bin, event_id, bipolar, electrodes, from_t, to_t, gk_sigma, njobs=njobs)
    if len(error_functions) == 0:
        error_functions = ERROR_RECONSTRUCT_METHODS
    results, results_fol = bps_load_results(cond, uuid, k, optimization_method, electrodes_normalize, electrodes_positive, optimization_params)
    results, errors, results_errors = sort_results(event_id, results, meg_data_dic, elec_data, time_split,
        error_calc_method, sort_only_accoring_to_pred, calc_all_errors, dtw_window)
    results_num = sum([np.all(result_errors < error_threshold) for result_errors in results_errors])
    print('{} results < error_threshold'.format(results_num))
    utils.save((errors, results_errors), os.path.join(results_fol, 'bps_errors_{}_{}_{}.pkl'.format(k, uuid, error_calc_method)))
    if write_errors_csv:
        csv_file = open(os.path.join(results_fol, 'bps_errors_{}_{}_{}.csv'.format(k, uuid, error_calc_method)), 'w')
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerows([['index', 'predicted', 'train'] + error_functions])
    parc = get_electrodes_parcellation(electrodes, bipolar)
    for res_ind, (result, result_errors) in enumerate(zip(results, results_errors)):
        good_result = result_errors[-1] < error_threshold if check_only_pred_score else \
            np.all(result_errors < error_threshold)
        if good_result:
            predicted_electrode, train_electrodes, elecs_errors, ps, cv_params = open_bps_result(result)
            if res_ind == 0:
                print('#freqs: {}'.format(ps.shape[1]))
            print('{}) {}->{}: {}'.format(res_ind, train_electrodes, predicted_electrode, result_errors))
            if write_errors_csv:
                 write_error_line(csv_writer, res_ind, cond, elec_data, predicted_electrode, train_electrodes, ps, meg_data_dic,
                    time_split, dtw_window)
            print_parc_info(parc, predicted_electrode, train_electrodes, k=-1)
            plot_leastsq_results(meg_data_dic, cond, elec_data, train_electrodes, ps, time_split,
                predicted_electrodes=[predicted_electrode], same_ps=True, res_ind=res_ind,
                do_save=do_save, uuid=uuid, do_plot=do_plot, optimization_method=optimization_method,
                save_in_pred_electrode_fol=save_in_pred_electrode_fol, error_functions=error_functions)
            if do_plot_together:
                plt.figure()
                plt.plot(elec_data[predicted_electrode][cond], label=predicted_electrode)
                plt.plot(elec_data[train_electrodes[0]][cond], label=train_electrodes[0])
                plt.plot(elec_data[train_electrodes[1]][cond], label=train_electrodes[1])
                plt.legend()
                plt.show()
            if do_plot_electrodes:
                plot_electrodes(bipolar, [predicted_electrode] + train_electrodes)

    if write_errors_csv:
        csv_file.close()


def print_parc_info(parc, predicted_electrode, train_electrodes, k=3):
    if parc:
        for electrode, elec_type in zip([predicted_electrode] + train_electrodes, ['pred'] + ['train'] * len(train_electrodes)):
            elec_parc = sorted([(float(prob), region) for region, prob in parc[electrode].items()], reverse=True)
            elec_parc_str = ['{}: {:.4f}'.format(region, prob) for prob, region in elec_parc]
            print('{}: {} in {}'.format(elec_type, electrode, elec_parc_str[:k]))


def write_error_line(csv_writer, res_ind, cond, elec_data, predicted_electrode, train_electrodes, ps, meg_data_dic,
        time_split, dtw_window):
    time_diff = np.diff(time_split)[0] if len(time_split) > 1 else 500
    meg = combine_meg_chunks(meg_data_dic[predicted_electrode], ps, time_split, time_diff)
    err_func = partial(electrode_reconstruction_error, electrode=predicted_electrode, meg=meg,
        electrode_data=elec_data[predicted_electrode][cond], electrode_ps=ps,
        meg_data_dic=meg_data_dic, time_split=time_split, time_diff=time_diff, dtw_window=dtw_window)
    errors_strs = ['{:.2f}'.format(err_func(error_calc_method=em)) for em in ERROR_RECONSTRUCT_METHODS]
    csv_writer.writerows([[res_ind, predicted_electrode, train_electrodes] + errors_strs])


def bps_load_results(cond, uuid, k, optimization_method, electrodes_normalize, electrodes_positive, optimization_params):
    results_fol = get_results_fol(optimization_method, electrodes_normalize, electrodes_positive)
    params_suffix = '' if len(optimization_params) == 0 else \
        ''.join(sorted(['_{}_{}'.format(param_key, param_val) for param_key, param_val in sorted(optimization_params.items())]) )
    results_file = os.path.join(results_fol, 'bps_{}_{}_{}{}.pkl'.format(cond, k, uuid, params_suffix))
    if os.path.isfile(results_file):
        results = utils.load(results_file)
    else: # Try to collect partial results
        results = []
        partial_files = glob.glob(os.path.join(results_fol, uuid, '*.pkl'))
        for partial_file in partial_files:
            results.extend(utils.load(partial_file))
        if len(results) == 0:
            raise Exception('No results found!')
    return results, results_fol


def sort_results(event_id, results, meg_data_dic, electrodes_data, time_split,
                 error_calc_method='RMS', only_accoring_to_pred=True, re_calc_all_errors=False,
                 dtw_window=10, max_error_threshold=np.inf):
    if not re_calc_all_errors:
        results_errors = [[res_errors[elec] for elec in train + [pred]] for pred, train, res_errors, _, _ in results
                          if res_errors[pred] < max_error_threshold]
        if only_accoring_to_pred:
            errors = [res_errors[pred] for pred, _, res_errors, _, _ in results
                if res_errors[pred] < max_error_threshold]
        else:
            errors = copy.copy(results_errors)
        sorted_results = [(pred, train, res_errors, ps, params) for pred, train, res_errors, ps, params in results
                          if res_errors[pred] < max_error_threshold]
    else:
        sorted_results, errors, results_errors = [], [], []
        cond = utils.first_key(event_id)
        time_diff = np.diff(time_split)[0] if len(time_split) > 1 else 500
        now = time.time()
        for ind, result in enumerate(results):
            predicted_electrode, train_electrodes, elecs_errors, electrode_ps, cv_params = open_bps_result(result)
            if ind % 100 == 0:
                print('sorting ({}) {}/{} {}'.format(error_calc_method, ind, len(results), time.time() - now))
            err = 0
            electrodes_errors = []
            for electrode in train_electrodes + [predicted_electrode]:
                elec_err = electrode_reconstruction_error(electrode, electrodes_data[electrode][cond], electrode_ps,
                    meg_data_dic, error_calc_method, time_split, time_diff, dtw_window)
                electrodes_errors.append(elec_err)
                if (only_accoring_to_pred and electrode == predicted_electrode) or not only_accoring_to_pred:
                    err += elec_err
            # Check if the predicted electrode's error is below the threshld
            if electrodes_errors[-1] < max_error_threshold:
                sorted_results.append(result)
                errors.append(err)
                results_errors.append(electrodes_errors)
    sorted_results = [res for (err, res) in sorted(zip(errors, sorted_results))]
    results_errors = [res_err for (err, res_err) in sorted(zip(errors, results_errors))]
    errors = sorted(errors)
    return sorted_results, np.array(errors), np.array(results_errors)


def open_bps_result(result):
    if len(result) == 5:
        predicted_electrode, train_electrodes, elecs_errors, electrode_ps, cv_params = result
    else:
        predicted_electrode, train_electrodes, elecs_errors, electrode_ps = result
        cv_params = []
    return predicted_electrode, train_electrodes, elecs_errors, electrode_ps, cv_params


def bps_collect_results_gs(uuid, event_id, optimization_method, error_calc_method, from_t, to_t, time_split, gk_sigma=3, bipolar=True,
        electrodes_positive=False, electrodes_normalize=False, dtw_window=10, do_plot=True, error_threshold=10, write_csv=False, njobs=4):
    cond = utils.first_key(event_id)
    time_diff = np.diff(time_split)[0] if len(time_split) > 1 else 500
    electrodes = get_all_electrodes_names(bipolar)
    elec_data = load_electrodes_data(event_id, bipolar, electrodes, from_t, to_t,
        subtract_min=electrodes_positive, normalize_data=electrodes_normalize)
    meg_data_dic = load_all_dics(freqs_bin, event_id, bipolar, electrodes, from_t, to_t, gk_sigma, njobs=njobs)
    results_fol = os.path.join(get_results_fol(optimization_method, electrodes_normalize, electrodes_positive),
        'params_grid_search_{}'.format(uuid))

    gs_results = defaultdict(lambda : defaultdict(list))
    gs_files = glob.glob(os.path.join(results_fol, '*.pkl'))
    gs_files = sorted(gs_files, key=lambda x:int(utils.namebase(x).split('_')[-3]))
    alphas = [int(utils.namebase(x).split('_')[-3]) for x in gs_files]
    if write_csv:
        csv_file = open(os.path.join(results_fol, 'grid_search.csv'), 'w')
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerows([['predicted', 'train'] + alphas])
    for result_fname in gs_files:
        alpha = int(utils.namebase(result_fname).split('_')[-3])
        results = utils.load(result_fname)
        # results, _, prediction_errors = sort_results(event_id, results, meg_data_dic, elec_data, time_split,
        #     error_calc_method=error_calc_method, only_accoring_to_pred=False, calc_all_errors=True, dtw_window=dtw_window)

        for result in results:
            predicted_electrode, train_electrodes, _, ps, _ = open_bps_result(result)
            errors = []
            for electrode in [predicted_electrode] + train_electrodes:
                errors.append(electrode_reconstruction_error(electrode, elec_data[electrode][cond], ps,
                    meg_data_dic, error_calc_method, time_split, time_diff, dtw_window))
            gs_results[(predicted_electrode, tuple(train_electrodes))][alpha] = (errors, ps)

    best_alpha = {}
    for (predicted_electrode, train_electrodes), pred_results in  gs_results.items():
        pred_results = utils.sort_dict_by_values(pred_results)
        pred_errors = [result_info[0][0] for result_info in pred_results.values()]
        train1_errors = [result_info[0][1] for result_info in pred_results.values()]
        train2_errors = [result_info[0][2] for result_info in pred_results.values()]
        # min_train_index = np.argmin(train_errors)
        # best_alpha[predicted_electrode] = alphas[min_train_index]
        if write_csv:
            result_errors_str = ['{:.2f}'.format(err) for err in pred_errors]
            csv_writer.writerows([[predicted_electrode, train_electrodes] + result_errors_str])
        if do_plot and min(pred_errors) < error_threshold:
            plt.figure()
            plt.plot(pred_errors, label='pred')
            plt.plot(train1_errors, label='train1')
            plt.plot(train2_errors, label='train2')
            plt.title('{} {}'.format(predicted_electrode, train_electrodes))
            plt.legend()
            plt.show()


def plot_reconstruction_for_different_freqs(event_id, electrode, two_electrodes, from_t, to_t, time_split,
        gk_sigma=3, bipolar=True, electrodes_positive=False, electrodes_normalize=False, njobs=4):
    cond = utils.first_key(event_id)
    electrodes = get_all_electrodes_names(bipolar)
    elec_data = load_electrodes_data(event_id, bipolar, electrodes, from_t, to_t,
        subtract_min=electrodes_positive, normalize_data=electrodes_normalize)
    meg_data_dic = load_all_dics(freqs_bin, event_id, bipolar, electrodes, from_t, to_t, gk_sigma, njobs=njobs)
    reconstruct_meg(event_id, [electrode], from_t, to_t, time_split, plot_results=True, all_meg_data=meg_data_dic,
        elec_data=elec_data, title='{}: {}'.format(cond, electrode))
    reconstruct_meg(event_id, two_electrodes, from_t, to_t, time_split, optimization_method='RidgeCV',
        plot_results=True, all_meg_data=meg_data_dic,elec_data=elec_data,
        title='{}: {} and {}'.format(cond, two_electrodes[0], two_electrodes[1]))
    freqs_inds = np.array([2, 6, 9, 10, 11, 15, 16])
    plt.plot(elec_data[electrode][cond])
    plt.plot(meg_data_dic[electrode][freqs_inds, :].T, '--')
    plt.legend([electrode] + np.array(CSD_FREQS)[freqs_inds].tolist())
    # plt.title('{}: {}'.format(cond, electrode))
    plt.show()


def plot_predictive_subset(electrodes, event_id, bipolar, from_t, to_t, time_split,
                           elec_data, meg_data_dic, gk_sigma=3, electrodes_positive=True,
                           electrodes_normalize=True, njobs=4):
    for predicted_electrode in electrodes:
        train_electrodes = [e for e in electrodes if e != predicted_electrode]
        reconstruct_meg(event_id, train_electrodes, from_t, to_t, time_split, predicted_electrodes=[predicted_electrode],
                        plot_results=True, bipolar=bipolar, dont_calc_new_csd=True, all_meg_data=meg_data_dic,
                        elec_data=elec_data, njobs=1)


def calc_lead_predictiveness(event_id, bipolar, from_t, to_t, time_split, gk_sigma=3,
        electrodes_positive=True, electrodes_normalize=True, k=3, error_threshold=10, njobs=4):
    cond = utils.first_key(event_id)
    electrodes = get_all_electrodes_names(True)
    elec_data = load_electrodes_data(event_id, bipolar, electrodes, from_t, to_t,
        subtract_min=electrodes_positive, normalize_data=electrodes_normalize)
    meg_data_dic = load_all_dics(freqs_bin, event_id, bipolar, electrodes, from_t, to_t, gk_sigma, njobs=njobs)
    electrodes = set(meg_data_dic.keys())
    elecs_groups = get_lead_groups(k)
    good_groups = []
    for elecs in elecs_groups:
        group_errors, group_pss, group_predicted = [], [], []
        if not np.all([e in electrodes for e in elecs]):
            continue
        for predicted_electrode in elecs:
            train_electrodes = [e for e in elecs if e != predicted_electrode]
            elec_errors, elec_ps = reconstruct_meg(event_id, train_electrodes, from_t, to_t, time_split, predicted_electrodes=[predicted_electrode],
                                                   plot_results=False, bipolar=bipolar, dont_calc_new_csd=True, all_meg_data=meg_data_dic,
                                                   elec_data=elec_data, njobs=1)
            group_errors.append(elec_errors[cond][predicted_electrode])
            group_pss.append(elec_ps[cond])
            group_predicted.append(predicted_electrode)
        good_groups.append((elecs, group_errors, group_pss, group_predicted))
        if min(group_errors) < error_threshold:
            print('low error!')
            print(group_errors, group_predicted)
    utils.save(good_groups, get_pkl_file('{}_calc_lead_predictiveness_{}.pkl'.format(cond, k)))


def plot_lead_predictiveness(event_id, bipolar, from_t, to_t, time_split, gk_sigma=3,
        electrodes_positive=True, electrodes_normalize=True, k=3, error_threshold=10, njobs=4):
    cond = utils.first_key(event_id)
    good_groups = utils.load(get_pkl_file('{}_calc_lead_predictiveness_{}.pkl'.format(cond, k)))
    electrodes = get_all_electrodes_names(True)
    elec_data = load_electrodes_data(event_id, bipolar, electrodes, from_t, to_t,
        subtract_min=electrodes_positive, normalize_data=electrodes_normalize)
    meg_data_dic = load_all_dics(freqs_bin, event_id, bipolar, electrodes, from_t, to_t, gk_sigma, njobs=njobs)
    for elecs, group_errors, group_pss, group_predicted in good_groups:
        if min(group_errors) < error_threshold:
            best_index = np.argmin(group_errors)
            best_predictive_elc = group_predicted[best_index]
            print(best_predictive_elc, group_errors[best_index])
            best_ps = group_pss[best_index]
            plot_leastsq_results(meg_data_dic, cond, elec_data, elecs, best_ps, time_split,
                same_ps=True, do_plot=True, do_save=False)


def get_lead_groups(k, bipolar):
    electrodes = get_all_electrodes_names(False)
    groups = defaultdict(list)
    elecs_groups = []
    for electrode in electrodes:
        elc_group, elc_num = utils.elec_group_number(electrode)
        groups[elc_group].append(elc_num)
    for group_name, nums in groups.items():
        max_num = max(nums) - k if bipolar else max(nums) - k + 1
        for num in range(max_num):
            if bipolar:
                elecs_groups.append(['{}{}-{}{}'.format(group_name, num+l+1, group_name, num+l) for l in range(1, k+1)])
            else:
                elecs_groups.append(['{}{}'.format(group_name, num+l) for l in range(1, k+1)])
    return elecs_groups


def get_inter_lead_combs(inter_leads_groups, bipolar, calc_groups_product):
    electrodes = get_all_electrodes_names(False)
    groups = defaultdict(list)
    elecs_groups = []
    for electrode in electrodes:
        elc_group, elc_num = utils.elec_group_number(electrode)
        groups[elc_group].append(elc_num)
    for group in groups.keys():
        groups[group] = sorted(groups[group])
    for inter_leads_group in inter_leads_groups:
        if calc_groups_product:
            electrodes_groups = [['{}{}'.format(group, k) for k in groups[group]] for group in inter_leads_group]
            elecs_groups.extend(product(*electrodes_groups))
        else:
            max_k = min(map(max, [groups[inter_leads_group[i]] for i in range(len(inter_leads_group))]))
            for k in range(1, max_k + 1):
                if not bipolar:
                    elecs_groups.append(['{}{}'.format(group, k) for group in inter_leads_group])
    return elecs_groups


def analyze_best_predictive_subset(k, event_id, bipolar, from_t, to_t, time_split, gk_sigma=3, electrodes_positive=True, electrodes_normalize=True, njobs=1):
    cond = utils.first_key(event_id)
    elec_data = load_electrodes_data(event_id, bipolar, from_t, to_t,
        subtract_min=electrodes_positive, normalize_data=electrodes_normalize)
    electrodes = elec_data.keys()
    meg_data_dic = load_all_dics(freqs_bin, event_id, bipolar, electrodes, from_t, to_t, gk_sigma, njobs=njobs)
    min_err = min([float(utils.namebase(fname).split('_')[-1]) for fname in glob.glob(get_pkl_file('best_predictive_subset_{}_*.pkl'.format(k)))])
    min_comb, min_err, min_ps = utils.load(get_pkl_file('best_predictive_subset_{}_{}.pkl'.format(k, min_err)))
    plot_leastsq_results(meg_data_dic, cond, elec_data, min_comb, min_ps, time_split,
        same_ps=True, do_plot=True, do_save=False)


def load_electrodes_data(events_id, bipolar, electrodes=None, from_t=None, to_t=None,
        subtract_min=False, normalize_data=False, subtract_mean=False):
    meg_elec_conditions_translate = {'interference':1, 'neutral':0}
    d = np.load(os.path.join(BLENDER_SUB_FOL, 'electrodes{}_data.npz'.format('_bipolar' if bipolar else '')))
    data, names, elecs_conditions = d['data'], d['names'], d['conditions']
    elec_data = defaultdict(dict)
    if from_t is None:
        from_t = 0
    if to_t is None:
        to_t = -1
    if electrodes is None:
        electrodes = names
    if subtract_mean:
        data -= np.mean(data, axis=0)
    for electrode, elec_data_mat in zip(electrodes, data):
        for cond in events_id:
            ind = meg_elec_conditions_translate[cond]
            data = elec_data_mat[:, ind]
            if not from_t is None and not to_t is None:
                data = data[from_t: to_t]
            if subtract_min:
                data = data - min(data)
            if normalize_data:
                data = data * 1.0/max(data)
            elec_data[electrode][cond] = data

    return elec_data


def analyze_best_groups(event_id, bipolar, from_t, to_t, time_split, err_threshold=10, groups_panelty=5, only_sig_electrodes=False,
        electrodes_positive=True, electrodes_normalize=True, gk_sigma=3, njobs=4):
    cond = utils.first_key(event_id)
    # electrodes_parc = get_electrodes_parcellation(electrodes, bipolar)
    elec_data = load_electrodes_data(event_id, bipolar, from_t, to_t,
        subtract_min=electrodes_positive, normalize_data=electrodes_normalize)
    meg_data_dic = load_all_dics(freqs_bin, event_id, bipolar, elec_data.keys(), from_t, to_t, gk_sigma, njobs=njobs)
    print('analyze_best_groups', cond, err_threshold)
    res_fname = get_pkl_file(
        '{}_find_best_groups_{}_{}_{}_gp{}.pkl'.format(cond, len(time_split), err_threshold,
        'only_sig' if only_sig_electrodes else 'all', groups_panelty))
    if os.path.isfile(res_fname):
        groups, groups_ps, groups_err = utils.load(res_fname)
        print(len(groups))
        print(map(len, groups))
        print(map(np.mean, groups_err))
        for electrodes_group, electrodes_ps, group_err in zip(groups, groups_ps, groups_err):
            # for elec in electrodes_group:
            #     print(elec, electrodes_parc[elec])
            # print(electrodes_group, group_err)
            # plot_leastsq_results(meg_data_dic, cond, elec_data, electrodes_group, electrodes_ps, time_split,
            #     same_ps=True, do_plot=True)
            for elec in electrodes_group:
                electrodes_train = [e for e in electrodes_group if e!=elec]
                reconstruct_meg(event_id, electrodes_train, from_t, to_t, time_split, predicted_electrodes=[elec],
                                plot_results=True, bipolar=bipolar, dont_calc_new_csd=True, all_meg_data=meg_data_dic,
                                elec_data=elec_data, calc_ridge=False, njobs=njobs)
                reconstruct_meg(event_id, electrodes_train, from_t, to_t, time_split, predicted_electrodes=[elec],
                                plot_results=True, bipolar=bipolar, dont_calc_new_csd=True, all_meg_data=meg_data_dic,
                                elec_data=elec_data, calc_ridge=True, njobs=njobs)

            # plot_electrodes(bipolar, electrodes_group)

    else:
        print('No such file {}'.format(res_fname))




    # elecctrodes = get_all_electrodes_names(bipolar)
    # elec_data = load_all_electrodes(electrodes, positive=True, normalize_data=True)
    # for cond in events_id:
    #     cum_errors, cum_electrodes, ps_electrodes = utils.load(get_pkl_file('{}_find_best_fit_electrodes_{}.pkl'.format(cond, len(time_split))))
    #     plt.figure()
    #     plt.plot(cum_errors)
    #     plt.title(cond)
    #     plt.show()
    #     meg_data_dic = load_all_dics(freqs_bin, cond, bipolar, electrodes, from_t, to_t, gk_sigma, njobs)
    #     # plot_leastsq_results(meg_data_dic, cond, elec_data, cum_electrodes, ps_electrodes, time_split)


def plot_leastsq_results(meg_data_dic, cond, elec_data, electrodes, electrodes_ps, time_split, optimization_method,
        predicted_electrodes=[], same_ps=True, do_plot=True, do_save=False, uuid='', title=None, full_title=True,
        error_functions=(), res_ind=0, save_in_pred_electrode_fol=False, dtw_window=10, tight_layout=True):
    electrodes = list(electrodes)
    if uuid=='':
        uuid = utils.rand_letters(5)
    time_diff = np.diff(time_split)[0] if len(time_split) > 1 else 500
    # ps_range = range(0, len(time_split) * (len(CSD_FREQS) + 1), len(CSD_FREQS) + 1)
    if same_ps:
        electrodes_ps = [electrodes_ps] * (len(electrodes) + len(predicted_electrodes))
    pics_num_x, pics_num_y = utils.how_many_subplots(len(electrodes) + len(predicted_electrodes))
    f, axs = plt.subplots(pics_num_x, pics_num_y, sharex='col', sharey='row', figsize=(12, 8))
    if pics_num_x==1 and pics_num_y==1:
        axs = [axs]
    elif pics_num_x>1 and pics_num_y>1:
        axs = list(itertools.chain(*axs))
    if do_save:
        fig_fol = os.path.join(get_figs_fol(), optimization_method, uuid)
        utils.make_dir(fig_fol)
    if len(error_functions)==0:
        error_functions = ERROR_RECONSTRUCT_METHODS
    electrode_types = ['training'] * len(electrodes) + ['prediction'] * len(predicted_electrodes)
    for electrode, electrode_ps, ax, electrode_type in zip(electrodes + predicted_electrodes,
            electrodes_ps, axs, electrode_types):
        meg = combine_meg_chunks(meg_data_dic[electrode], electrode_ps, time_split, time_diff)
        err_func = partial(electrode_reconstruction_error, electrode=electrode, meg=meg,
            electrode_data=elec_data[electrode][cond], electrode_ps=electrode_ps,
            meg_data_dic=meg_data_dic, time_split=time_split, time_diff=time_diff, dtw_window=dtw_window)
        errors = {em:err_func(error_calc_method=em) for em in error_functions}
        errors_str = ''.join(['{}:{:.2f} '.format(em, errors[em]) for em in error_functions])
        ax.plot(elec_data[electrode][cond], label=electrode)
        ax.plot(meg, label='plsq')
        # ax.set_ylim([0, 1])
        # ax.legend()
        if title is None:
            if full_title:
                ax.set_title('{}) {}: {} '.format(res_ind, electrode_type, electrode) + errors_str)
            else:
                ax.set_title('{}: {} ({:.2f})'.format(electrode_type, electrode, err_func(error_calc_method=error_functions[0])))
    if not title is None:
        axs[0].set_title(title)
    if tight_layout:
        plt.tight_layout()
    if do_save:
        if save_in_pred_electrode_fol == True and len(predicted_electrodes)==1:
            elec_fol = os.path.join(fig_fol, predicted_electrodes[0])
            utils.make_dir(elec_fol)
            fig_fname = os.path.join(elec_fol, '{}_{}.png'.format(uuid, res_ind))
        else:
            fig_fname = os.path.join(fig_fol, '{}_{}.png'.format(uuid, res_ind))
        f.savefig(fig_fname)
        print('saved to {}'.format(fig_fname))
    if do_plot and not do_save:
        plt.show()
    else:
        plt.close()


def combine_meg_chunks(electrod_meg_data, electrode_ps, time_split, time_diff):
    meg = []
    for ps, from_t, to_t in zip(electrode_ps, time_split, time_split + time_diff):
        meg_chunk = meg_pred(ps, electrod_meg_data[:, from_t:to_t])
        meg = meg_chunk if len(meg) == 0 else np.hstack((meg, meg_chunk))
    return meg


def get_all_electrodes_names(bipolar):
    subject_mri_dir = os.path.join(SUBJECTS_MRI_DIR, MRI_SUBJECT)
    positions_file_name = 'electrodes{}_positions.npz'.format('_bipolar' if bipolar else '')
    positions_file_name = os.path.join(subject_mri_dir, 'electrodes', positions_file_name)
    d = np.load(positions_file_name)
    names = d['names']
    names = [elc.astype(str) for elc in names if not elc in BAD_ELECS]
    return names


def calc_p_for_each_electrode(event_id, bipolar, from_t, to_t, time_split, gk_sigma=3,
        electrodes_positive=True, electrodes_normalize=True, outliers=[],
        only_sig_electrodes=False, plot_results=False, njobs=4):
    cond = utils.first_key(event_id)
    if only_sig_electrodes:
        sig_electrodes = find_significant_electrodes(event_id, bipolar, from_t, to_t, do_plot=False, do_save=False, plot_only_sig=False)
        electrodes = sig_electrodes[cond]
    else:
        electrodes = get_all_electrodes_names(bipolar)

    errors, pss = [], []
    electrodes = [elc for elc in electrodes if elc not in outliers]
    elec_data = load_electrodes_data(event_id, bipolar, electrodes, from_t, to_t,
        subtract_min=electrodes_positive, normalize_data=electrodes_normalize)
    meg_data_dic = load_all_dics(freqs_bin, event_id, bipolar, electrodes, from_t, to_t, gk_sigma, dont_calc_new_csd=True, njobs=njobs)
    electrodes = meg_data_dic.keys()
    chunked_electrodes = utils.chunks(electrodes, len(electrodes) / njobs)
    params = [(chunk_electrodes, bipolar, event_id, from_t, to_t, time_split, gk_sigma, meg_data_dic, elec_data) for chunk_electrodes in chunked_electrodes]
    results = utils.run_parallel(_calc_p_for_each_electrode_parallel, params, njobs)
    for chunk_errors, chunk_pss in results:
        errors.extend(chunk_errors)
        pss.extend(chunk_pss)
    utils.save((electrodes, errors, pss), get_pkl_file('{}_calc_p_for_each_electrode.pkl'.format(cond)))


def _calc_p_for_each_electrode_parallel(chunked_params):
    errors, pss = [], []
    electrodes, bipolar, event_id, from_t, to_t, time_split, gk_sigma, meg_data_dic, elec_data = chunked_params
    cond = utils.first_key(event_id)
    for electrode in electrodes:
        elec_errors, elec_ps = reconstruct_meg(event_id, [electrode], from_t, to_t, time_split, gk_sigma=gk_sigma,
                                               plot_results=False, bipolar=bipolar, dont_calc_new_csd=True, all_meg_data=meg_data_dic,
                                               elec_data=elec_data, njobs=1)
        errors.append(elec_errors[cond][electrode])
        pss.append(elec_ps[cond])
    return errors, pss


def analyze_p_for_each_electrode(event_id, bipolar, from_t, to_t, time_split, gk_sigma=3,
        electrodes_positive=True, electrodes_normalize=True, njobs=4):
    cond = utils.first_key(event_id)
    electrodes, errors, pss = utils.load(get_pkl_file('{}_calc_p_for_each_electrode.pkl'.format(cond)))
    elec_data = load_electrodes_data(event_id, bipolar, electrodes, from_t, to_t,
        subtract_min=electrodes_positive, normalize_data=electrodes_normalize)
    meg_data_dic = load_all_dics(freqs_bin, event_id, bipolar, electrodes, from_t, to_t, gk_sigma, dont_calc_new_csd=True, njobs=njobs)
    # plot_leastsq_results(meg_data_dic, cond, elec_data, electrodes, pss, time_split,
    #     same_ps=False, do_plot=True, do_save=False)
    X = utils.stack(pss)
    find_best_n_componenets(X, electrodes, event_id, bipolar, from_t, to_t, time_split, meg_data_dic, elec_data, gk_sigma)

    # utils.plot_3d_PCA(X)
    # res, best_gmm, bic = utils.calc_clusters_bic(X)

def analyze_best_n_componenets(event_id, bipolar, from_t, to_t, time_split, gk_sigma=3,
        electrodes_positive=True, electrodes_normalize=True, njobs=4):
    cond = utils.first_key(event_id)
    all_clusters, all_errors = utils.load(get_pkl_file('{}_best_n_componenets'.format(cond)))
    electrodes, errors, pss = utils.load(get_pkl_file('{}_calc_p_for_each_electrode.pkl'.format(cond)))
    elec_data = load_electrodes_data(event_id, bipolar, electrodes, from_t, to_t,
        subtract_min=electrodes_positive, normalize_data=electrodes_normalize)
    meg_data_dic = load_all_dics(freqs_bin, event_id, bipolar, electrodes, from_t, to_t, gk_sigma, dont_calc_new_csd=True, njobs=njobs)

    X = utils.stack(pss)
    for k, cluster_error in enumerate(all_errors):
        print(k, cluster_error)
    plt.plot(all_errors)
    plt.show()

    gmm = mixture.GMM(n_components=22, covariance_type='spherical')
    gmm.fit(X)
    clusters = gmm.predict(X)
    unique_clusters = np.unique(clusters)
    cluster_errors = []
    for cluster in unique_clusters:
        cluster_electrodes = np.array(electrodes)[np.where(clusters == cluster)].tolist()
        elec_errors, elec_ps = reconstruct_meg(event_id, cluster_electrodes, from_t, to_t, time_split, gk_sigma=gk_sigma,
                                               plot_results=True, bipolar=bipolar, dont_calc_new_csd=True, all_meg_data=meg_data_dic,
                                               elec_data=elec_data, njobs=1)
        cluster_errors.append(max(elec_errors[cond].values()))


def find_best_n_componenets(X, electrodes, event_id, bipolar, from_t, to_t, time_split, meg_data_dic, elec_data, gk_sigma):
    cond = utils.first_key(event_id)
    all_errors = []
    all_clusters = []
    for n_components in range(1, X.shape[0]):
        gmm = mixture.GMM(n_components=n_components, covariance_type='spherical')
        gmm.fit(X)
        clusters = gmm.predict(X)
        unique_clusters = np.unique(clusters)
        cluster_errors = []
        for cluster in unique_clusters:
            cluster_electrodes = np.array(electrodes)[np.where(clusters == cluster)].tolist()
            elec_errors, elec_ps = reconstruct_meg(event_id, cluster_electrodes, from_t, to_t, time_split, gk_sigma=gk_sigma,
                                                   plot_results=False, bipolar=bipolar, dont_calc_new_csd=True, all_meg_data=meg_data_dic,
                                                   elec_data=elec_data, njobs=1)
            cluster_errors.append(max(elec_errors[cond].values()))
        print(n_components, max(cluster_errors))
        all_clusters.append(clusters)
        all_errors.append(max(cluster_errors))
    utils.save((all_clusters, all_errors), get_pkl_file('{}_best_n_componenets'.format(cond)))
    plt.plot(all_errors)
    plt.show()


def find_best_subset(event_id, k, bipolar, from_t, to_t, time_split, gk_sigma=3, only_first_subset=False,
        electrodes_positive=True, electrodes_normalize=True, outliers=[], split_to_learn_and_pred=False,
        only_sig_electrodes=False, plot_results=False, njobs=4):
    cond = utils.first_key(event_id)
    if only_sig_electrodes:
        sig_electrodes = find_significant_electrodes(event_id, bipolar, from_t, to_t, do_plot=False, do_save=False, plot_only_sig=False)
        electrodes = sig_electrodes[cond]
    else:
        electrodes = get_all_electrodes_names(bipolar)

    electrodes = [elc for elc in electrodes if elc not in outliers]
    if k==0:
        k = len(electrodes)
    elec_data = load_electrodes_data(event_id, bipolar, electrodes, from_t, to_t,
        subtract_min=electrodes_positive, normalize_data=electrodes_normalize)
    meg_data_dic = load_all_dics(freqs_bin, event_id, bipolar, electrodes, from_t, to_t, gk_sigma, njobs)
    run, min_error = 0, np.inf
    cutof = 1 # int(len(used_electrodes)/2)
    while(True):
        if only_first_subset:
            used_electrodes_sets = [np.random.choice(electrodes, k, replace=False).tolist()]
        else:
            used_electrodes_sets = utils.find_subsets(electrodes, k)
        errors = 0
        elec_pss = []
        for used_electrodes in used_electrodes_sets:
            if split_to_learn_and_pred:
                predicted_electrodes = np.random.choice(used_electrodes, cutof, replace=False).tolist()
                learned_electrodes = [elc for elc in used_electrodes if elc not in predicted_electrodes]
                elec_errors, elec_ps =  reconstruct_meg(event_id, learned_electrodes, from_t, to_t, time_split, gk_sigma=gk_sigma,
                                                        plot_results=False, bipolar=bipolar, dont_calc_new_csd=True, all_meg_data=meg_data_dic,
                                                        elec_data=elec_data, predicted_electrodes=predicted_electrodes, njobs=njobs)
            else:
                elec_errors, elec_ps = reconstruct_meg(event_id, used_electrodes, from_t, to_t, time_split, gk_sigma=gk_sigma,
                                                       plot_results=plot_results, bipolar=bipolar, dont_calc_new_csd=True, all_meg_data=meg_data_dic,
                                                       elec_data=elec_data, njobs=njobs)
            errors += sum([elec_errors[cond][electrode] for electrode in used_electrodes])
            elec_pss.append(elec_ps[cond])
        if errors < min_error:
            min_error = errors
            print('new min was found! {}'.format(errors))
            utils.save((used_electrodes_sets, errors, elec_pss), best_subset_fname(
                    cond, k, time_split, split_to_learn_and_pred, cutof, only_sig_electrodes))
        run += 1
        print(cond, run, k, '' if only_first_subset else 's', min_error)


def analyze_best_subset(event_id, k, bipolar, from_t, to_t, time_split, gk_sigma=3, only_sig_electrodes=True, split_to_learn_and_pred=True,
        only_first_subset=False, electrodes_positive=True, electrodes_normalize=True, plot_locations=False, do_plot=True, njobs=3):
    cond = utils.first_key(event_id)
    electrodes_sets, electrodes_errors, electrodes_pss = utils.load(best_subset_fname(
            cond, k, time_split, split_to_learn_and_pred, only_sig_electrodes, 1))
    electrodes = utils.flat_list_of_lists(electrodes_sets)
    elec_data = load_electrodes_data(event_id, bipolar, electrodes, from_t, to_t,
        subtract_min=electrodes_positive, normalize_data=electrodes_normalize)
    meg_data_dic = load_all_dics(freqs_bin, event_id, bipolar, electrodes, from_t, to_t, gk_sigma, njobs)
    for electrodes_set, electrodes_ps in zip(electrodes_sets, electrodes_pss):
        if plot_locations:
            plot_electrodes(bipolar, electrodes_set)
        # plot_leastsq_results(meg_data_dic, cond, elec_data, electrodes_set, electrodes_ps, time_split, same_ps=True, do_plot=do_plot)


def plot_electrodes(bipolar, electrodes=None):
    names, pos, _ = get_electrodes_positions(MRI_SUBJECT, bipolar)
    utils.plot_3d_scatter(pos, names.tolist(), electrodes)


def best_subset_fname(cond, k, time_split, split_to_learn_and_pred=False, only_sig_electrodes=False, cutof=0,
        only_first_subset=False, electrodes_normalize=True, electrodes_positive=True):
    if electrodes_normalize and electrodes_positive:
        fname = get_pkl_file('{}_{}{}_find_best_subset_{}{}{}{}.pkl'.format(
            cond, k, '' if only_first_subset else 's', len(time_split), '_split' if split_to_learn_and_pred else '',
            '_{}'.format(cutof) if cutof > 0 else '',
            '' if only_sig_electrodes else '_all_elecs'))
    else:
        fname = get_pkl_file('{}_{}{}_find_best_subset_{}{}{}{}_norm_{}_positive_{}.pkl'.format(
            cond, k, '' if only_first_subset else 's', len(time_split), '_split_' if split_to_learn_and_pred else '',
            '_{}'.format(cutof) if cutof > 0 else '',
            '' if only_sig_electrodes else '_all_elecs',
            int(electrodes_normalize), int(electrodes_positive)))
    return fname


def get_pkl_file(fname):
    return os.path.join(utils.get_files_fol(), fname)


def find_best_groups_parralel(params):
    event_id, bipolar, from_t, to_t, time_split, err_threshold, groups_panelty = params
    random.seed(utils.rand_letters(5))
    find_best_groups(event_id, bipolar, from_t, to_t, time_split, err_threshold=err_threshold,
        groups_panelty=groups_panelty, only_sig_electrodes=False, electrodes_positive=True,
        electrodes_normalize=True, gk_sigma=3, njobs=1)


def calc_noise_epoches_from_empty_room(events_id, data_raw_fname, empty_room_raw_fname, from_t, to_t,
        overwrite_epochs=False):
    from mne.event import make_fixed_length_events
    from mne.io import Raw

    epochs_noise_dic = {}
    epochs_noise_fnames = [get_cond_fname(EPO_NOISE, event) for event in events_id.keys()]
    if np.all([os.path.isfile(fname) for fname in epochs_noise_fnames]) and not overwrite_epochs:
        for event in events_id.keys():
            epochs_noise_dic[event] = mne.read_epochs(get_cond_fname(EPO_NOISE, event))
    else:
        raw = Raw(data_raw_fname)
        raw_noise = Raw(empty_room_raw_fname)
        # raw_noise.info['bads'] = ['MEG0321']  # 1 bad MEG channel
        picks = mne.pick_types(raw.info, meg=True)#, exclude='bads')
        events_noise = make_fixed_length_events(raw_noise, 1)
        epochs_noise = mne.Epochs(raw_noise, events_noise, 1, from_t,
            to_t, proj=True, picks=picks, baseline=None, preload=True)
        for event, event_id in events_id.items():
            # then make sure the number of epochs is the same
            epochs = mne.read_epochs(get_cond_fname(EPO, event))
            epochs_noise_dic[event] = epochs_noise[:len(epochs.events)]
            epochs_noise_dic[event].save(get_cond_fname(EPO_NOISE, event))
    return epochs_noise_dic


def calc_empty_room_noise_csd(events_id, epochs_from_t, epochs_to_t,
        freq_bins, win_lengths, overwrite_csds=False, overwrite_epochs=False):
    noise_csds = defaultdict(list)
    epochs_noise = calc_noise_epoches_from_empty_room(events_id, RAW, RAW_NOISE, epochs_from_t, epochs_to_t,
        overwrite_epochs=overwrite_epochs)
    for event in events_id.keys():
        if not os.path.isfile(get_cond_fname(NOISE_CSD_EMPTY_ROOM, event)) or overwrite_csds:
            for freq_bin, win_length in zip(freq_bins, win_lengths):
                noise_csd = compute_epochs_csd(epochs_noise[event], mode='multitaper', #mode='fourier',
                   fmin=freq_bin[0], fmax=freq_bin[1], fsum=True, tmin=-win_length, tmax=0.0, n_fft=None)
                noise_csds[event].append(noise_csd)
            print('saving csd to {}'.format(get_cond_fname(NOISE_CSD_EMPTY_ROOM, event)))
            utils.save(noise_csds[event], get_cond_fname(NOISE_CSD_EMPTY_ROOM, event))
        else:
            noise_csds[event] = utils.load(get_cond_fname(NOISE_CSD_EMPTY_ROOM, event))
        print(tuple(c.data.shape for c in noise_csds[event]))
    return noise_csds


def calc_td_dics(events_id, bipolar, epochs_from_t, epochs_to_t, csd_from_t, csd_to_t, tstep,
        freq_bins, win_lengths, subtract_evoked=False, overwrite_epochs=False, overwrite_csds=False):
    region = 'bipolar_electrodes' if bipolar else 'regular_electrodes'
    noise_csds = calc_empty_room_noise_csd(events_id, epochs_from_t, epochs_to_t, freq_bins=freq_bins,
        win_lengths=win_lengths, overwrite_csds=overwrite_csds, overwrite_epochs=overwrite_epochs)
    stcs = {}
    data_fol = os.path.join(SUBJECT_MEG_FOL, 'subcorticals', 'dics', 'all_{}'.format(region))
    utils.make_dir(data_fol)
    for event in events_id.keys():
        forward = mne.read_forward_solution(get_cond_fname(FWD_X, event, region=region), surf_ori=True)
        epochs = mne.read_epochs(get_cond_fname(EPO, event))
        stcs[event] = tf.tf_dics(event, epochs, forward, noise_csds[event], csd_from_t, csd_to_t, tstep, win_lengths,
            freq_bins=freq_bins, subtract_evoked=subtract_evoked, mode='multitaper', reg=0.001, subject=MRI_SUBJECT,
            data_fol=data_fol, overwrite_csds=False, overwrite_dics_sp=False, overwrite_stc=True)


def plot_td_dics(events_id, bipolar, tmin_plot, tmax_plot, freq_bins):
    from mne.viz import plot_source_spectrogram
    from mne.source_estimate import _make_stc

    # Plotting source spectrogram for source with maximum activity
    # Note that tmin and tmax are set to display a time range that is smaller than
    # the one for which beamforming estimates were calculated. This ensures that
    # all time bins shown are a result of smoothing across an identical number of
    # time windows.
    region = 'bipolar_electrodes' if bipolar else 'regular_electrodes'
    data_fol = os.path.join(SUBJECT_MEG_FOL, 'subcorticals', 'dics', 'all_{}'.format(region))
    stcs = []
    for event in events_id.keys():
        stcs.append(tf.load_stcs(event, freq_bins, data_fol, MRI_SUBJECT))
        # plot_source_spectrogram(stcs[-1], freq_bins, tmin=tmin_plot, tmax=tmax_plot,
        #                     source_index=None, colorbar=True)
    stcs_diff = []
    for stc1, stc2 in zip(stcs[0], stcs[1]):
        stc_diff = _make_stc(stc1.data - stc2.data, vertices=stc1.vertices,
                            tmin=stc1.tmin, tstep=stc1.tstep, subject=stc1.subject)
        stcs_diff.append(stc_diff)
    plot_source_spectrogram(stcs_diff, freq_bins, tmin=tmin_plot, tmax=tmax_plot,
                        source_index=None, colorbar=True)


def find_best_freqs_subset(event_id, bipolar, freqs_bins, from_t, to_t, time_split, combs,
        optimization_method='RidgeCV', optimization_params={}, k=3, gk_sigma=3, njobs=6):
    freqs_bins = sorted(freqs_bins)
    all_electrodes = get_all_electrodes_names(bipolar)
    elec_data = load_electrodes_data(event_id, bipolar, all_electrodes, from_t, to_t,
            subtract_min=False, normalize_data=False)
    meg_data_dic = load_all_dics(freqs_bins, event_id, bipolar, all_electrodes, from_t, to_t, gk_sigma,
        dont_calc_new_csd=True, njobs=njobs)

    uuid = utils.rand_letters(5)
    results_fol = get_results_fol(optimization_method)
    partial_results_fol = os.path.join(results_fol, 'best_freqs_subset_{}'.format(uuid))
    utils.make_dir(results_fol)
    utils.make_dir(partial_results_fol)

    cond = utils.first_key(event_id)
    all_freqs_bins_subsets = list(utils.superset(freqs_bins))
    random.shuffle(all_freqs_bins_subsets)
    N = len(all_freqs_bins_subsets)
    print('There are {} freqs subsets'.format(N))
    all_freqs_bins_subsets_chunks = utils.chunks(all_freqs_bins_subsets, int(len(all_freqs_bins_subsets) / njobs))
    params = [Bunch(event_id=event_id, bipolar=bipolar, freqs_bins_chunks=freqs_bins_subsets_chunk, cond=cond,
            from_t=from_t, to_t=to_t, freqs_bins=freqs_bins, partial_results_fol=partial_results_fol,
            time_split=time_split, only_sig_electrodes=False, only_from_same_lead=True, electrodes_positive=False,
            electrodes_normalize=False, gk_sigma=gk_sigma, k=k, do_plot_results=False, do_save_partial_results=False,
            optimization_params=optimization_params, check_only_pred_score=True, njobs=1, N=int(N / njobs),
            elec_data=elec_data, meg_data_dic=meg_data_dic, all_electrodes=all_electrodes,
            optimization_method=optimization_method, error_calc_method='rol_corr', error_threshold=30, combs=combs) for
            freqs_bins_subsets_chunk in all_freqs_bins_subsets_chunks]
    results = utils.run_parallel(_find_best_freqs_subset_parallel, params, njobs)
    all_results = []
    for chunk_results in results:
        all_results.extend(chunk_results)
    params_suffix = utils.params_suffix(optimization_params)
    output_file = os.path.join(results_fol, 'best_freqs_subset_{}_{}_{}{}.pkl'.format(cond, uuid, k, params_suffix))
    print('saving results to {}'.format(output_file))
    utils.save((chunk_results, freqs_bins), output_file)


def _find_best_freqs_subset_parallel(p):
    chunk_results = []
    uuid = utils.rand_letters(5)
    output_file = os.path.join(p.partial_results_fol,
        'best_freqs_subset_{}_{}.pkl'.format(p.cond, uuid))
    now = time.time()
    for run, freqs_bin in enumerate(p.freqs_bins_chunks):
        freqs_indices = [p.freqs_bins.index(fb) for fb in freqs_bin]
        meg_data_dic = get_sub_meg_data_dic(p.meg_data_dic, freqs_indices)
        if run % 10 == 0 and len(chunk_results) > 0:
            utils.time_to_go(now, run, p.N)
            utils.save((chunk_results, p.freqs_bins), output_file)
        results = find_best_predictive_subset(event_id=p.event_id, bipolar=p.bipolar, freqs_bins=freqs_bin, from_t=p.from_t, to_t=p.to_t,
            time_split=p.time_split, only_sig_electrodes=False, only_from_same_lead=True, electrodes_positive=False,
            electrodes_normalize=False, gk_sigma=p.gk_sigma, k=p.k, do_plot_results=False, do_save_partial_results=False,
            optimization_params=p.optimization_params, check_only_pred_score=True, njobs=1, vebrose=False, uuid_len=5,
            optimization_method=p.optimization_method, error_calc_method='rol_corr', error_threshold=100, combs=p.combs,
            save_results=False, elec_data=p.elec_data, meg_data_dic=meg_data_dic, all_electrodes=p.all_electrodes)
        results = [result + (freqs_bin, ) for result in results]
        chunk_results.extend(results)
    utils.save((chunk_results, p.freqs_bins), output_file)
    return chunk_results


def get_sub_meg_data_dic(orig_meg_data_dic, freqs_indices):
    meg_data_dic = copy.deepcopy(orig_meg_data_dic)
    for elec in meg_data_dic.keys():
        meg_data_dic[elec] = meg_data_dic[elec][freqs_indices, :]
    return meg_data_dic


def pickup_freqs_subsets(event_id, uuid, optimization_method, optimization_params, k=3):
    results_fol = get_results_fol(optimization_method)
    partial_results_fol = os.path.join(results_fol, 'best_freqs_subset_{}'.format(uuid))
    cond = utils.first_key(event_id)
    all_results = []
    freqs_bins = set()
    for results_fname in glob.glob(os.path.join(partial_results_fol, 'best_freqs_subset*.pkl')):
        try:
            results, freqs_bins = utils.load(results_fname)
        except:
            results = utils.load(results_fname)
            for result in results:
                freqs_bins |= set([f for f in result[-1]])
        all_results.extend(results)
    params_suffix = utils.params_suffix(optimization_params)
    output_file = os.path.join(results_fol, 'best_freqs_subset_{}_{}_{}{}.pkl'.format(cond, uuid, k, params_suffix))
    freqs_bins = sorted([(fmin, fmax) for fmin, fmax in freqs_bins])
    print('saving {} results to {}'.format(len(all_results), output_file))
    print('freqs_bins = {}'.format(freqs_bins))
    utils.save((all_results, freqs_bins), output_file)


def get_results_fol(optimization_method, electrodes_normalize=False, electrodes_positive=False):
    return os.path.join(utils.get_files_fol(), '{}_norm_{}_positive_{}'.format(
        optimization_method, str(electrodes_normalize)[0], str(electrodes_positive)[0]))


def load_best_freqs_subset(event_id, uuid, optimization_method, optimization_params, from_t, to_t, time_split,
        k=3, bipolar=False, gk_sigma=3, do_plot=False, verbose=False, recalculate=False, write_errors_csv=False,
        new_optimization_method='', new_error_calc_method='', resort=False, top_k=np.inf, top_err=np.inf,
        group_by_predicted=True, best_k_in_group=1, save_plots_in_pred_electrode_fol=False, do_save_plots=False, njobs=4):
    results_fol = get_results_fol(optimization_method)
    cond = utils.first_key(event_id)
    params_suffix = utils.params_suffix(optimization_params)
    results_fname = os.path.join(results_fol, 'best_freqs_subset_{}_{}_{}{}.pkl'.format(cond, uuid, k, params_suffix))
    print('loading {}'.format(results_fname))
    results, all_freqs_bins = utils.load(results_fname)
    print('{} results for {}'.format(len(results), all_freqs_bins))
    all_best_freqs_bins = set()

    if do_plot or recalculate or resort:
        electrodes = get_all_electrodes_names(bipolar)
        elec_data = load_electrodes_data(event_id, bipolar, electrodes, from_t, to_t,
            subtract_min=False, normalize_data=False)

    if top_err < np.inf:
        results = sorted(results, key=lambda x:x[2][x[0]])
        errors = np.array([x[2][x[0]] for x in results])
        top_ind = np.where(errors > top_err)[0][0]
        print('top ind for max_err {} is {}'.format(top_err, top_ind))
        results = results[:top_ind]

    if group_by_predicted:
        only_best_results = []
        results = sorted(results, key=lambda x:x[0])
        for predicted, pred_results in groupby(results, lambda x:x[0]):
            pred_results = sorted(pred_results, key=lambda x:x[2][x[0]])
            # pred_results = sorted(pred_results, key=lambda x:len(x[-1]))
            only_best_results.extend(pred_results[:best_k_in_group])
        results = only_best_results
        print('new results len after group by: {}'.format(len(results)))

    if resort:
        results = freqs_subsets_sort_results(event_id, results, electrodes, elec_data, all_freqs_bins, gk_sigma,
            from_t, to_t, time_split, new_optimization_method, new_error_calc_method, optimization_params,
            bipolar, njobs)
    else:
        results = sorted(results, key=lambda x:x[2][x[0]])

    if write_errors_csv:
        csv_file = open(os.path.join(results_fol, 'freqs_subsets_{}_{}.csv'.format(k, uuid)), 'w')
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerows([['index', 'predicted', 'train', 'predicted RMS', '', '', 'freqs bins']])

    for res_ind, best_result in enumerate(results):
        predicted, train, errors, best_ps, params, best_freqs_bin = best_result
        for fb in best_freqs_bin:
            all_best_freqs_bins.add(fb)
        if do_plot or recalculate:
            meg_data_dic = load_all_dics(best_freqs_bin, event_id, bipolar, electrodes, from_t, to_t, gk_sigma, njobs)
        if verbose:
            print('best result:')
            print(best_freqs_bin)
            errors_str = ','.join(['{:.2f}'.format(errors[elec] if recalculate else errors[elec]) for elec in train + [predicted]])
            print('{}->{}: {}'.format(train, predicted, errors_str))
        if recalculate:
            errors, ps, params = reconstruct_meg(event_id, best_freqs_bin, train, from_t, to_t, time_split,
                optimization_method=new_optimization_method, error_calc_method=new_error_calc_method,
                optimization_params=optimization_params, predicted_electrodes=[predicted],
                plot_results=do_plot, bipolar=bipolar, dont_calc_new_csd=True, res_ind=res_ind,
                all_meg_data=meg_data_dic, elec_data=elec_data, njobs=1, uuid=uuid,
                save_plots_in_pred_electrode_fol=save_plots_in_pred_electrode_fol, do_save_plots=do_save_plots)
            print(ps[cond])
        elif do_plot and not do_save_plots:
            plot_leastsq_results(meg_data_dic, cond, elec_data, train, best_ps, time_split, optimization_method,
                predicted_electrodes=[predicted],save_in_pred_electrode_fol=save_plots_in_pred_electrode_fol,
                do_save=do_save_plots)
        if write_errors_csv:
            errors_strs = ['{:.2f}'.format(errors[cond][elec]) for elec in [predicted] + train]
            csv_writer.writerows([[res_ind, predicted, train] + errors_strs + list(best_freqs_bin)])
        if res_ind == top_k:
            break

    print('freqs_bins in all {} results:'.format(top_k))
    print(sorted(all_best_freqs_bins))
    print('not used freqs: {}'.format(set(all_freqs_bins) - all_best_freqs_bins))
    if write_errors_csv:
        csv_file.close()


def print_result(errors, train, predicted, cond=None):
    err = errors[list(errors)[0]] if predicted not in errors else errors
    errors_str = ','.join(['{:.2f}'.format(err[elec]) for elec in train + [predicted]])
    print('{}{}->{}: {}'.format('' if cond is None else '{} '.format(cond), train, predicted, errors_str))


def cut_best_freqs_subset(events_id, uuid, optimization_method, optimization_params, k, max_err):
    results_fol = get_results_fol(optimization_method)
    params_suffix = utils.params_suffix(optimization_params)
    for cond in events_id.keys():
        results_fname = os.path.join(results_fol, 'best_freqs_subset_{}_{}_{}{}.pkl'.format(cond, uuid, k, params_suffix))
        if os.path.isfile(results_fname):
            print('loading {}'.format(results_fname))
            results, all_freqs_bins = utils.load(results_fname)
            print('{} results for {}'.format(len(results), all_freqs_bins))

            results = sorted(results, key=lambda x:x[2][x[0]])
            errors = np.array([x[2][x[0]] for x in results])
            top_ind = np.where(errors > max_err)[0][0]
            print('top ind for max_err {} is {}'.format(max_err, top_ind))
            results = results[:top_ind]
            utils.save((results, all_freqs_bins), results_fname)
        else:
            print("{} does't exist!".format(results_fname))


def best_freqs_subset_cv(events_id, uuid, all_freqs_bins, optimization_method, error_calc_method, optimization_params,
    from_t, to_t, time_split, k=3, bipolar=False, gk_sigma=3, max_err=30, verbose=False, njobs=4):

    results_fol = get_results_fol(optimization_method)
    params_suffix = utils.params_suffix(optimization_params)
    electrodes = get_all_electrodes_names(bipolar)
    elec_data = load_electrodes_data(events_id, bipolar, electrodes, from_t, to_t)

    all_meg_data_dic = {}
    for cond in events_id.keys():
        event_id = {cond: events_id[cond]}
        all_meg_data_dic[cond] = load_all_dics(all_freqs_bins, event_id, bipolar, electrodes, from_t, to_t, gk_sigma, njobs)

    for cond in events_id.keys():
        event_id = {cond: events_id[cond]}
        other_cond = [c for c in events_id.keys() if c !=cond][0]
        other_event_id = {other_cond: events_id[other_cond]}
        results_fname = os.path.join(results_fol, 'best_freqs_subset_{}_{}_{}{}.pkl'.format(cond, uuid, k, params_suffix))
        if not os.path.isfile(results_fname):
            print("{} doesn't exist!".format(results_fname))
            continue
        print('loading {}'.format(results_fname))
        results, all_freqs_bins = utils.load(results_fname)
        print('{} results for {}'.format(len(results), all_freqs_bins))

        results = sorted(results, key=lambda x:x[2][x[0]])
        errors = np.array([x[2][x[0]] for x in results])
        top_inds = np.where(errors > max_err)[0]
        if len(top_inds) > 0:
            results = results[:top_inds[0]]
        print('top ind for max_err {} is {}'.format(max_err, len(results)))

        all_pred_results = []
        results = sorted(results, key=lambda x:x[0])
        for predicted, pred_results in groupby(results, lambda x:x[0]):
            pred_results = sorted(pred_results, key=lambda x:x[2][x[0]])
            all_pred_results.append(pred_results)
        all_pred_results_chunks = utils.chunks(all_pred_results, int(len(all_pred_results) / njobs))
        params = [(pred_results_chunk, all_meg_data_dic, elec_data, all_freqs_bins, cond, other_cond, other_event_id,
                from_t, to_t, time_split, optimization_method, error_calc_method, optimization_params, bipolar,
                max_err, uuid, verbose) for pred_results_chunk in all_pred_results_chunks]
        all_results = utils.run_parallel(_best_freqs_subset_cv_parallel, params, njobs)

        all_good_results = defaultdict(list)
        for good_results in all_results:
            all_good_results[cond].extend(good_results[cond])

        results_fol = get_results_fol(optimization_method)
        params_suffix = utils.params_suffix(optimization_params)
        good_results_fname = os.path.join(results_fol, 'best_freqs_subset_cv_{}_{}_{}{}.pkl'.format(cond, uuid, k, params_suffix))
        utils.save(good_results, good_results_fname)
        print('{} good result for max_err {}'.format(len(all_good_results[cond]), max_err))


def _best_freqs_subset_cv_parallel(params):
    pred_results_chunk, all_meg_data_dic, elec_data, all_freqs_bins, cond, other_cond, other_event_id, from_t, to_t, time_split,\
       optimization_method, error_calc_method, optimization_params, bipolar, max_err, uuid, verbose = params
    good_results = defaultdict(list)
    for pred_results in pred_results_chunk:
        for result in pred_results:
            predicted, train, errors, ps, params, freqs_bins = result
            freqs_indices = [all_freqs_bins.index(fb) for fb in freqs_bins]
            other_meg_data_dic = get_sub_meg_data_dic(all_meg_data_dic[other_cond], freqs_indices)
            other_errors, other_ps, other_params = reconstruct_meg(other_event_id, freqs_bins, train, from_t, to_t,
                time_split, optimization_method=optimization_method, error_calc_method=error_calc_method,
                optimization_params=optimization_params, predicted_electrodes=[predicted],
                plot_results=False, bipolar=bipolar, dont_calc_new_csd=True,
                all_meg_data=other_meg_data_dic, elec_data=elec_data, njobs=1, uuid=uuid)
            if other_errors[other_cond][predicted] < max_err and errors[predicted] < max_err:
                if verbose:
                    print('results for {}'.format(sorted(freqs_bins)))
                    print_result(errors, train, predicted, cond)
                print_result(other_errors, train, predicted, other_cond)
                good_results[cond].append((result, other_errors[other_cond], other_ps[other_cond], other_params))
    return good_results


def load_best_freqs_subset_cv(events_id, uuid, all_freqs_bins, optimization_method, optimization_params,
        from_t, to_t, time_split, gk_sigma, max_err, k=3, bipolar=False, njobs=4):
    results_fol = get_results_fol(optimization_method)
    params_suffix = utils.params_suffix(optimization_params)
    electrodes = get_all_electrodes_names(bipolar)
    elec_data = load_electrodes_data(events_id, bipolar, electrodes, from_t, to_t)
    all_meg_data_dic = {}
    for cond in events_id.keys():
        event_id = {cond: events_id[cond]}
        all_meg_data_dic[cond] = load_all_dics(all_freqs_bins, event_id, bipolar, electrodes, from_t, to_t, gk_sigma, njobs)
    for cond in events_id.keys():
        other_cond = [c for c in events_id.keys() if c !=cond][0]
        results_fname = os.path.join(results_fol, 'best_freqs_subset_cv_{}_{}_{}{}.pkl'.format(cond, uuid, k, params_suffix))
        if not os.path.isfile(results_fname):
            continue
        results = utils.load(results_fname)[cond]
        print('{} good result for max_err {}'.format(len(results), max_err))
        results = sorted(results, key=lambda x:x[0][0])
        for predicted, pred_results in groupby(results, lambda x:x[0][0]):
            pred_results = list(pred_results)
            print('{}: {} good results'.format(predicted, len(pred_results)))
            pred_results = sort_both_conds(pred_results, other_cond)
            (predicted, train, best_errors, best_ps, best_params, best_freqs_bins),\
                best_other_errors, other_best_ps, _ = pred_results[0]
            print(best_freqs_bins)
            print_result(best_errors, train, predicted, cond)
            print_result(best_other_errors, train, predicted, other_cond)
            freqs_indices = [all_freqs_bins.index(fb) for fb in best_freqs_bins]
            cond_meg_data_dic = get_sub_meg_data_dic(all_meg_data_dic[cond], freqs_indices)
            other_meg_data_dic = get_sub_meg_data_dic(all_meg_data_dic[other_cond], freqs_indices)
            plot_both_cond_prediction(predicted, cond_meg_data_dic, other_meg_data_dic, cond, other_cond, elec_data,
                best_ps, other_best_ps[other_cond], time_split, optimization_method)


def sort_both_conds(pred_results, other_cond):
    both_errors = []
    for result, other_errors, other_ps, other_params in pred_results:
        predicted, train, errors, ps, params, freqs_bins = result
        cond_error = errors[predicted]
        other_cond_error = other_errors[predicted]
        both_errors.append(cond_error * other_cond_error)
    return [res for err, res in sorted(zip(both_errors, pred_results))]


def plot_both_cond_prediction(predicted_electrode, cond_meg_data_dic, other_meg_data_dic, cond, other_cond, elec_data,
        cond_ps, other_cond_ps, time_split, optimization_method, do_plot=True, do_save=False,
        uuid='', title=None, full_title=True, error_functions=(), res_ind=0, save_in_pred_electrode_fol=False,
        dtw_window=10, tight_layout=True):
    if uuid=='':
        uuid = utils.rand_letters(5)
    time_diff = np.diff(time_split)[0] if len(time_split) > 1 else 500
    pics_num_x, pics_num_y = utils.how_many_subplots(2)
    f, axs = plt.subplots(pics_num_x, pics_num_y, sharex='col', sharey='row', figsize=(12, 8))
    if do_save:
        fig_fol = os.path.join(get_figs_fol(), optimization_method, uuid)
        utils.make_dir(fig_fol)
    if len(error_functions) == 0:
        error_functions = ERROR_RECONSTRUCT_METHODS
    electrodes_data = [elec_data[predicted_electrode][cond], elec_data[predicted_electrode][other_cond]]
    meg_data_dics = [cond_meg_data_dic, other_meg_data_dic]
    conds = [cond, other_cond]
    pss = [cond_ps, other_cond_ps]
    for electrode_data, meg_data_dic, current_cond, ps, ax in zip(electrodes_data, meg_data_dics, conds, pss, axs):
        meg = combine_meg_chunks(meg_data_dic[predicted_electrode], ps, time_split, time_diff)
        err_func = partial(electrode_reconstruction_error, electrode=predicted_electrode, meg=meg,
            electrode_data=electrode_data, electrode_ps=ps,
            meg_data_dic=meg_data_dic, time_split=time_split, time_diff=time_diff, dtw_window=dtw_window)
        errors = {em:err_func(error_calc_method=em) for em in error_functions}
        errors_str = ''.join(['{}:{:.2f} '.format(em, errors[em]) for em in error_functions])
        ax.plot(elec_data[predicted_electrode][current_cond], label=predicted_electrode)
        ax.plot(meg, label='plsq')
        if title is None:
            ax.set_title('{}: {} '.format(current_cond, predicted_electrode) + errors_str)
    if not title is None:
        axs[0].set_title(title)
    if tight_layout:
        plt.tight_layout()
    if do_save:
        if save_in_pred_electrode_fol:
            elec_fol = os.path.join(fig_fol, predicted_electrode)
            utils.make_dir(elec_fol)
            fig_fname = os.path.join(elec_fol, '{}_{}.png'.format(uuid, res_ind))
        else:
            fig_fname = os.path.join(fig_fol, '{}_{}.png'.format(uuid, res_ind))
        f.savefig(fig_fname)
        print('saved to {}'.format(fig_fname))
    if do_plot and not do_save:
        plt.show()
    else:
        plt.close()


def freqs_subsets_sort_results(event_id, results, electrodes, elec_data, all_freqs_bins, gk_sigma, from_t, to_t,
        time_split, new_optimization_method, new_error_calc_method, optimization_params, bipolar, njobs=4):

    cond = utils.first_key(event_id)
    orig_meg_data_dic = load_all_dics(all_freqs_bins, event_id, bipolar, electrodes, from_t, to_t, gk_sigma, njobs)
    results_errors = []
    for result in tqdm(results):
        predicted, train, errors, ps, params, freqs_bin = result
        freqs_indices = [all_freqs_bins.index(fb) for fb in freqs_bin]
        meg_data_dic = get_sub_meg_data_dic(orig_meg_data_dic, freqs_indices)
        errors, ps, params = reconstruct_meg(event_id, freqs_bin, train, from_t, to_t, time_split,
            optimization_method=new_optimization_method, error_calc_method=new_error_calc_method,
            optimization_params=optimization_params, predicted_electrodes=[predicted],
            plot_results=False, bipolar=bipolar, dont_calc_new_csd=True,
            all_meg_data=meg_data_dic, elec_data=elec_data, njobs=1)
        results_errors.append(errors[cond][predicted])
    return sorted(res for (err, res) in zip(results, results_errors))


def fix_names(bipolar):
    #laf = lof
    #lmt = lpt
    dics = glob.glob(os.path.join(SUBJECT_MEG_FOL, 'subcorticals', 'dics',
        'bipolar' if bipolar else 'regular', 'dics_*.npy'))
    for dic_fname in dics:
        if 'LAF' in dic_fname:
            new_dic_fname = dic_fname.replace('LAF', 'LOF')
            os.rename(dic_fname, new_dic_fname)
        elif 'LMT' in dic_fname:
            new_dic_fname = dic_fname.replace('MTF', 'LPT')
            os.rename(dic_fname, new_dic_fname)


def main():
    from_t, to_t = 500, 1000# -500, 2000
    bipolar = False
    gk_sigma = 3
    njobs = int(utils.how_many_cores() / 2)
    logging.basicConfig(filename='errors.log',level=logging.ERROR)
    random.seed(datetime.now())
    events_id = dict(neutral=2, interference=1)

    # electrode = 'LAT3-LAT2' if bipolar else 'LAT3'
    # electrodes = ['LAT3-LAT2', 'LPT2-LPT1', 'LAT2-LAT1', 'LAT4-LAT3']
    # predicted_electrodes = []#['LAT2-LAT1', 'LAT4-LAT3']

    use_fwd_for_region = False
    sub_corticals_codes_file = os.path.join(BLENDER_ROOT_DIR, 'sub_cortical_codes.txt')
    time_split = np.arange(0, 500, 100) # np.arange(0, 500, 500)
    CSD_FREQS_NO_INF = [(0, 4), (1, 3), (2, 6), (3, 5), (4,8), (6,10), (8,12), (10, 14), (12, 16), (12, 25),
                 (25, 40), (40, 100), (60, 100), (80, 120), (100, 140)]#, (80, np.inf), (0, np.inf)]
    CSD_FREQS_INF = [(80, np.inf), (0, np.inf)]
    CSD_FREQS = CSD_FREQS_NO_INF + CSD_FREQS_INF
    CSD_FREQS_DALAL = [(4, 8), (8, 12), (12, 30), (30, 55), (65, 300)]  # Hz

    # find_fit(events_id, bipolar, from_t, to_t, time_split, err_threshold=10, plot_results=False)
    # analyze_leastsq_results(events_id, time_split)

    cond = utils.first_key(events_id)# list(events_id)[0]
    # neutral = 'neutral'
    event_id = {cond: events_id[cond]}
    # nice_combs = [['RMF5-RMF4', 'RMF4-RMF3', 'RMF6-RMF5'],['RMT4-RMT3','RMT3-RMT2','RMT5-RMT4'],['RAT6-RAT5','RAT5-RAT4', 'RAT7-RAT6'], ['RMF6-RMF5','RMF5-RMF4', 'RMF7-RMF6']]
    # groups_panelty = 1
    only_sig_electrodes = False
    # params = [({cond: events_id[cond]}, bipolar, from_t, to_t, time_split, err_threshold, groups_panelty) for
    #           (cond, err_threshold) in product(events_id.keys(), [3,5,7,10])]
    # utils.run_parallel(find_best_groups_parralel, params, 8)
    # find_best_groups(event_id, bipolar, from_t, to_t, time_split, err_threshold=10, groups_panelty=1, only_sig_electrodes=False, njobs=njobs)
    # analyze_best_groups(event_id, bipolar, from_t, to_t, time_split, err_threshold=5, groups_panelty=1)

    optimization_params={'window':30, 'alpha':5}
    freqs_bin = CSD_FREQS_DALAL
    find_bps = partial(find_best_predictive_subset, event_id=event_id, bipolar=bipolar, freqs_bins=freqs_bin, from_t=from_t, to_t=to_t,
        time_split=time_split, only_sig_electrodes=False, only_from_same_lead=True, electrodes_positive=False,
        electrodes_normalize=False, electrodes_subtract_mean=False, gk_sigma=3, k=3, do_plot_results=False, do_save_partial_results=False,
        optimization_params=optimization_params, check_only_pred_score=True, njobs=4)
    bps_collect_results = partial(best_predictive_subset_collect_results, event_id=event_id, bipolar=bipolar, freqs_bin=freqs_bin,
        from_t=from_t, to_t=to_t, time_split=time_split, sort_only_accoring_to_pred=True, calc_all_errors=False,
        dtw_window=10, electrodes_positive=False, electrodes_normalize=False, njobs=1,
        electrodes_subtract_mean=True, optimization_params=optimization_params,
        do_save=False, do_plot=True, save_in_pred_electrode_fol=True, write_errors_csv=False, do_plot_electrodes=False,
        error_functions=ERROR_RECONSTRUCT_METHODS)

    mg78_inter_leads = [['RAT', 'RMT', 'RPT'], ['RAF', 'RMF', 'RPF'], ['ROF', 'RAF', 'RMF']]
    mg78_inter_leads_combs = get_inter_lead_combs(mg78_inter_leads, bipolar, False)

    # find_bps(optimization_method='rol_corr', error_calc_method='rol_corr', error_threshold=100, combs=mg78_inter_leads_combs)#, elec_data=elec_data)
    # find_bps(optimization_method='RidgeCV', error_calc_method='rol_corr', error_threshold=30, combs=mg78_inter_leads_combs)
    # bps_collect_results(uuid='3f70f', k=3, optimization_method='rol_corr', error_calc_method='rol_corr', error_threshold=100)#, elec_data=elec_data)
    # bps_collect_results(uuid='7df21', k=3, optimization_method='RidgeCV', error_calc_method='rol_corr', error_threshold=30)
    # bps_collect_results(uuid='6d6a1', optimization_method='rol_corr', error_calc_method='rol_corr', error_threshold=10)

    # find_best_freqs_subset(event_id, bipolar, CSD_FREQS_NO_INF, from_t, to_t, time_split,
    #     mg78_inter_leads_combs, 'RidgeCV', optimization_params, k=3, gk_sigma=3, njobs=7)
    # pickup_freqs_subsets(event_id, '20793', 'RidgeCV', optimization_params)
    # cut_best_freqs_subset(events_id, '20793', 'RidgeCV', optimization_params, k=3, max_err=50)
    # load_best_freqs_subset(event_id, '20793', 'RidgeCV', optimization_params, from_t, to_t, time_split, k=3,
    # load_best_freqs_subset(event_id, '9222b', 'RidgeCV', optimization_params, from_t, to_t, time_split, k=3,
    #     bipolar=False, gk_sigma=3, do_plot=True, recalculate=False, write_errors_csv=False, top_err=25,
    #     group_by_predicted=True, new_optimization_method='rol_corr', new_error_calc_method='rol_corr',
    #     verbose=True, save_plots_in_pred_electrode_fol=False, do_save_plots=False, best_k_in_group=1, njobs=4)
    # best_freqs_subset_cv(events_id, '20793', CSD_FREQS_NO_INF, 'RidgeCV', 'rol_corr', optimization_params,
    #     from_t, to_t, time_split, k=3, bipolar=False, gk_sigma=3, max_err=50, njobs=4)
    load_best_freqs_subset_cv(events_id, '20793', CSD_FREQS_NO_INF, 'RidgeCV', optimization_params,
        from_t, to_t, time_split, gk_sigma, max_err=50)

    # all_electrodes = get_all_electrodes_names(bipolar)
    # elec_data = load_electrodes_data(event_id, bipolar, all_electrodes, from_t, to_t,
    #     subtract_min=False, normalize_data=False, subtract_mean=False)
    # new_combs = []
    # for comb in get_inter_lead_combs(mg78_inter_leads, bipolar, False):
    #     # avg = np.mean([elec_data[comb[l]][cond] for l in range(3)], axis=0)
    #     # for l in range(3):
    #     #     elec_data[comb[l]][cond] -= avg
    #     elec_data[comb[0]][cond] -= elec_data[comb[2]][cond]
    #     elec_data[comb[1]][cond] -= elec_data[comb[2]][cond]
    #     elec_data[comb[2]][cond] = elec_data[comb[0]][cond]
    #     new_combs.append([comb[0], comb[1]])
        # plt.figure()
        # for l in range(3):
        #     plt.plot(elec_data[comb[l]][cond], label=comb[l])
        # plt.legend()
        # plt.show()

    # for alpha in range(21):
    #     optimization_params={'window':30, 'alpha':alpha}
    #     print(optimization_params)
    #     find_bps(optimization_method='rol_corr', error_calc_method='rol_corr', error_threshold=100, optimization_params=optimization_params)

    # plot_reconstruction_for_different_freqs(event_id, 'RMF5-RMF4', ['RMT3-RMT2', 'RMT6-RMT5'], from_t, to_t, time_split)
    # 3df71
    # analyze_best_predictive_subset(4, event_id, bipolar, from_t, to_t, time_split)
    # plot_predictive_subset(['ROF5-ROF4', 'LAT3-LAT2', 'LAT2-LAT1', 'LAT4-LAT3'], 4, event_id, bipolar, from_t, to_t, time_split, njobs=4)

    # calc_lead_predictiveness(event_id, bipolar, from_t, to_t, time_split, gk_sigma=3,
    #     electrodes_positive=True, electrodes_normalize=True, k=7, njobs=4)
    # plot_lead_predictiveness(event_id, bipolar, from_t, to_t, time_split, gk_sigma=3,
    #     electrodes_positive=True, electrodes_normalize=True, k=5, error_threshold=10, njobs=4)

    # calc_p_for_each_electrode(event_id, bipolar, from_t, to_t, time_split, gk_sigma=3, njobs=4)
    # analyze_p_for_each_electrode(event_id, bipolar, from_t, to_t, time_split, gk_sigma=3, njobs=4)
    # analyze_best_n_componenets(event_id, bipolar, from_t, to_t, time_split, njobs=4)

    # find_best_subset(event_id, 1, bipolar, from_t, to_t, time_split, gk_sigma=3, split_to_learn_and_pred=False, only_sig_electrodes=False, plot_results=True, njobs=njobs)
    # analyze_best_subset(event_id, 4, bipolar, from_t, to_t, time_split, split_to_learn_and_pred=False, only_sig_electrodes=False, plot_locations=True, do_plot=False)

    # sig_elecs = find_significant_electrodes(bipolar, from_t, to_t, do_plot=False, do_save=False, plot_only_sig=False)
    # plot_electrodes(bipolar, sig_elecs[cond])

    # learn_and_pred(events_id, bipolar, from_t, to_t, time_split)
    # sig = find_significant_electrodes(bipolar, from_t, to_t)

    # check_freqs(events_id, electrodes, from_t, to_t, time_split, predicted_electrodes=predicted_electrodes, gk_sigma=3, bipolar=bipolar, njobs=1)

    # names, pos, pos_org = get_electrodes_positions(MRI_SUBJECT, bipolar)
    bipolar = False
    # electrodes = get_all_electrodes_names(bipolar)
    # calc_all_fwds(events_id, electrodes, bipolar, from_t, to_t, time_split, overwrite_fwd=True, njobs=7)
    # calc_electrodes_fwd(MRI_SUBJECT, electrodes, events_id, bipolar=False, overwrite_fwd=True, read_if_exist=False, n_jobs=4)
    # calc_dics_freqs_csd(events_id, electrodes, bipolar, from_t, to_t, time_split, freqs_bands=CSD_FREQS_DALAL,
    #     overwrite_csds=False, overwrite_dics=False, gk_sigma=3, njobs=7)
    # calc_dics_freqs_csd(events_id, electrodes, bipolar, from_t, to_t, time_split, freqs_bands=CSD_FREQS_INF,
    #     overwrite_csds=False, overwrite_dics=False, gk_sigma=3, njobs=1)
    # region = 'bipolar_electrodes' if bipolar else 'regular_electrodes'
    # calc_dics_freqs_csd(events_id, [region], bipolar, from_t, to_t, time_split, freqs_bands=CSD_FREQS_DALAL,
    #     overwrite_csds=True, overwrite_dics=True, gk_sigma=3, njobs=1)

    win_lengths = [0.3, 0.3, 0.2, 0.15, 0.1]
    freqs = CSD_FREQS_DALAL
    epochs_from_t, epochs_to_t, tstep = -0.5, 2.0, 0.05
    csd_from_t, csd_to_t = -0.5, 2.0
    tmin_plot, tmax_plot = -0.25, 1.75
    # calc_td_dics(events_id, bipolar, epochs_from_t, epochs_to_t, csd_from_t, csd_to_t, tstep,
    #     win_lengths=win_lengths, freq_bins=freqs, overwrite_csds=True, overwrite_epochs=False)
    # plot_td_dics(events_id, bipolar, tmin_plot, tmax_plot, freq_bins=freqs)

    # calc_all_electrodes_fwd(MRI_SUBJECT, events_id, overwrite_fwd=False, n_jobs=6)
    # calc_electrode_fwd(MRI_SUBJECT, electrode, events_id, bipolar, overwrite_fwd=False)

    # check_electrodes()
    # check_bipolar_meg(events_id, electrode)
    # comp_lcmv_dics_electrode(events_id, electrode, bipolar)


    # plot_activation_one_fig(cond, meg_data_norm, elec_data_norm, electrode, 500)


    # cond = 'interference'
    # meg_data = load_all_subcorticals(subject_meg_fol, sub_corticals_codes_file, cond, from_t, to_t, normalize=True)
    # plot_activation_one_fig(cond, meg_data, elec_data, 'elec', from_t, to_t)

    # meg_data = call_lcmv(forward, data_cov, noise_cov, evoked, epochs)
    # plot_activation(events_id, meg_data, elec_data, 'elec', from_t, to_t, 'lcmv')

    # meg_data = test_all_verts(forward, data_cov, noise_cov, evoked, epochs)


if __name__ == '__main__':
    MEG_SUBJECT = 'ep001'
    MRI_SUBJECT = 'mg78'
    constrast='interference'
    raw_cleaning_method='nTSSS'
    task = meg_preproc.TASK_MSIT
    fname_format = '{subject}_msit_{raw_cleaning_method}_{constrast}_{cond}_1-15-{ana_type}.{file_type}'
    SUBJECT_MEG_FOL = os.path.join(SUBJECTS_MEG_DIR, TASKS[task], MEG_SUBJECT)
    SUBJECT_MRI_FOL = os.path.join(SUBJECTS_MRI_DIR, MRI_SUBJECT)
    BLENDER_SUB_FOL = os.path.join(BLENDER_ROOT_DIR, MRI_SUBJECT)
    MEG_ELEC_CONDS_TRANS = {'noninterference':0, 'interference':1}
    EVENTS_TRANS = {'noninterference':'neutral', 'interference':'interference'}
    EVENTS_TRANS_INV = {v:k for k, v in EVENTS_TRANS.items()}
    meg_preproc.init_globals(MEG_SUBJECT, MRI_SUBJECT, fname_format, True, raw_cleaning_method, constrast,
                             SUBJECTS_MEG_DIR, TASKS, task, SUBJECTS_MRI_DIR, BLENDER_ROOT_DIR)
    from src.preproc.meg_preproc import RAW, RAW_NOISE, FWD_X, EVO, EPO, EPO_NOISE, DATA_COV, NOISE_COV, \
        DATA_CSD, NOISE_CSD, NOISE_CSD_EMPTY_ROOM
    now = time.time()
    main()
    print('Finish! {}'.format(time.time() - now))