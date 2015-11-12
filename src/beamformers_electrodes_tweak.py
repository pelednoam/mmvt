import numpy as np
import os
import mne
import multiprocessing
from functools import partial
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from mne.beamformer import lcmv
from mne.time_frequency import compute_epochs_csd

import utils
from meg_preproc import (calc_cov, calc_csd, get_cond_fname, get_file_name, make_forward_solution_to_specific_points, TASKS)
import meg_preproc
from compare_meg_electrodes import load_electrode_data, plot_activation

LINKS_DIR = utils.get_links_dir()
SUBJECTS_MEG_DIR = os.path.join(LINKS_DIR, 'meg')
SUBJECTS_MRI_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
FREE_SURFER_HOME = utils.get_link_dir(LINKS_DIR, 'freesurfer', 'FREESURFER_HOME')
BLENDER_ROOT_DIR = os.path.join(LINKS_DIR, 'mmvt')


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


def read_vars(events_id, region, read_csd=True):
    for event in events_id.keys():
        forward, noise_csd, data_csd = None, None, None
        if not region is None:
            forward = mne.read_forward_solution(get_cond_fname(FWD_X, event, region=region)) #, surf_ori=True)
        epochs = mne.read_epochs(get_cond_fname(EPO, event))
        evoked = mne.read_evokeds(get_cond_fname(EVO, event), baseline=(None, 0))[0]
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


def load_electrode_msit_data(bipolar, electrode, root_fol, from_t=None, to_t=None, positive=False, normalize_data=False, do_plot=False):
    meg_conditions = ['interference', 'neutral']
    meg_elec_conditions_translate = {'interference':1, 'neutral':0}
    elec_data = load_electrode_data(root_fol, electrode, bipolar, meg_conditions, meg_elec_conditions_translate,
        from_t, to_t, normalize_data, positive, do_plot)
    return elec_data


def get_electrode_fwd(subject, electrode, events_id, bipolar=False, overwrite_fwd=False):
    fwd_elec_fnames = [get_cond_fname(FWD_X, cond, region=electrode) for cond in events_id.keys()]
    if not np.all([os.path.isfile(fname) for fname in fwd_elec_fnames]) or overwrite_fwd:
        subject_mri_dir = os.path.join(SUBJECTS_MRI_DIR, subject)
        positions_file_name = 'electrodes{}_positions.npz'.format('_bipolar' if bipolar else '')
        positions_file_name = os.path.join(subject_mri_dir, 'electrodes', positions_file_name)
        d = np.load(positions_file_name)
        names, pos = d['names'], d['pos']
        index = np.where(names==electrode)[0][0]
        elec_pos = np.array([pos[index]])
        make_forward_solution_to_specific_points(events_id, elec_pos, electrode, EPO, FWD_X,
            n_jobs=4, usingEEG=True)


def call_lcmv(forward, data_cov, noise_cov, evoked, epochs, all_verts=False, pick_ori=None, whitened_data_cov_reg=0.01):
    stc = lcmv(evoked, forward, noise_cov, data_cov, reg=whitened_data_cov_reg, pick_ori=pick_ori, rank=None)
    data = stc.data[:len(stc.vertices)]
    data = data.T if all_verts else data.mean(0).T
    return data


def call_dics(forward, evoked, noise_csd, data_csd, cond='', data_fname='', all_verts=False):
    from mne.beamformer import dics
    if data_fname == '':
        fmin, fmax = map(int, np.round(data_csd.frequencies)[[0, -1]])
        data_fname = os.path.join(SUBJECT_MEG_FOL, 'subcorticals', 'dics_{}-{}-{}-{}.npy'.format(cond, electrode, fmin, fmax))
    if not os.path.isfile(data_fname):
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
        for (key, data), color in zip(meg_data.iteritems(), colors):
            plt.plot(xaxis, meg_data[key], label=key, color=color)
    else:
        plt.plot(xaxis, meg_data, label='MEG', color='r')
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
    for (label, meg_data), color in zip(meg_data_all.iteritems(), colors):
        plt.plot(xaxis, meg_data, label=label, color=color)
    plt.plot(xaxis, elec_data, label=electrode, color='k')
    plt.axvline(x=0, linestyle='--', color='k')
    plt.legend()
    # plt.set_title('{}-{}'.format(cond, params_option), fontsize=20)
    plt.xlabel('Time(ms)', fontsize=20)
    plt.show()


def plot_all_vertices(meg_data, elec_data, from_i, to_i):
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
    return {'all': call_lcmv(forward, data_cov, noise_cov, evoked, epochs, all_verts=True)}


def normalize_meg_data(meg_data, elec_data, from_t, to_t, sigma=0):
    if sigma != 0:
        meg_data = gaussian_filter1d(meg_data, sigma)
    meg_data = meg_data[from_t:to_t]
    meg_data *= 1/max(meg_data)
    meg_data -= meg_data[0] - elec_data[0]
    return meg_data


def normalize_elec_data(elec_data, from_t, to_t):
    elec_data = elec_data[from_t:to_t]
    elec_data = elec_data - min(elec_data)
    elec_data *= 1/max(elec_data)
    return elec_data

def smooth_meg_data(meg_data):
    meg_data_all = {}
    for sigma in [8, 10, 12]:
        meg_data_all[sigma] = gaussian_filter1d(meg_data, sigma)
    return meg_data_all


def check_electrodes():
    meg_data_all, elec_data_all = {}, {}
    electrodes = ['LAT1', 'LAT2', 'LAT3', 'LAT4']
    vars = read_vars(events_id, None)
    for cond, forward, evoked, epochs, data_cov, noise_cov, data_csd, noise_csd in vars:
        for electrode in electrodes:
            get_electrode_fwd(MRI_SUBJECT, electrode, events_id, bipolar, overwrite_fwd=False)
            forward = mne.read_forward_solution(get_cond_fname(FWD_X, cond, region=electrode)) #, surf_ori=True)
            elec_data = load_electrode_msit_data(bipolar, electrode, BLENDER_SUB_FOL, positive=True, normalize_data=True)
            meg_data = call_dics(forward, evoked, noise_csd, data_csd, cond)
            elec_data_norm, meg_data_norm = normalize_data(elec_data[cond], meg_data, from_t, to_t)
            meg_data_norm = gaussian_filter1d(meg_data_norm, 10)
            meg_data_all[electrode] = meg_data_norm
            elec_data_all[electrode] = elec_data_norm
        plot_activation_options(meg_data_all, elec_data_all, electrodes, 500, elec_opts=True)


def check_freqs(events_id, electrode, from_t, to_t, gk_sigma=3, bipolar=False, njobs=6):
    vars = list(read_vars(events_id, electrode, read_csd=False))
    elec_data = load_electrode_msit_data(bipolar, electrode, BLENDER_SUB_FOL, positive=True, normalize_data=True)
    freqs1 = [(0, 4), (2, 6), (4,8), (6,10), (8,12), (10, 14), (12, 16), (12, 25), (25, 40), (40, 100), (80, np.inf), (0, np.inf)]
    freqs2 = [(1, 3), (3, 5), (40, 100), (60, 100), (80, 120), (100, 140)]
    freqs3 = [(0, np.inf), (1, 3), (3, 5), (100, 140)]
    freqs = set(freqs1) | set(freqs2) | set(freqs3)
    for cond, forward, evoked, epochs, data_cov, noise_cov, _, _ in vars:
        meg_data = {}
        elec_data_norm = normalize_elec_data(elec_data[cond], from_t, to_t)
        params = [(cond, forward, evoked, epochs, data_cov, noise_cov, elec_data_norm, electrode,
                   from_t, to_t, gk_sigma, fmin, fmax) for fmin, fmax in freqs]
        results = utils.run_parallel(_par_calc_dics_frqs, params, njobs)
        for data, fmin, fmax in results:
            meg_data[(fmin, fmax)] = data

        # meg_data['sum'] = sum([data for data in meg_data.values()])
        plot_activation_cond(cond, meg_data, elec_data_norm, electrode, 500)


def _par_calc_dics_frqs(p):
    cond, forward, evoked, epochs, data_cov, noise_cov, elec_data_norm, electrode, from_t, to_t, gk_sigma, fmin, fmax = p
    data_fname = os.path.join(SUBJECT_MEG_FOL, 'subcorticals', 'dics_{}-{}-{}-{}.npy'.format(cond, electrode, fmin, fmax))
    print('dics for {}-{}-{}-{}'.format(cond, electrode, fmin, fmax))
    if not os.path.isfile(data_fname):
        noise_csd = compute_epochs_csd(epochs, 'multitaper', tmin=-0.5, tmax=0.0, fmin=fmin, fmax=fmax)
        data_csd = compute_epochs_csd(epochs, 'multitaper', tmin=0.0, tmax=1.0, fmin=fmin, fmax=fmax)
        data = call_dics(forward, evoked, noise_csd, data_csd, data_fname=data_fname)
    else:
        data = np.load(data_fname)
    data = normalize_meg_data(data, elec_data_norm, from_t, to_t, gk_sigma)
    plot_activation_cond(cond, {(fmin, fmax):data}, elec_data_norm, electrode, 500, do_plot=False, plt_fname='{}.png'.format(data_fname[:-4]))
    return data, fmin, fmax


def comp_lcmv_dics_electrode(events_id, electrode, bipolar):
    elec_data = load_electrode_msit_data(bipolar, electrode, BLENDER_SUB_FOL, positive=True, normalize_data=True)
    get_electrode_fwd(MRI_SUBJECT, electrode, events_id, bipolar, overwrite_fwd=False)
    vars = read_vars(events_id, electrode)
    for cond, forward, evoked, epochs, data_cov, noise_cov, data_csd, noise_csd in vars:
        meg_data_lcmv = call_lcmv(forward, data_cov, noise_cov, evoked, epochs)
        meg_data_dics = call_dics(forward, evoked, noise_csd, data_csd, cond)
        elec_data_norm = normalize_elec_data(elec_data[cond], from_t, to_t)
        meg_data_lcmv_norm = normalize_meg_data(meg_data_lcmv, elec_data_norm, from_t, to_t, 3)
        meg_data_dics_norm = normalize_meg_data(meg_data_dics, elec_data_norm, from_t, to_t, 3)
        plot_activation_cond(cond, {'lcmv': meg_data_lcmv_norm, 'dics': meg_data_dics_norm}, elec_data_norm, electrode, 500)


if __name__ == '__main__':
    MEG_SUBJECT = 'ep001'
    MRI_SUBJECT = 'mg78'
    constrast='interference'
    raw_cleaning_method='nTSSS'
    task = meg_preproc.TASK_MSIT
    fname_format = '{subject}_msit_{raw_cleaning_method}_{constrast}_{cond}_1-15-{ana_type}.{file_type}'
    SUBJECT_MEG_FOL = os.path.join(SUBJECTS_MEG_DIR, TASKS[task], MEG_SUBJECT)
    events_id = dict(interference=1) # dict(interference=1, neutral=2)
    meg_preproc.init_globals(MEG_SUBJECT, MRI_SUBJECT, fname_format, True, raw_cleaning_method, constrast,
                             SUBJECTS_MEG_DIR, TASKS, task, SUBJECTS_MRI_DIR, BLENDER_ROOT_DIR)
    from meg_preproc import FWD_X, EVO, EPO, DATA_COV, NOISE_COV, DATA_CSD, NOISE_CSD
    region = 'Left-Hippocampus'
    from_t, to_t = 500, 1000# -500, 2000
    bipolar = False
    electrode = 'LAT3-LAT2' if bipolar else 'LAT3'
    use_fwd_for_region = False
    BLENDER_SUB_FOL = os.path.join(BLENDER_ROOT_DIR, MRI_SUBJECT)
    sub_corticals_codes_file = os.path.join(BLENDER_ROOT_DIR, 'sub_cortical_codes.txt')
    check_freqs(events_id, electrode, from_t, to_t, gk_sigma=3, bipolar=False, njobs=6)
    # check_electrodes()
    # comp_lcmv_dics_electrode(events_id, electrode, bipolar)


    # plot_activation_one_fig(cond, meg_data_norm, elec_data_norm, electrode, 500)


    # cond = 'interference'
    # meg_data = load_all_subcorticals(subject_meg_fol, sub_corticals_codes_file, cond, from_t, to_t, normalize=True)
    # plot_activation_one_fig(cond, meg_data, elec_data, 'elec', from_t, to_t)

    # meg_data = call_lcmv(forward, data_cov, noise_cov, evoked, epochs)
    # plot_activation(events_id, meg_data, elec_data, 'elec', from_t, to_t, 'lcmv')

    # meg_data = test_all_verts(forward, data_cov, noise_cov, evoked, epochs)
