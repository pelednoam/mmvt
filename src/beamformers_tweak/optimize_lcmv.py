import numpy as np
import os
import mne
from functools import partial
import matplotlib.pyplot as plt

import utils
from beamformers import calc_subcortical_region_lcmv
from meg_preproc import calc_cov, get_cond_fname, get_file_name
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


def read_vars(events_id, region, fwd_fname, evo_fname, epo_fname, data_cov_fname, noise_cov_fnmae):
    for event in events_id.keys():
        forward = mne.read_forward_solution(get_cond_fname(fwd_fname, event, region=region)) #, surf_ori=True)
        epochs = mne.read_epochs(get_cond_fname(epo_fname, event))
        evoked = mne.read_evokeds(get_cond_fname(evo_fname, event), baseline=(None, 0))[0]
        noise_cov = calc_cov(get_cond_fname(data_cov_fname, event), event, epochs, None, 0)
        data_cov = calc_cov(get_cond_fname(noise_cov_fnmae, event), event, epochs, 0.0, 1.0)
        yield event, forward, data_cov, noise_cov, evoked, epochs


def get_fif_name(subject, fname_format, raw_cleaning_method, constrast, task):
    root_dir = os.path.join(SUBJECTS_MEG_DIR, task, subject)
    return partial(get_file_name, root_dir=root_dir, fname_format=fname_format, subject=subject,
        file_type='fif', raw_cleaning_method=raw_cleaning_method, constrast=constrast)


def init_msit(subject, constrast, raw_cleaning_method, region):
    events_id = dict(interference=1) # dict(interference=1, neutral=2)
    fname_format = '{subject}_msit_{raw_cleaning_method}_{constrast}_{cond}_1-15-{ana_type}.{file_type}'
    _get_fif_name = get_fif_name(subject, fname_format, raw_cleaning_method, constrast, 'MSIT')
    fwd_fname = _get_fif_name('{}-fwd'.format(region))
    evo_fname = _get_fif_name('ave')
    epo_fname = _get_fif_name('epo')
    data_cov_fname = _get_fif_name('data-cov')
    noise_cov_fnmae = _get_fif_name('noise-cov')
    return events_id, fwd_fname, evo_fname, epo_fname, data_cov_fname, noise_cov_fnmae


def load_electrode_msit_data(root_fol, from_t, to_t, positive=False, normalize_data=False, do_plot=False):
    bipolar = False
    electrode = 'LAT3-LAT2' if bipolar else 'LAT3'
    meg_conditions = ['interference', 'neutral']
    meg_elec_conditions_translate = {'interference':1, 'neutral':0}
    elec_data = load_electrode_data(root_fol, electrode, bipolar, meg_conditions, meg_elec_conditions_translate,
        from_t, to_t, normalize_data, positive, do_plot)
    return elec_data


def call_lcmv(forward, data_cov, noise_cov, evoked, epochs, all_verts=False, pick_ori=None, whitened_data_cov_reg=0.01):
    data = calc_subcortical_region_lcmv(forward, data_cov, noise_cov,
        evoked, epochs, pick_ori, whitened_data_cov_reg)
    data = data.T if all_verts else data.mean(0).T
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


def plot_activation_options(cond, meg_data_all, elec_data, electrode, from_i, to_i):
    xaxis = np.arange(from_i, to_i) - 500
    T = len(xaxis)
    f, axs = plt.subplots(len(meg_data_all.keys()), sharex=True)#, sharey=True)
    if len(meg_data_all.keys())==1:
        axs = [axs]
    for ind, ((params_option), ax) in enumerate(zip(meg_data_all.keys(), axs)):
        meg_data = meg_data_all[params_option]
        ax.plot(xaxis, meg_data[from_i: to_i], label=params_option, color='r')
        ax.plot(xaxis, elec_data[cond][from_i: to_i], label=electrode, color='b')
        ax.axvline(x=0, linestyle='--', color='k')
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.legend()
        plt.setp(ax.get_legend().get_texts(), fontsize='20')
        ax.set_title('{}-{}'.format(cond, params_option), fontsize=20)
        plt.xlabel('Time(ms)', fontsize=20)
    plt.xlim((from_t, to_t))
    plt.show()


def plot_activation_one_fig(cond, meg_data_all, elec_data, electrode, from_i, to_i):
    xaxis = np.arange(from_i, to_i) - 500
    colors = utils.get_spaced_colors(len(meg_data_all.keys()))
    plt.figure()
    for (label, meg_data), color in zip(meg_data_all.iteritems(), colors):
        plt.plot(xaxis, meg_data, label=label, color=color)
    plt.plot(xaxis, elec_data[cond], label=electrode, color='k')
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

if __name__ == '__main__':
    meg_subject = 'ep001'
    mri_subject = 'mg78'
    constrast='interference'
    raw_cleaning_method='nTSSS'
    region = 'Left-Hippocampus'
    from_t, to_t = 500, 1000# -500, 2000
    root_fol = os.path.join(BLENDER_ROOT_DIR, mri_subject)
    subject_meg_fol = os.path.join(SUBJECTS_MEG_DIR, 'MSIT', meg_subject)
    sub_corticals_codes_file = os.path.join(BLENDER_ROOT_DIR, 'sub_cortical_codes.txt')
    elec_data = load_electrode_msit_data(root_fol, from_t, to_t, positive=True, normalize_data=True)
    events_id, fwd_fname, evo_fname, epo_fname, data_cov_fname, noise_cov_fnmae = init_msit(meg_subject, constrast, raw_cleaning_method, region)
    vars = read_vars(events_id, region, fwd_fname, evo_fname, epo_fname, data_cov_fname, noise_cov_fnmae)
    for cond, forward, data_cov, noise_cov, evoked, epochs in vars:
        meg_data = test_all_verts(forward, data_cov, noise_cov, evoked, epochs)

    # cond = 'interference'
    # meg_data = load_all_subcorticals(subject_meg_fol, sub_corticals_codes_file, cond, from_t, to_t, normalize=True)
    # plot_activation_one_fig(cond, meg_data, elec_data, 'elec', from_t, to_t)

    # meg_data = call_lcmv(events_id, forward, data_cov, noise_cov, evoked, epochs)
    # plot_activation(events_id, meg_data, elec_data, 'elec', from_t, to_t, 'lcmv')