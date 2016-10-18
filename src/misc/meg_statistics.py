from surfer import Brain
import os.path as op
import numpy as np
from scipy import stats as stats
import scipy
import glob
from multiprocessing import Pool
import os
import traceback
import gc
from functools import partial
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
from collections import defaultdict

import mne
from mne import spatial_tris_connectivity, grade_to_tris
from mne.stats import spatio_temporal_cluster_test, spatio_temporal_cluster_1samp_test, summarize_clusters_stc
from mne.minimum_norm import (write_inverse_operator, make_inverse_operator, apply_inverse,
        apply_inverse_epochs, read_inverse_operator)
from mne.source_estimate import _make_stc, VolSourceEstimate

# data_path = mne.datasets.sample.data_path()


# from surfer import Brain
# from surfer import viz

import utils

COMP_ROOT = utils.get_exisiting_dir(('/homes/5/npeled/space3', '/home/noam'))
LOCAL_SUBJECTS_DIR = op.join(COMP_ROOT, 'subjects')
REMOTE_ROOT_DIR = '/autofs/space/lilli_001/users/DARPA-MEG/ecr'
LOCAL_ROOT_DIR = op.join(COMP_ROOT, 'MEG/ECR/group')
BLENDER_DIR = op.join(COMP_ROOT, 'visualization_blender')
SUBJECTS_DIR = '/autofs/space/lilli_001/users/DARPA-MEG/freesurfs'
os.environ['SUBJECTS_DIR'] = LOCAL_SUBJECTS_DIR

blender_template = op.join(BLENDER_DIR, 'fsaverage', '{patient}_{cond_name}')


def get_subjects():
    if not op.isfile(op.join(LOCAL_ROOT_DIR, 'subjects.npy')):
        epos = glob.glob(op.join(REMOTE_ROOT_DIR, 'ave', '*_ecr_nTSSS_conflict-epo.fif'))
        subjects = [utils.namebase(s).split('_')[0] for s in epos]
        np.save(op.join(LOCAL_ROOT_DIR, 'subjects'), subjects)
        print(subjects)
    else:
        subjects = np.load(op.join(LOCAL_ROOT_DIR, 'subjects.npy'))
    return subjects

def get_healthy_controls():
    return [subject for subject in get_subjects() if subject[:2]=='hc']


def get_patients():
    return [subject for subject in get_subjects() if subject[:2]!='hc']


def calc_all(events_id, tmin=None, tmax=None, overwrite=False,inverse_method='dSPM',
        baseline=(None, 0), apply_for_epochs=False, apply_SSP_projection_vectors=True, add_eeg_ref=True, n_jobs=1):
    params = [(subject, events_id, tmin, tmax, overwrite, inverse_method, baseline, apply_for_epochs,
               apply_SSP_projection_vectors, add_eeg_ref) for subject in get_subjects()]
    utils.parallel_run(pool, _calc_all, params, n_jobs)


def _calc_all(params):
    subject, events_id, tmin, tmax, overwrite, \
        inverse_method, baseline, apply_for_epochs, apply_SSP_projection_vectors, add_eeg_ref = params

    # todo: should calulcate this value
    epochs_num = 36
    if apply_for_epochs:
        file_exists = len(glob.glob(op.join(LOCAL_ROOT_DIR, 'stc_epochs', '{}_{}_*_{}-stc.h5'.format(subject, events_id.keys()[0], inverse_method)))) == epochs_num
    else:
        file_exists = op.isfile(op.join(LOCAL_ROOT_DIR, 'stc', '{}_{}_{}-stc.h5'.format(subject, events_id.keys()[0], inverse_method)))
    if file_exists:
        print('stcs was already calcualted for {}'.format(subject))
    else:
        epochs = _calc_epoches((subject, events_id, tmin, tmax))
        if not apply_for_epochs:
            evoked = _calc_evoked((subject, events_id, epochs))
        else:
            evoked = None
        inv = _calc_inverse((subject, epochs, overwrite))
        _create_stcs((subject, events_id, epochs, evoked, inv, inverse_method,
            baseline, apply_for_epochs, apply_SSP_projection_vectors, add_eeg_ref))


def _calc_evoked(events_id, epochs):
    params = [(subject, events_id, epochs) for subject in get_subjects()]
    utils.parallel_run(pool, _calc_inverse, params, n_jobs)


def _calc_evoked(params):
    subject, events_id, epochs = params
    evoked = {}
    for cond_name in events_id.keys():
        evoked[cond_name] = epochs[cond_name].average()
        evo = op.join(LOCAL_ROOT_DIR, 'evo', '{}_ecr_{}-ave.fif'.format(subject, cond_name))
        mne.write_evokeds(evo, evoked[cond_name])
    return evoked


def calc_inverse(epochs=None, overwrite=False):
    params = [(subject, epochs, overwrite) for subject in get_subjects()]
    utils.parallel_run(pool, _calc_inverse, params, n_jobs)


def _calc_inverse(params):
    subject, epochs, overwrite = params
    epo = op.join(REMOTE_ROOT_DIR, 'ave', '{}_ecr_nTSSS_conflict-epo.fif'.format(subject))
    fwd = op.join(REMOTE_ROOT_DIR, 'fwd', '{}_ecr-fwd.fif'.format(subject))
    local_inv_file_name = op.join(LOCAL_ROOT_DIR, 'inv', '{}_ecr_nTSSS_conflict-inv.fif'.format(subject))

    if os.path.isfile(local_inv_file_name) and not overwrite:
        inverse_operator = read_inverse_operator(local_inv_file_name)
        print('inv already calculated for {}'.format(subject))
    else:
        if epochs is None:
            epochs = mne.read_epochs(epo)
        noise_cov = mne.compute_covariance(epochs.crop(None, 0, copy=True))
        inverse_operator = None
        if not os.path.isfile(fwd):
            print('no fwd for {}'.format(subject))
        else:
            forward = mne.read_forward_solution(fwd)
            inverse_operator = make_inverse_operator(epochs.info, forward, noise_cov,
                loose=None, depth=None)
            write_inverse_operator(local_inv_file_name, inverse_operator)
    return inverse_operator
    #     forward_sub = mne.read_forward_solution(get_cond_fname(FWD_SUB, cond))
    #     inverse_operator_sub = make_inverse_operator(epochs.info, forward_sub, noise_cov,
    #         loose=None, depth=None)
    #     write_inverse_operator(get_cond_fname(INV_SUB, cond), inverse_operator_sub)


def calc_epoches(events_id, tmin, tmax, n_jobs=1):
    params = [(subject, events_id, tmin, tmax) for subject in get_subjects()]
    utils.parallel_run(pool, _calc_epoches, params, n_jobs)


def _calc_epoches(params):
    subject, events_id, tmin, tmax = params
    out_file = op.join(LOCAL_ROOT_DIR, 'epo', '{}_ecr_nTSSS_conflict-epo.fif'.format(subject))
    if not op.isfile(out_file):
        events = mne.read_events(op.join(REMOTE_ROOT_DIR, 'events', '{}_ecr_nTSSS_conflict-eve.fif'.format(subject)))
        raw = mne.io.Raw(op.join(REMOTE_ROOT_DIR, 'raw', '{}_ecr_nTSSS_raw.fif'.format(subject)), preload=False)
        picks = mne.pick_types(raw.info, meg=True)
        epochs = find_epoches(raw, picks, events, events_id, tmin=tmin, tmax=tmax)
        epochs.save(out_file)
    else:
        epochs = mne.read_epochs(out_file)
    return epochs

def find_epoches(raw, picks,  events, event_id, tmin, tmax, baseline=(None, 0)):
    # remove events that are not in the events table
    event_id = dict([(k, ev) for (k, ev) in event_id.iteritems() if ev in np.unique(events[:, 2])])
    return mne.Epochs(raw=raw, events=events, event_id=event_id, tmin=tmin, tmax=tmax, proj=True,
                      picks=picks, baseline=baseline, preload=True, reject=None) # reject can be dict(mag=4e-12)


def create_stcs(events_id, epochs=None, evoked=None, inv=None, inverse_method='dSPM',
        baseline=(None, 0), apply_for_epochs=False, apply_SSP_projection_vectors=True, add_eeg_ref=True):
    params = [(subject, events_id, epochs, evoked, inv, inverse_method,
        baseline, apply_for_epochs, apply_SSP_projection_vectors, add_eeg_ref) for subject in get_subjects()]
    utils.parallel_run(pool, _create_stcs, params, n_jobs)


def _create_stcs(params):
    subject, events_id, epochs, evoked, inv, inverse_method, baseline, apply_for_epochs, apply_SSP_projection_vectors, add_eeg_ref = params
    snr = 3.0
    lambda2 = 1.0 / snr ** 2

    local_inv_file_name = op.join(LOCAL_ROOT_DIR, 'inv', '{}_ecr_nTSSS_conflict-inv.fif'.format(subject))
    if inv is None and os.path.isfile(local_inv_file_name):
        inv = read_inverse_operator(local_inv_file_name)
    if inv is None:
        return
    print([s['vertno'] for s in inv['src']])
    for cond_name in events_id.keys():
        if not apply_for_epochs:
            local_stc_file_name = op.join(LOCAL_ROOT_DIR, 'stc', '{}_{}_{}'.format(subject, cond_name, inverse_method))
            if os.path.isfile('{}-lh.stc'.format(local_stc_file_name)):
                print('stc was already calculated for {}'.format(subject))
            else:
                if evoked is None:
                    evoked_cond = mne.read_evokeds(op.join(LOCAL_ROOT_DIR, 'evo', '{}_ecr_{}-ave.fif'.format(subject, cond_name)))
                else:
                    evoked_cond = evoked[cond_name]
                stcs = apply_inverse(evoked_cond, inv, lambda2, inverse_method, pick_ori=None)
                stcs.save(local_stc_file_name, ftype='h5')
        else:
            local_stc_template = op.join(LOCAL_ROOT_DIR, 'stc_epochs', '{}_{}_{}_{}'.format(subject, cond_name, '{epoch_ind}', inverse_method))
            if len(glob.glob(local_stc_template.format(epoch_ind='*') == 36)):
                print('stc was already calculated for {}'.format(subject))
            else:
                stcs = apply_inverse_epochs(epochs[cond_name], inv, lambda2, inverse_method, pick_ori=None, return_generator=True)
                for epoch_ind, stc in enumerate(stcs):
                    if not os.path.isfile(local_stc_template.format(epoch_ind=epoch_ind)):
                        stc.save(local_stc_template.format(epoch_ind=epoch_ind), ftype='h5')


def morph_stcs_to_fsaverage(events_id, stc_per_epoch=False, inverse_method='dSPM', subjects_dir='', n_jobs=1):
    if subjects_dir is '':
        subjects_dir = os.environ['SUBJECTS_DIR']
    for subject in get_subjects():
        for cond_name in events_id.keys():
            print('morphing {}, {}'.format(subject, cond_name))
            if not stc_per_epoch:
                morphed_stc_file_name = op.join(LOCAL_ROOT_DIR, 'stc_morphed', '{}_{}_morphed_{}'.format(subject, cond_name, inverse_method))
                if op.isfile('{}-stc.h5'.format(morphed_stc_file_name)):
                    print('{} {} already morphed'.format(subject, cond_name))
                else:
                    local_stc_file_name = op.join(LOCAL_ROOT_DIR, 'stc', '{}_{}_{}'.format(subject, cond_name, inverse_method))
                    if op.isfile('{}-stc.h5'.format(local_stc_file_name)):
                        stc = mne.read_source_estimate(local_stc_file_name)
                        stc_morphed = mne.morph_data(subject, 'fsaverage', stc, grade=5, smooth=20,
                            subjects_dir=subjects_dir)
                        stc_morphed.save(morphed_stc_file_name, ftype='h5')
                    else:
                        print("can't find stc file for {}, {}".format(subject, cond_name))
            else:
                stcs = glob.glob(op.join(LOCAL_ROOT_DIR, 'stc_epochs', '{}_{}_*_{}-stc.h5'.format(subject, cond_name, inverse_method)))
                params = [(subject, cond_name, stc_file_name, inverse_method, subjects_dir) for stc_file_name in stcs]
                utils.parallel_run(pool, _morphed_epochs_files, params, n_jobs)


def _morphed_epochs_files(params):
    subject, cond_name, stc_file_name, inverse_method, subjects_dir = params
    print('morphing {}'.format(stc_file_name))
    epoch_id = utils.namebase(stc_file_name).split('_')[2]
    morphed_stc_file_name = op.join(LOCAL_ROOT_DIR, 'stc_epochs_morphed',  '{}_{}_{}_{}'.format(subject, cond_name, epoch_id, inverse_method))
    if not op.isfile('{}-stc.h5'.format(morphed_stc_file_name)):
        stc = mne.read_source_estimate(stc_file_name)
        stc_morphed = mne.morph_data(subject, 'fsaverage', stc, grade=5, smooth=20,
            subjects_dir=subjects_dir)
        stc_morphed.save(morphed_stc_file_name, ftype='h5')
    else:
        print('{} {} {} already morphed'.format(subject, cond_name, epoch_id))


def calc_average_hc_epochs_stc(events_id, inverse_method='dSPM'):
    epochs_num = len(glob.glob(op.join(LOCAL_ROOT_DIR, 'stc_epochs', '{}_{}_*_{}-stc.h5'.format(get_healthy_controls()[0], events_id.keys()[0], inverse_method))))
    epochs_num = 36
    for cond_name in events_id.keys():
        for epoch_id in range(epochs_num):
            averaged_stc_file_name = op.join(LOCAL_ROOT_DIR, 'stc_hc_average_epochs',  '{}_{}_{}'.format(cond_name, epoch_id, inverse_method))
            if op.isfile('{}.npy'.format(averaged_stc_file_name)):
                print('averaged hc stc was already calculated for {} {}'.format(cond_name, epoch_id))
            else:
                stc_files = glob.glob(op.join(LOCAL_ROOT_DIR, 'stc_epochs_morphed', '*_{}_{}_{}-stc.h5'.format(cond_name, epoch_id, inverse_method)))
                # Get only the healthy controls
                stc_files = [stc_file_name for stc_file_name in stc_files if utils.namebase(stc_file_name).split('_')[0] in get_healthy_controls()]
                stcs = []
                for ind, stc_file_name in enumerate(stc_files):
                    if op.isfile(stc_file_name):
                        stc = mne.read_source_estimate(stc_file_name)
                        stcs.append(stc.data)
                stcs = np.array(stcs)
                stc_avg = stcs.data.mean(0).reshape(stc.data.shape)
                np.save(averaged_stc_file_name, stc_avg)


def get_hc_morphed_stcs(tmin, tmax, cond_name, inverse_method='dSPM'):
    hcs = get_healthy_controls()
    stcs = []
    for subject in hcs:
        stc_file_name = op.join(LOCAL_ROOT_DIR, 'stc_morphed', '{}_{}_morphed_{}-stc.h5'.format(subject, cond_name, inverse_method))
        if op.isfile(stc_file_name):
            stc = mne.read_source_estimate(stc_file_name)
            stc.crop(tmin, tmax)
            data = stc.data.T # time x space
            stcs.append(data)
        else:
            print("can't find the file {}".format(stc_file_name))
    return np.array(stcs)


def get_morphed_epochs_stcs(tmin, tmax, cond_name, subsects_to_choose_from=None, stcs_num=None, inverse_method='dSPM'):
    if subsects_to_choose_from is None:
        subsects_to_choose_from = get_subjects()
    epochs = glob.glob(op.join(LOCAL_ROOT_DIR, 'stc_epochs_morphed', '*_{}_*_{}-stc.h5'.format(cond_name, inverse_method)))
    epochs_subset = [ep for ep in epochs if utils.namebase(ep).split('_')[0] in subsects_to_choose_from]
    if len(epochs_subset) == 0 or stcs_num is not None and len(epochs_subset) < stcs_num:
        print('subset to choose from is smaller({}) than {}! {}'.format(len(epochs_subset), stcs_num, subsects_to_choose_from))
        return None
    if stcs_num is not None:
        epochs_subset = random.sample(epochs_subset, stcs_num)
    for ind, epoc_helthy_file_name in enumerate(epochs_subset):
        print('reading stc {}/{}'.format(ind, len(epochs_subset)))
        stc = mne.read_source_estimate(epoc_helthy_file_name)
        stc.crop(tmin, tmax)
        if ind==0:
            stcs = np.zeros(stc.data.shape + (len(epochs_subset), ))
        # data = stc.data.T # time x space
        stcs[:, :, ind] = stc.data
    return stcs


def permutation_test_on_source_data_with_spatio_temporal_clustering(controls_data, patient_data, patient, cond_name,
                tstep, n_permutations, inverse_method='dSPM', n_jobs=6):
    try:
        print('permutation_test: patient {}, cond {}'.format(patient, cond_name))
        connectivity = spatial_tris_connectivity(grade_to_tris(5))
        #    Note that X needs to be a list of multi-dimensional array of shape
        #    samples (subjects_k) x time x space, so we permute dimensions
        print(controls_data.shape, patient_data.shape)
        X = [controls_data, patient_data]

        p_threshold = 0.05
        f_threshold = stats.distributions.f.ppf(1. - p_threshold / 2.,
                                                controls_data.shape[0] - 1, 1)
        print('Clustering. thtreshold = {}'.format(f_threshold))
        T_obs, clusters, cluster_p_values, H0 = clu =\
            spatio_temporal_cluster_test(X, connectivity=connectivity, n_jobs=n_jobs, threshold=10, n_permutations=n_permutations)

        results_file_name = op.join(LOCAL_ROOT_DIR, 'clusters_results', '{}_{}_{}'.format(patient, cond_name, inverse_method))
        np.savez(results_file_name, T_obs=T_obs, clusters=clusters, cluster_p_values=cluster_p_values, H0=H0)
        #    Now select the clusters that are sig. at p < 0.05 (note that this value
        #    is multiple-comparisons corrected).
        good_cluster_inds = np.where(cluster_p_values < 0.05)[0]

        ###############################################################################
        # Visualize the clusters

        print('Visualizing clusters.')

        #    Now let's build a convenient representation of each cluster, where each
        #    cluster becomes a "time point" in the SourceEstimate
        fsave_vertices = [np.arange(10242), np.arange(10242)]
        stc_all_cluster_vis = summarize_clusters_stc(clu, tstep=tstep,
                                                     vertices=fsave_vertices,
                                                     subject='fsaverage')
        stc_all_cluster_vis.save(op.join(LOCAL_ROOT_DIR, 'stc_clusters', '{}_{}_{}'.format(patient, cond_name, inverse_method)), ftype='h5')

        # #    Let's actually plot the first "time point" in the SourceEstimate, which
        # #    shows all the clusters, weighted by duration
        # # blue blobs are for condition A != condition B
        # brain = stc_all_cluster_vis.plot('fsaverage', 'inflated', 'both',
        #                                  subjects_dir=subjects_dir, clim='auto',
        #                                  time_label='Duration significant (ms)')
        # brain.set_data_time_index(0)
        # brain.show_view('lateral')
        # brain.save_image('clusters.png')
    except:
        print('bummer! {}, {}'.format(patient, cond_name))
        print(traceback.format_exc())


def run_permutation_tests(tmin=None, tmax=None, calc_per_cond=True, calc_constrast=False, calc_mean=False, n_permutations=1024, inverse_method='dSPM', n_jobs=1):
    perm_test = partial(permutation_test_on_source_data_with_spatio_temporal_clustering, n_permutations=n_permutations,
                       inverse_method=inverse_method, n_jobs=n_jobs)
    for patient in ['pp005']: #get_patients():
        for cond_id, cond_name in enumerate(events_id.keys()):
            controls_data = get_hc_morphed_stcs(tmin, tmax, cond_name, inverse_method)
            controls_data = np.abs(controls_data) # only magnitude
            patient_stc_file_name = op.join(LOCAL_ROOT_DIR, 'stc_morphed', '{}_{}_morphed_{}-stc.h5'.format(patient, cond_name, inverse_method))
            patient_stc = mne.read_source_estimate(patient_stc_file_name)
            patient_stc.crop(tmin, tmax)
            tstep = patient_stc.tstep
            patient_data = patient_stc.data.T
            patient_data = patient_data[np.newaxis, :, :]
            patient_data = np.abs(patient_data) # only magnitude
            if calc_per_cond:
                perm_test(controls_data, patient_data, patient, cond_name, tstep)
            if cond_id==0 and (calc_constrast or calc_mean):
                controls_data_shape = controls_data.shape
                patient_data_shape = patient_data.shape
                patient_data_all = np.zeros((2,) + patient_data_shape)
                controls_data_all = np.zeros((2,) + controls_data_shape)
            if calc_constrast or calc_mean:
                patient_data_all[cond_id] = patient_data
                controls_data_all[cond_id] = controls_data
            del controls_data, patient_data, patient_stc
            gc.collect()

        if calc_constrast:
            x1 = np.diff(controls_data_all, axis=0).reshape(controls_data_shape)
            x2 = np.diff(patient_data_all, axis=0).reshape(patient_data_shape)
            cond_name = 'contrast'
        elif calc_mean:
            x1 = np.mean(controls_data_all, axis=0).reshape(controls_data_shape)
            x2 = np.mean(patient_data_all, axis=0).reshape(patient_data_shape)
            cond_name = 'mean'
        else:
            continue

        perm_test(x1, x2, patient, cond_name, tstep)
        del x1, x2
        gc.collect()


def run_permutation_ttest(tmin=None, tmax=None, p_threshold = 0.05, n_permutations=1024, inverse_method='dSPM', n_jobs=1):
    for cond_id, cond_name in enumerate(events_id.keys()):
        #todo: calc the 36
        controls_data = get_morphed_epochs_stcs(tmin, tmax, cond_name, get_healthy_controls(),
            36, inverse_method)
        controls_data = abs(controls_data)
        for patient in get_patients():
            try:
                print(patient, cond_name)
                patient_data = get_morphed_epochs_stcs(tmin, tmax, cond_name, [patient], None, inverse_method)
                patient_data = abs(patient_data)
                print(patient_data.shape, controls_data.shape)
                data = controls_data - patient_data
                del patient_data
                gc.collect()
                data = np.transpose(data, [2, 1, 0])
                connectivity = spatial_tris_connectivity(grade_to_tris(5))
                t_threshold = -stats.distributions.t.ppf(p_threshold / 2., data.shape[0] - 1)
                T_obs, clusters, cluster_p_values, H0 = \
                    spatio_temporal_cluster_1samp_test(data, connectivity=connectivity, n_jobs=n_jobs,
                        threshold=t_threshold, n_permutations=n_permutations)
                results_file_name = op.join(LOCAL_ROOT_DIR, 'permutation_ttest_results', '{}_{}_{}'.format(patient, cond_name, inverse_method))
                np.savez(results_file_name, T_obs=T_obs, clusters=clusters, cluster_p_values=cluster_p_values, H0=H0)
                good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
                print('good_cluster_inds: {}'.format(good_cluster_inds))
            except:
                print('bummer! {}, {}'.format(patient, cond_name))
                print(traceback.format_exc())


def read_permutation_ttest_reusults(events_id, subjects_dir, inverse_method='dSPM'):
    for cond_id, cond_name, patient, hc, res in patients_hcs_conds_gen(events_id, inverse_method):
        print(patient, hc, cond_name)
        good_cluster_inds = np.where(res['cluster_p_values'] < 0.05)[0]
        print('good cluster {}, min: {}'.format(len(good_cluster_inds), min(res['cluster_p_values'])))
        try:
            plot_ttest_results_using_pysurfer(patient, cond_name, res, subjects_dir)
        except:
            pass



def plot_ttest_results_using_pysurfer(patient, cond_name, res, subjects_dir):
    clu = (res['T_obs'], res['clusters'], res['cluster_p_values'], res['H0'])
    fsave_vertices = [np.arange(10242), np.arange(10242)]
    stc_all_cluster_vis = summarize_clusters_stc(clu, tstep=tstep,
        vertices=fsave_vertices, subject='fsaverage')
    # Let's actually plot the first "time point" in the SourceEstimate, which
    # shows all the clusters, weighted by duration
    # blue blobs are for condition A < condition B, red for A > B
    brain = stc_all_cluster_vis.plot('fsaverage', 'inflated', 'split', 'mne', views=['lat', 'med'],
        subjects_dir=subjects_dir, clim='auto',time_label='Duration significant (ms)')
    brain.set_data_time_index(0)
    brain.show_view('lateral')
    print('tada!')
    brain.save_image(op.join(LOCAL_ROOT_DIR, 'permutation_ttest_results', '{}_{}_clusters.jpg'.format(patient, cond_name)))
    brain.close()


def filter_out_clusters_from_ttest_results(events_id, p_thresh=0.05, inverse_method='dSPM'):
    for cond_id, cond_name, patient, hc, res in patients_hcs_conds_gen(events_id, False, inverse_method):
        n_times, n_vertices = res.T_obs.shape
        good_cluster_inds = np.where(res.cluster_p_values < p_thresh)[0]
        if len(good_cluster_inds) > 0:
            data = np.zeros((n_vertices, n_times))
            for ii, cluster_ind in enumerate(good_cluster_inds):
                v_inds = res.clusters[cluster_ind][1]
                t_inds = res.clusters[cluster_ind][0]
                data[v_inds, t_inds] = res.T_obs[t_inds, v_inds]
            data = scipy.sparse.csr_matrix(data)
            np.save(op.join(LOCAL_ROOT_DIR, 'permutation_ttest_results',
                '{}_{}_{}_{}_clusters'.format(patient, hc, cond_name, inverse_method)), data)
        else:
            print('no sigificant clusters for {} {}'.format(patient, cond_name))


def plot_perm_ttest_results(events_id, inverse_method='dSPM', plot_type='scatter_plot'):
    print('plot_perm_ttest_results')
    all_data = defaultdict(dict)
    fsave_vertices = [np.arange(10242), np.arange(10242)]
    fs_pts = mne.vertex_to_mni(fsave_vertices, [0, 1], 'fsaverage', LOCAL_SUBJECTS_DIR) # 0 for lh
    for cond_id, cond_name, patient, hc, data in patients_hcs_conds_gen(events_id, True, inverse_method):
        all_data[patient][hc] = data[()]
    print(all_data.keys())
    for patient, pat_data in all_data.iteritems():
        print(patient)
        fol = op.join(LOCAL_ROOT_DIR, 'permutation_ttest_results', patient)
        utils.make_dir(fol)
        if op.isfile(op.join(fol, 'perm_ttest_points.npz')):
            d = np.load(op.join(fol, 'perm_ttest_points.npz'))
            if plot_type == 'scatter_plot':
                points, values = d['points'][()], d['values'][()]
            elif plot_type == 'pysurfer':
                vertices, vertives_values = d['vertices'][()], d['vertives_values'][()]
        else:
            points, values, vertices, vertives_values = calc_points(pat_data, fs_pts)
            np.savez(op.join(fol, 'perm_ttest_points'), points=points, values=values, vertices=vertices, vertives_values=vertives_values)
        max_vals = 8 # int(np.percentile([max(v) for v in values.values()], 70))
        print(max_vals)
        fol = op.join(fol, '{}_figures'.format(plot_type))
        utils.make_dir(fol)
        if plot_type == 'scatter_plot':
            scatter_plot_perm_ttest_results(points, values, fs_pts, max_vals, fol)
        elif plot_type == 'pysurfer':
            pysurfer_plot_perm_ttest_results(vertices, vertives_values, max_vals, fol)


def pysurfer_plot_perm_ttest_results(vertices, vertives_values, max_vals, fol):
    T = max(vertices.keys())
    for t in range(T+1):
        print(t)
        brain = Brain('fsaverage', 'split', 'pial', curv=False, offscreen=False, views=['lat', 'med'], title='{} ms'.format(t))
        for hemi in ['rh', 'lh']:
            if t in vertices:
                brain.add_data(np.array(vertives_values[t][hemi]), hemi=hemi, min=1, max=max_vals, remove_existing=True,
                         colormap="YlOrRd", alpha=1, vertices=np.array(vertices[t][hemi]))
        brain.save_image(os.path.join(fol, '{}.jpg'.format(t)))
        brain.close()


def scatter_plot_perm_ttest_results(points, values, fs_pts, max_vals, fol):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(fs_pts[0, :, 0], fs_pts[0, :, 1], fs_pts[0, :, 2], c='white', edgecolors='none', alpha=0.01)
    ax.scatter(fs_pts[1, :, 0], fs_pts[1, :, 1], fs_pts[1, :, 2], c='white', edgecolors='none', alpha=0.01)
    m = utils.get_scalar_map(1, max_vals, color_map='YlOrRd')
    for ind, t in enumerate(points.keys()):
        print(t)
        colors = utils.arr_to_colors(values[t], 1,  max_vals, colors_map='YlOrRd')
        sca = ax.scatter(points[t][:, 0], points[t][:, 1], points[t][:, 2], c=colors)
        if ind == 0:
            m.set_array(colors)
            plt.colorbar(m, ticks=range(1, max_vals + 1), shrink=.5, aspect=10)
            title_obj = plt.title('t={}'.format(t))
        else:
            plt.setp(title_obj, text='t={}'.format(t))
        plt.savefig(op.join(fol, '{}.jpg'.format(t)))
        sca.remove()
    plt.close()


def calc_points(patient_data, fs_pts):
    from collections import Counter
    # pts = defaultdict(dict)
    vt = defaultdict(list)
    vt_hemi = {}
    for ind, (hc, data) in enumerate(patient_data.iteritems()):
        print('{} / {}'.format(ind, len(patient_data.keys()) - 1))
        x = np.array(data.todense())
        inds = np.where(x)
        ts = np.unique(inds[1])
        for t in ts:
            if not t in vt_hemi:
                vt_hemi[t] = {'rh': [], 'lh': []}
            # hc_t[t].append(hc)
            t_inds = np.where(inds[1] == t)[0]
            v_inds = inds[0][t_inds]
            vt[t].extend(v_inds)
            vt_hemi[t]['rh'].extend(v_inds[v_inds < 10242])
            vt_hemi[t]['lh'].extend(v_inds[v_inds >= 10242] - 10242)
            # pts[t][hc] = fs_pts[0][v_inds[v_inds < 10242]]
            # pts[t][hc] = np.vstack((pts[t][hc], fs_pts[1][v_inds[v_inds >= 10242] - 10242]))
    points, values = defaultdict(list), defaultdict(list)
    vertices, vertives_values = {}, {}
    for t in vt.keys():
        if not t in vertices:
            vertices[t] = {'rh': [], 'lh': []}
        if not t in vertives_values:
            vertives_values[t] = {'rh': [], 'lh': []}

        if t % 10 == 0:
            print(t)
        cnt = Counter(vt[t])
        cnt_rh = Counter(vt_hemi[t]['rh'])
        cnt_lh = Counter(vt_hemi[t]['lh'])
        for v, num in cnt.iteritems():
            if not t in points:
                points[t] = vertice_to_fs_point(v, fs_pts)
            else:
                points[t] = np.vstack((points[t], vertice_to_fs_point(v, fs_pts)))
            values[t].append(num)
        for cnt, hemi in zip([cnt_lh, cnt_rh], ['rh', 'lh']):
            for v, num in cnt.iteritems():
                vertices[t][hemi].append(v)
                vertives_values[t][hemi].append(num)
    return points, values, vertices, vertives_values


def init_hemi_time_list_dic(t, dic):
    if not t in dic:
        dic[t] = {'rh': [], 'lh': []}
    return dic


def vertice_to_fs_point(v, fs_pts):
    if v < 10242:
        return fs_pts[0][v]
    else:
        return fs_pts[1][v - 10242]

def smooth_ttest_results(tmin, tstep, subjects_dir, inverse_method='dSPM', n_jobs=1):
    for cond_id, cond_name in enumerate(events_id.keys()):
        for patient in get_patients():
            results_file_name = op.join(LOCAL_ROOT_DIR, 'permutation_ttest_results', '{}_{}_{}_clusters.npy'.format(patient, cond_name, inverse_method))
            if op.isfile(results_file_name):
                data = np.load(results_file_name)
                print('smoothing {} {}'.format(patient, cond_name))
                fsave_vertices = [np.arange(10242), np.arange(10242)]
                stc = _make_stc(data, fsave_vertices, tmin=tmin, tstep=tstep, subject='fsaverage')
                vertices_to = mne.grade_to_vertices('fsaverage', grade=None, subjects_dir=subjects_dir)
                print(stc.data.shape, vertices_to[0].shape)
                stc_smooth = mne.morph_data('fsaverage', 'fsaverage', stc, n_jobs=n_jobs, grade=vertices_to, subjects_dir=subjects_dir)
                stc_smooth.save(op.join(LOCAL_ROOT_DIR, 'results_for_blender', '{}_{}_{}'.format(patient, cond_name, inverse_method)), ftype='h5')
            else:
                print('no results for {} {}'.format(patient, cond_name))


def save_ttest_result_for_blender(events_id, cm_big='YlOrRd', cm_small='PuBu', threshold=2, norm_by_percentile=True,
        norm_percs=(1,99), inverse_method='dSPM', do_print=False, n_jobs=1):
    for cond_id, cond_name in enumerate(events_id.keys()):
        for patient in get_patients():
            results_file_name = op.join(LOCAL_ROOT_DIR, 'results_for_blender', '{}_{}_{}'.format(patient, cond_name, inverse_method))
            if op.isfile('{}-stc.h5'.format(results_file_name)):
                print('{}, {}'.format(patient, cond_name))
                stc = mne.read_source_estimate(results_file_name)
                data_max, data_min = utils.get_activity_max_min(stc, norm_by_percentile, norm_percs, threshold)
                print(data_max, data_min)
                scalar_map_big = utils.get_scalar_map(threshold, data_max, cm_big)
                scalar_map_small = utils.get_scalar_map(data_min, -threshold, cm_small)
                for hemi in ['rh', 'lh']:
                    utils.check_stc_vertices(stc, hemi, op.join(BLENDER_DIR, 'fsaverage', '{}.pial.ply'.format(hemi)))
                    data = utils.stc_hemi_data(stc, hemi)
                    fol = '{}'.format(os.path.join(BLENDER_DIR, 'fsaverage', '{}_{}'.format(patient, cond_name), 'activity_map_{}').format(hemi))
                    utils.delete_folder_files(fol)
                    params = [(data[:, t], t, fol, scalar_map_big, scalar_map_small, threshold, do_print) for t in xrange(data.shape[1])]
                    utils.parallel_run(pool, _calc_activity_colors, params, n_jobs)
            else:
                print('no results for {} {}'.format(patient, cond_name))


def _calc_activity_colors(params):
    data, t, fol, scalar_map_big, scalar_map_small, threshold, do_print = params
    colors = utils.arr_to_colors_two_colors_maps(data, scalar_map_big=scalar_map_big,
        scalar_map_small=scalar_map_small, threshold=threshold, default_val=1)[:,:3]
    if do_print and t % 10 == 0:
        print(t)
    # colors = utils.arr_to_colors(data[:, t], 0, data_max, scalar_map=scalar_map)[:,:3]
    colors = np.hstack((np.reshape(data, (data.shape[0], 1)), colors))
    np.save(os.path.join(fol, 't{}'.format(t)), colors)


def calc_fsaverage_labels_indices(surf_name='pial', labels_from_annot=False, labels_fol='', parc='aparc250', subjects_dir=''):
    labels_fol = os.path.join(subjects_dir, 'fsaverage', 'label', parc) if labels_fol=='' else labels_fol
    if (labels_from_annot):
        labels = mne.read_labels_from_annot('fsaverage', parc=parc, hemi='both', surf_name=surf_name)
    else:
        labels = utils.read_labels(labels_fol)
    fsave_vertices = [np.arange(10242), np.arange(10242)]
    labels_vertices, labels_names = utils.get_labels_vertices(labels, fsave_vertices)
    np.savez(op.join(LOCAL_ROOT_DIR, 'fsaverage_labels_indices'), labels_vertices=labels_vertices, labels_names=labels_names)


def calc_labels_avg(events_id, tmin, inverse_method='dSPM', do_plot=False):
    d = np.load(op.join(LOCAL_ROOT_DIR, 'fsaverage_labels_indices.npz'))
    labels_vertices, labels_names = d['labels_vertices'], d['labels_names']

    if do_plot:
        plt.close('all')
        plt.figure()

    res_fol = op.join(LOCAL_ROOT_DIR, 'permutation_ttest_results')
    for cond_id, cond_name in enumerate(events_id.keys()):
        for patient in get_patients():
            blender_fol = blender_template.format(patient=patient, cond_name=cond_name)
            results_file_name = op.join(res_fol, '{}_{}_{}.npz'.format(patient, cond_name, inverse_method))
            clusters_file_name = op.join(res_fol, '{}_{}_{}_clusters.npy'.format(patient, cond_name, inverse_method))
            if op.isfile(results_file_name) and op.isfile(clusters_file_name):
                print('{}, {}'.format(patient, cond_name))
                clusters = np.load(clusters_file_name)
                ttest_res = np.load(results_file_name)
                for data, data_name in zip([ttest_res['T_obs'].T, clusters], ['ttest_res', 'clusters']):
                    fsave_vertices = utils.fsaverage_vertices()
                    stc = _make_stc(data, fsave_vertices, tmin=tmin, tstep=tstep, subject='fsaverage')
                    labels_data = np.zeros((len(labels_vertices), stc.data.shape[1], 2))
                    for ind, (vertidx, label_name) in enumerate(zip(labels_vertices, labels_names)):
                        if vertidx is not None:
                            labels_data[ind] = utils.get_max_min(stc.data[vertidx, :])
                        if do_plot:
                            plt.plot(labels_data[ind, :, 0], label='{} p<c'.format(label_name))
                            plt.plot(labels_data[ind, :, 1], label='{} p>c'.format(label_name))
                    for hemi in ['rh', 'lh']:
                        indices = [ind for ind, l in enumerate(labels_names) if hemi in l]
                        labels = [str(l) for l in labels_names if hemi in l]
                        np.savez(op.join(blender_fol, 'labels_data_{}_{}.npz'.format(hemi, data_name)),
                            data=labels_data[indices], names=labels, conditions=['p<c', 'p>c'])

                    if do_plot:
                        plt.legend()
                        plt.xlabel('time (ms)')
                        plt.title('{} {}'.format(patient, cond_name))
                        plt.show()
                        print('show!')

                # Make clusters to be the default files for blender
                for hemi in ['rh', 'lh']:
                    utils.copy_file(op.join(blender_fol, 'labels_data_{}_clusters.npz'.format(hemi)),
                                    op.join(blender_fol, 'labels_data_{}.npz'.format(hemi)))



# def analyze_clustering_results(events_id, inverse_method='dSPM'):
#     for patient in get_patients():
#         for cond_name in events_id.keys() + ['contrast', 'mean']:
#             results_file_name = op.join(LOCAL_ROOT_DIR, 'clusters_results', '{}_{}_{}.npz'.format(patient, cond_name, inverse_method))
#             if op.isfile(results_file_name):
#                 print(patient, cond_name)
#                 res = np.load(results_file_name)
#                 #T_obs=T_obs, clusters=clusters, cluster_p_values=cluster_p_values, H0=H0
#                 print(len(res['cluster_p_values']), min(res['cluster_p_values']))
#                 good_cluster_inds = np.where(res['cluster_p_values'] < 0.05)[0]
#                 print('good_cluster_inds: {}'.format(good_cluster_inds))


def patients_hcs_conds_gen(events_id, read_clusters=False, inverse_method='dSPM'):
    loops = [enumerate(events_id.keys()), get_patients(), get_healthy_controls()]
    for (cond_id, cond_name), patient, hc in itertools.product(*loops):
        if read_clusters:
            results_file_name = op.join(LOCAL_ROOT_DIR, 'permutation_ttest_results',
                '{}_{}_{}_{}_clusters.npy'.format(patient, hc, cond_name, inverse_method))
        else:
            results_file_name = op.join(LOCAL_ROOT_DIR, 'permutation_ttest_results',
                '{}_{}_{}_{}.npz'.format(patient, hc, cond_name, inverse_method))
        if op.isfile(results_file_name):
            if read_clusters:
                data = np.load(results_file_name)
                yield cond_id, cond_name, patient, hc, data
            else:
                res = utils.dic2bunch(np.load(results_file_name))
                yield cond_id, cond_name, patient, hc, res
    else:
        print('no results for {} {} {}'.format(patient, hc, cond_name))


def create_local_folds():
    for fol in ['eop', 'evo', 'inv', 'stc', 'stc_epochs', 'stc_morphed',
                'stc_epochs_morphed', 'stc_clusters', 'clusters_results',
                'permutation_ttest_results', 'results_for_blender']:
        utils.make_dir(op.join(LOCAL_ROOT_DIR, fol))


if __name__ == '__main__':
    T_MAX = 2
    T_MIN = -0.5
    T_MAX_CROP = 1
    T_MIN_CROP = 0
    events_id = dict(Fear=1, Happy=2)
    inverse_method='dSPM'
    tstep = 0.001
    n_jobs = 6
    pool = Pool(n_jobs) if n_jobs>1 else None

    # src = mne.setup_source_space('fsaverage', surface='pial', spacing='ico5', n_jobs=2)

    print('healthy subjects:')
    print(get_healthy_controls())
    print('patients:')
    print(get_patients())
    # create_local_folds()
    # create_stcs(events_id)
    # calc_all(events_id, T_MIN, T_MAX, apply_for_epochs=True)
    # morph_stcs_to_fsaverage(events_id, stc_per_epoch=True, inverse_method=inverse_method, subjects_dir=SUBJECTS_DIR, n_jobs=n_jobs)
    # calc_average_hc_epochs_stc(events_id)
    # run_permutation_ttest(tmin=T_MIN_CROP, tmax=T_MAX_CROP, p_threshold = 0.005, n_permutations=50, inverse_method=inverse_method, n_jobs=1)
    # filter_out_clusters_from_ttest_results(events_id, p_thresh=0.05, inverse_method=inverse_method)
    # read_permutation_ttest_reusults(events_id, SUBJECTS_DIR, inverse_method)
    plot_perm_ttest_results(events_id, inverse_method, plot_type='pysurfer')
    # smooth_ttest_results(T_MIN_CROP, tstep, SUBJECTS_DIR, n_jobs=1)
    # save_ttest_result_for_blender(events_id, norm_by_percentile=False, norm_percs=(1,99), inverse_method=inverse_method, n_jobs=n_jobs)
    # calc_fsaverage_labels_indices(surf_name='pial', labels_from_annot=False, labels_fol='', parc='aparc250', subjects_dir=LOCAL_SUBJECTS_DIR)
    # calc_labels_avg(events_id, T_MIN_CROP, inverse_method='dSPM', do_plot=False)
    # analyze_clustering_results(events_id, inverse_method='dSPM')


    # misc
    # run_permutation_tests(tmin=T_MIN_CROP, tmax=T_MAX_CROP, calc_per_cond=False, calc_constrast=True, calc_mean=False, n_permutations=10, n_jobs=1)
    # utils.convert_stcs_to_h5(LOCAL_ROOT_DIR, ['stc_epochs', 'stc', 'stc_morphed', 'stc_epochs_morphed'])
