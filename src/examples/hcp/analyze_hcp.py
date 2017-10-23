import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import mne
import hcp
import hcp.preprocessing as preproc

from src.utils import utils
from src.utils import preproc_utils as pu
from src.preproc import meg


links_dir = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(links_dir, 'subjects', 'SUBJECTS_DIR')
MMVT_DIR = utils.get_link_dir(links_dir, 'mmvt')
HCP_DIR = utils.get_link_dir(links_dir, 'hcp')
MEG_DIR = utils.get_link_dir(links_dir, 'meg')

recordings_path = op.join(HCP_DIR, 'hcp-meg')

# subject = '100307'  # our test subject
# meg_sub_fol = op.join(MEG_DIR, subject)
# fwd_fname = op.join(meg_sub_fol, '{}-fwd.fif'.format(subject))
# inv_fname = op.join(meg_sub_fol, '{}-inv.fif'.format(subject))
# noise_cov_fname = op.join(meg_sub_fol, '{}-noise-cov.fif'.format(subject, task))
# stc_fname = op.join(meg_sub_fol, '{}-{}.stc'.format(subject, task))


def collect_events(hcp_params):
    # we first collect events
    trial_infos = list()
    for run_index in [0, 1]:
        trial_info = hcp.read_trial_info(run_index=run_index, **hcp_params)
        trial_infos.append(trial_info)


    # trial_info is a dict
    # it contains a 'comments' vector that maps on the columns of 'codes'
    # 'codes is a matrix with its length corresponding to the number of trials
    print(trial_info['stim']['comments'][:10])  # which column?
    print(set(trial_info['stim']['codes'][:, 3]))  # check values

    # so according to this we need to use the column 7 (index 6)
    # for the time sample and column 4 (index 3) to get the image types
    # with this information we can construct our event vectors

    all_events = list()
    for trial_info in trial_infos:
        events = np.c_[
            trial_info['stim']['codes'][:, 6] - 1,  # time sample
            np.zeros(len(trial_info['stim']['codes'])),
            trial_info['stim']['codes'][:, 3]  # event codes
        ].astype(int)
        events = events[np.argsort(events[:, 0])]  # chronological order
        # for some reason in the HCP data the time events may not always be unique
        unique_subset = np.nonzero(np.r_[1, np.diff(events[:, 0])])[0]
        events = events[unique_subset]  # use diff to find first unique events
        all_events.append(events)

    return all_events


def using_preprocessed_epochs(all_events, eventss, baseline, hcp_params):
    from collections import  defaultdict
    evokeds_from_epochs_hcp = defaultdict(list)
    all_epochs_hcp = list()

    for run_index, events in zip([0, 1], all_events):

        unique_subset = np.nonzero(np.r_[1, np.diff(events[:, 0])])[0]

        epochs_hcp = hcp.read_epochs(run_index=run_index, **hcp_params)
        # subset epochs, add events and id
        subset = np.in1d(events[:, 2], list(eventss.values()))

        epochs_hcp = epochs_hcp[unique_subset][subset]
        epochs_hcp.events[:, 2] = events[subset, 2]
        epochs_hcp.event_id = eventss
        break
        # all_epochs_hcp.append(epochs_hcp)

    for event_name, event_id in eventss.items():
        evoked = epochs_hcp[event_name].average()
        evoked.baseline = baseline
        evoked.apply_baseline()
        evoked = preproc.interpolate_missing(evoked, **hcp_params)
        evokeds_from_epochs_hcp[event_name].append(evoked)

    return epochs_hcp, evokeds_from_epochs_hcp


def analyze_task(subject, args, hcp_params):
    flags = {}
    events = dict(face=1, tools=2)
    baseline = (-0.5, 0)

    all_events = collect_events(hcp_params)
    files_exist = all([op.isfile(meg.get_cond_fname(meg.EPO, event)) and
                       op.isfile(meg.get_cond_fname(meg.EVO, event)) for event in events.keys()])
    if not files_exist:
        epochs_hcp, evokeds_from_epochs_hcp = using_preprocessed_epochs(all_events, events, baseline, hcp_params)
        for event in events.keys():
            epochs_hcp[event].save(meg.get_cond_fname(meg.EPO, event))
            mne.write_evokeds(meg.get_cond_fname(meg.EVO, event), evokeds_from_epochs_hcp[event])

    flags = meg.calc_fwd_inv_wrapper(subject, args, events, flags)
    flags, stcs_conds, _ = meg.calc_stc_per_condition_wrapper(subject, events, args.inverse_method, args, flags)
    flags = meg.calc_labels_avg_per_condition_wrapper(subject, events, args.atlas, args.inverse_method, stcs_conds, args, flags)


def analyze_rest(subject, args, hcp_params, run_index=0):
    flags = {}
    raw = None
    # raw = hcp.read_raw(run_index=run_index, **hcp_params)
    # annots = hcp.read_annot(run_index=run_index, **hcp_params)
    # epochs = hcp.read_epochs(run_index=run_index, **hcp_params)
    meg.calc_fwd_inv_wrapper(subject, args)
    args.snr = 1.0  # use smaller SNR for raw data
    # args.n_jobs = 1
    meg.calc_labels_avg_for_rest_wrapper(args, raw)
    print('sdf')


def compute_noise_cov(subject, hcp_path, noise_cov_fname=''):
    if noise_cov_fname == '':
        noise_cov_fname = meg.NOISE_COV.format(cond='empty_room')
    if op.isfile(noise_cov_fname):
        noise_cov = mne.read_cov(noise_cov_fname)
        return noise_cov
    utils.make_dir(utils.get_parent_fol(noise_cov_fname))

    raw_noise = hcp.read_raw(subject=subject, hcp_path=hcp_path, data_type='noise_empty_room')
    raw_noise.load_data()
    # apply ref channel correction and drop ref channels
    preproc.apply_ref_correction(raw_noise)
    raw_noise.filter(0.50, None, method='iir',
                     iir_params=dict(order=4, ftype='butter'), n_jobs=1)
    raw_noise.filter(None, 60, method='iir',
                     iir_params=dict(order=4, ftype='butter'), n_jobs=1)
    ##############################################################################
    # Note that using the empty room noise covariance will inflate the SNR of the
    # evkoked and renders comparisons  to `baseline` rather uninformative.
    noise_cov = mne.compute_raw_covariance(raw_noise, method='empirical')
    noise_cov.save(noise_cov_fname)
    return noise_cov


def init(subject, task):
    args = pu.init_args(meg.read_cmd_args(dict(
        subject=subject,
        atlas='laus125',
        task=task,
        files_includes_cond=True,
        inverse_method='MNE')))

    fname_format_cond = '{subject}_hcp_{cond}-{ana_type}.{file_type}'
    fname_format = '{subject}_hcp-{ana_type}.{file_type}'
    meg.init_globals_args(
        subject, '', fname_format, fname_format_cond, MEG_DIR, SUBJECTS_DIR, MMVT_DIR, args)

    hcp_params = dict(hcp_path=HCP_DIR, subject=subject, data_type=task)
    return args, hcp_params

if __name__ == '__main__':


    # https://db.humanconnectome.org/
    subject = '100307'
    # task = 'task_working_memory'
    task = 'rest'
    args, hcp_params = init(subject, task)
    # analyze_task(subject, args, hcp_params)
    analyze_rest(subject, args, hcp_params)
    # test_apply_ica(dict(hcp_path=HCP_DIR, subject=subject, data_type='rest'))
