import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import mne
import hcp
import hcp.preprocessing as preproc

from src.utils import utils
from src.preproc import meg


links_dir = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(links_dir, 'subjects', 'SUBJECTS_DIR')
MMVT_DIR = utils.get_link_dir(links_dir, 'mmvt')
HCP_DIR = utils.get_link_dir(links_dir, 'hcp')
MEG_DIR = utils.get_link_dir(links_dir, 'meg')

recordings_path = op.join(HCP_DIR, 'hcp-meg')
subject = '100307'  # our test subject
task = 'task_working_memory'
meg_sub_fol = op.join(MEG_DIR, subject)
fwd_fname = op.join(meg_sub_fol, '{}-fwd.fif'.format(subject))
inv_fname = op.join(meg_sub_fol, '{}-inv.fif'.format(subject))
noise_cov_fname = op.join(meg_sub_fol, '{}-noise-cov.fif'.format(subject, task))
stc_fname = op.join(meg_sub_fol, '{}-{}.stc'.format(subject, task))

hcp_params = dict(hcp_path=HCP_DIR, subject=subject, data_type=task)

args = meg.read_cmd_args(dict(
    subject=subject,
    task='task',
    conditions='face',
    # atlas='laus250',
    inverse_method='MNE',
    t_min=-1.5, t_max=2.5,
    reject=False,
    overwrite_epochs=False))
fname_format, fname_format_cond, conditions = meg.init(subject, args)
print('epo: {}, evo: {}'.format(meg.EPO, meg.EVO))

##############################################################################
# We first reprocess the data from scratch
#
# That is, almost from scratch. We're relying on the ICA solutions and
# data annotations.
#
# In order to arrive at the final ERF we need to pool over two runs.
# for each run we need to read the raw data, all annotations, apply
# the reference sensor compensation, the ICA, bandpass filter, baseline
# correction and decimation (downsampling)

# these values are looked up from the HCP manual


def collect_events():
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


def reprocess_the_data_from_scratch(all_events, event_id, tmin, tmax, baseline, decim):
    # now we can go ahead
    evokeds = list()
    all_epochs = list()
    for run_index, events in zip([0, 1], all_events):

        raw = hcp.read_raw(run_index=run_index, **hcp_params)
        raw.load_data()
        # apply ref channel correction and drop ref channels
        # preproc.apply_ref_correction(raw)

        annots = hcp.read_annot(run_index=run_index, **hcp_params)
        # construct MNE annotations
        bad_seg = (annots['segments']['all']) / raw.info['sfreq']
        annotations = mne.Annotations(
            bad_seg[:, 0], (bad_seg[:, 1] - bad_seg[:, 0]),
            description='bad')

        raw.annotations = annotations
        raw.info['bads'].extend(annots['channels']['all'])
        raw.pick_types(meg=True, ref_meg=False)

        #  Note: MNE complains on Python 2.7
        raw.filter(0.50, None, method='iir',
                   iir_params=dict(order=4, ftype='butter'), n_jobs=1)
        raw.filter(None, 60, method='iir',
                   iir_params=dict(order=4, ftype='butter'), n_jobs=1)

        # read ICA and remove EOG ECG
        # note that the HCP ICA assumes that bad channels have already been removed
        ica_mat = hcp.read_ica(run_index=run_index, **hcp_params)

        # We will select the brain ICs only
        exclude = annots['ica']['ecg_eog_ic']
        preproc.apply_ica_hcp(raw, ica_mat=ica_mat, exclude=exclude)

        # now we can epoch
        events = np.sort(events, 0)
        epochs = mne.Epochs(raw, events=events[events[:, 2] == 1],
                            event_id=event_id, tmin=tmin, tmax=tmax,
                            reject=None, baseline=baseline, decim=decim,
                            preload=True)

        all_epochs.append(epochs)
        evoked = epochs.average()
        # now we need to add back out channels for comparison across runs.
        evoked = preproc.interpolate_missing(evoked, **hcp_params)
        evokeds.append(evoked)
    return all_epochs, evokeds


##############################################################################
# Now we can compute the same ERF based on the preprocessed epochs
#
# These are obtained from the 'tmegpreproc' pipeline.
# Things are pythonized and simplified however, so
def using_preprocessed_epochs():
    evokeds_from_epochs_hcp = list()
    all_epochs_hcp = list()

    for run_index, events in zip([0, 1], all_events):

        unique_subset = np.nonzero(np.r_[1, np.diff(events[:, 0])])[0]
        # use diff to find first unique events
        this_events = events[unique_subset]
        subset = np.in1d(events[:, 2], event_id.values())

        epochs_hcp = hcp.read_epochs(run_index=run_index, **hcp_params)

        # subset epochs, add events and id
        epochs_hcp = epochs_hcp[unique_subset][subset]
        epochs_hcp.events[:, 2] = events[subset, 2]
        epochs_hcp.event_id = event_id
        all_epochs_hcp.append(epochs_hcp)

        evoked = epochs_hcp['face'].average()

        del epochs_hcp
        # These epochs have different channels.
        # We use a designated function to re-apply the channels and interpolate
        # them.

        evoked.baseline = baseline
        evoked.apply_baseline()
        evoked = preproc.interpolate_missing(evoked, **hcp_params)

        evokeds_from_epochs_hcp.append(evoked)

    return all_epochs_hcp, evokeds_from_epochs_hcp


##############################################################################
# Finally we can read the actual official ERF file
#
# These are obtained from the 'eravg' pipelines.
# We read the matlab file, MNE-HCP is doing some conversions, and then we
# search our condition of interest. Here we're looking at the image as onset.
# and we want the average, not the standard deviation.

def official_ERF():
    hcp_evokeds = hcp.read_evokeds(onset='stim', **hcp_params)

    for ev in hcp_evokeds:
        if not ev.comment == 'Wrkmem_LM-TIM-face_BT-diff_MODE-mag':
            continue

    # Once more we add and interpolate missing channels
    evoked_hcp = preproc.interpolate_missing(ev, **hcp_params)
    return evoked_hcp


##############################################################################
# Time to compare the outputs
#

def compare_the_outputs(evokeds_from_epochs_hcp, evoked_hcp):
    evoked = mne.combine_evoked(evokeds, weights='equal')
    evoked_from_epochs_hcp = mne.combine_evoked(
        evokeds_from_epochs_hcp, weights='equal')

    fig1, axes = plt.subplots(3, 1, figsize=(12, 8))

    evoked.plot(axes=axes[0], show=False)
    axes[0].set_title('MNE-HCP')

    evoked_from_epochs_hcp.plot(axes=axes[1], show=False)
    axes[1].set_title('HCP epochs')

    evoked_hcp.plot(axes=axes[2], show=False)
    axes[2].set_title('HCP evoked')
    fig1.canvas.draw()

    plt.show()

    # now some correlations

    plt.figure()
    r1 = np.corrcoef(evoked_from_epochs_hcp.data.ravel(),
                     evoked_hcp.data.ravel())[0][1]
    plt.plot(evoked_from_epochs_hcp.data.ravel()[::10] * 1e15,
             evoked_hcp.data.ravel()[::10] * 1e15,
             linestyle='None', marker='o', alpha=0.1,
             mec='orange', color='orange')
    plt.annotate("r=%0.3f" % r1, xy=(-300, 250))
    plt.ylabel('evoked from HCP epochs')
    plt.xlabel('evoked from HCP evoked')
    plt.show()

    plt.figure()
    r1 = np.corrcoef(evoked.data.ravel(), evoked_hcp.data.ravel())[0][1]
    plt.plot(evoked.data.ravel()[::10] * 1e15,
             evoked_hcp.data.ravel()[::10] * 1e15,
             linestyle='None', marker='o', alpha=0.1,
             mec='orange', color='orange')
    plt.annotate("r=%0.3f" % r1, xy=(-300, 250))
    plt.ylabel('evoked from scratch with MNE-HCP')
    plt.xlabel('evoked from HCP evoked file')
    plt.show()


if __name__ == '__main__':
    tmin, tmax = -1.5, 2.5
    decim = 4
    event_id = dict(face=1)
    baseline = (-0.5, 0)

    all_events = collect_events()
    # all_epochs, evokeds = reprocess_the_data_from_scratch(all_events, event_id, tmin, tmax, baseline, decim)
    # all_epochs_hcp, evokeds_from_epochs_hcp = using_preprocessed_epochs
    evoked_hcp = official_ERF()
    print('Finish!')
    # epochs.save(meg.EPO)
    mne.write_evokeds(meg.EVO, evoked_hcp)
