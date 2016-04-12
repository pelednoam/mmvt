import numpy as np
import mne
import os.path as op
from src.preproc import meg_preproc
from src import utils

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
FREE_SURFER_HOME = utils.get_link_dir(LINKS_DIR, 'freesurfer', 'FREESURFER_HOME')
SUBJECTS_MEG_DIR = op.join(LINKS_DIR, 'meg')
BLENDER_ROOT_DIR = op.join(LINKS_DIR, 'mmvt')


def find_events_indices(events_fname):
    data = np.genfromtxt(events_fname, dtype=np.int, delimiter=',', skip_header=1, usecols=(0, 2, 6), names=None, autostrip=True)
    data[:, 0] = range(len(data))
    data = data[data[:, -1] == 1]
    indices = {}
    event_names = {1:'low_risk', 2:'med_risk', 3:'high_risk'}
    for risk, event_name in event_names.items():
        event_indices = data[data[:, 1] == risk, 0]
        indices[event_name] = event_indices
    return indices


def calc_evoked(indices, epochs_fname):
    epochs = mne.read_epochs(epochs_fname, preload=False)
    print(epochs.events.shape)
    for event_name, event_indices in indices.items():
        evoked_fname = meg_preproc.get_cond_fname(meg_preproc.EVO, event_name)
        print('Saving {} evoked to {}'.format(event_name, evoked_fname))
        mne.write_evokeds(evoked_fname, epochs[event_indices].average())


def plot_evoked(indices):
    for event_name in indices.keys():
        evoked_fname = meg_preproc.get_cond_fname(meg_preproc.EVO, event_name)
        print('Reading {}'.format(event_name))
        evoked = mne.read_evokeds(evoked_fname)[0]
        evoked.plot()


if __name__ == '__main__':
    subject = 'pp009'
    raw_cleaning_method = 'tsss'
    task = 'ARC'
    fname_format = '{subject}_arc_rer_{raw_cleaning_method}_{cond}-{ana_type}.{file_type}'
    epochs_fname = 'pp009_arc_rer_tsss-epo.fif'
    events_fname = 'pp009_arc_rer_tsss-epo.csv'
    root_fol = '/autofs/space/sophia_002/users/DARPA-MEG/arc/ave/'
    root_fol = op.join(SUBJECTS_MEG_DIR, task, subject)
    meg_preproc.init_globals(subject, fname_format=fname_format, raw_cleaning_method=raw_cleaning_method,
                             subjects_meg_dir=SUBJECTS_MEG_DIR, task=task, subjects_mri_dir=SUBJECTS_DIR,
                             BLENDER_ROOT_DIR=BLENDER_ROOT_DIR, files_includes_cond=True)
    print('evoked fname: {}'.format(meg_preproc.EVO))
    indices = find_events_indices(op.join(root_fol, events_fname))
    # calc_evoked(indices, op.join(root_fol, epochs_fname))
    plot_evoked(indices)