import numpy as np
import mne
import os.path as op
from src.preproc import meg_preproc
from src.preproc import anatomy_preproc
from src import utils
import glob
import shutil
import traceback

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
        if len(event_indices) == 0:
            print('No events for {}!'.format(event_name))
            return None
    return indices


def calc_evoked(indices, epochs_fname, overwrite_epochs=False, overwrite_evoked=False):
    epochs = mne.read_epochs(epochs_fname, preload=False)
    print(epochs.events.shape)
    for event_name, event_indices in indices.items():
        evoked_event_fname = meg_preproc.get_cond_fname(meg_preproc.EVO, event_name)
        epochs_event_fname = meg_preproc.get_cond_fname(meg_preproc.EPO, event_name)
        if not op.isfile(epochs_event_fname) or overwrite_epochs:
            print('Saving {} epochs to {}, events num: {}'.format(event_name, epochs_event_fname, len(event_indices)))
            event_epochs = epochs[event_indices]
            event_epochs.save(epochs_event_fname)
        if not op.isfile(evoked_event_fname) or overwrite_evoked:
            print('Saving {} evoked to {}'.format(event_name, evoked_event_fname))
            mne.write_evokeds(evoked_event_fname, event_epochs.average())


def average_all_evoked_responses(root_fol, moving_average_win_size=50):
    import matplotlib.pyplot as plt
    for hemi in utils.HEMIS:
        evoked_files = glob.glob(op.join(root_fol, '*labels_data_{}.npz'.format(hemi)))
        all_data = None
        all_data_win = None
        for evoked_ind, evoked_fname in enumerate(evoked_files):
            f = np.load(evoked_fname)
            data = f['data'] # labels x time x conditions
            if all_data is None:
                all_data = np.zeros((*data.shape, len(evoked_files)))
            for cond_ind in range(data.shape[2]):
                for label_ind in range(data.shape[0]):
                    x = data[label_ind, :, cond_ind]
                    x *= np.sign(x[np.argmax(np.abs(x))])
                    all_data[label_ind, :, cond_ind, evoked_ind] = x
                win_avg = utils.moving_avg(all_data[:, :, cond_ind, evoked_ind], moving_average_win_size)
                if all_data_win is None:
                    all_data_win = np.zeros((*win_avg.shape, data.shape[2], len(evoked_files)))
                all_data_win[:, :, cond_ind, evoked_ind] = win_avg
        mean_evoked = np.mean(all_data, 3)
        mean_win_evoked = np.mean(all_data_win, 3)
        np.savez(op.join(root_fol, 'avg_labels_data.npz'), data=mean_evoked, names=f['names'], conditions=f['conditions'])
        np.savez(op.join(root_fol, 'avg_labels_data_win_{}.npz'.format(moving_average_win_size)),
                 data=mean_win_evoked, names=f['names'], conditions=f['conditions'])
        plt.figure()
        for label_data in mean_win_evoked:
            plt.plot(label_data)
        plt.savefig(op.join(root_fol, '{}_avg_evoked_win_{}.jpg'.format(hemi, moving_average_win_size)))


def plot_evoked(indices):
    for event_name in indices.keys():
        evoked_fname = meg_preproc.get_cond_fname(meg_preproc.EVO, event_name)
        print('Reading {}'.format(event_name))
        evoked = mne.read_evokeds(evoked_fname)[0]
        evoked.plot()


def create_evoked_responses(root_fol, task, atlas, events_id, fname_format, fwd_fol, neccesary_files,
            remote_subjects_dir, fsaverage, raw_cleaning_method, inverse_method,
            overwrite_epochs=False, overwrite_evoked=False):
    errors = []
    hc_subjects_epo_filess = glob.glob(op.join(root_fol, 'hc*arc*epo.fif'))
    for subject_epo_fname in hc_subjects_epo_filess:
        try:
            subject = utils.namebase(subject_epo_fname).split('_')[0]
            calc_subject_evoked_response(subject, root_fol, task, atlas, events_id, fname_format, fwd_fol, neccesary_files,
                remote_subjects_dir, fsaverage, raw_cleaning_method, inverse_method,
                overwrite_epochs, overwrite_evoked)
        except:
            print('******* Error with {} *******'.format(subject))
            print(traceback.format_exc())
            errors.append(subject)

    for subject_err in errors:
        print('Error with {}'.format(subject_err))


def calc_subject_evoked_response(subject, root_fol, task, atlas, events_id, fname_format, fwd_fol, neccesary_files,
            remote_subjects_dir, fsaverage, raw_cleaning_method, inverse_method, indices=None,
            overwrite_epochs=False, overwrite_evoked=False):
    meg_preproc.init_globals(subject, fname_format=fname_format, raw_cleaning_method=raw_cleaning_method,
                             subjects_meg_dir=SUBJECTS_MEG_DIR, task=task, subjects_mri_dir=SUBJECTS_DIR,
                             BLENDER_ROOT_DIR=BLENDER_ROOT_DIR, files_includes_cond=True, fwd_no_cond=True)
    epochs_fname = '{}_arc_rer_{}-epo.fif'.format(subject, raw_cleaning_method)
    events_fname = '{}_arc_rer_{}-epo.csv'.format(subject, raw_cleaning_method)
    if indices is None:
        indices = find_events_indices(op.join(root_fol, events_fname))
    if not indices is None:
        utils.make_dir(op.join(SUBJECTS_MEG_DIR, task, subject))
        utils.make_dir(op.join(SUBJECTS_DIR, subject, 'mmvt'))
        utils.make_dir(op.join(BLENDER_ROOT_DIR, subject))
        # utils.prepare_local_subjects_folder(
        #     neccesary_files, subject, remote_subjects_dir, SUBJECTS_DIR, print_traceback=False)
        # anatomy_preproc.freesurfer_surface_to_blender_surface(subject, overwrite=False)
        # anatomy_preproc.create_annotation_file_from_fsaverage(subject, atlas, fsaverage, False, False, False, True)
        # calc_evoked(indices, op.join(root_fol, epochs_fname), overwrite_epochs, overwrite_evoked)
        fwd_fname = '{}_arc_rer_tsss-fwd.fif'.format(subject)
        if not op.isfile(op.join(SUBJECTS_MEG_DIR, task, subject, fwd_fname)):
            shutil.copy(op.join(fwd_fol, fwd_fname), op.join(SUBJECTS_MEG_DIR, task, subject, fwd_fname))
        meg_preproc.calc_inverse_operator(events_id, calc_for_cortical_fwd=True, calc_for_sub_cortical_fwd=False)
        stcs = meg_preproc.calc_stc_per_condition(events_id, inverse_method)
        for hemi in utils.HEMIS:
            meg_preproc.calc_labels_avg_per_condition(
                atlas, hemi, 'pial', events_id, labels_from_annot=False, labels_fol='', stcs=stcs,
                inverse_method=inverse_method, do_plot=False)


def copy_evokes(task, root_fol, target_subject, raw_cleaning_method):
    hc_subjects_epo_filess = glob.glob(op.join(root_fol, 'hc*arc*epo.fif'))
    for subject_epo_fname in hc_subjects_epo_filess:
        subject = utils.namebase(subject_epo_fname).split('_')[0]
        events_fname = '{}_arc_rer_{}-epo.csv'.format(subject, raw_cleaning_method)
        indices = find_events_indices(op.join(root_fol, events_fname))
        if not indices is None:
            for hemi in utils.HEMIS:
                shutil.copy(op.join(SUBJECTS_MEG_DIR, task, subject, 'labels_data_{}.npz'.format(hemi)),
                            op.join(BLENDER_ROOT_DIR, target_subject, 'meg_evoked_files', '{}_labels_data_{}.npz'.format(subject, hemi)))


if __name__ == '__main__':
    target_subject = 'pp009'
    raw_cleaning_method = 'tsss'
    task = 'ARC'
    atlas = 'arc_april2016'
    fsaverage = 'fscopy'
    inverse_method = 'dSPM'
    # fname_format = '{subject}_arc_rer_{raw_cleaning_method}_{cond}-{ana_type}.{file_type}'
    fname_format, events_id, event_digit = meg_preproc.get_fname_format(task)
    root_fol = '/autofs/space/sophia_002/users/DARPA-MEG/arc/ave/'
    fwd_fol = '/autofs/space/sophia_002/users/DARPA-MEG/arc/fwd/'
    remote_subjects_dir = '/autofs/space/lilli_001/users/DARPA-Recons'
    neccesary_files = {'..': ['sub_cortical_codes.txt'], 'mri': ['aseg.mgz', 'norm.mgz', 'ribbon.mgz'],
        'surf': ['rh.pial', 'lh.pial', 'rh.sphere.reg', 'lh.sphere.reg', 'lh.white', 'rh.white']}
    overwrite_epochs = True
    overwrite_evoked = True
    # # root_fol = op.join(SUBJECTS_MEG_DIR, task, subject)

    create_evoked_responses(root_fol, task, atlas, events_id, fname_format,
        fwd_fol, neccesary_files, remote_subjects_dir, fsaverage, raw_cleaning_method, inverse_method,
        overwrite_epochs, overwrite_evoked)
    copy_evokes(task, root_fol, target_subject, raw_cleaning_method)
    # average_all_evoked_responses(op.join(BLENDER_ROOT_DIR, target_subject, 'meg_evoked_files'))

    # calc_subject_evoked_response('hc022', root_fol, task, atlas, events_id, fname_format, fwd_fol, neccesary_files,
    #         remote_subjects_dir, fsaverage, raw_cleaning_method, inverse_method, indices=[],
    #         overwrite_epochs=False, overwrite_evoked=False)