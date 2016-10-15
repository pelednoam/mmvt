import os
import os.path as op
import numpy as np
import mne.io
import glob

from src.utils import utils
from src.preproc import meg_preproc as meg

LINKS_DIR = utils.get_links_dir()
SUBJECTS_MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
SUBJECTS_EEG_DIR = utils.get_link_dir(LINKS_DIR, 'eeg')
if SUBJECTS_EEG_DIR == '':
    print('No EEG folder, using MEG folder')
    SUBJECTS_EEG_DIR = SUBJECTS_MEG_DIR
if SUBJECTS_EEG_DIR == '':
    raise Exception('No EEG folder (not MEG)!')
SUBJECT_EEG_DIR = ''
SUBJECTS_MRI_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
FREESURFER_HOME = utils.get_link_dir(LINKS_DIR, 'freesurfer', 'FREESURFER_HOME')
print('FREE_SURFER_HOME: {}'.format(FREESURFER_HOME))
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')
os.environ['SUBJECTS_DIR'] = SUBJECTS_MRI_DIR


def read_eeg_sensors_layout(mri_subject):
    if not op.isfile(meg.INFO):
        raw = mne.io.read_raw_fif(meg.RAW)
        info = raw.info
        utils.save(info, meg.INFO)
    else:
        info = utils.load(meg.INFO)
    eeg_picks = mne.io.pick.pick_types(info, meg=False, eeg=True)
    eeg_pos = np.array([info['chs'][k]['loc'][:3] for k in eeg_picks])
    eeg_names = np.array([info['ch_names'][k] for k in eeg_picks])
    fol = op.join(MMVT_DIR, mri_subject, 'eeg')
    utils.make_dir(fol)
    output_fname = op.join(fol, 'eeg_positions.npz')
    if len(eeg_pos) > 0:
        trans_files = glob.glob(op.join(SUBJECTS_MRI_DIR, '*COR*.fif'))
        if len(trans_files) == 1:
            trans = mne.transforms.read_trans(trans_files[0])
            head_mri_t = mne.transforms._ensure_trans(trans, 'head', 'mri')
            eeg_pos = mne.transforms.apply_trans(head_mri_t, eeg_pos)
            eeg_pos *= 1000
            np.savez(output_fname, pos=eeg_pos, names=eeg_names)
            return True
    return False


def save_evoked_to_blender(mri_subject, events, evoked=None, norm_by_percentile=True, norm_percs=(1,99)):
    fol = op.join(MMVT_DIR, mri_subject, 'eeg')
    utils.make_dir(fol)
    if '{cond}' in meg.EVO:
        for event_ind, event_id in enumerate(events.keys()):
            if evoked is None:
                evo = mne.read_evokeds(meg.get_cond_fname(meg.EVO, event_id))
            else:
                evo = evoked[event_id]
            if event_ind == 0:
                ch_names = evo[0].ch_names
                data = np.zeros((evo[0].data.shape[0], evo[0].data.shape[1], 2))
            data[:, :, event_ind] = evo[0].data
    else:
        if evoked is None:
            evoked = mne.read_evokeds(meg.EVO)
        data = evoked.data
    data_max, data_min = utils.get_data_max_min(data, norm_by_percentile, norm_percs)
    max_abs = utils.get_max_abs(data_max, data_min)
    data = data / max_abs
    np.save(op.join(fol, 'eeg_data.npy'), data)
    np.savez(op.join(fol, 'eeg_data_meta.npz'), names=ch_names, conditions=list(events.keys()))
    return True


def main(subject, mri_subject, inverse_method, args):

    fname_format, fname_format_cond, conditions = meg.get_fname_format_args(args)
    meg.init_globals_args(subject, mri_subject, fname_format, fname_format_cond, SUBJECTS_EEG_DIR, SUBJECTS_MRI_DIR,
                     MMVT_DIR, args)
    meg.SUBJECTS_MEG_DIR = SUBJECTS_EEG_DIR
    stat = meg.STAT_AVG if len(conditions) == 1 else meg.STAT_DIFF
    evoked, epochs = None, None
    flags = {}

    if utils.should_run(args, 'read_eeg_sensors_layout'):
        flags['read_eeg_sensors_layout'] = read_eeg_sensors_layout(mri_subject)

    if utils.should_run(args, 'calc_evoked'):
        necessary_files = meg.calc_evoked_necessary_files(args)
        meg.get_meg_files(subject, necessary_files, args, conditions)
        flags['calc_evoked'], evoked, epochs = meg.calc_evoked_args(conditions, args)

    if utils.should_run(args, 'save_evoked_to_blender'):
        flags['save_evoked_to_blender'] = save_evoked_to_blender(mri_subject, conditions, evoked)

    return flags


if __name__ == '__main__':
    args = meg.read_cmd_args()
    args.pick_meg = False
    args.pick_eeg = True
    args.reject = False
    meg.run_on_subjects(args, locals()['main'])
    print('finish!')
