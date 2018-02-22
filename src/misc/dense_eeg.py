import sys
import os.path as op
import glob
import shutil
from functools import partial

from src.preproc import eeg
from src.preproc import meg
from src.utils import utils
from src.mmvt_addon.scripts import import_sensors


LINKS_DIR = utils.get_links_dir()
EEG_DIR = utils.get_link_dir(LINKS_DIR, 'eeg')


def init_flags(subject, task, conditions, remote_subject_dir=''):
    args = eeg.read_cmd_args(subject=subject)
    args.task = task
    args.sub_dirs_for_tasks = True
    eeg.init(subject, args)
    args.remote_subject_dir = remote_subject_dir
    args.conditions = list(conditions.values())
    args.t_min, args.t_max = -0.2, 0.2
    args.l_freq, args.h_freq = 1, 40
    args.inverse_method = ['MNE', 'sLORETA']
    args.inv_loose = 0.6
    args.calc_max_min_diff = False
    args.calc_evoked_for_all_epoches = args.calc_stc_for_all = False
    args.calc_stc_diff = False
    args.overwrite_epochs, args.overwrite_evoked, args.overwrite_sensors, args.overwrite_stc = False, True, False, False
    args.normalize_data = False
    args.use_raw_for_noise_cov = True
    return args


def set_args(task, conditions, raw_file, eve_fname):
    args = init_flags(subject, task, conditions, remote_subject_dir)
    fol = utils.make_dir(op.join(EEG_DIR, subject, args.task))
    raw_file, eve_fname = copy_files(fol, raw_file, eve_fname)
    args.raw_template = args.raw_fname = raw_file  # todo: why both?
    args.eve_template = eve_fname
    args.stc_template = op.join(
        meg.SUBJECT_MEG_FOLDER, '{}_{}_{}-{}.stc'.format(subject, task, '{cond}', '{method}'))
    return args


def copy_files(fol, raw_file, eve_fname):
    if not op.isfile(op.join(fol, utils.namebase_with_ext(raw_file))):
        shutil.copy(raw_file, fol)
    if not op.isfile(op.join(fol, utils.namebase_with_ext(eve_fname))):
        shutil.copy(eve_fname, fol)
    raw_file = op.join(fol, utils.namebase_with_ext(raw_file))
    eve_fname = op.join(fol, utils.namebase_with_ext(eve_fname))
    return raw_file, eve_fname


def main(eeg_fol, subject, remote_subject_dir=''):
    raw_files = glob.glob(op.join(eeg_fol, '*_raw.fif'))
    conditions = {'spikes1': 1001, 'spikes2': 1002}
    first_time = True
    for raw_file in raw_files:
        eve_fname = f'{raw_file[:-4]}-annot.fif'
        if not op.isfile(eve_fname):
            print(f'No annot file for {raw_file}!')
            continue
        task = '_'.join(utils.namebase(raw_file).split('_')[1:-1])
        args = set_args(task, conditions, raw_file, eve_fname)
        if first_time:
            eeg.read_eeg_sensors_layout(subject, args)
            # import_sensors.wrap_blender_call(subject, 'eeg', load_data=False, overwrite_sensors=args.overwrite_sensors)
            first_time = False
        eeg.calc_evokes(subject, conditions, args)
        eeg.calc_fwd_inv(subject, args, conditions)
        for im in args.inverse_method:
            eeg.calc_stc_per_condition(subject, conditions, im, args)


if __name__ == '__main__':
    eeg_fol = '/cluster/neuromind/npeled/MGH_Highdensity_source/EEG'
    subject = 'mgh128'
    remote_subject_dir = '/cluster/neuromind/npeled/MGH_Highdensity_source/MGH128'
    # eeg_fol = op.join(EEG_DIR, subject)
    main(eeg_fol, subject, remote_subject_dir)