import os
import os.path as op
import time
import matplotlib.pyplot as plt
import glob
import shutil
import numpy as np
from functools import partial
from collections import defaultdict, Iterable
from itertools import product
import traceback
import mne
import types
import scipy.io as sio
import nibabel as nib

from mne.minimum_norm import (make_inverse_operator, apply_inverse,
                              write_inverse_operator, read_inverse_operator)
from mne.preprocessing import ICA
from mne.preprocessing import create_ecg_epochs, create_eog_epochs

from src.utils import utils
from src.utils import preproc_utils as pu
from src.utils import labels_utils as lu
from src.utils import args_utils as au
from src.utils import freesurfer_utils as fu
from src.preproc import anatomy as anat

SUBJECTS_MRI_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()

LINKS_DIR = utils.get_links_dir()
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
LOOKUP_TABLE_SUBCORTICAL = op.join(MMVT_DIR, 'sub_cortical_codes.txt')

STAT_AVG, STAT_DIFF = range(2)
HEMIS = ['rh', 'lh']

SUBJECT, MRI_SUBJECT, SUBJECT_MEG_FOLDER, RAW, RAW_ICA, INFO, EVO, EVE, COV, EPO, EPO_NOISE, FWD, FWD_EEG, FWD_SUB, FWD_X,\
FWD_SMOOTH, INV, INV_EEG, INV_SMOOTH, INV_EEG_SMOOTH, INV_SUB, INV_X, EMPTY_ROOM, MRI, SRC, SRC_SMOOTH, BEM, STC, \
STC_HEMI, STC_HEMI_SAVE, STC_HEMI_SMOOTH, STC_HEMI_SMOOTH_SAVE, STC_ST,\
COR, LBL, STC_MORPH, ACT, ASEG, DATA_COV, NOISE_COV, DATA_CSD, NOISE_CSD, MEG_TO_HEAD_TRANS = [''] * 43

locating_meg_file = None
locating_subject_file = None


def init_globals_args(subject, mri_subject, fname_format, fname_format_cond, subjects_meg_dir, subjects_mri_dir,
                      mmvt_dir, args):
    return init_globals(subject, mri_subject, fname_format, fname_format_cond, args.raw_fname_format,
                 args.fwd_fname_format, args.inv_fname_format, args.events_file_name, args.files_includes_cond,
                 args.cleaning_method, args.contrast, subjects_meg_dir, args.task, subjects_mri_dir, mmvt_dir,
                 args.fwd_no_cond, args.inv_no_cond, args.data_per_task, args.sub_dirs_for_tasks)


def init_globals(subject, mri_subject='', fname_format='', fname_format_cond='', raw_fname_format='',
                 fwd_fname_format='', inv_fname_format='', events_fname='', files_includes_cond=False,
                 cleaning_method='', contrast='', subjects_meg_dir='', task='', subjects_mri_dir='', mmvt_dir='',
                 fwd_no_cond=False, inv_no_cond=False, data_per_task=False, sub_dirs_for_tasks=False):
    global SUBJECT, MRI_SUBJECT, SUBJECT_MEG_FOLDER, RAW, RAW_ICA, INFO, EVO, EVE, COV, EPO, EPO_NOISE, FWD, FWD_EEG, FWD_SUB,\
        FWD_X, FWD_SMOOTH, INV, INV_EEG, INV_SMOOTH, INV_EEG_SMOOTH, INV_SUB, INV_X, EMPTY_ROOM, MRI, SRC, SRC_SMOOTH,\
        BEM, STC, STC_HEMI, STC_HEMI_SAVE, STC_HEMI_SMOOTH, STC_HEMI_SMOOTH_SAVE, STC_ST, COR, AVE, LBL, STC_MORPH,\
        ACT, ASEG, MMVT_SUBJECT_FOLDER, DATA_COV, NOISE_COV, DATA_CSD, NOISE_CSD, MEG_TO_HEAD_TRANS, \
        locating_meg_file, locating_subject_file
    if files_includes_cond:
        fname_format = fname_format_cond
    SUBJECT = subject
    MRI_SUBJECT = mri_subject if mri_subject!='' else subject
    os.environ['SUBJECT'] = SUBJECT
    if task != '':
        if data_per_task:
            SUBJECT_MEG_FOLDER = op.join(subjects_meg_dir, task, SUBJECT)
        elif sub_dirs_for_tasks:
            SUBJECT_MEG_FOLDER = op.join(subjects_meg_dir, SUBJECT, task)
        else:
            SUBJECT_MEG_FOLDER = op.join(subjects_meg_dir, SUBJECT)
    else:
        SUBJECT_MEG_FOLDER = op.join(subjects_meg_dir, SUBJECT)
    locating_meg_file = partial(utils.locating_file, parent_fol=SUBJECT_MEG_FOLDER)
    # if not op.isdir(SUBJECT_MEG_FOLDER):
    #     SUBJECT_MEG_FOLDER = op.join(subjects_meg_dir)
    # if not op.isdir(SUBJECT_MEG_FOLDER):
    #     raise Exception("Can't find the subject's MEG folder! {}".format(SUBJECT_MEG_FOLDER))
    utils.make_dir(SUBJECT_MEG_FOLDER)
    print('Subject meg dir: {}'.format(SUBJECT_MEG_FOLDER))
    SUBJECT_MRI_FOLDER = op.join(subjects_mri_dir, MRI_SUBJECT)
    locating_subject_file = partial(utils.locating_file, parent_fol=SUBJECT_MRI_FOLDER)
    MMVT_SUBJECT_FOLDER = op.join(mmvt_dir, MRI_SUBJECT)
    _get_fif_name_cond = partial(get_file_name, fname_format=fname_format, file_type='fif',
        cleaning_method=cleaning_method, contrast=contrast, raw_fname_format=raw_fname_format,
                 fwd_fname_format=fwd_fname_format, inv_fname_format=inv_fname_format)
    _get_fif_name_no_cond = partial(_get_fif_name_cond, cond='')
    _get_fif_name = _get_fif_name_cond if files_includes_cond else _get_fif_name_no_cond
    _get_txt_name = partial(get_file_name, fname_format=fname_format, file_type='txt',
        cleaning_method=cleaning_method) #, contrast=contrast)
    _get_stc_name = partial(get_file_name, fname_format=fname_format_cond, file_type='stc',
                            cleaning_method=cleaning_method, contrast=contrast)
    _get_pkl_name = partial(get_file_name, fname_format=fname_format_cond, file_type='pkl',
                            cleaning_method=cleaning_method, contrast=contrast)
    _get_pkl_name_no_cond = partial(_get_pkl_name, cond='')

    RAW = _get_fif_name('raw', contrast='', cond='')
    RAW_ICA = _get_fif_name('ica-raw', contrast='', cond='')
    alt_raw_fname = '{}.fif'.format(RAW[:-len('-raw.fif')])
    if not op.isfile(RAW) and op.isfile(alt_raw_fname):
        RAW = alt_raw_fname
    INFO = _get_pkl_name_no_cond('raw-info')
    EVE = _get_txt_name('eve', cond='', contrast=contrast) if events_fname == '' else events_fname
    if not op.isfile(EVE):
        EVE = _get_txt_name('eve', cond='', contrast='')
    EVO = _get_fif_name('ave')
    COV = _get_fif_name('cov')
    DATA_COV = _get_fif_name('data-cov')
    NOISE_COV = _get_fif_name_no_cond('noise-cov')
    DATA_CSD = _get_pkl_name('data-csd')
    NOISE_CSD = _get_pkl_name('noise-csd')
    EPO = _get_fif_name('epo')
    EPO_NOISE = _get_fif_name('noise-epo')
    FWD = _get_fif_name_no_cond('fwd') if fwd_no_cond else _get_fif_name_cond('fwd')
    FWD_EEG = _get_fif_name_no_cond('eeg-fwd') if fwd_no_cond else _get_fif_name_cond('eeg-fwd')
    FWD_SUB = _get_fif_name_no_cond('sub-cortical-fwd') if fwd_no_cond else _get_fif_name_cond('sub-cortical-fwd')
    FWD_X = _get_fif_name_no_cond('{region}-fwd') if fwd_no_cond else _get_fif_name_cond('{region}-fwd')
    FWD_SMOOTH = _get_fif_name_no_cond('smooth-fwd') if inv_no_cond else _get_fif_name_cond('smooth-fwd')
    INV = _get_fif_name_no_cond('inv') if inv_no_cond else _get_fif_name_cond('inv')
    INV_EEG = _get_fif_name_no_cond('eeg-inv') if inv_no_cond else _get_fif_name_cond('eeg-inv')
    INV_SUB = _get_fif_name_no_cond('sub-cortical-inv') if inv_no_cond else _get_fif_name_cond('sub-cortical-inv')
    INV_X = _get_fif_name_no_cond('{region}-inv') if inv_no_cond else _get_fif_name_cond('{region}-inv')
    INV_SMOOTH = _get_fif_name_no_cond('smooth-inv') if inv_no_cond else _get_fif_name_cond('smooth-inv')
    INV_EEG_SMOOTH = _get_fif_name_no_cond('eeg-smooth-inv') if inv_no_cond else _get_fif_name_cond('eeg-smooth-inv')
    EMPTY_ROOM = _get_fif_name_no_cond('empty-raw').replace('-{}-'.format(task), '-').replace('_{}'.format(cleaning_method), '')
    STC = _get_stc_name('{method}')
    STC_HEMI = _get_stc_name('{method}-{hemi}')
    STC_HEMI_SAVE = op.splitext(STC_HEMI)[0].replace('-{hemi}','')
    STC_HEMI_SMOOTH = _get_stc_name('{method}-smoothed-{hemi}')
    STC_HEMI_SMOOTH_SAVE = op.splitext(STC_HEMI_SMOOTH)[0].replace('-{hemi}','')
    STC_MORPH = op.join(MEG_DIR, task, '{}', '{}-{}-inv.stc') # cond, method
    STC_ST = _get_pkl_name('{method}_st')
    # LBL = op.join(SUBJECT_MEG_FOLDER, 'labels_data_{}_{}_{}.npz') # atlas, extract_method, hemi
    LBL = op.join(MMVT_SUBJECT_FOLDER, 'meg', 'labels_data_{}_{}_{}.npz')  # atlas, extract_method, hemi
    ACT = op.join(MMVT_SUBJECT_FOLDER, 'activity_map_{}') # hemi
    # MRI files
    MRI = op.join(SUBJECT_MRI_FOLDER, 'mri', 'transforms', '{}-trans.fif'.format(MRI_SUBJECT))
    SRC = op.join(SUBJECT_MRI_FOLDER, 'bem', '{}-oct-6p-src.fif'.format(MRI_SUBJECT))
    SRC_SMOOTH = op.join(SUBJECT_MRI_FOLDER, 'bem', '{}-all-src.fif'.format(MRI_SUBJECT))
    BEM = op.join(SUBJECT_MRI_FOLDER, 'bem', '{}-5120-5120-5120-bem-sol.fif'.format(MRI_SUBJECT))
    COR = op.join(SUBJECT_MRI_FOLDER, 'mri', 'T1-neuromag', 'sets', 'COR.fif')
    if not op.isfile(COR):
        COR = op.join(SUBJECT_MEG_FOLDER, '{}-cor-trans.fif'.format(SUBJECT))
    ASEG = op.join(SUBJECT_MRI_FOLDER, 'ascii')
    MEG_TO_HEAD_TRANS = op.join(SUBJECT_MEG_FOLDER, 'trans', 'meg_to_head_trans.npy')
    print_files_names()


def print_files_names():
    print_file(RAW, 'raw')
    print_file(EVE, 'events')
    print_file(FWD, 'forward')
    print_file(INV, 'inverse')
    print_file(EPO, 'epochs')
    print_file(EVO, 'evoked')
    # MRI files
    print_file(MRI, 'subject MRI transform')
    print_file(SRC, 'subject MRI source')
    print_file(BEM, 'subject MRI BEM model')
    print_file(COR, 'subject MRI co-registration')


def print_file(fname, file_desc):
    print('{}: {} {}'.format(file_desc, fname, " !!!! doesn't exist !!!!" if not op.isfile(fname) else ""))


def get_file_name(ana_type, subject='', file_type='fif', fname_format='', cond='{cond}', cleaning_method='',
                  contrast='', root_dir='', raw_fname_format='', fwd_fname_format='', inv_fname_format=''):
    if fname_format == '':
        fname_format = '{subject}-{ana_type}.{file_type}'
    if subject=='':
        subject = SUBJECT
    args = {'subject':subject, 'ana_type':ana_type, 'file_type':file_type,
        'cleaning_method':cleaning_method, 'contrast':contrast}
    if '{cond}' in fname_format:
        args['cond'] = cond
    if ana_type == 'raw' and raw_fname_format != '':
        fname_format = raw_fname_format
    elif ana_type == 'fwd' and fwd_fname_format != '':
        fname_format = fwd_fname_format
    elif ana_type == 'inv' and inv_fname_format != '':
        fname_format = inv_fname_format
    if '{ana_type}' not in fname_format and '{file_type}' not in fname_format:
        fname_format = '{}-{}.{}'.format(fname_format, '{ana_type}', '{file_type}')
    fname = fname_format.format(**args)
    while '__' in fname:
        fname = fname.replace('__', '_')
    if '_-' in fname:
        fname = fname.replace('_-', '-')
    if root_dir == '':
        root_dir = SUBJECT_MEG_FOLDER
    return op.join(root_dir, fname)


def load_raw(raw_fname='', bad_channels=[], l_freq=None, h_freq=None, raw_template='*raw.fif'):
    # read the data
    raw_fname, raw_exist = locating_meg_file(raw_fname, glob_pattern=raw_template)
    if not raw_exist:
        raise Exception("Coulnd't find the raw file! {}".format(raw_fname))
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    if not op.isfile(INFO):
        utils.save(raw.info, INFO)
    if len(bad_channels) > 0:
        raw.info['bads'] = args.bad_channels
    if l_freq or h_freq:
        raw = raw.filter(l_freq, h_freq)
    return raw


def calcNoiseCov(epoches):
    noiseCov = mne.compute_covariance(epoches, tmax=None)
    # regularize noise covariance
    # noiseCov = mne.cov.regularize(noiseCov, evoked.info,
    #     mag=0.05, proj=True) # grad=0.05, eeg=0.1
    noiseCov.save(COV)
    # allEpoches = findEpoches(raw, picks, events, dict(onset=20), tmin=0, tmax=3.5)
    # evoked = allEpoches['onset'].average()
    # evoked.save(EVO)


# def calc_demi_epoches(windows_length, windows_shift, windows_num, raw, tmin, baseline,
#                       pick_meg=True, pick_eeg=False, pick_eog=False, reject=True,
#                       reject_grad=4000e-13, reject_mag=4e-12, reject_eog=150e-6, task='', epoches_fname=EPO):
#     demi_events, demi_events_ids = create_demi_events(raw, windows_length, windows_shift)
#     demi_tmax = windows_length / 1000.0
#     calc_epoches(raw, demi_events_ids, tmin, demi_tmax, baseline, False, demi_events, None, pick_meg, pick_eeg,
#                  pick_eog, reject, reject_grad, reject_mag, reject_eog, False, epoches_fname, task,
#                  windows_length, windows_shift, windows_num)


def create_demi_events(raw, windows_length, windows_shift, epoches_nun=0):
    import math
    # T = raw._data.shape[1]
    T = raw.last_samp - raw.first_samp + 1
    if epoches_nun == 0:
        epoches_nun = math.floor((T - windows_length) / windows_shift + 1)
    demi_events = np.zeros((epoches_nun, 3), dtype=np.uint32)
    for win_ind in range(epoches_nun):
        demi_events[win_ind] = [win_ind * windows_shift, win_ind * windows_shift + windows_length, 0]
    # for win_ind, win in enumerate(range(0, max_time, W * 2)):
        # demi_events[win_ind * 2] = [win, win + W, 0]
        # demi_events[win_ind * 2 + 1] = [win + W + 1, win + W * 2, 1]
    # demi_events_ids = {'demi_1': 0, 'demi_2': 1}
    demi_events[:, :2] += raw.first_samp
    demi_conditions = {'demi': 0}
    return demi_events, demi_conditions


def calc_epoches(raw, conditions, tmin, tmax, baseline, read_events_from_file=False, events=None,
                 stim_channels=None, pick_meg=True, pick_eeg=False, pick_eog=False, reject=True,
                 reject_grad=4000e-13, reject_mag=4e-12, reject_eog=150e-6, remove_power_line_noise=True,
                 power_line_freq=60, epoches_fname=None, task='', windows_length=1000, windows_shift=500,
                 windows_num=0, overwrite_epochs=False, eve_template='*eve.fif'):
    epoches_fname = EPO if epoches_fname is None else epoches_fname
    if op.isfile(epoches_fname) and not overwrite_epochs:
        epochs = mne.read_epochs(epoches_fname)
        return epochs
    picks = mne.pick_types(raw.info, meg=pick_meg, eeg=pick_eeg, eog=pick_eog, exclude='bads')
    # events[:, 2] = [str(ev)[event_digit] for ev in events[:, 2]]
    if reject:
        reject_dict = {}
        if reject_mag > 0:
            reject_dict['mag'] = reject_mag
        if reject_grad > 0:
            reject_dict['grad'] = reject_grad
        if pick_eog and reject_eog > 0:
            reject_dict['eog'] = reject_eog
    else:
        reject_dict = None
    # epochs = find_epoches(raw, picks, events, events, tmin=tmin, tmax=tmax)
    if remove_power_line_noise:
        raw.notch_filter(np.arange(power_line_freq, power_line_freq * 4 + 1, power_line_freq), picks=picks)
        # raw.notch_filter(np.arange(60, 241, 60), picks=picks)
    if read_events_from_file:
        events_fname, event_fname_exist = locating_meg_file(EVE, glob_pattern=eve_template)
        if events is None and event_fname_exist:
            # events_fname = events_fname if events_fname != '' else EVE
            print('read events from {}'.format(events_fname))
            events = mne.read_events(events_fname)
    else:
        if events is None:
            try:
                print('Finding events in {}'.format(stim_channels))
                events = mne.find_events(raw, stim_channel=stim_channels)
            except:
                print('No stim channels found!')
                events = np.array([])
    if events.shape[0] == 0:
        if task != 'rest':
            ans = input('Are you sure you want to have only one epoch, containing all the data (y/n)? ')
            if ans != 'y':
                return None
        events, conditions = create_demi_events(raw, windows_length, windows_shift, windows_num)
        tmax = windows_length / 1000.0

    if tmax - tmin <= 0:
        raise Exception('tmax-tmin must be greater than zero!')
    if list(conditions.keys())[0] == 'all' and len(np.unique(events[:, 2])) == 1:
        event_id = np.unique(events[:, 2])[0]
        events_conditions = {k: event_id for k, v in conditions.items()}
    else:
        events_conditions = {k: v for k,v in conditions.items() if v in np.unique(events[:, 2])}
    epochs = mne.Epochs(raw, events, events_conditions, tmin, tmax, proj=True, picks=picks,
                        baseline=baseline, preload=True, reject=reject_dict)
    print('{} good epochs'.format(len(epochs)))
    save_epochs(epochs, epoches_fname)
    return epochs


def save_epochs(epochs, epoches_fname=None):
    epoches_fname = EPO if epoches_fname is None else epoches_fname
    if '{cond}' in epoches_fname:
        for event in epochs.event_id: #events.keys():
            epochs[event].save(get_cond_fname(epoches_fname, event))
    else:
        try:
            epochs.save(epoches_fname)
        except:
            print(traceback.format_exc())

# def createEventsFiles(behavior_file, pattern):
#     make_ecr_events(RAW, behavior_file, EVE, pattern)

def calc_epochs_necessary_files(args):
    necessary_files = []
    if args.calc_epochs_from_raw:
        necessary_files.append(RAW)
        if args.read_events_from_file:
            necessary_files.append(EVE)
    else:
        necessary_files.append(EPO)
    return necessary_files


def calc_epochs_wrapper_args(conditions, args, raw=None):
    return calc_epochs_wrapper(
        conditions, args.t_min, args.t_max, args.baseline, raw, args.read_events_from_file,
        None, args.stim_channels,
        args.pick_meg, args.pick_eeg, args.pick_eog, args.reject,
        args.reject_grad, args.reject_mag, args.reject_eog, args.remove_power_line_noise, args.power_line_freq,
        args.bad_channels, args.l_freq, args.h_freq, args.task, args.windows_length, args.windows_shift,
        args.windows_num, args.overwrite_epochs, args.epo_fname, args.raw_fname, args.eve_template)


def calc_epochs_wrapper(
        conditions, tmin, tmax, baseline, raw=None, read_events_from_file=False, events_mat=None,
        stim_channels=None, pick_meg=True, pick_eeg=False, pick_eog=False,
        reject=True, reject_grad=4000e-13, reject_mag=4e-12, reject_eog=150e-6, remove_power_line_noise=True,
        power_line_freq=60, bad_channels=[], l_freq=None, h_freq=None, task='', windows_length=1000, windows_shift=500,
        windows_num=0, overwrite_epochs=False, epo_fname='', raw_fname='', eve_template='*eve.fif'):
    # Calc evoked data for averaged data and for each condition
    try:
        epo_fname = get_epo_fname(epo_fname)
        if '{cond}' not in epo_fname:
            epo_exist = op.isfile(epo_fname)
        else:
            epo_exist = all([op.isfile(get_cond_fname(epo_fname, cond)) for cond in conditions.keys()])
        if epo_exist and not overwrite_epochs:
            if '{cond}' in epo_fname:
                epo_exist = True
                epochs = {}
                for cond in conditions.keys():
                    if op.isfile(get_cond_fname(epo_fname, cond)):
                        epochs[cond] = mne.read_epochs(get_cond_fname(epo_fname, cond))
                    else:
                        epo_exist = False
                        break
            else:
                if op.isfile(epo_fname):
                    epochs = mne.read_epochs(epo_fname)
        if not epo_exist or overwrite_epochs:
            if raw is None:
                if raw_fname == '':
                    raw_fname = get_raw_fname(raw_fname)
                raw = load_raw(raw_fname, bad_channels, l_freq, h_freq)
            epochs = calc_epoches(raw, conditions, tmin, tmax, baseline, read_events_from_file, events_mat,
                                  stim_channels, pick_meg, pick_eeg, pick_eog, reject,
                                  reject_grad, reject_mag, reject_eog, remove_power_line_noise, power_line_freq,
                                  epo_fname, task, windows_length, windows_shift, windows_num, overwrite_epochs,
                                  eve_template)
        # if task != 'rest':
        #     all_evoked = calc_evoked_from_epochs(epochs, conditions)
        # else:
        #     all_evoked = None
        flag = True
    except:
        print(traceback.format_exc())
        epochs = None
        flag = False

    return flag, epochs


def calc_evokes(epochs, events, mri_subject, normalize_data=True, evoked_fname='', norm_by_percentile=False,
                norm_percs=None, modality='meg', calc_max_min_diff=True, calc_evoked_for_all_epoches=False,
                overwrite_evoked=False, task='', set_eeg_reference=True):
    try:
        evoked_fname = get_evo_fname(evoked_fname)
        fol = utils.make_dir(op.join(MMVT_DIR, mri_subject, modality))
        task_str = f'{task}_' if task != '' else ''
        mmvt_files = [op.join(fol, f'{task_str}sensors_evoked_data.npy'),
                      op.join(fol, f'{task_str}sensors_evoked_data_meta.npz'),
                      op.join(fol, f'{task_str}sensors_evoked_minmax.npy')]
        if op.isfile(evoked_fname) and all([op.isfile(f) for f in mmvt_files]) and not overwrite_evoked:
            evoked = mne.read_evokeds(evoked_fname)
            return True, evoked
        events_keys = list(events.keys())
        if epochs is None:
            epochs = mne.read_epochs(EPO)
        if any([event not in epochs.event_id for event in events_keys]):
            print('Not all the events can be found in the epochs! (events = {})'.format(events_keys))
            events_keys = list(set(epochs.event_id.keys()) & set(events.keys()))
            # return False, None
        evokes = [epochs[event].average() for event in events_keys] # if event in list(epochs.event_id.keys())]
        if set_eeg_reference:
            try:
                for evoked in evokes:
                    mne.set_eeg_reference(evoked, ref_channels=None)
                    evoked.apply_proj()
            except:
                print("_calc_inverse_operator: Can't add eeg_reference to the evoked")

        if calc_evoked_for_all_epoches:
            epochs_all = mne.concatenate_epochs([epochs[event] for event in events_keys])
            evokes_all = epochs_all.average()
            evokes_all_fname = op.join(utils.get_parent_fol(evoked_fname), f'{mri_subject}-all-eve.fif')
            mne.write_evokeds(evokes_all_fname, evokes_all)
        else:
            evokes_all = None
        save_evokes_to_mmvt(evokes, events_keys, mri_subject, evokes_all, normalize_data, norm_by_percentile,
                            norm_percs, modality, calc_max_min_diff, task)
        if '{cond}' in evoked_fname:
            # evokes = {event: epochs[event].average() for event in events_keys}
            for event, evoked in zip(events_keys, evokes):
                mne.write_evokeds(get_cond_fname(evoked_fname, event), evoked)
        else:
            # evokes = [epochs[event].average() for event in events_keys]
            mne.write_evokeds(evoked_fname, evokes)
    except:
        print(traceback.format_exc())
        return False, None
    else:
        if '{cond}' in evoked_fname:
            flag = all([op.isfile(get_cond_fname(evoked_fname, event)) for event in evokes.keys()])
        else:
            flag = op.isfile(evoked_fname)
    return flag, evokes


def save_evokes_to_mmvt(evokes, events_keys, mri_subject, evokes_all=None, normalize_data=False,
                        norm_by_percentile=False, norm_percs=None, modality='meg', calc_max_min_diff=True, task=''):
    fol = utils.make_dir(op.join(MMVT_DIR, mri_subject, modality))
    first_evokes = evokes if isinstance(evokes, mne.evoked.EvokedArray) else evokes[0]
    if modality == 'meg':
        channels_indices = [k for k, name in enumerate(evokes[0].ch_names) if name.startswith('MEG')]
        if len(channels_indices) == 0:
            print(f'None of the channels names starts with {channels_prefix}!')
    else:
        channels_indices = range(len(first_evokes.ch_names))
    ch_names = [first_evokes.ch_names[k] for k in channels_indices]
    dt = np.diff(first_evokes.times[:2])[0]
    if isinstance(evokes, mne.evoked.EvokedArray):
        data = evokes.data[channels_indices]
    else:
        data_shape = len(events_keys) + (0 if evokes_all is None else 1)
        data = np.zeros((len(channels_indices), first_evokes.data.shape[1], data_shape))
        for event_ind, event in enumerate(events_keys):
            data[:, :, event_ind] = evokes[event_ind].data[channels_indices]
        if evokes_all is not None:
            data[:, :, data_shape - 1] = evokes_all.data[channels_indices]
    if normalize_data:
        data = utils.normalize_data(data, norm_by_percentile, norm_percs)
    else:
        factor = 6 if modality == 'eeg' else 12 # micro V for EEG, fT (Magnetometers) and fT/cm (Gradiometers) for MEG
        data *= np.power(10, factor)
    if calc_max_min_diff:
        data_diff = np.diff(data) if len(events_keys) > 1 else np.squeeze(data)
        data_max, data_min = utils.get_data_max_min(data_diff, norm_by_percentile, norm_percs)
    else:
        data_max, data_min = utils.get_data_max_min(data, norm_by_percentile, norm_percs)
    max_abs = utils.get_max_abs(data_max, data_min)
    if evokes_all is not None:
        events_keys.append('all')
    task = f'{task}_' if task != '' else ''
    np.save(op.join(fol, f'{task}sensors_evoked_data.npy'), data)
    np.savez(op.join(fol, f'{task}sensors_evoked_data_meta.npz'), names=ch_names, conditions=events_keys, dt=dt)
    np.save(op.join(fol, f'{task}sensors_evoked_minmax.npy'), [-max_abs, max_abs])


def equalize_epoch_counts(events, method='mintime'):
    if '{cond}' not in EPO:
        epochs = mne.read_epochs(EPO)
    else:
        epochs = []
        for cond_name in events.keys():
            epochs_cond = mne.read_epochs(EPO.format(cond=cond_name))
            epochs.append(epochs_cond)
    mne.epochs.equalize_epoch_counts(epochs, method='mintime')
    if '{cond}' not in EPO == 0:
        epochs.save(EPO)
    else:
        for cond_name, epochs in zip(events.keys(), epochs):
            epochs.save(EPO.format(cond=cond_name))


def find_epoches(raw, picks,  events, event_id, tmin, tmax, baseline=(None, 0)):
    # remove events that are not in the events table
    event_id = dict([(k, ev) for (k, ev) in event_id.iteritems() if ev in np.unique(events[:, 2])])
    return mne.Epochs(raw=raw, events=events, event_id=event_id, tmin=tmin, tmax=tmax, proj=True,
                      picks=picks, baseline=baseline, preload=True, reject=None) # reject can be dict(mag=4e-12)


def check_src_ply_vertices_num(src):
    # check the vertices num with the ply files
    ply_vertives_num = utils.get_ply_vertices_num(op.join(MMVT_DIR, MRI_SUBJECT, 'surf', '{}.pial.ply'))
    if ply_vertives_num is not None:
        print(ply_vertives_num)
        src_vertices_num = [src_h['np'] for src_h in src]
        print(src_vertices_num)
        if not src_vertices_num[0] in ply_vertives_num.values() or \
                not src_vertices_num[1] in ply_vertives_num.values():
            raise Exception("src and ply files doesn't have the same vertices number! {}".format(SRC))
    else:
        print('No ply files to check the src!')


def make_smoothed_forward_solution(events, n_jobs=4, usingEEG=True, usingMEG=True):
    src = create_smooth_src(MRI_SUBJECT)
    if '{cond}' not in EPO:
        fwd = _make_forward_solution(src, EPO, usingMEG, usingEEG, n_jobs)
        mne.write_forward_solution(FWD_SMOOTH, fwd, overwrite=True)
    else:
        for cond in events.keys():
            fwd = _make_forward_solution(src, get_cond_fname(EPO, cond), usingMEG, usingEEG, n_jobs)
            mne.write_forward_solution(get_cond_fname(FWD_SMOOTH, cond), fwd, overwrite=True)
    return fwd


def create_smooth_src(subject, surface='pial', overwrite=False, fname=SRC_SMOOTH):
    src = mne.setup_source_space(subject, surface=surface, overwrite=overwrite, spacing='all', fname=fname)
    return src


def check_src(mri_subject, recreate_the_source_space=False, recreate_src_spacing='oct6', recreate_src_surface='white',
              n_jobs=2):
    # https://martinos.org/mne/stable/manual/cookbook.html#anatomical-information
    # src_fname, src_exist = locating_subject_file(SRC, '*-src.fif')
    src_fname = op.join(SUBJECTS_MRI_DIR, mri_subject, 'bem',
                        f'{mri_subject}-{recreate_src_spacing[:-1]}-{recreate_src_spacing[-1]}-src.fif')
    if not op.isfile(src_fname):
        src_fname = op.join(SUBJECTS_MRI_DIR, mri_subject, 'bem', '{}-{}-{}-src.fif'.format(
            mri_subject, recreate_src_spacing[:3], recreate_src_spacing[3:]))
    if not op.isfile(src_fname):
        src_files = glob.glob(op.join(SUBJECTS_MRI_DIR, mri_subject, 'bem', '*-src.fif'))
        if len(src_files) > 0:
            src_fname = utils.select_one_file(src_files, '', 'source files')
    if not recreate_the_source_space and op.isfile(src_fname):
        src = mne.read_source_spaces(src_fname)
    else:
        if not recreate_the_source_space:
            ans = input("Can't find the source file, recreate it (y/n)? (spacing={}, surface={}) ".format(
                recreate_src_spacing, recreate_src_surface))
        if recreate_the_source_space or ans == 'y':
            # oct_name, oct_num = recreate_src_spacing[:3], recreate_src_spacing[-1]
            # prepare_subject_folder(
            #     mri_subject, args.remote_subject_dir, op.join(SUBJECTS_MRI_DIR, mri_subject),
            #     {'bem': '{}-{}-{}-src.fif'.format(mri_subject, oct_name, oct_num)}, args)
            # https://martinos.org/mne/dev/manual/cookbook.html#source-localization
            utils.make_dir(op.join(SUBJECTS_MRI_DIR, MRI_SUBJECT, 'bem'))
            src = mne.setup_source_space(MRI_SUBJECT, spacing=recreate_src_spacing, surface=recreate_src_surface,
                                         overwrite=True, subjects_dir=SUBJECTS_MRI_DIR, n_jobs=n_jobs)
        else:
            raise Exception("Can't calculate the fwd solution without the source")
    return src


def check_bem(mri_subject, args={}):
    if len(args) == 0:
        args = utils.Bag(
            sftp=False, sftp_username='', sftp_domain='', sftp_password='',
            overwrite_fs_files=False, print_traceback=False, sftp_port=22)
    bem_fname, bem_exist = locating_subject_file(BEM, '*-bem-sol.fif')
    if not bem_exist:
        prepare_subject_folder(
            mri_subject, args.remote_subject_dir, SUBJECTS_MRI_DIR,
            {'bem': ['*-bem-sol.fif']}, args)
    if not op.isfile(bem_fname):
        bem_files = ['brain.surf', 'inner_skull.surf', 'outer_skin.surf', 'outer_skull.surf']
        watershed_files = ['{}_brain_surface', '{}_inner_skull_surface', '{}_outer_skin_surface',
                           '{}_outer_skull_surface']
        bem_fol = op.join(SUBJECTS_MRI_DIR, mri_subject, 'bem')
        bem_files_exist = np.all([op.isfile(op.join(bem_fol, bem_fname)) for bem_fname in bem_files])
        if not bem_files_exist:
            prepare_subject_folder(
                mri_subject, args.remote_subject_dir, SUBJECTS_MRI_DIR,
                {'bem': [f for f in bem_files]}, args)
        watershed_files_exist = np.all(
            [op.isfile(op.join(bem_fol, 'watershed', watershed_fname.format(mri_subject))) for watershed_fname in
             watershed_files])
        if not bem_files_exist and not watershed_files_exist:
            err_msg = '''BEM files don't exist, you should create it first using mne_watershed_bem.
                For that you need to open a terminal, define SUBJECTS_DIR, SUBJECT, source MNE, and run
                mne_watershed_bem.
                You can take a look here:
                http://perso.telecom-paristech.fr/~gramfort/mne/MRC/mne_anatomical_workflow.pdf '''
            raise Exception(err_msg)
        if not bem_files_exist and watershed_files_exist:
            for bem_file, watershed_file in zip(bem_files, watershed_files):
                utils.remove_file(bem_file)
                shutil.copy(op.join(bem_fol, 'watershed', watershed_file.format(mri_subject)),
                            op.join(bem_fol, bem_file))
        model = mne.make_bem_model(mri_subject, subjects_dir=SUBJECTS_MRI_DIR)
        bem = mne.make_bem_solution(model)
        mne.write_bem_solution(BEM, bem)
    return op.isfile(bem_fname)


def make_forward_solution(mri_subject, events=None, evo_fname='', fwd_fname='', sub_corticals_codes_file='',
                          usingMEG=True, usingEEG=True, calc_corticals=True,
                          calc_subcorticals=True, recreate_the_source_space=False, recreate_src_spacing='oct6',
                          recreate_src_surface='white', overwrite_fwd=False, n_jobs=4, args={}):
    fwd, fwd_with_subcortical = None, None
    evo_fname = get_evo_fname(evo_fname)
    fwd_fname = get_fwd_fname(fwd_fname, usingMEG, usingEEG)
    events_keys = events.keys() if events is not None else ['all']
    try:
        src = check_src(mri_subject, recreate_the_source_space, recreate_src_spacing,recreate_src_surface, n_jobs)
        check_src_ply_vertices_num(src)
        check_bem(mri_subject, args)
        sub_corticals = utils.read_sub_corticals_code_file(sub_corticals_codes_file)
        if '{cond}' not in evo_fname:
            if calc_corticals:
                if overwrite_fwd or not op.isfile(fwd_fname):
                    fwd = _make_forward_solution(src, evo_fname, usingMEG, usingEEG, n_jobs)
                    mne.write_forward_solution(fwd_fname, fwd, overwrite=True)
            if calc_subcorticals and len(sub_corticals) > 0:
                # add a subcortical volumes
                if overwrite_fwd or not op.isfile(FWD_SUB):
                    src_with_subcortical = add_subcortical_volumes(src, sub_corticals)
                    fwd_with_subcortical = _make_forward_solution(
                        src_with_subcortical, evo_fname, usingMEG, usingEEG, n_jobs)
                    mne.write_forward_solution(FWD_SUB, fwd_with_subcortical, overwrite=True)
        else:
            for cond in events_keys:
                if calc_corticals:
                    fwd_cond_fname = get_cond_fname(fwd_fname, cond)
                    if overwrite_fwd or not op.isfile(fwd_cond_fname):
                        fwd = _make_forward_solution(src, get_cond_fname(evo_fname, cond), usingMEG, usingEEG, n_jobs)
                        mne.write_forward_solution(fwd_cond_fname, fwd, overwrite=True)
                if calc_subcorticals and len(sub_corticals) > 0:
                    # add a subcortical volumes
                    fwd_cond_fname = get_cond_fname(FWD_SUB, cond)
                    if overwrite_fwd or not op.isfile(fwd_cond_fname):
                        src_with_subcortical = add_subcortical_volumes(src, sub_corticals)
                        fwd_with_subcortical = _make_forward_solution(
                            src_with_subcortical, get_cond_fname(evo_fname, cond), usingMEG, usingEEG, n_jobs)
                        mne.write_forward_solution(fwd_cond_fname, fwd_with_subcortical, overwrite=True)
        flag = True
    except:
        print(traceback.format_exc())
        print('Error in calculating fwd solution')
        flag = False

    return flag, fwd, fwd_with_subcortical


def make_forward_solution_to_specific_subcortrical(events, region, n_jobs=4, usingMEG=True, usingEEG=True):
    import nibabel as nib
    aseg_fname = op.join(SUBJECTS_MRI_DIR, MRI_SUBJECT, 'mri', 'aseg.mgz')
    aseg = nib.load(aseg_fname)
    aseg_hdr = aseg.get_header()
    for cond in events.keys():
        src = add_subcortical_volumes(None, [region])
        fwd = _make_forward_solution(src, get_cond_fname(EPO, cond), usingMEG, usingEEG, n_jobs)
        mne.write_forward_solution(get_cond_fname(FWD_X, cond, region=region), fwd, overwrite=True)
    return fwd


def make_forward_solution_to_specific_points(events, pts, region_name, epo_fname='', fwd_fname='',
        usingMEG=True, usingEEG=True, n_jobs=4):
    from mne.source_space import _make_discrete_source_space

    epo_fname = EPO if epo_fname == '' else epo_fname
    fwd_fname = FWD_X if fwd_fname == '' else fwd_fname

    # Convert to meters
    pts /= 1000.
    # Set orientations
    ori = np.zeros(pts.shape)
    ori[:, 2] = 1.0
    pos = dict(rr=pts, nn=ori)

    # Setup a discrete source
    sp = _make_discrete_source_space(pos)
    sp.update(dict(nearest=None, dist=None, use_tris=None, patch_inds=None,
                   dist_limit=None, pinfo=None, ntri=0, nearest_dist=None,
                   nuse_tri=None, tris=None, type='discrete',
                   seg_name=region_name))

    src = mne.SourceSpaces([sp])
    for cond in events.keys():
        fwd = _make_forward_solution(src, get_cond_fname(epo_fname, cond), usingMEG, usingEEG, n_jobs)
        mne.write_forward_solution(get_cond_fname(fwd_fname, cond, region=region_name), fwd, overwrite=True)
    return fwd


def _make_forward_solution(src, epo, usingMEG=True, usingEEG=True, n_jobs=6):
    bem = BEM
    if not op.isfile(BEM):
        bem = utils.select_one_file(op.join(SUBJECTS_MRI_DIR, MRI_SUBJECT, 'bem', '*-bem-sol.fif'), '', 'bem files')
        if op.isfile(bem):
            shutil.copy(bem, BEM)
        else:
            raise Exception("Can't find the BEM file!")
    fwd = mne.make_forward_solution(info=epo, trans=COR, src=src, bem=bem, # mri=MRI
                                    meg=usingMEG, eeg=usingEEG, mindist=5.0,
                                    n_jobs=n_jobs, overwrite=True)
    return fwd


def add_subcortical_surfaces(src, seg_labels):
    """Adds a subcortical volume to a cortical source space
    """
    from mne.source_space import _make_discrete_source_space

    # Read the freesurfer lookup table
    lut = utils.read_freesurfer_lookup_table()

    # Get the indices to the desired labels
    for label in seg_labels:

        # Get numeric index to label
        seg_name, seg_id = utils.get_numeric_index_to_label(label, lut)
        srf_file = op.join(ASEG, 'aseg_%.3d.srf' % seg_id)
        pts, _, _, _ = utils.read_srf_file(srf_file)

        # Convert to meters
        pts /= 1000.

        # Set orientations
        ori = np.zeros(pts.shape)
        ori[:, 2] = 1.0

        # Store coordinates and orientations as dict
        pos = dict(rr=pts, nn=ori)

        # Setup a discrete source
        sp = _make_discrete_source_space(pos)
        sp.update(dict(nearest=None, dist=None, use_tris=None, patch_inds=None,
                       dist_limit=None, pinfo=None, ntri=0, nearest_dist=None,
                       nuse_tri=None, tris=None, type='discrete',
                       seg_name=seg_name))

        # Combine source spaces
        src.append(sp)

    return src


def add_subcortical_volumes(org_src, seg_labels, spacing=5., use_grid=True):
    """Adds a subcortical volume to a cortical source space
    """
    # Get the subject
    import nibabel as nib
    from mne.source_space import _make_discrete_source_space

    if org_src is not None:
        src = org_src.copy()
    else:
        src = None

    # Find the segmentation file
    aseg_fname = op.join(SUBJECTS_MRI_DIR, MRI_SUBJECT, 'mri', 'aseg.mgz')
    aseg = nib.load(aseg_fname)
    aseg_hdr = aseg.get_header()
    sub_cortical_generator = utils.sub_cortical_voxels_generator(aseg, seg_labels, spacing, use_grid, FREESURFER_HOME)
    for pts, seg_name, seg_id in sub_cortical_generator:
        pts = utils.transform_voxels_to_RAS(aseg_hdr, pts)
        # Convert to meters
        pts /= 1000.
        # Set orientations
        ori = np.zeros(pts.shape)
        ori[:, 2] = 1.0

        # Store coordinates and orientations as dict
        pos = dict(rr=pts, nn=ori)

        # Setup a discrete source
        sp = _make_discrete_source_space(pos)
        sp.update(dict(nearest=None, dist=None, use_tris=None, patch_inds=None,
                       dist_limit=None, pinfo=None, ntri=0, nearest_dist=None,
                       nuse_tri=None, tris=None, type='discrete',
                       seg_name=seg_name))

        # Combine source spaces
        if src is None:
            src = mne.SourceSpaces([sp])
        else:
            src.append(sp)

    return src


def calc_noise_cov(epochs=None, noise_t_min=None, noise_t_max=0, noise_cov_fname='', args=None, raw=None):
    if noise_cov_fname == '':
        noise_cov_fname = NOISE_COV
    if op.isfile(noise_cov_fname):
        noise_cov = mne.read_cov(noise_cov_fname)
        return noise_cov
    if epochs is None:
        noise_cov = recalc_epochs_for_noise_cov(noise_t_min, noise_t_max, args, raw)
        noise_cov.save(noise_cov_fname)
        return noise_cov
    if len(epochs) > 1:
        # Check if we need to recalc the epoches...
        if epochs.tmin <= noise_t_min and epochs.tmax >= noise_t_max:
            noise_cov = mne.compute_covariance(epochs.crop(noise_t_min, noise_t_max))
        else:
            # Yes, we do...
            noise_cov = recalc_epochs_for_noise_cov(noise_t_min, noise_t_max, args, raw)
    else:
        if op.isfile(EPO_NOISE):
            demi_epochs = mne.read_epochs(EPO_NOISE)
        else:
            raise Exception("You should split first your epochs into small demi epochs, see calc_demi_epoches")
        noise_cov = calc_noise_cov(demi_epochs)
    noise_cov.save(noise_cov_fname)
    return noise_cov


def recalc_epochs_for_noise_cov(noise_t_min, noise_t_max, args, raw=None):
    noise_args = utils.Bag(args.copy())
    noise_args.t_min = noise_t_min
    noise_args.t_max = noise_t_max
    noise_args.overwrite_epochs = False
    noise_args.baseline = (noise_t_min, noise_t_max)
    epo_fname = get_epo_fname(args.epo_fname)
    noise_args.epo_fname = '{}noise-epo.fif'.format(epo_fname[:-len('epo.fif')])
    _, noise_epochs = calc_epochs_wrapper_args(noise_args.conditions, noise_args, raw=raw)
    noise_cov = mne.compute_covariance(noise_epochs)
    return noise_cov


def get_inv_fname(inv_fname='', fwd_usingMEG=True, fwd_usingEEG=True):
    if inv_fname == '':
        inv_fname = INV_EEG if fwd_usingEEG and not fwd_usingMEG else INV
    inv_fname, inv_exist = locating_meg_file(inv_fname, '*inv.fif')
    if not inv_exist:
        files = glob.glob(op.join(MEG_DIR, SUBJECT, '*{}*'.format(inv_fname)))
        if len(files) > 0:
            inv_fname = utils.select_one_file(files)
    return inv_fname


def get_raw_fname(raw_fname=''):
    if raw_fname == '':
        raw_fname = RAW
    raw_fname, raw_exist = locating_meg_file(raw_fname, '*raw.fif')
    return raw_fname


def get_fwd_fname(fwd_fname='', fwd_usingMEG=True, fwd_usingEEG=True):
    if fwd_fname == '':
        fwd_fname = FWD_EEG if fwd_usingEEG and not fwd_usingMEG else FWD
    fwd_fname, fwd_exist = locating_meg_file(fwd_fname, '*fwd.fif')
    return fwd_fname


def get_epo_fname(epo_fname=''):
    if epo_fname == '':
        epo_fname = EPO
    epo_fname, epo_exist = locating_meg_file(epo_fname, '*epo.fif')
    return epo_fname


def get_evo_fname(evo_fname=''):
    if evo_fname == '':
        evo_fname = EVO
    evo_fname, epo_exist = locating_meg_file(evo_fname, '*ave.fif')
    return evo_fname


def calc_inverse_operator(
        events=None, raw_fname='', epo_fname='', evo_fname='', fwd_fname='', inv_fname='', noise_cov_fname='',
        inv_loose=0.2, inv_depth=0.8, noise_t_min=None, noise_t_max=0, overwrite_inverse_operator=False,
        use_empty_room_for_noise_cov=False, use_raw_for_noise_cov=False, overwrite_noise_cov=False,
        calc_for_cortical_fwd=True, calc_for_sub_cortical_fwd=True, fwd_usingMEG=True, fwd_usingEEG=True,
        calc_for_spec_sub_cortical=False, cortical_fwd=None, subcortical_fwd=None, spec_subcortical_fwd=None,
        region=None, args=None):
    raw_fname = get_raw_fname(raw_fname)
    fwd_fname = get_fwd_fname(fwd_fname, fwd_usingMEG, fwd_usingEEG)
    inv_fname = get_inv_fname(inv_fname, fwd_usingMEG, fwd_usingEEG)
    evo_fname = get_evo_fname(evo_fname)
    epo_fname = get_epo_fname(epo_fname)
    if noise_cov_fname == '':
        noise_cov_fname = NOISE_COV
    if events is None:
        conds = ['all']
    else:
        conds = ['all'] if '{cond}' not in epo_fname else events.keys()
    flag = True
    for cond in conds:
        if (not overwrite_inverse_operator and
            (not calc_for_cortical_fwd or op.isfile(get_cond_fname(inv_fname, cond))) and \
            (not calc_for_sub_cortical_fwd or op.isfile(get_cond_fname(INV_SUB, cond))) and \
            (not calc_for_spec_sub_cortical or op.isfile(get_cond_fname(INV_X, cond, region=region)))):
                continue
        try:
            if overwrite_noise_cov or not op.isfile(noise_cov_fname):
                if use_empty_room_for_noise_cov:
                    empty_room_fname, empty_room_exist = locating_meg_file(
                        EMPTY_ROOM, '*empty*.fif', raise_exception=True)
                    raw_empty_room = mne.io.read_raw_fif(empty_room_fname, add_eeg_ref=False)
                    noise_cov = mne.compute_raw_covariance(raw_empty_room, tmin=0, tmax=None)
                    noise_cov.save(noise_cov_fname)
                elif use_raw_for_noise_cov:
                    raw = mne.io.read_raw_fif(raw_fname) # preload=True # add_eeg_ref=False
                    noise_cov = mne.compute_raw_covariance(raw, tmin=0, tmax=None)
                    noise_cov.save(noise_cov_fname)
                else:
                    epo = get_cond_fname(epo_fname, cond)
                    epochs = mne.read_epochs(epo)
                    noise_cov = calc_noise_cov(epochs, noise_t_min, noise_t_max, noise_cov_fname, args)
            else:
                noise_cov = mne.read_cov(noise_cov_fname)
            # todo: should use noise_cov = calc_cov(...
            if calc_for_cortical_fwd and not op.isfile(get_cond_fname(inv_fname, cond)):
                if cortical_fwd is None:
                    cortical_fwd = get_cond_fname(fwd_fname, cond)
                _calc_inverse_operator(
                    cortical_fwd, get_cond_fname(inv_fname, cond), get_cond_fname(evo_fname, cond), noise_cov, inv_loose, inv_depth)
            if calc_for_sub_cortical_fwd and not op.isfile(get_cond_fname(INV_SUB, cond)):
                if subcortical_fwd is None:
                    subcortical_fwd = get_cond_fname(FWD_SUB, cond)
                _calc_inverse_operator(subcortical_fwd, get_cond_fname(INV_SUB, cond), evo_fname, noise_cov, None, None)
            if calc_for_spec_sub_cortical and not op.isfile(get_cond_fname(INV_X, cond, region=region)):
                if spec_subcortical_fwd is None:
                    spec_subcortical_fwd = get_cond_fname(FWD_X, cond, region=region)
                _calc_inverse_operator(spec_subcortical_fwd, get_cond_fname(INV_X, cond, region=region), evo_fname,
                                       noise_cov, None, None)
            flag = True
        except:
            print(traceback.format_exc())
            print('Error in calculating inv for {}'.format(cond))
            flag = False
    return flag


def _calc_inverse_operator(fwd_name, inv_name, evoked_fname, noise_cov, inv_loose=0.2, inv_depth=0.8):
    fwd = mne.read_forward_solution(fwd_name)
    evoked = mne.read_evokeds(evoked_fname)[0]
    inverse_operator = make_inverse_operator(evoked.info, fwd, noise_cov,
        loose=inv_loose, depth=inv_depth)
    write_inverse_operator(inv_name, inverse_operator)


# def calc_stc(inverse_method='dSPM'):
#     snr = 3.0
#     lambda2 = 1.0 / snr ** 2
#     inverse_operator = read_inverse_operator(INV)
#     evoked = mne.read_evokeds(EVO, condition=0, baseline=(None, 0))
#     stc = apply_inverse(evoked, inverse_operator, lambda2, inverse_method,
#                         pick_ori=None)
#     stc.save(STC.format('all', inverse_method))


def calc_stc_per_condition(events=None, stc_t_min=None, stc_t_max=None, inverse_method='dSPM', baseline=(None, 0),
                           apply_SSP_projection_vectors=True, add_eeg_ref=True, pick_ori=None,
                           single_trial_stc=False, save_stc=True, snr=3.0, overwrite_stc=False,
                           stc_template='', epo_fname='', evo_fname='', inv_fname='',
                           fwd_usingMEG=True, fwd_usingEEG=True, apply_on_raw=False, raw=None, epochs=None,
                           modality='meg', calc_stc_for_all=False, calc_stcs_diff=True, n_jobs=6):
    # todo: If the evoked is the raw (no events), we need to seperate it into N events with different ids, to avoid memory error
    # Other options is to use calc_labels_avg_for_rest
    epo_fname = get_epo_fname(epo_fname)
    evo_fname = get_evo_fname(evo_fname)
    inv_fname = get_inv_fname(inv_fname, fwd_usingMEG, fwd_usingEEG)
    if stc_template == '':
        stc_template = STC
        stc_hemi_template = STC_HEMI
    else:
        stc_hemi_template = '{}{}'.format(stc_template[:-4], '-{hemi}.stc')
    events_keys = list(events.keys()) if events is not None and isinstance(events, dict) else ['all']
    stcs, stcs_num = {}, {}
    lambda2 = 1.0 / snr ** 2
    global_inverse_operator = False
    if '{cond}' not in inv_fname:
        if not op.isfile(inv_fname):
            print('No inverse operator was found!')
            return False, stcs, stcs_num
        inverse_operator = read_inverse_operator(inv_fname)
        global_inverse_operator = True
    if calc_stc_for_all:
        events_keys.append('all')
    for cond_name in events_keys:
        stc_fname = stc_template.format(cond=cond_name, method=inverse_method)[:-4]
        if op.isfile('{}.stc'.format(stc_fname)) and not overwrite_stc:
            stcs[cond_name] = mne.read_source_estimate(stc_fname)
            continue
        try:
            if not global_inverse_operator:
                if not op.isfile(inv_fname.format(cond=cond_name)):
                    print('No inverse operator was found!')
                    return False, stcs, stcs_num
                inverse_operator = read_inverse_operator(inv_fname.format(cond=cond_name))
            if single_trial_stc:
                if epochs is None:
                    epo_fname = epo_fname.format(cond=cond_name)
                    if not op.isfile(epo_fname):
                        print('single_trial_stc and not epoch file was found!')
                        return False, stcs, stcs_num
                    epo_cond = get_cond_fname(epo_fname, cond_name)
                    epochs = mne.read_epochs(epo_cond, apply_SSP_projection_vectors, add_eeg_ref)
                try:
                    mne.set_eeg_reference(epochs, ref_channels=None)
                except:
                    print('annot create EEG average reference projector (no EEG data found)')
                stcs[cond_name] = mne.minimum_norm.apply_inverse_epochs(epochs, inverse_operator, lambda2, inverse_method,
                    pick_ori=pick_ori, return_generator=True)
                stcs_num[cond_name] = epochs.events.shape[0]
                # if save_stc:
                #     utils.save(stcs[cond_name], STC_ST.format(cond=cond_name, method=inverse_method))
            else:
                if apply_on_raw:
                    stcs_num[cond_name] = mne.minimum_norm.apply_inverse_raw(
                        raw, inverse_operator, lambda2, inverse_method, pick_ori=pick_ori)
                else:
                    evoked = get_evoked_cond(
                        cond_name, evo_fname, epo_fname, baseline, apply_SSP_projection_vectors, add_eeg_ref)
                    if evoked is None and cond_name == 'all':
                        all_evokes_fname = op.join(SUBJECT_MEG_FOLDER, '{}-all-eve.fif'.format(SUBJECT))
                        if op.isfile(all_evokes_fname):
                            evoked = mne.read_evokeds(all_evokes_fname)[0]
                    if evoked is None:
                        continue
                    if not stc_t_min is None and not stc_t_max is None:
                        evoked = evoked.crop(stc_t_min, stc_t_max)
                    try:
                        # todo: should check if this was already done
                        mne.set_eeg_reference(evoked, ref_channels=None)
                    except:
                        print('Cannot create EEG average reference projector (no EEG data found)')
                    stcs[cond_name] = apply_inverse(evoked, inverse_operator, lambda2, inverse_method, pick_ori=pick_ori)
                if save_stc and not op.isfile(stc_fname):
                    print('Saving the source estimate to {}.stc'.format(stc_fname))
                    stcs[cond_name].save(stc_fname)
                    # mmvt_stc_name = ''
                    stcs[cond_name].save(op.join(MMVT_DIR, MRI_SUBJECT, modality, utils.namebase(stc_fname)))
            flag = True
        except:
            print(traceback.format_exc())
            print('Error with {}!'.format(cond_name))
            flag = False
    if calc_stcs_diff and len(events) > 1:
        calc_stc_diff_both_hemis(events, stc_hemi_template, inverse_method, overwrite_stc)
    return flag, stcs, stcs_num


def calc_stc_diff_both_hemis(events, stc_hemi_template, inverse_method, overwrite_stc=False):
    if events is None or (not isinstance(events, dict) and len(events) < 2):
        return False
    stc_hemi_template = stc_hemi_template.format(cond='{cond}', method=inverse_method, hemi='{hemi}')
    if all([utils.both_hemi_files_exist(
            stc_hemi_template.format(cond=cond, hemi='{hemi}'))
            for cond in events.keys()]) and len(glob.glob(stc_hemi_template.format(cond='*', hemi='?h'))) >= 4:
        conds = list(events.keys())
        if len(conds) == 2:
            for hemi in utils.HEMIS:
                diff_fname = stc_hemi_template.format(cond='{}-{}'.format(conds[0], conds[1]), hemi=hemi)
                if op.isfile(diff_fname) and not overwrite_stc:
                    continue
                calc_stc_diff(
                    stc_hemi_template.format(cond=conds[0], hemi=hemi),
                    stc_hemi_template.format(cond=conds[1], hemi=hemi), diff_fname)
    return utils.both_hemi_files_exist(stc_hemi_template.format(cond='{}-{}'.format(conds[0], conds[1]), hemi='{hemi}'))


def dipoles_fit(dipoles_times, dipoloes_title, evokes=None, noise_cov_fname='', evo_fname='', min_dist=5.,
                use_meg=True, use_eeg=False, vol_atlas_fname='', vol_atlas_lut_fname='', mask_roi='', do_plot=False,
                n_jobs=6):
    from mne.forward import make_forward_dipole
    import nibabel as nib

    if noise_cov_fname == '':
        noise_cov_fname = NOISE_COV
    evo_fname = get_evo_fname(evo_fname)
    if vol_atlas_fname == '':
        vol_atlas_fname = op.join(op.join(MMVT_DIR, MRI_SUBJECT, 'freeview', 'aparc.DKTatlas40+aseg.mgz'))
    if vol_atlas_lut_fname == '':
        vol_atlas_lut_fname = op.join(op.join(MMVT_DIR, MRI_SUBJECT, 'freeview', 'aparc.DKTatlas40ColorLUT.txt'))

    if not op.isfile(noise_cov_fname):
        print("The noise covariance cannot be found in {}!".format(NOISE_COV))
        return False
    if not op.isfile(BEM):
        print("The BEM solution cannot be found in {}!".format(BEM))
        return False
    if not op.isfile(COR):
        print("The MEG-to-head-trans matrix cannot be found in {}!".format(COR))
        return False
    if evokes is None:
        evokes = mne.read_evokeds(evo_fname)
    if not isinstance(evokes, Iterable):
        evokes = [evokes]

    for evoked in evokes:
        evoked.pick_types(meg=use_meg, eeg=use_eeg)
        for (t_min, t_max), dipole_title in zip(dipoles_times, dipoloes_title):
            dipole_fname = op.join(SUBJECT_MEG_FOLDER, 'dipole_{}_{}.pkl'.format(evoked.comment, dipole_title))
            dipole_fix_output_fname = op.join(SUBJECT_MEG_FOLDER, 'dipole_fix_{}_{}.pkl'.format(evoked.comment, dipole_title))
            dipole_stc_fwd_fname = op.join(SUBJECT_MEG_FOLDER, 'dipole_{}_fwd_stc_{}.pkl'.format(evoked.comment, dipole_title))
            dipole_location_figure_fname = op.join(SUBJECT_MEG_FOLDER, 'dipole_{}_{}.png'.format(evoked.comment, dipole_title))
            if not op.isfile(dipole_fname):
                evoked_t = evoked.copy()
                evoked_t.crop(t_min, t_max)
                dipole, residual = mne.fit_dipole(evoked_t, noise_cov_fname, BEM, COR, min_dist, n_jobs)
                utils.save((dipole, residual), dipole_fname)
            else:
                dipole, residual = utils.load(dipole_fname)
            if do_plot:
                dipole.plot_locations(COR, 'DC', SUBJECTS_MRI_DIR, mode='orthoview')
            dipole_pos_vox = dipole_pos_to_vox(dipole, COR)
            if op.isfile(vol_atlas_fname) and op.isfile(vol_atlas_lut_fname) and mask_roi != '':
                mask = nib.load(vol_atlas_fname).get_data()
                lut = utils.read_freesurfer_lookup_table(lut_fname=vol_atlas_lut_fname, return_dict=True, reverse_dict=True)
                _lut = utils.read_freesurfer_lookup_table(lut_fname=vol_atlas_lut_fname, return_dict=True)
                mask_code = lut[mask_roi]
            #
            #     print('asdf')
            if not op.isfile(dipole_fix_output_fname):
                # Estimate the time course of a single dipole with fixed position and orientation
                # (the one that maximized GOF)over the entire interval
                best_idx = np.argmax(dipole.gof)
                dipole_fixed, residual_fixed = mne.fit_dipole(
                    evoked, noise_cov_fname, BEM, COR, min_dist, pos=dipole.pos[best_idx], ori=dipole.ori[best_idx],
                    n_jobs=n_jobs)
                utils.save((dipole_fixed, residual_fixed), dipole_fix_output_fname)
            else:
                dipole_fixed, residual_fixed = utils.load(dipole_fix_output_fname)
            if not op.isfile(dipole_stc_fwd_fname):
                dipole_fwd, dipole_stc = make_forward_dipole(dipole, BEM, evoked.info, COR, n_jobs=n_jobs)
                utils.save((dipole_fwd, dipole_stc), dipole_stc_fwd_fname)
            else:
                dipole_fwd, dipole_stc = utils.load(dipole_stc_fwd_fname)
            if not op.isfile(dipole_location_figure_fname) and do_plot:
                dipole.plot_locations(COR, MRI_SUBJECT, SUBJECTS_MRI_DIR, mode='orthoview')
                plt.savefig(dipole_location_figure_fname)

            if do_plot:
                # save_diploe_loc(dipole, COR)
                best_idx = np.argmax(dipole.gof)
                best_time = dipole.times[best_idx]
                # plot_predicted_dipole(evoked, dipole_fwd, dipole_stc, best_time)
                dipole_fixed.plot()


def find_dipole_cortical_locations(atlas, cond, dipole_title, grow_label=True, label_r=5,
                                   inv_fname='', stc_hemi_template='', extract_mode='mean_flip', n_jobs=6):
    from mne.transforms import _get_trans, apply_trans
    from scipy.spatial.distance import cdist

    if inv_fname == '':
        inv_fname = INV
    dipole_fname = op.join(SUBJECT_MEG_FOLDER, 'dipole_{}_{}.pkl'.format(cond, dipole_title))
    vertices_labels_lookup = lu.create_vertices_labels_lookup(MRI_SUBJECT, atlas)
    dipole, _ = utils.load(dipole_fname)
    trans = _get_trans(COR, fro='head', to='mri')[0]
    scatter_points = apply_trans(trans['trans'], dipole.pos) * 1e3

    vertices, vertices_dist = {}, {}
    for hemi_ind, hemi in enumerate(['lh', 'rh']):
        verts, _ = utils.read_ply_file(op.join(MMVT_DIR, MRI_SUBJECT, 'surf', '{}.pial.ply'.format(hemi)))
        dists = cdist(scatter_points, verts)
        vertices[hemi_ind] = np.argmin(dists, 1)
        vertices_dist[hemi_ind] = np.min(dists, 1)
    hemis = np.argmin(np.vstack((vertices_dist[0], vertices_dist[1])), 0)
    sort_inds = np.argsort(dipole.gof)[::-1]
    print('****** {} ********'.format(cond))
    dipole_vert = -1
    for ind in sort_inds:
        hemi_ind = hemis[ind]
        hemi = ['lh', 'rh'][hemi_ind]
        dist = vertices_dist[hemi_ind][ind]
        vert = vertices[hemi_ind][ind]
        label = vertices_labels_lookup[hemi][vert]
        gof = dipole.gof[ind]
        if label.startswith('precentral'):
            print(ind, hemi, dist, label, gof)
            if dipole_vert == -1:
                dipole_vert = vert
                dipole_hemi = hemi
                print(cond, gof)
                break
    label_data = None
    if grow_label and dipole_vert != -1:
        label_name = 'precentral_dipole_{}'.format(cond)
        label_fname = op.join(MMVT_DIR, MRI_SUBJECT, 'labels', '{}-{}.label'.format(label_name, dipole_hemi))
        if not op.isfile(label_fname):
            dipole_label = lu.grow_label(
                MRI_SUBJECT, dipole_vert, dipole_hemi, label_name, label_r, n_jobs)
        else:
            dipole_label = mne.read_label(label_fname)
        inverse_operator = read_inverse_operator(inv_fname.format(cond=cond))
        src = inverse_operator['src']
        stc = mne.read_source_estimate(stc_hemi_template.format(cond=cond, hemi=dipole_hemi))
        label_data = stc.extract_label_time_course(dipole_label, src, mode=extract_mode, allow_empty=True)
        label_data = np.squeeze(label_data)
    return dipole_vert, label_data, label_fname, dipole_hemi


def plot_predicted_dipole(evoked, dipole_fwd, dipole_stc, best_time):
    from mne.simulation import simulate_evoked

    noise_cov = mne.read_cov(NOISE_COV)
    pred_evoked = simulate_evoked(dipole_fwd, dipole_stc, evoked.info, cov=noise_cov) #, nave=np.inf)

    # first plot the topography at the time of the best fitting (single) dipole
    plot_params = dict(times=best_time, ch_type='mag', outlines='skirt',
                       colorbar=True)#, vmin=vmin, vmax=vmin)
    evoked.plot_topomap(**plot_params)
    # compare this to the predicted field
    pred_evoked.plot_topomap(**plot_params)
    print('Comparison of measured and predicted fields at {:.0f} ms'.format(best_time * 1000.))


def dipole_pos_to_vox(dipole, trans):
    from mne.transforms import _get_trans, apply_trans

    orig_trans = utils.Bag(np.load(op.join(MMVT_DIR, MRI_SUBJECT, 'orig_trans.npz')))
    trans = _get_trans(trans, fro='head', to='mri')[0]
    scatter_points = apply_trans(trans['trans'], dipole.pos) * 1e3
    dipole_locs_vox = apply_trans(orig_trans.ras_tkr2vox, scatter_points)
    dipole_locs_vox = np.rint(np.array(dipole_locs_vox)).astype(int)
    return dipole_locs_vox


def save_diploe_loc(dipole, trans):
    from mne.transforms import _get_trans, apply_trans

    best_idx = np.argmax(dipole.gof)
    best_point = dipole_locs[best_idx]
    dipole_xyz = np.rint(best_point).astype(int)
    dipole_ori = apply_trans(trans['trans'], dipole.ori, move=False)
    print(dipole_xyz, dipole_ori)


def get_evoked_cond(cond_name, evo_fname='', epo_fname='', baseline=(None, 0), apply_SSP_projection_vectors=True,
                    add_eeg_ref=True):
    evo_fname = get_evo_fname(evo_fname)
    epo_fname = get_epo_fname(epo_fname)
    if '{cond}' not in evo_fname:
        if not op.isfile(evo_fname):
            print('get_evoked_cond: No evoked file found!')
            return None
        try:
            evoked = mne.read_evokeds(evo_fname, condition=cond_name, baseline=baseline)
        except:
            print('No evoked data with the condition {}'.format(cond_name))
            evoked = None
    else:
        evo_cond = get_cond_fname(evo_fname, cond_name)
        if op.isfile(evo_cond):
            evoked = mne.read_evokeds(evo_cond, baseline=baseline)[0]
        else:
            print('No evoked file, trying to use epo file')
            if not op.isfile(epo_fname):
                print('No epochs were found!')
                return None
            if '{cond}' not in epo_fname:
                epochs = mne.read_epochs(epo_fname, apply_SSP_projection_vectors, add_eeg_ref)
                evoked = epochs[cond_name].average()
            else:
                epo_cond = get_cond_fname(epo_fname, cond_name)
                epochs = mne.read_epochs(epo_cond, apply_SSP_projection_vectors, add_eeg_ref)
                evoked = epochs.average()
            mne.write_evokeds(evo_cond, evoked)
    return evoked


def get_cond_fname(fname, cond, **kargs):
    if '{cond}' in fname:
        kargs['cond'] = cond
    return fname.format(**kargs)


def calc_sub_cortical_activity(events, sub_corticals_codes_file=None, inverse_method='dSPM', pick_ori=None,
        evoked=None, epochs=None, regions=None, inv_include_hemis=True, n_dipoles=0):
    lut = utils.read_freesurfer_lookup_table()
    evoked_given = not evoked is None
    epochs_given = not epochs is None
    if regions is not None:
        sub_corticals = utils.lut_labels_to_indices(regions, lut)
    else:
        if not sub_corticals_codes_file is None:
            sub_corticals = utils.read_sub_corticals_code_file(sub_corticals_codes_file)
    if len(sub_corticals) == 0:
        return

    snr = 3.0
    lambda2 = 1.0 / snr ** 2
    global_evo = False
    if '{cond}' not in EVO:
        global_evo = True
        if not evoked_given:
            evoked = {event:mne.read_evokeds(EVO, condition=event, baseline=(None, 0)) for event in events.keys()}
        inv_fname = INV_SUB if len(sub_corticals) > 1 else INV_X.format(region=regions[0])
        if op.isfile(inv_fname):
            inverse_operator = read_inverse_operator(inv_fname)
        else:
            print('The Inverse operator file does not exist! {}'.format(inv_fname))
            return False

    for event in events.keys():
        sub_corticals_activity = {}
        if not global_evo:
            evo = get_cond_fname(EVO, event)
            if not evoked_given:
                evoked = {event:mne.read_evokeds(evo, baseline=(None, 0))[0]}
            inv_fname = get_cond_fname(INV_SUB, event) if len(sub_corticals) > 1 else \
                get_cond_fname(INV_X, event, region=regions[0])
            inverse_operator = read_inverse_operator(inv_fname)
        if inverse_method in ['lcmv', 'dics', 'rap_music']:
            if not epochs_given:
                epochs = mne.read_epochs(get_cond_fname(EPO, event))
            fwd_fname = get_cond_fname(FWD_SUB, event) if len(sub_corticals) > 1 else get_cond_fname(FWD_X, event, region=regions[0])
            forward = mne.read_forward_solution(fwd_fname)
        if inverse_method in ['lcmv', 'rap_music']:
            noise_cov = calc_cov(get_cond_fname(NOISE_COV, event), event, epochs, None, 0)

        if inverse_method == 'lcmv':
            from mne.beamformer import lcmv
            data_cov = calc_cov(get_cond_fname(DATA_COV, event), event, epochs, 0.0, 1.0)
            # pick_ori = None | 'normal' | 'max-power'
            stc = lcmv(evoked[event], forward, noise_cov, data_cov, reg=0.01, pick_ori='max-power')

        elif inverse_method == 'dics':
            from mne.beamformer import dics
            from mne.time_frequency import compute_epochs_csd
            data_csd = compute_epochs_csd(epochs, mode='multitaper', tmin=0.0, tmax=2.0,
                fmin=6, fmax=10)
            noise_csd = compute_epochs_csd(epochs, mode='multitaper', tmin=-0.5, tmax=0.0,
                fmin=6, fmax=10)
            stc = dics(evoked[event], forward, noise_csd, data_csd)

        elif inverse_method == 'rap_music':
            if len(sub_corticals) > 1:
                print('Need to do more work here for len(sub_corticals) > 1')
            else:
                from mne.beamformer import rap_music
                noise_cov = mne.compute_covariance(epochs.crop(None, 0, copy=True))
                n_dipoles = len(sub_corticals) if n_dipoles==0 else n_dipoles
                dipoles = rap_music(evoked[event], forward, noise_cov, n_dipoles=n_dipoles,
                    return_residual=False, verbose=True)
                for sub_cortical_ind, sub_cortical_code in enumerate(sub_corticals):
                    amp = dipoles[sub_cortical_ind].amplitude
                    amp = amp.reshape((1, amp.shape[0]))
                    sub_corticals_activity[sub_cortical_code] = amp
                    print(set([tuple(dipoles[sub_cortical_ind].pos[t]) for t in range(len(dipoles[sub_cortical_ind].times))]))
        else:
            stc = apply_inverse(evoked[event], inverse_operator, lambda2, inverse_method, pick_ori=pick_ori)

        if inverse_method not in ['rap_music']:
            # todo: maybe to flip?
            # stc.extract_label_time_course(label, src, mode='mean_flip')
            read_vertices_from = len(stc.vertices[0])+len(stc.vertices[1]) if inv_include_hemis else 0
            # 2 becasue the first two are the hemispheres
            sub_cortical_indices_shift = 2 if inv_include_hemis else 0
            for sub_cortical_ind, sub_cortical_code in enumerate(sub_corticals):
                if len(sub_corticals) > 1:
                    vertices_to_read = len(stc.vertices[sub_cortical_ind + sub_cortical_indices_shift])
                else:
                    vertices_to_read = len(stc.vertices)
                sub_corticals_activity[sub_cortical_code] = stc.data[
                    read_vertices_from: read_vertices_from + vertices_to_read]
                read_vertices_from += vertices_to_read

        subs_fol = op.join(SUBJECT_MEG_FOLDER, 'subcorticals', inverse_method)
        utils.make_dir(subs_fol)
        for sub_cortical_code, activity in sub_corticals_activity.items():
            sub_cortical, _ = utils.get_numeric_index_to_label(sub_cortical_code, lut)
            np.save(op.join(subs_fol, '{}-{}-{}.npy'.format(event, sub_cortical, inverse_method)), activity.mean(0))
            np.save(op.join(subs_fol, '{}-{}-{}-all-vertices.npy'.format(event, sub_cortical, inverse_method)), activity.mean(0))
            # np.save(op.join(subs_fol, '{}-{}-{}'.format(event, sub_cortical, inverse_method)), activity.mean(0))
            # np.save(op.join(subs_fol, '{}-{}-{}-all-vertices'.format(event, sub_cortical, inverse_method)), activity)


def calc_cov(cov_fname, cond, epochs, from_t, to_t, method='empirical', overwrite=False):
    cov_cond_fname = get_cond_fname(cov_fname, cond)
    if not op.isfile(cov_cond_fname) or overwrite:
        cov = mne.compute_covariance(epochs.crop(from_t, to_t), method=method)
        cov.save(cov_cond_fname)
    else:
        cov = mne.read_cov(cov_cond_fname)
    return cov


def calc_csd(csd_fname, cond, epochs, from_t, to_t, mode='multitaper', fmin=6, fmax=10, overwrite=False):
    from mne.time_frequency import compute_epochs_csd
    csd_cond_fname = get_cond_fname(csd_fname, cond)
    if not op.isfile(csd_cond_fname) or overwrite:
        csd = compute_epochs_csd(epochs, mode, tmin=from_t, tmax=to_t, fmin=fmin, fmax=fmax)
        utils.save(csd, csd_cond_fname)
    else:
        csd = utils.load(csd_cond_fname)
    return csd


def calc_specific_subcortical_activity(region, inverse_methods, events, plot_all_vertices=False,
        overwrite_fwd=False, overwrite_inv=False, overwrite_activity=False, inv_loose=0.2, inv_depth=0.8):
    if not x_opertor_exists(FWD_X, region, events) or overwrite_fwd:
        make_forward_solution_to_specific_subcortrical(events, region)
    if not x_opertor_exists(INV_X, region, events) or overwrite_inv:
        calc_inverse_operator(events, '', '', '', '', '', '', inv_loose, inv_depth, False, False, False, False, False,
                              True, region=region)
    for inverse_method in inverse_methods:
        files_exist = np.all([op.isfile(op.join(SUBJECT_MEG_FOLDER, 'subcorticals',
            '{}-{}-{}.npy'.format(cond, region, inverse_method))) for cond in events.keys()])
        if not files_exist or overwrite_activity:
            calc_sub_cortical_activity(events, None, inverse_method=inverse_method, pick_ori=pick_ori,
                regions=[region], inv_include_hemis=False)
        plot_sub_cortical_activity(events, None, inverse_method=inverse_method,
            regions=[region], all_vertices=plot_all_vertices)


def x_opertor_exists(operator, region, events):
    if not '{cond}' in operator:
        exists = op.isfile(op.join(SUBJECT_MEG_FOLDER, operator.format(region=region)))
    else:
        exists = np.all([op.isfile(op.join(SUBJECT_MEG_FOLDER,
            get_cond_fname(operator, cond, region=region))) for cond in events.keys()])
    return exists


def plot_sub_cortical_activity(events, sub_corticals_codes_file, inverse_method='dSPM', regions=None, all_vertices=False):
    lut = utils.read_freesurfer_lookup_table()
    if sub_corticals_codes_file is None:
        sub_corticals = regions
    else:
        sub_corticals = utils.read_sub_corticals_code_file(sub_corticals_codes_file)
    for label in sub_corticals:
        sub_cortical, _ = utils.get_numeric_index_to_label(label, lut)
        print(sub_cortical)
        activity = {}
        f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
        fig_name = '{} ({}): {}, {}, contrast {}'.format(sub_cortical, inverse_method,
            events.keys()[0], events.keys()[1], '-all-vertices' if all_vertices else '')
        ax1.set_title(fig_name)
        for event, ax in zip(events.keys(), [ax1, ax2]):
            data_file_name = '{}-{}-{}{}.npy'.format(event, sub_cortical, inverse_method, '-all-vertices' if all_vertices else '')
            activity[event] = np.load(op.join(SUBJECT_MEG_FOLDER, 'subcorticals', data_file_name))
            ax.plot(activity[event].T)
        ax3.plot(activity[events.keys()[0]].T - activity[events.keys()[1]].T)
        f.subplots_adjust(hspace=0.2)
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        fol = op.join(SUBJECT_MEG_FOLDER, 'figures')
        if not op.isdir(fol):
            os.mkdir(fol)
        plt.savefig(op.join(fol, fig_name))
        plt.close()


def save_subcortical_activity_to_blender(sub_corticals_codes_file, events, stat, inverse_method='dSPM',
        norm_by_percentile=True, norm_percs=(1,99), do_plot=False):
    if do_plot:
        plt.figure()

    sub_corticals = utils.read_sub_corticals_code_file(sub_corticals_codes_file)
    first_time = True
    names_for_blender = []
    lut = utils.read_freesurfer_lookup_table()
    for ind, sub_cortical_ind in enumerate(sub_corticals):
        sub_cortical_name, _ = utils.get_numeric_index_to_label(sub_cortical_ind, lut)
        names_for_blender.append(sub_cortical_name)
        for cond_id, cond in enumerate(events.keys()):
            data_fname = op.join(SUBJECT_MEG_FOLDER, 'subcorticals', inverse_method,
                '{}-{}-{}.npy'.format(cond, sub_cortical_name, inverse_method))
            if op.isfile(data_fname):
                x = np.load(op.join(SUBJECT_MEG_FOLDER, 'subcorticals', inverse_method,
                    '{}-{}-{}.npy'.format(cond, sub_cortical_name, inverse_method)))
                if first_time:
                    first_time = False
                    T = len(x)
                    data = np.zeros((len(sub_corticals), T, len(events.keys())))
                data[ind, :, cond_id] = x[:T]
            else:
                print('The file {} does not exist!'.format(data_fname))
                return
        if do_plot:
            plt.plot(data[ind, :, 0] - data[ind, :, 1], label='{}-{} {}'.format(
                events.keys()[0], events.keys()[1], sub_cortical_name))

    stat_data = utils.calc_stat_data(data, stat)
    # Normalize
    # todo: I don't think we should normalize stat_data
    # stat_data = utils.normalize_data(stat_data, norm_by_percentile, norm_percs)
    data = utils.normalize_data(data, norm_by_percentile, norm_percs)
    data_max, data_min = utils.get_data_max_min(stat_data, norm_by_percentile, norm_percs, symmetric=True)
    # if stat == STAT_AVG:
    #     colors = utils.mat_to_colors(stat_data, data_min, data_max, colorsMap=colors_map)
    # elif stat == STAT_DIFF:
    #     data_minmax = max(map(abs, [data_max, data_min]))
    #     colors = utils.mat_to_colors_two_colors_maps(stat_data, threshold=threshold,
    #         x_max=data_minmax,x_min = -data_minmax, cm_big=cm_big, cm_small=cm_small,
    #         default_val=1, flip_cm_big=flip_cm_big, flip_cm_small=flip_cm_small)

    np.savez(op.join(MMVT_SUBJECT_FOLDER, 'subcortical_meg_activity'), data=data,
        names=names_for_blender, conditions=list(events.keys()), data_minmax=data_max)

    if do_plot:
        plt.legend()
        plt.show()


# def plotActivationTS(stcs_meg):
#     plt.close('all')
#     plt.figure(figsize=(8, 6))
#     name = 'MEG'
#     stc = stcs_meg
#     plt.plot(1e3 * stc.times, stc.data[::150, :].T)
#     plt.ylabel('%s\ndSPM value' % str.upper(name))
#     plt.xlabel('time (ms)')
#     plt.show()
#
#
# def plot3DActivity(stc=None):
#     if (stc is None):
#         stc = read_source_estimate(STC)
#     # Plot brain in 3D with PySurfer if available. Note that the subject name
#     # is already known by the SourceEstimate stc object.
#     brain = stc.plot(surface='inflated', hemi='rh', subjects_dir=SUBJECT_MRI_DIR, subject=SUBJECT)
#     brain.scale_data_colormap(fmin=8, fmid=12, fmax=15, transparent=True)
#     brain.show_view('lateral')
#
#     # use peak getter to move vizualization to the time point of the peak
#     vertno_max, time_idx = stc.get_peak(hemi='rh', time_as_index=True)
#     brain.set_data_time_index(time_idx)
#
#     # draw marker at maximum peaking vertex
#     brain.add_foci(vertno_max, coords_as_verts=True, hemi='rh', color='blue',
#                     scale_factor=0.6)
# #     brain.save_movie(SUBJECT_FOLDER)
#     brain.save_image(getPngName('dSPM_map'))


# def morphTOTlrc(method='MNE'):
#     # use dSPM method (could also be MNE or sLORETA)
#     epoches = loadEpoches()
#     epo1 = epoches[CONDS[0]]
#     epo2 = epoches[CONDS[1]]
#
#     snr = 3.0
#     lambda2 = 1.0 / snr ** 2
#
#     inverse_operator = read_inverse_operator(INV)
#
#     #    Let's average and compute inverse, resampling to speed things up
#     evoked1 = epo1.average()
#     evoked1.resample(50)
#     condition1 = apply_inverse(evoked1, inverse_operator, lambda2, method)
#     evoked2 = epo2.average()
#     evoked2.resample(50)
#     condition2 = apply_inverse(evoked2, inverse_operator, lambda2, method)
#
#     cond1tlrc = mne.morph_data(SUBJECT, 'fsaverage', condition1, subjects_dir=SUBJECTS_DIR, n_jobs=4)
#     cond2tlrc = mne.morph_data(SUBJECT, 'fsaverage', condition2, subjects_dir=SUBJECTS_DIR, n_jobs=4)
#     cond1tlrc.save(op.join(SUBJECT_FOLDER, '{}_tlrc_{}'.format(method, CONDS[0])))
#     cond2tlrc.save(op.join(SUBJECT_FOLDER, '{}_tlrc_{}'.format(method, CONDS[1])))


# def morph_stc(subject_to, cond='all', grade=None, n_jobs=6, inverse_method='dSPM'):
#     stc = mne.read_source_estimate(STC.format(cond, inverse_method))
#     vertices_to = mne.grade_to_vertices(subject_to, grade=grade)
#     stc_to = mne.morph_data(SUBJECT, subject_to, stc, n_jobs=n_jobs, grade=vertices_to)
#     fol_to = op.join(MEG_DIR, TASK, subject_to)
#     if not op.isdir(fol_to):
#         os.mkdir(fol_to)
#     stc_to.save(STC_MORPH.format(subject_to, cond, inverse_method))


def calc_stc_for_all_vertices(stc, subject='', morph_to_subject='', n_jobs=6):
    subject = MRI_SUBJECT if subject == '' else MRI_SUBJECT
    morph_to_subject = subject if morph_to_subject == '' else morph_to_subject
    grade = None if morph_to_subject != 'fsaverage' else 5
    vertices_to = mne.grade_to_vertices(morph_to_subject, grade)
    return mne.morph_data(subject, morph_to_subject, stc, n_jobs=n_jobs, grade=vertices_to)


# def create_stc_t(stc, t, subject=''):
#     from mne import SourceEstimate
#     subject = MRI_SUBJECT if subject == '' else MRI_SUBJECT
#     data = np.concatenate([stc.lh_data[:, t:t + 1], stc.rh_data[:, t:t + 1]])
#     vertices = [stc.lh_vertno, stc.rh_vertno]
#     stc_t = SourceEstimate(data, vertices, stc.tmin + t * stc.tstep, stc.tstep, subject=subject, verbose=stc.verbose)
#     return stc_t


def create_stc_t(stc, t, subject=''):
    subject = MRI_SUBJECT if subject == '' else MRI_SUBJECT
    C = max([stc.rh_data.shape[0], stc.lh_data.shape[0]])
    stc_lh_data = stc.lh_data[:, t:t + 1] if stc.lh_data.shape[0] > 0 else np.zeros((C, 1))
    stc_rh_data = stc.rh_data[:, t:t + 1] if stc.rh_data.shape[0] > 0 else np.zeros((C, 1))
    data = np.concatenate([stc_lh_data, stc_rh_data])
    vertno = max([len(stc.lh_vertno), len(stc.rh_vertno)])
    lh_vertno = stc.lh_vertno if len(stc.lh_vertno) > 0 else np.arange(0, vertno)
    rh_vertno = stc.rh_vertno if len(stc.rh_vertno) > 0 else np.arange(0, vertno) + max(lh_vertno) + 1
    vertices = [lh_vertno, rh_vertno]
    stc_t = mne.SourceEstimate(data, vertices, stc.tmin + t * stc.tstep, stc.tstep, subject=subject, verbose=stc.verbose)
    return stc_t


def create_stc_t_from_data(subject, rh_data, lh_data, tstep=0.001):
    lh_vertno, rh_vertno = len(lh_data), len(rh_data)
    data = np.concatenate([lh_data, rh_data])
    if np.max(data) > 1e-6:
        data /= np.power(10, 9)
    data = np.reshape(data, (data.shape[0], 1))
    rh_verts, _ = utils.read_pial(subject, MMVT_DIR, 'rh')
    lh_verts, _ = utils.read_pial(subject, MMVT_DIR, 'lh')
    lh_vertno = np.arange(0, lh_vertno)
    rh_vertno = np.arange(0, rh_vertno) + max(lh_vertno) + 1
    vertices = [lh_vertno, rh_vertno]
    stc_t = mne.SourceEstimate(data, vertices, 0, tstep, subject=subject)
    return stc_t


@utils.files_needed({'surf': ['lh.sphere.reg', 'rh.sphere.reg']})
def morph_stc(subject, events, morph_to_subject, inverse_method='dSPM', grade=5, smoothing_iterations=None,
              overwrite=False, n_jobs=6):
    ret = True
    for ind, cond in enumerate(events.keys()):
        output_fname = STC_HEMI_SAVE.format(cond=cond, method=inverse_method).replace(SUBJECT, morph_to_subject)
        output_fname = op.join(SUBJECT_MEG_FOLDER, cond).replace(SUBJECT, morph_to_subject)
        if utils.both_hemi_files_exist('{}-{}.stc'.format(output_fname, '{hemi}')) and not overwrite:
            continue
        utils.make_dir(utils.get_parent_fol(output_fname))
        _morph_stc(subject, morph_to_subject, STC_HEMI.format(cond=cond, method=inverse_method, hemi='rh'),
                   output_fname, grade=grade, smooth=smoothing_iterations, n_jobs=n_jobs)
        ret = ret and utils.both_hemi_files_exist(output_fname)
    diff_template = op.join(SUBJECT_MEG_FOLDER, '{cond}-{hemi}.stc').replace(SUBJECT, morph_to_subject)
    calc_stc_diff_both_hemis(events, diff_template, inverse_method, overwrite)
    return ret


def _morph_stc(from_subject, to_subject, stc_fname, output_fname, grade=5, smooth=None, n_jobs=6):
    if op.isfile(stc_fname):
        stc = mne.read_source_estimate(stc_fname)
        stc_morphed = mne.morph_data(from_subject, to_subject, stc, grade=grade, smooth=smooth, n_jobs=n_jobs)
        stc_morphed.save(output_fname)
        print('Morphed stc file was saves in {}'.format(output_fname))
    else:
        print("Can't find {}!".format(stc_fname))


# @utils.timeit
def smooth_stc(events, stcs_conds=None, inverse_method='dSPM', t=-1, morph_to_subject='', n_jobs=6):
    try:
        stcs = {}
        for ind, cond in enumerate(events.keys()):
            output_fname = STC_HEMI_SMOOTH_SAVE.format(cond=cond, method=inverse_method)
            if morph_to_subject != '':
                output_fname = '{}-{}'.format(output_fname, morph_to_subject)
            if stcs_conds is not None:
                stc = stcs_conds[cond]
            else:
                # Can read only for the 'rh', it'll also read the second file for 'lh'. Strange...
                stc = mne.read_source_estimate(STC_HEMI.format(cond=cond, method=inverse_method, hemi='rh'))
            if t != -1:
                stc = create_stc_t(stc, t)
                output_fname = '{}-t{}'.format(output_fname, t)
            stc_smooth = calc_stc_for_all_vertices(stc, MRI_SUBJECT, morph_to_subject, n_jobs)
            check_stc_with_ply(stc_smooth, cond, morph_to_subject)
            stc_smooth.save(output_fname)
            stcs[cond] = stc_smooth
        flag = True
    except:
        print(traceback.format_exc())
        print('Error in calculating inv for {}'.format(cond))
        flag = False

    return flag, stcs


def check_stc_with_ply(stc, cond_name='', subject=''):
    mmvt_surf_fol = op.join(MMVT_DIR, MRI_SUBJECT if subject == '' else subject, 'surf')
    verts = {}
    for hemi in HEMIS:
        stc_vertices = stc.rh_vertno if hemi=='rh' else stc.lh_vertno
        print('{} {} stc vertices: {}'.format(hemi, cond_name, len(stc_vertices)))
        verts[hemi], _ = utils.read_ply_file(op.join(mmvt_surf_fol, '{}.pial.ply'.format(hemi)))
        print('{} {} ply vertices: {}'.format(hemi, cond_name, verts[hemi].shape[0]))
        if len(stc_vertices) != verts[hemi].shape[0]:
            raise Exception('check_stc_with_ply: Wrong number of vertices!')
    return verts


def save_activity_map(events, stat, stcs_conds=None, inverse_method='dSPM', smoothed_stc=True, morph_to_subject='',
                      stc_t=-1, norm_by_percentile=False, norm_percs=(1,99), plot_cb=False):
    try:
        if stat not in [STAT_DIFF, STAT_AVG]:
            raise Exception('stat not in [STAT_DIFF, STAT_AVG]!')
        stcs = get_stat_stc_over_conditions(
            events, stat, stcs_conds, inverse_method, smoothed_stc, morph_to_subject, stc_t)
        if stc_t == -1:
            save_activity_map_minmax(stcs, events, stat, stcs_conds, inverse_method, morph_to_subject,
                                     norm_by_percentile, norm_percs, plot_cb)
        subject = MRI_SUBJECT if morph_to_subject == '' else morph_to_subject
        for hemi in HEMIS:
            verts, faces = utils.read_pial(subject, MMVT_DIR, hemi)
            data = stcs[hemi]
            if verts.shape[0] != data.shape[0]:
                raise Exception('save_activity_map: wrong number of vertices!')
            else:
                print('Both {}.pial.ply and the stc file have {} vertices'.format(hemi, data.shape[0]))
            fol = '{}'.format(ACT.format(hemi))
            if morph_to_subject != '':
                fol = fol.replace(MRI_SUBJECT, morph_to_subject)
            if stc_t == -1:
                utils.delete_folder_files(fol)
                now = time.time()
                T = data.shape[1]
                for t in range(T):
                    utils.time_to_go(now, t, T, runs_num_to_print=10)
                    np.save(op.join(fol, 't{}'.format(t)), data[:, t])
            else:
                utils.make_dir(fol)
                np.save(op.join(fol, 't{}'.format(stc_t)), data)
        flag = True
    except:
        print(traceback.format_exc())
        print('Error in save_activity_map')
        flag = False
    return flag


def save_activity_map_minmax(stcs=None, events=None, stat=STAT_DIFF, stcs_conds=None, inverse_method='dSPM',
                             morph_to_subject='', norm_by_percentile=False, norm_percs=(1,99), plot_cb=False):
    from src.utils import color_maps_utils as cp
    from src.utils import figures_utils as figu

    subject = MRI_SUBJECT if morph_to_subject == '' else morph_to_subject
    output_fname = op.join(MMVT_DIR, subject, 'meg_activity_map_minmax.pkl')
    if stcs is None:
        if stat not in [STAT_DIFF, STAT_AVG]:
            raise Exception('stat not in [STAT_DIFF, STAT_AVG]!')
        stcs = get_stat_stc_over_conditions(events, stat, stcs_conds, inverse_method, False)
        if stcs is None and morph_to_subject != '':
            stcs = get_stat_stc_over_conditions(events, stat, stcs_conds, inverse_method, False, morph_to_subject)
        if stcs is None:
            print("Can't find the stc files!")
    data_max, data_min = utils.get_activity_max_min(stcs, norm_by_percentile, norm_percs)
    data_minmax = utils.get_max_abs(data_max, data_min)
    print('Saving data minmax, min: {}, max: {} to {}'.format(-data_minmax, data_minmax, output_fname))
    utils.save((-data_minmax, data_minmax), output_fname)
    if plot_cb:
        # todo: create colors map according to the parameters
        colors_map = cp.create_BuPu_YlOrRd_cm()
        figures_fol = op.join(MMVT_DIR, MRI_SUBJECT, 'figures')
        figu.plot_color_bar(data_minmax, -data_minmax, colors_map, fol=figures_fol)
    return op.isfile(output_fname)


def calc_activity_significance(events, inverse_method, stcs_conds=None):
    from mne import spatial_tris_connectivity, grade_to_tris
    from mne.stats import (spatio_temporal_cluster_1samp_test)
    from mne import bem
    from scipy import stats as stats

    paired_constart_fname = op.join(SUBJECT_MEG_FOLDER, 'paired_contrast.npy')
    n_subjects = 1
    if not op.isfile(paired_constart_fname):
        stc_template = STC_HEMI_SMOOTH
        if stcs_conds is None:
            stcs_conds = {}
            for cond_ind, cond in enumerate(events.keys()):
                # Reading only the rh, the lh will be read too
                print('Reading {}'.format(stc_template.format(cond=cond, method=inverse_method, hemi='lh')))
                stcs_conds[cond] = mne.read_source_estimate(stc_template.format(cond=cond, method=inverse_method, hemi='lh'))

        # Let's only deal with t > 0, cropping to reduce multiple comparisons
        for cond in events.keys():
            stcs_conds[cond].crop(0, None)
        conds = sorted(list(events.keys()))
        tmin = stcs_conds[conds[0]].tmin
        tstep = stcs_conds[conds[0]].tstep
        n_vertices_sample, n_times = stcs_conds[conds[0]].data.shape
        X = np.zeros((n_vertices_sample, n_times, n_subjects, 2))
        X[:, :, :, 0] += stcs_conds[conds[0]].data[:, :, np.newaxis]
        X[:, :, :, 1] += stcs_conds[conds[1]].data[:, :, np.newaxis]
        X = np.abs(X)  # only magnitude
        X = X[:, :, :, 0] - X[:, :, :, 1]  # make paired contrast
        #    Note that X needs to be a multi-dimensional array of shape
        #    samples (subjects) x time x space, so we permute dimensions
        X = np.transpose(X, [2, 1, 0])
        np.save(paired_constart_fname, X)
    else:
        X = np.load(paired_constart_fname)

    #    To use an algorithm optimized for spatio-temporal clustering, we
    #    just pass the spatial connectivity matrix (instead of spatio-temporal)
    print('Computing connectivity.')
    # tris = get_subject_tris()
    connectivity = None # spatial_tris_connectivity(tris)
    #    Now let's actually do the clustering. This can take a long time...
    #    Here we set the threshold quite high to reduce computation.
    p_threshold = 0.2
    t_threshold = -stats.distributions.t.ppf(p_threshold / 2., 10 - 1)
    print('Clustering.')
    T_obs, clusters, cluster_p_values, H0 = \
        spatio_temporal_cluster_1samp_test(X, connectivity=connectivity, n_jobs=6,
            threshold=t_threshold)
    #    Now select the clusters that are sig. at p < 0.05 (note that this value
    #    is multiple-comparisons corrected).
    good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
    # utils.save((clu, good_cluster_inds), op.join(SUBJECT_MEG_FOLDER, 'spatio_temporal_ttest.npy'))
    np.savez(op.join(SUBJECT_MEG_FOLDER, 'spatio_temporal_ttest'), T_obs=T_obs, clusters=clusters,
             cluster_p_values=cluster_p_values, H0=H0, good_cluster_inds=good_cluster_inds)
    print('good_cluster_inds: {}'.format(good_cluster_inds))


def get_subject_tris():
    from mne import read_surface
    _, tris_lh = read_surface(op.join(SUBJECTS_MRI_DIR, MRI_SUBJECT, 'surf', 'lh.white'))
    _, tris_rh = read_surface(op.join(SUBJECTS_MRI_DIR, MRI_SUBJECT, 'surf', 'rh.white'))
    # tris =  [tris_lh, tris_rh]
    tris = np.vstack((tris_lh, tris_rh))
    return tris


def save_vertex_activity_map(events, stat, stcs_conds=None, inverse_method='dSPM', number_of_files=100):
    try:
        if stat not in [STAT_DIFF, STAT_AVG]:
            raise Exception('stat not in [STAT_DIFF, STAT_AVG]!')
        stcs = get_stat_stc_over_conditions(events, stat, stcs_conds, inverse_method, smoothed=True)
        for hemi in HEMIS:
            verts, faces = utils.read_pial(MRI_SUBJECT, MMVT_DIR, hemi)
            data = stcs[hemi]
            if verts.shape[0]!=data.shape[0]:
                raise Exception('save_vertex_activity_map: wrong number of vertices!')
            else:
                print('Both {}.pial.ply and the stc file have {} vertices'.format(hemi, data.shape[0]))

            data_hash = defaultdict(list)
            fol = '{}_verts'.format(ACT.format(hemi))
            utils.delete_folder_files(fol)
            look_up = np.zeros((data.shape[0], 2), dtype=np.int)
            for vert_ind in range(data.shape[0]):
                file_num = vert_ind % number_of_files
                data_hash[file_num].append(data[vert_ind, :])
                look_up[vert_ind] = [file_num, len(data_hash[file_num])-1]
                if vert_ind % 10000 == 0:
                    print('{}: {} out of {}'.format(hemi, vert_ind, data.shape[0]))

            np.save('{}_verts_lookup'.format(ACT.format(hemi)), look_up)
            for file_num in range(number_of_files):
                file_name = op.join(fol, str(file_num))
                x = np.array(data_hash[file_num])
                np.save(file_name, x)
        flag = True
    except:
        print(traceback.format_exc())
        print('Error in save_vertex_activity_map')
        flag = False
    return flag


def get_stat_stc_over_conditions(events, stat, stcs_conds=None, inverse_method='dSPM', smoothed=False,
                                 morph_to_subject='', stc_t=-1):
    stcs = {}
    stc_template = STC_HEMI if not smoothed else STC_HEMI_SMOOTH
    for cond_ind, cond in enumerate(events.keys()):
        if stcs_conds is None:
            # Reading only the rh, the lh will be read too
            input_fname = stc_template.format(cond=cond, method=inverse_method, hemi='lh')
            if morph_to_subject != '':
                input_fname = '{}-{}-lh.stc'.format(input_fname[:-len('-lh.stc')], morph_to_subject)
            if stc_t != -1:
                input_fname = '{}-t{}-lh.stc'.format(input_fname[:-len('-lh.stc')], stc_t)
            if op.isfile(input_fname):
                print('Reading {}'.format(input_fname))
                stc = mne.read_source_estimate(input_fname)
            else:
                print('No such file {}!'.format(input_fname))
                return None
        else:
            stc = stcs_conds[cond]
        for hemi in HEMIS:
            data = stc.rh_data if hemi == 'rh' else stc.lh_data
            if hemi not in stcs:
                stcs[hemi] = np.zeros((data.shape[0], data.shape[1], len(events)))
            stcs[hemi][:, :, cond_ind] = data
    for hemi in HEMIS:
        if stat == STAT_AVG:
            # Average over the conditions
            stcs[hemi] = stcs[hemi].mean(2)
        elif stat == STAT_DIFF:
            # Calc the diff of the conditions
            stcs[hemi] = np.squeeze(np.diff(stcs[hemi], axis=2))
        else:
            raise Exception('Wrong value for stat, should be STAT_AVG or STAT_DIFF')
    return stcs


def rename_activity_files():
    fol = '/homes/5/npeled/space3/MEG/ECR/mg79/activity_map_rh'
    files = glob.glob(op.join(fol, '*.npy'))
    for file in files:
        name = '{}.npy'.format(file.split('/')[-1].split('-')[0])
        os.rename(file, op.join(fol, name))


# def calc_labels_avg(parc, hemi, surf_name, stc=None):
#     if stc is None:
#         stc = mne.read_source_estimate(STC)
#     labels = mne.read_labels_from_annot(SUBJECT, parc, hemi, surf_name)
#     inverse_operator = read_inverse_operator(INV)
#     src = inverse_operator['src']
#
#     plt.close('all')
#     plt.figure()
#
#     for ind, label in enumerate(labels):
#         # stc_label = stc.in_label(label)
#         mean_flip = stc.extract_label_time_course(label, src, mode='mean_flip')
#         mean_flip = np.squeeze(mean_flip)
#         if ind==0:
#             labels_data = np.zeros((len(labels), len(mean_flip)))
#         labels_data[ind, :] = mean_flip
#         plt.plot(mean_flip, label=label.name)
#
#     np.savez(LBL.format('all'), data=labels_data, names=[l.name for l in labels])
#     plt.legend()
#     plt.xlabel('time (ms)')
#     plt.show()


def morph_labels_from_fsaverage(atlas='aparc250', fs_labels_fol='', sub_labels_fol='', n_jobs=6):
    lu.morph_labels_from_fsaverage(MRI_SUBJECT, SUBJECTS_MRI_DIR, MMVT_DIR, atlas, fs_labels_fol, sub_labels_fol, n_jobs)


def labels_to_annot(parc_name, labels_fol='', overwrite=True):
    ret = lu.labels_to_annot(MRI_SUBJECT, SUBJECTS_MRI_DIR, parc_name, labels_fol, overwrite)
    if ret:
        return op.join(SUBJECTS_MRI_DIR, MRI_SUBJECT, 'label', '{}.{}.annot'.format('{hemi}', parc_name))
    else:
        return None


def calc_single_trial_labels_per_condition(atlas, events, stcs, extract_modes=('mean_flip'), src=None):
    global_inverse_operator = False
    if '{cond}' not in INV:
        global_inverse_operator = True
        if src is None:
            inverse_operator = read_inverse_operator(INV)
            src = inverse_operator['src']

    for extract_mode in extract_modes:
        for (cond_name, cond_id), stc in zip(events.items(), stcs.values()):
            if not global_inverse_operator:
                if src is None:
                    inverse_operator = read_inverse_operator(INV.format(cond=cond_name))
                    src = inverse_operator['src']
            labels = lu.read_labels(MRI_SUBJECT, SUBJECTS_MRI_DIR, atlas)
            labels_ts = mne.extract_label_time_course(stcs[cond_name], labels, src, mode=extract_mode,
                                                      return_generator=False, allow_empty=True)
            np.save(op.join(SUBJECT_MEG_FOLDER, 'labels_ts_{}_{}'.format(cond_name, extract_mode)), np.array(labels_ts))


def get_stc_conds(events, inverse_method, stc_hemi_template):
    stcs = {}
    hemi = 'lh' # both will be loaded
    for cond in events.keys():
        stc_fname = stc_hemi_template.format(cond=cond, method=inverse_method, hemi=hemi)
        if not op.isfile(stc_fname):
            print("Can't find the stc file! {}".format(stc_fname))
            template = op.join(SUBJECT_MEG_FOLDER, '*-{}.stc'.format(hemi))
            stc_fname = utils.select_one_file(stcs, template=template, files_desc='STC', print_title=True)
            if not op.isfile(stc_fname):
                return None
        stcs[cond] = mne.read_source_estimate(stc_fname)
    return stcs


def calc_labels_avg_per_cluster(subject, atlas, events, stc_names, extract_method, labels_output_fname_template=''):
    if labels_output_fname_template == '':
        labels_output_fname_template = LBL
    labels_data = {hemi:{} for hemi in utils.HEMIS}
    labels_names = defaultdict(list)
    for hemi in utils.HEMIS:
        labels_data[hemi][extract_method] = None
    conditions = list(events.keys())
    for stc_ind, stc_name in enumerate(stc_names):
        clusters_fname = op.join(
            MMVT_DIR, subject, 'meg', 'clusters', 'clusters_labels_{}.pkl'.format(stc_name))
        clusters = utils.load(clusters_fname)
        cond, inv_method, lfreq, hfreq = utils.namebase(clusters_fname)[len('clusters_labels_'):].split('-')
        cond_id = conditions.index(cond)
        clusters_info = [info for info in clusters.values if info.label_data is not None]
        for hemi in utils.HEMIS:
            if labels_data[hemi][extract_method] is None:
                labels_data[hemi][extract_method] = np.zeros(
                    (len(clusters_info), len(clusters_info[0].label_data), len(stc_names), len(events)))
        for ind, info in enumerate(clusters_info):
            labels_data[info.hemi][extract_method][ind, :, cond_id, stc_ind] = info.label_data
            label_name = '{name}_max_{max:.2f}_size_{size}'.format(**info)
            labels_names[info.hemi].append(label_name)
    for hemi in utils.HEMIS:
        labels_output_fname = labels_output_fname_template.format(atlas, extract_method, hemi)
        lables_mmvt_fname = op.join(MMVT_DIR, MRI_SUBJECT, 'meg', op.basename(labels_output_fname))
        np.savez(labels_output_fname, data=labels_data[hemi][extract_method],
                 names=labels_names, conditions=conditions)
        shutil.copy(labels_output_fname, lables_mmvt_fname)


def calc_labels_avg_per_condition(atlas, hemi, events=None, surf_name='pial', labels_fol='', stcs=None, stcs_num={},
        extract_modes=['mean_flip'], positive=False, moving_average_win_size=0, labels_data_template='', src=None,
        factor=1, inv_fname='', fwd_usingMEG=True, fwd_usingEEG=True, read_only_from_annot=True, overwrite=False,
        do_plot=False, n_jobs=1):
    def _check_all_files_exist():
        return all([op.isfile(op.join(MMVT_DIR, MRI_SUBJECT, 'meg', op.basename(
            labels_data_template.format(atlas, em, hemi)))) for em in extract_modes])
    if labels_data_template == '':
        labels_data_template = LBL
    if _check_all_files_exist() and not overwrite:
        return True
    try:
        inv_fname = get_inv_fname(inv_fname, fwd_usingMEG, fwd_usingEEG)
        global_inverse_operator = False
        if events is None:
            events = dict(all=0)
        if '{cond}' not in inv_fname:
            if not op.isfile(inv_fname):
                print('No inverse operator found!')
                return False
            global_inverse_operator = True
            if src is None:
                inverse_operator = read_inverse_operator(inv_fname)
                src = inverse_operator['src']

        if do_plot:
            utils.make_dir(op.join(SUBJECT_MEG_FOLDER, 'figures'))

        # if stcs is None:
        #     stcs = get_stc_conds(events, hemi, inverse_method)
        conds_incdices = {cond_id:ind for ind, cond_id in zip(range(len(stcs)), events.values())}
        conditions = []
        labels_data = {}
        labels = lu.read_labels(MRI_SUBJECT, SUBJECTS_MRI_DIR, atlas, hemi=hemi, surf_name=surf_name,
                                labels_fol=labels_fol, read_only_from_annot=read_only_from_annot, n_jobs=n_jobs)
        for (cond_name, cond_id), stc_cond in zip(events.items(), stcs.values()):
            if do_plot:
                plt.figure()
            conditions.append(cond_name)
            if not global_inverse_operator:
                if src is None:
                    if not op.isfile(inv_fname.format(cond=cond_name)):
                        print('No inverse operator found!')
                        return False
                    inverse_operator = read_inverse_operator(inv_fname.format(cond=cond_name))
                    src = inverse_operator['src']
            if len(labels) == 0:
                print('No labels were found for {} atlas!'.format(atlas))
                return False

            if isinstance(stc_cond, types.GeneratorType):
                stc_cond_num = stcs_num[cond_name]
            else:
                stc_cond = [stc_cond]
                stc_cond_num = 1
            for stc_ind, stc in enumerate(stc_cond):
                for em in extract_modes:
                    for ind, label in enumerate(labels):
                        label_data = stc.extract_label_time_course(label, src, mode=em, allow_empty=True)
                        label_data = np.squeeze(label_data)
                        # Set flip to be always positive
                        # mean_flip *= np.sign(mean_flip[np.argmax(np.abs(mean_flip))])
                        if em not in labels_data:
                            T = len(stc.times)
                            labels_data[em] = np.zeros((len(labels), T, len(stcs), stc_cond_num))
                        labels_data[em][ind, :, conds_incdices[cond_id], stc_ind] = label_data
                        if do_plot:
                            plt.plot(labels_data[em][ind, :, conds_incdices[cond_id]], label=label.name)

            if do_plot:
                plt.xlabel('time (ms)')
                plt.title('{}: {} {}'.format(cond_name, hemi, atlas))
                plt.legend()
                # plt.show()
                plt.savefig(op.join(SUBJECT_MEG_FOLDER, 'figures', '{}: {} {}.png'.format(cond_name, hemi, atlas)))

        save_labels_data(labels_data, hemi, labels, atlas, conditions, extract_modes, labels_data_template, factor,
                         positive, moving_average_win_size)
        flag = _check_all_files_exist()
    except:
        print(traceback.format_exc())
        print('Error in calc_labels_avg_per_condition inv')
        flag = False
    return flag


def save_labels_data(labels_data, hemi, labels_names, atlas, conditions, extract_modes, labels_data_template, factor=1, positive=False, moving_average_win_size=0):
    if not isinstance(labels_names[0], str):
        labels_names = [utils.to_str(l.name) for l in labels_names]
    for em in extract_modes:
        labels_data[em] = labels_data[em].squeeze()
        if np.max(labels_data[em]) < 1e-4:
            labels_data[em] *= np.power(10, factor)
        if positive or moving_average_win_size > 0:
            labels_data[em] = utils.make_evoked_smooth_and_positive(labels_data[em], positive, moving_average_win_size)
        labels_output_fname = labels_data_template.format(atlas, em, hemi)  # if labels_output_fname_template == '' else \
        # lables_mmvt_fname = op.join(
        #     MMVT_DIR, MRI_SUBJECT, 'meg', op.basename(labels_data_template.format(atlas, em, hemi)))
        print('Saving to {}'.format(labels_output_fname))
        utils.make_dir(utils.get_parent_fol(labels_output_fname))
        np.savez(labels_output_fname, data=labels_data[em], names=labels_names, conditions=conditions)
        # shutil.copyfile(labels_output_fname, lables_mmvt_fname)


def read_sensors_layout(mri_subject, args, pick_meg=True, pick_eeg=False, overwrite_sensors=False,
                        trans_file=''):
    if pick_eeg and pick_meg or (not pick_meg and not pick_eeg):
        raise Exception('read_sensors_layout: You should pick only meg or eeg!')
    if not op.isfile(INFO):
        raw_fname, raw_exist = locating_meg_file(RAW, args.raw_template)
        if not raw_exist:
            print('No raw or raw info file!')
            return False
        raw = mne.io.read_raw_fif(raw_fname)
        info = raw.info
        utils.save(info, INFO)
    else:
        info = utils.load(INFO)
    picks = mne.io.pick.pick_types(info, meg=pick_meg, eeg=pick_eeg)
    sensors_pos = np.array([info['chs'][k]['loc'][:3] for k in picks])
    sensors_names = np.array([info['ch_names'][k] for k in picks])
    if 'Event' in sensors_names:
        event_ind = np.where(sensors_names == 'Event')[0]
        sensors_names = np.delete(sensors_names, event_ind)
        sensors_pos = np.delete(sensors_pos, event_ind)
    if pick_meg:
        utils.make_dir(op.join(MMVT_DIR, mri_subject, 'meg'))
        output_fname = op.join(MMVT_DIR, mri_subject, 'meg', 'meg_sensors_positions.npz')
    else:
        utils.make_dir(op.join(MMVT_DIR, mri_subject, 'eeg'))
        output_fname = op.join(MMVT_DIR, mri_subject, 'eeg', 'eeg_sensors_positions.npz')
    if op.isfile(output_fname) and not overwrite_sensors:
        return True
    ret = False
    if len(sensors_pos) > 0:
        trans_file = find_trans_file(trans_file, args.remote_subject_dir)
        if not op.isfile(trans_file):
            print('No trans files!')
        else:
            trans = mne.transforms.read_trans(trans_file)
            head_mri_t = mne.transforms._ensure_trans(trans, 'head', 'mri')
            sensors_pos = mne.transforms.apply_trans(head_mri_t, sensors_pos)
            sensors_pos *= 1000
            np.savez(output_fname, pos=sensors_pos, names=sensors_names)
            ret = True
    else:
        print('No sensors found!')
    return ret


def find_trans_file(trans_file, remote_subject_dir=''):
    trans_file = COR if trans_file == '' else trans_file
    if not op.isfile(trans_file):
        trans_files = glob.glob(op.join(SUBJECTS_MRI_DIR, MRI_SUBJECT, '**', '*COR*.fif'), recursive=True)
        if len(trans_files) == 0 and remote_subject_dir != '':
            trans_files = glob.glob(op.join(remote_subject_dir, '**', '*COR*.fif'), recursive=True)
        if len(trans_files) == 0:
            trans_files = glob.glob(op.join(utils.get_parent_fol(COR), '**', '*COR*.fif'), recursive=True)
        bem_trans_files = glob.glob(op.join(SUBJECTS_MRI_DIR, MRI_SUBJECT, 'bem', '*-head.fif'))
        if len(bem_trans_files):
            trans_files += bem_trans_files
        else:
            trans_files += glob.glob(op.join(remote_subject_dir, 'bem', '*-head.fif'), recursive=True)
        ok_trans_files = []
        for trans_file in trans_files:
            try:
                trans = mne.transforms.read_trans(trans_file)
                head_mri_t = mne.transforms._ensure_trans(trans, 'head', 'mri')
                if trans is not None and not np.all(head_mri_t['trans'] == np.eye(4)):
                    ok_trans_files.append(trans_file)
            except:
                pass

        trans_file = utils.select_one_file(
            ok_trans_files, template='*COR*.fif', files_desc='MRI-Head transformation',
            file_func=lambda fname:read_trans(fname))
    return trans_file


def read_trans(fname):
    try:
        print(mne.transforms.read_trans(fname))
    except:
        print('Not a trans file')

# def plot_labels_data(plot_each_label=False):
#     plt.close('all')
#     for hemi in HEMIS:
#         plt.figure()
#         d = np.load(LBL.format(hemi))
#         for cond_id, cond_name in enumerate(d['conditions']):
#             figures_fol = op.join(SUBJECT_MEG_FOLDER, 'figures', hemi, cond_name)
#             if not op.isdir(figures_fol):
#                 os.makedirs(figures_fol)
#             for name, data in zip(d['names'], d['data'][:,:,cond_id]):
#                 if plot_each_label:
#                     plt.figure()
#                 plt.plot(data, label=name)
#                 if plot_each_label:
#                     plt.title('{}: {} {}'.format(cond_name, hemi, name))
#                     plt.xlabel('time (ms)')
#                     plt.savefig(op.join(figures_fol, '{}.jpg'.format(name)))
#                     plt.close()
#             # plt.legend()
#             if not plot_each_label:
#                 plt.title('{}: {}'.format(cond_name, hemi))
#                 plt.xlabel('time (ms)')
#                 plt.show()


def check_both_hemi_in_stc(events, inverse_method):
    for ind, cond in enumerate(events.keys()):
        stcs = {}
        for hemi in HEMIS:
            stcs[hemi] = mne.read_source_estimate(STC_HEMI.format(cond=cond, method=inverse_method, hemi=hemi))
        print(np.all(stcs['rh'].rh_data == stcs['lh'].rh_data))
        print(np.all(stcs['rh'].lh_data == stcs['lh'].lh_data))


def check_labels():
    data, names = [], []
    for hemi in HEMIS:
        # todo: What?
        f = np.load('/homes/5/npeled/space3/visualization_blender/fsaverage/pp003_Fear/labels_data_{}.npz'.format(hemi))
        data.append(f['data'])
        names.extend(f['names'])
    d = np.vstack((d for d in data))
    plt.plot((d[:,:,0]-d[:,:,1]).T)
    t_range = range(0, 1000)
    dd = d[:, t_range, 0]- d[:, t_range, 1]
    print(dd.shape)
    dd = np.sqrt(np.sum(np.power(dd, 2), 1))
    print(dd)
    objects_to_filtter_in = np.argsort(dd)[::-1][:2]
    print(objects_to_filtter_in)
    print(dd[objects_to_filtter_in])
    return objects_to_filtter_in,names


def test_labels_coloring(subject, atlas):
    T = 2500
    labels_fnames = glob.glob(op.join(SUBJECTS_MRI_DIR, subject, 'label', atlas, '*.label'))
    labels_names = defaultdict(list)
    for label_fname in labels_fnames:
        label = mne.read_label(label_fname)
        labels_names[label.hemi].append(label.name)

    for hemi in HEMIS:
        L = len(labels_names[hemi])
        data, data_no_t = np.zeros((L, T)), np.zeros((L))
        for ind in range(L):
            data[ind, :] = (np.sin(np.arange(T) / 100 - np.random.rand(1) * 100) +
                np.random.randn(T) / 100) * np.random.rand(1)
            data_no_t[ind] = data[ind, 0]
        colors = utils.mat_to_colors(data)
        colors_no_t = utils.arr_to_colors(data_no_t)[:, :3]
        np.savez(op.join(MMVT_SUBJECT_FOLDER, 'meg', 'meg_labels_coloring_{}.npz'.format(hemi)),
            data=data, colors=colors, names=labels_names[hemi])
        np.savez(op.join(MMVT_SUBJECT_FOLDER, 'meg', 'meg_labels_coloring_no_t{}.npz'.format(hemi)),
            data=data_no_t, colors=colors_no_t, names=labels_names[hemi])
        # plt.plot(range(T), data.T)
        # plt.show()


def misc():
    # check_labels()
    # Morph and move to mg79
    # morph_stc('mg79', 'all')
    # initGlobals('mg79')
    # readLabelsData()
    # plot3DActivity()
    # plot3DActivity()
    # morphTOTlrc()
    # stc = read_source_estimate(STC)
    # plot3DActivity(stc)
    # permuationTest()
    # check_both_hemi_in_stc(events)
    # lut = utils.read_freesurfer_lookup_table(FREESURFER_HOME)
    pass


def get_fname_format_args(args):
    return get_fname_format(
        args.task, args.fname_format,args.fname_format_cond, args.conditions, args.get_task_defaults)


def get_fname_format(task, fname_format='', fname_format_cond='', args_conditions=('all'), get_task_defaults=True):
    if get_task_defaults:
        if task == 'MSIT':
            # fname_format = '{subject}_msit_interference_1-15-{file_type}.fif' # .format(subject, fname (like 'inv'))
            # fname_format_cond = '{subject}_msit_{cleaning_method}_{contrast}_{cond}_1-15-{ana_type}.{file_type}'
            # fname_format = '{subject}_msit_{cleaning_method}_{contrast}_1-15-{ana_type}.{file_type}'
            fname_format_cond = '{subject}_msit_{cleaning_method}_{contrast}_{cond}_1-15-{ana_type}.{file_type}'
            fname_format = '{subject}_msit_{cleaning_method}_{contrast}_1-15-{ana_type}.{file_type}'
            conditions = dict(interference=1, neutral=2) # dict(congruent=1, incongruent=2), events = dict(Fear=1, Happy=2)
            # event_digit = 1
        elif task == 'ECR':
            fname_format_cond = '{subject}_ecr_{cond}_15-{ana_type}.{file_type}'
            fname_format = '{subject}_ecr_15-{ana_type}.{file_type}'
            # conditions = dict(Fear=1, Happy=2) # or dict(congruent=1, incongruent=2)
            conditions = dict(C=1, I=2)
            # event_digit = 3
        elif task == 'ARC':
            fname_format_cond = '{subject}_arc_rer_{cleaning_method}_{cond}-{ana_type}.{file_type}'
            fname_format = '{subject}_arc_rer_{cleaning_method}-{ana_type}.{file_type}'
            conditions = dict(low_risk=1, med_risk=2, high_risk=3)
        elif task == 'rest':
            fname_format = fname_format_cond = '{subject}_{cleaning_method}-rest-{ana_type}.{file_type}'
            conditions = dict(rest=1)
    else:
        if fname_format == '' or fname_format_cond == '':
            raise Exception('Empty fname_format and/or fname_format_cond!')
        # raise Exception('Unkown task! Known tasks are MSIT/ECR/ARC')
        # print('Unkown task! Known tasks are MSIT/ECR/ARC.')
        conditions = dict((cond_name, cond_id + 1) for cond_id, cond_name in enumerate(args_conditions))
    return fname_format, fname_format_cond, conditions


def prepare_subject_folder(subject, remote_subject_dir, local_subjects_dir, necessary_files, sftp_args):
    return utils.prepare_subject_folder(
        necessary_files, subject, remote_subject_dir, local_subjects_dir,
        sftp_args.sftp, sftp_args.sftp_username, sftp_args.sftp_domain, sftp_args.sftp_password,
        False, sftp_args.print_traceback, local_subject_dir=SUBJECT_MEG_FOLDER)


def get_meg_files(subject, necessary_fnames, args, events=None):
    fnames = []
    events_keys = events.keys() if events is not None and isinstance(events, dict) else ['all']
    for necessary_fname in necessary_fnames:
        fname = os.path.basename(necessary_fname)
        if '{cond}' in fname:
            fnames.extend([get_cond_fname(fname, event) for event in events_keys])
        else:
            fnames.append(fname)
    # local_fol = op.join(MEG_DIR, args.task)
    prepare_subject_folder(subject, args.remote_subject_meg_dir, MEG_DIR, {'.': fnames}, args)


def calc_fwd_inv_wrapper(subject, args, conditions=None, flags={}, mri_subject=''):
    if not utils.should_run(args, 'make_forward_solution') and not utils.should_run(args, 'calc_inverse_operator'):
        return flags
    if mri_subject == '':
        mri_subject = subject
    raw_fname = get_raw_fname(args.raw_fname)
    inv_fname = get_inv_fname(args.inv_fname, args.fwd_usingMEG, args.fwd_usingEEG)
    fwd_fname = get_fwd_fname(args.fwd_fname, args.fwd_usingMEG, args.fwd_usingEEG)
    epo_fname = get_epo_fname(args.epo_fname)
    evo_fname = get_evo_fname(args.evo_fname)
    noise_cov_fname = NOISE_COV if args.noise_cov_fname == '' else args.noise_cov_fname
    get_meg_files(subject, [inv_fname], args, conditions)
    if args.overwrite_inv or not op.isfile(inv_fname) or (args.inv_calc_subcorticals and not op.isfile(INV_SUB)):
        if utils.should_run(args, 'make_forward_solution') and (not op.isfile(fwd_fname) or args.overwrite_fwd):
            # prepare_subject_folder(
            #     mri_subject, args.remote_subject_dir, SUBJECTS_MRI_DIR,
            #     {op.join('mri', 'T1-neuromag', 'sets'): ['COR.fif']}, args)
            trans_file = find_trans_file(COR, args.remote_subject_dir)
            if trans_file is None:
                flags['make_forward_solution'] = False
                flags['calc_inverse_operator'] = False
                return flags
            if op.isfile(trans_file):
                trans_fol = utils.make_dir(op.join('mri', 'T1-neuromag', 'sets'))
                if utils.get_parent_fol(trans_file) != trans_fol:
                    if trans_file != trans_fol:
                        shutil.copy(trans_file, trans_fol)
                    if trans_file != COR:
                        shutil.copy(trans_file, COR)
            src_dic = dict(bem=['*-oct-6*-src.fif'])
            create_src_dic = dict(surf=['lh.{}'.format(args.recreate_src_surface), 'rh.{}'.format(args.recreate_src_surface),
                       'lh.sphere', 'rh.sphere'])
            for nec_file in [src_dic, create_src_dic]:
                file_exist = prepare_subject_folder(
                    mri_subject, args.remote_subject_dir, SUBJECTS_MRI_DIR,
                    nec_file, args)
                if file_exist:
                    break
            get_meg_files(subject, [evo_fname], args, conditions)
            sub_corticals_codes_file = op.join(MMVT_DIR, 'sub_cortical_codes.txt')
            flags['make_forward_solution'], fwd, fwd_subs = make_forward_solution(
                mri_subject, conditions, evo_fname, fwd_fname, sub_corticals_codes_file, args.fwd_usingMEG,
                args.fwd_usingEEG, args.fwd_calc_corticals, args.fwd_calc_subcorticals, args.fwd_recreate_source_space,
                args.recreate_src_spacing, args.recreate_src_surface, args.overwrite_fwd, args.n_jobs, args)

        if utils.should_run(args, 'calc_inverse_operator') and flags.get('make_forward_solution', True):
            get_meg_files(subject, [epo_fname, fwd_fname], args, conditions)
            flags['calc_inverse_operator'] = calc_inverse_operator(
                conditions, raw_fname, epo_fname, evo_fname, fwd_fname, inv_fname, noise_cov_fname, args.inv_loose,
                args.inv_depth, args.noise_t_min, args.noise_t_max, args.overwrite_inv,
                args.use_empty_room_for_noise_cov, args.use_raw_for_noise_cov,
                args.overwrite_noise_cov, args.inv_calc_cortical, args.inv_calc_subcorticals,
                args.fwd_usingMEG, args.fwd_usingEEG, args=args)
    return flags


def calc_evokes_wrapper(subject, conditions, args, flags={}, raw=None, mri_subject=''):
    if mri_subject == '':
        mri_subject = subject
    evoked, epochs = None, None
    if utils.should_run(args, 'calc_epochs'):
        necessary_files = calc_epochs_necessary_files(args)
        get_meg_files(subject, necessary_files, args, conditions)
        flags['calc_epochs'], epochs = calc_epochs_wrapper_args(conditions, args, raw)
    if utils.should_run(args, 'calc_epochs') and not flags['calc_epochs']:
        return flags, evoked, epochs

    if utils.should_run(args, 'calc_evokes'):
        flags['calc_evokes'], evoked = calc_evokes(
            epochs, conditions, mri_subject, args.normalize_data, args.evo_fname,
            args.norm_by_percentile, args.norm_percs, args.modality, args.calc_max_min_diff,
            args.calc_evoked_for_all_epoches, args.overwrite_evoked, args.task)

    return flags, evoked, epochs


def get_stc_hemi_template(stc_template):
    return '{}{}'.format(stc_template[:-4], '-{hemi}.stc')


def calc_stc_per_condition_wrapper(subject, conditions, inverse_method, args, flags={}, raw=None, epochs=None):
    stcs_conds, stcs_num = None, {}
    if utils.should_run(args, 'calc_stc_per_condition'):
        stc_hemi_template = get_stc_hemi_template(args.stc_template)
        if conditions is None:
            conditions = ['all']
        stc_exist = all([utils.both_hemi_files_exist(stc_hemi_template.format(
            cond=cond, method=im, hemi='{hemi}')) for (im, cond) in product(args.inverse_method, conditions)])
        if stc_exist and not args.overwrite_stc:
            return True, None, {}
        if isinstance(inverse_method, Iterable) and not isinstance(inverse_method, str):
            inverse_method = inverse_method[0]
        inv_fname = get_inv_fname(args.inv_fname, args.fwd_usingMEG, args.fwd_usingEEG)
        evo_fname = get_evo_fname(args.evo_fname)
        get_meg_files(subject, [inv_fname, evo_fname], args, conditions)
        flags['calc_stc_per_condition'], stcs_conds, stcs_num = calc_stc_per_condition(
            conditions, args.stc_t_min, args.stc_t_max, inverse_method, args.baseline,
            args.apply_SSP_projection_vectors, args.add_eeg_ref, args.pick_ori, args.single_trial_stc, args.save_stc,
            args.snr, args.overwrite_stc, args.stc_template, args.epo_fname,
            args.evo_fname, args.inv_fname, args.fwd_usingMEG, args.fwd_usingEEG, args.apply_on_raw, raw, epochs,
            args.modality, args.calc_stc_for_all, args.calc_stc_diff, args.n_jobs)
    return flags, stcs_conds, stcs_num


def calc_labels_avg_for_rest_wrapper(args, raw=None):
    return calc_labels_avg_for_rest(
        args.atlas, args.inverse_method, raw, args.pick_ori, args.extract_mode, args.snr, args.raw_fname,
        args.inv_fname, args.labels_data_template, args.overwrite_stc, args.overwrite_labels_data,
        args.fwd_usingMEG, args.fwd_usingEEG, cond_name='all', positive=False, moving_average_win_size=0,
        save_data_files=True, n_jobs=args.n_jobs)


def calc_labels_avg_for_rest(
        atlas, inverse_method, raw=None, pick_ori=None, extract_modes=['mean_flip'], snr=1, raw_fname='', inv_fname='',
        labels_data_template='', overwrite_stc=False, overwrite_labels_data=False, fwd_usingMEG=True, fwd_usingEEG=True,
        cond_name='all', positive=False, moving_average_win_size=0, save_data_files=True, do_plot_time_series=True,
        n_jobs=6):

    def collect_parallel_results(indices, results, labels_num):
        labels_data_hemi = {}
        for indices_chunk, labels_data_chunk in zip(indices, results):
            for em in extract_modes:
                if em not in labels_data_hemi:
                    labels_data_hemi[em] = np.zeros((labels_num, labels_data_chunk[em].shape[1]))
                labels_data_hemi[em][indices_chunk] = labels_data_chunk[em]
        return labels_data_hemi

    labels_output_fol_template = op.join(
        SUBJECT_MEG_FOLDER, 'rest_{}_labels_data_{}'.format(atlas, '{extract_mode}'))
    if labels_data_template == '':
        labels_data_template = LBL
    min_max_output_template = get_labels_minmax_template(labels_data_template)

    labels_num = lu.get_labels_num(MRI_SUBJECT, SUBJECTS_MRI_DIR, atlas)
    labels_files_exist = all([len(glob.glob(op.join(
        SUBJECT_MEG_FOLDER, labels_output_fol_template.format(extract_mode=em), '*.npy'))) == labels_num
        for em in extract_modes])
    labels_data_exist = all([utils.both_hemi_files_exist(labels_data_template.format(atlas, em, '{hemi}'))
                             for em in extract_modes])
    if (not labels_files_exist and not labels_data_exist) or overwrite_labels_data:
        inv_fname = get_inv_fname(inv_fname, fwd_usingMEG, fwd_usingEEG)
        if raw is None:
            raw_fname = get_raw_fname(raw_fname)
            if op.isfile(raw_fname):
                mne.io.read_raw_fif(raw_fname)
            else:
                raise Exception("Can't find the raw file! ({})".format(raw_fname))
        if not isinstance(inverse_method, str) and isinstance(inverse_method, Iterable):
            inverse_method = inverse_method[0]
        inverse_operator = read_inverse_operator(inv_fname.format(cond=cond_name))
        src = inverse_operator['src']
        lambda2 = 1.0 / snr ** 2
        labels_data = {}
        for hemi in utils.HEMIS:
            labels = lu.read_labels(MRI_SUBJECT, SUBJECTS_MRI_DIR, atlas, hemi=hemi)
            indices = np.array_split(np.arange(len(labels)), n_jobs)
            chunks = [([labels[ind] for ind in indices_chunk], raw, src, inverse_operator, lambda2, inverse_method,
                       extract_modes, pick_ori, save_data_files, labels_output_fol_template, overwrite_stc,
                       do_plot_time_series) for indices_chunk in indices]
            results = utils.run_parallel(calc_stc_labels_parallel, chunks, n_jobs)
            labels_data[hemi] = collect_parallel_results(indices, results, len(labels))

    elif (not labels_data_exist) or overwrite_labels_data:
        labels_data = {}
        for hemi in utils.HEMIS:
            labels_names = lu.get_labels_names(MRI_SUBJECT, SUBJECTS_MRI_DIR, atlas, hemi)
            indices = np.array_split(np.arange(len(labels_names)), n_jobs)
            chunks = [([labels_names[ind] for ind in indices_chunk], extract_modes, labels_output_fol_template,
                       do_plot_time_series) for indices_chunk in indices]
            results = utils.run_parallel(_load_labels_data_parallel, chunks, n_jobs)
            labels_data[hemi] = collect_parallel_results(indices, results, len(labels))

    for em in extract_modes:
        data_max = max([np.max(labels_data[hemi][em]) for hemi in utils.HEMIS])
        data_min = min([np.min(labels_data[hemi][em]) for hemi in utils.HEMIS])
        # data_minmax = utils.get_max_abs(data_max, data_min)
        # factor = -int(utils.ceil_floor(np.log10(data_minmax)))
        factor = 9 # to get nAmp
        min_max_output_fname = op.join(MMVT_DIR, MRI_SUBJECT, 'meg', min_max_output_template.format(atlas, em))
        np.savez(min_max_output_fname, labels_minmax=[data_min, data_max])
        if (not labels_data_exist) or overwrite_labels_data:
            for hemi in utils.HEMIS:
                labels_names = lu.get_labels_names(MRI_SUBJECT, SUBJECTS_MRI_DIR, atlas, hemi)
                save_labels_data(labels_data[hemi], hemi, labels_names, atlas, ['all'], extract_modes,
                                 labels_data_template, factor, positive, moving_average_win_size)

    output_fol = op.join(MMVT_DIR, MRI_SUBJECT, 'meg')
    flag = all([op.isfile(op.join(output_fol, op.basename(labels_data_template.format(atlas, em, hemi)))) for \
        em, hemi in product(extract_modes, utils.HEMIS)]) and \
        all([op.isfile(op.join(output_fol, min_max_output_template.format(atlas, em))) for em in extract_modes])
    return flag


def calc_stc_labels_parallel(p):
    (labels, raw, src, inverse_operator, lambda2, inverse_method, extract_modes, pick_ori,
     save_data_files, labels_output_fol_template, overwrite, do_plot_time_series) = p
    labels_data = {}
    for ind, label in enumerate(labels):
        stc = mne.minimum_norm.apply_inverse_raw(
            raw, inverse_operator, lambda2, inverse_method, label=label, pick_ori=pick_ori)
        for em in extract_modes:
            # label_data = stc.extract_label_time_course(label, src, mode=em, allow_empty=True)
            if em != 'mean_flip':
                print("{} isn't implemented yet for rest data!".format(em))
                continue
            labels_output_fol = labels_output_fol_template.format(extract_mode=em)
            utils.make_dir(labels_output_fol)
            label_data = extract_label_data(label, src, stc)
            label_data = np.squeeze(label_data)
            if do_plot_time_series:
                plot_label_data(label_data, label.name, em, labels_output_fol)
            if save_data_files:
                # fol = utils.make_dir(op.join(SUBJECT_MEG_FOLDER, 'rest_labels_data_{}'.format(em)))
                label_fname = op.join(labels_output_fol, '{}-{}.npy'.format(label.name, em))
                if not op.isfile(label_fname) and not overwrite:
                    np.save(label_fname, label_data)
            if em not in labels_data:
                T = len(stc.times)
                labels_data[em] = np.zeros((len(labels), T))
            labels_data[em][ind, :] = label_data
    return labels_data


def _load_labels_data_parallel(p):
    labels, extract_modes, labels_output_fol_template, do_plot_time_series = p
    labels_data = {}
    for em in extract_modes:
        labels_output_fol = labels_output_fol_template.format(extract_mode=em)
        for ind, label in enumerate(labels):
            label_fname = op.join(labels_output_fol, '{}-{}.npy'.format(label, em))
            label_data = np.load(label_fname).squeeze()
            if do_plot_time_series:
                plot_label_data(label_data, label, em, labels_output_fol)
            if em not in labels_data:
                labels_data[em] = np.zeros((len(labels), len(label_data)))
            try:
                labels_data[em][ind, :] = label_data
            except:
                print('error in {}'.format(label))
                raise Exception(traceback.format_exc())
    return labels_data


def plot_label_data(label_data, label, em, labels_output_fol):
    plt.figure()
    plt.plot(label_data)
    plt.title('{} {}'.format(label, em))
    plt.savefig(op.join(labels_output_fol, '{}-{}.png'.format(label, em)))
    plt.close()


def extract_label_data(label, src, stc):
    label_flip = mne.label_sign_flip(label, src)[:, None].squeeze()
    label_flip = np.tile(label_flip, (stc.data.shape[1], 1)).T
    label_tc = np.mean(label_flip * stc.data, axis=0)
    return label_tc


def calc_labels_avg_per_condition_wrapper(
        subject, conditions, atlas, inverse_method, stcs_conds, args, flags={}, stcs_num={}, raw=None, epochs=None,
        overwrite=False):
    if utils.should_run(args, 'calc_labels_avg_per_condition'):
        labels_data_exist = all([utils.both_hemi_files_exist(args.labels_data_template.format(
            args.atlas, me, '{hemi}')) for me in args.extract_mode])
        if labels_data_exist and not overwrite:
            if utils.should_run(args, 'calc_labels_min_max'):
                flags['calc_labels_min_max'] = calc_labels_minmax(
                    atlas, args.extract_mode, args.labels_data_template, args.overwrite_labels_data)
            return flags

        conditions_keys = conditions.keys() if conditions is not None else ['all']
        if isinstance(inverse_method, Iterable) and not isinstance(inverse_method, str):
            inverse_method = inverse_method[0]
        args.inv_fname = get_inv_fname(args.inv_fname, args.fwd_usingMEG, args.fwd_usingEEG)
        args.stc_hemi_template = STC_HEMI if args.stc_template == '' else get_stc_hemi_template(args.stc_template)
        stc_fnames = [args.stc_hemi_template.format(cond='{cond}', method=inverse_method, hemi=hemi)
                      for hemi in utils.HEMIS]
        get_meg_files(subject, stc_fnames + [args.inv_fname], args, conditions)
        if stcs_conds is None:
            stcs_conds = get_stc_conds(conditions, inverse_method, args.stc_hemi_template)
            if stcs_conds is None:
                return False
        # data_min = min([utils.min_stc(stc_cond) for stc_cond in stcs_conds.values()])
        # data_max = max([utils.max_stc(stc_cond) for stc_cond in stcs_conds.values()])
        # data_minmax = utils.get_max_abs(data_max, data_min)
        # factor = -int(utils.ceil_floor(np.log10(data_minmax)))
        factor = 9 # to get nAmp
        for hemi_ind, hemi in enumerate(HEMIS):
            flags['calc_labels_avg_per_condition_{}'.format(hemi)] = calc_labels_avg_per_condition(
                args.atlas, hemi, conditions, extract_modes=args.extract_mode,
                positive=args.evoked_flip_positive,
                moving_average_win_size=args.evoked_moving_average_win_size,
                labels_data_template=args.labels_data_template,
                stcs=stcs_conds, factor=factor, inv_fname=args.inv_fname,
                fwd_usingMEG=args.fwd_usingMEG, fwd_usingEEG=args.fwd_usingMEG,
                stcs_num=stcs_num, read_only_from_annot=args.read_only_from_annot, overwrite=overwrite,
                n_jobs=args.n_jobs)
            if stcs_conds and isinstance(stcs_conds[list(conditions_keys)[0]], types.GeneratorType) and hemi_ind == 0:
                # Create the stc generator again for the second hemi
                _, stcs_conds, stcs_num = calc_stc_per_condition_wrapper(
                    subject, conditions, inverse_method, args, flags, raw=raw, epochs=epochs)

    if utils.should_run(args, 'calc_labels_min_max'):
        flags['calc_labels_min_max'] = calc_labels_minmax(
            atlas, args.extract_mode, args.labels_data_template, args.overwrite_labels_data)
    return flags


def calc_labels_minmax(atlas, extract_modes, labels_data_template='', overwrite_labels_data=False):
    if labels_data_template == '':
        labels_data_template = LBL
    if isinstance(extract_modes, str):
        extract_modes = [extract_modes]
    min_max_output_template = get_labels_minmax_template(labels_data_template)
    for em in extract_modes:
        min_max_output_fname = min_max_output_template.format(atlas, em)
        min_max_mmvt_output_fname = op.join(MMVT_DIR, MRI_SUBJECT, 'meg', utils.namebase_with_ext(min_max_output_fname))
        if op.isfile(min_max_output_fname) and op.isfile(min_max_mmvt_output_fname) and not overwrite_labels_data:
            continue
        template = labels_data_template.format(atlas, em, '{hemi}') # op.join(MMVT_DIR, MRI_SUBJECT, 'meg', op.basename(LBL.format(atlas, em, '{hemi}')))
        if utils.both_hemi_files_exist(template):
            labels_data = [np.load(template.format(hemi=hemi)) for hemi in utils.HEMIS]
            labels_min = min([np.min(d['data']) for d in labels_data])
            labels_max = max([np.max(d['data']) for d in labels_data])
            labels_diff_min = min([np.min(np.diff(d['data'])) for d in labels_data])
            labels_diff_max = max([np.max(np.diff(d['data'])) for d in labels_data])
            np.savez(min_max_output_fname, labels_minmax=[labels_min, labels_max],
                     labels_diff_minmax=[labels_diff_min, labels_diff_max])
            print('{}: min: {}, max: {}'.format(em, labels_min, labels_max))
            print('{}: labels diff min: {}, labels diff max: {}'.format(em, labels_diff_min, labels_diff_max))
            # shutil.copy(min_max_output_fname, min_max_mmvt_output_fname)
        else:
            print("Can't find {}!".format(template))
    return np.all([op.isfile(op.join(MMVT_DIR, MRI_SUBJECT, 'meg', min_max_output_template.format(atlas, em)))
                   for em in extract_modes])


def get_labels_minmax_template(labels_data_template):
    return '{}_minmax.npz'.format(labels_data_template[:-7])


def calc_stc_diff(stc1_fname, stc2_fname, output_name):
    stc1 = mne.read_source_estimate(stc1_fname)
    stc2 = mne.read_source_estimate(stc2_fname)
    stc_diff = stc1 - stc2
    output_name = output_name[:-len('-lh.stc')]
    stc_diff.save(output_name)
    mmvt_fname = utils.make_dir(op.join(MMVT_DIR, MRI_SUBJECT, 'meg', utils.namebase(output_name)))
    utils.make_dir(utils.get_parent_fol(output_name))
    for hemi in utils.HEMIS:
        shutil.copy('{}-{}.stc'.format(output_name, hemi),
                    '{}-{}.stc'.format(mmvt_fname, hemi))
        print('Saving to {}'.format('{}-{}.stc'.format(mmvt_fname, hemi)))


def find_functional_rois_in_stc(subject, mri_subject, atlas, stc_name, threshold, threshold_is_precentile=True, time_index=None,
                                label_name_template='', peak_mode='abs', extract_mode='mean_flip',
                                min_cluster_max=0, min_cluster_size=0, clusters_label='', src=None,
                                inv_fname='', fwd_usingMEG=True, fwd_usingEEG=True,
                                recreate_src_spacing='oct6', n_jobs=6):
    import mne.stats.cluster_level as mne_clusters

    clusters_root_fol = op.join(MMVT_DIR, subject, 'meg', 'clusters')
    # todo: Should check for an overwrite flag. Not sure why, if the folder isn't being deleted, the code doesn't work
    # utils.delete_folder_files(clusters_root_fol)
    utils.make_dir(clusters_root_fol)
    stc_fname = op.join(MMVT_DIR, subject, 'meg', '{}-lh.stc'.format(stc_name))
    stc = mne.read_source_estimate(stc_fname)
    if time_index is None:
        if label_name_template == '':
            max_vert, time_index = stc.get_peak(
                time_as_index=True, vert_as_index=True, mode=peak_mode)
        else:
            max_vert, time_index = find_pick_activity(
                subject, stc, atlas, label_name_template, hemi='both', peak_mode=peak_mode)
    stc_t = create_stc_t(stc, time_index, subject)
    stc_t_smooth = calc_stc_for_all_vertices(stc_t, subject, subject, n_jobs)
    verts = check_stc_with_ply(stc_t_smooth, subject=subject)
    connectivity = load_connectivity(subject)
    if threshold_is_precentile:
        threshold = np.percentile(stc_t_smooth.data, threshold)
    clusters_name = '{}-{}'.format(stc_name, label_name_template.replace('*', '').replace('?', ''))
    clusters_fol = op.join(clusters_root_fol, clusters_name)
    # data_minmax = utils.get_max_abs(utils.min_stc(stc), utils.max_stc(stc))
    # factor = -int(utils.ceil_floor(np.log10(data_minmax)))
    threshold *= np.power(10, 9) # nAmp
    min_cluster_max = max(threshold, min_cluster_max)
    clusters_labels = utils.Bag(
        dict(stc_name=stc_name, threshold=threshold, time=time_index, label_name_template=label_name_template, values=[],
             min_cluster_max=min_cluster_max, min_cluster_size=min_cluster_size, clusters_label=clusters_label))
    for hemi in utils.HEMIS:
        stc_data = (stc_t_smooth.rh_data if hemi == 'rh' else stc_t_smooth.lh_data).squeeze() * np.power(10, 9)
        clusters, _ = mne_clusters._find_clusters(stc_data, threshold, connectivity=connectivity[hemi])
        if len(clusters) == 0:
            print('No clusters where found for {}-{}!'.format(stc_name, hemi))
            continue
        clusters_labels_hemi = lu.find_clusters_overlapped_labeles(
            subject, clusters, stc_data, atlas, hemi, verts[hemi], min_cluster_max, min_cluster_size, clusters_label,
            n_jobs)
        if clusters_labels_hemi is None or len(clusters_labels_hemi) == 0:
            print("Can't find overlapped_labeles in {}-{}!".format(stc_name, hemi))
        else:
            clusters_labels_hemi = extract_time_series_for_cluster(
                subject, mri_subject, stc, hemi, clusters_labels_hemi, clusters_fol, extract_mode[0], src, inv_fname,
                time_index, min_cluster_max, fwd_usingMEG, fwd_usingEEG, recreate_src_spacing=recreate_src_spacing)
            clusters_labels.values.extend(clusters_labels_hemi)
    clusters_labels_output_fname = op.join(clusters_root_fol, 'clusters_labels_{}.pkl'.format(stc_name, atlas))
    print('Saving clusters labels: {}'.format(clusters_labels_output_fname))
    # Change Bag to regular dict because we want to load the pickle file in Blender (argggg)
    for ind in range(len(clusters_labels.values)):
        clusters_labels.values[ind] = dict(**clusters_labels.values[ind])
    clusters_labels = dict(**clusters_labels)
    utils.save(clusters_labels, clusters_labels_output_fname)
    return op.isfile(clusters_labels_output_fname)


def find_pick_activity(subject, stc, atlas, label_name_template='', hemi='both', peak_mode='abs'):
    if isinstance(stc, str):
        stc = mne.read_source_estimate(stc)
    if label_name_template == '':
        max_vert, time_index = stc.get_peak(
            time_as_index=True, vert_as_index=True, mode=peak_mode)
    else:
        hemis = utils.HEMIS if hemi=='both' else [hemi]
        for hemi in hemis:
            data, vertices = get_stc_data_and_vertices(stc, hemi)
            label_vertices, label_vertices_indices = \
                lu.find_label_vertices(subject, atlas, hemi, vertices, label_name_template)
            label_data = data[label_vertices_indices]
            if peak_mode == 'abs':
                max_vert, time_index = utils.argmax2d(abs(label_data))
            elif peak_mode == 'pos':
                max_vert, time_index = utils.argmax2d(label_data)
            elif peak_mode == 'neg':
                max_vert, time_index = utils.argmax2d(-label_data)
            max_vert = label_vertices_indices[max_vert]
    return max_vert, time_index


def get_stc_data_and_vertices(stc, hemi):
    return (stc.lh_data, stc.lh_vertno) if hemi == 'lh' else (stc.rh_data, stc.rh_vertno)


def load_connectivity(subject):
    connectivity_fname = op.join(MMVT_DIR, subject, 'spatial_connectivity.pkl')
    if not op.isfile(connectivity_fname):
        from src.preproc import anatomy
        anatomy.create_spatial_connectivity(subject)
    connectivity_per_hemi = utils.load(connectivity_fname)
    return connectivity_per_hemi


def extract_time_series_for_cluster(subject, mri_subject, stc, hemi, clusters, clusters_fol, extract_mode='mean_flip',
                                    src=None, inv_fname='', time_index=-1, cluster_max_min=0, fwd_usingMEG=True,
                                    fwd_usingEEG=True, calc_contours=True, recreate_src_spacing='oct6'):
    utils.make_dir(clusters_fol)
    time_series_fol = op.join(clusters_fol, 'time_series_{}'.format(extract_mode))
    utils.make_dir(time_series_fol)
    inv_fname = get_inv_fname(inv_fname, fwd_usingMEG, fwd_usingEEG)
    if src is None:
        if op.isfile(inv_fname):
            inverse_operator = read_inverse_operator(inv_fname)
            src = inverse_operator['src']
        else:
            # todo: set the recreate_src_spacing according to the stc
            # https://martinos.org/mne/dev/manual/cookbook.html#source-localization
            src = check_src(mri_subject, recreate_src_spacing=recreate_src_spacing)
    for cluster_ind, cluster in enumerate(clusters):
        cluster = utils.Bag(cluster)
        # cluster: vertices, intersects, name, coordinates, max, hemi, size
        cluster_label = mne.Label(
            cluster.vertices, cluster.coordinates, hemi=cluster.hemi, name=cluster.name, subject=subject)
        label_data = x = stc.extract_label_time_course(cluster_label, src, mode=extract_mode, allow_empty=True).squeeze()
        if time_index == -1:
            cluster.ts_max = np.min(x) if abs(np.min(x)) > abs(np.max(x)) else np.max(x)
        else:
            cluster.ts_max = x[time_index] * np.power(10, 9)
        cluster_name = 'cluster_size_{}_max_{:.2f}_{}.label'.format(cluster.size, cluster.max, cluster.name)
        cluster_label.save(op.join(clusters_fol, cluster_name))
        if np.all(label_data == 0):
            cluster.label_data = None
        else:
            cluster.label_data = np.squeeze(label_data)
            cluster.label_data *= np.power(10, 9)
            np.save(op.join(time_series_fol, '{}.npy'.format(cluster_name)), cluster.label_data)
    if len(clusters) > 0 and calc_contours:
        new_atlas_name = 'clusters-{}-{}'.format(utils.namebase(clusters_fol), hemi)
        annot_files = labels_to_annot(new_atlas_name, clusters_fol)
        if annot_files is not None:
            for hemi in utils.HEMIS:
                annot_fname = annot_files.format(hemi=hemi)
                dest_annot_fname = op.join(clusters_fol, utils.namebase_with_ext(annot_fname))
                if not op.isfile(dest_annot_fname):
                    shutil.copy(annot_fname, dest_annot_fname)
        anat.calc_labeles_contours(subject, new_atlas_name, overwrite=True, verbose=False)
    return clusters


def fit_ica(raw=None, n_components=0.95, method='fastica', ica_fname='', raw_fname='', overwrite_ica=False,
            do_plot=False, examine_ica=False, n_jobs=6):
    from mne.preprocessing import read_ica
    if raw_fname == '':
        raw_fname = RAW
    if ica_fname == '':
        ica_fname = '{}-{}'.format(op.splitext(raw_fname)[0][:-4], 'ica.fif')
    if op.isfile(ica_fname) and not overwrite_ica:
        ica = read_ica(ica_fname)
    else:
        ica = ICA(n_components=n_components, method=method)
        raw.filter(1, 45, n_jobs=n_jobs, l_trans_bandwidth=0.5, h_trans_bandwidth=0.5,
                   filter_length='10s', phase='zero-double')
        picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, ref_meg=False,
                               stim=False, exclude='bads')
        ica.fit(raw, picks=picks, decim=3, reject=dict(mag=4e-12, grad=4000e-13))
    print(ica)
    if do_plot:
        fig = ica.plot_components(picks=range(max(ica.n_components_, 20)), inst=raw)
        fig_fname = '{}.{}'.format(op.splitext(ica_fname)[0], 'png')
        if not op.isfile(fig_fname):
            fig.savefig(fig_fname)
    if examine_ica:
        output_fname = op.join(utils.get_parent_fol(ica_fname), 'ica_comps.txt')
        comp_num = input('Type the ICA component you want to save: ')
        while not utils.is_int(comp_num):
            print('Please enter a valid integer')
        with open(output_fname, 'a') as f:
            f.write('{},{}\n'.format(utils.namebase_with_ext(ica_fname), comp_num))
    return ica, ica_fname


def find_raw_fname(raw=None, raw_fname='', raw_template='*raw.fif'):
    if op.isfile(raw_fname):
        return raw_fname, True
    if raw is None or (isinstance(raw, str) and not op.isfile(raw)):
        raw_fname, raw_exist = locating_meg_file(raw_fname, glob_pattern=raw_template)
        if not raw_exist:
            return '', False
    elif isinstance(raw, mne.io.fiff.raw.Raw):
        raw_fname = raw.filenames[0]
    if not op.isfile(raw_fname):
        return '', False
    else:
        return raw_fname, True


def remove_artifacts(raw=None, n_max_ecg=3, n_max_eog=1, n_components=0.95, method='fastica',
                     ecg_inds=[], eog_inds=[], eog_channel=None, remove_from_raw=True, save_raw=True, raw_fname='',
                     new_raw_fname='', ica_fname='', overwrite_ica=False, overwrite_raw=False, do_plot=False,
                     raw_template='*raw.fif', n_jobs=6):
    from mne.preprocessing import read_ica
    raw_fname, raw_exist = find_raw_fname(raw, raw_fname, raw_template)
    if not raw_exist:
        print("remove_artifacts: Can't find raw!")
        return False
    if ica_fname == '':
        raw_fname, raw_exist = locating_meg_file(raw_fname, glob_pattern=raw_template)
        ica_fname = '{}-{}'.format(op.splitext(raw_fname)[0][:-4], 'ica.fif')
    if op.isfile(ica_fname) and not overwrite_ica:
        ica = read_ica(ica_fname)
    else:
        if isinstance(raw, str) and op.isfile(raw):
            raw = mne.io.read_raw_fif(raw)
        elif raw is None:
            raw = load_raw(raw_fname, raw_template=raw_template)
        # 1) Fit ICA model using the FastICA algorithm.
        ica, ica_fname = fit_ica(raw, n_components, method, ica_fname, raw_fname, overwrite_ica, do_plot, n_jobs=n_jobs)
        picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, ref_meg=False,
                               stim=False, exclude='bads')
        ###############################################################################
        # 2) identify bad components by analyzing latent sources.
        # generate ECG epochs use detection via phase statistics
        if len(ecg_inds) == 0:
            ecg_epochs = create_ecg_epochs(raw, tmin=-.5, tmax=.5, picks=picks)
            ecg_inds, scores = ica.find_bads_ecg(ecg_epochs, method='ctps')
        else:
            scores = [1.0 / len(ecg_inds)] * len(ecg_inds)
        if len(ecg_inds) == 0:
            print('No ECG artifacts!')
        else:
            if do_plot:
                title = 'Sources related to %s artifacts (red)'
                ica.plot_scores(scores, exclude=ecg_inds, title=title % 'ecg', labels='ecg')
                show_picks = np.abs(scores).argsort()[::-1][:5]
                ica.plot_sources(raw, show_picks, exclude=ecg_inds, title=title % 'ecg')
                ica.plot_components(ecg_inds, title=title % 'ecg', colorbar=True)
            ecg_inds = ecg_inds[:n_max_ecg]
            ica.exclude += ecg_inds
        try:
            if len(eog_inds) == 0:
                # detect EOG by correlation # eog_ch = [c for c in raw.ch_names if 'EOG' in c]
                eog_inds, scores = ica.find_bads_eog(raw, ch_name=eog_channel)
            else:
                scores = [1.0 / len(eog_inds)] * len(eog_inds)
            if len(eog_inds) == 0:
                print('No EOG artifacts!')
            else:
                if do_plot:
                    ica.plot_scores(scores, exclude=eog_inds, title=title % 'eog', labels='eog')
                    show_picks = np.abs(scores).argsort()[::-1][:5]
                    ica.plot_sources(raw, show_picks, exclude=eog_inds, title=title % 'eog')
                    ica.plot_components(eog_inds, title=title % 'eog', colorbar=True)
                eog_inds = eog_inds[:n_max_eog]
                ica.exclude += eog_inds
        except:
            print("Can't remove EOG artifacts!")
        ###############################################################################
        # 3) Assess component selection and unmixing quality.
        # estimate average artifact
        if do_plot:
            ecg_evoked = ecg_epochs.average()
            print('We found %i ECG events' % ecg_evoked.nave)
            ica.plot_sources(ecg_evoked, exclude=ecg_inds)  # plot ECG sources + selection
            ica.plot_overlay(ecg_evoked, exclude=ecg_inds)  # plot ECG cleaning
            try:
                eog_evoked = create_eog_epochs(raw, tmin=-.5, tmax=.5, picks=picks, ch_name=eog_channel).average()
                print('We found %i EOG events' % eog_evoked.nave)
                ica.plot_sources(eog_evoked, exclude=eog_inds)  # plot EOG sources + selection
                ica.plot_overlay(eog_evoked, exclude=eog_inds)  # plot EOG cleaning
            except:
                print("Can't create eog epochs, no EEG")
            # check the amplitudes do not change
            ica.plot_overlay(raw)  # EOG artifacts remain
        ica.save(ica_fname)

    if len(ica.exclude) == 0:
        print("ICA didn't find any artifacts!")
    else:
        if remove_from_raw:
            raw = ica.apply(raw)
            if save_raw:
                if new_raw_fname == '':
                    new_raw_fname = raw_fname if overwrite_raw else RAW_ICA
                raw.save(new_raw_fname, overwrite=True)
    return True


def remove_artifacts_with_template_matching(ica_subjects='all', meg_root_fol=''):
    subject_ica_fname = '{}-{}'.format(op.splitext(RAW)[0][:-4], 'ica.fif')
    if not op.isfile(subject_ica_fname):
        print('You should first call remove_artifacts to create an ICA file')
        return False
    if meg_root_fol == '':
        meg_root_fol = MEG_DIR
    eog_teampltes = find_eog_template(ica_subjects, meg_root_fol)
    if eog_teampltes is None:
        print('EOG template is None!')
        return False
    raw_fnames_for_eog_template = list(eog_teampltes.keys())
    reference_ica, reference_eog_inds = eog_teampltes[raw_fnames_for_eog_template[0]]
    subject_ica = mne.preprocessing.read_ica(subject_ica_fname)
    icas = [reference_ica, subject_ica]
    template = (0, reference_eog_inds[0])
    fig_template, fig_detected = mne.preprocessing.corrmap(
        icas, template=template, label="blinks", show=True, threshold=.8, ch_type='mag')
    print('hmmm')


def find_eog_template(subjects='all', n_components=0.95, meg_root_fol='', method='fastica'):
    eog_teampltes_fname = op.join(MEG_DIR, 'eog_templates.pkl')
    if op.isfile(eog_teampltes_fname):
        eog_teampltes = utils.load(eog_teampltes_fname)
        return eog_teampltes
    # subject_ica_fname = '{}-{}'.format(op.splitext(RAW)[0][:-4], 'artifacts_removal-ica.fif')
    # subject_ica = mne.preprocessing.read_ica(subject_ica_fname)
    # n_components = subject_ica.n_components
    raw_files = collect_raw_files(subjects, meg_root_fol, excludes=(RAW))
    reject = dict(mag=5e-12, grad=4000e-13)
    valid_raw_files, eog_templates = {}, {}
    template_found = False
    for raw_fname in raw_files:
        raw_ica_fname = '{}-{}'.format(op.splitext(raw_fname)[0][:-4], 'ica.fif')
        if not op.isfile(raw_ica_fname) or raw_fname == RAW:
            continue
        try:
            raw = mne.io.read_raw_fif(raw_fname)
            eog_epochs = create_eog_epochs(raw, reject=reject)  # get single EOG trials
            if len(eog_epochs) > 0:
                valid_raw_files[raw_fname] = eog_epochs
                if op.isfile(raw_ica_fname):
                    ica = mne.preprocessing.read_ica(raw_ica_fname)
                else:
                    ica = ICA(n_components=n_components, method=method)
                    picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, ref_meg=False,
                                           stim=False, exclude='bads')
                    ica.fit(raw, picks=picks, decim=3, reject=dict(mag=4e-12, grad=4000e-13))
                eog_inds, scores = ica.find_bads_eog(eog_epochs)  # find via correlation
                if len(eog_inds) > 0:
                    eog_templates[raw_fname] = (ica, eog_inds)
                    template_found = True
                    break
                    # ica.plot_components(eog_inds, colorbar=True)
                    # eog_average = eog_epochs.average()
                    # ica.plot_sources(eog_average, exclude=eog_inds)
        except:
            print("Can't examine EOG artifacts on {}".format(raw_fname))
            print(traceback.format_exc())
    if template_found:
        utils.save(eog_templates, eog_teampltes_fname)
        return eog_templates
    else:
        print("Couldn't find any teamplte for EOG!")
        return None


def fit_ica_on_subjects(subjects='all', n_components=0.95, method='fastica', ica_fname='', meg_root_fol='',
                        overwrite_ica=False):
    from mne.preprocessing import ICA

    if ica_fname == '':
        ica_fname = 'ica.fif'
    ica_files_num, raw_files_num = 0, 0
    raw_files = collect_raw_files(subjects, meg_root_fol)
    for raw_fname in raw_files:
        subject_ica_fname = op.join(utils.get_parent_fol(raw_fname), ica_fname)
        if (not op.isfile(subject_ica_fname) or overwrite_ica) and 'empty' not in utils.namebase(raw_fname):
            raw_files_num += 1
    print('{} raw data were found!'.format(raw_files_num))
    for raw_fname in raw_files:
        try:
            subject_ica_fname = '{}-{}'.format(op.splitext(raw_fname)[0][:-4], 'artifacts_removal-ica.fif')
            if op.isfile(subject_ica_fname) and not overwrite_ica:
                ica_files_num += 1
                continue
            # Don't analyze empty room...
            if 'empty' in utils.namebase(raw_fname):
                continue
            print('Fitting ICA on {}'.format(raw_fname))
            raw = mne.io.read_raw_fif(raw_fname, preload=True)
            raw.filter(1, 45, n_jobs=1, l_trans_bandwidth=0.5, h_trans_bandwidth=0.5,
                       filter_length='10s', phase='zero-double')
            picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, ref_meg=False,
                                   stim=False, exclude='bads')
            ica = ICA(n_components=n_components, method=method)
            ica.fit(raw, picks=picks, decim=3, reject=dict(mag=4e-12, grad=4000e-13))
            if not op.isfile(subject_ica_fname) or overwrite_ica:
                ica.save(subject_ica_fname)
            ica_files_num += 1
            del raw, ica
        except:
            print(traceback.format_exc())
    print('{} ICA files were created'.format(ica_files_num))


def collect_raw_files(subjects='all', meg_root_fol='', excludes=()):
    if meg_root_fol == '':
        meg_root_fol = MEG_DIR
    if subjects == 'all' or 'all' in subjects:
        raw_files = glob.glob(op.join(meg_root_fol, '**', '*raw.fif'), recursive=True)
    else:
        raw_files = []
        for subject in subjects:
            subject_raw_file = utils.select_one_file(glob.glob(op.join(meg_root_fol, subject, '*raw.fif')))
            if subject_raw_file is not None:
                raw_files.append(subject_raw_file)
    raw_files = [raw_fname for raw_fname in raw_files if
                 'empty' not in utils.namebase(raw_fname) and utils.namebase(raw_fname) not in excludes]
    return raw_files


def load_fieldtrip_volumetric_data(subject, data_name, data_field_name,
                                   overwrite_nii_file=False, overwrite_surface=False, overwrite_stc=False):
    volumetric_meg_fname = op.join(MEG_DIR, subject, '{}.nii'.format(data_name))
    if not op.isfile(volumetric_meg_fname) or overwrite_nii_file:
        fname = op.join(MEG_DIR, subject, '{}.mat'.format(data_name))
        # load Matlab/Fieldtrip data
        mat = sio.loadmat(fname, squeeze_me=True, struct_as_record=False)
        ft_data = mat[data_name]
        data = getattr(ft_data, data_field_name)
        data[np.isnan(data)] = 0
        affine = ft_data.transform
        nib.save(nib.Nifti1Image(data, affine), volumetric_meg_fname)
    surface_output_template = op.join(MEG_DIR, subject, '{}_{}.mgz'.format(data_name, '{hemi}'))
    if not utils.both_hemi_files_exist(surface_output_template) or overwrite_surface:
        fu.project_on_surface(subject, volumetric_meg_fname, surface_output_template, overwrite_surf_data=True,
                              modality='meg')
    stcs_exist = utils.both_hemi_files_exist(op.join(MMVT_DIR, subject, 'meg', '{}-{}.stc'.format(data_name, '{hemi}')))
    if not stcs_exist or overwrite_stc:
        data = {}
        for hemi in utils.HEMIS:
            data[hemi] = np.load(op.join(MMVT_DIR, subject, 'meg', 'meg_{}_{}.npy'.format(data_name, hemi)))
            data[hemi][np.where(np.isnan(data[hemi]))] = 0
        stc = create_stc_t_from_data(subject, data['rh'], data['lh'])
        stc.save(op.join(MMVT_DIR, subject, 'meg', data_name))
    return stcs_exist


def init_main(subject, mri_subject, remote_subject_dir, args):
    if args.events_file_name != '':
        args.events_file_name = op.join(MEG_DIR, args.task, subject, args.events_file_name)
        if '{subject}' in args.events_file_name:
            args.events_file_name = args.events_file_name.format(subject=subject)
    args.remote_subject_dir = remote_subject_dir
    args.remote_subject_meg_dir = utils.build_remote_subject_dir(args.remote_subject_meg_dir, subject)
    prepare_subject_folder(mri_subject, remote_subject_dir, SUBJECTS_MRI_DIR,
                           args.mri_necessary_files, args)
    fname_format, fname_format_cond, conditions = get_fname_format_args(args)
    return fname_format, fname_format_cond, conditions


def init(subject, args, mri_subject='', remote_subject_dir=''):
    if mri_subject == '':
        mri_subject = subject
    fname_format, fname_format_cond, conditions = init_main(subject, mri_subject, remote_subject_dir, args)
    init_globals_args(
        subject, mri_subject, fname_format, fname_format_cond, MEG_DIR, SUBJECTS_MRI_DIR, MMVT_DIR, args)
    return fname_format, fname_format_cond, conditions


def main(tup, remote_subject_dir, args, flags=None):
    (subject, mri_subject), inverse_method = tup
    evoked, epochs, raw = None, None, None
    stcs_conds, stcs_conds_smooth = None, None
    if flags is None:
        flags = {}
    fname_format, fname_format_cond, conditions = init(subject, args, mri_subject, remote_subject_dir)
    # fname_format, fname_format_cond, conditions = init_main(subject, mri_subject, remote_subject_dir, args)
    # init_globals_args(
    #     subject, mri_subject, fname_format, fname_format_cond, MEG_DIR, SUBJECTS_MRI_DIR, MMVT_DIR, args)
    stat = STAT_AVG if len(conditions) == 1 else STAT_DIFF

    # flags: calc_evoked
    flags, evoked, epochs = calc_evokes_wrapper(subject, conditions, args, flags, mri_subject=mri_subject)
    # flags: make_forward_solution, calc_inverse_operator
    flags = calc_fwd_inv_wrapper(subject, args, conditions, flags, mri_subject)
    # flags: calc_stc_per_condition
    flags, stcs_conds, stcs_num = calc_stc_per_condition_wrapper(subject, conditions, inverse_method, args, flags)
    # flags: calc_labels_avg_per_condition
    flags = calc_labels_avg_per_condition_wrapper(
        subject, conditions, args.atlas, inverse_method, stcs_conds, args, flags, stcs_num, raw, epochs,
        args.overwrite_labels_data)

    if utils.should_run(args, 'read_sensors_layout'):
        flags['read_sensors_layout'] = read_sensors_layout(mri_subject, args)

    if utils.should_run(args, 'save_vertex_activity_map'):
        stc_fnames = [STC_HEMI_SMOOTH.format(cond='{cond}', method=inverse_method, hemi=hemi)
                      for hemi in utils.HEMIS]
        get_meg_files(subject, stc_fnames, args, conditions)
        flags['save_vertex_activity_map'] = save_vertex_activity_map(conditions, stat, stcs_conds_smooth, inverse_method)

    if 'calc_labels_avg_for_rest' in args.function:
        flags['calc_labels_avg_for_rest'] = calc_labels_avg_for_rest(
            args.atlas, inverse_method, None, args.pick_ori, args.extract_mode, args.snr, args.raw_fname,
            args.inv_fname, args.labels_data_template, args.overwrite_stc, args.overwrite_labels_data,
            args.fwd_usingMEG, args.fwd_usingEEG, cond_name='all', positive=False, moving_average_win_size=0,
            save_data_files=True, n_jobs=args.n_jobs)

    # functions that aren't in the main pipeline
    if 'smooth_stc' in args.function:
        stc_fnames = [STC_HEMI.format(cond='{cond}', method=inverse_method, hemi=hemi) for hemi in utils.HEMIS]
        get_meg_files(subject, stc_fnames, args, conditions)
        flags['smooth_stc'], stcs_conds_smooth = smooth_stc(
            conditions, stcs_conds, inverse_method, args.stc_t, args.morph_to_subject, args.n_jobs)

    if 'save_activity_map' in args.function:
        stc_fnames = [STC_HEMI_SMOOTH.format(cond='{cond}', method=inverse_method, hemi=hemi)
                      for hemi in utils.HEMIS]
        get_meg_files(subject, stc_fnames, args, conditions)
        flags['save_activity_map'] = save_activity_map(
            conditions, stat, stcs_conds_smooth, inverse_method, args.save_smoothed_activity, args.morph_to_subject,
            args.stc_t, args.norm_by_percentile, args.norm_percs)

    if 'find_functional_rois_in_stc' in args.function:
        flags['find_functional_rois_in_stc'] = find_functional_rois_in_stc(
            subject, mri_subject, args.atlas, args.stc_name, args.threshold, args.threshold_is_precentile,
            args.peak_stc_time_index, args.label_name_template, args.peak_mode,
            args.extract_mode, args.min_cluster_max, args.min_cluster_size, args.clusters_label,
            inv_fname=args.inv_fname, fwd_usingMEG=args.fwd_usingMEG, fwd_usingEEG=args.fwd_usingEEG,
            recreate_src_spacing=args.recreate_src_spacing, n_jobs=args.n_jobs)

    if 'print_files_names' in args.function:
        # also called in init_globals
        print_files_names()

    if 'calc_single_trial_labels_per_condition' in args.function:
        calc_single_trial_labels_per_condition(args.atlas, conditions, stcs_conds, extract_mode=args.extract_mode)

    sub_corticals_codes_file = op.join(MMVT_DIR, 'sub_cortical_codes.txt')
    if 'calc_sub_cortical_activity' in args.function:
        # todo: call get_meg_files
        calc_sub_cortical_activity(conditions, sub_corticals_codes_file, inverse_method, args.pick_ori, evoked, epochs)
    if 'save_subcortical_activity_to_blender' in args.function:
        save_subcortical_activity_to_blender(sub_corticals_codes_file, conditions, stat, inverse_method,
                                             args.norm_by_percentile, args.norm_percs)
    if 'plot_sub_cortical_activity' in args.function:
        plot_sub_cortical_activity(conditions, sub_corticals_codes_file, inverse_method=inverse_method)

    if 'calc_activity_significance' in args.function:
        calc_activity_significance(conditions, inverse_method, stcs_conds)

    if 'save_activity_map_minmax' in args.function:
        flags['save_activity_map_minmax'] = save_activity_map_minmax(
            None, conditions, stat, stcs_conds_smooth, inverse_method, args.morph_to_subject,
            args.norm_by_percentile, args.norm_percs, False)

    if 'remove_artifacts' in args.function:
        flags['remove_artifacts'] = remove_artifacts(
            n_max_ecg=args.ica_n_max_ecg, n_max_eog=args.ica_n_max_eog, n_components=args.ica_n_components,
            method=args.ica_method, remove_from_raw=args.remove_artifacts_from_raw, overwrite_ica=args.overwrite_ica,
            overwrite_raw=args.ica_overwrite_raw)

    if 'morph_stc' in args.function:
        flags['morph_stc'] = morph_stc(
            MRI_SUBJECT, conditions, args.morph_to_subject, args.inverse_method[0], args.grade,
            args.smoothing_iterations, args.overwrite_stc, args.n_jobs)

    if 'calc_stc_diff' in args.function:
        flags['calc_stc_diff'] = calc_stc_diff_both_hemis(
            conditions, STC_HEMI, args.inverse_method[0], overwrite_stc=args.overwrite_stc)

    if 'load_fieldtrip_volumetric_data' in args.function:
        flags['load_fieldtrip_volumetric_data'] = load_fieldtrip_volumetric_data(
            subject, args.fieldtrip_data_name, args.fieldtrip_data_field_name, args.overwrite_nii_file,
            args.overwrite_surface, args.overwrite_stc)

    if 'fit_ica_on_subjects' in args.function:
        fit_ica_on_subjects(args.subject, meg_root_fol=args.meg_root_fol)

    return flags


def read_cmd_args(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='MMVT anatomy preprocessing')
    parser.add_argument('-m', '--mri_subject', help='mri subject name', required=False, default=None, type=au.str_arr_type)
    parser.add_argument('-t', '--task', help='task name', required=False, default='')
    parser.add_argument('-c', '--conditions', help='conditions', required=False, default='all', type=au.str_arr_type)
    parser.add_argument('-i', '--inverse_method', help='inverse_method', required=False, default='dSPM', type=au.str_arr_type)
    parser.add_argument('--modality', help='', required=False, default='meg')
    parser.add_argument('--sub_dirs_for_tasks', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--get_task_defaults', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--fname_format', help='', required=False, default='{subject}-{ana_type}.{file_type}')
    parser.add_argument('--fname_format_cond', help='', required=False, default='{subject}_{cond}-{ana_type}.{file_type}')
    parser.add_argument('--data_per_task', help='task-subject-data', required=False, default=0, type=au.is_true)
    parser.add_argument('--raw_fname_format', help='', required=False, default='')
    parser.add_argument('--raw_fname', help='', required=False, default='')
    parser.add_argument('--fwd_fname_format', help='', required=False, default='')
    parser.add_argument('--inv_fname_format', help='', required=False, default='')
    parser.add_argument('--inv_fname', help='', required=False, default='')
    parser.add_argument('--fwd_fname', help='', required=False, default='')
    parser.add_argument('--epo_fname', help='', required=False, default='')
    parser.add_argument('--evo_fname', help='', required=False, default='')
    parser.add_argument('--noise_cov_fname', help='', required=False, default='')
    parser.add_argument('--calc_evoked_for_all_epoches', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_epochs', help='overwrite_epochs', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_evoked', help='overwrite_evoked', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_sensors', help='overwrite_sensors', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_fwd', help='overwrite_fwd', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_inv', help='overwrite_inv', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_stc', help='overwrite_stc', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_labels_data', help='overwrite_labels_data', required=False, default=0, type=au.is_true)
    parser.add_argument('--read_events_from_file', help='read_events_from_file', required=False, default=0, type=au.is_true)
    parser.add_argument('--events_file_name', help='events_file_name', required=False, default='')
    parser.add_argument('--windows_length', help='', required=False, default=1000, type=int)
    parser.add_argument('--windows_shift', help='', required=False, default=500, type=int)
    parser.add_argument('--windows_num', help='', required=False, default=0, type=int)
    parser.add_argument('--bad_channels', help='bad_channels', required=False, default=[], type=au.str_arr_type)
    parser.add_argument('--calc_epochs_from_raw', help='calc_epochs_from_raw', required=False, default=0, type=au.is_true)
    parser.add_argument('--l_freq', help='low freq filter', required=False, default=None, type=float)
    parser.add_argument('--h_freq', help='high freq filter', required=False, default=None, type=float)
    parser.add_argument('--pick_meg', help='pick meg events', required=False, default=1, type=au.is_true)
    parser.add_argument('--pick_eeg', help='pick eeg events', required=False, default=1, type=au.is_true)
    parser.add_argument('--pick_eog', help='pick eog events', required=False, default=0, type=au.is_true)
    parser.add_argument('--remove_power_line_noise', help='remove power line noise', required=False, default=1, type=au.is_true)
    parser.add_argument('--power_line_freq', help='power line freq', required=False, default=60, type=int)
    parser.add_argument('--stim_channels', help='stim_channels', required=False, default='STI001', type=au.str_arr_type)
    parser.add_argument('--reject', help='reject trials', required=False, default=1, type=au.is_true)
    parser.add_argument('--reject_grad', help='', required=False, default=4000e-13, type=float)
    parser.add_argument('--reject_mag', help='', required=False, default=4e-12, type=float)
    parser.add_argument('--reject_eog', help='', required=False, default=150e-6, type=float)
    parser.add_argument('--apply_SSP_projection_vectors', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--add_eeg_ref', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--pick_ori', help='', required=False, default=None)
    parser.add_argument('--t_min', help='', required=False, default=0.0, type=float)
    parser.add_argument('--t_max', help='', required=False, default=0.0, type=float)
    parser.add_argument('--noise_t_min', help='', required=False, default=None, type=au.float_or_none)
    parser.add_argument('--noise_t_max', help='', required=False, default=0, type=float)
    parser.add_argument('--snr', help='', required=False, default=3.0, type=float)
    parser.add_argument('--calc_stc_for_all', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--stc_t_min', help='', required=False, default=None, type=float)
    parser.add_argument('--stc_t_max', help='', required=False, default=None, type=float)
    parser.add_argument('--baseline_min', help='', required=False, default=None, type=float)
    parser.add_argument('--baseline_max', help='', required=False, default=0, type=au.float_or_none)
    parser.add_argument('--files_includes_cond', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--inv_no_cond', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--fwd_no_cond', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--contrast', help='', required=False, default='')
    parser.add_argument('--cleaning_method', help='', required=False, default='') # nTSSS
    parser.add_argument('--fwd_usingMEG', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--fwd_usingEEG', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--fwd_calc_corticals', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--fwd_calc_subcorticals', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--fwd_recreate_source_space', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--recreate_src_spacing', help='', required=False, default='oct6')
    parser.add_argument('--recreate_src_surface', help='', required=False, default='white')
    parser.add_argument('--inv_loose', help='', required=False, default=0.2, type=float)
    parser.add_argument('--inv_depth', help='', required=False, default=0.8, type=float)
    parser.add_argument('--use_raw_for_noise_cov', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--use_empty_room_for_noise_cov', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_noise_cov', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--inv_calc_cortical', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--inv_calc_subcorticals', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--evoked_flip_positive', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--evoked_moving_average_win_size', help='', required=False, default=0, type=int)
    parser.add_argument('--normalize_evoked', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--calc_max_min_diff', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--raw_template', help='', required=False, default='*raw.fif')
    parser.add_argument('--eve_template', help='', required=False, default='*eve.fif')
    parser.add_argument('--stc_template', help='', required=False, default='')
    parser.add_argument('--labels_data_template', help='', required=False, default='')
    parser.add_argument('--save_stc', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--stc_t', help='', required=False, default=-1, type=int)
    parser.add_argument('--calc_stc_diff', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--morph_to_subject', help='', required=False, default='')
    parser.add_argument('--single_trial_stc', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--apply_on_raw', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--extract_mode', help='', required=False, default='mean_flip', type=au.str_arr_type)
    parser.add_argument('--read_only_from_annot', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--colors_map', help='', required=False, default='OrRd')
    parser.add_argument('--save_smoothed_activity', help='', required=False, default=True, type=au.is_true)
    parser.add_argument('--normalize_data', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--norm_by_percentile', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--norm_percs', help='', required=False, default='1,99', type=au.int_arr_type)
    parser.add_argument('--remote_subject_meg_dir', help='remote_subject_dir', required=False, default='')
    parser.add_argument('--meg_root_fol', required=False, default='')
    # parser.add_argument('--sftp_sso', help='ask for sftp pass only once', required=False, default=0, type=au.is_true)
    parser.add_argument('--eeg_electrodes_excluded_from_mesh', help='', required=False, default='', type=au.str_arr_type)
    # **** ICA *****
    parser.add_argument('--ica_n_components', required=False, default=0.95, type=float)
    parser.add_argument('--ica_n_max_ecg', required=False, default=3, type=int)
    parser.add_argument('--ica_n_max_eog', required=False, default=1, type=int)
    parser.add_argument('--ica_method', required=False, default='fastica')
    parser.add_argument('--overwrite_ica', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--remove_artifacts_from_raw', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--ica_overwrite_raw', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--do_plot_ica', help='', required=False, default=0, type=au.is_true)
    # Clusters
    parser.add_argument('--stc_name', required=False, default='')
    parser.add_argument('--threshold', required=False, default=75, type=float)
    parser.add_argument('--threshold_is_precentile', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--peak_stc_time_index', required=False, default=None, type=au.int_or_none)
    parser.add_argument('--label_name_template', required=False, default='')
    parser.add_argument('--peak_mode', required=False, default='abs')
    parser.add_argument('--min_cluster_max', required=False, default=0, type=float)
    parser.add_argument('--min_cluster_size', required=False, default=0, type=int)
    parser.add_argument('--clusters_label', required=False, default='')
    # FieldTrip
    parser.add_argument('--fieldtrip_data_name', required=False, default='')
    parser.add_argument('--fieldtrip_data_field_name', required=False, default='')
    parser.add_argument('--overwrite_nii_file', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_surface', help='', required=False, default=0, type=au.is_true)
    # Smoothing / Morphing
    parser.add_argument('--grade', help='', required=False, default=5, type=au.int_or_none)
    parser.add_argument('--smoothing_iterations', help='', required=False, default=None, type=au.int_or_none)

    pu.add_common_args(parser)
    args = utils.Bag(au.parse_parser(parser, argv))
    args.mri_necessary_files = {'surf': ['rh.pial', 'lh.pial', 'rh.sphere.reg', 'lh.sphere.reg'],
                                'label': ['{}.{}.annot'.format(hemi, args.atlas) for hemi in utils.HEMIS]}
    if not args.mri_subject:
        args.mri_subject = args.subject
    if args.baseline_min is None and args.baseline_max is None:
        args.baseline = None
    else:
        args.baseline = (args.baseline_min, args.baseline_max)
    if args.task == 'rest':
        if not args.apply_on_raw:
            # todo: check why it was set to True
            # args.single_trial_stc = True
            args.calc_epochs_from_raw = True
        args.use_empty_room_for_noise_cov = True
        args.baseline_min = 0
        args.baseline_max = 0
    # todo: Was set as a remark, why?
    if args.n_jobs == -1:
        args.n_jobs = utils.get_n_jobs(args.n_jobs)
    if args.function == 'rest_functions':
        args.function = 'calc_epochs,make_forward_solution,calc_inverse_operator,calc_stc_per_condition,' + \
                        'calc_labels_avg_per_condition'
    # print(args)
    return args


def get_subjects_itr_func(args):
    subjects_itr = product(zip(args.subject, args.mri_subject), args.inverse_method)
    subject_func = lambda x:x[0][1]
    return subjects_itr, subject_func


def call_main(args):
    subjects_itr, subject_func = get_subjects_itr_func(args)
    pu.run_on_subjects(args, main, subjects_itr, subject_func)


if __name__ == '__main__':
    args = read_cmd_args()
    call_main(args)
    print('finish!')
