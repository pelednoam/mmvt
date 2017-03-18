import os
import os.path as op
import time
import matplotlib.pyplot as plt
import glob
import shutil
import numpy as np
from functools import partial
from collections import defaultdict
from itertools import product
import traceback
import mne
import types
from mne.minimum_norm import (make_inverse_operator, apply_inverse,
                              write_inverse_operator, read_inverse_operator)

from src.utils import utils
from src.utils import preproc_utils as pu
from src.utils import labels_utils as lu

SUBJECTS_MRI_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()

LINKS_DIR = utils.get_links_dir()
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
LOOKUP_TABLE_SUBCORTICAL = op.join(MMVT_DIR, 'sub_cortical_codes.txt')

STAT_AVG, STAT_DIFF = range(2)
HEMIS = ['rh', 'lh']

SUBJECT, MRI_SUBJECT, SUBJECT_MEG_FOLDER, RAW, INFO, EVO, EVE, COV, EPO, EPO_NOISE, FWD, FWD_EEG, FWD_SUB, FWD_X, FWD_SMOOTH, INV,\
INV_EEG, INV_SMOOTH, INV_EEG_SMOOTH, INV_SUB, INV_X, EMPTY_ROOM, MRI, SRC, SRC_SMOOTH, BEM, STC, STC_HEMI, STC_HEMI_SMOOTH, STC_HEMI_SMOOTH_SAVE, STC_ST,\
COR, LBL, STC_MORPH, ACT, ASEG, DATA_COV, NOISE_COV, DATA_CSD, NOISE_CSD, MEG_TO_HEAD_TRANS = [''] * 41


def init_globals_args(subject, mri_subject, fname_format, fname_format_cond, subjects_meg_dir, subjects_mri_dir,
                      mmvt_dir, args):
    return init_globals(subject, mri_subject, fname_format, fname_format_cond, args.raw_fname_format,
                 args.fwd_fname_format, args.inv_fname_format, args.events_file_name, args.files_includes_cond,
                 args.cleaning_method, args.contrast, subjects_meg_dir, args.task, subjects_mri_dir, mmvt_dir,
                 args.fwd_no_cond, args.inv_no_cond, args.data_per_task)


def init_globals(subject, mri_subject='', fname_format='', fname_format_cond='', raw_fname_format='',
                 fwd_fname_format='', inv_fname_format='', events_fname='', files_includes_cond=False,
                 cleaning_method='', contrast='', subjects_meg_dir='', task='', subjects_mri_dir='', mmvt_dir='',
                 fwd_no_cond=False, inv_no_cond=False, data_per_task=False):
    global SUBJECT, MRI_SUBJECT, SUBJECT_MEG_FOLDER, RAW, INFO, EVO, EVE, COV, EPO, EPO_NOISE, FWD, FWD_EEG, FWD_SUB, FWD_X,\
        FWD_SMOOTH, INV, INV_EEG, INV_SMOOTH, INV_EEG_SMOOTH, INV_SUB, INV_X, EMPTY_ROOM, MRI, SRC, SRC_SMOOTH, BEM, STC, STC_HEMI, STC_HEMI_SMOOTH, \
        STC_HEMI_SMOOTH_SAVE, STC_ST, COR, AVE, LBL, STC_MORPH, ACT, ASEG, MMVT_SUBJECT_FOLDER, DATA_COV, NOISE_COV, \
        DATA_CSD, NOISE_CSD, MEG_TO_HEAD_TRANS
    if files_includes_cond:
        fname_format = fname_format_cond
    SUBJECT = subject
    MRI_SUBJECT = mri_subject if mri_subject!='' else subject
    os.environ['SUBJECT'] = SUBJECT
    if task != '' and data_per_task:
        SUBJECT_MEG_FOLDER = op.join(subjects_meg_dir, task, SUBJECT)
    else:
        SUBJECT_MEG_FOLDER = op.join(subjects_meg_dir, SUBJECT)
    # if not op.isdir(SUBJECT_MEG_FOLDER):
    #     SUBJECT_MEG_FOLDER = op.join(subjects_meg_dir)
    # if not op.isdir(SUBJECT_MEG_FOLDER):
    #     raise Exception("Can't find the subject's MEG folder! {}".format(SUBJECT_MEG_FOLDER))
    utils.make_dir(SUBJECT_MEG_FOLDER)
    print('Subject meg dir: {}'.format(SUBJECT_MEG_FOLDER))
    SUBJECT_MRI_FOLDER = op.join(subjects_mri_dir, MRI_SUBJECT)
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
    NOISE_COV = _get_fif_name('noise-cov')
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
    STC_HEMI_SMOOTH = _get_stc_name('{method}-smoothed-{hemi}')
    STC_HEMI_SMOOTH_SAVE = op.splitext(STC_HEMI_SMOOTH)[0].replace('-{hemi}','')
    STC_MORPH = op.join(MEG_DIR, task, '{}', '{}-{}-inv.stc') # cond, method
    STC_ST = _get_pkl_name('{method}_st')
    LBL = op.join(SUBJECT_MEG_FOLDER, 'labels_data_{}_{}.npz') # atlas, extract_method, hemi
    ACT = op.join(MMVT_SUBJECT_FOLDER, 'activity_map_{}') # hemi
    # MRI files
    MRI = op.join(SUBJECT_MRI_FOLDER, 'mri', 'transforms', '{}-trans.fif'.format(MRI_SUBJECT))
    SRC = op.join(SUBJECT_MRI_FOLDER, 'bem', '{}-oct-6p-src.fif'.format(MRI_SUBJECT))
    SRC_SMOOTH = op.join(SUBJECT_MRI_FOLDER, 'bem', '{}-all-src.fif'.format(MRI_SUBJECT))
    BEM = op.join(SUBJECT_MRI_FOLDER, 'bem', '{}-5120-5120-5120-bem-sol.fif'.format(MRI_SUBJECT))
    COR = op.join(SUBJECT_MRI_FOLDER, 'mri', 'T1-neuromag', 'sets', 'COR.fif')
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
    if fname_format=='':
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


def load_raw(bad_channels=[], l_freq=None, h_freq=None):
    # read the data
    raw = mne.io.read_raw_fif(RAW, preload=True)
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
                 windows_num=0):
    epoches_fname = EPO if epoches_fname is None else epoches_fname

    picks = mne.pick_types(raw.info, meg=pick_meg, eeg=pick_eeg, eog=pick_eog, exclude='bads')
    # events[:, 2] = [str(ev)[event_digit] for ev in events[:, 2]]
    reject_dict = dict(grad=reject_grad, mag=reject_mag) if reject else None
    if not reject_dict is None and pick_eog:
        reject['eog'] = reject_eog
    # epochs = find_epoches(raw, picks, events, events, tmin=tmin, tmax=tmax)
    if remove_power_line_noise:
        raw.notch_filter(np.arange(power_line_freq, power_line_freq * 4 + 1, power_line_freq), picks=picks)
        # raw.notch_filter(np.arange(60, 241, 60), picks=picks)
    if read_events_from_file and events is None and op.isfile(EVE):
        # events_fname = events_fname if events_fname != '' else EVE
        print('read events from {}'.format(EVE))
        events = mne.read_events(EVE)
    else:
        if events is None:
            try:
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
    epochs = mne.Epochs(raw, events, conditions, tmin, tmax, proj=True, picks=picks,
                        baseline=baseline, preload=True, reject=reject_dict)
    if '{cond}' in epoches_fname:
        for event in epochs.event_id: #events.keys():
            epochs[event].save(get_cond_fname(epoches_fname, event))
    else:
        try:
            epochs.save(epoches_fname)
        except:
            print(traceback.format_exc())
    return epochs


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
        None, args.calc_epochs_from_raw, args.stim_channels,
        args.pick_meg, args.pick_eeg, args.pick_eog, args.reject,
        args.reject_grad, args.reject_mag, args.reject_eog, args.remove_power_line_noise, args.power_line_freq,
        args.bad_channels, args.l_freq, args.h_freq, args.task, args.windows_length, args.windows_shift,
        args.windows_num)


def calc_epochs_wrapper(
        conditions, tmin, tmax, baseline, raw=None, read_events_from_file=False, events_mat=None,
        calc_epochs_from_raw=False, stim_channels=None, pick_meg=True, pick_eeg=False, pick_eog=False,
        reject=True, reject_grad=4000e-13, reject_mag=4e-12, reject_eog=150e-6, remove_power_line_noise=True,
        power_line_freq=60, bad_channels=[], l_freq=None, h_freq=None, task='', windows_length=1000, windows_shift=500,
        windows_num=0):
    # Calc evoked data for averaged data and for each condition
    try:
        epo_exist = False
        if not calc_epochs_from_raw:
            if '{cond}' in EPO:
                epo_exist = True
                epochs = {}
                for cond in conditions.keys():
                    if op.isfile(get_cond_fname(EPO, cond)):
                        epochs[cond] = mne.read_epochs(get_cond_fname(EPO, cond))
                    else:
                        epo_exist = False
                        break
            else:
                epo_exist = op.isfile(EPO) and not calc_epochs_from_raw
                if epo_exist:
                    epochs = mne.read_epochs(EPO)
        if not epo_exist or calc_epochs_from_raw:
            if raw is None:
                raw = load_raw(bad_channels, l_freq, h_freq)
            epochs = calc_epoches(raw, conditions, tmin, tmax, baseline, read_events_from_file, events_mat,
                                  stim_channels, pick_meg, pick_eeg, pick_eog, reject,
                                  reject_grad, reject_mag, reject_eog, remove_power_line_noise, power_line_freq,
                                  None, task, windows_length, windows_shift, windows_num)
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


def calc_evokes(epochs, events, mri_subject, norm_by_percentile=False, norm_percs=None):
    try:
        events_keys = list(events.keys())
        if epochs is None:
            epochs = mne.read_epochs(EPO)
        evokes = [epochs[event].average() for event in events_keys]
        save_evokes_to_mmvt(evokes, events_keys, mri_subject, norm_by_percentile, norm_percs)
        if '{cond}' in EVO:
            # evokes = {event: epochs[event].average() for event in events_keys}
            for event, evoked in zip(events_keys, evokes):
                mne.write_evokeds(get_cond_fname(EVO, event), evoked)
        else:
            # evokes = [epochs[event].average() for event in events_keys]
            mne.write_evokeds(EVO, evokes)
    except:
        print(traceback.format_exc())
        return False
    else:
        if '{cond}' in EVO:
            flag = all([op.isfile(get_cond_fname(EVO, event)) for event in evokes.keys()])
        else:
            flag = op.isfile(EVO)
    return flag, evokes


def save_evokes_to_mmvt(evokes, events_keys, mri_subject, norm_by_percentile=False, norm_percs=None):
    fol = op.join(MMVT_DIR, mri_subject, 'meg')
    utils.make_dir(fol)
    meg_indices = [k for k, name in enumerate(evokes[0].ch_names) if name.startswith('MEG')]
    ch_names = [evokes[0].ch_names[k] for k in meg_indices]
    dt = np.diff(evokes[0].times[:2])[0]
    data = np.zeros((len(meg_indices), evokes[0].data.shape[1], len(events_keys)))
    for event_ind, event in enumerate(events_keys):
        data[:, :, event_ind] = evokes[event_ind].data[meg_indices]
    data = utils.normalize_data(data, norm_by_percentile, norm_percs)
    np.save(op.join(fol, 'meg_sensors_evoked_data.npy'), data)
    np.savez(op.join(fol, 'meg_sensors_evoked_data_meta.npz'), names=ch_names, conditions=events_keys, dt=dt)


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
    ply_vertives_num = utils.get_ply_vertices_num(op.join(SUBJECTS_MRI_DIR, MRI_SUBJECT, 'surf', '{}.pial.ply'))
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
    src_fname = SRC if op.isfile(SRC) else op.join(SUBJECTS_MRI_DIR, mri_subject, 'bem', '{}-{}-{}-src.fif'.format(
        mri_subject, recreate_src_spacing[:3], recreate_src_spacing[3:]))
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
            src = mne.setup_source_space(MRI_SUBJECT, spacing=recreate_src_spacing, surface=recreate_src_surface,
                                         overwrite=True, n_jobs=n_jobs)
        else:
            raise Exception("Can't calculate the fwd solution without the source")
    return src


def check_bem(mri_subject):
    if not op.isfile(BEM):
        prepare_subject_folder(
            mri_subject, args.remote_subject_dir, SUBJECTS_MRI_DIR,
            {'bem': [utils.namesbase_with_ext(BEM)]}, args)
    if not op.isfile(BEM):
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
        model = mne.make_bem_model(mri_subject)
        bem = mne.make_bem_solution(model)
        mne.write_bem_solution(BEM, bem)


def make_forward_solution(mri_subject, events, sub_corticals_codes_file='', usingMEG=True, usingEEG=True, calc_corticals=True,
        calc_subcorticals=True, recreate_the_source_space=False, recreate_src_spacing='oct6',
        recreate_src_surface='white', overwrite_fwd=False, n_jobs=4):
    fwd, fwd_with_subcortical = None, None
    fwd_fname = FWD if usingMEG else FWD_EEG
    try:
        src = check_src(mri_subject, recreate_the_source_space, recreate_src_spacing,recreate_src_surface, n_jobs)
        check_src_ply_vertices_num(src)
        check_bem(mri_subject)
        sub_corticals = utils.read_sub_corticals_code_file(sub_corticals_codes_file)
        if '{cond}' not in EPO:
            if calc_corticals:
                if overwrite_fwd or not op.isfile(fwd_fname):
                    fwd = _make_forward_solution(src, EPO, usingMEG, usingEEG, n_jobs)
                    mne.write_forward_solution(fwd_fname, fwd, overwrite=True)
            if calc_subcorticals and len(sub_corticals) > 0:
                # add a subcortical volumes
                if overwrite_fwd or not op.isfile(FWD_SUB):
                    src_with_subcortical = add_subcortical_volumes(src, sub_corticals)
                    fwd_with_subcortical = _make_forward_solution(src_with_subcortical, EPO, usingMEG, usingEEG, n_jobs)
                    mne.write_forward_solution(FWD_SUB, fwd_with_subcortical, overwrite=True)
        else:
            for cond in events.keys():
                if calc_corticals:
                    fwd_cond_fname = get_cond_fname(fwd_fname, cond)
                    if overwrite_fwd or not op.isfile(fwd_cond_fname):
                        fwd = _make_forward_solution(src, get_cond_fname(EPO, cond), usingMEG, usingEEG, n_jobs)
                        mne.write_forward_solution(fwd_cond_fname, fwd, overwrite=True)
                if calc_subcorticals and len(sub_corticals) > 0:
                    # add a subcortical volumes
                    fwd_cond_fname = get_cond_fname(FWD_SUB, cond)
                    if overwrite_fwd or not op.isfile(fwd_cond_fname):
                        src_with_subcortical = add_subcortical_volumes(src, sub_corticals)
                        fwd_with_subcortical = _make_forward_solution(src_with_subcortical, get_cond_fname(EPO, cond),
                                                                      usingMEG, usingEEG, n_jobs)
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
    fwd = mne.make_forward_solution(info=epo, trans=COR, src=src, bem=BEM, # mri=MRI
                                    meg=usingMEG, eeg=usingEEG, mindist=5.0,
                                    n_jobs=n_jobs, overwrite=True)
    return fwd


def add_subcortical_surfaces(src, seg_labels):
    """Adds a subcortical volume to a cortical source space
    """
    from mne.source_space import _make_discrete_source_space

    # Read the freesurfer lookup table
    lut = utils.read_freesurfer_lookup_table(FREESURFER_HOME)

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


def calc_noise_cov(epochs):
    if len(epochs) > 1:
        noise_cov = mne.compute_covariance(epochs.crop(None, 0)) #, copy=True))
    else:
        if op.isfile(EPO_NOISE):
            demi_epochs = mne.read_epochs(EPO_NOISE)
        else:
            raise Exception("You should split first your epochs into small demi epochs, see calc_demi_epoches")
        noise_cov = calc_noise_cov(demi_epochs)
    return noise_cov


def calc_inverse_operator(events, inv_loose=0.2, inv_depth=0.8, use_empty_room_for_noise_cov=False,
                          overwrite_noise_cov=False, calc_for_cortical_fwd=True, calc_for_sub_cortical_fwd=True,
                          fwd_usingMEG=True, fwd_usingEEG=True, calc_for_spec_sub_cortical=False, cortical_fwd=None, subcortical_fwd=None,
                          spec_subcortical_fwd=None, region=None):
    conds = ['all'] if '{cond}' not in EPO else events.keys()
    fwd_fname = FWD_EEG if fwd_usingEEG and not fwd_usingMEG else FWD
    inv_fname = INV_EEG if fwd_usingEEG and not fwd_usingMEG else INV
    for cond in conds:
        if (not calc_for_cortical_fwd or op.isfile(get_cond_fname(inv_fname, cond))) and \
                (not calc_for_sub_cortical_fwd or op.isfile(get_cond_fname(INV_SUB, cond))) and \
                (not calc_for_spec_sub_cortical or op.isfile(get_cond_fname(INV_X, cond, region=region))):
            continue
        try:
            epo = get_cond_fname(EPO, cond)
            epochs = mne.read_epochs(epo)
            if use_empty_room_for_noise_cov:
                raw_empty_room = mne.io.read_raw_fif(EMPTY_ROOM, add_eeg_ref=False)
                noise_cov = mne.compute_raw_covariance(raw_empty_room, tmin=0, tmax=None)
                noise_cov.save(NOISE_COV)
            elif overwrite_noise_cov or not op.isfile(NOISE_COV):
                noise_cov = calc_noise_cov(epochs)
                noise_cov.save(NOISE_COV)
            else:
                noise_cov = mne.read_cov(NOISE_COV)
            # todo: should use noise_cov = calc_cov(...
            if calc_for_cortical_fwd and not op.isfile(get_cond_fname(inv_fname, cond)):
                if cortical_fwd is None:
                    cortical_fwd = get_cond_fname(fwd_fname, cond)
                _calc_inverse_operator(cortical_fwd, get_cond_fname(inv_fname, cond), epochs, noise_cov, inv_loose, inv_depth)
            if calc_for_sub_cortical_fwd and not op.isfile(get_cond_fname(INV_SUB, cond)):
                if subcortical_fwd is None:
                    subcortical_fwd = get_cond_fname(FWD_SUB, cond)
                _calc_inverse_operator(subcortical_fwd, get_cond_fname(INV_SUB, cond), epochs, noise_cov, None, None)
            if calc_for_spec_sub_cortical and not op.isfile(get_cond_fname(INV_X, cond, region=region)):
                if spec_subcortical_fwd is None:
                    spec_subcortical_fwd = get_cond_fname(FWD_X, cond, region=region)
                _calc_inverse_operator(spec_subcortical_fwd, get_cond_fname(INV_X, cond, region=region), epochs,
                                       noise_cov, None, None)
            flag = True
        except:
            print(traceback.format_exc())
            print('Error in calculating inv for {}'.format(cond))
            flag = False
    return flag


def _calc_inverse_operator(fwd_name, inv_name, epochs, noise_cov, inv_loose=0.2, inv_depth=0.8):
    fwd = mne.read_forward_solution(fwd_name)
    inverse_operator = make_inverse_operator(epochs.info, fwd, noise_cov,
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


def calc_stc_per_condition(events, stc_t_min=None, stc_t_max=None, inverse_method='dSPM', baseline=(None, 0),
                           apply_SSP_projection_vectors=True, add_eeg_ref=True, pick_ori=None,
                           single_trial_stc=False, save_stc=True):
    # todo: If the evoked is the raw (no events), we need to seperate it into N events with different ids, to avoid memory error
    if single_trial_stc:
        from mne.minimum_norm import apply_inverse_epochs
    stcs, stcs_num = {}, {}
    snr = 3.0
    lambda2 = 1.0 / snr ** 2
    global_inverse_operator = False
    if '{cond}' not in INV:
        inverse_operator = read_inverse_operator(INV)
        global_inverse_operator = True
    for cond_name in events.keys():
        try:
            if not global_inverse_operator:
                inverse_operator = read_inverse_operator(INV.format(cond=cond_name))
            if single_trial_stc:
                epo_cond = get_cond_fname(EPO, cond_name)
                epochs = mne.read_epochs(epo_cond, apply_SSP_projection_vectors, add_eeg_ref)
                stcs[cond_name] = apply_inverse_epochs(epochs, inverse_operator, lambda2, inverse_method,
                    pick_ori=pick_ori, return_generator=True)
                stcs_num[cond_name] = epochs.events.shape[0]
                # if save_stc:
                #     utils.save(stcs[cond_name], STC_ST.format(cond=cond_name, method=inverse_method))
            else:
                evoked = get_evoked_cond(cond_name, baseline, apply_SSP_projection_vectors, add_eeg_ref)
                if not stc_t_min is None and not stc_t_max is None:
                    evoked = evoked.crop(stc_t_min, stc_t_max)
                stcs[cond_name] = apply_inverse(evoked, inverse_operator, lambda2, inverse_method, pick_ori=pick_ori)
                if save_stc:
                    stcs[cond_name].save(STC.format(cond=cond_name, method=inverse_method)[:-4])
            flag = True
        except:
            print(traceback.format_exc())
            print('Error with {}!'.format(cond_name))
            flag = False
    return flag, stcs, stcs_num


def get_evoked_cond(cond_name, baseline=(None, 0), apply_SSP_projection_vectors=True, add_eeg_ref=True):
    if '{cond}' not in EVO:
        try:
            evoked = mne.read_evokeds(EVO, condition=cond_name, baseline=baseline)
        except:
            print('No evoked data with the condition {}'.format(cond_name))
            evoked = None
    else:
        evo_cond = get_cond_fname(EVO, cond_name)
        if op.isfile(evo_cond):
            evoked = mne.read_evokeds(evo_cond, baseline=baseline)[0]
        else:
            print('No evoked file, trying to use epo file')
            if '{cond}' not in EPO:
                epochs = mne.read_epochs(EPO, apply_SSP_projection_vectors, add_eeg_ref)
                evoked = epochs[cond_name].average()
            else:
                epo_cond = get_cond_fname(EPO, cond_name)
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
    lut = utils.read_freesurfer_lookup_table(FREESURFER_HOME)
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
        calc_inverse_operator(events, inv_loose, inv_depth, False, False, False, True, region=region)
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
    lut = utils.read_freesurfer_lookup_table(FREESURFER_HOME)
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
    lut = utils.read_freesurfer_lookup_table(FREESURFER_HOME)
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


def morph_stc(subject_to, cond='all', grade=None, n_jobs=6, inverse_method='dSPM'):
    stc = mne.read_source_estimate(STC.format(cond, inverse_method))
    vertices_to = mne.grade_to_vertices(subject_to, grade=grade)
    stc_to = mne.morph_data(SUBJECT, subject_to, stc, n_jobs=n_jobs, grade=vertices_to)
    fol_to = op.join(MEG_DIR, TASK, subject_to)
    if not op.isdir(fol_to):
        os.mkdir(fol_to)
    stc_to.save(STC_MORPH.format(subject_to, cond, inverse_method))


def calc_stc_for_all_vertices(stc, n_jobs=6):
    vertices_to = mne.grade_to_vertices(MRI_SUBJECT, grade=None)
    return mne.morph_data(MRI_SUBJECT, MRI_SUBJECT, stc, n_jobs=n_jobs, grade=vertices_to)


def smooth_stc(events, stcs_conds=None, inverse_method='dSPM', n_jobs=6):
    try:
        stcs = {}
        for ind, cond in enumerate(events.keys()):
            if stcs_conds is not None:
                stc = stcs_conds[cond]
            else:
                # Can read only for the 'rh', it'll also read the second file for 'lh'. Strange...
                stc = mne.read_source_estimate(STC_HEMI.format(cond=cond, method=inverse_method, hemi='rh'))
            stc_smooth = calc_stc_for_all_vertices(stc, n_jobs)
            check_stc_with_ply(stc_smooth, cond)
            stc_smooth.save(STC_HEMI_SMOOTH_SAVE.format(cond=cond, method=inverse_method))
            stcs[cond] = stc_smooth
        flag = True
    except:
        print(traceback.format_exc())
        print('Error in calculating inv for {}'.format(cond))
        flag = False

    return flag, stcs


def check_stc_with_ply(stc, cond_name):
    for hemi in HEMIS:
        stc_vertices = stc.rh_vertno if hemi=='rh' else stc.lh_vertno
        print('{} {} stc vertices: {}'.format(hemi, cond_name, len(stc_vertices)))
        ply_vertices, _ = utils.read_ply_file(op.join(MMVT_SUBJECT_FOLDER, 'surf', '{}.pial.ply'.format(hemi)))
        print('{} {} ply vertices: {}'.format(hemi, cond_name, len(stc_vertices)))
        if len(stc_vertices) != ply_vertices.shape[0]:
            raise Exception('check_stc_with_ply: Wrong number of vertices!')
    print('check_stc_with_ply: ok')


def save_activity_map(events, stat, stcs_conds=None, inverse_method='dSPM', norm_by_percentile=False, norm_percs=(1,99),
                      plot_cb=False):
    try:
        if stat not in [STAT_DIFF, STAT_AVG]:
            raise Exception('stat not in [STAT_DIFF, STAT_AVG]!')
        stcs = get_stat_stc_over_conditions(events, stat, stcs_conds, inverse_method, smoothed=True)
        save_activity_map_minmax(stcs, events, stat, stcs_conds, inverse_method, norm_by_percentile,
                                 norm_percs, plot_cb)
        for hemi in HEMIS:
            verts, faces = utils.read_pial_npz(MRI_SUBJECT, MMVT_DIR, hemi)
            data = stcs[hemi]
            if verts.shape[0] != data.shape[0]:
                raise Exception('save_activity_map: wrong number of vertices!')
            else:
                print('Both {}.pial.ply and the stc file have {} vertices'.format(hemi, data.shape[0]))
            fol = '{}'.format(ACT.format(hemi))
            utils.delete_folder_files(fol)
            # data = data / data_max
            now = time.time()
            T = data.shape[1]
            for t in range(T):
                utils.time_to_go(now, t, T, runs_num_to_print=10)
                np.save(op.join(fol, 't{}'.format(t)), data[:, t])

        flag = True
    except:
        print(traceback.format_exc())
        print('Error in save_activity_map')
        flag = False
    return flag


def save_activity_map_minmax(stcs=None, events=None, stat=STAT_DIFF, stcs_conds=None, inverse_method='dSPM',
                             norm_by_percentile=False, norm_percs=(1,99), plot_cb=False):
    from src.utils import color_maps_utils as cp
    from src.utils import figures_utils as figu

    output_fname = op.join(MMVT_DIR, MRI_SUBJECT, 'meg_activity_map_minmax.pkl')
    if stcs is None:
        if stat not in [STAT_DIFF, STAT_AVG]:
            raise Exception('stat not in [STAT_DIFF, STAT_AVG]!')
        stcs = get_stat_stc_over_conditions(events, stat, stcs_conds, inverse_method, smoothed=True)
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
            verts, faces = utils.read_pial_npz(MRI_SUBJECT, MMVT_DIR, hemi)
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


def get_stat_stc_over_conditions(events, stat, stcs_conds=None, inverse_method='dSPM', smoothed=False):
    stcs = {}
    stc_template = STC_HEMI if not smoothed else STC_HEMI_SMOOTH
    for cond_ind, cond in enumerate(events.keys()):
        if stcs_conds is None:
            # Reading only the rh, the lh will be read too
            print('Reading {}'.format(stc_template.format(cond=cond, method=inverse_method, hemi='lh')))
            stc = mne.read_source_estimate(stc_template.format(cond=cond, method=inverse_method, hemi='lh'))
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
    utils.labels_to_annot(MRI_SUBJECT, SUBJECTS_MRI_DIR, parc_name, labels_fol, overwrite)


def calc_single_trial_labels_per_condition(atlas, events, stcs, extract_mode='mean_flip', src=None):
    global_inverse_operator = False
    if '{cond}' not in INV:
        global_inverse_operator = True
        if src is None:
            inverse_operator = read_inverse_operator(INV)
            src = inverse_operator['src']

    for (cond_name, cond_id), stc in zip(events.items(), stcs.values()):
        if not global_inverse_operator:
            if src is None:
                inverse_operator = read_inverse_operator(INV.format(cond=cond_name))
                src = inverse_operator['src']
        labels = lu.read_labels(MRI_SUBJECT, SUBJECTS_MRI_DIR, atlas)
        labels_ts = mne.extract_label_time_course(stcs[cond_name], labels, src, mode=extract_mode,
                                                  return_generator=False, allow_empty=True)
        np.save(op.join(SUBJECT_MEG_FOLDER, 'labels_ts_{}'.format(cond_name)), np.array(labels_ts))


def calc_labels_avg_per_condition(atlas, hemi, events, surf_name='pial', labels_fol='', stcs=None, stcs_num={},
        extract_mode='mean_flip', inverse_method='dSPM', positive=False, moving_average_win_size=0,
        norm_by_percentile=True, norm_percs=(1,99), labels_output_fname_template='', src=None,
        do_plot=False, n_jobs=1):
    try:
        labels_output_fname = LBL.format(atlas, hemi) if labels_output_fname_template == '' else \
            labels_output_fname_template.format(hemi=hemi)
        labels_norm_output_fname = op.join(SUBJECT_MEG_FOLDER, 'labels_data_norm_{}_{}.npz'.format(atlas, hemi))
        lables_mmvt_fname = op.join(MMVT_DIR, MRI_SUBJECT, op.basename(LBL.format(atlas, hemi)))
        # if op.isfile(labels_output_fname) and op.isfile(labels_norm_output_fname):
        #     if not op.isfile(lables_mmvt_fname):
        #         shutil.copyfile(labels_output_fname, lables_mmvt_fname)
        #         return True
        if stcs is None:
            stcs = {}
            for cond in events.keys():
                stc_fname = STC_HEMI.format(cond=cond, method=inverse_method, hemi=hemi)
                if not op.isfile(stc_fname):
                    raise Exception("Can't find the stc file! {}".format(stc_fname))
                stcs[cond] = mne.read_source_estimate(stc_fname)

        global_inverse_operator = False
        if '{cond}' not in INV:
            global_inverse_operator = True
            if src is None:
                inverse_operator = read_inverse_operator(INV)
                src = inverse_operator['src']

        if do_plot:
            utils.make_dir(op.join(SUBJECT_MEG_FOLDER, 'figures'))

        conds_incdices = {cond_id:ind for ind, cond_id in zip(range(len(stcs)), events.values())}
        conditions = []
        labels_data = None

        for (cond_name, cond_id), stc_cond in zip(events.items(), stcs.values()):
            if do_plot:
                plt.figure()
            conditions.append(cond_name)
            if not global_inverse_operator:
                if src is None:
                    inverse_operator = read_inverse_operator(INV.format(cond=cond_name))
                    src = inverse_operator['src']
            # labels = lu.read_hemi_labels(MRI_SUBJECT, SUBJECTS_MRI_DIR, atlas, hemi, surf_name, labels_fol)
            labels = lu.read_labels(MRI_SUBJECT, SUBJECTS_MRI_DIR, atlas, hemi=hemi, surf_name=surf_name,
                                    labels_fol=labels_fol, read_only_from_annot=True, n_jobs=n_jobs)
            if len(labels) == 0:
                raise Exception('No labels!!!')

            if isinstance(stc_cond, types.GeneratorType):
                stc_cond_num = stcs_num[cond_name]
            else:
                stc_cond = [stc_cond]
                stc_cond_num = 1
            for stc_ind, stc in enumerate(stc_cond):
                for ind, label in enumerate(labels):
                    mean_flip = stc.extract_label_time_course(label, src, mode=extract_mode, allow_empty=True)
                    mean_flip = np.squeeze(mean_flip)
                    # Set flip to be always positive
                    # mean_flip *= np.sign(mean_flip[np.argmax(np.abs(mean_flip))])
                    if labels_data is None:
                        T = len(stc.times)
                        labels_data = np.zeros((len(labels), T, len(stcs), stc_cond_num))
                    labels_data[ind, :, conds_incdices[cond_id], stc_ind] = mean_flip
                    if do_plot:
                        plt.plot(labels_data[ind, :, conds_incdices[cond_id]], label=label.name)

            if do_plot:
                plt.xlabel('time (ms)')
                plt.title('{}: {} {}'.format(cond_name, hemi, atlas))
                plt.legend()
                # plt.show()
                plt.savefig(op.join(SUBJECT_MEG_FOLDER, 'figures', '{}: {} {}.png'.format(cond_name, hemi, atlas)))

        labels_data = labels_data.squeeze()
        if positive or moving_average_win_size > 0:
            labels_data = utils.make_evoked_smooth_and_positive(labels_data, positive, moving_average_win_size)
        print('Saving to {}'.format(labels_output_fname))
        labels_names = [utils.to_str(l.name) for l in labels]
        np.savez(labels_output_fname, data=labels_data, names=labels_names, conditions=conditions)
        # Normalize the data
        data_max, data_min = utils.get_data_max_min(labels_data, norm_by_percentile, norm_percs)
        max_abs = utils.get_max_abs(data_max, data_min)
        labels_data = labels_data / max_abs
        np.savez(labels_norm_output_fname, data=labels_data, names=labels_names, conditions=conditions)
        shutil.copyfile(labels_output_fname, lables_mmvt_fname)
        flag = True
    except:
        print(traceback.format_exc())
        print('Error in calc_labels_avg_per_condition inv')
        flag = False
    return flag


def read_sensors_layout(subject, mri_subject, args, pick_meg=True, pick_eeg=False):
    if pick_eeg and pick_meg or (not pick_meg and not pick_eeg):
        raise Exception('read_sensors_layout: You should pick only meg or eeg!')
    if not op.isfile(INFO):
        raw = mne.io.read_raw_fif(RAW)
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
        output_fname = op.join(MMVT_DIR, mri_subject, 'eeg', 'eeg_positions.npz')
    ret = False
    if len(sensors_pos) > 0:
        # trans_files = glob.glob(op.join(SUBJECTS_MRI_DIR, '*COR*.fif'))
        trans_file = COR
        if not op.isfile(trans_file):
            trans_pat = op.join(MEG_DIR, args.task, subject, '*COR*.fif')
            trans_files = glob.glob(trans_pat)
            if len(trans_files) == 1:
                trans_file = trans_files[0]
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


def check_both_hemi_in_stc(events):
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
        np.savez(op.join(MMVT_SUBJECT_FOLDER, 'meg_labels_coloring_{}.npz'.format(hemi)),
            data=data, colors=colors, names=labels_names[hemi])
        np.savez(op.join(MMVT_SUBJECT_FOLDER, 'meg_labels_coloring_no_t{}.npz'.format(hemi)),
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
        args.task, args.fname_format,args.fname_format_cond, args.conditions)


def get_fname_format(task, fname_format='', fname_format_cond='', args_conditions=('all')):
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
        # conditions = dict(all=1)
    if args_conditions[0] != 'all':
        conditions = dict((cond_name, cond_id + 1) for cond_id, cond_name in enumerate(args_conditions))
    return fname_format, fname_format_cond, conditions


def prepare_subject_folder(subject, remote_subject_dir, local_subjects_dir, necessary_files, sftp_args):
    return utils.prepare_subject_folder(
        necessary_files, subject, remote_subject_dir, local_subjects_dir,
        sftp_args.sftp, sftp_args.sftp_username, sftp_args.sftp_domain, sftp_args.sftp_password,
        False, sftp_args.print_traceback)


def get_meg_files(subject, necessary_fnames, args, events):
    fnames = []
    for necessary_fname in necessary_fnames:
        fname = os.path.basename(necessary_fname)
        if '{cond}' in fname:
            fnames.extend([get_cond_fname(fname, event) for event in events.keys()])
        else:
            fnames.append(fname)
    local_fol = op.join(MEG_DIR, args.task)
    prepare_subject_folder(subject, args.remote_subject_meg_dir, local_fol, {'.': fnames}, args)


def calc_fwd_inv_wrapper(subject, mri_subject, conditions, args, flags):
    inv_fname = INV_EEG if args.fwd_usingEEG and not args.fwd_usingMEG else INV
    get_meg_files(subject, [inv_fname], args, conditions)
    if args.overwrite_inv or not op.isfile(inv_fname) or (args.inv_calc_subcorticals and not op.isfile(INV_SUB)):
        if utils.should_run(args, 'make_forward_solution'):
            prepare_subject_folder(
                mri_subject, args.remote_subject_dir, SUBJECTS_MRI_DIR,
                {op.join('mri', 'T1-neuromag', 'sets'): ['COR.fif']}, args)
            src_dic = dict(bem=['{}-oct-6p-src.fif'.format(mri_subject)])
            create_src_dic = dict(surf=['lh.{}'.format(args.recreate_src_surface), 'rh.{}'.format(args.recreate_src_surface),
                       'lh.sphere', 'rh.sphere'])
            for nec_file in [src_dic, create_src_dic]:
                file_exist = prepare_subject_folder(
                    mri_subject, args.remote_subject_dir, SUBJECTS_MRI_DIR,
                    nec_file, args)
                if file_exist:
                    break
            get_meg_files(subject, [EPO], args, conditions)
            sub_corticals_codes_file = op.join(MMVT_DIR, 'sub_cortical_codes.txt')
            flags['make_forward_solution'], fwd, fwd_subs = make_forward_solution(
                mri_subject, conditions, sub_corticals_codes_file, args.fwd_usingMEG, args.fwd_usingEEG,
                args.fwd_calc_corticals, args.fwd_calc_subcorticals, args.fwd_recreate_source_space,
                args.recreate_src_spacing, args.recreate_src_surface, args.overwrite_fwd, args.n_jobs)

        if utils.should_run(args, 'calc_inverse_operator'):
            get_meg_files(subject, [EPO, FWD], args, conditions)
            flags['calc_inverse_operator'] = calc_inverse_operator(
                conditions, args.inv_loose, args.inv_depth, args.use_empty_room_for_noise_cov,
                args.overwrite_noise_cov, args.inv_calc_cortical, args.inv_calc_subcorticals,
                args.fwd_usingMEG, args.fwd_usingEEG)
    return flags


def calc_evokes_wrapper(subject, mri_subject, conditions, args, flags):
    evoked, epochs = None, None
    if utils.should_run(args, 'calc_epochs'):
        necessary_files = calc_epochs_necessary_files(args)
        get_meg_files(subject, necessary_files, args, conditions)
        flags['calc_epochs'], epochs = calc_epochs_wrapper_args(conditions, args)

    if utils.should_run(args, 'calc_evokes'):
        flags['calc_evokes'], evoked = calc_evokes(
            epochs, conditions, mri_subject, args.norm_by_percentile, args.norm_percs)

    return flags, evoked, epochs


def calc_stc_per_condition_wrapper(subject, conditions, inverse_method, args, flags):
    stcs_conds, stcs_num = None, {}
    if utils.should_run(args, 'calc_stc_per_condition'):
        get_meg_files(subject, [INV, EVO], args, conditions)
        flags['calc_stc_per_condition'], stcs_conds, stcs_num = calc_stc_per_condition(
            conditions, args.stc_t_min, args.stc_t_max, inverse_method, args.baseline, args.apply_SSP_projection_vectors,
            args.add_eeg_ref, args.pick_ori, args.single_trial_stc, args.save_stc)
    return flags, stcs_conds, stcs_num


def calc_labels_avg_per_condition_wrapper(subject, conditions, atlas, inverse_method, stcs_conds, args, flags,
                                          stcs_num={}):
    if utils.should_run(args, 'calc_labels_avg_per_condition'):
        stc_fnames = [STC_HEMI.format(cond='{cond}', method=inverse_method, hemi=hemi) for hemi in utils.HEMIS]
        get_meg_files(subject, stc_fnames + [INV], args, conditions)
        for hemi_ind, hemi in enumerate(HEMIS):
            flags['calc_labels_avg_per_condition_{}'.format(hemi)] = calc_labels_avg_per_condition(
                args.atlas, hemi, conditions, extract_mode=args.extract_mode,
                inverse_method=inverse_method, positive=args.evoked_flip_positive,
                moving_average_win_size=args.evoked_moving_average_win_size,
                norm_by_percentile=args.norm_by_percentile, norm_percs=args.norm_percs,
                stcs=stcs_conds, stcs_num=stcs_num, n_jobs=args.n_jobs)
            if stcs_conds and isinstance(stcs_conds[list(conditions.keys())[0]], types.GeneratorType) and hemi_ind == 0:
                # Create the stc generator again for the second hemi
                _, stcs_conds, stcs_num = calc_stc_per_condition_wrapper(
                    subject, conditions, inverse_method, args, flags)

    if utils.should_run(args, 'calc_labels_min_max'):
        min_max_output_fname = op.join(MMVT_DIR, MRI_SUBJECT, 'meg_labels_{}_minmax.npz'.format(atlas))
        if utils.both_hemi_files_exist(op.join(MMVT_DIR, MRI_SUBJECT, op.basename(LBL.format(atlas, '{hemi}')))):
            labels_data_rh = np.load(op.join(MMVT_DIR, MRI_SUBJECT, op.basename(LBL.format(atlas, 'rh'))))
            labels_data_lh = np.load(op.join(MMVT_DIR, MRI_SUBJECT, op.basename(LBL.format(atlas, 'lh'))))
            labels_min = min([np.min(labels_data_rh['data']), np.min(labels_data_lh['data'])])
            labels_max = max([np.max(labels_data_rh['data']), np.max(labels_data_lh['data'])])
            labels_diff_min = min([np.min(np.diff(labels_data_rh['data'])), np.min(np.diff(labels_data_lh['data']))])
            labels_diff_max = max([np.max(np.diff(labels_data_rh['data'])), np.max(np.diff(labels_data_lh['data']))])
            np.savez(min_max_output_fname, labels_minmax=[labels_min, labels_max],
                     labels_diff_minmax=[labels_diff_min, labels_diff_max])
        else:
            print("Can't find {}!".format(op.join(MMVT_DIR, MRI_SUBJECT, op.basename(LBL.format(atlas, '{hemi}')))))
        flags['calc_labels_min_max'] = op.isfile(min_max_output_fname)

    return flags


# def calc_labels_data_from_activity_map(mri_subject, atlas):
#     for hemi in utils.HEMIS:
#         labels = lu.read_labels(mri_subject, SUBJECTS_MRI_DIR, atlas, hemi=hemi)
#         activity_map_fol = op.join(MMVT_DIR, mri_subject, 'activity_map_{}'.format(hemi))
#         activity_files = glob.glob(op.join(activity_map_fol, 't*.npy'))
#         labels_data =


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


def main(tup, remote_subject_dir, args, flags):
    (subject, mri_subject), inverse_method = tup
    evoked, epochs, raw = None, None, None
    stcs_conds, stcs_conds_smooth = None, None
    fname_format, fname_format_cond, conditions = init_main(subject, mri_subject, remote_subject_dir, args)
    init_globals_args(
        subject, mri_subject, fname_format, fname_format_cond, MEG_DIR, SUBJECTS_MRI_DIR, MMVT_DIR, args)
    stat = STAT_AVG if len(conditions) == 1 else STAT_DIFF

    # flags: calc_evoked
    flags, evoked, epochs = calc_evokes_wrapper(subject, mri_subject, conditions, args, flags)
    # flags: make_forward_solution, calc_inverse_operator
    flags = calc_fwd_inv_wrapper(subject, mri_subject, conditions, args, flags)
    # flags: calc_stc_per_condition
    flags, stcs_conds, stcs_num = calc_stc_per_condition_wrapper(subject, conditions, inverse_method, args, flags)
    # flags: calc_labels_avg_per_condition
    flags = calc_labels_avg_per_condition_wrapper(subject, conditions, args.atlas, inverse_method, stcs_conds, args, flags, stcs_num)

    if utils.should_run(args, 'read_sensors_layout'):
        flags['read_sensors_layout'] = read_sensors_layout(subject, mri_subject, args)

    if utils.should_run(args, 'smooth_stc'):
        stc_fnames = [STC_HEMI.format(cond='{cond}', method=inverse_method, hemi=hemi) for hemi in utils.HEMIS]
        get_meg_files(subject, stc_fnames, args, conditions)
        flags['smooth_stc'], stcs_conds_smooth = smooth_stc(conditions, stcs_conds, inverse_method, args.n_jobs)

    if utils.should_run(args, 'save_activity_map'):
        stc_fnames = [STC_HEMI_SMOOTH.format(cond='{cond}', method=inverse_method, hemi=hemi)
                      for hemi in utils.HEMIS]
        get_meg_files(subject, stc_fnames, args, conditions)
        flags['save_activity_map'] = save_activity_map(
            conditions, stat, stcs_conds_smooth, inverse_method, args.norm_by_percentile, args.norm_percs)

    if utils.should_run(args, 'save_vertex_activity_map'):
        stc_fnames = [STC_HEMI_SMOOTH.format(cond='{cond}', method=inverse_method, hemi=hemi)
                      for hemi in utils.HEMIS]
        get_meg_files(subject, stc_fnames, args, conditions)
        flags['save_vertex_activity_map'] = save_vertex_activity_map(conditions, stat, stcs_conds_smooth, inverse_method)

    # functions that aren't in the main pipeline

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
            None, conditions, stat, stcs_conds_smooth, inverse_method, args.norm_by_percentile, args.norm_percs, False)

    return flags


def read_cmd_args(argv=None):
    import argparse
    from src.utils import args_utils as au
    parser = argparse.ArgumentParser(description='MMVT anatomy preprocessing')
    parser.add_argument('-m', '--mri_subject', help='mri subject name', required=False, default=None, type=au.str_arr_type)
    parser.add_argument('-t', '--task', help='task name', required=False, default='')
    parser.add_argument('-c', '--conditions', help='conditions', required=False, default='all', type=au.str_arr_type)
    parser.add_argument('-i', '--inverse_method', help='inverse_method', required=False, default='dSPM', type=au.str_arr_type)
    parser.add_argument('--fname_format', help='', required=False, default='{subject}-{ana_type}.{file_type}')
    parser.add_argument('--fname_format_cond', help='', required=False, default='{subject}_{cond}-{ana_type}.{file_type}')
    parser.add_argument('--data_per_task', help='task-subject-data', required=False, default=0, type=au.is_true)
    parser.add_argument('--raw_fname_format', help='', required=False, default='')
    parser.add_argument('--fwd_fname_format', help='', required=False, default='')
    parser.add_argument('--inv_fname_format', help='', required=False, default='')
    parser.add_argument('--overwrite_fwd', help='overwrite_fwd', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_inv', help='overwrite_inv', required=False, default=0, type=au.is_true)
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
    parser.add_argument('--stim_channels', help='stim_channels', required=False, default=None, type=au.str_arr_type)
    parser.add_argument('--reject', help='reject trials', required=False, default=1, type=au.is_true)
    parser.add_argument('--reject_grad', help='', required=False, default=4000e-13, type=float)
    parser.add_argument('--reject_mag', help='', required=False, default=4e-12, type=float)
    parser.add_argument('--reject_eog', help='', required=False, default=150e-6, type=float)
    parser.add_argument('--apply_SSP_projection_vectors', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--add_eeg_ref', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--pick_ori', help='', required=False, default=None)
    parser.add_argument('--t_min', help='', required=False, default=0.0, type=float)
    parser.add_argument('--t_max', help='', required=False, default=0.0, type=float)
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
    parser.add_argument('--use_empty_room_for_noise_cov', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_noise_cov', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--inv_calc_cortical', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--inv_calc_subcorticals', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--evoked_flip_positive', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--evoked_moving_average_win_size', help='', required=False, default=0, type=int)
    parser.add_argument('--normalize_evoked', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--save_stc', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--single_trial_stc', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--extract_mode', help='', required=False, default='mean_flip')
    parser.add_argument('--colors_map', help='', required=False, default='OrRd')
    parser.add_argument('--norm_by_percentile', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--norm_percs', help='', required=False, default='1,99', type=au.int_arr_type)
    parser.add_argument('--remote_subject_meg_dir', help='remote_subject_dir', required=False, default='')
    # parser.add_argument('--sftp_sso', help='ask for sftp pass only once', required=False, default=0, type=au.is_true)
    parser.add_argument('--eeg_electrodes_excluded_from_mesh', help='', required=False, default='', type=au.str_arr_type)
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
        args.single_trial_stc = True
        calc_epochs_from_raw = True
        args.use_empty_room_for_noise_cov = True
        args.baseline_min = 0
        args.baseline_max = 0
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
