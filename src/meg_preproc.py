import os
import utils
import matplotlib.pyplot as plt
import glob
import shutil
import numpy as np
from functools import partial
from collections import defaultdict
import mne
from mne.minimum_norm import (make_inverse_operator, apply_inverse,
                              write_inverse_operator, read_inverse_operator, apply_inverse_epochs)

from make_ecr_events import make_ecr_events

SUBJECTS_MEG_DIR = utils.get_exisiting_dir(['/homes/5/npeled/space3/MEG', '/home/noam/subjects/meg'])
SUBJECTS_MRI_DIR = utils.get_exisiting_dir(['/homes/5/npeled/space3/subjects', '/home/noam/subjects/mri']) # '/autofs/space/lilli_001/users/DARPA-MEG/freesurfs'
FREE_SURFER_HOME = utils.get_exisiting_dir([os.environ.get('FREESURFER_HOME', ''),
    '/usr/local/freesurfer/stable5_3_0', '/home/noam/freesurfer'])
print('FREE_SURFER_HOME: {}'.format(FREE_SURFER_HOME))
BEHAVIOR_FILE = utils.get_exisiting_file(
    ['/space/lilli/1/users/DARPA-MEG/ecr/behavior/hc004_ECR_MEG_2015_03_08_14_11_13.csv',
     '/home/noam/subjects/meg/ECR/hc004_ECR_MEG_2015_03_08_14_11_13.csv'])
LOOKUP_TABLE_SUBCORTICAL = os.path.join(utils.get_exisiting_dir([
    '/autofs/space/franklin_003/users/npeled/visualization_blender',
    '/media/noam/e2d1c2d2-8ac0-46ac-806b-74c5a8a3db9d/home/noam/blender_visualiztion_tool']), 'sub_cortical_codes.txt')

os.environ['SUBJECTS_DIR'] = SUBJECTS_MRI_DIR
BLENDER_ROOT_FOLDER = '/homes/5/npeled/space3/visualization_blender/'
TASK_MSIT, TASK_ECR = range(2)
TASKS = {TASK_MSIT: 'MSIT', TASK_ECR: 'ECR'}

SUBJECT, MRI_SUBJECT, SUBJECT_MEG_FOLDER, RAW, EVO, EVE, COV, EPO, FWD, FWD_SUB, INV, INV_SUB, \
MRI, SRC, BEM, STC, STC_HEMI, STC_HEMI_SMOOTH, STC_HEMI_SMOOTH_SAVE, COR, LBL, STC_MORPH, ACT, ASEG = '', '', '', \
    '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''

def initGlobals(subject, mri_subject, fname_format=''):
    global SUBJECT, MRI_SUBJECT, SUBJECT_MEG_FOLDER, RAW, EVO, EVE, COV, EPO, FWD, FWD_SUB, INV, INV_SUB, \
        MRI, SRC, BEM, STC, STC_HEMI, STC_HEMI_SMOOTH, STC_HEMI_SMOOTH_SAVE, COR, AVE, LBL, STC_MORPH, ACT, ASEG, \
        BLENDER_SUBJECT_FOLDER
    SUBJECT = subject
    MRI_SUBJECT = mri_subject if mri_subject!='' else subject
    os.environ['SUBJECT'] = SUBJECT
    SUBJECT_MEG_FOLDER = os.path.join(SUBJECTS_MEG_DIR, TASKS[TASK], SUBJECT)
    SUBJECT_MRI_FOLDER = os.path.join(SUBJECTS_MRI_DIR, MRI_SUBJECT)
    BLENDER_SUBJECT_FOLDER = os.path.join(BLENDER_ROOT_FOLDER, MRI_SUBJECT)
    _get_fif_name = partial(get_file_name, fname_format=fname_format, file_type='fif')
    _get_stc_name = partial(get_file_name, fname_format=fname_format, file_type='stc')
    RAW = _get_fif_name('raw')
    EVO = _get_fif_name('ave')
    EVE = _get_fif_name('eve')
    COV = _get_fif_name('cov')
    EPO = _get_fif_name('epo')
    FWD = _get_fif_name('fwd')
    FWD_SUB = _get_fif_name('sub-cortical-fwd')
    INV = _get_fif_name('inv')
    INV_SUB = _get_fif_name('sub-cortical-inv')
    STC = _get_stc_name('{method}')
    STC_HEMI = _get_stc_name('{method}-{hemi}')
    STC_HEMI_SMOOTH = _get_stc_name('{method}-smoothed-{hemi}')
    STC_HEMI_SMOOTH_SAVE = get_file_name('{method}-smoothed', '', fname_format)[:-1]
    STC_MORPH = os.path.join(SUBJECTS_MEG_DIR, TASKS[TASK], '{}', '{}-{}-inv.stc') # cond, method
    LBL = os.path.join(SUBJECT_MEG_FOLDER, 'labels_data_{}.npz')
    ACT = os.path.join(BLENDER_SUBJECT_FOLDER, 'activity_map_{}') # hemi
    # MRI files
    MRI = os.path.join(SUBJECT_MRI_FOLDER, 'mri', 'transforms', '{}-trans.fif'.format(SUBJECT))
    SRC = os.path.join(SUBJECT_MRI_FOLDER, 'bem', '{}-oct-6p-src.fif'.format(SUBJECT))
    BEM = os.path.join(SUBJECT_MRI_FOLDER, 'bem', '{}-5120-5120-5120-bem-sol.fif'.format(SUBJECT))
    COR = os.path.join(SUBJECT_MRI_FOLDER, 'mri', 'T1-neuromag', 'sets', 'COR.fif')
    ASEG = os.path.join(SUBJECT_MRI_FOLDER, 'ascii')


def get_file_name(ana_type, file_type='fif', fname_format=''):
    if fname_format=='':
        fname_format = '{subject}-{ana_type}.{file_type}'
    if utils.how_many_curlies(fname_format) == 2:
        fname = fname_format.format(subject=SUBJECT, ana_type=ana_type, file_type=file_type)
    else:
        fname = fname_format.format(subject=SUBJECT, ana_type=ana_type, cond='{cond}', file_type=file_type)
    return os.path.join(SUBJECT_MEG_FOLDER, fname)


def load_raw():
    # read the data
    raw = mne.io.Raw(RAW, preload=True)
    print(raw)
    return raw


def filter(raw):
    raw.filter(l_freq=1.0, h_freq=50.0)
    raw.save(RAW, overwrite=True)


def calcNoiseCov(epoches):
    noiseCov = mne.compute_covariance(epoches, tmax=None)
    # regularize noise covariance
    # noiseCov = mne.cov.regularize(noiseCov, evoked.info,
    #     mag=0.05, proj=True) # grad=0.05, eeg=0.1
    noiseCov.save(COV)
    # allEpoches = findEpoches(raw, picks, events, dict(onset=20), tmin=0, tmax=3.5)
    # evoked = allEpoches['onset'].average()
    # evoked.save(EVO)


def calc_epoches(raw, event_digit, events_id, tmin, tmax, readEventsFromFile=False, eventsFileName='', fileToSave=''):
    if (fileToSave == ''):
        fileToSave = EPO
    if (readEventsFromFile):
        print('read events from {}'.format(eventsFileName))
        try:
            events = mne.read_events(eventsFileName)
        except:
            print('error reading events file! {}'.format(eventsFileName))
            events = None
    else:
        events = mne.find_events(raw, stim_channel='STI001')
    if (events is not None):
        print(events)
        picks = mne.pick_types(raw.info, meg=True)
        events[:, 2] = [str(ev)[event_digit] for ev in events[:, 2]]
        epochs = find_epoches(raw, picks, events, events_id, tmin=tmin, tmax=tmax)
        # compute evoked response and noise covariance,and plot evoked
        epochs.save(fileToSave)
        return epochs
    else:
        return None


def createEventsFiles(behavior_file, pattern):
    make_ecr_events(RAW, behavior_file, EVE, pattern)


def calc_evoked(event_digit, events_id, tmin, tmax, raw=None, read_events_from_file=False, eventsFileName=''):
    # Calc evoked data for averaged data and for each condition
    if raw is None:
        raw = load_raw()
    epochs = calc_epoches(raw, event_digit, events_id, tmin, tmax, read_events_from_file, eventsFileName)
    all_evoked = calc_evoked_from_epochs(epochs, events_id)
    return all_evoked, epochs


def calc_evoked_from_epochs(epochs, events_id):
    evoked = epochs.average()
    evoked1 = epochs[events_id.keys()[0]].average()
    evoked2 = epochs[events_id.keys()[1]].average()
    all_evoked = [evoked, evoked1, evoked2]
    mne.write_evokeds(EVO, all_evoked)
    return all_evoked


def equalize_epoch_counts(events_id, method='mintime'):
    if utils.how_many_curlies(EPO) == 0:
        epochs = mne.read_epochs(EPO)
    else:
        epochs = []
        for cond_name in events_id.keys():
            epochs_cond = mne.read_epochs(EPO.format(cond=cond_name))
            epochs.append(epochs_cond)
    mne.epochs.equalize_epoch_counts(epochs, method='mintime')
    if utils.how_many_curlies(EPO) == 0:
        epochs.save(EPO)
    else:
        for cond_name, epochs in zip(events_id.keys(), epochs):
            epochs.save(EPO.format(cond=cond_name))
            

def find_epoches(raw, picks,  events, event_id, tmin, tmax, baseline=(None, 0)):
    # remove events that are not in the events table
    event_id = dict([(k, ev) for (k, ev) in event_id.iteritems() if ev in np.unique(events[:, 2])])
    return mne.Epochs(raw=raw, events=events, event_id=event_id, tmin=tmin, tmax=tmax, proj=True,
                      picks=picks, baseline=baseline, preload=True, reject=None) # reject can be dict(mag=4e-12)


def check_src_ply_vertices_num(src):
    # check the vertices num with the ply files
    ply_vertives_num = utils.get_ply_vertices_num(os.path.join(SUBJECTS_MRI_DIR, MRI_SUBJECT, 'surf', '{}.pial.ply'))
    if ply_vertives_num is not None:
        print(ply_vertives_num)
        src_vertices_num = [src_h['np'] for src_h in src]
        print(src_vertices_num)
        if not src_vertices_num[0] in ply_vertives_num.values() or \
                not src_vertices_num[1] in ply_vertives_num.values():
            raise Exception("src and ply files doesn't have the same vertices number! {}".format(SRC))
    else:
        print('No ply files to check the src!')


def make_forward_solution(events_id, sub_corticals_codes_file='', n_jobs=4, usingEEG=True, calc_only_subcorticals=False):
    # setup the cortical surface source space
    # src = mne.setup_source_space(MRI_SUBJECT, surface='pial',  overwrite=True) # overwrite=True)
    fwd, fwd_with_subcortical = None, None
    src = mne.read_source_spaces(SRC)
    check_src_ply_vertices_num(src)
    sub_corticals = utils.read_sub_corticals_code_file(sub_corticals_codes_file)
    if utils.how_many_curlies(EPO) == 0:
        if not calc_only_subcorticals:
            fwd = _make_forward_solution(src, EPO, usingEEG, n_jobs)
            mne.write_forward_solution(FWD, fwd, overwrite=True)
        if len(sub_corticals) > 0:
            # add a subcortical volumes
            src_with_subcortical = add_subcortical_volumes(src, sub_corticals)
            fwd_with_subcortical = _make_forward_solution(src_with_subcortical, EPO, usingEEG, n_jobs)
            mne.write_forward_solution(FWD_SUB, fwd_with_subcortical, overwrite=True)
    else:
        for cond in events_id.keys():
            if not calc_only_subcorticals:
                fwd = _make_forward_solution(src, get_cond_fname(EPO, cond), usingEEG, n_jobs)
                mne.write_forward_solution(get_cond_fname(FWD, cond), fwd, overwrite=True)
            if len(sub_corticals) > 0:
                # add a subcortical volumes
                src_with_subcortical = add_subcortical_volumes(src, sub_corticals)
                fwd_with_subcortical = _make_forward_solution(src_with_subcortical, get_cond_fname(EPO, cond), usingEEG, n_jobs)
                mne.write_forward_solution(get_cond_fname(FWD_SUB, cond), fwd_with_subcortical, overwrite=True)

    return fwd, fwd_with_subcortical


def _make_forward_solution(src, epo, usingEEG=True, n_jobs=6):
    fwd = mne.make_forward_solution(info=epo, trans=COR, src=src, bem=BEM, # mri=MRI
                                    meg=True, eeg=usingEEG, mindist=5.0,
                                    n_jobs=n_jobs, overwrite=True)
    return fwd


def add_subcortical_surfaces(src, seg_labels):
    """Adds a subcortical volume to a cortical source space
    """
    from mne.source_space import _make_discrete_source_space

    # Read the freesurfer lookup table
    lut = utils.read_freesurfer_lookup_table(FREE_SURFER_HOME)

    # Get the indices to the desired labels
    for label in seg_labels:

        # Get numeric index to label
        seg_name, seg_id = utils.get_numeric_index_to_label(label, lut)
        srf_file = os.path.join(ASEG, 'aseg_%.3d.srf' % seg_id)
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

    src = org_src.copy()

    # Find the segmentation file
    aseg_fname = os.path.join(SUBJECTS_MRI_DIR, MRI_SUBJECT, 'mri', 'aseg.mgz')
    aseg = nib.load(aseg_fname)
    aseg_hdr = aseg.get_header()
    sub_cortical_generator = utils.sub_cortical_voxels_generator(aseg, seg_labels, spacing, use_grid, FREE_SURFER_HOME)
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
        src.append(sp)

    return src


def calc_inverse_operator(events_id, calc_for_sub_cortical_fwd=True, calc_for_corticla_fwd=True):
    conds = ['all'] if utils.how_many_curlies(EPO)==0 else events_id.keys()
    for cond in conds:
        epo = get_cond_fname(EPO, cond)
        epochs = mne.read_epochs(epo)
        noise_cov = mne.compute_covariance(epochs.crop(None, 0, copy=True))
        if calc_for_corticla_fwd:
            forward = mne.read_forward_solution(get_cond_fname(FWD, cond))
            inverse_operator = make_inverse_operator(epochs.info, forward, noise_cov,
                loose=None, depth=None)
            write_inverse_operator(get_cond_fname(INV, cond), inverse_operator)
        if calc_for_sub_cortical_fwd:
            forward_sub = mne.read_forward_solution(get_cond_fname(FWD_SUB, cond))
            inverse_operator_sub = make_inverse_operator(epochs.info, forward_sub, noise_cov,
                loose=None, depth=None)
            write_inverse_operator(get_cond_fname(INV_SUB, cond), inverse_operator_sub)


def calc_stc(inverse_method='dSPM'):
    snr = 3.0
    lambda2 = 1.0 / snr ** 2
    inverse_operator = read_inverse_operator(INV)
    evoked = mne.read_evokeds(EVO, condition=0, baseline=(None, 0))
    stc = apply_inverse(evoked, inverse_operator, lambda2, inverse_method,
                        pick_ori=None)
    stc.save(STC.format('all', inverse_method))


def calc_stc_per_condition(events_id, inverse_method='dSPM', baseline=(None, 0), apply_SSP_projection_vectors=True,
                           add_eeg_ref=True):
    stcs = {}
    snr = 3.0
    lambda2 = 1.0 / snr ** 2
    global_inverse_operator = False
    if utils.how_many_curlies(INV) == 0:
        inverse_operator = read_inverse_operator(INV)
        global_inverse_operator = True
    for cond_name in events_id.keys():
        if not global_inverse_operator:
            inverse_operator = read_inverse_operator(INV.format(cond=cond_name))
        evoked = get_evoked_cond(cond_name, baseline, apply_SSP_projection_vectors, add_eeg_ref)
        stcs[cond_name] = apply_inverse(evoked, inverse_operator, lambda2, inverse_method,
                                        pick_ori=None)
        stcs[cond_name].save(STC.format(cond=cond_name, method=inverse_method)[:-4])
    return stcs


def get_evoked_cond(cond_name, baseline=(None, 0), apply_SSP_projection_vectors=True, add_eeg_ref=True):
    if utils.how_many_curlies(EVO) == 0:
        try:
            evoked = mne.read_evokeds(EVO, condition=cond_name, baseline=baseline)
        except:
            print('No evoked data with the condition {}'.format(cond_name))
            evoked = None
    else:
        evo_cond = get_cond_fname(EVO, cond_name)
        if os.path.isfile(evo_cond):
            evoked = mne.read_evokeds(evo_cond, baseline=baseline)[0]
        else:
            print('No evoked file, trying to use epo file')
            if utils.how_many_curlies(EPO) == 0:
                epochs = mne.read_epochs(EPO, apply_SSP_projection_vectors, add_eeg_ref)
                evoked = epochs[cond_name].average()
            else:
                epo_cond = get_cond_fname(EPO, cond_name)
                epochs = mne.read_epochs(epo_cond, apply_SSP_projection_vectors, add_eeg_ref)
                evoked = epochs.average()
            mne.write_evokeds(evo_cond, evoked)
    return evoked


def get_cond_fname(fname, cond):
    return fname if utils.how_many_curlies(fname) == 0 else fname.format(cond=cond)


def calc_sub_cortical_activity(events_id, sub_corticals_codes_file, inverse_method='dSPM'):
    sub_corticals = utils.read_sub_corticals_code_file(sub_corticals_codes_file)
    if len(sub_corticals) == 0:
        return

    snr = 3.0
    lambda2 = 1.0 / snr ** 2
    global_evo = False
    if utils.how_many_curlies(EVO) == 0:
        global_evo = True
        evoked = {event:mne.read_evokeds(EVO, condition=event, baseline=(None, 0)) for event in events_id.keys()}
        inverse_operator = read_inverse_operator(INV_SUB)

    lut = utils.read_freesurfer_lookup_table(FREE_SURFER_HOME)
    for event in events_id.keys():
        if not global_evo:
            evo = get_cond_fname(EVO, event)
            evoked = {event:mne.read_evokeds(evo, baseline=(None, 0))[0] for event in [event]}
            inverse_operator = read_inverse_operator(get_cond_fname(INV_SUB, event))

        if inverse_method=='lcmv':
            from mne.beamformer import lcmv
            epochs = mne.read_epochs(get_cond_fname(EPO, event))
            noise_cov = mne.compute_covariance(epochs.crop(None, 0, copy=True))
            data_cov = mne.compute_covariance(epochs, tmin=0.0, tmax=1.0, method='shrunk')
            forward = mne.read_forward_solution(get_cond_fname(FWD_SUB, event))
            stc = lcmv(evoked, forward, noise_cov, data_cov, reg=0.01)
        elif inverse_method=='dics':
            from mne.beamformer import dics
            from mne.time_frequency import compute_epochs_csd
            epochs = mne.read_epochs(get_cond_fname(EPO, event))
            data_csd = compute_epochs_csd(epochs, mode='multitaper', tmin=0.0, tmax=2.0,
                                          fmin=6, fmax=10)
            noise_csd = compute_epochs_csd(epochs, mode='multitaper', tmin=-0.5, tmax=0.0,
                                           fmin=6, fmax=10)
            forward = mne.read_forward_solution(get_cond_fname(FWD_SUB, event))
            stc = dics(evoked, forward, noise_csd, data_csd)
        else:
            stc = apply_inverse(evoked[event], inverse_operator, lambda2, inverse_method)
        # stc.extract_label_time_course(label, src, mode='mean_flip')
        read_vertices_from = len(stc.vertices[0])+len(stc.vertices[1])
        sub_corticals_activity = {}
        for sub_cortical_ind, sub_cortical_code in enumerate(sub_corticals):
            # +2 becasue the first two are the hemispheres
            sub_corticals_activity[sub_cortical_code] = stc.data[
                read_vertices_from: read_vertices_from + len(stc.vertices[sub_cortical_ind + 2])]
            read_vertices_from += len(stc.vertices[sub_cortical_ind + 2])

        if not os.path.isdir:
            os.mkdir(os.path.join(SUBJECT_MEG_FOLDER, 'subcorticals'))
        for sub_cortical_code, activity in sub_corticals_activity.iteritems():
            sub_cortical, _ = utils.get_numeric_index_to_label(sub_cortical_code, lut)
            np.save(os.path.join(SUBJECT_MEG_FOLDER, 'subcorticals', '{}-{}-{}'.format(event, sub_cortical, inverse_method)), activity.mean(0))
            np.save(os.path.join(SUBJECT_MEG_FOLDER, 'subcorticals', '{}-{}-{}-all-vertices'.format(event, sub_cortical, inverse_method)), activity)


def plot_sub_cortical_activity(events_id, sub_corticals_codes_file, inverse_method='dSPM'):
    lut = utils.read_freesurfer_lookup_table(FREE_SURFER_HOME)
    sub_corticals = utils.read_sub_corticals_code_file(sub_corticals_codes_file)
    for label in sub_corticals:
        sub_cortical, _ = utils.get_numeric_index_to_label(label, lut)
        print(sub_cortical)
        activity = {}
        f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
        fig_name = '{} ({}): {}, {}, contrast'.format(sub_cortical, inverse_method, events_id.keys()[0], events_id.keys()[1])
        ax1.set_title(fig_name)
        for event, ax in zip(events_id.keys(), [ax1, ax2]):
            activity[event] = np.load(os.path.join(SUBJECT_MEG_FOLDER, 'subcorticals', '{}-{}-{}-all-vertices.npy'.format(event, sub_cortical, inverse_method)))
            ax.plot(activity[event].T)
        ax3.plot(activity[events_id.keys()[0]].T - activity[events_id.keys()[1]].T)
        f.subplots_adjust(hspace=0.2)
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        fol = os.path.join(SUBJECT_MEG_FOLDER, 'figures')
        if not os.path.isdir(fol):
            os.mkdir(fol)
        plt.savefig(os.path.join(fol, fig_name))
        plt.close()


def save_subcortical_activity_to_blender(sub_corticals_codes_file, events_id, inverse_method='dSPM',
        colors_map='OrRd', norm_by_percentile=True, norm_percs=(1,99), do_plot=False):
    if do_plot:
        plt.figure()

    sub_corticals = utils.read_sub_corticals_code_file(sub_corticals_codes_file)
    first_time = True
    names_for_blender = []
    lut = utils.read_freesurfer_lookup_table(FREE_SURFER_HOME)
    for ind, sub_cortical_ind in enumerate(sub_corticals):
        sub_cortical_name, _ = utils.get_numeric_index_to_label(sub_cortical_ind, lut)
        names_for_blender.append(sub_cortical_name)
        for cond_id, cond in enumerate(events_id.keys()):
            x = np.load(os.path.join(SUBJECT_MEG_FOLDER, 'subcorticals',
                '{}-{}-{}.npy'.format(cond, sub_cortical_name, inverse_method)))
            if first_time:
                first_time = False
                T = len(x)
                data = np.zeros((len(sub_corticals), T, len(events_id.keys())))
            data[ind, :, cond_id] = x[:T]
        if do_plot:
            plt.plot(data[ind, :, 0]-data[ind, :, 1], label='{}-{} {}'.format(
                events_id.keys()[0], events_id.keys()[1], sub_cortical_name))

    avg_data = np.mean(data, 2)
    # Normalize
    avg_data = utils.normalize_data(avg_data, norm_by_percentile, norm_percs)
    data = utils.normalize_data(data, norm_by_percentile, norm_percs)
    min_x, max_x = np.percentile(avg_data, norm_percs[0]), np.percentile(avg_data, norm_percs[1])
    colors = utils.mat_to_colors(avg_data, min_x, max_x, colorsMap=colors_map)
    np.savez(os.path.join(BLENDER_SUBJECT_FOLDER, 'subcortical_meg_activity'), data=data, colors=colors,
        names=names_for_blender, conditions=events_id.keys())

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
#     cond1tlrc.save(os.path.join(SUBJECT_FOLDER, '{}_tlrc_{}'.format(method, CONDS[0])))
#     cond2tlrc.save(os.path.join(SUBJECT_FOLDER, '{}_tlrc_{}'.format(method, CONDS[1])))


def morph_stc(subject_to, cond='all', grade=None, n_jobs=6, inverse_method='dSPM'):
    stc = mne.read_source_estimate(STC.format(cond, inverse_method))
    vertices_to = mne.grade_to_vertices(subject_to, grade=grade)
    stc_to = mne.morph_data(SUBJECT, subject_to, stc, n_jobs=n_jobs, grade=vertices_to)
    fol_to = os.path.join(SUBJECTS_MEG_DIR, TASK, subject_to)
    if not os.path.isdir(fol_to):
        os.mkdir(fol_to)
    stc_to.save(STC_MORPH.format(subject_to, cond, inverse_method))


def calc_stc_for_all_vertices(stc, n_jobs=6):
    vertices_to = mne.grade_to_vertices(MRI_SUBJECT, grade=None)
    return mne.morph_data(MRI_SUBJECT, MRI_SUBJECT, stc, n_jobs=n_jobs, grade=vertices_to)


def smooth_stc(events_id, stcs=None, inverse_method='dSPM', n_jobs=6):
    stcs = {}
    for ind, cond in enumerate(events_id.keys()):
        if stcs is not None:
            stc = stcs[cond]
        else:
            # Can read only for the 'rh', it'll also read the second file for 'lh'. Strange...
            stc = mne.read_source_estimate(STC_HEMI.format(cond=cond, method=inverse_method, hemi='rh'))
        stc_smooth = calc_stc_for_all_vertices(stc, n_jobs)
        check_stc_with_ply(stc_smooth, cond)
        stc_smooth.save(STC_HEMI_SMOOTH_SAVE.format(cond=cond, method=inverse_method))
        stcs[cond] = stc_smooth
    return stcs


def check_stc_with_ply(stc, cond_name):
    for hemi in ['rh', 'lh']:
        stc_vertices = stc.rh_vertno if hemi=='rh' else stc.lh_vertno
        print('{} {} stc vertices: {}'.format(hemi, cond_name, len(stc_vertices)))
        ply_vertices, _ = utils.read_ply_file(os.path.join(BLENDER_SUBJECT_FOLDER, '{}.pial.ply'.format(hemi)))
        print('{} {} ply vertices: {}'.format(hemi, cond_name, len(stc_vertices)))
        if len(stc_vertices) != ply_vertices.shape[0]:
            raise Exception('check_stc_with_ply: Wrong number of vertices!')
    print('check_stc_with_ply: ok')


def save_activity_map(events_id, stcs_conds=None, colors_map='OrRd', inverse_method='dSPM',
        norm_by_percentile=True, norm_percs=(1,99)):
    stcs = get_average_stc_over_conditions(events_id, stcs_conds, inverse_method, smoothed=True)
    data_max, data_min = utils.get_activity_max_min(stcs, norm_by_percentile, norm_percs)
    scalar_map = utils.get_scalar_map(data_min, data_max, colors_map)
    for hemi in ['rh', 'lh']:
        verts, faces = utils.read_ply_file(os.path.join(SUBJECTS_MRI_DIR, MRI_SUBJECT, 'surf', '{}.pial.ply'.format(hemi)))
        data = stcs[hemi]
        if verts.shape[0]!=data.shape[0]:
            raise Exception('save_activity_map: wrong number of vertices!')
        else:
            print('Both {}.pial.ply and the stc file have {} vertices'.format(hemi, data.shape[0]))
        fol = '{}'.format(ACT.format(hemi))
        utils.delete_folder_files(fol)
        # data = data / data_max
        for t in xrange(data.shape[1]):
            colors = utils.arr_to_colors(data[:, t], 0, data_max, scalar_map=scalar_map)[:,:3]
            colors = np.hstack((np.reshape(data[:, t], (data[:, t].shape[0], 1)), colors))
            np.save(os.path.join(fol, 't{}'.format(t)), colors)
            if t % 10 == 0:
                print('{}: {} out of {}'.format(hemi, t, data.shape[1]))


def save_vertex_activity_map(events_id, stcs_conds=None, inverse_method='dSPM', number_of_files=100):
    if stcs_conds is None:
        stcs_conds = {}
        for cond in events_id.keys():
            stcs_conds[cond] = np.load(STC_HEMI_SMOOTH_SAVE.format(cond=cond, method=inverse_method))
    stcs = get_average_stc_over_conditions(events_id, stcs_conds, inverse_method, smoothed=True)
    # data_max, data_min = utils.get_activity_max_min(stc_rh, stc_lh, norm_by_percentile, norm_percs)

    for hemi in ['rh', 'lh']:
        verts, faces = utils.read_ply_file(os.path.join(SUBJECTS_MRI_DIR, MRI_SUBJECT, 'surf', '{}.pial.ply'.format(hemi)))
        data = stcs[hemi]
        if verts.shape[0]!=data.shape[0]:
            raise Exception('save_vertex_activity_map: wrong number of vertices!')
        else:
            print('Both {}.pial.ply and the stc file have {} vertices'.format(hemi, data.shape[0]))

        data_hash = defaultdict(list)
        fol = '{}_verts'.format(ACT.format(hemi))
        utils.delete_folder_files(fol)
        # data = data / data_max
        look_up = np.zeros((data.shape[0], 2), dtype=np.int)
        for vert_ind in xrange(data.shape[0]):
            file_num = vert_ind % number_of_files
            data_hash[file_num].append(data[vert_ind, :])
            look_up[vert_ind] = [file_num, len(data_hash[file_num])-1]
            if vert_ind % 10000 == 0:
                print('{}: {} out of {}'.format(hemi, vert_ind, data.shape[0]))

        np.save('{}_verts_lookup'.format(ACT.format(hemi)), look_up)
        for file_num in xrange(number_of_files):
            file_name = os.path.join(fol, str(file_num))
            x = np.array(data_hash[file_num])
            np.save(file_name, x)


def get_average_stc_over_conditions(events_id, stcs_conds=None, inverse_method='dSPM', smoothed=False):
    stcs = {}
    stc_template = STC_HEMI if not smoothed else STC_HEMI_SMOOTH
    for cond_ind, cond in enumerate(events_id.keys()):
        if stcs is None:
            # Reading only the rh, the lh will be read too
            print('Reading {}'.format(stc_template.format(cond=cond, method=inverse_method, hemi='rh')))
            stc = mne.read_source_estimate(stc_template.format(cond=cond, method=inverse_method, hemi='rh'))
        else:
            stc = stcs_conds[cond]
        for hemi in ['rh', 'lh']:
            data = stc.rh_data if hemi == 'rh' else stc.lh_data
            if hemi not in stcs:
                stcs[hemi] = np.zeros((data.shape[0], data.shape[1], 2))
            stcs[hemi][:, :, cond_ind] = data
        # Average over the conditions
        stcs[hemi] = stcs[hemi].mean(2)

    return stcs


def rename_activity_files():
    fol = '/homes/5/npeled/space3/MEG/ECR/mg79/activity_map_rh'
    files = glob.glob(os.path.join(fol, '*.npy'))
    for file in files:
        name = '{}.npy'.format(file.split('/')[-1].split('-')[0])
        os.rename(file, os.path.join(fol, name))


def calc_labels_avg(parc, hemi, surf_name, stc=None):
    if stc is None:
        stc = mne.read_source_estimate(STC)
    labels = mne.read_labels_from_annot(SUBJECT, parc, hemi, surf_name)
    inverse_operator = read_inverse_operator(INV)
    src = inverse_operator['src']

    plt.close('all')
    plt.figure()

    for ind, label in enumerate(labels):
        # stc_label = stc.in_label(label)
        mean_flip = stc.extract_label_time_course(label, src, mode='mean_flip')
        mean_flip = np.squeeze(mean_flip)
        if ind==0:
            labels_data = np.zeros((len(labels), len(mean_flip)))
        labels_data[ind, :] = mean_flip
        plt.plot(mean_flip, label=label.name)

    np.savez(LBL.format('all'), data=labels_data, names=[l.name for l in labels])
    plt.legend()
    plt.xlabel('time (ms)')
    plt.show()


def morph_labels_from_fsaverage(aparc_name='aparc250', fs_labels_fol='', sub_labels_fol='', n_jobs=6):
    utils.morph_labels_from_fsaverage(MRI_SUBJECT, SUBJECTS_MRI_DIR, aparc_name, fs_labels_fol, sub_labels_fol, n_jobs)
    # labels_fol = os.path.join(SUBJECTS_MRI_DIR, 'fsaverage', 'label', aparc_name) if fs_labels_fol=='' else fs_labels_fol
    # sub_labels_fol = os.path.join(SUBJECTS_MRI_DIR, MRI_SUBJECT, 'label', aparc_name) if sub_labels_fol=='' else sub_labels_fol
    # if not os.path.isdir(sub_labels_fol):
    #     os.mkdir(sub_labels_fol)
    # for label_file in glob.glob(os.path.join(labels_fol, '*.label')):
    #     fs_label = mne.read_label(label_file)
    #     fs_label.values.fill(1.0)
    #     sub_label = fs_label.morph('fsaverage', MRI_SUBJECT, grade=None, n_jobs=n_jobs, subjects_dir=SUBJECTS_MRI_DIR)
    #     sub_label.save(os.path.join(sub_labels_fol, '{}.label'.format(sub_label.name)))


def labels_to_annot(parc_name, labels_fol='', overwrite=True):
    utils.labels_to_annot(MRI_SUBJECT, SUBJECTS_MRI_DIR, parc_name, labels_fol, overwrite)
    # labels_fol = os.path.join(SUBJECTS_MRI_DIR, MRI_SUBJECT, 'label', 'aparc250') if labels_fol=='' else labels_fol
    # labels = []
    # for label_file in glob.glob(os.path.join(labels_fol, '*.label')):
    #     label = mne.read_label(label_file)
    #     print(label.name)
    #     labels.append(label)
    #
    # mne.write_labels_to_annot(subject=MRI_SUBJECT, labels=labels, parc=parc_name, overwrite=overwrite,
    #                           subjects_dir=SUBJECTS_MRI_DIR)

def calc_labels_avg_per_condition(parc, hemi, surf_name, events_id, labels_fol='', labels_from_annot=True, stcs=None,
        extract_mode='mean_flip', inverse_method='dSPM', norm_by_percentile=True, norm_percs=(1,99), do_plot=False):
    labels_fol = os.path.join(SUBJECTS_MRI_DIR, MRI_SUBJECT, 'label', 'aparc250') if labels_fol=='' else labels_fol
    if stcs is None:
        stcs = {}
        for cond in events_id.keys():
            stcs[cond] = mne.read_source_estimate(STC_HEMI.format(cond=cond, method=inverse_method, hemi=hemi))

    if (labels_from_annot):
        labels = mne.read_labels_from_annot(MRI_SUBJECT, parc, hemi, surf_name)
    else:
        labels = []
        for label_file in glob.glob(os.path.join(labels_fol, '*{}.label'.format(hemi))):
            label = mne.read_label(label_file)
            labels.append(label)

    global_inverse_operator = False
    if utils.how_many_curlies(INV) == 0:
        global_inverse_operator = True
        inverse_operator = read_inverse_operator(INV)
        src = inverse_operator['src']

    if do_plot:
        plt.close('all')
        plt.figure()

    T = len(stcs[stcs.keys()[0]].times)
    labels_data = np.zeros((len(labels), T, len(stcs)))
    conds_incdices = {cond_id:ind for ind, cond_id in zip(range(len(stcs)), events_id.values())}
    for (cond_name, cond_id), stc in zip(events_id.iteritems(), stcs.values()):
        if not global_inverse_operator:
            inverse_operator = read_inverse_operator(INV.format(cond=cond_name))
            src = inverse_operator['src']
        for ind, label in enumerate(labels):
            mean_flip = stc.extract_label_time_course(label, src, mode=extract_mode)
            mean_flip = np.squeeze(mean_flip)
            labels_data[ind, :, conds_incdices[cond_id]] = mean_flip
            if do_plot:
                plt.plot(labels_data[ind, :, conds_incdices[cond_id]], label=label.name)

        if do_plot:
            # plt.legend()
            plt.xlabel('time (ms)')
            plt.title('{}: {} {}'.format(cond_name, hemi, parc))
            plt.show()

    data_max, data_min = utils.get_data_max_min(labels_data, norm_by_percentile, norm_percs)
    max_abs = utils.get_max_abs(data_max, data_min)
    labels_data = labels_data / max_abs
    np.savez(LBL.format(hemi), data=labels_data, names=[l.name for l in labels], conditions=events_id.keys())


def plot_labels_data(plot_each_label=False):
    plt.close('all')
    for hemi in ['rh', 'lh']:
        plt.figure()
        d = np.load(LBL.format(hemi))
        for cond_id, cond_name in enumerate(d['conditions']):
            figures_fol = os.path.join(SUBJECT_MEG_FOLDER, 'figures', hemi, cond_name)
            if not os.path.isdir(figures_fol):
                os.makedirs(figures_fol)
            for name, data in zip(d['names'], d['data'][:,:,cond_id]):
                if plot_each_label:
                    plt.figure()
                plt.plot(data, label=name)
                if plot_each_label:
                    plt.title('{}: {} {}'.format(cond_name, hemi, name))
                    plt.xlabel('time (ms)')
                    plt.savefig(os.path.join(figures_fol, '{}.jpg'.format(name)))
                    plt.close()
            # plt.legend()
            if not plot_each_label:
                plt.title('{}: {}'.format(cond_name, hemi))
                plt.xlabel('time (ms)')
                plt.show()


def check_both_hemi_in_stc(events_id):
    for ind, cond in enumerate(events_id.keys()):
        stcs = {}
        for hemi in ['rh', 'lh']:
            stcs[hemi] = mne.read_source_estimate(STC_HEMI.format(cond=cond, method=inverse_method, hemi=hemi))
        print(np.all(stcs['rh'].rh_data == stcs['lh'].rh_data))
        print(np.all(stcs['rh'].lh_data == stcs['lh'].lh_data))


def check_labels():
    data, names = [], []
    for hemi in ['rh', 'lh']:
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


if __name__ == '__main__':
    TASK = TASK_MSIT
    if TASK==TASK_MSIT:
        # fname_format = '{subject}_msit_interference_1-15-{file_type}.fif' # .format(subject, fname (like 'inv'))
        fname_format = '{subject}_msit_{cond}_1-15-{ana_type}.{file_type}'
        events_id = dict(interference=1, neutral=2) # dict(congruent=1, incongruent=2), events_id = dict(Fear=1, Happy=2)
        event_digit = 1
    elif TASK==TASK_ECR:
        fname_format = '{cond}-{ana_type}.{file_type}'
        events_id = dict(Fear=1, Happy=2) # or dict(congruent=1, incongruent=2)
        event_digit = 3

    # initGlobals('ep001', 'mg78', fname_format)
    # initGlobals('hc004', 'hc004', fname_format)
    initGlobals('ep001', 'mg78', fname_format)
    # initGlobals('fsaverage', 'fsaverage', fname_format)
    inverse_method='dSPM'
    T_MAX = 2
    T_MIN = -0.5
    # sub_corticals = [18, 54] # 18, 'Left-Amygdala', 54, 'Right-Amygdala
    sub_corticals_codes_file = os.path.join(BLENDER_ROOT_FOLDER, 'sub_cortical_codes.txt')
    aparc_name = 'aparc250'
    n_jobs = 6
    stcs = None
    # 1) Load the raw data
    # raw = loadRaw()
    # print(raw.info['sfreq'])
    # filter(raw)
    # createEventsFiles(behavior_file=BEHAVIOR_FILE, pattern='1.....')
    # evoked, epochs = calc_evoked(event_digit=event_digit, events_id=events_id,
    #                     tmin=T_MIN, tmax=T_MAX, read_events_from_file=True, eventsFileName=EVE)

    # make_forward_solution(events_id, sub_corticals_codes_file, n_jobs, calc_only_subcorticals=True)
    # calc_inverse_operator(events_id, calc_for_sub_cortical_fwd=True, calc_for_corticla_fwd=False)
    # calc_sub_cortical_activity(events_id, sub_corticals_codes_file, inverse_method=inverse_method)
    # plot_sub_cortical_activity(events_id, sub_corticals_codes_file, inverse_method=inverse_method)
    # save_subcortical_activity_to_blender(sub_corticals_codes_file, events_id, inverse_method=inverse_method,
    #     colors_map='OrRd', norm_by_percentile=True, norm_percs=(3,97), do_plot=False)

    # *) equalize_epoch_counts
    # equalize_epoch_counts(events_id, method='mintime')
    # *) Calc stc for all the conditions
    # calc_stc(inverse_method)
    # *) Calc stc per condition
    # stcs = calc_stc_per_condition(events_id, inverse_method)
    # morph_labels_from_fsaverage(aparc_name=aparc_name)
    # labels_to_annot(parc_name=aparc_name, overwrite=True)
    # *) Calc labels average
    # # calc_labels_avg(aparc_name, 'rh', 'pial')
    # *) Calc labelse average per condition
    # for hemi in ['rh', 'lh']:
    #     calc_labels_avg_per_condition(aparc_name, hemi, 'pial', events_id, labels_from_annot=False, labels_fol='', stcs=None, inverse_method=inverse_method, do_plot=False)
    # plot_labels_data(plot_each_label=True)
    # *) Save the activity map
    stcs_conds=None
    # stcs_conds = smooth_stc(events_id, stcs, inverse_method=inverse_method)
    # save_activity_map(events_id, stcs_conds, inverse_method=inverse_method)
    # save_vertex_activity_map(events_id, stcs_conds, number_of_files=100)


    # *) misc
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
    # check_both_hemi_in_stc(events_id)
    # lut = utils.read_freesurfer_lookup_table(FREE_SURFER_HOME)
    # for l in lut:
    #     print l
    print('finish!')
