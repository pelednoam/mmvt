import numpy as np
import os
import matplotlib.pyplot as plt
from functools import partial
import re

import mne
from mne.transforms import apply_trans
from mne.minimum_norm import (make_inverse_operator, apply_inverse,
                              write_inverse_operator, read_inverse_operator)

SUBJECTS_MRI_DIR = '/autofs/space/lilli_001/users/DARPA-MEG/freesurfs'
FREE_SURFER_HOME = '/usr/local/freesurfer/stable5_3_0'
os.environ['SUBJECTS_DIR'] = SUBJECTS_MRI_DIR

def make_forward_solution(events_id, src_fn, fwd_sub_fn, epochs_fn, cor_fn, bem_fn, sub_corticals_codes_file, usingEEG=True, n_jobs=4):
    # The cortical surface source space can be setup if doens't exist
    # src = mne.setup_source_space(MRI_SUBJECT, surface='pial',  overwrite=True) # overwrite=True)
    fwd_with_subcortical = None
    src = mne.read_source_spaces(src_fn)
    sub_corticals = read_sub_corticals_code_file(sub_corticals_codes_file)
    for cond in events_id.keys():
        if len(sub_corticals) > 0:
            # add a subcortical volumes
            src_with_subcortical = add_subcortical_volumes(src, sub_corticals)
            fwd_with_subcortical = _make_forward_solution(src, epochs_fn, cor_fn, bem_fn, usingEEG=usingEEG, n_jobs=n_jobs)
            mne.write_forward_solution(fwd_sub_fn.format(cond=cond), fwd_with_subcortical, overwrite=True)

    return fwd_with_subcortical


def calc_inverse_operator(events_id, epochs_fn, fwd_sub_fn, inv_fn, min_crop_t=None, max_crop_t=0):
    for cond in events_id.keys():
        epochs = mne.read_epochs(epochs_fn.format(cond=cond))
        noise_cov = mne.compute_covariance(epochs.crop(min_crop_t, max_crop_t, copy=True))
        forward_sub = mne.read_forward_solution(fwd_sub_fn.format(cond=cond))
        inverse_operator_sub = make_inverse_operator(epochs.info, forward_sub, noise_cov,
            loose=None, depth=None)
        write_inverse_operator(inv_fn.format(cond=cond), inverse_operator_sub)


def calc_sub_cortical_activity(events_id, evoked_fn, inv_fn, sub_corticals_codes_file, baseline_min_t=None,
        baseline_max_t = 0, snr = 3.0, inverse_method='dSPM'):

    sub_corticals = read_sub_corticals_code_file(sub_corticals_codes_file)
    if len(sub_corticals) == 0:
        return

    lambda2 = 1.0 / snr ** 2
    lut = read_freesurfer_lookup_table(FREE_SURFER_HOME)
    for cond in events_id.keys():
        evo = evoked_fn.format(cond=cond)
        evoked = {event:mne.read_evokeds(evo, baseline=(baseline_min_t, baseline_max_t))[0] for event in [event]}
        inverse_operator = read_inverse_operator(inv_fn.format(cond=cond))
        stc = apply_inverse(evoked[cond], inverse_operator, lambda2, inverse_method)
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
            sub_cortical, _ = get_numeric_index_to_label(sub_cortical_code, lut)
            np.save(os.path.join(SUBJECT_MEG_FOLDER, 'subcorticals', '{}-{}-{}'.format(cond, sub_cortical, inverse_method)), activity.mean(0))
            np.save(os.path.join(SUBJECT_MEG_FOLDER, 'subcorticals', '{}-{}-{}-all-vertices'.format(cond, sub_cortical, inverse_method)), activity)


def _make_forward_solution(src, epochs_fn, cor_fn, bem_fn, usingEEG=True, n_jobs=6):
    fwd = mne.make_forward_solution(info=epochs_fn, trans=cor_fn, src=src, bem=bem_fn,
                                    meg=True, eeg=usingEEG, mindist=5.0,
                                    n_jobs=n_jobs, overwrite=True)
    return fwd


def read_sub_corticals_code_file(sub_corticals_codes_file, read_also_names=False):
    if os.path.isfile(sub_corticals_codes_file):
        codes = np.genfromtxt(sub_corticals_codes_file, usecols=(1), delimiter=',', dtype=int)
        codes = map(int, codes)
        if read_also_names:
            names = np.genfromtxt(sub_corticals_codes_file, usecols=(0), delimiter=',', dtype=str)
            names = map(str, names)
            sub_corticals = {code:name for code, name in zip(codes, names)}
        else:
            sub_corticals = codes
    else:
        sub_corticals = []
    return sub_corticals


def add_subcortical_volumes(org_src, seg_labels, spacing=5., use_grid=True):
    """Adds a subcortical volume to a cortical source space
    """
    # Get the subject
    import nibabel as nib
    from mne.source_space import _make_discrete_source_space

    src = org_src.copy()

    # Find the segmentation file
    aseg_fname = os.path.join(SUBJECTS_MRI_DIR, SUBJECT, 'mri', 'aseg.mgz')
    aseg = nib.load(aseg_fname)
    aseg_hdr = aseg.get_header()
    sub_cortical_generator = sub_cortical_voxels_generator(aseg, seg_labels, spacing, use_grid, FREE_SURFER_HOME)
    for pts, seg_name, seg_id in sub_cortical_generator:
        pts = transform_voxels_to_RAS(aseg_hdr, pts)
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


def sub_cortical_voxels_generator(aseg, seg_labels, spacing=5, use_grid=True, freesurfer_home=''):
    if freesurfer_home=='':
        freesurfer_home = os.environ['FREE_SURFER_HOME']

    # Read the segmentation data using nibabel
    aseg_data = aseg.get_data()

    # Read the freesurfer lookup table
    lut = read_freesurfer_lookup_table(freesurfer_home)

    # Generate a grid using spacing
    grid = None
    if use_grid:
        grid = generate_grid_using_spacing(spacing, aseg_data.shape)

    # Get the indices to the desired labels
    for label in seg_labels:
        seg_name, seg_id = get_numeric_index_to_label(label, lut)
        pts = calc_label_voxels(seg_id, aseg_data, grid)
        yield pts, seg_name, seg_id


def read_freesurfer_lookup_table(freesurfer_home):
    lut_fname = os.path.join(freesurfer_home, 'FreeSurferColorLUT.txt')
    lut = np.genfromtxt(lut_fname, dtype=None, usecols=(0, 1),
                        names=['id', 'name'])
    return lut


def generate_grid_using_spacing(spacing, shp):
    # Generate a grid using spacing
    kernel = np.zeros((int(spacing), int(spacing), int(spacing)))
    kernel[0, 0, 0] = 1
    sx, sy, sz = shp
    nx, ny, nz = np.ceil((sx/spacing, sy/spacing, sz/spacing))
    grid = np.tile(kernel, (nx, ny, nz))
    grid = grid[:sx, :sy, :sz]
    grid = grid.astype('bool')
    return grid


def get_numeric_index_to_label(label, lut):
    if type(label) == str:
        seg_name = label
        seg_id = lut['id'][lut['name'] == seg_name][0]
    elif type(label) == int:
        seg_id = label
        seg_name = lut['name'][lut['id'] == seg_id][0]
    return seg_name, seg_id


def calc_label_voxels(seg_id, aseg_data, grid=None):
    # Get indices to label
    ix = aseg_data == seg_id
    if grid is not None:
        ix *= grid  # downsample to grid
    pts = np.array(np.where(ix)).T
    return pts


def transform_voxels_to_RAS(aseg_hdr, pts):
    # Transform data to RAS coordinates
    trans = aseg_hdr.get_vox2ras_tkr()
    pts = apply_trans(trans, pts)
    return pts


def plot_sub_cortical_activity(events_id, sub_corticals_codes_file, inverse_method='dSPM'):
    lut = read_freesurfer_lookup_table(FREE_SURFER_HOME)
    sub_corticals = read_sub_corticals_code_file(sub_corticals_codes_file)
    for label in sub_corticals:
        sub_cortical, _ = get_numeric_index_to_label(label, lut)
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


def get_file_name(subject, ana_type, file_type='fif', fname_format=''):
    if fname_format=='':
        fname_format = '{subject}-{ana_type}.{file_type}'
    if how_many_curlies(fname_format) == 2:
        fname = fname_format.format(subject=subject, ana_type=ana_type, file_type=file_type)
    else:
        fname = fname_format.format(subject=subject, ana_type=ana_type, cond='{cond}', file_type=file_type)
    return os.path.join(SUBJECT_MEG_FOLDER, fname)

def how_many_curlies(str):
    return len(re.findall('\{*\}', str))


if __name__ == '__main__':
    inverse_method='dSPM'
    n_jobs = 6
    snr = 3.0

    TASK_MSIT, TASK_ECR = range(2)
    TASK = TASK_MSIT
    if TASK==TASK_MSIT:
        fname_format = '{subject}_msit_{cond}_1-15-{ana_type}.{file_type}'
        events_id = dict(interference=1, neutral=2)
        event_digit = 1
    elif TASK==TASK_ECR:
        fname_format = '{cond}-{ana_type}.{file_type}'
        events_id = dict(Fear=1, Happy=2) # or dict(congruent=1, incongruent=2)
        event_digit = 3

    # Should be filled in
    SUBJECT = ''
    SUBJECT_MRI_FOLDER = os.path.join(SUBJECTS_MRI_DIR, SUBJECT)
    SUBJECTS_MEG_DIR = ''
    SUBJECT_MEG_FOLDER = ''

    _get_fif_name = partial(get_file_name, subject=SUBJECT, fname_format=fname_format, file_type='fif')
    src_fn = os.path.join(SUBJECT_MRI_FOLDER, 'bem', '{}-oct-6p-src.fif'.format(SUBJECT))
    bem_fn = os.path.join(SUBJECT_MRI_FOLDER, 'bem', '{}-5120-5120-5120-bem-sol.fif'.format(SUBJECT))
    cor_fn = os.path.join(SUBJECT_MRI_FOLDER, 'mri', 'T1-neuromag', 'sets', 'COR.fif')
    epochs_fn = _get_fif_name('epo')
    evoked_fn = _get_fif_name('ave')
    fwd_sub_fn =  _get_fif_name('sub-cortical-fwd')
    inv_sub_fn = _get_fif_name('sub-cortical-inv')

    #todo: Should be changed!!!
    baseline_min_t = None
    baseline_max_t = 0
    min_crop_t = 0
    max_crop_t = 1

    subcortical_codes_fol = '' # should be filled in
    sub_corticals_codes_file = os.path.join(subcortical_codes_fol, 'sub_cortical_codes.txt')

    make_forward_solution(events_id, src_fn, fwd_sub_fn, epochs_fn, cor_fn, bem_fn, sub_corticals_codes_file,
        usingEEG=True, n_jobs=n_jobs)
    calc_inverse_operator(events_id, epochs_fn, fwd_sub_fn, inv_sub_fn, min_crop_t, max_crop_t)
    calc_sub_cortical_activity(events_id, evoked_fn, inv_sub_fn, sub_corticals_codes_file, baseline_min_t,
        baseline_max_t, snr=snr, inverse_method=inverse_method)
    plot_sub_cortical_activity(events_id, sub_corticals_codes_file, inverse_method=inverse_method)
