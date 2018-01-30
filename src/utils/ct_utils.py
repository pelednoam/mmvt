import os
import os.path as op
import nibabel as nib
import numpy as np
import shutil
from scipy import ndimage

from src.utils import utils
from src.utils import freesurfer_utils as fu


def remove_large_negative_values_from_ct(ct_fname, new_ct_fname='', threshold=-200, overwrite=False):
    '''
    Opens the CT, checks for values less than threshold. Sets all values less than
    threshold to threshold instead. This is helpful for registration as extremely 
    large negative values present in the CT but not in the MR skew the mutual
    information algorithm.

    Parameters
    ----------
    ct_fname : Str
        The filename containing the CT scan
    new_ct_fname: Str
        The output fname
    '''
    if not op.isfile(ct_fname):
        print(f'The CT could not be found in {ct_fname}!')
        return ''
    if new_ct_fname == '':
        new_ct_fname = op.join(utils.get_parent_fol(ct_fname), 'ct_no_large_negative_values.mgz')
    if op.isfile(new_ct_fname):
        if overwrite:
            os.remove(new_ct_fname)
        else:
            return new_ct_fname
    h = nib.load(ct_fname)
    ct_data = h.get_data()
    ct_data[ct_data < threshold] = threshold
    ct_new = nib.Nifti1Image(ct_data, header=h.get_header(), affine=h.get_affine())
    nib.save(ct_new, new_ct_fname)
    return new_ct_fname


def register_ct_to_mr_using_mutual_information(
        subject, subjects_dir, ct_fname, output_fname='', lta_name='', overwrite=False, cost_function='nmi',
        print_only=False):
    '''
    Performs the registration between CT and MR using the normalized mutual
    information cost option in freesurfer's mri_robust_register. Saves the
    output to a temporary file which is subsequently examined and the
    linear registration is returned.

    Freesurfer should already be sourced.

    Parameters
    ----------
    ct_fname : Str
        The filename containing the CT scan
    subject : Str
        The freesurfer subject.
    subjects_dir : Str
        The freesurfer subjects_dir.
    overwrite : Bool
        When true, will do the computation and not search for a saved value.
        Defaults to false.
    cost_function : Enum
        uses mi or nmi or blank cost function. If blank, then its not actually
        using MI at all, just rigid 6 parameter dof registration (with resampling
        tricks and so on)

    Returns
    -------
    affine : 4x4 np.ndarray
        The matrix containing the affine transformation from CT to MR space.
    '''

    xfms_dir = utils.make_dir(op.join(subjects_dir, subject, 'mri', 'transforms'))
    if lta_name == '':
        lta_name = 'ct2mr.lta'
    lta_fname = op.join(xfms_dir, lta_name)
    if op.isfile(lta_fname) and op.isfile(output_fname):
        if overwrite:
            os.remove(lta_fname)
            os.remove(output_fname)
        else:
            return True

    rawavg = op.join(subjects_dir, subject, 'mri', 'T1.mgz') #''rawavg.mgz')
    if output_fname == '':
        output_fname = op.join(utils.get_parent_fol(ct_fname), 'ct_reg_to_mr.mgz')

    fu.robust_register(subject, subjects_dir, ct_fname, rawavg, output_fname, lta_name, cost_function, print_only)
    if print_only:
        return True
    else:
        return op.isfile(lta_fname) and op.isfile(output_fname)


def get_data_and_header(subject, mmvt_dir, subjects_dir, ct_name='ct_reg_to_mr.mgz'):
    fname = op.join(mmvt_dir, subject, 'ct', ct_name)
    if not op.isfile(fname):
        subjects_fname = op.join(subjects_dir, subject, 'mri', ct_name)
        if op.isfile(subjects_fname):
            shutil.copy(subjects_fname, fname)
        else:
            print("Can't find subject's CT! ({})".format(fname))
            return None, None
    header = nib.load(fname)
    data = header.get_data()
    return data, header


def isotropization(ct_fname, isotropization_type=1, iso_vector_override=None):
    ct =  nib.load(ct_fname)
    ct_data = ct.get_data()
    initial_shape = ct_data.shape
    ct_new_data, iso_img = None, None
    vox2ras = ct.affine

    if isotropization_type == 1: #'By voxel':
        max_axis = np.max(ct_data.shape)

        # WARNING: this is not the true isotropization
        # factor. To get the true isotropization factor we would have to
        # trust the image to tell us its correct slice thickness.
        # But this is usually a good approximation
        zf = ct_data.shape / np.array([max_axis, max_axis, max_axis])
        ct_new_data = ndimage.interpolation.zoom(ct_data, zf)

    elif isotropization_type == 2: #'By header':
        # check orientation
        rd, ad, sd = get_std_orientation(vox2ras)
        vox2ras_rstd = np.array(list(map(lambda ix: np.squeeze(vox2ras[ix, :3]), (rd, ad, sd))))
        vox2ras_dg = np.abs(np.diag(vox2ras_rstd)[:3])
        min_axis = np.min(vox2ras_dg)
        max_axis = np.max(vox2ras_dg)
        # zf = vox2ras_dg / min_axis
        zf = vox2ras_dg/ max_axis
        # zf = min_axis / vox2ras_dg
        if np.all(zf == 1):
            print('CT header is isotropic, no linearization to do')
        else:
            ct_new_data = ndimage.interpolation.zoom(ct_data, zf)
            # ct_new_data = ct_data

    elif isotropization_type == 3: #'Manual override':
        initial_shape = ct_data.shape
        zf = np.array(iso_vector_override)
        if np.all(zf == 1):
            print('CT header is isotropic, no linearization to do')
        else:
            ct_new_data = ndimage.interpolation.zoom(ct_data, zf)

    if ct_new_data is not None:
        new_shape = ct_new_data.shape
        # for i in range(3):
        #     vox2ras[i,i] *= 1/zf[i]
        print('Inital CT shape:{}, new shape:{}'.format(initial_shape, new_shape))
        iso_img = nib.Nifti1Image(ct_new_data, vox2ras, ct.header)

    return iso_img


def get_std_orientation(affine):
    rd, = np.where(np.abs(affine[:,0]) == np.max(np.abs(affine[:,0])))
    ad, = np.where(np.abs(affine[:,1]) == np.max(np.abs(affine[:,1])))
    sd, = np.where(np.abs(affine[:,2]) == np.max(np.abs(affine[:,2])))
    return (rd[0], ad[0], sd[0])
