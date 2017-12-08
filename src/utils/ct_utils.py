import os.path as op
import nibabel as nib
import shutil

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
        return
    if new_ct_fname == '':
        new_ct_fname = op.join(utils.get_parent_fol(ct_fname), 'ct_no_large_negative_values.mgz')
    if op.isfile(new_ct_fname) and not overwrite:
        return new_ct_fname
    h = nib.load(ct_fname)
    ct_data = h.get_data()
    ct_data[ct_data < threshold] = threshold
    ct_new = nib.Nifti1Image(ct_data, header=h.get_header(), affine=h.get_affine())
    nib.save(ct_new, new_ct_fname)
    return new_ct_fname


def register_ct_to_mr_using_mutual_information(
        subject, subjects_dir, ct_fname, output_fname='', lta_name='', overwrite=False, cost_function='nmi'):
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
    if op.isfile(lta_fname) and op.isfile(output_fname) and not overwrite:
        return fu.read_lta_file(lta_fname)

    rawavg = op.join(subjects_dir, subject, 'mri', 'T1.mgz') #''rawavg.mgz')
    if output_fname == '':
        output_fname = op.join(utils.get_parent_fol(ct_fname), 'ct_reg_to_mr.mgz')

    fu.robust_register(subject, subjects_dir, ct_fname, rawavg, output_fname, lta_name, cost_function)
    if op.isfile(lta_fname) and op.isfile(output_fname):
        return fu.read_lta_file(lta_fname)
    else:
        return None


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
