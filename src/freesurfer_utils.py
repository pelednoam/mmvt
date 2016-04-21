import os
import os.path as op
from tempfile import mktemp

from subprocess import Popen, PIPE, check_output
import gzip
import numpy as np
import shutil
import nibabel as nib
from nibabel.spatialimages import ImageFileError

import logging
logger = logging.getLogger('surfer')

from src import utils

mni305_to_subject_reg = 'reg-mni305.2mm --s {subject} --reg mn305_to_{subject}.dat'
mni305_to_subject = 'mri_vol2vol --mov {mni305_sig_file} --reg mn305_to_{subject}.dat --o {subject_sig_file} --fstarg'
mris_ca_label = 'mris_ca_label {subject} {hemi} sphere.reg {freesurfer_home}/average/{hemi}.{atlas_type}.gcs {subjects_dir}/{subject}/label/{hemi}.{atlas}.annot -orig white'


# https://github.com/nipy/PySurfer/blob/master/surfer/io.py
def project_volume_data(filepath, hemi, reg_file=None, subject_id=None,
                        projmeth="frac", projsum="avg", projarg=[0, 1, .1],
                        surf="white", smooth_fwhm=3, mask_label=None,
                        target_subject=None, output_fname=None, verbose=None):
    """Sample MRI volume onto cortical manifold.
    Note: this requires Freesurfer to be installed with correct
    SUBJECTS_DIR definition (it uses mri_vol2surf internally).
    Parameters
    ----------
    filepath : string
        Volume file to resample (equivalent to --mov)
    hemi : [lh, rh]
        Hemisphere target
    reg_file : string
        Path to TKreg style affine matrix file
    subject_id : string
        Use if file is in register with subject's orig.mgz
    projmeth : [frac, dist]
        Projection arg should be understood as fraction of cortical
        thickness or as an absolute distance (in mm)
    projsum : [avg, max, point]
        Average over projection samples, take max, or take point sample
    projarg : single float or sequence of three floats
        Single float for point sample, sequence for avg/max specifying
        start, stop, and step
    surf : string
        Target surface
    smooth_fwhm : float
        FWHM of surface-based smoothing to apply; 0 skips smoothing
    mask_label : string
        Path to label file to constrain projection; otherwise uses cortex
    target_subject : string
        Subject to warp data to in surface space after projection
    verbose : bool, str, int, or None
        If not None, override default verbose level (see surfer.verbose).
    """

    env = os.environ
    if 'FREESURFER_HOME' not in env:
        raise RuntimeError('FreeSurfer environment not defined. Define the '
                           'FREESURFER_HOME environment variable.')
    # Run FreeSurferEnv.sh if not most recent script to set PATH
    if not env['PATH'].startswith(os.path.join(env['FREESURFER_HOME'], 'bin')):
        cmd = ['bash', '-c', 'source {} && env'.format(
               os.path.join(env['FREESURFER_HOME'], 'FreeSurferEnv.sh'))]
        envout = check_output(cmd)
        env = dict(line.split(b'=', 1) for line in envout.split(b'\n') if b'=' in line)

    # Set the basic commands
    cmd_list = ["mri_vol2surf",
                "--mov", filepath,
                "--hemi", hemi,
                "--surf", surf]

    # Specify the affine registration
    if reg_file is not None:
        cmd_list.extend(["--reg", reg_file])
    elif subject_id is not None:
        cmd_list.extend(["--regheader", subject_id])
    else:
        raise ValueError("Must specify reg_file or subject_id")

    # Specify the projection
    proj_flag = "--proj" + projmeth
    if projsum != "point":
        proj_flag += "-"
        proj_flag += projsum
    if hasattr(projarg, "__iter__"):
        proj_arg = list(map(str, projarg))
    else:
        proj_arg = [str(projarg)]
    cmd_list.extend([proj_flag] + proj_arg)

    # Set misc args
    if smooth_fwhm:
        cmd_list.extend(["--surf-fwhm", str(smooth_fwhm)])
    if mask_label is not None:
        cmd_list.extend(["--mask", mask_label])
    if target_subject is not None:
        cmd_list.extend(["--trgsubject", target_subject])

    # Execute the command
    if output_fname is None:
        output_fname = mktemp(prefix="pysurfer-v2s", suffix='.mgz')
    cmd_list.extend(["--o", output_fname])
    logger.info(" ".join(cmd_list))
    p = Popen(cmd_list, stdout=PIPE, stderr=PIPE, env=env)
    stdout, stderr = p.communicate()
    out = p.returncode
    if out:
        raise RuntimeError('mri_vol2surf command failed: {}, {}, {}'.format(out, stdout, stderr))


    # Read in the data
    surf_data = read_scalar_data(output_fname)
    if output_fname is None:
        os.remove(output_fname)
    return surf_data


# https://github.com/nipy/PySurfer/blob/master/surfer/io.py
def read_scalar_data(filepath):
    """Load in scalar data from an image.
    Parameters
    ----------
    filepath : str
        path to scalar data file
    Returns
    -------
    scalar_data : numpy array
        flat numpy array of scalar data
    """
    try:
        scalar_data = nib.load(filepath).get_data()
        scalar_data = np.ravel(scalar_data, order="F")
        return scalar_data

    except ImageFileError:
        ext = os.path.splitext(filepath)[1]
        if ext == ".mgz":
            openfile = gzip.open
        elif ext == ".mgh":
            openfile = open
        else:
            raise ValueError("Scalar file format must be readable "
                             "by Nibabel or .mg{hz} format")

    fobj = openfile(filepath, "rb")
    # We have to use np.fromstring here as gzip fileobjects don't work
    # with np.fromfile; same goes for try/finally instead of with statement
    try:
        v = np.fromstring(fobj.read(4), ">i4")[0]
        if v != 1:
            # I don't actually know what versions this code will read, so to be
            # on the safe side, let's only let version 1 in for now.
            # Scalar data might also be in curv format (e.g. lh.thickness)
            # in which case the first item in the file is a magic number.
            raise NotImplementedError("Scalar data file version not supported")
        ndim1 = np.fromstring(fobj.read(4), ">i4")[0]
        ndim2 = np.fromstring(fobj.read(4), ">i4")[0]
        ndim3 = np.fromstring(fobj.read(4), ">i4")[0]
        nframes = np.fromstring(fobj.read(4), ">i4")[0]
        datatype = np.fromstring(fobj.read(4), ">i4")[0]
        # Set the number of bytes per voxel and numpy data type according to
        # FS codes
        databytes, typecode = {0: (1, ">i1"), 1: (4, ">i4"), 3: (4, ">f4"),
                               4: (2, ">h")}[datatype]
        # Ignore the rest of the header here, just seek to the data
        fobj.seek(284)
        nbytes = ndim1 * ndim2 * ndim3 * nframes * databytes
        # Read in all the data, keep it in flat representation
        # (is this ever a problem?)
        scalar_data = np.fromstring(fobj.read(nbytes), typecode)
    finally:
        fobj.close()

    return scalar_data



def transform_mni_to_subject(subject, subjects_dir, volue_fol, volume_fname='sig.mgz',
        subject_contrast_file_name='sig_subject.mgz', print_only=False):
    mni305_sig_file = os.path.join(volue_fol, volume_fname)
    subject_sig_file = os.path.join(volue_fol, subject_contrast_file_name)
    rs = utils.partial_run_script(locals(), print_only=print_only)
    rs(mni305_to_subject_reg)
    rs(mni305_to_subject)
    subject_fol = op.join(subjects_dir, subject, 'mmvt')
    utils.make_dir(subject_fol)
    shutil.move(op.join(utils.get_parent_fol(), 'mn305_to_{}.dat'.format(subject)),
                op.join(subject_fol, 'mn305_to_{}.dat'.format(subject)))


def transform_subject_to_mni_coordinates(subject, coords, subjects_dir, print_only=False):
    rs = utils.partial_run_script(locals(), print_only=print_only)
    # rs(mni305_to_subject_reg)
    subject_fol = op.join(subjects_dir, subject, 'mmvt')
    utils.make_dir(subject_fol)
    subject_trans_file = op.join(subject_fol, 'mn305_to_{}.dat'.format(subject))
    # shutil.move(op.join(utils.get_parent_fol(), 'mn305_to_{}.dat'.format(subject)), subject_trans_file)
    trans_mat = np.genfromtxt(subject_trans_file, delimiter=' ', skip_header=4, skip_footer=1)
    inv_trans = np.linalg.inv(trans_mat)
    coords_mni = nib.affines.apply_affine(inv_trans, coords)
    return coords_mni


def create_annotation_file(subject, atlas, subjects_dir='', freesurfer_home='', overwrite_annot_file=True, print_only=False):
    '''
    Creates the annot file by using the freesurfer mris_ca_label function

    Parameters
    ----------
    subject: subject name
    atlas: One of the three atlases included with freesurfer:
        Possible values are aparc.DKTatlas40, aparc.a2009s or aparc
        https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation
    subjects_dir: subjects dir. If empty, get it from the environ
    freesurfer_home: freesurfer home. If empty, get it from the environ
    overwrite_annot_file: If False and the annot file already exist, the function return True. If True, the function
        delete first the annot files if exist.
    print_only: If True, the function will just prints the command without executing it

    Returns
    -------
        True if the new annot files exist
    '''
    atlas_types = {'aparc': 'curvature.buckner40.filled.desikan_killiany',
                   'aparc.a2009s': 'destrieux.simple.2009-07-28',
                   'aparc.DKTatlas40': 'DKTatlas40'}
    atlas_type = atlas_types[atlas]
    check_env_var('FREESURFER_HOME', freesurfer_home)
    check_env_var('SUBJECTS_DIR', subjects_dir)
    annot_files_exist = True
    for hemi in ['rh','lh']:
        annot_fname = op.join(subjects_dir, subject, 'label', '{}.{}.annot'.format(hemi, atlas))
        if overwrite_annot_file and op.isfile(annot_fname):
            os.remove(annot_fname)
        rs = utils.partial_run_script(locals(), print_only=print_only)
        rs(mris_ca_label)
        annot_files_exist = annot_files_exist and op.isfile(annot_fname)
    return annot_files_exist


def check_env_var(var_name, var_val):
    if var_val == '':
        var_val = os.environ.get(var_name, '')
        if var_val  == '':
            raise Exception('No {}!'.format(var_name))