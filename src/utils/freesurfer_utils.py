import os
import os.path as op
from tempfile import mktemp

from subprocess import Popen, PIPE, check_output
import gzip
import numpy as np
import shutil
import traceback
import nibabel as nib
from nibabel.spatialimages import ImageFileError

import logging
logger = logging.getLogger('surfer')

from src.utils import utils

mni305_to_subject_reg = 'reg-mni305.2mm --s {subject} --reg mn305_to_{subject}.dat'
mni305_to_subject = 'mri_vol2vol --mov {mni305_sig_file} --reg mn305_to_{subject}.dat --o {subject_sig_file} --fstarg'
mris_ca_label = 'mris_ca_label {subject} {hemi} sphere.reg {freesurfer_home}/average/{hemi}.{atlas_type}.gcs {subjects_dir}/{subject}/label/{hemi}.{atlas}.annot -orig white'

mri_pretess = 'mri_pretess {mask_fname} {region_id} {norm_fname} {tmp_fol}/{region_id}_filled.mgz'
mri_tessellate = 'mri_tessellate {tmp_fol}/{region_id}_filled.mgz {region_id} {tmp_fol}/{region_id}_notsmooth'
mris_smooth = 'mris_smooth -nw {tmp_fol}/{region_id}_notsmooth {tmp_fol}/{region_id}_smooth'
mris_convert = 'mris_convert {tmp_fol}/{region_id}_smooth {tmp_fol}/{region_id}.asc'

mri_vol2surf_pet = 'mri_vol2surf --mov {volume_fname} --hemi {hemi} --projfrac {projfrac} --o {output_fname} --cortex --regheader {subject} --trgsubject {subject}'

warp_buckner_atlas_cmd = 'mri_vol2vol --mov {subjects_dir}/{subject}/mri/norm.mgz --s {subject} ' + \
                     '--targ {bunker_atlas_fname} --m3z talairach.m3z ' + \
                     '--o {subjects_dir}/{subject}/mri/{wrap_map_name} --nearest --inv-morph'


def project_pet_volume_data(subject, volume_fname, hemi, output_fname=None, projfrac=0.5, print_only=False):
    temp_output = output_fname is None
    if output_fname is None:
        output_fname = mktemp(prefix="pysurfer-v2s", suffix='.mgz')
    rs = utils.partial_run_script(locals(), print_only=print_only)
    rs(mri_vol2surf_pet)
    surf_data = read_scalar_data(output_fname)
    if temp_output:
        os.remove(output_fname)
    return surf_data


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


def transform_subject_to_mni_coordinates(subject, coords, subjects_dir):
    import mne.transforms
    xfm = mne.source_space._read_talxfm(subject, subjects_dir, 'nibabel')
    return mne.transforms.apply_trans(xfm['trans'], coords)


def transform_subject_to_subject_coordinates(from_subject, to_subject, coords, subjects_dir):
    import mne.transforms
    xfm_from = mne.source_space._read_talxfm(from_subject, subjects_dir, 'nibabel')
    xfm_to = mne.source_space._read_talxfm(to_subject, subjects_dir, 'nibabel')
    xfm_to_inv = mne.transforms.invert_transform(xfm_to)
    mni_coords = mne.transforms.apply_trans(xfm_from['trans'], coords)
    to_subject_coords = mne.transforms.apply_trans(xfm_to_inv['trans'], mni_coords)
    return to_subject_coords


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


def aseg_to_srf(subject, subjects_dir, output_fol, region_id, mask_fname, norm_fname,
                overwrite_subcortical_objs=False):
    ret = True
    tmp_fol = op.join(subjects_dir, subject, 'tmp', utils.rand_letters(6))
    utils.make_dir(tmp_fol)
    rs = utils.partial_run_script(locals())
    output_fname = op.join(output_fol, '{}.srf'.format(region_id))
    tmp_output_fname = op.join(tmp_fol, '{}.asc'.format(region_id))
    if overwrite_subcortical_objs:
        utils.remove_file(output_fname)
    try:
        rs(mri_pretess)
        rs(mri_tessellate)
        rs(mris_smooth)
        rs(mris_convert)
        if op.isfile(tmp_output_fname):
            shutil.move(tmp_output_fname, output_fname)
            shutil.rmtree(tmp_fol)
        else:
            ret = False
    except:
        print('Error in aseg_to_srf! subject: {}'.format(subject))
        print(traceback.format_exc())
        ret = False

    return ret


def warp_buckner_atlas_output_fname(subject, subjects_dir, subregions_num=7, cerebellum_segmentation='loose'):
    return op.join(subjects_dir, subject, 'mri', 'Buckner2011_atlas_{}_{}.nii.gz'.format(
        subregions_num, cerebellum_segmentation))


def warp_buckner_atlas(subject, subjects_dir, bunker_atlas_fname, wrap_map_fname, print_only=False):
    norm_fname = op.join(subjects_dir, subject, 'mri', 'norm.mgz')
    if not op.isfile(norm_fname):
        print("Error in warp_buckner_atlas, can't find the file {}".format(norm_fname))
        return False
    trans_fname = op.join(subjects_dir, subject, 'mri', 'transforms', 'talairach.m3z')
    if not op.isfile(trans_fname):
        print("Error in warp_buckner_atlas, can't find the file {}".format(trans_fname))
        return False
    wrap_map_name = op.basename(wrap_map_fname)
    rs = utils.partial_run_script(locals(), print_only=print_only)
    rs(warp_buckner_atlas_cmd)
    if not op.isfile(wrap_map_fname):
        raise Exception('Error in warp_buckner_atlas!')
    else:
        print('warp_buckner_atlas output fname: {}'.format(wrap_map_fname))
        return True


def get_tr(fmri_fname):
    img = nib.load(fmri_fname)
    hdr = img.get_header()
    tr = float(hdr._header_data.tolist()[-1][0])
    return tr


def mri_convert(org_fname, new_fname, overwrite=False):
    if not op.isfile(new_fname) or overwrite:
        utils.run_script('mri_convert {} {}'.format(org_fname, new_fname))


def nii_gz_to_mgz(fmri_fname):
    new_fmri_fname = '{}mgz'.format(fmri_fname[:-len('nii.gz')])
    if not op.isfile(new_fmri_fname):
        mri_convert(fmri_fname, new_fmri_fname)
    return new_fmri_fname


def mgz_to_nii_gz(fmri_fname):
    new_fmri_fname = '{}nii.gz'.format(fmri_fname[:-len('mgz')])
    if not op.isfile(new_fmri_fname):
        mri_convert(fmri_fname, new_fmri_fname)
    return new_fmri_fname