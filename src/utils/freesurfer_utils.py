import os
import os.path as op
from tempfile import mktemp

from subprocess import Popen, PIPE, check_output
import gzip
import numpy as np
import shutil
import traceback
import nibabel as nib
import time
from nibabel.spatialimages import ImageFileError

import logging
logger = logging.getLogger('surfer')

from src.utils import utils
import nibabel.freesurfer as nib_fs


mni305_to_subject_reg = 'reg-mni305.2mm --s {subject} --reg mn305_to_{subject}.dat'
mni305_to_subject = 'mri_vol2vol --mov {mni305_sig_file} --reg mn305_to_{subject}.dat --o {subject_sig_file} --fstarg'
mris_ca_label = 'mris_ca_label {subject} {hemi} sphere.reg {freesurfer_home}/average/{hemi}.{atlas_type}.gcs {subjects_dir}/{subject}/label/{hemi}.{atlas}.annot -orig white'

mri_pretess = 'mri_pretess {mask_fname} {region_id} {norm_fname} {tmp_fol}/{region_id}_filled.mgz'
mri_tessellate = 'mri_tessellate {tmp_fol}/{region_id}_filled.mgz {region_id} {tmp_fol}/{region_id}_notsmooth'
mris_smooth = 'mris_smooth -nw {tmp_fol}/{region_id}_notsmooth {tmp_fol}/{region_id}_smooth'
# mris_convert = 'mris_convert {tmp_fol}/{region_id}_smooth {tmp_fol}/{region_id}.asc'
_mris_convert = 'mris_convert {org_surf_fname} {new_surf_fname}'

mri_vol2surf_pet = 'mri_vol2surf --mov {volume_fname} --hemi {hemi} --projfrac {projfrac} --o {output_fname} --cortex --regheader {subject} --trgsubject {subject}'

warp_buckner_atlas_cmd = 'mri_vol2vol --mov {subjects_dir}/{subject}/mri/norm.mgz --s {subject} ' + \
                     '--targ {bunker_atlas_fname} --m3z talairach.m3z ' + \
                     '--o {subjects_dir}/{subject}/mri/{wrap_map_name} --nearest --inv-morph'

mri_surf2surf = 'mri_surf2surf --srcsubject {source_subject} --srcsurfval {source_fname} --trgsubject {target_subject} --trgsurfval {target_fname} --hemi {hemi}'
mri_vol2vol = 'mri_vol2vol --mov {source_volume_fname} --s {subject} --targ {target_volume_fname} --o {output_volume_fname} --nearest'

mri_segstats = 'mri_segstats --i {fmri_fname} --avgwf {output_txt_fname} --annot {target_subject} {hemi} {atlas} --sum {output_sum_fname}'
mri_aparc2aseg = 'mri_aparc2aseg --s {subject} --annot {atlas} --o {atlas}+aseg.mgz'

mris_flatten = 'mris_flatten {hemi}.inflated.patch {hemi}.flat.patch'

mri_robust_register = 'mri_robust_register --mov "{source_fname}" --dst "{target_fname}" --lta {lta_fname} ' + \
                      '--satit --vox2vox --mapmov "{output_fname}" --cost {cost_function}'

# Creating the seghead surface
mkheadsurf = 'mkheadsurf -subjid {subject} -srcvol T1.mgz'


@utils.check_for_freesurfer
def project_on_surface(subject, volume_file, surf_output_fname, target_subject=None, overwrite_surf_data=False,
                       modality='fmri', subjects_dir='', mmvt_dir='', **kargs):
    if target_subject is None:
        target_subject = subject
    if subjects_dir == '':
        subjects_dir = utils.get_link_dir(utils.get_links_dir(), 'subjects', 'SUBJECTS_DIR')
    if mmvt_dir == '':
        mmvt_dir = utils.get_link_dir(utils.get_links_dir(), 'mmvt')
    utils.make_dir(op.join(mmvt_dir, subject, 'fmri'))
    os.environ['SUBJECTS_DIR'] = subjects_dir
    os.environ['SUBJECT'] = subject
    for hemi in utils.HEMIS:
        if not op.isfile(surf_output_fname.format(hemi=hemi)) or overwrite_surf_data:
            print('project {} to {}'.format(volume_file, hemi))
            if modality != 'pet':
                surf_data = project_volume_data(volume_file, hemi, subject_id=subject, surf="pial", smooth_fwhm=3,
                    target_subject=target_subject, output_fname=surf_output_fname.format(hemi=hemi))
            else:
                surf_data = project_pet_volume_data(subject, volume_file, hemi, surf_output_fname.format(hemi=hemi))
            nans = np.sum(np.isnan(surf_data))
            if nans > 0:
                print('there are {} nans in {} surf data!'.format(nans, hemi))
        surf_data = np.squeeze(nib.load(surf_output_fname.format(hemi=hemi)).get_data())
        output_fname = op.join(mmvt_dir, subject, modality, '{}_{}'.format(modality, op.basename(
            surf_output_fname.format(hemi=hemi))))
        npy_output_fname = op.splitext(output_fname)[0]
        if not op.isfile('{}.npy'.format(npy_output_fname)) or overwrite_surf_data:
            print('Saving surf data in {}.npy'.format(npy_output_fname))
            utils.make_dir(utils.get_parent_fol(npy_output_fname))
            np.save(npy_output_fname, surf_data)


@utils.check_for_freesurfer
def project_pet_volume_data(subject, volume_fname, hemi, output_fname=None, projfrac=0.5, print_only=False, **kargs):
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
@utils.check_for_freesurfer
def project_volume_data(filepath, hemi, reg_file=None, subject_id=None,
                        projmeth="frac", projsum="avg", projarg=[0, 1, .1],
                        surf="white", smooth_fwhm=3, mask_label=None,
                        target_subject=None, output_fname=None, verbose=None, **kargs):
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
    logger.info(' '.join(cmd_list))
    print(' '.join(cmd_list))
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
def read_scalar_data(filepath, **kargs):
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


# def transform_mni_to_subject(subject, subjects_dir, volume_fol, volume_fname='sig.mgz',
#         subject_contrast_file_name='sig_subject.mgz', print_only=False, **kargs):
#     mni305_sig_file = os.path.join(volume_fol, volume_fname)
#     subject_sig_file = os.path.join(volume_fol, subject_contrast_file_name)
#     rs = utils.partial_run_script(locals(), print_only=print_only)
#     rs(mni305_to_subject_reg)
#     rs(mni305_to_subject)
    # subject_fol = op.join(subjects_dir, subject, 'mmvt')
    # utils.make_dir(subject_fol)
    # shutil.move(op.join(utils.get_parent_fol(), 'mn305_to_{}.dat'.format(subject)),
    #             op.join(subject_fol, 'mn305_to_{}.dat'.format(subject)))


def transform_mni_to_subject(subject, subjects_dir, mni305_sig_file, subject_sig_file, print_only=False, **kargs):
    rs = utils.partial_run_script(locals(), print_only=print_only)
    rs(mni305_to_subject_reg)
    rs(mni305_to_subject)


@utils.tryit(None)
def transform_subject_to_mni_coordinates(subject, coords, subjects_dir, **kargs):
    import mne.transforms
    talairach_fname = op.join(subjects_dir, subject, 'mri', 'transforms', 'talairach.xfm')
    if op.isfile(talairach_fname):
        xfm = mne.source_space._read_talxfm(subject, subjects_dir, 'nibabel')
        return mne.transforms.apply_trans(xfm['trans'], coords)
    else:
        print('transform_subject_to_mni_coordinates: No {}!'.format(talairach_fname))
        return None
    # MNI305RAS = TalXFM * Norig * inv(Torig) * [tkrR tkrA tkrS 1]'
    # TalXFM: subject/orig/transforms/talairach.xfm Norig: mri_info --vox2ras orig.mgz Torig: mri_info --vox2ras-tkr orig.mgz


def transform_subject_to_ras_coordinates(subject, coords, subjects_dir, **kargs):
    t1_fname = op.join(subjects_dir, subject, 'mri', 'T1.mgz')
    if op.isfile(t1_fname):
        ndim = coords.ndim
        if ndim == 1:
            coords = np.array([coords])
        t1_header = nib.load(t1_fname).get_header()
        vox = apply_trans(np.linalg.inv(t1_header.get_vox2ras_tkr()), coords)
        ras = apply_trans(t1_header.get_vox2ras(), vox)
        if ndim == 1:
            ras = ras[0]
        return ras
    else:
        print('transform_subject_to_ras_coordinates: No {}!'.format(t1_fname))
        return None


def apply_trans(trans, points, **kargs):
    if points.ndim == 1:
        points = points.reshape((1, 3))
    points = np.hstack((points, np.ones((len(points), 1))))
    points = np.dot(trans, points.T).T
    return points[:, :3]


# def transform_subject_to_subject_coordinates(from_subject, to_subject, coords, subjects_dir):
#     import mne.transforms
#     xfm_from = mne.source_space._read_talxfm(from_subject, subjects_dir, 'nibabel')
#     xfm_to = mne.source_space._read_talxfm(to_subject, subjects_dir, 'nibabel')
#     xfm_to_inv = mne.transforms.invert_transform(xfm_to)
#     mni_coords = mne.transforms.apply_trans(xfm_from['trans'], coords)
#     to_subject_coords = mne.transforms.apply_trans(xfm_to_inv['trans'], mni_coords)
#     return to_subject_coords


def transform_subject_to_subject_coordinates(
        from_subject, to_subject, coords, subjects_dir, return_trans=False, **kargs):
    t1_from_fname = op.join(subjects_dir, from_subject, 'mri', 'T1.mgz')
    t1_to_fname = op.join(subjects_dir, to_subject, 'mri', 'T1.mgz')
    if op.isfile(t1_from_fname) and op.isfile(t1_to_fname):
        if isinstance(coords, list):
            coords = np.array(coords)
        ndim = coords.ndim
        if ndim == 1:
            coords = np.array([coords])
        t1_from_header = nib.load(t1_from_fname).get_header()
        t1_to_header = nib.load(t1_to_fname).get_header()
        trans = t1_to_header.get_vox2ras_tkr() @ t1_to_header.get_ras2vox() @ t1_from_header.get_vox2ras() @ \
                np.linalg.inv(t1_from_header.get_vox2ras_tkr())
        trans[np.isclose(trans, np.zeros(trans.shape))] = 0
        # apply_trans(trans, coords)
        vox_from = apply_trans(np.linalg.inv(t1_from_header.get_vox2ras_tkr()), coords)
        ras = apply_trans(t1_from_header.get_vox2ras(), vox_from)
        vox_to = apply_trans(t1_to_header.get_ras2vox(), ras)
        tk_ras_to = apply_trans(t1_to_header.get_vox2ras_tkr(), vox_to)
        if ndim == 1:
            tk_ras_to = tk_ras_to[0]
    else:
        print('Both {} and {} should exist!'.format(t1_from_fname, t1_from_fname))
        return None
    if return_trans:
        return tk_ras_to, trans
    else:
        return tk_ras_to


@utils.check_for_freesurfer
def create_annotation_file(subject, atlas, subjects_dir='', freesurfer_home='', overwrite_annot_file=True,
                           print_only=False, **kargs):
    '''
    Creates the annot file by using the freesurfer mris_ca_label function

    Parameters
    ----------
    subject: subject name
    atlas: One of the three atlases included with freesurfer:
        Possible values are aparc.DKTatlas, aparc.a2009s or aparc
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
                   'aparc.DKTatlas': 'DKTatlas'}
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


def check_env_var(var_name, var_val, **kargs):
    if var_val == '':
        var_val = os.environ.get(var_name, '')
        if var_val  == '':
            raise Exception('No {}!'.format(var_name))


@utils.check_for_freesurfer
def aseg_to_srf(subject, subjects_dir, output_fol, region_id, lookup, mask_fname, norm_fname,
                overwrite_subcortical_objs=False, **kargs):
    ret = True
    tmp_fol = op.join(subjects_dir, subject, 'tmp', utils.rand_letters(6))
    utils.make_dir(tmp_fol)
    rs = utils.partial_run_script(locals())
    # output_fname = op.join(output_fol, '{}.srf'.format(region_id))
    # tmp_output_fname = op.join(tmp_fol, '{}.asc'.format(region_id))
    # if overwrite_subcortical_objs:
    #     utils.remove_file(output_fname)
    try:
        rs(mri_pretess)
        rs(mri_tessellate)
        rs(mris_smooth)
        fs_file = op.join(tmp_fol, '{}_smooth'.format(region_id))
        verts, faces = nib_fs.read_geometry(fs_file)
        num = int(op.basename(fs_file).split('_')[0])
        if num not in lookup:
            print('Error in the subcorticals lookup table!')
            return False
        new_name = lookup.get(num, '')
        utils.write_ply_file(verts, faces, op.join(output_fol, '{}.ply'.format(new_name)), True)
        # mris_convert = 'mris_convert {tmp_fol}/{region_id}_smooth {tmp_fol}/{region_id}.asc'
        # rs(mris_convert)
        # if op.isfile(tmp_output_fname):
        #     shutil.move(tmp_output_fname, output_fname)
        if op.isdir(tmp_fol):
            shutil.rmtree(tmp_fol)
        else:
            ret = False
    except:
        print('Error in aseg_to_srf! subject: {}'.format(subject))
        print(traceback.format_exc())
        ret = False

    return ret


def warp_buckner_atlas_output_fname(subject, subjects_dir, subregions_num=7, cerebellum_segmentation='loose', **kargs):
    return op.join(subjects_dir, subject, 'mri', 'Buckner2011_atlas_{}_{}.nii.gz'.format(
        subregions_num, cerebellum_segmentation))


@utils.check_for_freesurfer
def warp_buckner_atlas(subject, subjects_dir, bunker_atlas_fname, wrap_map_fname, print_only=False, **kargs):
    norm_fname = op.join(subjects_dir, subject, 'mri', 'norm.mgz')
    if not op.isfile(norm_fname):
        print("Error in warp_buckner_atlas, can't find the file {}".format(norm_fname))
        return False
    trans_fname = op.join(subjects_dir, subject, 'mri', 'transforms', 'talairach.m3z')
    if not op.isfile(trans_fname):
        print("Error in warp_buckner_atlas, can't find the file {}".format(trans_fname))
        return False
    wrap_map_name = op.basename(wrap_map_fname)
    # todo: check why we need to chage directory here
    current_dir = os.getcwd()
    os.chdir(op.join(subjects_dir, subject, 'mri') )
    rs = utils.partial_run_script(locals(), print_only=print_only)
    rs(warp_buckner_atlas_cmd)
    os.chdir(current_dir)
    if not op.isfile(wrap_map_fname):
        raise Exception('Error in warp_buckner_atlas!')
    else:
        print('warp_buckner_atlas output fname: {}'.format(wrap_map_fname))
        return True


def get_tr(fmri_fname, **kargs):
    try:
        img = nib.load(fmri_fname)
        hdr = img.get_header()
        tr = float(hdr._header_data.tolist()[-1][0]) / 1000.0 # To sec
        return tr
    except:
        output = utils.run_script('mri_info --tr {}'.format(fmri_fname))
        try:
            output = output.decode('ascii')
        except:
            pass
        output = output.replace('\n', '')
        tr = float(output) / 1000
        return tr


@utils.check_for_freesurfer
def mri_convert(org_fname, new_fname, overwrite=False, print_only=False, **kargs):
    cmd = 'mri_convert "{}" "{}"'.format(org_fname, new_fname)
    if print_only:
        print(cmd)
        return
    if not op.isfile(new_fname) or overwrite:
        utils.run_script(cmd)
    if op.isfile(new_fname):
        return new_fname
    else:
        raise Exception("mri_convert: {} wan't created".format(new_fname))


def mri_convert_to(org_fname, new_file_type, overwrite=False, **kargs):
    if org_fname.endswith('nii.gz'):
        new_fname = nii_gz_to_mgz_name(org_fname)
    else:
        new_fname = '{}.{}'.format(op.splitext(org_fname)[0], new_file_type)
    new_fname = mri_convert(org_fname, new_fname, overwrite)
    return new_fname


def nii_gz_to_mgz_name(fmri_fname, **kargs):
    return '{}mgz'.format(fmri_fname[:-len('nii.gz')])


def nii_to_mgz_name(fmri_fname, **kargs):
    return '{}mgz'.format(fmri_fname[:-len('nii')])


def nii_gz_to_mgz(fmri_fname, **kargs):
    new_fmri_fname = nii_gz_to_mgz_name(fmri_fname)
    if not op.isfile(new_fmri_fname):
        mri_convert(fmri_fname, new_fmri_fname)
    return new_fmri_fname


def nii_to_mgz(fmri_fname, **kargs):
    new_fmri_fname = nii_to_mgz_name(fmri_fname)
    if not op.isfile(new_fmri_fname):
        mri_convert(fmri_fname, new_fmri_fname)
    return new_fmri_fname


def mgz_to_nii_gz(fmri_fname, **kargs):
    new_fmri_fname = '{}nii.gz'.format(fmri_fname[:-len('mgz')])
    if not op.isfile(new_fmri_fname):
        mri_convert(fmri_fname, new_fmri_fname)
    return new_fmri_fname


@utils.check_for_freesurfer
def surf2surf(source_subject, target_subject, hemi, source_fname, target_fname, cwd=None, print_only=False, **kargs):
    if source_subject != target_subject:
        rs = utils.partial_run_script(locals(), cwd=cwd, print_only=print_only)
        rs(mri_surf2surf)
        if not op.isfile(target_fname):
            raise Exception('surf2surf: Target file was not created!')


@utils.check_for_freesurfer
def vol2vol(subject, source_volume_fname, target_volume_fname, output_volume_fname, cwd=None, print_only=False, **kargs):
    if source_volume_fname != target_volume_fname:
        rs = utils.partial_run_script(locals(), cwd=cwd, print_only=print_only)
        rs(mri_vol2vol)
        if not op.isfile(output_volume_fname):
            raise Exception('vol2vol: Target file was not created!')



def calc_labels_avg(target_subject, hemi, atlas, fmri_fname, res_dir, cwd, overwrite=True, output_txt_fname='',
                    output_sum_fname='', ret_files_name=False, **kargs):
    def get_labels_names(line):
        label_name = line.split()[4]
        label_nums = utils.find_num_in_str(label_name)
        label_num = label_nums[-1] if len(label_nums) > 0 else ''
        if label_num != '':
            name_len = label_name.find('_{}'.format(label_num)) + len(str(label_num)) + 1
            label_name = '{}-{}'.format(label_name[:name_len], hemi)
        return label_name

    if output_txt_fname == '':
        output_txt_fname = op.join(res_dir, '{}_{}_{}.txt'.format(utils.namebase(fmri_fname), atlas, hemi))
    if output_sum_fname == '':
        output_sum_fname = op.join(res_dir, '{}_{}_{}.sum'.format(utils.namebase(fmri_fname), atlas, hemi))
    if not op.isfile(output_txt_fname) or not op.isfile(output_sum_fname) or overwrite:
        print('Running mri_segstats on {} ({})'.format(fmri_fname, utils.file_modification_time(fmri_fname)))
        utils.partial_run_script(locals(), cwd=cwd)(mri_segstats)
    if not op.isfile(output_txt_fname):
        raise Exception('The output file was not created!')
    labels_data = np.genfromtxt(output_txt_fname).T
    labels_names = utils.read_list_from_file(output_sum_fname, get_labels_names, 'rb')
    if ret_files_name:
        return labels_data, labels_names, output_txt_fname, output_sum_fname
    else:
        return labels_data, labels_names


@utils.check_for_freesurfer
@utils.files_needed({'surf': ['lh.white', 'rh.white'], 'mri': ['ribbon.mgz']})
def create_aparc_aseg_file(subject, atlas, subjects_dir, overwrite_aseg_file=False, print_only=False, **kargs):
    if not utils.both_hemi_files_exist(op.join(subjects_dir, subject, 'label', '{}.{}.annot'.format('{hemi}', atlas))):
        print('No annot file was found for {}!'.format(atlas))
        return False, ''
    # The atlas var need to be in the locals for the APARC2ASEG call
    aparc_aseg_file = '{}+aseg.mgz'.format(atlas)
    mri_file_fol = op.join(subjects_dir, subject, 'label')
    aparc_aseg_fname = op.join(mri_file_fol, aparc_aseg_file)
    rs = utils.partial_run_script(locals(), print_only=print_only, cwd=mri_file_fol)
    if not op.isfile(aparc_aseg_fname) or overwrite_aseg_file:
        now = time.time()
        rs(mri_aparc2aseg)
        if op.isfile(aparc_aseg_fname) and op.getmtime(aparc_aseg_fname) > now:
            return True, aparc_aseg_fname
        else:
            print('Failed to create {}'.format(aparc_aseg_fname))
            return False, ''
    return True, aparc_aseg_fname


def parse_patch(filename, **kargs):
    import struct
    with open(filename, 'rb') as fp:
        header, = struct.unpack('>i', fp.read(4))
        nverts, = struct.unpack('>i', fp.read(4))
        data = np.fromstring(fp.read(), dtype=[('vert', '>i4'), ('x', '>f4'), ('y', '>f4'), ('z', '>f4')])
        assert len(data) == nverts
        return data


def write_patch(filename, pts, edges=set(), **kargs):
    from tqdm import tqdm
    import struct
    with open(filename, 'wb') as fp:
        fp.write(struct.pack('>2i', -1, len(pts)))
        for i, pt in tqdm(pts, total=len(pts)):
            if i not in edges:
                fp.write(struct.pack('>i3f', -i-1, *pt))
            else:
                fp.write(struct.pack('>i3f', i+1, *pt))


def read_patch(subject, hemi, subjects_dir, surface_type='pial', patch_fname='', **kargs):
    pts, polys = nib_fs.read_geometry(op.join(subjects_dir, subject, 'surf', '{}.{}'.format(hemi, surface_type)))

    if patch_fname == '':
        patch_fname = op.join(subjects_dir, subject, 'surf', '{}.cortex.patch.flat'.format(hemi))
    patch = parse_patch(patch_fname)
    verts = patch[patch['vert'] > 0]['vert'] - 1
    edges = -patch[patch['vert'] < 0]['vert'] - 1

    idx = np.zeros((len(pts),), dtype=bool)
    idx[verts] = True
    idx[edges] = True
    valid = idx[polys.ravel()].reshape(-1, 3).all(1)
    polys = polys[valid]
    # idx = np.zeros((len(pts),))
    # idx[verts] = 1
    # idx[edges] = -1

    for i, x in enumerate(['x', 'y', 'z']):
        pts[verts, i] = patch[patch['vert'] > 0][x]
        pts[edges, i] = patch[patch['vert'] < 0][x]
    return pts, polys


def mris_convert(org_surf_fname, new_surf_fname, print_only=False, **kargs):
    # mris_convert = 'mris_convert {org_surf_fname} {new_surf_fname}'
    rs = utils.partial_run_script(locals(), print_only=print_only)
    rs(_mris_convert)
    return op.isfile(new_surf_fname)


def write_surf(filename, pts, polys, comment=b'', **kargs):
    import struct
    with open(filename, 'wb') as fp:
        fp.write(b'\xff\xff\xfe')
        fp.write(comment+b'\n\n')
        fp.write(struct.pack('>2I', len(pts), len(polys)))
        fp.write(pts.astype(np.float32).byteswap())#.tostring())
        fp.write(polys.astype(np.uint32).byteswap())#.tostring())
        fp.write(b'\n')


def flat_brain(subject, hemi, subjects_dir, print_only=False, **kargs):
    # mris_flatten lh.inflated.patch lh.flat.patch
    surf_dir = op.join(subjects_dir, subject, 'surf')
    rs = utils.partial_run_script(locals(), cwd=surf_dir, print_only=print_only)
    rs(mris_flatten)
    return get_flat_patch_fname(subject, hemi, subjects_dir)


def get_flat_patch_fname(subject, hemi, subjects_dir, **kargs):
    surf_dir = op.join(subjects_dir, subject, 'surf')
    return op.join(surf_dir, '{}.flat.patch'.format(hemi))


def test_patch(subject, **kargs):
    from src.utils import preproc_utils as pu
    SUBJECTS_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()

    flat_patch_cut_vertices = utils.load(op.join(MMVT_DIR, subject, 'flat_patch_cut_vertices.pkl'))
    for hemi in utils.HEMIS:
        # verts, faces = read_patch(hemi, SUBJECTS_DIR)
        # write_patch(op.join(MMVT_DIR, 'fsaverage', 'surf', '{}.flat.test.pial'.format(hemi)), verts, faces)

        # fs_verts, fs_faces = utils.read_pial('fsaverage', MMVT_DIR, hemi, surface_type='inflated')
        print('Reading inflated surf')
        fs_verts, fs_faces = nib.freesurfer.read_geometry(op.join(SUBJECTS_DIR, subject, 'surf', '{}.inflated'.format(hemi)))
        # flat_verts, flat_faces = read_patch(hemi, SUBJECTS_DIR, surface_type='inflated')
        # good_verts = set(np.unique(flat_faces))
        # bad_verts = np.setdiff1d(np.arange(0, flat_verts.shape[0]), good_verts)
        # bad_faces_inds = set(utils.flat_list_of_lists([verts_faces_lookup[hemi][v] for v in bad_verts]))
        patch_fname = op.join(SUBJECTS_DIR, subject, 'surf', '{}.inflated.patch'.format(hemi))
        print('Writing patch')
        flat_patch_cut_vertices_hemi = set(flat_patch_cut_vertices[hemi])
        write_patch(patch_fname, [(ind, v) for ind, v in enumerate(fs_verts) if ind not in flat_patch_cut_vertices_hemi], set())#flat_faces)

        print('Reading patch')
        patch_verts, patch_faces = read_patch(subject, hemi, SUBJECTS_DIR, surface_type='inflated', patch_fname=patch_fname)

        print('Writing ply')
        patch_verts *= 0.1
        utils.write_ply_file(patch_verts, patch_faces, op.join(MMVT_DIR, subject, 'surf', '{}.flat.pial.test.ply').format(hemi))
    print('Finish!')


def read_lta_file(lta_fame, **kargs):
    # https://github.com/pelednoam/ielu/blob/master/ielu/geometry.py#L182
    affine = np.zeros((4,4))
    with open(lta_fame) as fd:
        for i,ln in enumerate(fd):
            if i < 8:
                continue
            elif i > 11:
                break
            affine[i-8,:] = np.array(list(map(float, ln.strip().split())))
    return affine


@utils.check_for_freesurfer
def robust_register(subject, subjects_dir, source_fname, target_fname, output_fname, lta_name,
                    cost_function='nmi', print_only=False, **kargs):
    xfms_dir = op.join(subjects_dir, subject, 'mri', 'transforms')
    utils.make_dir(xfms_dir)
    lta_fname = op.join(xfms_dir, lta_name)
    rs = utils.partial_run_script(locals(), print_only=print_only)
    rs(mri_robust_register)
    return True if print_only else op.isfile(lta_fname)


@utils.check_for_freesurfer
def create_seghead(subject, subjects_dir=None, print_only=False, **kargs):
    if subjects_dir is None:
        subjects_dir = utils.get_link_dir(utils.get_links_dir(), 'subjects', 'SUBJECTS_DIR')
    os.environ['SUBJECTS_DIR'] = subjects_dir
    rs = utils.partial_run_script(locals(), print_only=print_only)
    rs(mkheadsurf)


def is_fs_atlas(atlas):
    return atlas in ['aparc.DKTatlas', 'aparc', 'aparc.a2009s']


def read_surface(subject, subjects_dir, surf_type='pial'):
    verts, faces = {}, {}
    for hemi in utils.HEMIS:
        surf_fname = op.join(subjects_dir, subject, 'surf', '{}.{}'.format(hemi, surf_type))
        if not op.isfile(surf_fname):
            print('{} does not exist!'.format(surf_fname))
            return None, None
        verts[hemi], faces[hemi] = nib.freesurfer.read_geometry(surf_fname)
    return verts, faces


if __name__ == '__main__':
    import argparse
    from src.utils.utils import Bag
    from src.utils import args_utils as au

    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=True)
    parser.add_argument('-f', '--function', help='function name', required=True, type=au.str_arr_type)
    args = Bag(au.parse_parser(parser))
    for func in args.function:
        locals()[func](**args)


