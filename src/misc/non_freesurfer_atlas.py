import os.path as op
import nibabel as nib
import numpy as np
import shutil
import os

from src.utils import utils

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
MMVT_DIR = op.join(LINKS_DIR, 'mmvt')


MRI_PRETESS = 'mri_pretess {hemi}/{region}_mask.nii.gz 1 {colin_norm_fname} tmp/{region}_{hemi}_filled.mgz'
MRI_TESSELLATE = 'mri_tessellate {hemi}/{region}_mask.nii.gz 1 tmp/{region}_{hemi}_notsmooth'
MRIS_SMOOTH = 'mris_smooth -nw tmp/{region}_{hemi}_notsmooth tmp/{region}_{hemi}_smooth'
MRIS_CONVERT = 'mris_convert tmp/{region}_{hemi}_smooth tmp/{region}_{hemi}.asc'


def prepare_mask_file(region_fname, overwrite=False):
    namebase, image_type = get_namebase(region_fname)
    output_fname = op.join(get_fol_name(region_fname), '{}_mask.{}'.format(namebase, image_type))
    if not op.isfile(output_fname) or overwrite:
        reg = nib.load(region_fname)
        data = reg.get_data()
        h = reg.get_header()
        data[np.where(np.isnan(data))] = 0
        data[np.where(data >= 0.1)] = 1
        data[np.where(data < 0.1)] = 0
        mask = nib.Nifti1Image(data, affine=reg.get_affine())
        print('Saving image to {}'.format(output_fname))
        nib.save(mask, output_fname)


def to_ras(points, org_vox2ras_tkr, shifter_vox2tk_ras):
    points = apply_trans(np.linalg.inv(org_vox2ras_tkr), points)
    points = apply_trans(shifter_vox2tk_ras, points)
    return points


def apply_trans(trans, points):
    return np.array([np.dot(trans, np.append(p, 1))[:3] for p in points])


# def org_vox2ras_tkr():
#     # mri_info --vox2ras-tkr rh/Striatum_mask.mgz
#     return np.array(
#         [[-0.300, 0.000, 0.000, 25.200],
#         [0.000, 0.000, 0.300, -29.700],
#         [0.000, -0.300, 0.000, 37.200],
#         [0.000, 0.000, 0.000, 1.000]])


def get_shifted_vox2tk_ras(zero_in_T1_tk_ras, voxels_sizes):
    return np.array(
        [[-voxels_sizes[0], 0.0, 0.0, zero_in_T1_tk_ras[0]],
         [0.0, voxels_sizes[1], 0.0, zero_in_T1_tk_ras[1]],
         [0.0, 0.0, voxels_sizes[2], zero_in_T1_tk_ras[2]],
         [0.0, 0.0, 0.0, 1.0]])


# def vox2ras_tkr():
#     # To calculate this matrix, open colin27 T1 and the mask file.
#     # Set the cursor position of the new region to 0, 0, 0
#     # The values in TkReg RAS (T1) will be the offset (right column)
#     # 0.3, 0.3, 0.3 are the voxel sizes
#
#     # right striatum
#     return np.array(
#         [[-0.300, 0.000, 0.000, 44.750],
#          [0.000, 0.300, 0.000, -21.2500],
#          [0.000, 0.000, 0.300, -41.250],
#          [0.000, 0.000, 0.000, 1.000]])
#
#     # left striatum
#     return np.array(
#         [[-0.300, 0.000, 0.000, 4.850],
#          [0.000, 0.300, 0.000, -22.2500],
#          [0.000, 0.000, 0.300, -41.250],
#          [0.000, 0.000, 0.000, 1.000]])


def get_vox2ras(fname):
    output = utils.run_script('mri_info --vox2ras {}'.format(fname))
    return read_transform_matrix_from_output(output)


def get_vox2ras_tkr(fname):
    output = utils.run_script('mri_info --vox2ras-tkr {}'.format(fname))
    return read_transform_matrix_from_output(output)


def ras_to_tkr_ras(fname):
    ras2vox = np.linalg.inv(get_vox2ras(fname))
    vox2tkras = get_vox2ras_tkr(fname)
    return np.dot(ras2vox, vox2tkras)


def calc_shifted_vox2tk_ras(region_fname, colin_T1_fname):
    region_vox2ras = get_vox2ras(region_fname)
    zero_in_ras = apply_trans(region_vox2ras, [[0, 0, 0]])
    zero_in_T1_voxels = apply_trans(np.linalg.inv(get_vox2ras(colin_T1_fname)), zero_in_ras)
    zero_in_T1_tk_ras = apply_trans(get_vox2ras_tkr(colin_T1_fname), zero_in_T1_voxels)
    voxels_sizes = [abs(v) for v in np.diag(region_vox2ras)[:-1]]
    shifted_vox2tk_ras = get_shifted_vox2tk_ras(zero_in_T1_tk_ras[0], voxels_sizes)
    return shifted_vox2tk_ras


def read_transform_matrix_from_output(output):
    import re
    str_mat = output.decode('ascii').split('\n')
    for i in range(len(str_mat)):
        str_mat[i] = re.findall(r'[+-]?[0-9.]+', str_mat[i])
    del str_mat[-1]
    return np.array(str_mat).astype(float)


def mask_to_srf(atlas_fol, region, hemi, colin_norm_fname):
    os.chdir(atlas_fol)
    rs = utils.partial_run_script(locals())
    # rs(MRI_PRETESS)
    rs(MRI_TESSELLATE)
    rs(MRIS_SMOOTH)
    rs(MRIS_CONVERT)
    shutil.move(op.join(atlas_fol, 'tmp', '{}_{}.asc'.format(region, hemi)),
                op.join(atlas_fol, 'tmp', '{}_{}.srf'.format(region, hemi)))


def convert_to_ply(srf_fname, ply_fname, org_vox2ras_tkr, shifter_vox2tk_ras):
    verts, faces, verts_num, faces_num = utils.read_srf_file(srf_fname)
    verts = apply_trans(np.linalg.inv(org_vox2ras_tkr), verts)
    verts = apply_trans(shifter_vox2tk_ras, verts)
    utils.write_ply_file(verts, faces, ply_fname)


def get_fol_name(fname):
    return op.sep.join(fname.split(op.sep)[:-1])


def get_namebase(fname):
    basename = op.basename(fname)
    if basename.endswith('nii.gz'):
        return basename[:-len('nii.gz') - 1], 'nii.gz'
    elif basename.endswith('mgz'):
        return basename[:-len('mgz') - 1], 'mgz'
    else:
        raise Exception('Unknown image type!')


def transform_to_another_subject(subject, region, subjects_dir):
    import mne.source_space
    import mne.transforms
    colin27_xfm = mne.source_space._read_talxfm('colin27', subjects_dir, 'nibabel')
    xfm = mne.source_space._read_talxfm(subject, subjects_dir, 'nibabel')
    for hemi in ['lh', 'rh']:
        verts, faces = utils.read_ply_file(op.join(MMVT_DIR, 'colin27', 'subcortical', '{}_{}.ply'.format(region, hemi)))
        verts = apply_trans(colin27_xfm['trans'], verts)
        verts = apply_trans(np.linalg.inv(xfm['trans']), verts)
        utils.write_ply_file(verts, faces, op.join(MMVT_DIR, subject, 'subcortical', '{}_{}.ply'.format(region, hemi)))


def main(region, atlas_fol, colin_T1_fname, overwrite=False):
    for hemi in ['lh', 'rh']:
        region_fname = op.join(atlas_fol, hemi, '{}.nii.gz'.format(region))
        prepare_mask_file(region_fname, overwrite=overwrite)
        mask_to_srf(atlas_fol, region, hemi, op.join(SUBJECTS_DIR, 'colin27', 'mri', 'norm.mgz'))
        region_vox2ras_tkr = get_vox2ras_tkr(region_fname)
        shifted_vox2tk_ras = calc_shifted_vox2tk_ras(region_fname, colin_T1_fname)
        convert_to_ply(op.join(atlas_fol, 'tmp', '{}_{}.srf'.format(region, hemi)),
                       op.join(atlas_fol, 'tmp', '{}_{}.ply'.format(region, hemi)),
                       region_vox2ras_tkr, shifted_vox2tk_ras)
        shutil.copy(op.join(atlas_fol, 'tmp', '{}_{}.ply'.format(region, hemi)),
                    op.join(MMVT_DIR, 'colin27', 'subcortical', '{}_{}.ply'.format(region, hemi)))


if __name__ == '__main__':
    region =  'STN' #'Striatum'
    atlas_fol = op.join(MMVT_DIR, 'maps', 'ATAG_Nonlinear_Keuken_2014')
    colin_T1_fname = op.join(SUBJECTS_DIR, 'colin27', 'mri', 'T1.mgz')
    # main(region, atlas_fol, colin_T1_fname)
    transform_to_another_subject('mg99', region, SUBJECTS_DIR)
    print('Finish!')