import os.path as op
import nibabel as nib
import numpy as np
import shutil
import os

from src.utils import utils
from src.utils import trans_utils as tu

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
    points = tu.apply_trans(np.linalg.inv(org_vox2ras_tkr), points)
    points = tu.apply_trans(shifter_vox2tk_ras, points)
    return points


def get_shifted_vox2tk_ras(zero_in_T1_tk_ras, voxels_sizes):
    return np.array(
        [[-voxels_sizes[0], 0.0, 0.0, zero_in_T1_tk_ras[0]],
         [0.0, voxels_sizes[1], 0.0, zero_in_T1_tk_ras[1]],
         [0.0, 0.0, voxels_sizes[2], zero_in_T1_tk_ras[2]],
         [0.0, 0.0, 0.0, 1.0]])


def calc_shifted_vox2tk_ras(region_fname, colin_T1_fname):
    region_vox2ras = tu.get_vox2ras(region_fname)
    zero_in_ras = utils.apply_trans(region_vox2ras, [[0, 0, 0]])
    zero_in_T1_voxels = utils.apply_trans(np.linalg.inv(tu.get_vox2ras(colin_T1_fname)), zero_in_ras)
    zero_in_T1_tk_ras = utils.apply_trans(tu.get_vox2ras_tkr(colin_T1_fname), zero_in_T1_voxels)
    voxels_sizes = [abs(v) for v in np.diag(region_vox2ras)[:-1]]
    shifted_vox2tk_ras = get_shifted_vox2tk_ras(zero_in_T1_tk_ras[0], voxels_sizes)
    return shifted_vox2tk_ras


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
    verts = tu.apply_trans(np.linalg.inv(org_vox2ras_tkr), verts)
    verts = tu.apply_trans(shifter_vox2tk_ras, verts)
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
    colin27_xfm = tu.get_talxfm('colin27', subjects_dir)
    xfm = tu.get_talxfm('colin27', subjects_dir)
    for hemi in ['lh', 'rh']:
        verts, faces = utils.read_ply_file(op.join(MMVT_DIR, 'colin27', 'subcortical', '{}_{}.ply'.format(region, hemi)))
        verts = tu.apply_trans(colin27_xfm, verts)
        verts = tu.apply_trans(np.linalg.inv(xfm), verts)
        utils.write_ply_file(verts, faces, op.join(MMVT_DIR, subject, 'subcortical', '{}_{}.ply'.format(region, hemi)))


def main(region, atlas_fol, colin_T1_fname, overwrite=False):
    for hemi in ['lh', 'rh']:
        region_fname = op.join(atlas_fol, hemi, '{}.nii.gz'.format(region))
        prepare_mask_file(region_fname, overwrite=overwrite)
        mask_to_srf(atlas_fol, region, hemi, op.join(SUBJECTS_DIR, 'colin27', 'mri', 'norm.mgz'))
        region_vox2ras_tkr = tu.get_vox2ras_tkr(region_fname)
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
    transform_to_another_subject('mg78', region, SUBJECTS_DIR)
    print('Finish!')