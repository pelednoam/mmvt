import os.path as op
import nibabel as nib
import numpy as np
import mne

from src.utils import utils


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


def to_ras(points):
    points = apply_trans(np.linalg.inv(vox2ras_used()), points)
    points = apply_trans(vox2ras(), points)
    return points


def apply_trans(trans, points):
    return np.array([np.dot(trans, np.append(p, 1))[:3] for p in points])


def vox2ras_used():
    return np.array(
        [[-0.300, 0.000, 0.000, 25.200],
        [0.000, 0.000, 0.300, -29.700],
        [0.000, -0.300, 0.000, 37.200],
        [0.000, 0.000, 0.000, 1.000]])


def vox2ras():
    return np.array(
        [[-0.300, 0.000, 0.000, 5.1000],
         [0.000, 0.300, 0.000, -40.000],
         [0.000, 0.000, 0.300, -23.000],
         [0.000, 0.000, 0.000, 1.000]])


def convert_to_ply(srf_fname, region_name, affine_trans=None):
    ply_fname = op.join(get_fol_name(srf_fname), '{}.ply'.format(region_name))
    verts, faces, verts_num, faces_num = utils.read_srf_file(srf_fname)
    verts = to_ras(verts)
    # if affine_trans:
    #     new_verts = mne.transforms.apply_trans(affine_trans, verts)
    # verts[:, [1, 2]] = verts[:, [2, 1]]
    utils.write_ply_file(verts, faces, ply_fname)
    # utils.srf2ply(srf_fname, )


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


if __name__ == '__main__':
    atals_fol = '/homes/5/npeled/space1/mmvt/maps/ATAG_Nonlinear_Keuken_2014'
    for hemi in ['lh', 'rh']:
        region_fname = op.join(atals_fol, hemi, 'Striatum.nii.gz')
        # prepare_mask_file(region_fname, overwrite=True)
    convert_to_ply(op.join(atals_fol, 'tmp', 'Striatum_notsmooth.srf'), 'striatum_lh')
    print('Finish!')