import os.path as op
import numpy as np
import mne
import nibabel as nib
from src.utils import utils


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


def read_transform_matrix_from_output(output):
    import re
    try:
        str_mat = output.decode('ascii').split('\n')
    except:
        str_mat = output.split('\n')
    for i in range(len(str_mat)):
        str_mat[i] = re.findall(r'[+-]?[0-9.]+', str_mat[i])
    del str_mat[-1]
    return np.array(str_mat).astype(float)


def apply_trans(trans, points):
    return np.array([np.dot(trans, np.append(p, 1))[:3] for p in points])


def get_talxfm(subject, subjects_dir, return_trans_obj=False):
    trans = mne.source_space._read_talxfm(subject, subjects_dir, 'nibabel')
    if not return_trans_obj:
        trans = trans['trans']
    return trans


def tkras_to_mni(points, subject, subjects_dir):
    # https://mail.nmr.mgh.harvard.edu/pipermail/freesurfer/2012-June/024293.html
    # MNI305RAS = TalXFM * orig_vox2ras * inv(orig_vox2tkras) * [tkrR tkrA tkrS 1]
    tal_xfm = get_talxfm(subject, subjects_dir)
    orig_vox2ras = get_vox2ras(op.join(subjects_dir, subject, 'mri', 'orig.mgz'))
    orig_vox2tkras = get_vox2ras_tkr(op.join(subjects_dir, subject, 'mri', 'orig.mgz'))
    trans = tal_xfm @ orig_vox2ras @ inv(orig_vox2tkras)
    points = apply_trans(trans, points)
    # points = apply_trans(tal_xfm, points)
    # points = apply_trans(orig_vox2ras, points)
    # points = apply_trans(np.linalg.inv(orig_vox2tkras), points)

    return points


def mni_to_tkras(points, subject, subjects_dir, tal_xfm=None, orig_vox2ras=None, orig_vox2tkras=None):
    # https://mail.nmr.mgh.harvard.edu/pipermail/freesurfer/2012-June/024293.html
    # MNI305RAS = TalXFM * orig_vox2ras * inv(orig_vox2tkras) * [tkrR tkrA tkrS 1]
    if tal_xfm is None:
        tal_xfm = get_talxfm(subject, subjects_dir)
    if orig_vox2ras is None:
        orig_vox2ras = get_vox2ras(op.join(subjects_dir, subject, 'mri', 'orig.mgz'))
    if orig_vox2tkras is None:
        orig_vox2tkras = get_vox2ras_tkr(op.join(subjects_dir, subject, 'mri', 'orig.mgz'))
    # Only in python 3.5:
    # trans = tal_xfm @ orig_vox2ras @ np.linalg.inv(orig_vox2tkras)
    # trans = tal_xfm.dot(orig_vox2ras).dot(np.linalg.inv(orig_vox2tkras))
    # trans = np.linalg.inv(trans)
    trans = inv(tal_xfm @ orig_vox2ras @ inv(orig_vox2tkras))
    # points = apply_trans(orig_vox2tkras, points)
    # points = apply_trans(np.linalg.inv(orig_vox2ras), points)
    # points = apply_trans(np.linalg.inv(tal_xfm), points)
    points = apply_trans(trans, points)
    return points


def inv(x):
    return np.linalg.inv(x)


def mni305_to_mni152_matrix():
    # http://freesurfer.net/fswiki/CoordinateSystems
    # The folowing matrix is V152*inv(T152)*R*T305*inv(V305), where V152 and V305 are the vox2ras matrices from the
    # 152 and 305 spaces, T152 and T305 are the tkregister-vox2ras matrices from the 152 and 305 spaces,
    # and R is from $FREESURFER_HOME/average/mni152.register.dat
    M = [[0.9975, - 0.0073, 0.0176, -0.0429],
         [0.0146, 1.0009, -0.0024, 1.5496],
         [-0.0130, -0.0093, 0.9971, 1.1840],
         [0, 0, 0, 1.0000]]
    return np.array(M)


def mni305_to_mni152(points):
    return apply_trans(mni305_to_mni152_matrix(), points)


def mni152_mni305(points):
    return apply_trans(np.linalg.inv(mni305_to_mni152_matrix()), points)


def tkras_to_vox(points, subject_orig_header=None, subject='', subjects_dir=''):
    if subject_orig_header is None:
        subject_orig_header = get_subject_mri_header(subject, subjects_dir)
    vox2ras_tkr = subject_orig_header.get_vox2ras_tkr()
    ras_tkr2vox = np.linalg.inv(vox2ras_tkr)
    vox = apply_trans(ras_tkr2vox, points)
    return vox


def vox_to_ras(points, subject_orig_header=None, subject='', subjects_dir=''):
    if subject_orig_header is None:
        subject_orig_header = get_subject_mri_header(subject, subjects_dir)
    vox2ras = subject_orig_header.get_vox2ras()
    ras = apply_trans(vox2ras, points)
    return ras


def get_subject_mri_header(subject, subjects_dir, image_name='T1.mgz'):
    image_fname = op.join(subjects_dir, subject, 'mri', image_name)
    if op.isfile(image_fname):
        d = nib.load(image_fname)# 'orig.mgz'))
        subject_orig_header = d.get_header()
    else:
        print("get_subject_mri_header: Can't find image! ({})".format(image_fname))
        subject_orig_header = None
    return subject_orig_header


if __name__ == '__main__':
    from src.utils import preproc_utils as pu
    SUBJECTS_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()
    subject = 'mg78'
    points = [[-17.85, -52.18, 42.08]]
    true_ras = [[-16.92, -44.43, 29.06]]
    true_vox = [[146, 86, 76]]
    # point = [-13.1962, -66.5584, 33.3018]

    h = get_subject_mri_header(subject, SUBJECTS_DIR)
    vox = tkras_to_vox(points, h)
    print('vox: {}'.format(vox))

    ras = vox_to_ras(vox, h)
    print('ras: {}'.format(ras))

    import mne
    from mne.source_space import vertex_to_mni, combine_transforms, Transform, apply_trans

    mni2 = vertex_to_mni(23633, 0, 'mg78', SUBJECTS_DIR)
    print('mni2: {}'.format(mni2))