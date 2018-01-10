from src.mmvt_addon.dell import find_electrodes_in_ct as fect
import numpy as np
import os.path as op
from itertools import product


def test1(ct_data, threshold):
    voxels = np.array([[90, 75, 106]])
    maxs = fect.find_all_local_maxima(ct_data, voxels, threshold, find_nei_maxima=True)
    print(maxs)


def test2(ct_data, ct_header, brain, aseg, threshold, min_distance):
    ct_voxels = fect.find_voxels_above_threshold(ct_data, threshold)
    ct_voxels = fect.mask_voxels_outside_brain(ct_voxels, ct_header, brain, aseg)
    voxels = fect.find_all_local_maxima(ct_data, ct_voxels, threshold, find_nei_maxima=True, max_iters=100)
    voxels = fect.remove_neighbors_voexls(ct_data, voxels)
    print('asdf')


def test3(ct_data, threshold, ct_header, brain, aseg=None, user_fol=''):
    ct_voxels = fect.find_voxels_above_threshold(ct_data, threshold)
    brain_header = brain.get_header()
    brain_mask = brain.get_data()
    ct_vox2ras, ras2t1_vox, vox2t1_ras_tkr = fect.get_trans(ct_header, brain_header)
    ct_ras = fect.apply_trans(ct_vox2ras, ct_voxels)
    t1_vox = np.rint(fect.apply_trans(ras2t1_vox, ct_ras)).astype(int)
    t1_tkreg = fect.apply_trans(vox2t1_ras_tkr, t1_vox)
    t1_voxels_outside_pial, _ = fect.get_t1_voxels_outside_pial(user_fol, brain_header, brain_mask, aseg, t1_tkreg, t1_vox)
    t1_voxels_outside_pial = set([tuple(v) for v in t1_voxels_outside_pial])
    print('sdf')


def test4(ct_data):
    voxel1 = np.array([102, 99, 131])
    voxel2 = np.array([104, 102, 131])
    points = np.array(list(product(*[range(voxel1[k], voxel2[k] + 1) for k in range(3)])))
    inds, _, _ = fect.points_in_cylinder(voxel1, voxel2, points, 1, metric='cityblock')
    path = points[inds]
    diffs = [pt2 - pt1 for pt1, pt2 in zip(path[:-1], path[1:])]
    path = path[[np.all(d>=0) for d in diffs]]
    path_ct_data = [ct_data[tuple(p)] for p in path]
    print(path)


def test5(ct_data, threshold):
    vox1 = np.array([102, 99, 131])
    vox2 = np.array([104, 102, 131])
    is_there_path = fect.find_path(vox1, vox2, ct_data, threshold)
    max1 = fect.find_local_maxima_in_ct(ct_data, vox1)
    max2 = fect.find_local_maxima_in_ct(ct_data, vox2)
    print(is_there_path)


def test6(elc1, elc2, threshold, output_fol, min_distance=3, error_radius=2):
    (electrodes, names, hemis, threshold) = utils.load(op.join(output_fol, '{}_electrodes.pkl'.format(int(threshold))))
    elc_ind1, elc_ind2 = names.index(elc1), names.index(elc2)
    group, too_close_points, dists, dists_to_cylinder = \
        fect.find_group_between_pair(elc_ind1, elc_ind2, electrodes, error_radius, min_distance)
    print(group, too_close_points, dists, dists_to_cylinder)


def test7(elc_name, output_fol, threshold, ct_header, brain, aseg, user_fol):
    (electrodes, names, hemis, threshold) = utils.load(op.join(output_fol, '{}_electrodes.pkl'.format(int(threshold))))
    elc_ind = names.index(elc_name)
    t1_tkras_coords = np.array([electrodes[elc_ind]])
    ct_voxels = fect.t1_ras_tkr_to_ct_voxels(t1_tkras_coords, ct_header, brain.header)
    voxels = fect.mask_voxels_outside_brain(ct_voxels, ct_header, brain, user_fol, aseg)
    print(voxels)


if __name__ == '__main__':
    from src.utils import utils
    import nibabel as nib
    import matplotlib.pyplot as plt
    subject = 'mg105'
    threshold_percentile = 99.9
    min_distance = 3
    error_r = 2

    links_dir = utils.get_links_dir()
    mmvt_dir = utils.get_link_dir(links_dir, 'mmvt')
    subjects_dir = utils.get_link_dir(links_dir, 'subjects', 'SUBJECTS_DIR')

    ct_name = 'ct_reg_to_mr.mgz'
    brain_mask_name = 'brain.mgz'
    aseg_name = 'aseg.mgz'
    brain_mask_fname = op.join(subjects_dir, subject, 'mri', brain_mask_name)
    aseg_fname = op.join(subjects_dir, subject, 'mri', aseg_name)
    ct_fname = op.join(mmvt_dir, subject, 'ct', ct_name)
    ct = nib.load(ct_fname)
    ct_data = ct.get_data()
    brain = nib.load(brain_mask_fname)
    aseg = nib.load(aseg_fname).get_data() if op.isfile(aseg_fname) else None
    threshold = np.percentile(ct_data, threshold_percentile)
    print('threshold: {}'.format(threshold))
    output_fol = op.join(mmvt_dir, subject, 'ct', 'finding_electrodes_in_ct')

    # test1(ct_data, threshold)
    # test2(ct_data, ct.header, brain, aseg, threshold, min_distance)
    # test3(ct_data, threshold, ct.header, brain, aseg, op.join(mmvt_dir, subject))
    # test5(ct_data, threshold)
    test6('RUN72', 'RUN79', threshold, output_fol, min_distance, error_r)
    # test7('RUN133', output_fol, threshold, ct.header, brain, aseg, op.join(mmvt_dir, subject))
