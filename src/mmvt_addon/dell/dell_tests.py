from src.mmvt_addon.dell import find_electrodes_in_ct as fect
import numpy as np
import os.path as op
import nibabel as nib
from itertools import product
from scipy.spatial.distance import cdist


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


def find_path(ct_data, threshold):
    vox1 = np.array([102, 99, 131])
    vox2 = np.array([104, 102, 131])
    is_there_path = fect.find_path(vox1, vox2, ct_data, threshold)
    max1 = fect.find_local_maxima_in_ct(ct_data, vox1)
    max2 = fect.find_local_maxima_in_ct(ct_data, vox2)
    print(is_there_path)


def find_group_between_pair(elc1, elc2, threshold, output_fol, min_distance=3, error_radius=2):
    (electrodes, names, hemis, threshold) = utils.load(op.join(output_fol, '{}_electrodes.pkl'.format(int(threshold))))
    elc_ind1, elc_ind2 = names.index(elc1), names.index(elc2)
    group, too_close_points, dists, dists_to_cylinder = \
        fect.find_group_between_pair(elc_ind1, elc_ind2, electrodes, error_radius, min_distance)
    print(group, too_close_points, dists, dists_to_cylinder)


# @utils.profileit(root_folder=op.join(MMVT_DIR, 'profileit'))
def mask_voxels_outside_brain(elc_name, output_fol, threshold, ct_header, brain, aseg, user_fol, subject_fol):
    (electrodes, names, hemis, threshold) = utils.load(op.join(output_fol, '{}_electrodes.pkl'.format(int(threshold))))
    # elc_ind = names.index(elc_name)
    # t1_tkras_coords = np.array([electrodes[elc_ind]])
    # ct_voxels = fect.t1_ras_tkr_to_ct_voxels(t1_tkras_coords, ct_header, brain.header)
    ct_voxels = fect.t1_ras_tkr_to_ct_voxels(electrodes, ct_header, brain.header)
    voxels = fect.mask_voxels_outside_brain(ct_voxels, ct_header, brain, user_fol, subject_fol, aseg)
    print(voxels)


def find_closest_points_on_cylinder(elc1, elc2, threshold, output_fol, error_radius):
    (electrodes, names, hemis, threshold) = utils.load(op.join(output_fol, '{}_electrodes.pkl'.format(int(threshold))))
    elc_ind1, elc_ind2 = names.index(elc1), names.index(elc2)
    _, cylinder, _ = fect.points_in_cylinder(
        electrodes[elc_ind1], electrodes[elc_ind2], electrodes, error_radius)
    points_inside_cylinder, too_close_points, dists, dists_to_cylinder = \
        fect.find_group_between_pair(elc_ind1, elc_ind2, electrodes, error_radius, min_distance)
    fect.find_closest_points_on_cylinder(electrodes, points_inside_cylinder, cylinder)


def calc_dist_on_cylinder(elc1, elc2, threshold, output_fol, error_radius):
    (electrodes, names, hemis, threshold) = utils.load(op.join(output_fol, '{}_electrodes.pkl'.format(int(threshold))))
    elc_ind1, elc_ind2 = names.index(elc1), names.index(elc2)
    groups, noise = utils.load(op.join(output_fol, '{}_groups.pkl'.format(int(threshold))))
    groups_mask = [(elc_ind1 in g) for g in groups]
    if sum(groups_mask) == 1:
        group = [g for g, m in zip(groups, groups_mask) if m][0]
        print('Electrodes belongs to {}-{}'.format(names[group[0]], names[group[1]]))
    else:
        print('No group was found!')
        return
    _, cylinder, _ = fect.points_in_cylinder(electrodes[group[0]], electrodes[group[-1]], electrodes, error_radius)
    closest_points = fect.find_closest_points_on_cylinder(electrodes, [elc_ind1, elc_ind2], cylinder)
    dist = np.linalg.norm(closest_points[0]-closest_points[1])
    print(dist)


def check_if_outside_pial(threshold, user_fol, output_fol, subject_fol, ct_header, brain, aseg, sigma=2):
    (electrodes, names, hemis, threshold) = utils.load(op.join(output_fol, '{}_electrodes.pkl'.format(int(threshold))))
    all_voxels = fect.t1_ras_tkr_to_ct_voxels(electrodes, ct_header, brain.header)
    voxels = fect.t1_ras_tkr_to_ct_voxels(electrodes, ct_header, brain.header)
    voxels_in, voxels_in_indices = fect.mask_voxels_outside_brain(
        voxels, ct_header, brain, user_fol, subject_fol, aseg, None, sigma)
    indices_outside_brain = list(set(range(len(voxels))) - set(voxels_in_indices))
    outside_voxels = all_voxels[indices_outside_brain]
    outside_voxels_norm = [np.linalg.norm(v) for v in outside_voxels]
    plt.hist(outside_voxels_norm, bins=40)
    plt.show()
    print('asdf')


def check_dist_to_pial_vertices(elc_name, subject_fol, threshold):
    (electrodes, names, hemis, threshold) = utils.load(op.join(output_fol, '{}_electrodes.pkl'.format(int(threshold))))
    hemi_indices, close_verts_indices, all_dists, dural_mask = get_t1_voxels_inside_dural(electrodes, subject_fol)
    elc_ind = names.index(elc_name)
    t1_tkras_coords = np.array([electrodes[elc_ind]])
    verts, faces, normals = {}, {}, {}
    for hemi in ['lh', 'rh']:
        verts[hemi], faces[hemi] = nib.freesurfer.read_geometry(op.join(subject_fol, 'surf', '{}.dural'.format(hemi)))
        normals[hemi] = fect.calc_normals(verts[hemi], faces[hemi])

    hemi = hemis[elc_ind]
    dists = cdist(t1_tkras_coords, verts[hemi])
    close_verts = np.argmin(dists, axis=1)
    is_inside = point_in_mesh(t1_tkras_coords[0], verts[hemi][close_verts[0]], normals[hemi][close_verts[0]])
    # vert_norm = np.linalg.norm(vertices[close_verts][0])
    # elc_norm = np.linalg.norm(t1_tkras_coords[0])
    print(is_inside)


def get_t1_voxels_inside_dural(t1_tkreg, subject_fol):
    verts, normals, min_dists, close_verts_indices = {}, {}, {}, {}
    hemis = ['lh', 'rh']
    for hemi_ind, hemi in enumerate(hemis):
        verts[hemi_ind], faces = nib.freesurfer.read_geometry(op.join(subject_fol, 'surf', '{}.dural'.format(hemi)))
        # todo: check the normals vs Blender normals
        normals[hemi_ind] = fect.calc_normals(verts[hemi_ind], faces)
        dists = cdist(t1_tkreg, verts[hemi_ind])
        min_dists[hemi] = np.min(dists, axis=1)
        close_verts_indices[hemi] = np.argmin(dists, axis=1)
    all_dists = [min_dists[hemi] for hemi in hemis]
    hemi_indices = np.argmin(all_dists, axis=0)
    close_verts_indices = np.array([close_verts_indices[hemi] for hemi in hemis]).T
    close_verts_indices = [close_vert_indices[h] for close_vert_indices, h in zip(close_verts_indices, hemi_indices)]
    dural_mask = [point_in_mesh(u, verts[hemi][vert_ind], normals[hemi][vert_ind]) for u, vert_ind, hemi in
                  zip(t1_tkreg, close_verts_indices, hemi_indices)]
    return [hemis[h] for h in hemi_indices], close_verts_indices, np.array(all_dists).T, dural_mask


def point_in_mesh(point, closest_vert, closeset_vert_normal):
    # https://blender.stackexchange.com/questions/31693/how-to-find-if-a-point-is-inside-a-mesh
    p2 = point - closest_vert
    v = p2.dot(closeset_vert_normal)
    print(v)
    return not(v < 0.0)


if __name__ == '__main__':
    from src.utils import utils
    import nibabel as nib
    import matplotlib.pyplot as plt
    subject = 'mg105'
    threshold_percentile = 99.9
    min_distance = 2.5
    error_r = 2
    min_elcs_for_lead = 4
    max_dist_between_electrodes = 15

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
    user_fol = op.join(mmvt_dir, subject)
    subject_fol = op.join(subjects_dir, subject)

    # test1(ct_data, threshold)
    # test2(ct_data, ct.header, brain, aseg, threshold, min_distance)
    # test3(ct_data, threshold, ct.header, brain, aseg, user_fol)
    # find_path(ct_data, threshold)
    # find_group_between_pair('RUN72', 'RUN79', threshold, output_fol, min_distance, error_r)
    mask_voxels_outside_brain('RUN133', output_fol, threshold, ct.header, brain, aseg, user_fol, subject_fol)
    # find_closest_points_on_cylinder('RUN42', 'RUN112', threshold, output_fol, error_r)
    # calc_dist_on_cylinder('RUN57', 'RUN82', threshold, output_fol, error_r)
    # check_if_outside_pial(threshold, user_fol, output_fol, subject_fol, ct.header, brain, aseg, sigma=2)
    # check_dist_to_pial_vertices('LUN195', subject_fol, threshold)