from src.mmvt_addon.dell import find_electrodes_in_ct as fect
import numpy as np
import os.path as op
import nibabel as nib
from itertools import product
from scipy.spatial.distance import cdist

from src.utils import utils

MMVT_DIR = utils.get_link_dir(utils.get_links_dir(), 'mmvt')


def find_local_maxima_from_voxels(voxel, ct_data, threshold, find_nei_maxima=True):
    voxels = np.array([voxel])
    maxs = fect.find_all_local_maxima(ct_data, voxels, threshold, find_nei_maxima)
    print(maxs)


def test2(ct_data, ct_header, brain, aseg, threshold, min_distance):
    ct_voxels = fect.find_voxels_above_threshold(ct_data, threshold)
    ct_voxels = fect.mask_voxels_outside_brain(ct_voxels, ct_header, brain, aseg)
    voxels = fect.find_all_local_maxima(ct_data, ct_voxels, threshold, find_nei_maxima=True, max_iters=100)
    voxels = fect.remove_neighbors_voxels(ct_data, voxels)
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
    hemi_indices, close_verts_indices, all_dists, dural_mask = fect.get_t1_voxels_inside_dural(electrodes, subject_fol)
    elc_ind = names.index(elc_name)
    t1_tkras_coords = np.array([electrodes[elc_ind]])
    verts, faces, normals = {}, {}, {}
    for hemi in ['lh', 'rh']:
        verts[hemi], faces[hemi] = nib.freesurfer.read_geometry(op.join(subject_fol, 'surf', '{}.dural'.format(hemi)))
        normals[hemi] = fect.calc_normals(verts[hemi], faces[hemi])

    hemi = hemis[elc_ind]
    dists = cdist(t1_tkras_coords, verts[hemi])
    close_verts = np.argmin(dists, axis=1)
    is_inside = fect.point_in_mesh(t1_tkras_coords[0], verts[hemi][close_verts[0]], normals[hemi][close_verts[0]])
    # vert_norm = np.linalg.norm(vertices[close_verts][0])
    # elc_norm = np.linalg.norm(t1_tkras_coords[0])
    print(is_inside)


def calc_groups_dist_to_dura(elc_name, output_fol, threshold):
    (electrodes, names, hemis, threshold) = utils.load(op.join(output_fol, '{}_electrodes.pkl'.format(int(threshold))))
    groups, noise = utils.load(op.join(output_fol, '{}_groups.pkl'.format(int(threshold))))
    elc_ind = names.index(elc_name)
    group, in_group_ind = find_electrode_group(elc_ind, groups)

    verts = {}
    for hemi in ['lh', 'rh']:
        # verts[hemi], _ = nib.freesurfer.read_geometry(op.join(subject_fol, 'surf', '{}.dural'.format(hemi)))
        verts[hemi], _ = nib.freesurfer.read_geometry(op.join(subject_fol, 'bem', 'watershed',  'mg105_brain_surface'))

    dists = cdist([electrodes[elc_ind]], verts[hemis[elc_ind]])
    close_verts_dists = np.min(dists, axis=1)
    close_verts_ind = np.argmin(dists, axis=1)
    print('{}: {} ({})'.format(names[elc_ind], close_verts_dists, verts[hemis[elc_ind]][close_verts_ind]))

    mean_dists = []
    for group in groups:
        dists = cdist(electrodes[group], verts[hemis[group[0]]])
        close_verts_dists = np.min(dists, axis=1)
        print('{}-{}: {}'.format(names[group[0]], names[group[-1]], close_verts_dists))
        mean_dists.append(np.max(close_verts_dists))
    plt.barh(np.arange(len(groups)), mean_dists, align='center', alpha=0.5)
    plt.yticks(np.arange(len(groups)), ('{}-{}'.format(names[group[0]], names[group[-1]]) for group in groups))
    plt.title('max groups dist to dural surface')
    plt.show()
    print('asdf')

def find_electrode_group(elc_ind, groups):
    groups_mask = [(elc_ind in g) for g in groups]
    if sum(groups_mask) == 1:
        group = [g for g, m in zip(groups, groups_mask) if m][0]
        in_group_ind = group.index(elc_ind)
    return group, in_group_ind


@utils.profileit(root_folder=op.join(MMVT_DIR, 'profileit'))
def get_electrodes_above_threshold(ct_data, ct_header, brain, threshold, user_fol, subject_fol, binary_erosion=True):
    # ct_data = fect.binary_erosion(ct_data, threshold)
    print('find_voxels_above_threshold...')
    ct_voxels = fect.find_voxels_above_threshold(ct_data, threshold)
    print('{} voxels were found above {}'.format(len(ct_voxels), threshold))
    print('Finding local maxima')
    ct_voxels = fect.find_all_local_maxima(
        ct_data, ct_voxels, threshold, find_nei_maxima=True, max_iters=100)
    print('{} local maxima were found'.format(len(ct_voxels)))
    ct_voxels = fect.remove_neighbors_voxels(ct_data, ct_voxels)
    print('{} local maxima after removing neighbors'.format(len(ct_voxels)))
    #utils.plot_3d_scatter(ct_voxels)
    # utils.gradient_scatter3d(ct_voxels, ct_data)
    print('mask_voxels_outside_brain...')
    ct_voxels, _ = fect.mask_voxels_outside_brain(
        ct_voxels, ct_header, brain, subject_fol, sigma=1)
    print('{} voxels in the brain were found'.format(len(ct_voxels)))
    utils.gradient_scatter3d(ct_voxels, ct_data)
    print('asdf')


def get_voxel_neighbors_ct_values(voxel, ct_data):
    fect.get_voxel_neighbors_ct_values(ct_data, voxel)


def load_find_electrode_lead_log(output_fol, logs_fol, log_name, threshold, elc1_ind=0, elc2_ind=0):
    (_, names, _, threshold) = utils.load(op.join(output_fol, '{}_electrodes.pkl'.format(int(threshold))))
    (elc_ind, electrodes, elctrodes_hemis, groups, error_radius, min_elcs_for_lead, max_dist_between_electrodes, min_distance,
     do_post_search) = utils.load(op.join(output_fol, logs_fol, '{}.pkl'.format(log_name)))

    elcs_already_in_groups = set(fect.flat_list_of_lists(groups))
    if elc1_ind in elcs_already_in_groups or elc2_ind in elcs_already_in_groups:
        print('elcs are in elcs_already_in_groups!')
    electrodes_list = set(range(len(electrodes))) - elcs_already_in_groups
    for ind in (elc1_ind, elc2_ind):
        if ind not in electrodes_list:
            print('{} are not in electrodes_list!'.format(elc_ind))

    group, noise, dists, dists_to_cylinder, gof, best_elc_ind = fect.find_electrode_group(
        elc_ind, electrodes, elctrodes_hemis, groups, error_radius, min_elcs_for_lead, max_dist_between_electrodes, min_distance,
        do_post_search)

    print('elcs_already_in_groups: {}'.format(elcs_already_in_groups))
    print('{} points between {} and {}: {}'.format(len(group), names[group[0]], names[group[-1]], [names[p] for p in group]))


def check_voxel_dist_to_dural(voxel, subject_fol, ct_header, brain_header, sigma):
    verts, faces, normals = fect.get_dural_surface(subject_fol, do_calc_normals=True)
    electrodes_t1_tkreg = fect.ct_voxels_to_t1_ras_tkr(voxel, ct_header, brain_header)
    for hemi in ['rh', 'lh']:
        dists = cdist([electrodes_t1_tkreg], verts[hemi])
        close_verts_indices = np.argmin(dists, axis=1)[0]
        inside, v = fect.point_in_mesh(electrodes_t1_tkreg, verts[hemi][close_verts_indices],
                                       normals[hemi][close_verts_indices], sigma=sigma, return_v=True)
        print(hemi, inside, v)


def check_voxels_around_electrodes_in_group(ct_data, output_fol, threshold, ct_header, brain_header):
    (electrodes, names, hemis, threshold) = utils.load(op.join(output_fol, '{}_electrodes.pkl'.format(int(threshold))))
    groups, noise = utils.load(op.join(output_fol, '{}_groups.pkl'.format(int(threshold))))
    elcs_group, elcs_colors = [], []
    # elcs_nei = np.zeros((len(utils.flat_list_of_lists(groups)), 3))
    elcs_nei = []
    groups_nei = []
    colors = utils.get_distinct_colors(len(groups))
    groups_names = ['{}-{}'.format(names[g[0]], names[g[-1]]) for g in groups]
    for group_ind, (group, color) in enumerate(zip(groups, colors)):
        group_voxels = fect.t1_ras_tkr_to_ct_voxels([electrodes[g] for g in group], ct_header, brain_header)
        groups_elecs_nei = np.array([fect.get_voxel_neighbors_ct_values(ct_data, elc_voxel, r=4) for elc_voxel in group_voxels])
        elcs_group.append([np.mean(groups_elecs_nei), np.var(groups_elecs_nei)])
        # for elc_voxel in group_voxels:
        #     elc_nei = fect.get_voxel_neighbors_ct_values(ct_data, elc_voxel, r=2)
        #     # elcs_nei.append(elc_nei)
        #     elcs_nei.append([np.sum(elc_nei), np.mean(elc_nei), np.var(elc_nei)])
        #     elcs_group.append('{}-{}'.format(names[group[0]], names[group[-1]]))
        #     elcs_colors.append(color)
    elcs_group = np.array(elcs_group)
    groups_names = np.array(groups_names)
    utils.plot_3d_scatter(elcs_group, names=groups_names)
    # elcs_nei = np.array(elcs_nei)
    # utils.plot_3d_PCA(elcs_nei, colors=elcs_colors, legend_labels=elcs_group)
    print('asdf')


def check_voxels_around_electrodes(ct_data, output_fol, threshold, ct_header, brain_header):
    (electrodes, names, hemis, threshold) = utils.load(op.join(output_fol, '{}_electrodes.pkl'.format(int(threshold))))
    groups, noise = utils.load(op.join(output_fol, '{}_groups.pkl'.format(int(threshold))))
    ct_voxels = fect.t1_ras_tkr_to_ct_voxels(electrodes, ct_header, brain_header)
    nei_features = []
    for ct_voxel in ct_voxels:
        ct_voxel_nei = fect.get_voxel_neighbors_ct_values(ct_data, ct_voxel, r=4)
        nei_features.append([np.mean(ct_voxel_nei), np.std(ct_voxel_nei)])

    from sklearn import mixture
    gmm = mixture.GaussianMixture(n_components=2, covariance_type='full')
    gmm.fit(nei_features)
    Y = gmm.predict_proba(nei_features)
    centroids = gmm.means_

    nei_features = np.array(nei_features)
    fig = plt.figure()
    plt.scatter(nei_features[:, 0], nei_features[:, 1])
    plt.scatter(centroids[:, 0], centroids[:, 1], c='r')
    plt.xlabel('mean')
    plt.ylabel('std')
    plt.show()


def point_in_surface_cylinder(subject, elc1_name, elc2_name, mmvt_dir, user_fol, threshold, radius_sq):
    (electrodes, names, hemis, threshold) = utils.load(op.join(output_fol, '{}_electrodes.pkl'.format(int(threshold))))
    elc1_ind = names.index(elc1_name)
    elc2_ind = names.index(elc2_name)

    verts_neighbors_fname = op.join(mmvt_dir, subject, 'verts_neighbors_{hemi}.pkl')
    verts = fect.read_pial_verts(user_fol)
    pt1, pt2 = [], []
    hemi = 'rh'
    dists = cdist([pt1, pt2], verts[hemi])


if __name__ == '__main__':
    from src.utils import utils
    import nibabel as nib
    import matplotlib.pyplot as plt
    subject = 'nmr01209' #'mg105' # 'nmr01183'
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
    threshold = 3000 # np.percentile(ct_data, threshold_percentile)
    threshold = 1826
    print('threshold: {}'.format(threshold))
    output_fol = op.join(mmvt_dir, subject, 'ct', 'finding_electrodes_in_ct')
    user_fol = op.join(mmvt_dir, subject)
    subject_fol = op.join(subjects_dir, subject)

    # find_local_maxima_from_voxels([97, 88, 125], ct_data, threshold, find_nei_maxima=False)
    # test2(ct_data, ct.header, brain, aseg, threshold, min_distance)
    # test3(ct_data, threshold, ct.header, brain, aseg, user_fol)
    # find_path(ct_data, threshold)
    # find_group_between_pair('RUN72', 'RUN79', threshold, output_fol, min_distance, error_r)
    # mask_voxels_outside_brain('RUN133', output_fol, threshold, ct.header, brain, aseg, user_fol, subject_fol)
    # find_closest_points_on_cylinder('RUN42', 'RUN112', threshold, output_fol, error_r)
    # calc_dist_on_cylinder('RUN57', 'RUN82', threshold, output_fol, error_r)
    # check_if_outside_pial(threshold, user_fol, output_fol, subject_fol, ct.header, brain, aseg, sigma=2)
    # check_dist_to_pial_vertices('LUN195', subject_fol, threshold)
    # check_voxel_dist_to_dural([130, 85, 157], subject_fol, ct.header, brain.header, sigma=1)
    # calc_groups_dist_to_dura('RUN98', output_fol, threshold)
    # get_electrodes_above_threshold(ct_data, ct.header, brain, threshold, user_fol, subject_fol)
    # get_voxel_neighbors_ct_values([97, 88, 125], ct_data)
    # load_find_electrode_lead_log(output_fol, 'f7ea9', '_find_electrode_lead_302-335_302_2951', threshold)
    # check_voxels_around_electrodes_in_group(ct_data, output_fol, threshold, ct.header, brain.header)
    # check_voxels_around_electrodes(ct_data, output_fol, threshold, ct.header, brain.header)
    point_in_surface_cylinder(subject, 'G38', 'G26', mmvt_dir, user_fol, threshold, error_r)