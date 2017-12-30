import numpy as np
import os.path as op
from collections import Counter
from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.externals import joblib
from sklearn import mixture
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import time


def find_voxels_above_threshold(ct_data, threshold):
    return np.array(np.where(ct_data > threshold)).T


def mask_voxels_outside_brain(voxels, ct_header, brain, user_fol, aseg=None):
    brain_header = brain.get_header()
    brain_mask = brain.get_data()
    ct_vox2ras, ras2t1_vox, vox2t1_ras_tkr = get_trans(ct_header, brain_header)
    ct_ras = apply_trans(ct_vox2ras, voxels)
    t1_vox = np.rint(apply_trans(ras2t1_vox, ct_ras)).astype(int)
    t1_tkreg = apply_trans(vox2t1_ras_tkr, t1_vox)
    t1_voxels_outside_pial = get_t1_voxels_outside_pial(user_fol, brain_header, brain_mask, aseg, t1_vox, t1_tkreg)
    t1_voxels_outside_pial = set([tuple(v) for v in t1_voxels_outside_pial])
    if aseg is None:
        voxels = np.array([v for v, t1_v in zip(voxels, t1_vox) if brain_mask[tuple(t1_v)] > 0])
    else:
        voxels = np.array([v for v, t1_v in zip(voxels, t1_vox) if brain_mask[tuple(t1_v)] > 0 and
                           aseg[tuple(t1_v)] not in [7, 8] and tuple(t1_v) not in t1_voxels_outside_pial])
    return voxels


def get_t1_voxels_outside_pial(user_fol, brain_header, brain_mask, aseg, t1_vox, t1_tkreg, vertices=None):
    unknown = np.array([t1_t for t1_v, t1_t in zip(t1_vox, t1_tkreg)
                        if brain_mask[tuple(t1_v)] > 0 and aseg[tuple(t1_v)] == 0])
    if vertices is None:
        print('Loading pial vertices for finding electrodes hemis')
        verts = read_pial_verts(user_fol)
        vertices = np.concatenate((verts['rh'], verts['lh']))
    dists = cdist(unknown, vertices)
    close_verts = np.argmin(dists, axis=1)
    outside_pial = [u for u, v in zip(unknown, close_verts) if np.linalg.norm(u) - np.linalg.norm(vertices[v]) > 0]
    voxel_outside_pial = apply_trans(np.linalg.inv(brain_header.get_vox2ras_tkr()), outside_pial)
    return voxel_outside_pial


def get_trans(ct_header, brain_header):
    ct_vox2ras = ct_header.get_vox2ras()
    ras2t1_vox = np.linalg.inv(brain_header.get_vox2ras())
    vox2t1_ras_tkr = brain_header.get_vox2ras_tkr()
    return ct_vox2ras, ras2t1_vox, vox2t1_ras_tkr


def apply_trans(trans, points):
    points = np.hstack((points, np.ones((len(points), 1))))
    points = np.dot(trans, points.T).T
    return points[:, :3]


def find_all_local_maxima(ct_data, voxels, threshold=0, find_nei_maxima=False, max_iters=100):
    maxs = set()
    for run, voxel in enumerate(voxels):
        if find_nei_maxima:
            max_voxel = find_local_nei_maxima_in_ct(ct_data, voxel, threshold, max_iters)
        else:
            max_voxel = find_local_maxima_in_ct(ct_data, voxel, max_iters)
        if max_voxel is not None:
            maxs.add(tuple(max_voxel))
    maxs = np.array([np.array(vox) for vox in maxs])
    return maxs


def remove_neighbors_voexls(ct_data, voxels):
    dists = cdist(voxels, voxels, 'cityblock')
    inds = np.where(dists == 1)
    if len(inds[0]) > 0:
        pairs = list(set([tuple(sorted([inds[0][k], inds[1][k]])) for k in range(len(inds[0]))]))
        to_remove = [pair[0] if ct_data[tuple(voxels[pair[0]])] < ct_data[tuple(voxels[pair[1]])]
                     else pair[1] for pair in pairs]
        voxels = np.delete(voxels, to_remove, axis=0)
    return voxels


def clustering(data, ct_data, n_components, get_centroids=True, clustering_method='knn', output_fol='', threshold=0,
               covariance_type='full'):
    if clustering_method == 'gmm':
        centroids, Y = gmm_clustering(data, n_components, covariance_type, threshold, output_fol)
    elif clustering_method == 'knn':
        centroids, Y = knn_clustering(data, n_components, output_fol, threshold)
    if get_centroids:
        centroids = np.rint(centroids).astype(int)
        # for ind, centroid in enumerate(centroids):
        #     centroids[ind] = find_local_maxima_in_ct(ct_data, centroid, threshold)
    else: # get max CT intensity
        centroids = np.zeros(centroids.shape, dtype=np.int)
        labels = np.unique(Y)
        for ind, label in enumerate(labels):
            voxels = data[Y == label]
            centroids[ind] = voxels[np.argmax([ct_data[tuple(voxel)] for voxel in voxels])]
    print(np.all([ct_data[tuple(voxel)] > threshold for voxel in centroids]))
    return centroids, Y


def knn_clustering(data, n_components, output_fol='', threshold=0):
    kmeans = KMeans(n_clusters=n_components, random_state=0)
    if output_fol != '':
        output_fname = op.join(output_fol, 'kmeans_model_{}.pkl'.format(int(threshold)))
        if not op.isfile(output_fname):
            kmeans.fit(data)
            print('Saving knn model to {}'.format(output_fname))
            joblib.dump(kmeans, output_fname, compress=9)
        else:
            kmeans = joblib.load(output_fname)
    else:
        kmeans.fit(data)
    Y = kmeans.predict(data)
    centroids = kmeans.cluster_centers_
    return centroids, Y


def gmm_clustering(data, n_components, covariance_type='full', output_fol='', threshold=0):
    gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=covariance_type)
    if output_fol != '':
        output_fname = op.join(output_fol, 'gmm_model_{}.pkl'.format(int(threshold)))
        if not op.isfile(output_fname):
            gmm.fit(data)
            print('Saving gmm model to {}'.format(output_fname))
            joblib.dump(gmm, output_fname, compress=9)
        else:
            gmm = joblib.load(output_fname)
    else:
        gmm.fit(data)
    Y = gmm.predict(data)
    centroids = gmm.means_
    return centroids, Y


def find_local_maxima_in_ct(ct_data, voxel, max_iters=100):
    peak_found, iter_num = False, 0
    voxel_max = voxel.copy()
    while not peak_found and iter_num < max_iters:
        max_ct_data = ct_data[tuple(voxel_max)]
        max_diffs = (0, 0, 0)
        neighbors = get_voxel_neighbors_ct_values(ct_data, voxel_max, True)
        for ct_value, delta in neighbors:
            if ct_value > max_ct_data:
                max_ct_data = ct_value
                max_diffs = delta
        peak_found = max_diffs == (0, 0, 0)
        voxel_max += max_diffs
        iter_num += 1
    if not peak_found:
        # print('Peak was not found!')
        voxel_max = None
    return voxel_max


def find_local_nei_maxima_in_ct(ct_data, voxel, threshold=0, max_iters=100):
    peak_found, iter_num = False, 0
    voxel_max = voxel.copy()
    while not peak_found and iter_num < max_iters:
        max_nei_ct_data = sum(get_voxel_neighbors_ct_values(ct_data, voxel_max, False))
        max_diffs = (0, 0, 0)
        neighbors = get_voxel_neighbors_ct_values(ct_data, voxel_max, True)
        for ct_val, delta in neighbors:
            neighbors_neighbors_ct_val = sum(get_voxel_neighbors_ct_values(ct_data, voxel+delta, False))
            if neighbors_neighbors_ct_val > max_nei_ct_data and ct_val > threshold:
                max_nei_ct_data = neighbors_neighbors_ct_val
                max_diffs = delta
        peak_found = max_diffs == (0, 0, 0)
        voxel_max += max_diffs
        iter_num += 1
    if not peak_found:
        # print('Peak was not found!')
        voxel_max = None
    return voxel_max


def get_voxel_neighbors_ct_values(ct_data, voxel, include_new_voxel=False):
    x, y, z = np.rint(voxel).astype(int)
    if include_new_voxel:
        return [(ct_data[x + dx, y + dy, z + dz], (dx, dy, dz))
                for dx, dy, dz in product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])]
    else:
        return [ct_data[x + dx, y + dy, z + dz] for dx, dy, dz in product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])]


def ct_voxels_to_t1_ras_tkr(centroids, ct_header, brain_header):
    ct_vox2ras, ras2t1_vox, vox2t1_ras_tkr = get_trans(ct_header, brain_header)
    centroids_ras = apply_trans(ct_vox2ras, centroids)
    centroids_t1_vox = apply_trans(ras2t1_vox, centroids_ras)
    centroids_t1_ras_tkr = apply_trans(vox2t1_ras_tkr, centroids_t1_vox)
    return centroids_t1_ras_tkr


def t1_ras_tkr_to_ct_voxels(t1_tkras_coords, ct_header, brain_header):
    ndim = t1_tkras_coords.ndim
    if ndim == 1:
        t1_tkras_coords = np.array([t1_tkras_coords])
    t1_vox = apply_trans(np.linalg.inv(brain_header.get_vox2ras_tkr()), t1_tkras_coords)
    t1_ras = apply_trans(brain_header.get_vox2ras(), t1_vox)
    ct_vox = np.rint(apply_trans(np.linalg.inv(ct_header.get_vox2ras()), t1_ras)).astype(int)
    if ndim == 1:
        ct_vox = ct_vox[0]
    return ct_vox


def apply_trans(trans, points):
    points = np.hstack((points, np.ones((len(points), 1))))
    points = np.dot(trans, points.T).T
    return points[:, :3]


def find_electrodes_hemis(user_fol, electrodes, groups=None, overwrite=False):
    output_fname = op.join(user_fol, 'ct', 'finding_electrodes_in_ct', 'hemis_model.pkl')
    if not op.isfile(output_fname) or overwrite:
        print('Loading pial vertices for finding electrodes hemis')
        verts = read_pial_verts(user_fol)
        vertices = np.concatenate((verts['rh'], verts['lh']))
        hemis = np.array([0] * len(verts['rh']) + [1] * len(verts['lh']))
        scaler = StandardScaler()
        vertices = scaler.fit_transform(vertices)
        electrodes = scaler.fit_transform(electrodes)
        clf = svm.SVC(kernel='linear')
        print('Fitting the vertices...')
        clf.fit(vertices, hemis)
        joblib.dump(clf, output_fname, compress=9)
    else:
        clf = joblib.load(output_fname)
    elctrodes_hemis = clf.predict(electrodes)
    hemis = ['rh' if elc_hemi == 0 else 'lh' for elc_hemi in elctrodes_hemis]
    if groups is not None:
        groups_hemis = [Counter([hemis[elc] for elc in group]).most_common()[0][0] for group in groups]
    else:
        groups_hemis = []
    return hemis, groups_hemis


def read_pial_verts(user_fol):
    verts = {}
    for hemi in ['rh', 'lh']:
        pial_npz_fname = op.join(user_fol, 'surf', '{}.{}.npz'.format(hemi, 'pial'))
        d = np.load(pial_npz_fname)
        verts[hemi] = d['verts']
    return verts


def find_group_between_pair(elc_ind1, elc_ind2, electrodes, error_radius=3, min_distance=2, names=[]):
    points_inside, cylinder = points_in_cylinder(
        electrodes[elc_ind1], electrodes[elc_ind2], electrodes, error_radius, return_cylinder=True)
    # if len(names) > 0:
    #     for p in points_inside:
    #         in_cube = point_in_cube(electrodes[elc_ind1], electrodes[elc_ind2], electrodes[p], error_radius)
    #         print(names[p], in_cube)
    sort_indices = np.argsort([np.linalg.norm(electrodes[p] - electrodes[elc_ind1]) for p in points_inside])
    points_inside = [points_inside[ind] for ind in sort_indices]
    elcs_inside = electrodes[points_inside]
    dists = calc_group_dists(elcs_inside)
    print([names[p] for p in points_inside])
    print(dists)
    points_inside, too_close_points = remove_too_close_points(
        electrodes, points_inside, cylinder, min_distance)
    group = points_inside.tolist() if not isinstance(points_inside, list) else points_inside.copy()
    return group, too_close_points, dists


def find_electrode_group(elc_ind, electrodes, elctrodes_hemis, groups=[], error_radius=3, min_elcs_for_lead=4, max_dist_between_electrodes=15,
                         min_distance=2):
    max_electrodes_inside = 0
    best_points_insides, best_group_too_close_points, best_group_dists, best_cylinder = [], [], [], None
    elcs_already_in_groups = set(flat_list_of_lists(groups))
    electrodes_list = list(set(range(len(electrodes))) - elcs_already_in_groups)
    for i in electrodes_list:
        for j in electrodes_list[i+1:]:
            if not point_in_cube(electrodes[i], electrodes[j], electrodes[elc_ind], error_radius):
                continue
            points_inside, cylinder = points_in_cylinder(
                electrodes[i], electrodes[j], electrodes, error_radius, return_cylinder=True)
            if elc_ind not in points_inside:
                continue
            if len(set(points_inside) & elcs_already_in_groups) > 0:
                continue
            if len(points_inside) < min_elcs_for_lead:
                continue
            same_hemi = all_items_equall([elctrodes_hemis[p] for p in points_inside])
            if not same_hemi:
                continue
            sort_indices = np.argsort([np.linalg.norm(electrodes[p] - electrodes[i]) for p in points_inside])
            points_inside = [points_inside[ind] for ind in sort_indices]
            elcs_inside = electrodes[points_inside]
            # elcs_inside = sorted(elcs_inside, key=lambda x: np.linalg.norm(x - electrodes[i]))
            dists = calc_group_dists(elcs_inside)
            if max(dists) > max_dist_between_electrodes:
                continue
            points_inside_before_removing_too_close_points = points_inside.copy()
            points_inside, too_close_points = remove_too_close_points(
                electrodes, points_inside, cylinder, min_distance)
            if len(points_inside) < min_elcs_for_lead:
                continue
            if len(points_inside) > max_electrodes_inside:
                max_electrodes_inside = len(points_inside)
                best_points_insides = points_inside
                bset_points_inside_before_removing_too_close_points = points_inside_before_removing_too_close_points.copy()
                best_group_too_close_points = too_close_points.copy()
                best_group_dists = calc_group_dists(electrodes[points_inside])
                best_cylinder = cylinder
    best_group = best_points_insides.tolist() if not isinstance(best_points_insides, list) else best_points_insides.copy()
    # For debug only
    remove_too_close_points(electrodes, bset_points_inside_before_removing_too_close_points, best_cylinder, min_distance)
    return best_group, best_group_too_close_points, best_group_dists


def point_in_cube(pt1, pt2, k, e=0):
    # return all([p_in_the_middle(pt1[k], pt2[k], r[k]) for k in range(3)])
    # faster:
    return p_in_the_middle(pt1[0], pt2[0], k[0], e) and p_in_the_middle(pt1[1], pt2[1], k[1], e) and \
           p_in_the_middle(pt1[2], pt2[2], k[2], e)


def p_in_the_middle(x, y, z, e=0):
    return x+e >= z >= y-e if x > y else x-e <= z <= y+e


def points_in_cylinder(pt1, pt2, points, radius_sq, return_cylinder=False, N=100):
    dist = np.linalg.norm(pt1 - pt2)
    elc_ori = (pt2 - pt1) / dist # norm(elc_ori)=1mm
    # elc_line = np.array([pt1 + elc_ori*t for t in np.linspace(0, dist, N)])
    elc_line = (pt1.reshape(3, 1) + elc_ori.reshape(3, 1) @ np.linspace(0, dist, N).reshape(1, N)).T
    dists = np.min(cdist(elc_line, points), 0)
    if return_cylinder:
        return np.where(dists <= radius_sq)[0], elc_line
    else:
        return np.where(dists <= radius_sq)[0]


def point_in_cylinder(pt1, pt2, r, q):
    vec = pt2 - pt1
    const = r * np.linalg.norm(vec)
    return np.dot(q - pt1, vec) >= 0 and np.dot(q - pt2, vec) <= 0 and np.linalg.norm(np.cross(q - pt1, vec)) <= const


def calc_group_dists(electrodes_group):
    return [np.linalg.norm(pt2 - pt1) for pt1, pt2 in zip(electrodes_group[:-1], electrodes_group[1:])]


def remove_too_close_points(electrodes, points_inside_cylinder, cylinder, min_distance):
    elecs_to_remove = []
    # Remove too close points
    elcs_inside = electrodes[points_inside_cylinder]
    dists = cdist(elcs_inside, elcs_inside)
    dists += np.eye(len(elcs_inside)) * min_distance * 2
    inds = np.where(dists < min_distance)
    while len(inds[0]) > 0:
        points_examined, points_to_remove = set(), []
        pairs = list(set([tuple(sorted([inds[0][k], inds[1][k]])) for k in range(len(inds[0]))]))
        print('remove_too_close_points: {}'.format(pairs))
        pairs_electrodes = [[elcs_inside[p[k]] for k in range(2)] for p in pairs]
        for pair_electrode, pair in zip(pairs_electrodes, pairs):
            if pair[0] in points_examined or pair[1] in points_examined:
                continue
            pair_dist_to_cylinder = np.min(cdist(np.array(pair_electrode), cylinder), axis=1)
            # print(pair, pair_dist_to_cylinder)
            ind = np.argmax(pair_dist_to_cylinder)
            points_to_remove.append(pair[ind])
            for k in range(2):
                points_examined.add(pair[k])
        elecs_to_remove += [points_inside_cylinder[p] for p in points_to_remove]
        if len(points_to_remove) > 0:
            points_inside_cylinder = np.delete(points_inside_cylinder, points_to_remove, axis=0)
        elcs_inside = electrodes[points_inside_cylinder]
        dists = cdist(elcs_inside, elcs_inside)
        dists += np.eye(len(elcs_inside)) * min_distance * 2
        inds = np.where(dists < min_distance)

    # rectangles = find_rectangles_in_group(electrodes, points_inside_cylinder, points_to_remove)
    # points_to_remove.extend(rectangles)
    # elecs_to_remove = np.array(points_inside_cylinder)[points_to_remove]
    # if len(points_to_remove) > 0:
    #     points_inside_cylinder = np.delete(points_inside_cylinder, points_to_remove, axis=0)
    elecs_to_remove = np.array(elecs_to_remove)
    return points_inside_cylinder, elecs_to_remove


def find_rectangles_in_group(electrodes, points_inside_cylinder, point_removed, ratio=0.7):
    points_inside_cylinder = [e for e in points_inside_cylinder if e not in
                              set([points_inside_cylinder[k] for k in point_removed])]
    elcs_inside = electrodes[points_inside_cylinder]
    points_to_remove = []
    dists = cdist(elcs_inside, elcs_inside)
    for ind in range(len(dists) - 2):
        if dists[ind, ind+2] < (dists[ind, ind+1] + dists[ind+1, ind+2]) * ratio:
            points_to_remove.append(ind+1)
    return points_to_remove


############# Utils ##############

def time_to_go(now, run, runs_num, runs_num_to_print=10, thread=-1):
    if run % runs_num_to_print == 0 and run != 0:
        time_took = time.time() - now
        more_time = time_took / run * (runs_num - run)
        if thread > 0:
            print('{}: {}/{}, {:.2f}s, {:.2f}s to go!'.format(thread, run, runs_num, time_took, more_time))
        else:
            print('{}/{}, {:.2f}s, {:.2f}s to go!'.format(run, runs_num, time_took, more_time))


def flat_list_of_lists(l):
    return sum(l, [])


def all_items_equall(arr):
    return all([x == arr[0] for x in arr])

################ tests ####################


def test1(ct_data, threshold):
    voxels = np.array([[90, 75, 106]])
    maxs = find_all_local_maxima(ct_data, voxels, threshold, find_nei_maxima=True)
    print(maxs)


def test2(ct_data, ct_header, brain, aseg, threshold, min_distance):
    ct_voxels = find_voxels_above_threshold(ct_data, threshold)
    ct_voxels = mask_voxels_outside_brain(ct_voxels, ct_header, brain, aseg)
    voxels = find_all_local_maxima(ct_data, ct_voxels, threshold, find_nei_maxima=True, max_iters=100)
    voxels = remove_neighbors_voexls(ct_data, voxels)
    print('asdf')


def test3(ct_data, threshold, ct_header, brain, aseg=None, user_fol=''):
    ct_voxels = find_voxels_above_threshold(ct_data, threshold)
    brain_header = brain.get_header()
    brain_mask = brain.get_data()
    ct_vox2ras, ras2t1_vox, vox2t1_ras_tkr = get_trans(ct_header, brain_header)
    ct_ras = apply_trans(ct_vox2ras, ct_voxels)
    t1_vox = np.rint(apply_trans(ras2t1_vox, ct_ras)).astype(int)
    t1_tkreg = apply_trans(vox2t1_ras_tkr, t1_vox)
    t1_voxels_outside_pial = get_t1_voxels_outside_pial(user_fol, brain_header, brain_mask, aseg, t1_vox, t1_tkreg)
    t1_voxels_outside_pial = set([tuple(v) for v in t1_voxels_outside_pial])
    print('sdf')


if __name__ == '__main__':
    from src.utils import utils
    import nibabel as nib
    import matplotlib.pyplot as plt
    subject = 'mg105'
    threshold_percentile = 99.9
    min_distance = 3

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
    # test1(ct_data, threshold)
    # test2(ct_data, ct.header, brain, aseg, threshold, min_distance)
    test3(ct_data, threshold, ct.header, brain, aseg, op.join(mmvt_dir, subject))