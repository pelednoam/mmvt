import numpy as np
import os.path as op
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.externals import joblib
from sklearn import mixture
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import time


def find_voxels_above_threshold(ct_data, threshold):
    return np.array(np.where(ct_data > threshold)).T


def mask_voxels_outside_brain(voxels, ct_header, brain, aseg=None):
    brain_header = brain.get_header()
    brain_mask = brain.get_data()
    ct_vox2ras, ras2t1_vox, _ = get_trans(ct_header, brain_header)
    ct_ras = apply_trans(ct_vox2ras, voxels)
    t1_vox = np.rint(apply_trans(ras2t1_vox, ct_ras)).astype(int)
    if aseg is None:
        voxels = np.array([v for v, t1_v in zip(voxels, t1_vox) if brain_mask[tuple(t1_v)] > 0])
    else:
        voxels = np.array([v for v, t1_v in zip(voxels, t1_vox) if brain_mask[tuple(t1_v)] > 0 and
                           aseg[tuple(t1_v)] not in [0, 7, 8]])
    return voxels


def get_trans(ct_header, brain_header):
    ct_vox2ras = ct_header.get_vox2ras()
    ras2t1_vox = np.linalg.inv(brain_header.get_vox2ras())
    vox2t1_ras_tkr = brain_header.get_vox2ras_tkr()
    return ct_vox2ras, ras2t1_vox, vox2t1_ras_tkr


def apply_trans(trans, points):
    points = np.hstack((points, np.ones((len(points), 1))))
    points = np.dot(trans, points.T).T
    return points[:, :3]


def clustering(data, ct_data, n_components, clustering_method='knn', output_fol='', threshold=0, covariance_type='full'):
    if clustering_method == 'gmm':
        centroids, Y = gmm_clustering(data, n_components, covariance_type, threshold, output_fol)
    elif clustering_method == 'knn':
        centroids, Y = knn_clustering(data, n_components, output_fol, threshold)
    centroids = np.zeros(centroids.shape, dtype=np.int)
    labels = np.unique(Y)
    # centroid_inds = []
    for ind, label in enumerate(labels):
        voxels = data[Y == label]
        centroids[ind] = voxels[np.argmax([ct_data[tuple(voxel)] for voxel in voxels])]
        # centroid_inds.append(np.where(np.all(data == centroids[ind], axis=1))[0][0])
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


def ct_voxels_to_t1_ras_tkr(centroids, ct_header, brain_header):
    ct_vox2ras, ras2t1_vox, vox2t1_ras_tkr = get_trans(ct_header, brain_header)
    centroids_ras = apply_trans(ct_vox2ras, centroids)
    centroids_t1_vox = apply_trans(ras2t1_vox, centroids_ras)
    centroids_t1_ras_tkr = apply_trans(vox2t1_ras_tkr, centroids_t1_vox)
    return centroids_t1_ras_tkr


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


def find_electrode_group(elc_ind, electrodes, elctrodes_hemis, groups=[], error_radius=3, min_elcs_for_lead=4, max_dist_between_electrodes=15,
                         min_distance=2):
    max_electrodes_inside = 0
    best_group, best_group_too_close_points, best_group_dists = [], [], []
    elcs_already_in_groups = set(flat_list_of_lists(groups))
    electrodes_list = list(set(range(len(electrodes))) - elcs_already_in_groups)
    for i in electrodes_list:
        for j in electrodes_list[i+1:]:
            if not point_in_cube(electrodes[i], electrodes[j], electrodes[elc_ind]):
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
            points_inside, too_close_points = remove_too_close_points(
                electrodes, points_inside, cylinder, min_distance)
            if len(points_inside) < min_elcs_for_lead:
                continue
            if len(points_inside) > max_electrodes_inside:
                max_electrodes_inside = len(points_inside)
                best_group = points_inside.tolist() if not isinstance(points_inside, list) else points_inside.copy()
                best_group_too_close_points = too_close_points.copy()
                best_group_dists = calc_group_dists(electrodes[points_inside])
    return best_group, best_group_too_close_points, best_group_dists


def point_in_cube(pt1, pt2, r):
    # return all([p_in_the_middle(pt1[k], pt2[k], r[k]) for k in range(3)])
    # faster:
    return p_in_the_middle(pt1[0], pt2[0], r[0]) and p_in_the_middle(pt1[1], pt2[1], r[1]) and \
           p_in_the_middle(pt1[2], pt2[2], r[2])


def p_in_the_middle(x, y, z):
    return x >= z >= y if x > y else x <= z <= y


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
    # Remove too close points
    elcs_inside = electrodes[points_inside_cylinder]
    points_to_remove = []
    dists = cdist(elcs_inside, elcs_inside)
    dists += np.eye(len(elcs_inside)) * min_distance * 2
    inds = np.where(dists <= min_distance)
    if len(inds[0]) > 0:
        pairs = list(set([tuple(sorted([inds[0][k], inds[1][k]])) for k in range(len(inds[0]))]))
        pairs_electrodes = [[elcs_inside[p[k]] for k in range(2)] for p in pairs]
        for pair_electrode, pair in zip(pairs_electrodes, pairs):
            ind = np.argmin(np.min(cdist(np.array(pair_electrode), cylinder), axis=1))
            points_to_remove.append(pair[ind])
    if len(points_to_remove) > 0:
        points_inside_cylinder = np.delete(points_inside_cylinder, points_to_remove, axis=0)
    return points_inside_cylinder, points_to_remove


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
