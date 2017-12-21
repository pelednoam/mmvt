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


def mask_voxels_outside_brain(voxels, ct_header, brain):
    brain_header = brain.get_header()
    brain_mask = brain.get_data()
    ct_vox2ras, ras2t1_vox, _ = get_trans(ct_header, brain_header)
    ct_ras = apply_trans(ct_vox2ras, voxels)
    t1_vox = np.rint(apply_trans(ras2t1_vox, ct_ras)).astype(int)
    voxels = np.array([v for v, t1_v in zip(voxels, t1_vox) if brain_mask[tuple(t1_v)] > 0])
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
    centroid_inds = []
    for ind, label in enumerate(labels):
        voxels = data[Y == label]
        centroids[ind] = voxels[np.argmax([ct_data[tuple(voxel)] for voxel in voxels])]
        centroid_inds.append(np.where(np.all(data == centroids[ind], axis=1))[0][0])
    return centroids, Y


def knn_clustering(data, n_components, output_fol='', threshold=0):
    if output_fol != '':
        output_fname = op.join(output_fol, 'kmeans_model_{}.pkl'.format(int(threshold)))
        if not op.isfile(output_fname):
            kmeans = KMeans(n_clusters=n_components, random_state=0).fit(data)
            print('Saving knn model to {}'.format(output_fname))
            joblib.dump(kmeans, output_fname, compress=9)
        else:
            kmeans = joblib.load(output_fname)
    Y = kmeans.predict(data)
    centroids = kmeans.cluster_centers_
    return centroids, Y


def gmm_clustering(data, n_components, covariance_type='full', output_fol='', threshold=0):
    if output_fol != '':
        output_fname = op.join(output_fol, 'gmm_model_{}.pkl'.format(int(threshold)))
        if not op.isfile(output_fname):
            gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=covariance_type)
            gmm.fit(data)
            print('Saving gmm model to {}'.format(output_fname))
            joblib.dump(gmm, output_fname, compress=9)
        else:
            gmm = joblib.load(output_fname)
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


def find_electrode_group(elc_ind, electrodes, groups=[], error_radius=3, min_elcs_for_lead=4, max_dist_between_electrodes=15,
                         min_distance=2):
    max_electrodes_inside = 0
    best_group = None
    elcs_already_in_groups = set(flat_list_of_lists(groups))
    electrodes_list = list(set(range(len(electrodes))) - elcs_already_in_groups)
    now = time.time()
    run = -1
    N = len(np.triu_indices(len(electrodes_list), 1)[0])
    for i in electrodes_list:
        for j in electrodes_list[i+1:]:
            run += 1
            time_to_go(now, run, N, runs_num_to_print=1000)
            points_inside, cylinder = point_in_cylinder(
                electrodes[i], electrodes[j], electrodes, error_radius, return_cylinder=True)
            # for p in points_inside:
            #     point_in_cylinder2(electrodes[i], electrodes[j], electrodes[p], error_radius)
            if elc_ind not in points_inside:
                continue
            if len(set(points_inside) & elcs_already_in_groups) > 0:
                continue
            if len(points_inside) < min_elcs_for_lead:
                continue
            elcs_inside = electrodes[points_inside]
            elcs_inside = sorted(elcs_inside, key=lambda x: np.linalg.norm(x - electrodes[i]))
            dists = calc_group_dists(elcs_inside)
            if max(dists) > max_dist_between_electrodes:  # max_dist_between_electrodes:
                continue
            points_inside = remove_too_close_points(electrodes, points_inside, cylinder, min_distance)
            if len(points_inside) < min_elcs_for_lead:
                continue
            if len(points_inside) > max_electrodes_inside:
                max_electrodes_inside = len(points_inside)
                best_group = points_inside.tolist()
    return best_group


def point_in_cylinder(pt1, pt2, points, radius_sq, return_cylinder=False, N=100):
    dist = np.linalg.norm(pt1 - pt2)
    elc_ori = (pt2 - pt1) / dist # norm(elc_ori)=1mm
    # elc_line = np.array([pt1 + elc_ori*t for t in np.linspace(0, dist, N)])
    elc_line = (pt1.reshape(3, 1) + elc_ori.reshape(3, 1) @ np.linspace(0, dist, N).reshape(1, N)).T
    dists = np.min(cdist(elc_line, points), 0)
    if return_cylinder:
        return np.where(dists <= radius_sq)[0], elc_line
    else:
        return np.where(dists <= radius_sq)[0]


def point_in_cylinder2(pt1, pt2, testpt, radius_sq):
    # Name: CylTest_CapsFirst
    # Orig: Greg James - gjames@NVIDIA.com
    # Lisc: Free code - no warranty & no money back.  Use it all you want
    #
    # This function tests if the 3D point 'testpt' lies within an arbitrarily
    # oriented cylinder. The cylinder is defined by an axis from 'pt1' to 'pt2',
    # the axis having a length squared of 'lengthsq' (pre-compute for each cylinder
    # to avoid repeated work!), and radius squared of 'radius_sq'.
    #    The function tests against the end caps first, which is cheap -> only
    # a single dot product to test against the parallel cylinder caps.  If the
    # point is within these, more work is done to find the distance of the point
    # from the cylinder axis.
    #    Fancy Math (TM) makes the whole test possible with only two dot-products
    # a subtract, and two multiplies.  For clarity, the 2nd mult is kept as a
    # divide.  It might be faster to change this to a mult by also passing in
    # 1/lengthsq and using that instead.
    #    Elminiate the first 3 subtracts by specifying the cylinder as a base
    # point on one end cap and a vector to the other end cap (pass in {dx,dy,dz}
    # instead of 'pt2' ).
    #
    # The dot product is constant along a plane perpendicular to a vector.
    # The magnitude of the cross product divided by one vector length is
    # constant along a cylinder surface defined by the other vector as axis.

    lengthsq = np.linalg.norm(pt2-pt1)
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    dz = pt2[2] - pt1[2]
    pdx = testpt[0] - pt1[0]
    pdy = testpt[1] - pt1[1]
    pdz = testpt[2] - pt1[2]

    # Dot the d and pd vectors to see if point lies behind the
    # cylinder cap at pt1.x, pt1.y, pt1.z
    dot = pdx * dx + pdy * dy + pdz * dz

    # If dot is less than zero the point is behind the pt1 cap.
    # If greater than the cylinder axis line segment length squared
    # then the point is outside the other end cap at pt2.
    if dot < 0.0 or dot > lengthsq:
        return False
    else:
        # Point lies within the parallel caps, so find
        # distance squared from point to line, using the fact that sin^2 + cos^2 = 1
        # the dot = cos() * |d||pd|, and cross*cross = sin^2 * |d|^2 * |pd|^2
        # Carefull: '*' means mult for scalars and dotproduct for vectors
        # In short, where dist is pt distance to cyl axis:
        # dist = sin( pd to d ) * |pd|
        # distsq = dsq = (1 - cos^2( pd to d)) * |pd|^2
        # dsq = ( 1 - (pd * d)^2 / (|pd|^2 * |d|^2) ) * |pd|^2
        # dsq = pd * pd - dot * dot / lengthsq
        #  where lengthsq is d*d or |d|^2 that is passed into this function

        # distance squared to the cylinder axis:
        dsq = abs(pdx*pdx + pdy*pdy + pdz*pdz - dot*dot/lengthsq)
        return dsq < radius_sq


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
    return points_inside_cylinder


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
