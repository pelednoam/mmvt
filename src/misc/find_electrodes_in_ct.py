import numpy as np
import nibabel as nib
from sklearn.cluster import FeatureAgglomeration
from sklearn import mixture
import os.path as op
from src.utils import utils
from src.utils import trans_utils as tu
import csv
import matplotlib.pyplot as plt

links_dir = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(links_dir, 'subjects', 'SUBJECTS_DIR')
MMVT_DIR = utils.get_link_dir(links_dir, 'mmvt')


# ward = FeatureAgglomeration(n_clusters=1000, linkage='ward')
# ward.fit(ct_data)
# print(ward)


def find_nearest_electrde_in_ct(ct_data, x, y, z, max_iters=100):
    from itertools import product
    peak_found, iter_num = True, 0
    while peak_found and iter_num < max_iters:
        max_ct_data = ct_data[x, y, z]
        max_diffs = (0, 0, 0)
        for dx, dy, dz in product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]):
            print(x+dx, y+dy, z+dz, ct_data[x+dx, y+dy, z+dz], max_ct_data)
            if ct_data[x+dx, y+dy, z+dz] > max_ct_data:
                max_ct_data = ct_data[x+dx, y+dy, z+dz]
                max_diffs = (dx, dy, dz)
        peak_found = max_diffs == (0, 0, 0)
        if not peak_found:
            x, y, z = x+max_diffs[0], y+max_diffs[1], z+max_diffs[2]
            print(max_ct_data, x, y, z)
        iter_num += 1
    if not peak_found:
        print('Peak was not found!')
    print(iter_num, max_ct_data, x, y, z)


def write_freeview_points(centroids):
    fol = op.join(MMVT_DIR, subject, 'freeview')
    utils.make_dir(fol)

    with open(op.join(fol, 'electrodes.dat'), 'w') as fp:
        writer = csv.writer(fp, delimiter=' ')
        writer.writerows(centroids)
        writer.writerow(['info'])
        writer.writerow(['numpoints', len(centroids)])
        writer.writerow(['useRealRAS', '0'])


def clustering(data, ct_data, n_components, covariance_type='full'):
    from collections import Counter
    cv_types = ['spherical', 'tied', 'diag', 'full']
    gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=covariance_type)
    gmm.fit(data)
    # print(gmm)
    Y = gmm.predict(data)
    # utils.plot_3d_scatter(data, Y)
    # plot_3d(gmm, data, Y)
    centroids = np.zeros(gmm.means_.shape, dtype=np.int)
    for ind, label in enumerate(np.unique(Y)):
        voxels = data[Y==label]
        centroids[ind] = voxels[np.argmax([ct_data[tuple(voxel)] for voxel in voxels])]
    # utils.plot_3d_scatter(centroids, [len(data[Y==label]) for label in np.unique(Y)])
    return centroids


def plot_3d(clf, X, Y_):
    import itertools
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    fig, splot = plt.subplots(1,1)
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                                  'darkorange'])
    for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
                                               color_iter)):
        v, w = np.linalg.eigh(cov)
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan2(w[0][1], w[0][0])
        angle = 180. * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(.5)
        splot.add_artist(ell)
    plt.show()


def mask_ct(voxels, ct_header, brain):
    brain_header = brain.get_header()
    brain_mask = brain.get_data()
    ct_vox2ras, ras2t1_vox, _ = get_trans(ct_header, brain_header)
    ct_ras = tu.apply_trans(ct_vox2ras, voxels)
    t1_vox = tu.apply_trans(ras2t1_vox, ct_ras).astype(int)
    # voxels = voxels[np.where(brain_mask[t1_vox] > 0)]
    voxels = np.array([v for v, t1_v in zip(voxels, t1_vox) if brain_mask[tuple(t1_v)] > 0])
    return voxels


def get_trans(ct_header, brain_header):
    ct_vox2ras = ct_header.get_vox2ras()
    ras2t1_vox = np.linalg.inv(brain_header.get_vox2ras())
    vox2t1_ras_tkr = brain_header.get_vox2ras_tkr()
    return ct_vox2ras, ras2t1_vox, vox2t1_ras_tkr


@utils.check_for_freesurfer
def run_freeview():
    utils.run_script('freeview -v T1.mgz:opacity=0.3 ct.mgz brain.mgz -c electrodes.dat')


def export_electrodes(subject, electrodes, groups, groups_hemis):
    import csv
    fol = utils.make_dir(op.join(MMVT_DIR, subject, 'electrodes'))
    csv_fname = op.join(fol, '{}_RAS.csv'.format(subject))
    with open(csv_fname, 'w') as csv_file:
        wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        wr.writerow(['Electrode Name','R','A','S'])
        groups_inds = {'R':0, 'L':0}
        for group, group_hemi in zip(groups, groups_hemis):
            group_hemi = 'R' if group_hemi == 'rh' else 'L'
            group_name = 'G{}'.format(chr(ord('A') + groups_inds[group_hemi]))
            elcs_names = ['{}{}{}'.format(group_hemi, group_name, k+1) for k in range(len(group))]
            for ind, elc_ind in enumerate(group):
                wr.writerow([elcs_names[ind], *['{:.2f}'.format(loc) for loc in electrodes[elc_ind]]])
            # utils.plot_3d_scatter(electrodes, names=elcs_names, labels_indices=group)
            groups_inds[group_hemi] += 1


def left_or_right(subject, electrodes, groups):
    from sklearn.preprocessing import StandardScaler
    from sklearn import svm
    from sklearn.externals import joblib
    from collections import Counter
    utils.make_dir(op.join(MMVT_DIR, subject, 'electrodes'))
    output_fname = op.join(MMVT_DIR, subject, 'electrodes', 'electrodes_hemis_model.pkl')
    if not op.isfile(output_fname):
        vertices_rh, _ = utils.read_pial(subject, MMVT_DIR, 'rh')
        vertices_lh, _ = utils.read_pial(subject, MMVT_DIR, 'lh')
        vertices = np.concatenate((vertices_rh, vertices_lh))
        hemis = np.array([0] * len(vertices_rh) + [1] * len(vertices_lh))
        scaler = StandardScaler()
        vertices = scaler.fit_transform(vertices)
        electrodes = scaler.fit_transform(electrodes)
        clf = svm.SVC(kernel='linear')
        clf.fit(vertices, hemis)
        joblib.dump(clf, output_fname, compress=9)
    else:
        clf = joblib.load(output_fname)
    elctrodes_hemis = clf.predict(electrodes)
    hemis = ['rh' if elc_hemi == 0 else 'lh' for elc_hemi in elctrodes_hemis]
    groups_hemis = [Counter([hemis[elc] for elc in group]).most_common()[0][0] for group in groups]
    return groups_hemis


def find_electrodes_groups(electrodes, error_radius=3, min_elcs_for_lead=4, threshold_dist_between_electrodes=20):
    from src.utils import trig_utils as tu
    groups = []
    for i in range(len(electrodes)):
        for j in range(i+1, len(electrodes)):
            points_inside = tu.point_in_cylinder(electrodes[i], electrodes[j], electrodes, error_radius)
            if len(points_inside) > min_elcs_for_lead:
                elcs_inside = electrodes[points_inside]
                elcs_inside = sorted(elcs_inside, key=lambda x: np.linalg.norm(x - electrodes[i]))
                dists = calc_group_dists(elcs_inside) # [np.linalg.norm(pt2 - pt1) for pt1, pt2 in zip(elcs_inside[:-1], elcs_inside[1:])]
                if max(dists) > threshold_dist_between_electrodes:
                    continue
                # utils.plot_3d_scatter(electrodes, names=points_inside, labels_indices=points_inside)
                groups.append(set(points_inside.tolist()))

    final_groups = []
    for group in groups:
        for final_group in final_groups:
            if intersects(group, final_group):
                final_groups.remove(final_group)
                final_groups.append(group | final_group)
                break
        else:
            final_groups.append(group)


    non_electrodes = [k for k in range(len(electrodes)) if all([k not in g for g in final_groups])]
    # utils.plot_3d_scatter(electrodes, names=non_electrodes, labels_indices=non_electrodes)
    if len(non_electrodes) > 0:
        electrodes = np.delete(electrodes, non_electrodes, axis=0)
        return find_electrodes_groups(electrodes, error_radius, min_elcs_for_lead, threshold_dist_between_electrodes)
    else:
        plot_groups(electrodes, final_groups)
    # Sort the electrodes for each group
    groups, first_indices = [], []
    for group in final_groups:
        group = list(group)
        # find the most inner electrode
        ind0 = np.argmin([np.linalg.norm(elc) for elc in electrodes[group]])
        first_indices.append(group[ind0])
        sort_indices = [t[0] for t in sorted(enumerate(electrodes[group]), key=lambda x: np.linalg.norm(electrodes[group[ind0]] - x[1]))]
        group = [group[ind] for ind in sort_indices]
        groups.append(group)
    # utils.plot_3d_scatter(electrodes, names=first_indices, labels_indices=first_indices)
    # find_extra_missing_electrodes(electrodes, groups)
    return electrodes, groups


def calc_group_dists(electrodes_group):
    return [np.linalg.norm(pt2 - pt1) for pt1, pt2 in zip(electrodes_group[:-1], electrodes_group[1:])]


def plot_groups(electrodes, final_groups):
    electrodes_groups = [[group_num for group_num, group in enumerate(final_groups) if elc_ind in group][0] for
                         elc_ind in range(len(electrodes))]
    groups_num = len(set(electrodes_groups))
    groups_colors = dist_colors(groups_num)
    electrodes_colors = [groups_colors[electrodes_groups[elc_ind]] for elc_ind in range(len(electrodes))]
    utils.plot_3d_scatter(electrodes, colors=electrodes_colors)


def find_extra_missing_electrodes(electrodes, groups):
    groups_dists = [calc_group_dists(electrodes[group]) for group in groups]
    groups_dists_flat = utils.flat_list_of_lists(groups_dists)
    # dists_threshold = np.median(groups_dists_flat) * np.std(groups_dists_flat)
    for group, group_dists in zip(groups, groups_dists):
        missing_electrodes = np.where(group_dists > np.median(groups_dists_flat) * 1.9)[0]
        if len(missing_electrodes) > 0:
            print(group_dists)
            utils.plot_3d_scatter(electrodes, names=missing_electrodes, labels_indices=missing_electrodes)


def dist_colors(colors_num):
    import colorsys
    Hs = np.linspace(0, 360, colors_num + 1)[:-1] / 360
    Ls = 0.5
    return [colorsys.hls_to_rgb(Hs[ind], Ls, 1) for ind in range(colors_num)]


def intersects(g1, g2):
    return len(g1 & g2) > 0 or len(g1 & g2) > 0


def ct_voxels_to_t1_ras_tkr(centroids, ct_header, brain_header):
    ct_vox2ras, ras2t1_vox, vox2t1_ras_tkr = get_trans(ct_header, brain_header)
    centroids_ras = tu.apply_trans(ct_vox2ras, centroids)
    centroids_t1_vox = tu.apply_trans(ras2t1_vox, centroids_ras)
    centroids_t1_ras_tkr = tu.apply_trans(vox2t1_ras_tkr, centroids_t1_vox)
    return centroids_t1_ras_tkr


def t1_ras_tkr_to_ct_voxels(centroids, ct_header, brain_header):
    ct_vox2ras, ras2t1_vox, vox2t1_ras_tkr = get_trans(ct_header, brain_header)
    centroids_t1_vox = tu.apply_trans(np.linalg.inv(vox2t1_ras_tkr), centroids)
    centroids_ras = tu.apply_trans(brain_header.get_vox2ras(), centroids_t1_vox)
    centroids_ct_vox = tu.apply_trans(np.linalg.inv(ct_vox2ras), centroids_ras).astype(int)
    return centroids_ct_vox


def sanity_check(electrodes, ct_header, brain_header, ct_data, threshold=2000):
    ct_voxels = t1_ras_tkr_to_ct_voxels(electrodes, ct_header, brain_header)
    ct_values = [ct_data[tuple(vox)] for vox in ct_voxels]
    plt.hist(ct_values)
    plt.show()
    if len(np.where(ct_values < threshold)[0]) > 0:
        print('asdf')


def check_ct_voxels(ct_data, voxels):
    plt.hist([ct_data[tuple(vox)] for vox in voxels])
    plt.show()


def main(ct_fname, brain_mask_fname, n_components, threshold=2000):
    ct = nib.load(ct_fname)
    ct_header = ct.get_header()
    ct_data = ct.get_data()
    brain = nib.load(brain_mask_fname)
    brain_header = brain.get_header()

    voxels = np.where(ct_data > threshold)
    voxels = np.array(voxels).T
    voxels = mask_ct(voxels, ct_header, brain)
    # utils.plot_3d_scatter(voxels)
    electrodes = clustering(voxels, ct_data, n_components)
    electrodes = ct_voxels_to_t1_ras_tkr(electrodes, ct_header, brain_header)
    sanity_check(electrodes, ct_header, brain_header, ct_data, threshold=threshold)
    electrodes, groups = find_electrodes_groups(electrodes)
    groups_hemis = left_or_right(subject, electrodes, groups)
    write_freeview_points(electrodes)
    export_electrodes(subject, electrodes, groups, groups_hemis)
    # run_freeview()


if __name__ == '__main__':
    subject = 'nmr01183'
    ct_fname = '/home/npeled/mmvt/nmr01183/freeview/ct.mgz'
    brain_mask_fname = '/home/npeled/mmvt/nmr01183/freeview/brain.mgz'

    import os
    os.chdir(op.join(MMVT_DIR, subject, 'freeview'))
    main(ct_fname, brain_mask_fname, n_components=52)