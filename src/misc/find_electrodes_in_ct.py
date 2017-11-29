import numpy as np
import nibabel as nib
import csv
import os
import os.path as op
import glob
import matplotlib.pyplot as plt
from collections import Counter
import traceback
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.externals import joblib
from sklearn import mixture

from src.utils import utils
from src.utils import trans_utils as tu
from src.utils import trig_utils as trig

links_dir = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(links_dir, 'subjects', 'SUBJECTS_DIR')
MMVT_DIR = utils.get_link_dir(links_dir, 'mmvt')


def gmm_optimization(data, n_components):
    from collections import defaultdict
    bics = defaultdict(list)
    cts = ['spherical', 'tied', 'diag', 'full']
    for n in range(1, n_components * 2):
        for covariance_type in cts:
            gmm = mixture.GaussianMixture(n_components=n, covariance_type=covariance_type)
            gmm.fit(data)
            bics[covariance_type].append(gmm.bic(data))
    mins_bic = [(min(bics[ct]), np.argmin(bics[ct])) for ct in cts]
    ct = np.argmin([t[0] for t in mins_bic])
    n_components = mins_bic[ct][1]
    gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cts[ct])
    gmm.fit(data)

    # plt.figure()
    # for covariance_type in ['spherical', 'tied', 'diag', 'full']:
    #     plt.plot(range(1, n_components * 2), bics[covariance_type], label=covariance_type)
    # plt.legend()


def clustering(data, ct_data, n_components, output_fol, iter_num=0, covariance_type='full'):
    gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=covariance_type)
    gmm.fit(data)
    Y = gmm.predict(data)
    centroids = np.zeros(gmm.means_.shape, dtype=np.int)
    labels = np.unique(Y)
    centroid_inds = []
    for ind, label in enumerate(labels):
        voxels = data[Y == label]
        centroids[ind] = voxels[np.argmax([ct_data[tuple(voxel)] for voxel in voxels])]
        centroid_inds.append(np.where(np.all(data == centroids[ind], axis=1))[0][0])
    colors = ['r' if ind in centroid_inds else '0.7' for ind in range(len(data))]
    utils.plot_3d_scatter(data, colors=colors, fname=op.join(output_fol, 'gmm_{}.png'.format(iter_num)))
    return centroids, Y


def mask_ct(voxels, ct_header, brain):
    brain_header = brain.get_header()
    brain_mask = brain.get_data()
    ct_vox2ras, ras2t1_vox, _ = get_trans(ct_header, brain_header)
    ct_ras = tu.apply_trans(ct_vox2ras, voxels)
    t1_vox = np.rint(tu.apply_trans(ras2t1_vox, ct_ras)).astype(int)
    voxels = np.array([v for v, t1_v in zip(voxels, t1_vox) if brain_mask[tuple(t1_v)] > 0])
    return voxels


def get_trans(ct_header, brain_header):
    ct_vox2ras = ct_header.get_vox2ras()
    ras2t1_vox = np.linalg.inv(brain_header.get_vox2ras())
    vox2t1_ras_tkr = brain_header.get_vox2ras_tkr()
    return ct_vox2ras, ras2t1_vox, vox2t1_ras_tkr


def get_electrodes_groups(electrodes, groups):
    return [[group_num for group_num, group in enumerate(groups) if elc_ind in group][0] for
                         elc_ind in range(len(electrodes))]


def get_electrodes_colors(electrodes, groups):
    electrodes_groups = get_electrodes_groups(electrodes, groups)
    groups_colors = dist_colors(len(groups))
    return [groups_colors[electrodes_groups[elc_ind]] for elc_ind in range(len(electrodes))]


def export_electrodes(subject, electrodes, groups, groups_hemis, output_fol):
    csv_fname = op.join(output_fol, '{}_RAS.csv'.format(subject))
    for fname in glob.glob(op.join(output_fol, '*.dat')):
        os.remove(fname)
    electrodes_colors = get_electrodes_colors(electrodes, groups)
    with open(csv_fname, 'w') as csv_file:
        wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        wr.writerow(['Electrode Name','R','A','S'])
        groups_inds = {'R':0, 'L':0}
        for group, group_hemi in zip(groups, groups_hemis):
            group_hemi = 'R' if group_hemi == 'rh' else 'L'
            group_name = '{}G{}'.format(group_hemi, chr(ord('A') + groups_inds[group_hemi]))
            elcs_names = ['{}{}'.format(group_name, k+1) for k in range(len(group))]
            for ind, elc_ind in enumerate(group):
                wr.writerow([elcs_names[ind], *['{:.2f}'.format(loc) for loc in electrodes[elc_ind]]])
            write_freeview_points(electrodes, group, group_name)
            ind0 = np.argmin([np.linalg.norm(elc) for elc in electrodes[group]])
            utils.plot_3d_scatter(electrodes, names=[elcs_names[0]], labels_indices=[group[ind0]], fname=op.join(
                output_fol, '{}.png'.format(group_name)), colors=electrodes_colors)
            groups_inds[group_hemi] += 1


def write_freeview_points(electrodes, group, group_name):
    fol = op.join(MMVT_DIR, subject, 'freeview')
    utils.make_dir(fol)
    group_electrodes = [electrodes[elc_ind] for elc_ind in group]
    with open(op.join(fol, '{}.dat'.format(group_name)), 'w') as fp:
        writer = csv.writer(fp, delimiter=' ')
        writer.writerows(group_electrodes)
        writer.writerow(['info'])
        writer.writerow(['numpoints', len(group_electrodes)])
        writer.writerow(['useRealRAS', '0'])


@utils.check_for_freesurfer
def run_freeview(subject):
    os.chdir(op.join(MMVT_DIR, subject, 'freeview'))
    electrodes_files = glob.glob(op.join(MMVT_DIR, subject, 'freeview', '*.dat'))
    utils.run_script('freeview -v T1.mgz:opacity=0.3 ct.mgz -c {}'.format(' '.join(electrodes_files)))


def find_electrodes_hemis(subject, electrodes, groups, overwrite=False):
    output_fname = op.join(MMVT_DIR, subject, 'electrodes', 'finding_electrodes_in_ct', 'hemis_model.pkl')
    if not op.isfile(output_fname) or overwrite:
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


def plot_hemis_sep_plane(clf, electrodes, vertices=[]):
    from mpl_toolkits.mplot3d import Axes3D

    d = clf.intercept_
    normal = clf.coef_
    sv = clf.support_vectors_

    # create x,y
    xx, yy = np.meshgrid(range(-70, 70), range(-10, 60))
    # calculate corresponding z
    zz = (-normal[:, 0] * xx - normal[:, 1] * yy - d) * 1. / normal[:, 2]

    # plot the surface
    # plt3d = plt.figure().gca(projection='3d')
    fig = plt.figure()
    ax = Axes3D(fig)
    if len(vertices) > 0:
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='0.7', s=1, alpha=0.1)
    ax.scatter(electrodes[:, 0], electrodes[:, 1], electrodes[:, 2])
    ax.plot_wireframe(xx, yy, zz, rcount=5, ccount=5, color='purple')
    plt.show()


def find_electrodes_groups(electrodes, output_fol, error_radius=3, min_elcs_for_lead=4, max_dist_between_electrodes=20):
    sub_groups = find_electrodes_sub_groups(
        electrodes, min_elcs_for_lead, max_dist_between_electrodes, error_radius)
    groups = join_electrodes_sub_groups(sub_groups)
    non_electrodes = [k for k in range(len(electrodes)) if all([k not in g for g in groups])]
    if len(non_electrodes) > 0:
        utils.plot_3d_scatter(electrodes, names=non_electrodes, labels_indices=non_electrodes,
                              fname=op.join(output_fol, 'non-electrodes.png'))
        return electrodes, electrodes[non_electrodes], groups
    plot_groups(electrodes, groups, output_fol)
    groups = sort_groups(electrodes, groups)
    for g_ind, g in enumerate(groups):
        plot_cylinder((electrodes[g[0]], electrodes[g[-1]]), error_radius, electrodes,
                      fname=op.join(output_fol, 'cylinder_{}.png'.format(g_ind)))
    return electrodes, [], groups


def plot_cylinder(base_points, R, points=[], alpha=0.5, cmap='seismic', plot_axis=False, fname=''):
    # https://stackoverflow.com/questions/32317247/how-to-draw-a-cylinder-using-matplotlib-along-length-of-point-x1-y1-and-x2-y2
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for p0, p1 in base_points:
        # vector in direction of axis
        v = p1 - p0
        # find magnitude of vector
        mag = np.linalg.norm(v)
        # unit vector in direction of axis
        v = v / mag
        # make some vector not in the same direction as v
        not_v = np.array([1, 0, 0])
        if (v == not_v).all():
            not_v = np.array([0, 1, 0])
        # make vector perpendicular to v
        n1 = np.cross(v, not_v)
        # normalize n1
        n1 /= np.linalg.norm(n1)
        # make unit vector perpendicular to v and n1
        n2 = np.cross(v, n1)
        # surface ranges over t from 0 to length of axis and 0 to 2*pi
        t = np.linspace(0, mag, 100)
        theta = np.linspace(0, 2 * np.pi, 100)
        # use meshgrid to make 2d arrays
        t, theta = np.meshgrid(t, theta)
        # generate coordinates for surface
        X, Y, Z = [p0[i] + v[i] * t + R * np.sin(theta) * n1[i] + R * np.cos(theta) * n2[i] for i in [0, 1, 2]]
        ax.plot_surface(X, Y, Z, alpha=alpha, cmap=cmap)
    # plot axis
    if plot_axis:
        ax.plot(*zip(p0, p1), color='red')
    if len(points) > 0:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    if fname == '':
        plt.show()
    else:
        plt.savefig(fname)


def find_electrodes_sub_groups(electrodes, min_elcs_for_lead, max_dist_between_electrodes, error_radius):
    groups = []
    for i in range(len(electrodes)):
        for j in range(i+1, len(electrodes)):
            points_inside = trig.point_in_cylinder(electrodes[i], electrodes[j], electrodes, error_radius)
            if len(points_inside) > min_elcs_for_lead:
                elcs_inside = electrodes[points_inside]
                elcs_inside = sorted(elcs_inside, key=lambda x: np.linalg.norm(x - electrodes[i]))
                dists = calc_group_dists(elcs_inside)
                if max(dists) > max_dist_between_electrodes:
                    continue
                groups.append(set(points_inside.tolist()))
    return groups


def join_electrodes_sub_groups(groups):
    final_groups = []
    for group in groups:
        for final_group in final_groups:
            if intersects(group, final_group):
                final_groups.remove(final_group)
                final_groups.append(group | final_group)
                break
        else:
            final_groups.append(group)
    return final_groups


def sort_groups(electrodes, groups):
    sorted_groups, first_indices = [], []
    for group in groups:
        group = list(group)
        # find the most inner electrode
        ind0 = np.argmin([np.linalg.norm(elc) for elc in electrodes[group]])
        first_indices.append(group[ind0])
        sort_indices = [t[0] for t in sorted(enumerate(electrodes[group]),
                                             key=lambda x: np.linalg.norm(electrodes[group[ind0]] - x[1]))]
        group = [group[ind] for ind in sort_indices]
        sorted_groups.append(group)
    return sorted_groups


def calc_group_dists(electrodes_group):
    return [np.linalg.norm(pt2 - pt1) for pt1, pt2 in zip(electrodes_group[:-1], electrodes_group[1:])]


def erase_voxels_from_ct(non_electrodes, ct_voxels, clusters, ct_data, ct_header, brain_header):
    if len(non_electrodes) == 0:
        return ct_data
    non_electrodes = t1_ras_tkr_to_ct_voxels(non_electrodes, ct_header, brain_header)
    non_electrodes_found = 0
    for ind, label in enumerate(np.unique(clusters)):
        voxels = ct_voxels[clusters == label]
        max_voxel = voxels[np.argmax([ct_data[tuple(voxel)] for voxel in voxels])]
        inds = np.where(np.all(non_electrodes == max_voxel, axis=1))[0]
        if len(inds) > 0:
            non_electrodes_found += 1
            print('Cleaning CT voxel: {}'.format(label))
            for voxel in voxels:
                ct_data[tuple(voxel)] = 0
    if non_electrodes_found < len(non_electrodes):
        print("erase_voxels_from_ct: Couldn't find all the non-electrodes!")
    return ct_data


def plot_groups(electrodes, final_groups, output_fol):
    electrodes_groups = get_electrodes_groups(electrodes, final_groups)
    groups_num = len(set(electrodes_groups))
    groups_colors = dist_colors(groups_num)
    try:
        electrodes_colors = [groups_colors[electrodes_groups[elc_ind]] for elc_ind in range(len(electrodes))]
        utils.plot_3d_scatter(electrodes, colors=electrodes_colors, fname=op.join(output_fol, 'groups.png'))
    except:
        print('error in plot_groups!')
        err = traceback.format_exc()
        utils.save((electrodes, final_groups, err), op.join(output_fol, 'plot_groups_log.pkl'))


def find_missing_electrodes(electrodes, groups, ct_data, ct_header, brain_header, threshold, output_fol,
                            med_dist_ratio=1.9, max_iter_num=10):
    electrodes_num = len(electrodes)
    found = True
    iter_num, electrodes_added = 0, 0
    while found and iter_num < max_iter_num:
        found = False
        groups_dists = [calc_group_dists(electrodes[group]) for group in groups]
        for group_ind, (group, group_dists) in enumerate(zip(groups, groups_dists)):
            missing_electrodes_indices = np.where(group_dists > np.median(group_dists) * med_dist_ratio)[0]
            if len(missing_electrodes_indices) > 0:
                print(group_dists)
                me_indices = [group[ms] for ms in missing_electrodes_indices]
                electrodes_colors = get_electrodes_colors(electrodes, groups)
                utils.plot_3d_scatter(electrodes, names=me_indices, labels_indices=me_indices, fname=op.join(
                    output_fol, 'missing_electrodes_{}.png'.format(group_ind)), colors=electrodes_colors)
                for mis_elc in missing_electrodes_indices:
                    new_electrode = (electrodes[group[mis_elc]] + electrodes[group[mis_elc + 1]]) / 2
                    new_electrode_ct_voxel = t1_ras_tkr_to_ct_voxels([new_electrode], ct_header, brain_header)[0]
                    new_electrode_ct_voxel = find_nearest_electrde_in_ct(ct_data, new_electrode_ct_voxel, max_iters=100)
                    if ct_data[new_electrode_ct_voxel] < threshold:
                        print('New electrode ct intensity ({}) < threshold!'.format(ct_data[new_electrode_ct_voxel]))
                    new_electrode = ct_voxels_to_t1_ras_tkr([new_electrode_ct_voxel], ct_header, brain_header)[0]
                    electrodes = np.vstack((electrodes, new_electrode))
                    groups[group_ind].insert(mis_elc + 1, electrodes_num)
                    # utils.plot_3d_scatter(electrodes, names=[electrodes_num], labels_indices=[electrodes_num])
                    # print(calc_group_dists(electrodes[groups[group_ind]]))
                    electrodes_num += 1
                    electrodes_added += 1
                found = True
                break
        iter_num += 1
    print('find_missing_electrodes: {} electrodes {} added'.format(
        electrodes_added, 'was' if electrodes_added == 1 else 'were'))
    return electrodes, groups


def find_extra_electrodes(electrodes, groups, output_fol, med_dist_ratio=0.3, max_iter_num=10):
    found = True
    iter_num, electrodes_removed = 0, 0
    while found and iter_num < max_iter_num:
        found = False
        groups_dists = [calc_group_dists(electrodes[group]) for group in groups]
        print(min(utils.flat_list_of_lists(groups_dists)))
        for group_ind, (group, group_dists) in enumerate(zip(groups, groups_dists)):
            extra_electrodes_indices = np.where(group_dists < np.median(group_dists) * med_dist_ratio)[0]
            if len(extra_electrodes_indices) > 0:
                print(group_dists)
                ext_indices = [group[ext] for ext in extra_electrodes_indices]
                electrodes_colors = get_electrodes_colors(electrodes, groups)
                utils.plot_3d_scatter(electrodes, names=ext_indices, labels_indices=ext_indices, fname=op.join(
                    output_fol, 'extra_electrodes_{}.png'.format(group_ind)), colors=electrodes_colors)
                ext_elc = extra_electrodes_indices[0]
                target_electrode = electrodes[group[ext_elc - 1]] if ext_elc > 0 else electrodes[group[ext_elc + 2]]
                dist1 = np.linalg.norm(electrodes[group[ext_elc]] - target_electrode)
                dist2 = np.linalg.norm(electrodes[group[ext_elc + 1]] - target_electrode)
                remove_group_ind = ext_elc if dist1 < dist2 else ext_elc + 1
                electrode_removed_ind = group[remove_group_ind]
                electrodes = np.delete(electrodes, (electrode_removed_ind), axis=0)
                del groups[group_ind][remove_group_ind]
                groups = fix_groups_after_deleting_electrode(groups, electrode_removed_ind)
                electrodes_removed += 1
                # utils.plot_3d_scatter(electrodes)
                found = True
                break
        iter_num += 1
    print('find_extra_electrodes: {} electrodes were deleted'.format(electrodes_removed))
    return electrodes, groups


def fix_groups_after_deleting_electrode(groups, item_removed_ind):
    for group_ind in range(len(groups)):
        groups[group_ind] = [g-1 if g > item_removed_ind else g for g in groups[group_ind]]
        # if groups[group_ind] > item_removed_ind:
        #     groups[group_ind] -= 1
    return groups


def find_nearest_electrde_in_ct(ct_data, voxel, max_iters=100):
    from itertools import product
    peak_found, iter_num = True, 0
    x, y, z = voxel
    while peak_found and iter_num < max_iters:
        max_ct_data = ct_data[x, y, z]
        max_diffs = (0, 0, 0)
        for dx, dy, dz in product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]):
            # print(x+dx, y+dy, z+dz, ct_data[x+dx, y+dy, z+dz], max_ct_data)
            if ct_data[x + dx, y + dy, z + dz] > max_ct_data:
                max_ct_data = ct_data[x + dx, y + dy, z + dz]
                max_diffs = (dx, dy, dz)
        peak_found = max_diffs == (0, 0, 0)
        if not peak_found:
            x, y, z = x + max_diffs[0], y + max_diffs[1], z + max_diffs[2]
            # print(max_ct_data, x, y, z)
        iter_num += 1
    if not peak_found:
        print('Peak was not found!')
    # print(iter_num, max_ct_data, x, y, z)
    return x, y, z


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
    centroids_ct_vox = np.rint(tu.apply_trans(np.linalg.inv(ct_vox2ras), centroids_ras)).astype(int)
    return centroids_ct_vox


def sanity_check(electrodes, ct_header, brain_header, ct_data, threshold, output_fol):
    ct_voxels = t1_ras_tkr_to_ct_voxels(electrodes, ct_header, brain_header)
    ct_values = np.array([ct_data[tuple(vox)] for vox in ct_voxels])
    if len(np.where(ct_values < threshold)[0]) > 0:
        print("There are electrodes with ct intensity < threshold!")
    plt.figure()
    plt.hist(ct_values)
    plt.savefig(op.join(output_fol, 'electrodes_final_intensity.png'))


def check_ct_voxels(ct_data, voxels):
    plt.hist([ct_data[tuple(vox)] for vox in voxels])
    plt.show()


def find_voxels_above_threshold(ct_data, threshold):
    return np.array(np.where(ct_data > threshold)).T


def load_object_and_export(subject, output_fol):
    electrodes, groups, groups_hemis = utils.load(op.join(output_fol, 'objects.pkl'))
    export_electrodes(subject, electrodes, groups, groups_hemis, output_fol)


def find_depth_electrodes_in_ct(
        ct_fname, brain_mask_fname, n_components, output_fol, threshold=2000, max_iters=5,cylinder_error_radius=3,
        min_elcs_for_lead=4, max_dist_between_electrodes=20, overwrite=False):
    ct = nib.load(ct_fname)
    ct_header = ct.get_header()
    ct_data = ct.get_data()
    brain = nib.load(brain_mask_fname)
    brain_header = brain.get_header()
    non_electrodes, iter_num = [None], 0

    while len(non_electrodes) > 0 and iter_num < max_iters:
        ct_voxels = find_voxels_above_threshold(ct_data, threshold)
        ct_voxels_in_brain = mask_ct(ct_voxels, ct_header, brain)
        ct_electrodes, clusters = clustering(ct_voxels_in_brain, ct_data, n_components, output_fol, iter_num)
        electrodes = ct_voxels_to_t1_ras_tkr(ct_electrodes, ct_header, brain_header)
        electrodes, non_electrodes, groups = find_electrodes_groups(
            electrodes, output_fol, cylinder_error_radius, min_elcs_for_lead, max_dist_between_electrodes)
        ct_data = erase_voxels_from_ct(non_electrodes, ct_voxels_in_brain, clusters, ct_data, ct_header, brain_header)
        iter_num += 1
    if iter_num == max_iters:
        print("The algorithm didn't converge, non electrodes couldn't be cleaned!")
    electrodes, groups = find_extra_electrodes(electrodes, groups, output_fol, med_dist_ratio=0.3, max_iter_num=10)
    electrodes, groups = find_missing_electrodes(
        electrodes, groups, ct_data, ct_header, brain_header, threshold, output_fol,
        med_dist_ratio=1.9, max_iter_num=10)
    sanity_check(electrodes, ct_header, brain_header, ct_data, threshold, output_fol)
    groups_hemis = find_electrodes_hemis(subject, electrodes, groups, overwrite)
    export_electrodes(subject, electrodes, groups, groups_hemis, output_fol)
    utils.save((subject, electrodes, groups, groups_hemis), op.join(output_fol, 'objects.pkl'))
    # run_freeview(subject)
    print('Finish!')


def load_objects_and_plot_hemis_sep(subject, input_fol):
    electrodes, groups, groups_hemis = utils.load(op.join(input_fol, 'objects.pkl'))
    model_fname = op.join(MMVT_DIR, subject, 'electrodes', 'finding_electrodes_in_ct', 'hemis_model.pkl')
    vertices_rh, _ = utils.read_pial(subject, MMVT_DIR, 'rh')
    vertices_lh, _ = utils.read_pial(subject, MMVT_DIR, 'lh')
    vertices = np.concatenate((vertices_rh, vertices_lh))
    clf = joblib.load(model_fname)
    plot_hemis_sep_plane(clf, electrodes, vertices)


if __name__ == '__main__':
    subject = 'nmr01183'
    ct_fname = op.join(MMVT_DIR, subject, 'freeview', 'ct.mgz')
    if not op.isfile(ct_fname):
        ct_fname = op.join(SUBJECTS_DIR, subject, 'mri', 'ct.mgz')
    if not op.isfile(ct_fname):
        raise Exception("Can't find ct.mgz!")

    brain_mask_fname = op.join(MMVT_DIR, subject, 'freeview', 'brain.mgz')
    if not op.isfile(brain_mask_fname):
        brain_mask_fname = op.join(SUBJECTS_DIR, subject, 'mri', 'brain.mgz')
    if not op.isfile(brain_mask_fname):
        raise Exception("Can't find brain.mgz!")

    output_fol = utils.make_dir(op.join(
        MMVT_DIR, subject, 'electrodes', 'finding_electrodes_in_ct', utils.rand_letters(5)))
    # find_depth_electrodes_in_ct(
    #     ct_fname, brain_mask_fname, n_components=52, output_fol=output_fol, threshold=2000, max_iters=5,
    #     cylinder_error_radius=3, min_elcs_for_lead=4, max_dist_between_electrodes=20, overwrite=False)

    input_fol = '/home/npeled/Documents/finding_electrodes_in_ct/a1cae'
    # load_object_and_export(subject, input_fol)
    load_objects_and_plot_hemis_sep(subject, input_fol)