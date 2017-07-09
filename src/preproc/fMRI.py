import os
import os.path as op
import mne
import mne.stats.cluster_level as mne_clusters
import nibabel as nib
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import shutil
import glob
import traceback
import subprocess

from src.utils import utils
from src.utils import freesurfer_utils as fu
from src.preproc import meg as meg
from src.utils import preproc_utils as pu
from src.utils import labels_utils as lu

try:
    from sklearn.neighbors import BallTree
except:
    print('No sklearn!')

try:
    from surfer import Brain
    from surfer import viz
    # from surfer import project_volume_data
    SURFER = True
except:
    SURFER = False
    print('no pysurfer!')


SUBJECTS_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()
SUBJECTS_MEG_DIR = utils.get_link_dir(utils.get_links_dir(), 'meg')
FMRI_DIR = utils.get_link_dir(utils.get_links_dir(), 'fMRI')

FSAVG_VERTS = 10242
FSAVG5_VERTS = 163842

_bbregister = 'bbregister --mov {fsl_input}.nii --bold --s {subject} --init-fsl --lta register.lta'
_mri_robust_register = 'mri_robust_register --mov {fsl_input}.nii --dst $SUBJECTS_DIR/colin27/mri/orig.mgz' +\
                       ' --lta register.lta --satit --vox2vox --cost mi --mapmov {subject}_reg_mi.mgz'


def get_hemi_data(subject, hemi, source, surf_name='pial', name=None, sign="abs", min=None, max=None):
    brain = Brain(subject, hemi, surf_name, curv=False, offscreen=True)
    print('Brain {} verts: {}'.format(hemi, brain.geo[hemi].coords.shape[0]))
    hemi = brain._check_hemi(hemi)
    # load data here
    scalar_data, name = brain._read_scalar_data(source, hemi, name=name)
    print('fMRI contrast map vertices: {}'.format(len(scalar_data)))
    min, max = brain._get_display_range(scalar_data, min, max, sign)
    if sign not in ["abs", "pos", "neg"]:
        raise ValueError("Overlay sign must be 'abs', 'pos', or 'neg'")
    surf = brain.geo[hemi]
    old = viz.OverlayData(scalar_data, surf, min, max, sign)
    return old, brain


def calc_fmri_min_max(subject, contrast, fmri_contrast_file_template, task='', norm_percs=(3, 97),
                      norm_by_percentile=True, symetric_colors=True, new_name=''):
    data = None
    for hemi in utils.HEMIS:
        if isinstance(fmri_contrast_file_template, dict):
            hemi_fname = fmri_contrast_file_template[hemi]
        elif isinstance(fmri_contrast_file_template, str):
            hemi_fname = fmri_contrast_file_template.format(hemi=hemi)
        else:
            raise Exception('Wrong type of template!')
        file_type = utils.file_type(hemi_fname)
        if file_type == 'npy':
            x = np.load(hemi_fname)
        else:
            fmri = nib.load(hemi_fname)
            x = fmri.get_data().ravel()
        verts, _ = utils.read_ply_file(op.join(MMVT_DIR, subject, 'surf', '{}.pial.ply'.format(hemi)))
        if x.shape[0] != verts.shape[0]:
            if x.shape[0] in [FSAVG5_VERTS, FSAVG_VERTS]:
                temp_barin = 'fsaverage5' if x.shape[0] == FSAVG5_VERTS else 'fsaverage'
                raise Exception(
                    "It seems that the fMRI contrast was made on {}, and not on the subject.\n".format(temp_barin) +
                    "You can run the fMRI preproc on the template barin, or morph the fMRI contrast map to the subject.")
            else:
                raise Exception("fMRI contrast map ({}) and the {} pial surface ({}) doesn't have the " +
                                "same vertices number!".format(len(x), hemi, verts.shape[0]))
        data = x if data is None else np.hstack((x, data))
    data_min, data_max = utils.calc_min_max(data, norm_percs=norm_percs, norm_by_percentile=norm_by_percentile)
    print('calc_fmri_min_max: min: {}, max: {}'.format(data_min, data_max))
    data_minmax = utils.get_max_abs(data_max, data_min)
    if symetric_colors and np.sign(data_max) != np.sign(data_min):
        data_max, data_min = data_minmax, -data_minmax
    # todo: the output_fname was changed, check where it's being used!
    new_name = new_name if new_name != '' else '{}{}'.format('{}_'.format(task) if task != '' else '', contrast)
    output_fname = op.join(MMVT_DIR, subject, 'fmri', '{}_minmax.pkl'.format(new_name))
    print('Saving {}'.format(output_fname))
    utils.make_dir(op.join(MMVT_DIR, subject, 'fmri'))
    utils.save((data_min, data_max), output_fname)


def save_fmri_hemi_data(subject, hemi, contrast_name, fmri_fname, task, output_fol=''):
    if not op.isfile(fmri_fname):
        print('No such file {}!'.format(fmri_fname))
        return
    morphed_fmri_fname = '{0}_morphed_to_{2}{1}'.format(*op.splitext(fmri_fname), subject)
    # if not op.isfile(morphed_fmri_fname):
    fmri = nib.load(fmri_fname)
    x = fmri.get_data().ravel()
    morph_from_subject = check_vertices_num(subject, hemi, x)
    org_subject_fname = ''
    if subject != morph_from_subject:
        # Save for data for the morph_from_subject
        org_subject_fname = 'fmri_{}_{}_{}_{}.npy'.format(subject, task, contrast_name, hemi)
        _save_fmri_hemi_data(morph_from_subject, hemi, x, contrast_name, task, output_fol, org_subject_fname)
        if not op.isfile(morphed_fmri_fname):
            fu.surf2surf(morph_from_subject, subject, hemi, fmri_fname, morphed_fmri_fname, cwd=None, print_only=False)
            fmri = nib.load(morphed_fmri_fname)
            x = fmri.get_data().ravel()
        else:
            fmri = nib.load(morphed_fmri_fname)
            x = fmri.get_data().ravel()

    subject_fname =  _save_fmri_hemi_data(subject, hemi, x, contrast_name, task, output_fol)
    org_subject_fname = op.join(MMVT_DIR, morph_from_subject, 'fmri', org_subject_fname)
    return subject_fname, org_subject_fname, morph_from_subject


def _save_fmri_hemi_data(subject, hemi, x, contrast_name, task, output_fol='', output_name=''):
    verts, faces = utils.read_pial_npz(subject, MMVT_DIR, hemi)
    if len(verts) != x.shape[0]:
        raise Exception('Wrong number of vertices!')
    if output_fol == '':
        output_fol = op.join(MMVT_DIR, subject, 'fmri')
    utils.make_dir(output_fol)
    if output_name == '':
        output_name = 'fmri_{}_{}_{}.npy'.format(task, contrast_name, hemi)
    output_name = op.join(output_fol, output_name)
    print('Saving {}'.format(output_name))
    np.save(output_name, x)
    return output_name


def init_clusters(subject, input_fname):
    contrast_per_hemi, verts_per_hemi = {}, {}
    for hemi in utils.HEMIS:
        fmri_fname = input_fname.format(hemi=hemi)
        if utils.file_type(input_fname) == 'npy':
            x = np.load(fmri_fname)
            contrast_per_hemi[hemi] = x #[:, 0]
        else:
            # try nibabel
            x = nib.load(fmri_fname)
            contrast_per_hemi[hemi] = x.get_data().ravel()
        pial_npz_fname = op.join(MMVT_DIR, subject, 'surf', '{}.pial.npz'.format(hemi))
        if not op.isfile(pial_npz_fname):
            print('No pial npz file (), creating one'.format(pial_npz_fname))
            verts, faces = utils.read_ply_file(op.join(MMVT_DIR, subject, 'surf', '{}.pial.ply'.format(hemi)))
            np.savez(pial_npz_fname[:-4], verts=verts, faces=faces)
        d = np.load(pial_npz_fname)
        verts_per_hemi[hemi] = d['verts']
    connectivity_fname = op.join(MMVT_DIR, subject, 'spatial_connectivity.pkl')
    if not op.isfile(connectivity_fname):
        from src.preproc import anatomy
        anatomy.create_spatial_connectivity(subject)
    connectivity_per_hemi = utils.load(connectivity_fname)
    return contrast_per_hemi, connectivity_per_hemi, verts_per_hemi


def find_clusters(subject, contrast_name, t_val, atlas, task, volume_name='', input_fol='', load_from_annotation=True, n_jobs=1):
    contrast_name = contrast_name if volume_name == '' else volume_name
    volume_name = volume_name if volume_name != '' else contrast_name
    if input_fol == '':
        input_fol = op.join(MMVT_DIR, subject, 'fmri')
    input_fname = op.join(input_fol, 'fmri_{}_{}_{}.npy'.format(task, contrast_name, '{hemi}'))
    contrast, connectivity, verts = init_clusters(subject, input_fname)
    clusters_labels = dict(threshold=t_val, values=[])
    for hemi in utils.HEMIS:
        clusters, _ = mne_clusters._find_clusters(contrast[hemi], t_val, connectivity=connectivity[hemi])
        # blobs_output_fname = op.join(input_fol, 'blobs_{}_{}.npy'.format(contrast_name, hemi))
        # print('Saving blobs: {}'.format(blobs_output_fname))
        # save_clusters_for_blender(clusters, contrast[hemi], blobs_output_fname)
        clusters_labels_hemi = find_clusters_overlapped_labeles(
            subject, clusters, contrast[hemi], atlas, hemi, verts[hemi], load_from_annotation, n_jobs)
        if clusters_labels_hemi is None:
            print("Can't find clusters in {}!".format(hemi))
        else:
            clusters_labels['values'].extend(clusters_labels_hemi)

    clusters_labels_output_fname = op.join(
        MMVT_DIR, subject, 'fmri', 'clusters_labels_{}_{}_{}.pkl'.format(task, volume_name, atlas))
    print('Saving clusters labels: {}'.format(clusters_labels_output_fname))
    utils.save(clusters_labels, clusters_labels_output_fname)


# def find_clusters_tval_hist(subject, contrast_name, output_fol, input_fol='', n_jobs=1):
#     contrast, connectivity, _ = init_clusters(subject, contrast_name, input_fol)
#     clusters = {}
#     tval_values = np.arange(2, 20, 0.1)
#     now = time.time()
#     for ind, tval in enumerate(tval_values):
#         try:
#             # utils.time_to_go(now, ind, len(tval_values), 5)
#             clusters[tval] = {}
#             for hemi in utils.HEMIS:
#                 clusters[tval][hemi], _ = mne_clusters._find_clusters(
#                     contrast[hemi], tval, connectivity=connectivity[hemi])
#             print('tval: {:.2f}, len rh: {}, lh: {}'.format(tval, max(map(len, clusters[tval]['rh'])),
#                                                         max(map(len, clusters[tval]['rh']))))
#         except:
#             print('error with tval {}'.format(tval))
#     utils.save(clusters, op.join(output_fol, 'clusters_tval_hist.pkl'))


def load_clusters_tval_hist(input_fol):
    from itertools import chain
    clusters = utils.load(op.join(input_fol, 'clusters_tval_hist.pkl'))
    res = []
    for t_val, clusters_tval in clusters.items():
        tval = float('{:.2f}'.format(t_val))
        max_size = max([max([len(c) for c in clusters_tval[hemi]]) for hemi in utils.HEMIS])
        avg_size = np.mean(list(chain.from_iterable(([[len(c) for c in clusters_tval[hemi]] for hemi in utils.HEMIS]))))
        clusters_num = sum(map(len, [clusters_tval[hemi] for hemi in utils.HEMIS]))
        res.append((tval, max_size, avg_size, clusters_num))
    res = sorted(res)
    # res = sorted([(t_val, max([len(c) for c in [c_tval[hemi] for hemi in utils.HEMIS]])) for t_val, c_tval in clusters.items()])
    # tvals = [float('{:.2f}'.format(t_val)) for t_val, c_tval in clusters.items()]
    max_sizes = [r[1] for r in res]
    avg_sizes = [r[2] for r in res]
    tvals = [float('{:.2f}'.format(r[0])) for r in res]
    clusters_num = [r[3] for r in res]
    fig, ax1 = plt.subplots()
    ax1.plot(tvals, max_sizes, 'b')
    ax1.set_ylabel('max size', color='b')
    ax2 = ax1.twinx()
    ax2.plot(tvals, clusters_num, 'r')
    # ax2.plot(tvals, avg_sizes, 'g')
    ax2.set_ylabel('#clusters', color='r')
    plt.show()
    print('sdfsd')


def save_clusters_for_blender(clusters, contrast, output_file):
    vertices_num = len(contrast)
    data = np.ones((vertices_num, 4)) * -1
    colors = utils.get_spaced_colors(len(clusters))
    for ind, (cluster, color) in enumerate(zip(clusters, colors)):
        x = contrast[cluster]
        cluster_max = max([abs(np.min(x)), abs(np.max(x))])
        cluster_data = np.ones((len(cluster), 1)) * cluster_max
        cluster_color = np.tile(color, (len(cluster), 1))
        data[cluster, :] = np.hstack((cluster_data, cluster_color))
    np.save(output_file, data)


def find_clusters_overlapped_labeles(subject, clusters, contrast, atlas, hemi, verts, load_from_annotation=True,
                                     n_jobs=1):
    cluster_labels = []
    annot_fname = op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot'.format(hemi, atlas))
    if load_from_annotation and op.isfile(annot_fname):
        labels = mne.read_labels_from_annot(subject, annot_fname=annot_fname, surf_name='pial')
    else:
        # todo: read only the labels from the current hemi
        labels = lu.read_labels_parallel(subject, SUBJECTS_DIR, atlas, hemi, n_jobs=n_jobs)
        labels = [l for l in labels if l.hemi == hemi]

    if len(labels) == 0:
        print('No labels!')
        return None
    for cluster in clusters:
        x = contrast[cluster]
        cluster_max = np.min(x) if abs(np.min(x)) > abs(np.max(x)) else np.max(x)
        inter_labels, inter_labels_tups = [], []
        for label in labels:
            overlapped_vertices = np.intersect1d(cluster, label.vertices)
            if len(overlapped_vertices) > 0:
                if 'unknown' not in label.name:
                    inter_labels_tups.append((len(overlapped_vertices), label.name))
                    # inter_labels.append(dict(name=label.name, num=len(overlapped_vertices)))
        inter_labels_tups = sorted(inter_labels_tups)[::-1]
        for inter_labels_tup in inter_labels_tups:
            inter_labels.append(dict(name=inter_labels_tup[1], num=inter_labels_tup[0]))
        if len(inter_labels) > 0:
            # max_inter = max([(il['num'], il['name']) for il in inter_labels])
            cluster_labels.append(dict(vertices=cluster, intersects=inter_labels, name=inter_labels[0]['name'],
                coordinates=verts[cluster], max=cluster_max, hemi=hemi, size=len(cluster)))
        else:
            print('No intersected labels!')
    return cluster_labels


def create_functional_rois(subject, contrast_name, clusters_labels_fname='', func_rois_folder=''):
    if clusters_labels_fname == '':
        clusters_labels = utils.load(op.join(
            MMVT_DIR, subject, 'fmri', 'clusters_labels_{}.npy'.format(contrast_name)))
    if func_rois_folder == '':
        func_rois_folder = op.join(SUBJECTS_DIR, subject, 'mmvt', 'fmri', 'functional_rois', '{}_labels'.format(contrast_name))
    utils.delete_folder_files(func_rois_folder)
    for cl in clusters_labels:
        cl_name = 'fmri_{}_{:.2f}'.format(cl['name'], cl['max'])
        new_label = mne.Label(cl['vertices'], cl['coordinates'], hemi=cl['hemi'], name=cl_name,
            filename=None, subject=subject, verbose=None)
        new_label.save(op.join(func_rois_folder, cl_name))


def show_fMRI_using_pysurfer(subject, input_file, hemi='both'):
    brain = Brain(subject, hemi, "pial", curv=False, offscreen=False)
    brain.toggle_toolbars(True)
    if hemi=='both':
        for hemi in ['rh', 'lh']:
            print('adding {}'.format(input_file.format(hemi=hemi)))
            brain.add_overlay(input_file.format(hemi=hemi), hemi=hemi)
    else:
        print('adding {}'.format(input_file.format(hemi=hemi)))
        brain.add_overlay(input_file.format(hemi=hemi), hemi=hemi)


def mri_convert_hemis(contrast_file_template, contrasts=None, existing_format='nii.gz'):
    for hemi in utils.HEMIS:
        if contrasts is None:
            contrasts = ['']
        for contrast in contrasts:
            if '{contrast}' in contrast_file_template:
                contrast_fname = contrast_file_template.format(hemi=hemi, contrast=contrast, format='{format}')
            else:
                contrast_fname = contrast_file_template.format(hemi=hemi, format='{format}')
            if not op.isfile(contrast_fname.format(format='mgz')):
                convert_fmri_file(contrast_fname, existing_format, 'mgz')


# def mri_convert(volume_fname, from_format='nii.gz', to_format='mgz'):
#     try:
#         print('convert {} to {}'.format(volume_fname.format(format=from_format), volume_fname.format(format=to_format)))
#         utils.run_script('mri_convert {} {}'.format(volume_fname.format(format=from_format),
#                                                     volume_fname.format(format=to_format)))
#     except:
#         print('Error running mri_convert!')


def convert_fmri_file(input_fname_template, from_format='nii.gz', to_format='mgz'):
    try:
        output_fname = input_fname_template.format(format=to_format)
        intput_fname = input_fname_template.format(format=from_format)
        output_files = glob.glob(output_fname)
        if len(output_files) == 0:
            inputs_files = glob.glob(intput_fname)
            if len(inputs_files) == 1:
                intput_fname = inputs_files[0]
                utils.run_script('mri_convert {} {}'.format(intput_fname, output_fname))
                return output_fname
            elif len(inputs_files) == 0:
                print('No imput file was found! {}'.format(intput_fname))
                return ''
            else:
                print('Too many input files were found! {}'.format(intput_fname))
                return ''
        else:
            return output_files[0]
    except:
        print('Error running mri_convert!')
        return ''


def calculate_subcorticals_surface_activity(subject, volume_file, subcortical_codes_file='', aseg_stats_file_name='',
        method='max', k_points=100, do_plot=False):
    x = nib.load(volume_file)
    x_data = x.get_data()

    if do_plot:
        fig = plt.figure()
        ax = Axes3D(fig)

    sig_subs = []
    if subcortical_codes_file != '':
        subcortical_codes = np.genfromtxt(subcortical_codes_file, dtype=str, delimiter=',')
        seg_labels = map(str, subcortical_codes[:, 0])
    elif aseg_stats_file_name != '':
        aseg_stats = np.genfromtxt(aseg_stats_file_name, dtype=str, delimiter=',', skip_header=1)
        seg_labels = map(str, aseg_stats[:, 0])
    else:
        raise Exception('No segmentation file!')
    # Find the segmentation file
    aseg_fname = op.join(SUBJECTS_DIR, subject, 'mri', 'aseg.mgz')
    aseg = nib.load(aseg_fname)
    aseg_hdr = aseg.get_header()
    out_folder = op.join(SUBJECTS_DIR, subject, 'subcortical_fmri_activity')
    if not op.isdir(out_folder):
        os.mkdir(out_folder)
    sub_cortical_generator = utils.sub_cortical_voxels_generator(aseg, seg_labels, 5, False, FREESURFER_HOME)
    for pts, seg_name, seg_id in sub_cortical_generator:
        print(seg_name)
        verts, _ = utils.read_ply_file(op.join(SUBJECTS_DIR, subject, 'subcortical', '{}.ply'.format(seg_name)))
        vals = np.array([x_data[i, j, k] for i, j, k in pts])
        is_sig = np.max(np.abs(vals)) >= 2
        print(seg_name, seg_id, np.mean(vals), is_sig)
        pts = utils.transform_voxels_to_RAS(aseg_hdr, pts)
        # plot_points(verts,pts)
        verts_vals = calc_vert_vals(verts, pts, vals, method=method, k_points=k_points)
        print('verts vals: {}+-{}'.format(verts_vals.mean(), verts_vals.std()))
        if sum(abs(verts_vals)>2) > 0:
            sig_subs.append(seg_name)
        verts_colors = utils.arr_to_colors_two_colors_maps(verts_vals, threshold=2)
        verts_data = np.hstack((np.reshape(verts_vals, (len(verts_vals), 1)), verts_colors))
        np.save(op.join(out_folder, seg_name), verts_data)
        if do_plot:
            plot_points(verts, colors=verts_colors, fig_name=seg_name, ax=ax)
        # print(pts)
    utils.rmtree(op.join(MMVT_DIR, subject, 'subcortical_fmri_activity'))
    shutil.copytree(out_folder, op.join(MMVT_DIR, subject, 'subcortical_fmri_activity'))
    if do_plot:
        plt.savefig('/home/noam/subjects/mri/mg78/subcortical_fmri_activity/figures/brain.jpg')
        plt.show()


def calculate_subcorticals_activity(subject, fmri_file_template, measures=['mean'], subcortical_codes_fname='', overwrite=False):
    fmri_file_template = op.join(FMRI_DIR, subject, fmri_file_template)
    volume_files = find_volume_files_from_template(fmri_file_template)
    if len(volume_files) == 1:
        volume_file = volume_files[0]
    elif len(volume_files) == 0:
        print("Can't find the volume file! {}".format(fmri_file_template))
        return False
    else:
        print('More than one file was found! {}'.format(fmri_file_template))
        return False

    x = nib.load(volume_file)
    x_data = x.get_data()

    if subcortical_codes_fname == '':
        subcortical_codes_fname = op.join(MMVT_DIR, 'sub_cortical_codes.txt')
    if not op.isfile(subcortical_codes_fname):
        print("Can't find the subcortical codes file! {}".format(subcortical_codes_fname))
    subcortical_codes = np.genfromtxt(subcortical_codes_fname, dtype=str, delimiter=',')
    seg_labels = list(map(str, subcortical_codes[:, 0]))

    # Find the segmentation file
    aseg_fname = op.join(SUBJECTS_DIR, subject, 'mri', 'aseg.mgz')
    aseg = nib.load(aseg_fname)
    out_folder = op.join(MMVT_DIR, subject, 'fmri')
    if not op.isdir(out_folder):
        os.mkdir(out_folder)
    if np.any(x_data.shape[:3] != aseg.shape):
        new_aseg_fname = op.join(FMRI_DIR, subject, 'aseg.mgz')
        if not op.isfile(new_aseg_fname):
            fu.vol2vol(subject, aseg_fname, volume_file, new_aseg_fname)
        aseg = nib.load(new_aseg_fname)

    out_fnames = []
    if isinstance(measures, str):
        measures = [measures]
    for measure in measures:
        labels_data, seg_names = [], []
        out_fname = op.join(out_folder, 'subcorticals_{}.npz'.format(measure))
        out_fnames.append(out_fname)
        if op.isfile(out_fname) and not overwrite:
            continue
        sub_cortical_generator = utils.sub_cortical_voxels_generator(aseg, seg_labels, use_grid=False)
        for pts, seg_name, seg_id in sub_cortical_generator:
            seg_names.append(seg_name)
            x = np.array([x_data[i, j, k] for i, j, k in pts])
            if measure == 'mean':
                labels_data.append(np.mean(x, 0))
            elif measure.startswith('pca'):
                import sklearn.decomposition as deco
                remove_cols = np.where(np.all(x == np.mean(x, 0), 0))[0]
                x = np.delete(x, remove_cols, 1)
                x = (x - np.mean(x, 0)) / np.std(x, 0)
                comps = 1 if '_' not in measure else int(measure.split('_')[1])
                pca = deco.PCA(comps)
                x = x.T
                x_r = pca.fit(x).transform(x)
                labels_data.append(x_r)
        labels_data = np.array(labels_data)
        np.savez(out_fname, data=labels_data, names=seg_names)
        print('Writing to {}, {}'.format(out_fname, labels_data.shape))
    return all([op.isfile(o) for o in out_fnames])


def calc_vert_vals(verts, pts, vals, method='max', k_points=100):
    ball_tree = BallTree(pts)
    dists, pts_inds = ball_tree.query(verts, k=k_points, return_distance=True)
    near_vals = vals[pts_inds]
    # sig_dists = dists[np.where(abs(near_vals)>2)]
    cover = len(np.unique(pts_inds.ravel()))/float(len(pts))
    print('{}% of the points are covered'.format(cover*100))
    if method=='dist':
        n_dists = 1/(dists**2)
        norm = 1/np.sum(n_dists, 1)
        norm = np.reshape(norm, (len(norm), 1))
        n_dists = norm * n_dists
        verts_vals = np.sum(near_vals * n_dists, 1)
    elif method=='max':
        verts_vals = near_vals[range(near_vals.shape[0]), np.argmax(abs(near_vals), 1)]
    return verts_vals


def plot_points(subject, verts, pts=None, colors=None, fig_name='', ax=None):
    if ax is None:
        fig = plt.figure()
        ax = Axes3D(fig)
    colors = 'tomato' if colors is None else colors
    # ax.plot(verts[:, 0], verts[:, 1], verts[:, 2], 'o', color=colors, label='verts')
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], s=20, c=colors, label='verts')
    if pts is not None:
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'o', color='blue', label='voxels')
        plt.legend()
    if ax is None:
        plt.savefig(op.join(MMVT_DIR, subject, 'fmri', '{}.jpg'.format(fig_name)))
        plt.close()


def project_on_surface(subject, volume_file, surf_output_fname,
                       target_subject=None, overwrite_surf_data=False, is_pet=False):
    if target_subject is None:
        target_subject = subject
    utils.make_dir(op.join(MMVT_DIR, subject, 'fmri'))
    for hemi in utils.HEMIS:
        print('project {} to {}'.format(volume_file, hemi))
        if not op.isfile(surf_output_fname.format(hemi=hemi)) or overwrite_surf_data:
            if not is_pet:
                surf_data = fu.project_volume_data(volume_file, hemi, subject_id=subject, surf="pial", smooth_fwhm=3,
                    target_subject=target_subject, output_fname=surf_output_fname.format(hemi=hemi))
            else:
                surf_data = fu.project_pet_volume_data(subject, volume_file, hemi, surf_output_fname.format(hemi=hemi))
            nans = np.sum(np.isnan(surf_data))
            if nans > 0:
                print('there are {} nans in {} surf data!'.format(nans, hemi))
        else:
            surf_data = np.squeeze(nib.load(surf_output_fname.format(hemi=hemi)).get_data())
        output_fname = op.join(MMVT_DIR, subject, 'fmri', op.basename(surf_output_fname.format(hemi=hemi)))
        if not op.isfile(output_fname) or overwrite_surf_data:
            print('Saving surf data in {}'.format(output_fname))
            np.save(output_fname, surf_data)


def load_images_file(image_fname):
    for hemi in ['rh', 'lh']:
        x = nib.load(image_fname.format(hemi=hemi))
        nans = np.sum(np.isnan(np.array(x.dataobj)))
        if nans > 0:
            print('there are {} nans in {} image!'.format(nans, hemi))


def mask_volume(volume, mask, masked_volume):
    vol_nib = nib.load(volume)
    vol_data = vol_nib.get_data()
    mask_nib = nib.load(mask)
    mask_data = mask_nib.get_data().astype(np.bool)
    vol_data[mask_data] = 0
    vol_nib.data = vol_data
    nib.save(vol_nib, masked_volume)


def load_and_show_npy(subject, npy_file, hemi):
    x = np.load(npy_file)
    brain = Brain(subject, hemi, "pial", curv=False, offscreen=False)
    brain.toggle_toolbars(True)
    brain.add_overlay(x[:, 0], hemi=hemi)


def copy_volume_to_blender(subject, volume_fname_template, contrast='', overwrite_volume_mgz=True):
    if op.isfile(volume_fname_template.format(format='mgh')) and \
            (not op.isfile(volume_fname_template.format(format='mgz')) or overwrite_volume_mgz):
        fu.mri_convert(volume_fname_template, 'mgh', 'mgz')
        format = 'mgz'
    else:
        # volume_files = glob.glob(op.join(volume_fname_template.replace('{format}', '*')))
        volume_files = find_volume_files_from_template(volume_fname_template.replace('{format}', '*'))
        if len(volume_files) == 0:
            print('No volume file! Should be in {}'.format(volume_fname_template.replace('{format}', '*')))
            return ''
        if len(volume_files) > 1:
            print('Too many volume files!')
            return ''
        else:
            format = utils.file_type(volume_files[0])
    volume_fname = volume_fname_template.format(format=format)
    blender_volume_fname = op.basename(volume_fname) if contrast=='' else '{}.{}'.format(contrast, format)
    utils.make_dir(op.join(MMVT_DIR, subject, 'freeview'))
    shutil.copyfile(volume_fname, op.join(MMVT_DIR, subject, 'freeview', blender_volume_fname))
    return volume_fname


def project_volume_to_surface_get_files(subject, remote_subject_dir, args):
    necessary_files = {'mri': ['orig.mgz'],
                       'surf': ['lh.pial', 'rh.pial', 'lh.thickness', 'rh.thickness']}
    return utils.prepare_subject_folder(
        necessary_files, subject, remote_subject_dir, SUBJECTS_DIR,
        args.sftp, args.sftp_username, args.sftp_domain, args.sftp_password,
        args.overwrite_fs_files, args.print_traceback, args.sftp_port)


# mri/orig.mgz
def project_volume_to_surface(subject, volume_fname_template, overwrite_surf_data=True,
                              target_subject='', remote_fmri_dir='', is_pet=False):
    if os.environ.get('FREESURFER_HOME', '') == '':
        raise Exception('Source freesurfer and rerun')
    if target_subject == '':
        target_subject = subject
    remote_fmri_dir = op.join(FMRI_DIR, subject) if remote_fmri_dir == '' else remote_fmri_dir
    full_input_fname_template = op.join(remote_fmri_dir, volume_fname_template)
    full_input_fname_template = full_input_fname_template.replace('{format}', '*')
    full_input_fname_template = full_input_fname_template.format(subject=subject)
    print('input_fname_template: {}'.format(full_input_fname_template))
    volume_fname = utils.look_for_one_file(full_input_fname_template, 'fMRI volume files', pick_the_first_one=False,
                                           search_func=find_volume_files_from_template)
    utils.make_dir(op.join(FMRI_DIR, subject))
    local_fname = op.join(FMRI_DIR, subject, utils.namesbase_with_ext(volume_fname))
    if not op.isfile(local_fname):
        shutil.copy(volume_fname, local_fname)
    volume_fname = local_fname

    # volume_fname = copy_volume_to_blender(subject, volume_fname_template, contrast, overwrite_volume)
    target_subject_prefix = '_{}'.format(target_subject) if subject != target_subject else ''
    surf_output_fname = op.join(utils.get_parent_fol(volume_fname), '{}{}_{}.mgz'.format(
        utils.namebase(volume_fname), target_subject_prefix, '{hemi}'))

    project_on_surface(subject, volume_fname, surf_output_fname,
                       target_subject, overwrite_surf_data=overwrite_surf_data, is_pet=is_pet)
    utils.make_dir(op.join(MMVT_DIR, subject, 'freeview'))
    shutil.copy(volume_fname, op.join(MMVT_DIR, subject, 'freeview', op.basename(volume_fname)))
    return utils.both_hemi_files_exist(surf_output_fname)

# fu.transform_mni_to_subject('colin27', data_fol, volume_fname, '{}_{}'.format(target_subject, volume_fname))
    # load_images_file(surf_output_fname)


def calc_meg_activity_for_functional_rois(subject, meg_subject, atlas, task, contrast_name, contrast, inverse_method):
    fname_format, fname_format_cond, events_id, event_digit = meg.get_fname_format(task)
    raw_cleaning_method = 'tsss' # 'nTSSS'
    files_includes_cond = True
    meg.init_globals(meg_subject, subject, fname_format, fname_format_cond, files_includes_cond, raw_cleaning_method, contrast_name,
        SUBJECTS_MEG_DIR, task, SUBJECTS_DIR, MMVT_DIR)
    root_fol = op.join(SUBJECTS_DIR, subject, 'mmvt', 'fmri', 'functional_rois')
    labels_fol = op.join(root_fol, '{}_labels'.format(contrast))
    labels_output_fname = op.join(root_fol, '{}_labels_data_{}'.format(contrast, '{hemi}'))
    # src = meg.create_smooth_src(subject)
    for hemi in ['rh', 'lh']:
        meg.calc_labels_avg_per_condition(atlas, hemi, 'pial', events_id, labels_from_annot=False,
            labels_fol=labels_fol, stcs=None, inverse_method=inverse_method,
            labels_output_fname_template=labels_output_fname)


def copy_volumes(subject, contrast_file_template, contrast, volume_fol, volume_name):
    contrast_format = 'mgz'
    volume_type = 'mni305'
    volume_file = contrast_file_template.format(contrast=contrast, hemi=volume_type, format='{format}')
    if not op.isfile(volume_file.format(format=contrast_format)):
        fu.mri_convert(volume_file, 'nii.gz', contrast_format)
    volume_fname = volume_file.format(format=contrast_format)
    subject_volume_fname = op.join(volume_fol, '{}_{}'.format(subject, volume_name))
    if not op.isfile(subject_volume_fname):
        volume_fol, volume_name = op.split(volume_fname)
        fu.transform_mni_to_subject(subject, volume_fol, volume_name, '{}_{}'.format(subject, volume_name))
    blender_volume_fname = op.join(MMVT_DIR, subject, 'freeview', '{}.{}'.format(contrast, contrast_format))
    if not op.isfile(blender_volume_fname):
        print('copy {} to {}'.format(subject_volume_fname, blender_volume_fname))
        shutil.copyfile(subject_volume_fname, blender_volume_fname)


def analyze_4d_data(subject, atlas, input_fname_template, measures=['mean'], template_brain='',
                          overwrite=False, remote_fmri_dir='', do_plot=False, do_plot_all_vertices=False,
                          excludes=('corpuscallosum', 'unknown'), input_format='nii.gz'):
    utils.make_dir(op.join(MMVT_DIR, subject, 'fmri'))
    morph_from_subject = subject if template_brain == '' else template_brain
    figures_dir = op.join(remote_fmri_dir, subject, 'figures')
    input_fname_template_file = find_4d_fmri_file(subject, input_fname_template, template_brain, remote_fmri_dir)
    for hemi in utils.HEMIS:
        fmri_fname = input_fname_template_file.format(hemi=hemi)
        fmri_fname = convert_fmri_file(fmri_fname, from_format=input_format)
        print('loading {} ({})'.format(fmri_fname), utils.file_modification_time(fmri_fname))
        x = nib.load(fmri_fname).get_data()
        morph_from_subject = check_vertices_num(subject, hemi, x, morph_from_subject)
        # print(max([max(label.vertices) for label in labels]))
        labels = []
        for em in measures:
            output_fname = op.join(MMVT_DIR, subject, 'fmri', 'labels_data_{}_{}_{}.npz'.format(atlas, em, hemi))
            if op.isfile(output_fname) and not overwrite:
                print('{} already exist'.format(output_fname))
                continue
            if len(labels) == 0:
                labels = lu.read_hemi_labels(morph_from_subject, SUBJECTS_DIR, atlas, hemi)
                if len(labels) == 0:
                    print('No {} {} labels were found!'.format(morph_from_subject, atlas))
                    return False
            labels_data, labels_names = lu.calc_time_series_per_label(
                x, labels, em, excludes, figures_dir, do_plot, do_plot_all_vertices)
            np.savez(output_fname, data=labels_data, names=labels_names)
            print('{} was saved'.format(output_fname))

    return np.all([utils.both_hemi_files_exist(op.join(MMVT_DIR, subject, 'fmri', 'labels_data_{}_{}_{}.npz'.format(
        atlas, em, '{hemi}'))) for em in measures])


def find_4d_fmri_file(subject, input_fname_template, template_brain='', remote_fmri_dir=''):
    remote_fmri_dir = op.join(FMRI_DIR, subject) if remote_fmri_dir == '' else remote_fmri_dir
    if input_fname_template == '':
        input_fname_template = '*{hemi}*'
    input_fname_template = input_fname_template.format(
        subject=subject, morph_to_subject=template_brain, hemi='{hemi}')
    full_input_fname_template = op.join(remote_fmri_dir, input_fname_template)

    input_fname_template_files = find_hemi_files_from_template(full_input_fname_template)
    if len(input_fname_template_files) > 1:
        print('More the one file was found! {}'.format(full_input_fname_template))
        print(input_fname_template_files)
        return ''
    elif len(input_fname_template_files) == 0:
        print("Can't find template files! {}".format(full_input_fname_template))
        print(subject, input_fname_template, template_brain, remote_fmri_dir)
        return ''
    return input_fname_template_files[0]


@utils.tryit(False)
def calc_labels_mean_freesurfer_get_files(
        args, remote_subject_dir, subject, atlas, input_fname_template, template_brain='', target_subject='',
        remote_fmri_dir=''):
    input_fname_template_file = find_4d_fmri_file(subject, input_fname_template, template_brain, remote_fmri_dir)
    fmri_fname = input_fname_template_file.format(hemi='rh')
    if not op.isfile(fmri_fname):
        target_subject = subject
    if target_subject == '':
        x = nib.load(fmri_fname)
        if x.shape[0] in [FSAVG5_VERTS, FSAVG_VERTS]:
            target_subject = 'fsaverage5' if x.shape[0] == FSAVG5_VERTS else 'fsaverage'
        else:
            target_subject = subject

    annot_template_fname = op.join(SUBJECTS_DIR, target_subject, 'label', '{}.{}.annot'.format('{hemi}', atlas))
    if not utils.both_hemi_files_exist(annot_template_fname):
        print('{} does not exist!'.format(annot_template_fname))
        necessary_files = {'label': ['lh.{}.annot'.format(atlas), 'rh.{}.annot'.format(atlas)]}
        utils.prepare_subject_folder(
            necessary_files, target_subject, remote_subject_dir, SUBJECTS_DIR,
            args.sftp, args.sftp_username, args.sftp_domain, args.sftp_password,
            args.overwrite_fs_files, args.print_traceback, args.sftp_port)
    return utils.both_hemi_files_exist(annot_template_fname)


@utils.tryit(False)
def calc_labels_mean_freesurfer(
        subject, atlas, input_fname_template, template_brain='', target_subject='',
        remote_fmri_dir='', overwrite=True, excludes=('corpuscallosum', 'unknown')):
    input_fname_template_file = find_4d_fmri_file(subject, input_fname_template, template_brain, remote_fmri_dir)
    fmri_fname = input_fname_template_file.format(hemi='rh')
    if not op.isfile(fmri_fname):
        print("Can't find fmri file ({}), abort!".format(fmri_fname))
        return False
    if target_subject == '':
        x = nib.load(fmri_fname)
        if x.shape[0] in [FSAVG5_VERTS, FSAVG_VERTS]:
            target_subject = 'fsaverage5' if x.shape[0] == FSAVG5_VERTS else 'fsaverage'
        else:
            target_subject = subject
    res_dir = op.join(FMRI_DIR, subject)
    utils.make_dir(res_dir)
    output_fname_hemi = op.join(MMVT_DIR, subject, 'fmri', 'labels_data_{}_mean_{}.npz'.format(atlas, '{hemi}'))
    for hemi in utils.HEMIS:
        output_fname = output_fname_hemi.format(hemi=hemi)
        if op.isfile(output_fname) and not overwrite:
            continue
        fmri_fname = input_fname_template_file.format(hemi=hemi)
        labels_data, labels_names = fu.calc_labels_avg(
            target_subject, hemi, atlas, fmri_fname, res_dir, SUBJECTS_DIR, overwrite)
        labels_names, labels_data = lu.remove_exclude_labels_and_data(labels_names, labels_data, excludes)
        np.savez(output_fname, data=labels_data, names=labels_names)
        print('{} was saved ({} labels)'.format(output_fname, len(labels_names)))
    return utils.both_hemi_files_exist(output_fname_hemi)


def load_labels_ts(subject, atlas, labels_order_fname, extract_measure='mean', excludes=('corpuscallosum', 'unknown'),
                   indices_to_remove_from_data=(0,4,113,117), backup_existing_files=True, pick_the_first_one=False):
    if isinstance(extract_measure, list):
        if len(extract_measure) == 1:
            extract_measure = extract_measure[0]
        elif len(extract_measure) == 0:
            print('extract_measure is an empty list!')
            return False
        elif len(extract_measure) > 1:
            print('The load_labels_ts can get only one extract_measure!')
            return False
    st_template = op.join(FMRI_DIR, subject, '*{}*.txt'.format(atlas))
    st_file = utils.look_for_one_file(st_template, 'st', pick_the_first_one)
    if st_file is None:
        return False
    labels_data = np.genfromtxt(st_file).T
    ret = save_labels_data(
        subject, atlas, labels_data, labels_order_fname, extract_measure, excludes,
        indices_to_remove_from_data, backup_existing_files)
    return ret


def save_labels_data(
        subject, atlas, labels_data, labels_order_fname, extract_measure='mean', excludes=('corpuscallosum', 'unknown'),
        indices_to_remove_from_data=(0, 4, 113, 117), backup_existing_files=True):
    if len(indices_to_remove_from_data) > 0:
        labels_data = np.delete(labels_data, indices_to_remove_from_data, 0)
    labels = utils.read_list_from_file(labels_order_fname)
    labels, indices = lu.remove_exclude_labels(labels, excludes)
    remove_indices = list(set(range(len(labels))) - set(indices))
    labels_data = np.delete(labels_data, remove_indices, 0)
    if len(labels) != labels_data.shape[0]:
        print('len(labels_order) ({}) != fmri_data.shape[0] {}!'.format(len(labels), labels_data.shape[0]))
        return False
    else:
        print('len(labels_order) = fmri_data.shape[0] = {}'.format(len(labels)))
    indices = lu.get_lh_rh_indices(labels)
    labels = np.array(labels)
    output_fname_hemi = op.join(MMVT_DIR, subject, 'fmri', 'labels_data_{}_{}_{}.npz'.format(
        atlas, extract_measure, '{hemi}'))
    for hemi in utils.HEMIS:
        output_fname = output_fname_hemi.format(hemi=hemi)
        if backup_existing_files and op.isfile(output_fname):
            backup_fname = utils.add_str_to_file_name(output_fname, '_backup')
            shutil.copy(output_fname, backup_fname.format(hemi=hemi))
        labels_data_hemi = labels_data[indices[hemi]]
        labels_names_hemi = labels[indices[hemi]]
        np.savez(output_fname, data=labels_data_hemi, names=labels_names_hemi)
        print('{} was saved ({} labels)'.format(output_fname, len(labels_names_hemi)))
    return utils.both_hemi_files_exist(output_fname_hemi)


def check_vertices_num(subject, hemi, x, morph_from_subject=''):
    if x.shape[0] == FSAVG_VERTS:
        morph_from_subject = 'fsaverage'
    elif x.shape[0] == FSAVG5_VERTS:
        morph_from_subject = 'fsaverage5'
    else:
        verts, faces = utils.read_pial_npz(subject, MMVT_DIR, hemi)
        if x.shape[0] == verts.shape[0]:
            morph_from_subject = subject
        else:
            if morph_from_subject != '':
                verts, faces = utils.read_pial_npz(morph_from_subject, MMVT_DIR, hemi)
                if x.shape[0] != verts.shape[0]:
                    raise Exception("Can't find the subject to morph from!")
            else:
                raise Exception("Can't find the subject to morph from!")
    return morph_from_subject


def calc_labels_minmax(subject, atlas, extract_modes):
    for em in extract_modes:
        min_max_output_fname = op.join(MMVT_DIR, subject, 'fmri', 'labels_data_{}_{}_minmax.npy'.format(atlas, em))
        template = op.join(MMVT_DIR, subject, 'fmri', op.basename('labels_data_{}_{}_{}.npz'.format(atlas, em, '{hemi}')))
        if utils.both_hemi_files_exist(template):
            labels_data = [np.load(template.format(hemi=hemi)) for hemi in utils.HEMIS]
            np.save(min_max_output_fname, [min([np.min(d['data']) for d in labels_data]),
                                           max([np.max(d['data']) for d in labels_data])])
        else:
            print("Can't find {}!".format(template))
    return np.all([op.isfile(op.join(MMVT_DIR, subject, 'fmri', 'labels_data_{}_{}_minmax.npy'.format(atlas, em)))
                   for em in extract_modes])


def save_dynamic_activity_map(subject, fmri_file_template='', template='fsaverage', format='mgz', overwrite=False):
    if fmri_file_template == '':
        fmri_file_template = '*{hemi}*{format}'
    input_fname_template = fmri_file_template.format(
        subject=subject, morph_to_subject=template, hemi='{hemi}', format=format)
    minmax_fname = op.join(MMVT_DIR, subject, 'fmri', 'activity_map_minmax.npy')
    data_min, data_max = [], []
    for hemi in utils.HEMIS:
        fol = op.join(MMVT_DIR, subject, 'fmri', 'activity_map_{}'.format(hemi))
        # Check if there is a morphed file
        fmri_fname = get_fmri_fname(subject, '*morphed*{}*{}*{}'.format(subject, hemi, format),
                                            raise_exception=False)
        if not op.isfile(fmri_fname):
            fmri_fname = get_fmri_fname(subject, input_fname_template.format(hemi=hemi))
        data = nib.load(fmri_fname).get_data().squeeze()
        T = data.shape[1]
        if not overwrite and len(glob.glob(op.join(fol, '*.npy'))) == T and op.isfile(minmax_fname):
            continue
        verts, faces = utils.read_pial_npz(subject, MMVT_DIR, hemi)
        file_verts_num, subject_verts_num = data.shape[0], verts.shape[0]
        if file_verts_num != subject_verts_num:
            if file_verts_num == FSAVG_VERTS:
                target_subject = 'fsaverage'
            elif file_verts_num == FSAVG5_VERTS:
                target_subject = 'fsaverage5'
            else:
                raise Exception('save_activity_map: wrong number of vertices!')
            if os.environ.get('FREESURFER_HOME', '') == '':
                raise Exception('Source freesurfer and rerun')
            sp = fmri_fname.split(hemi)
            sep = '.' if '.' in utils.namebase(fmri_fname) else '_'
            target_fname = '{}{}{}{}{}'.format(sp[0], 'morphed{}to{}{}'.format(sep, sep, subject), sep, hemi, sp[1])
            print('Morphing data from {} to {} -> {}'.format(target_subject, subject, target_fname))
            fu.surf2surf(target_subject, subject, hemi, fmri_fname, target_fname, cwd=None, print_only=False)
            if op.isfile(target_fname):
                fmri_fname = target_fname
            else:
                raise Exception('surf2surf: Target file was not created!')
            data = nib.load(fmri_fname).get_data().squeeze()
        assert (data.shape[0] == subject_verts_num)
        data_min.append(np.min(data))
        data_max.append(np.max(data))
        utils.delete_folder_files(fol)
        now = time.time()
        T = data.shape[1]
        for t in range(T):
            utils.time_to_go(now, t, T, runs_num_to_print=10)
            np.save(op.join(fol, 't{}'.format(t)), data[:, t])

    np.save(minmax_fname, [min(data_min), max(data_max)])
    return np.all([len(glob.glob(op.join(MMVT_DIR, subject, 'fmri', 'activity_map_{}'.format(hemi), '*.npy'))) == T
                   for hemi in utils.HEMIS])


def find_template_files(template_fname):
    def find_files(template_fname):
        return [f for f in glob.glob(template_fname) if op.isfile(f) and utils.file_type(f) in ['mgz', 'nii.gz', 'nii']]

    files = find_files(template_fname)
    if len(files) == 0:
        print('Adding * to the end of the template_fname')
        files = find_files('{}*'.format(template_fname))
    print('find_template_files: {}, template: {}'.format(files, template_fname))
    return files


def find_hemi_files_from_template(template_fname):
    try:
        return find_hemi_files(find_template_files(template_fname.replace('{hemi}', '*')))
    except:
        print('Error in find_hemi_files_from_template: {}'.format(template_fname))
        print(traceback.format_exc())
        return []


def find_hemi_files(files):
    if len(files) < 2:
        print('len(files) should be >= 2!')
        print(files)
        return []
    files = get_unique_files_into_mgz(files)
    hemis_files = []
    rh_files = [f for f in files if lu.get_hemi_from_name(utils.namebase(f)) == 'rh'] #  '_rh' in utils.namebase(f) or '.rh' in utils.namebase(f)]
    parent_fol = utils.get_parent_fol(rh_files[0])
    for rh_file in rh_files:
        lh_file = lu.change_hemi(utils.namebase(rh_file)) # rh_file.replace('_rh', '_lh').replace('.rh', '.lh')
        lh_file = op.join(parent_fol, '{}.{}'.format(lh_file, utils.file_type(rh_file)))
        if op.isfile(lh_file):
            hemis_files.append(rh_file.replace('rh', '{hemi}'))
    print('find_hemi_files return {}'.format(hemis_files))
    return hemis_files


def find_volume_files(files):
    # if convert_to_mgz:
    #     files = get_unique_files_into_mgz(files)
    return [f for f in files if ('_rh' not in utils.namebase(f) and '_lh' not in utils.namebase(f)) and
            ('.rh' not in utils.namebase(f) and '.lh' not in utils.namebase(f))]


def find_volume_files_from_template(template_fname):
    return find_volume_files(find_template_files(template_fname))


def get_fmri_fname(subject, fmri_file_template, no_files_were_found_func=None, raise_exception=True):
    fmri_fname = ''
    full_fmri_file_template = op.join(FMRI_DIR, subject, fmri_file_template)
    files = find_volume_files_from_template(full_fmri_file_template)
    files_num = len(set([utils.namebase(f) for f in files]))
    if files_num == 1:
        fmri_fname = files[0]
    elif files_num == 0:
        if no_files_were_found_func is None:
            if raise_exception:
                raise Exception("Can't find any file in {}!".format(fmri_file_template))
        else:
            return no_files_were_found_func()
    elif files_num > 1:
        if raise_exception:
            raise Exception("More than one file can be found in {}! {}".format(full_fmri_file_template, files))
    return fmri_fname


def clean_4d_data(subject, atlas, fmri_file_template, trg_subject='fsaverage5', fsd='rest',
                             fwhm=6, lfp=0.08, nskip=4, remote_fmri_dir='', overwrite=False, print_only=False):
    # fsd: functional subdirectory

    def no_files_were_found():
        print('Trying to find remote files in {}'.format(op.join(remote_fmri_dir, fsd, '001', fmri_file_template)))
        files = find_volume_files_from_template(op.join(remote_fmri_dir, fsd, '001', fmri_file_template)) + \
                find_volume_files_from_template(op.join(remote_fmri_dir, fmri_file_template))
        print('files: {}'.format(files))
        files_num = len(set([utils.namebase(f) for f in files]))
        if files_num == 1:
            fmri_fname = op.join(FMRI_DIR, subject, files[0].split(op.sep)[-1])
            utils.make_dir(op.join(FMRI_DIR, subject))
            shutil.copy(files[0], fmri_fname)
        else:
            print("Can't find any file in {}!".format(fmri_file_template))
            return ''
            # raise Exception("Can't find any file in {}!".format(fmri_file_template))


    def create_folders_tree(fmri_fname):
        # Fisrt it's needed to create the freesurfer folders tree for the preproc-sess
        fol = utils.make_dir(op.join(FMRI_DIR, subject, fsd, '001'))
        if not op.isfile(op.join(fol, 'f.nii.gz')):
            if utils.file_type(fmri_fname) == 'mgz':
                fmri_fname = fu.mgz_to_nii_gz(fmri_fname)
            shutil.copy(fmri_fname, op.join(fol, 'f.nii.gz'))
        if not op.isfile(op.join(FMRI_DIR, subject, 'subjectname')):
            with open(op.join(FMRI_DIR, subject, 'subjectname'), 'w') as sub_file:
                sub_file.write(subject)

    def create_analysis_info_file(fsd, trg_subject, tr, fwhm=6, lfp=0.08, nskip=4):
        rs = utils.partial_run_script(locals(), cwd=FMRI_DIR, print_only=print_only)
        for hemi in utils.HEMIS:
            rs('mkanalysis-sess -analysis {fsd}_{hemi} -notask -TR {tr} -surface {trg_subject} {hemi} -fsd {fsd}' +
               ' -per-run -nuisreg global.waveform.dat 1 -nuisreg wm.dat 1 -nuisreg vcsf.dat 1 -lpf {lfp} -mcextreg' +
               ' -fwhm {fwhm} -nskip {nskip} -stc up -force', hemi=hemi)

    def find_trg_subject(trg_subject):
        if not op.isdir(op.join(SUBJECTS_DIR, trg_subject)):
            if op.isdir(op.join(FREESURFER_HOME, 'subjects', trg_subject)):
                os.symlink(op.join(FREESURFER_HOME, 'subjects', trg_subject),
                           op.join(SUBJECTS_DIR, trg_subject))
            else:
                raise Exception("The target subject {} doesn't exist!".format(trg_subject))

    def copy_output_files():
        new_fname_template = op.join(FMRI_DIR, subject, '{}.sm{}.{}.{}.mgz'.format(
            fsd, int(fwhm), trg_subject, '{hemi}'))
        for hemi in utils.HEMIS:
            new_fname = new_fname_template.format(hemi=hemi)
            if not op.isfile(new_fname):
                res_fname = op.join(FMRI_DIR, subject, fsd, '{}_{}'.format(fsd, hemi), 'res', 'res-001.nii.gz')
                if op.isfile(res_fname):
                    fu.nii_gz_to_mgz(res_fname)
                    res_fname = utils.change_fname_extension(res_fname, 'mgz')
                    shutil.copy(res_fname, new_fname)
        return utils.both_hemi_files_exist(new_fname_template)

    def no_output(*args):
        return not op.isfile(op.join(FMRI_DIR, subject, fsd, *args))

    def run(cmd, *output_args, **kargs):
        if no_output(*output_args) or overwrite:
            rs(cmd, **kargs)
            if no_output(*output_args):
                raise Exception('{}\nNo output created in {}!!\n\n'.format(
                    cmd, op.join(FMRI_DIR, subject, fsd, *output_args)))

    if os.environ.get('FREESURFER_HOME', '') == '':
        raise Exception('Source freesurfer and rerun')
    trg_subject = subject if trg_subject == '' else trg_subject
    find_trg_subject(trg_subject)
    if fmri_file_template == '':
        fmri_file_template = '*'
    fmri_fname = get_fmri_fname(subject, fmri_file_template, no_files_were_found, raise_exception=False)
    if fmri_fname == '':
        return False
    output_files_exist = copy_output_files()
    if output_files_exist:
        return True
    create_folders_tree(fmri_fname)
    rs = utils.partial_run_script(locals(), cwd=FMRI_DIR, print_only=print_only)
    # if no_output('001', 'fmcpr.sm{}.mni305.2mm.nii.gz'.format(int(fwhm))):
    run('preproc-sess -surface {trg_subject} lhrh -s {subject} -fwhm {fwhm} -fsd {fsd} -mni305 -per-run',
        '001', 'fmcpr.sm{}.mni305.2mm.nii.gz'.format(int(fwhm)))
    run('plot-twf-sess -s {subject} -dat f.nii.gz -mc -fsd {fsd} && killall display', 'fmcpr.mcdat.png')
    run('plot-twf-sess -s {subject} -dat f.nii.gz -fsd {fsd} -meantwf && killall display', 'global.waveform.dat.png')

    # registration
    run('tkregister-sess -s {subject} -per-run -fsd {fsd} -bbr-sum > {subject}/{fsd}/reg_quality.txt',
        'reg_quality.txt')

    # Computes seeds (regressors) that can be used for functional connectivity analysis or for use as nuisance regressors.
    if no_output('001', 'wm.dat'):
        rs('fcseed-config -wm -overwrite -fcname wm.dat -fsd {fsd} -cfg {subject}/wm_{fsd}.cfg')
        run('fcseed-sess -s {subject} -cfg {subject}/wm_{fsd}.cfg', '001', 'wm.dat')
    if no_output('001', 'vcsf.dat'):
        rs('fcseed-config -vcsf -overwrite -fcname vcsf.dat -fsd {fsd} -mean -cfg {subject}/vcsf_{fsd}.cfg')
        run('fcseed-sess -s {subject} -cfg {subject}/vcsf_{fsd}.cfg', '001', 'vcsf.dat')

    tr = get_tr(subject, fmri_fname) / 1000 # To sec
    create_analysis_info_file(fsd, trg_subject, tr, fwhm, lfp, nskip)
    for hemi in utils.HEMIS:
        # computes the average signal intensity maps
        run('selxavg3-sess -s {subject} -a {fsd}_{hemi} -svres -no-con-ok',
            '{}_{}'.format(fsd, hemi), 'res', 'res-001.nii.gz', hemi=hemi)

    return copy_output_files()
    # for hemi in utils.HEMIS:
    #     # new_fname = utils.add_str_to_file_name(fmri_fname, '_{}'.format(hemi))
    #     new_fname = op.join(FMRI_DIR, subject, '{}.sm{}.{}.{}.mgz'.format(fsd, int(fwhm), trg_subject, hemi))
    #     if not op.isfile(new_fname):
    #         res_fname = op.join(FMRI_DIR, subject, fsd, '{}_{}'.format(fsd, hemi), 'res', 'res-001.nii.gz')
    #         fu.nii_gz_to_mgz(res_fname)
    #         res_fname = utils.change_fname_extension(res_fname, 'mgz')
    #         shutil.copy(res_fname, new_fname)



def get_tr(subject, fmri_fname):
    try:
        tr_fname = utils.add_str_to_file_name(fmri_fname, '_tr', 'pkl')
        if op.isfile(tr_fname):
            return utils.load(tr_fname)
        if utils.is_file_type(fmri_fname, 'nii.gz'):
            old_fmri_fname = fmri_fname
            fmri_fname = '{}mgz'.format(fmri_fname[:-len('nii.gz')])
            if not op.isfile(fmri_fname):
                fu.mri_convert(old_fmri_fname, fmri_fname)
        if utils.is_file_type(fmri_fname, 'mgz'):
            fmri_fname = op.join(FMRI_DIR, subject, fmri_fname)
            tr = fu.get_tr(fmri_fname)
            # print('fMRI fname: {}'.format(fmri_fname))
            print('tr: {}'.format(tr))
            utils.save(tr, tr_fname)
            return tr
        else:
            print('file format not supported!')
            return None
    except:
        print(traceback.format_exc())
        return None


def fmri_pipeline(subject, atlas, contrast_file_template, task='', contrast='', fsfast=True, t_val=2,
         fmri_files_fol='', load_labels_from_annotation=True, n_jobs=2):
    '''

    Parameters
    ----------
    subject: subject's name
    atlas: pacellation name
    contrast_file_template: template for the contrast file name. To get a full name the user should run:
          contrast_file_template.format(hemi=hemi, constrast=constrast, format=format)
    t_val: tval cutt off for finding clusters
    surface_name: Just for output name
    contrast_format: The contrast format (mgz, nii, nii.gz, ...)
    existing_format: The exsiting format (mgz, nii, nii.gz, ...)
    fmri_files_fol: The fmri files output folder
    load_labels_from_annotation: For finding the intersected labels, if True the function tries to read the labels from
        the annotation file, if False it tries to read the labels files.
    Returns
    -------

    '''
    fol = op.join(FMRI_DIR, args.task, subject)
    if not op.isdir(fol):
        raise Exception('You should first put the fMRI contrast files in {}'.format(fol))
    contrasts_files = {}
    if fsfast and op.isdir(op.join(fol, 'bold')):
        # todo: What to do with group-avg in fsfast?
        contrasts = set([utils.namebase(f) for f in glob.glob(op.join(fol, 'bold', '*'))])
        for contrast in contrasts:
            contrast_files = glob.glob(op.join(fol, 'bold', '*{}*'.format(contrast), 'sig.*'), recursive=True)
            contrasts_files[contrast] = dict(
                volume_files=find_volume_files(contrast_files),
                hemis_files=find_hemi_files(contrast_files))
    else:
        contrast = contrast if contrast != '' else contrast_file_template.replace('*', '').replace('?', '')
        contrasts_files[contrast] = dict(
            volume_files=find_volume_files_from_template(op.join(fol, contrast_file_template)),
            hemis_files=find_hemi_files_from_template(op.join(fol, contrast_file_template)))
        if not contrasts_files[contrast]['hemis_files']:
            raise Exception('No contrast maps projected to the hemispheres were found in {}'.format(
                op.join(fol, contrast_file_template)))

    utils.make_dir(op.join(MMVT_DIR, subject, 'freeview'))
    for contrast, contrast_dict in contrasts_files.items():
        volume_files, hemis_files_templates = contrast_dict['volume_files'], contrast_dict['hemis_files']
        for volume_file in volume_files:
            fu.mri_convert_to(volume_file, 'mgz')
            shutil.copyfile(volume_file, op.join(MMVT_DIR, subject, 'freeview', '{}.{}'.format(contrast, format)))
        hemis_files_templates = [t for t in hemis_files_templates if not t.endswith('_morphed_to_{}.mgz'.format(subject))]
        for hemis_files_teamplate in hemis_files_templates:
            new_hemis_fname, new_hemis_org_subject_fname = {}, {}
            for hemi in utils.HEMIS:
                new_hemi_fname = fu.mri_convert_to(hemis_files_teamplate.format(hemi=hemi), 'mgz')
                new_hemis_fname[hemi], new_hemis_org_subject_fname[hemi], morphed_from_subject = \
                    save_fmri_hemi_data(subject, hemi, contrast, new_hemi_fname, task, output_fol=fmri_files_fol)
            calc_fmri_min_max(
                subject, contrast, new_hemis_fname, task=task, norm_percs=args.norm_percs,
                norm_by_percentile=args.norm_by_percentile, symetric_colors=args.symetric_colors)
            if morphed_from_subject != subject:
                calc_fmri_min_max(
                    morphed_from_subject, contrast, new_hemis_org_subject_fname, task=task, norm_percs=args.norm_percs,
                    norm_by_percentile=args.norm_by_percentile, symetric_colors=args.symetric_colors)
        # todo: save clusters also for morphed_from_subject
        find_clusters(subject, contrast, t_val, atlas, task, '', fmri_files_fol, load_labels_from_annotation, n_jobs)
    # todo: check what to return
    return True


def fmri_pipeline_all(subject, atlas, task='*', contrast='*', filter_dic=None, new_name='',
                      norm_by_percentile=False, norm_percs=None, symetric_colors=True):

    def remove_dups(all_names):
        all_names = list(set(all_names))
        all_names = [t for t in all_names if not ('-and-' in t and all([tt in all_names for tt in t.split('-and-')]))]
        return '-and-'.join(sorted(all_names))

    def change_cluster_values_names(cluster, uid):
        for blob in cluster['values']:
            blob['name'] = '{}-{}'.format(uid, blob['name'])

    hemi_all_data = {}
    file_names = [utils.namebase(f) for f in glob.glob(
        op.join(MMVT_DIR, subject, 'fmri', 'fmri_{}_{}_rh.npy'.format(task, contrast)))]
    all_tasks = remove_dups([f.split('_')[1] for f in file_names])
    all_contrasts = remove_dups([f.split('_')[2] for f in file_names])
    new_name = new_name if new_name != '' else '{}_{}'.format(all_tasks, all_contrasts)
    for hemi in utils.HEMIS:
        hemi_fnames = glob.glob(op.join(MMVT_DIR, subject, 'fmri', 'fmri_{}_{}_{}.npy'.format(task, contrast, hemi)))
        hemi_all_data[hemi] = np.load(hemi_fnames[0])
        for hemi_fname in hemi_fnames[1:]:
            hemi_data = np.load(hemi_fname)
            hemi_all_data[hemi] = [x1 if abs(x1) > abs(x2) else x2 for x1,x2 in zip(hemi_data, hemi_all_data[hemi])]
        output_name = 'fmri_{}_{}.npy'.format(new_name, hemi)
        np.save(op.join(MMVT_DIR, subject, 'fmri', output_name), hemi_all_data[hemi])
    new_hemis_fname = op.join(MMVT_DIR, subject, 'fmri', 'fmri_{}_{}.npy'.format(new_name, '{hemi}'))
    calc_fmri_min_max(
        subject, all_contrasts, new_hemis_fname, task=all_tasks, norm_percs=norm_percs,
        norm_by_percentile=norm_by_percentile, symetric_colors=symetric_colors, new_name=new_name)
    all_clusters_fnames = glob.glob(op.join(MMVT_DIR, subject, 'fmri', 'clusters_labels_*_{}.pkl'.format(atlas)))
    all_clusters_fnames = [f for f in all_clusters_fnames if '-and-' not in utils.namebase(f)]
    all_clusters_uids = ['-'.join(n.split('_')[:2]) for n in
                         [utils.namebase(f)[len('clusters_labels_'):] for f in all_clusters_fnames]]
    all_clusters = utils.load(all_clusters_fnames[0])
    change_cluster_values_names(all_clusters, all_clusters_uids[0])
    all_clusters = filter_clusters(all_clusters, filter_dic)
    for cluster_fname, cluster_uid in zip(all_clusters_fnames[1:], all_clusters_uids[1:]):
        cluster = utils.load(cluster_fname)
        change_cluster_values_names(cluster, cluster_uid)
        cluster = filter_clusters(cluster, filter_dic)
        if all_clusters['threshold'] != cluster['threshold']:
            print("Not all the cluster have the same threshold, can't join them!")
            return False
        all_clusters['values'] += cluster['values']
    utils.save(all_clusters, op.join(MMVT_DIR, subject, 'fmri', 'clusters_labels_{}_{}.pkl'.format(
        new_name, atlas)))


def filter_clusters(clusters, filter_dic):
    if filter_dic is None:
        return clusters
    new_cluster = dict()
    new_cluster['threshold'] = clusters['threshold']
    new_cluster['values'] = []
    uid = '-'.join(clusters['values'][0]['name'].split('-')[:2])
    for cluster in clusters['values']:
        if uid not in filter_dic:
            continue
        # for roi in ['dACC', 'OFC', 'dmPFC', 'vlPFC']:
        #     if roi in cluster['name']:
        #         print(cluster['name'], '{0:.2f}'.format(cluster['max']))
        for val in filter_dic[uid]:
            _tval = '{0:.2f}'.format(cluster['max']) == '{0:.2f}'.format(val['tval'])
            _name = cluster['name'] == '{}-{}-{}'.format(uid, val['name'], val['hemi'])
            if _tval and _name:
                print('Cluster found! {}'.format(cluster['name']))
                if 'new_name' in val:
                    cluster['name'] = '{}-{}-{}'.format(uid, val['new_name'], val['hemi'])
                new_cluster['values'].append(cluster)
    return new_cluster


def get_unique_files_into_mgz(files):
    from collections import defaultdict
    contrast_files_dic = defaultdict(list)
    for contrast_file in files:
        ft = utils.file_type(contrast_file)
        contrast_files_dic[contrast_file[:-len(ft) - 1]].append(ft)
    for contrast_file, fts in contrast_files_dic.items():
        if 'mgz' not in fts and 'nii.gz' in fts:
            # fu.mri_convert_to('{}.{}'.format(contrast_file, fts[0]), 'mgz')
            fu.nii_gz_to_mgz('{}.nii.gz'.format(contrast_file))
    files = ['{}.mgz'.format(contrast_file) for contrast_file in contrast_files_dic.keys()]
    print('get_unique_files_into_mgz: {}'.format(files))
    return files


def misc(args):
    contrast_name = 'interference'
    contrasts = {'non-interference-v-base': '-a 1', 'interference-v-base': '-a 2',
                 'non-interference-v-interference': '-a 1 -c 2', 'task.avg-v-base': '-a 1 -a 2'}
    fol = op.join(FMRI_DIR, args.task, args.subject[0])
    contrast_file_template = op.join(fol, 'bold',
        '{contrast_name}.sm05.{hemi}'.format(contrast_name=contrast_name, hemi='{hemi}'), '{contrast}', 'sig.{format}')
    # contrast_file_template = op.join(fol, 'sig.{hemi}.{format}')


    contrast_name = 'group-avg'
    # main(subject, atlas, None, contrast_file_template, t_val=14, surface_name='pial', existing_format='mgh')
    # find_clusters_tval_hist(subject, contrast_name, fol, input_fol='', n_jobs=1)
    # load_clusters_tval_hist(fol)

    # contrast = 'non-interference-v-interference'
    inverse_method = 'dSPM'
    # meg_subject = 'ep001'

    # overwrite_volume_mgz = False
    # data_fol = op.join(FMRI_DIR, task, 'healthy_group')
    # contrast = 'pp003_vs_healthy'
    # contrast = 'pp009_ARC_High_Risk_Linear_Reward_contrast'
    # contrast = 'pp009_ARC_PPI_highrisk_L_VLPFC'

    # create_functional_rois(subject, contrast, data_fol)

    # # todo: find the TR automatiaclly
    # TR = 1.75

    # show_fMRI_using_pysurfer(subject, '/homes/5/npeled/space3/fMRI/ECR/hc004/bold/congruence.sm05.lh/congruent-v-incongruent/sig.mgz', 'rh')

    # fsfast.run(subject, root_dir=ROOT_DIR, par_file = 'msit.par', contrast_name=contrast_name, tr=TR, contrasts=contrasts, print_only=False)
    # fsfast.plot_contrast(subject, ROOT_DIR, contrast_name, contrasts, hemi='rh')
    # mri_convert_hemis(contrast_file_template, list(contrasts.keys())


    # show_fMRI_using_pysurfer(subject, input_file=contrast_file, hemi='lh')
    # root = op.join('/autofs/space/franklin_003/users/npeled/fMRI/MSIT/pp003')
    # volume_file = op.join(root, 'sig.anat.mgz')
    # mask_file = op.join(root, 'VLPFC.mask.mgz')
    # masked_file = op.join(root, 'sig.anat.masked.mgz')
    # contrast_file = op.join(root, 'sig.{hemi}.mgz')
    # contrast_masked_file = op.join(root, 'sig.masked.{hemi}.mgz')

    # for hemi in ['rh', 'lh']:
    #     save_fmri_colors(subject, hemi, contrast_masked_file.format(hemi=hemi), 'pial', threshold=2)
    # Show the fRMI in pysurfer
    # show_fMRI_using_pysurfer(subject, input_file=contrast_masked_file, hemi='both')

    # load_and_show_npy(subject, '/homes/5/npeled/space3/visualization_blender/mg79/fmri_lh.npy', 'lh')

    # mask_volume(volume_file, mask_file, masked_file)
    # show_fMRI_using_pysurfer(subject, input_file='/autofs/space/franklin_003/users/npeled/fMRI/MSIT/pp003/sig.{hemi}.masked.mgz', hemi='both')
    # calculate_subcorticals_activity(subject, '/homes/5/npeled/space3/MSIT/mg78/bold/interference.sm05.mni305/non-interference-v-interference/sig.anat.mgh',
    #              '/autofs/space/franklin_003/users/npeled/MSIT/mg78/aseg_stats.csv')
    # calculate_subcorticals_activity(subject, '/home/noam/fMRI/MSIT/mg78/bold/interference.sm05.mni305/non-interference-v-interference/sig.anat.mgh',
    #              '/home/noam/fMRI/MSIT/mg78/aseg_stats.csv')
    # volume_file = nib.load('/autofs/space/franklin_003/users/npeled/fMRI/MSIT/mg78/bold/interference.sm05.mni305/non-interference-v-interference/sig_subject.mgz')
    # vol_data, vol_header = volume_file.get_data(), volume_file.get_header()

    # contrast_file=contrast_file_template.format(
    #     contrast='non-interference-v-interference', hemi='mni305', format='mgz')
    # calculate_subcorticals_activity(subject, volume_file, subcortical_codes_file=op.join(BLENDER_DIR, 'sub_cortical_codes.txt'),
    #     method='dist')

    # SPM_ROOT = '/homes/5/npeled/space3/spm_subjects'
    # for subject_fol in utils.get_subfolders(SPM_ROOT):
    #     subject = utils.namebase(subject_fol)
    #     print(subject)
    #     contrast_masked_file = op.join(subject_fol, '{}_VLPFC_{}.mgz'.format(subject, '{hemi}'))
    #     show_fMRI_using_pysurfer(subject, input_file=contrast_masked_file, hemi='rh')
    # brain = Brain('fsaverage', 'both', "pial", curv=False, offscreen=False)


def main(subject, remote_subject_dir, args, flags):
    volume_name = args.volume_name if args.volume_name != '' else subject
    fol = op.join(FMRI_DIR, args.task, subject)
    remote_fmri_dir = op.join(FMRI_DIR, subject) if args.remote_fmri_dir == '' else \
        utils.build_remote_subject_dir(args.remote_fmri_dir, subject)
    print('remote_fmri_dir: {}'.format(remote_fmri_dir))
    if args.contrast_template == '':
        if args.fsfast:
            fmri_contrast_file_template = op.join(fol, 'bold', '{contrast_name}.sm05.{hemi}'.format(
                contrast_name=args.contrast_name, hemi='{hemi}'), '{contrast}', 'sig.{format}')
        else:
            fmri_contrast_file_template = op.join(fol, '{}_{}.mgz'.format(volume_name, '{hemi}'))
    else:
        fmri_contrast_file_template = args.contrast_template

    # todo: should find automatically the existing_format
    if 'fmri_pipeline' in args.function:
        flags['fmri_pipeline'] = fmri_pipeline(
            subject, args.atlas, fmri_contrast_file_template, args.task, args.contrast, args.fsfast,
            args.threshold, n_jobs=args.n_jobs)

    if utils.should_run(args, 'project_volume_to_surface'):
        ret = project_volume_to_surface_get_files(subject, remote_subject_dir, args)
        if not ret:
            flags['project_volume_to_surface'] = False
        else:
            flags['project_volume_to_surface'] = project_volume_to_surface(
                subject, args.fmri_file_template, args.overwrite_surf_data, args.target_subject, remote_fmri_dir,
                args.is_pet)

    if utils.should_run(args, 'calc_fmri_min_max'):
        #todo: won't work, need to find the hemis files first
        flags['calc_fmri_min_max'] = calc_fmri_min_max(
            subject, volume_name, fmri_contrast_file_template, task=args.task, norm_percs=args.norm_percs,
            norm_by_percentile=args.norm_by_percentile, symetric_colors=args.symetric_colors)

    if utils.should_run(args, 'find_clusters'):
        flags['find_clusters'] = find_clusters(subject, args.contrast, args.threshold, args.atlas, args.task, volume_name)

    if 'fmri_pipeline_all' in args.function:
        flags['fmri_pipeline_all'] = fmri_pipeline_all(subject, args.atlas, filter_dic=None)

    if 'analyze_4d_data' in args.function:
        flags['analyze_4d_data'] = analyze_4d_data(
            subject, args.atlas, args.fmri_file_template, args.labels_extract_mode, args.template_brain,
            args.overwrite_labels_data, remote_fmri_dir, args.resting_state_plot,
            args.resting_state_plot_all_vertices, args.excluded_labels, args.input_format)

    if 'calc_labels_minmax' in args.function:
        flags['calc_labels_minmax'] = calc_labels_minmax(subject, args.atlas, args.labels_extract_mode)

    if 'save_dynamic_activity_map' in args.function:
        flags['save_dynamic_activity_map'] = save_dynamic_activity_map(
            subject, args.fmri_file_template, template='fsaverage', format='mgz',
            overwrite=args.overwrite_activity_data)

    if 'clean_4d_data' in args.function:
        flags['clean_4d_data'] = clean_4d_data(
            subject, args.atlas, args.fmri_file_template, args.template_brain, args.fsd,
            args.fwhm, args.lfp, args.nskip, remote_fmri_dir, args.overwrite_4d_preproc, args.print_only)

    if 'calc_meg_activity' in args.function:
        meg_subject = args.meg_subject
        if meg_subject == '':
            print('You must set MEG subject (--meg_subject) to run calc_meg_activity function!')
        else:
            flags['calc_meg_activity'] = calc_meg_activity_for_functional_rois(
                subject, meg_subject, args.atlas, args.task, args.contrast_name, args.contrast, args.inverse_method)

    if 'calculate_subcorticals_activity' in args.function:
        flags['calculate_subcorticals_activity'] = calculate_subcorticals_activity(
            subject, args.fmri_file_template, measures=args.labels_extract_mode, overwrite=args.overwrite_subs_data)

    if 'copy_volumes' in args.function:
        flags['copy_volumes'] = copy_volumes(subject, fmri_contrast_file_template)

    if 'get_tr' in args.function:
        tr = get_tr(subject, args.fmri_fname)
        flags['get_tr'] = not tr is None

    if 'load_labels_ts' in args.function:
        flags['load_labels_ts'] = load_labels_ts(
            subject, args.atlas, args.labels_order_fname, args.labels_extract_mode, args.excluded_labels,
            args.labels_indices_to_remove_from_data, args.backup_existing_files, args.pick_the_first_one)

    if 'calc_labels_mean_freesurfer' in args.function:
        ret = calc_labels_mean_freesurfer_get_files(
            args, remote_subject_dir, subject, args.atlas, args.fmri_file_template, args.template_brain,
            args.target_subject, remote_fmri_dir)
        if not ret:
            print('Not all the necessary files exist!')
            flags['calc_labels_mean_freesurfer'] = False
        else:
            flags['calc_labels_mean_freesurfer'] = calc_labels_mean_freesurfer(
                subject, args.atlas, args.fmri_file_template, args.template_brain,
                args.target_subject, remote_fmri_dir, args.overwrite_labels_data, args.excluded_labels)
    return flags


def read_cmd_args(argv=None):
    import argparse
    from src.utils import args_utils as au

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-c', '--contrast', help='contrast map', required=False, default='')
    parser.add_argument('-n', '--contrast_name', help='contrast map', required=False, default='')
    parser.add_argument('-t', '--task', help='task', required=False, default='', type=au.str_arr_type)
    parser.add_argument('--threshold', help='clustering threshold', required=False, default=2, type=float)
    parser.add_argument('--fsfast', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--is_pet', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--contrast_template', help='', required=False, default='')
    parser.add_argument('--existing_format', help='existing format', required=False, default='mgz')
    parser.add_argument('--input_format', help='input format', required=False, default='nii.gz')
    parser.add_argument('--volume_type', help='volume type', required=False, default='mni305')
    parser.add_argument('--volume_name', help='volume file name', required=False, default='')
    parser.add_argument('--surface_name', help='surface_name', required=False, default='pial')
    parser.add_argument('--meg_subject', help='meg_subject', required=False, default='')
    parser.add_argument('--inverse_method', help='inverse method', required=False, default='dSPM')
    parser.add_argument('--remote_fmri_dir', help='remote fMRI folder', required=False, default='')

    parser.add_argument('--overwrite_surf_data', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_colors_file', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_volume', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_subs_data', help='', required=False, default=0, type=au.is_true)

    parser.add_argument('--norm_by_percentile', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--norm_percs', help='', required=False, default='1,99', type=au.int_arr_type)
    parser.add_argument('--symetric_colors', help='', required=False, default=1, type=au.is_true)

    # Resting state flags
    parser.add_argument('--fmri_file_template', help='', required=False, default='')
    parser.add_argument('--fsd', help='functional subdirectory', required=False, default='rest')
    parser.add_argument('--labels_extract_mode', help='', required=False, default='mean', type=au.str_arr_type)
    parser.add_argument('--morph_labels_from_subject', help='', required=False, default='fsaverage')
    parser.add_argument('--morph_labels_to_subject', help='', required=False, default='')
    parser.add_argument('--resting_state_plot', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--resting_state_plot_all_vertices', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--excluded_labels', help='', required=False, default='corpuscallosum,unknown', type=au.str_arr_type)
    parser.add_argument('--overwrite_labels_data', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_activity_data', help='', required=False, default=0, type=au.is_true)
    # parser.add_argument('--raw_fwhm', help='Raw Full Width at Half Maximum for Spatial Smoothing', required=False, default=5, type=float)
    parser.add_argument('--template_brain', help='', required=False, default='')
    parser.add_argument('--target_subject', help='', required=False, default='')
    # parser.add_argument('--fsd', help='functional subdirectory', required=False, default='rest')
    parser.add_argument('--fwhm', help='', required=False, default=6, type=float)
    parser.add_argument('--lfp', help='', required=False, default=0.08, type=float)
    parser.add_argument('--nskip', help='', required=False, default=4, type=int)
    parser.add_argument('--print_only', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_4d_preproc', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--backup_existing_files', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--pick_the_first_one', help='', required=False, default=0, type=au.is_true)

    # Misc flags
    parser.add_argument('--fmri_fname', help='', required=False, default='')
    parser.add_argument('--labels_order_fname', help='', required=False, default='')
    parser.add_argument('--labels_indices_to_remove_from_data', help='', required=False, default='', type=au.int_arr_type)
    pu.add_common_args(parser)
    args = utils.Bag(au.parse_parser(parser, argv))
    args.necessary_files = {'surf': ['lh.sphere.reg', 'rh.sphere.reg']}
    if 'clean_4d_data' in args.function or args.function == 'prepare_subject_folder':
        args.necessary_files = {'surf': ['rh.thickness', 'lh.thickness', 'rh.white', 'lh.white', 'lh.sphere.reg', 'rh.sphere.reg'],
                                'mri': ['brainmask.mgz', 'orig.mgz', 'aparc+aseg.mgz'],
                                'mri:transforms': ['talairach.xfm'],
                                'label': ['lh.cortex.label', 'rh.cortex.label']}
        # 'label': ['lh.cortex.label', 'rh.cortex.label']
    if args.is_pet:
        args.fsfast = False
    # print(args)
    for sub in args.subject:
        if '*' in sub:
            args.subject.remove(sub)
            args.subject.extend([fol.split(op.sep)[-1] for fol in glob.glob(op.join(FMRI_DIR, sub))])
    return args


if __name__ == '__main__':
    args = read_cmd_args()
    pu.run_on_subjects(args, main)
    print('finish!')
