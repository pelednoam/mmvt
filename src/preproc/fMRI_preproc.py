import platform
import sys

print(sys.version)
print(platform.python_version())
try:
    from surfer import Brain
    from surfer import viz
    # from surfer import project_volume_data
    SURFER = True
except:
    SURFER = False
    print('no pysurfer!')


import os
import os.path as op

import mne
import mne.stats.cluster_level as mne_clusters
import nibabel as nib

# from mne import spatial_tris_connectivity, grade_to_tris

import numpy as np
import time
# import pickle
# import math
# import glob
# import fsfast
try:
    from sklearn.neighbors import BallTree
except:
    print('No sklearn!')

import shutil

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.utils import utils
from src.utils import freesurfer_utils as fu
from src.preproc import meg_preproc as meg

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
SUBJECTS_MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
FREE_SURFER_HOME = utils.get_link_dir(LINKS_DIR, 'freesurfer', 'FREESURFER_HOME')
BLENDER_ROOT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')
FMRI_DIR = utils.get_link_dir(LINKS_DIR, 'fMRI')
os.environ['FREESURFER_HOME'] = FREE_SURFER_HOME
os.environ['SUBJECTS_DIR'] = SUBJECTS_DIR
# SUBJECTS_DIR = '/homes/5/npeled/space3/subjects'
# # SUBJECTS_DIR = '/autofs/space/lilli_001/users/DARPA-MEG/freesurfs'
# # SUBJECTS_DIR =  '/home/noam/subjects/mri'
# # SUBJECT = 'ep001'
# os.environ['SUBJECTS_DIR'] = SUBJECTS_DIR
# ROOT_DIR = [f for f in ['/homes/5/npeled/space3/fMRI/MSIT', '/home/noam/fMRI/MSIT'] if op.isdir(f)][0]
# BLENDER_DIR = '/homes/5/npeled/space3/visualization_blender'
# FREE_SURFER_HOME = utils.get_exisiting_dir([os.environ.get('FREESURFER_HOME', ''),
#     '/usr/local/freesurfer/stable5_3_0', '/home/noam/freesurfer'])


_bbregister = 'bbregister --mov {fsl_input}.nii --bold --s {subject} --init-fsl --lta register.lta'
_mri_robust_register = 'mri_robust_register --mov {fsl_input}.nii --dst $SUBJECTS_DIR/colin27/mri/orig.mgz' +\
                       ' --lta register.lta --satit --vox2vox --cost mi --mapmov {subject}_reg_mi.mgz'


conds = ['congruent-v-base', 'incongruent-v-base',  'congruent-v-incongruent', 'task.avg-v-base']
x, xfs = {}, {}
# show_fsaverage = False

# MRI_FILE_RH_FS = '/homes/5/npeled/space3/ECR_fsaverage/hc001/bold/congruence.sm05.rh/congruent-v-base/sig.nii.gz'
# MRI_FILE_LH_FS = '/homes/5/npeled/space3/ECR_fsaverage/hc001/bold/congruence.sm05.lh/congruent-v-base/sig.nii.gz'

# fMRI_FILE = '/homes/5/npeled/space3/ECR/hc001/bold/congruence.sm05.{}/congruent-v-base/sig_mg79.mgz'

# x = nib.load('/homes/5/npeled/Desktop/sig_fsaverage.nii.gz')
# xfs = nib.load('/homes/5/npeled/Desktop/sig_subject.nii.gz')


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


def save_fmri_colors(subject, hemi, contrast_name, fmri_file, surf_name='pial', threshold=2, output_fol=''):
    if not op.isfile(fmri_file.format(hemi)):
        print('No such file {}!'.format(fmri_file.format(hemi)))
        return
    fmri = nib.load(fmri_file.format(hemi))
    x = fmri.get_data().ravel()
    if output_fol == '':
        output_fol = op.join(BLENDER_ROOT_DIR, subject, 'fmri')
    utils.make_dir(output_fol)
    output_name = op.join(output_fol, 'fmri_{}_{}'.format(contrast_name, hemi))
    _save_fmri_colors(subject, hemi, x, threshold, output_name, surf_name=surf_name)


def _save_fmri_colors(subject, hemi, x, threshold, output_file='', verts=None, surf_name='pial'):
    if verts is None:
        # Try to read the hemi ply file to check if the vertices number is correct    
        ply_file = op.join(SUBJECTS_DIR, subject, 'surf', '{}.{}.ply'.format(hemi, surf_name))
        if op.isfile(ply_file):
            verts, _ = utils.read_ply_file(ply_file)
            if len(x) != verts.shape[0]:
                raise Exception("fMRI contrast map and the hemi doens't have the same vertices number!")
        else:
            print("No ply file, Can't check the vertices number")

    colors = utils.arr_to_colors_two_colors_maps(x, cm_big='YlOrRd', cm_small='PuBu',
        threshold=threshold, default_val=1)
    colors = np.hstack((x.reshape((len(x), 1)), colors))
    if output_file != '':
        op.join(BLENDER_ROOT_DIR, subject, 'fmri_{}.npy'.format(hemi))
    print('Saving {}'.format(output_file))
    np.save(output_file, colors)


def init_clusters(subject, contrast_name, input_fol):
    input_fname = op.join(input_fol, 'fmri_{}_{}.npy'.format(contrast_name, '{hemi}'))
    contrast_per_hemi, verts_per_hemi = {}, {}
    for hemi in utils.HEMIS:
        fmri_fname = input_fname.format(hemi=hemi)
        if utils.file_type(input_fname) == 'npy':
            x = np.load(fmri_fname)
            contrast_per_hemi[hemi] = x[:, 0]
        else:
            # try nibabel
            x = nib.load(fmri_fname)
            contrast_per_hemi[hemi] = x.get_data().ravel()
        pial_npz_fname = op.join(BLENDER_ROOT_DIR, subject, '{}.pial.npz'.format(hemi))
        if not op.isfile(pial_npz_fname):
            print('No pial npz file (), creating one'.format(pial_npz_fname))
            verts, faces = utils.read_ply_file(op.join(BLENDER_ROOT_DIR, subject, '{}.pial.ply'.format(hemi)))
            np.savez(pial_npz_fname[:-4], verts=verts, faces=faces)
        d = np.load(pial_npz_fname)
        verts_per_hemi[hemi] = d['verts']
    connectivity_fname = op.join(BLENDER_ROOT_DIR, subject, 'spatial_connectivity.pkl')
    if not op.isfile(connectivity_fname):
        from src.preproc import anatomy_preproc
        anatomy_preproc.create_spatial_connectivity(subject)
    connectivity_per_hemi = utils.load(connectivity_fname)
    return contrast_per_hemi, connectivity_per_hemi, verts_per_hemi


def find_clusters(subject, contrast_name, t_val, atlas, input_fol='', load_from_annotation=False, n_jobs=1):
    if input_fol == '':
        input_fol = op.join(BLENDER_ROOT_DIR, subject, 'fmri')
    contrast, connectivity, verts = init_clusters(subject, contrast_name, input_fol)
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
    # todo: should be pkl, not npy
    clusters_labels_output_fname = op.join(
        BLENDER_ROOT_DIR, subject, 'fmri', 'clusters_labels_{}.npy'.format(contrast_name))
    print('Saving clusters labels: {}'.format(clusters_labels_output_fname))
    utils.save(clusters_labels, clusters_labels_output_fname)


def find_clusters_tval_hist(subject, contrast_name, output_fol, input_fol='', n_jobs=1):
    contrast, connectivity, _ = init_clusters(subject, contrast_name, input_fol)
    clusters = {}
    tval_values = np.arange(2, 20, 0.1)
    now = time.time()
    for ind, tval in enumerate(tval_values):
        try:
            # utils.time_to_go(now, ind, len(tval_values), 5)
            clusters[tval] = {}
            for hemi in utils.HEMIS:
                clusters[tval][hemi], _ = mne_clusters._find_clusters(
                    contrast[hemi], tval, connectivity=connectivity[hemi])
            print('tval: {:.2f}, len rh: {}, lh: {}'.format(tval, max(map(len, clusters[tval]['rh'])),
                                                        max(map(len, clusters[tval]['rh']))))
        except:
            print('error with tval {}'.format(tval))
    utils.save(clusters, op.join(output_fol, 'clusters_tval_hist.pkl'))


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


def find_clusters_overlapped_labeles(subject, clusters, contrast, atlas, hemi, verts, load_from_annotation=False,
                                     n_jobs=1):
    cluster_labels = []
    annot_fname = op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot'.format(hemi, atlas))
    if load_from_annotation and op.isfile(annot_fname):
        labels = mne.read_labels_from_annot(subject, annot_fname=annot_fname, surf_name='pial')
    else:
        # todo: read only the labels from the current hemi
        labels = utils.read_labels_parallel(subject, SUBJECTS_DIR, atlas, n_jobs)
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
            BLENDER_ROOT_DIR, subject, 'fmri', 'clusters_labels_{}.npy'.format(contrast_name)))
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
                mri_convert(contrast_fname, existing_format, 'mgz')


def mri_convert(volume_fname, from_format='nii.gz', to_format='mgz'):
    try:
        print('convert {} to {}'.format(volume_fname.format(format=from_format), volume_fname.format(format=to_format)))
        utils.run_script('mri_convert {} {}'.format(volume_fname.format(format=from_format),
                                                    volume_fname.format(format=to_format)))
    except:
        print('Error running mri_convert!')


def calculate_subcorticals_activity(subject, volume_file, subcortical_codes_file='', aseg_stats_file_name='',
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
    sub_cortical_generator = utils.sub_cortical_voxels_generator(aseg, seg_labels, 5, False, FREE_SURFER_HOME)
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
    utils.rmtree(op.join(BLENDER_ROOT_DIR, subject, 'subcortical_fmri_activity'))
    shutil.copytree(out_folder, op.join(BLENDER_ROOT_DIR, subject, 'subcortical_fmri_activity'))
    if do_plot:
        plt.savefig('/home/noam/subjects/mri/mg78/subcortical_fmri_activity/figures/brain.jpg')
        plt.show()


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


def plot_points(verts, pts=None, colors=None, fig_name='', ax=None):
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
        plt.savefig('/home/noam/subjects/mri/mg78/subcortical_fmri_activity/figures/{}.jpg'.format(fig_name))
        plt.close()


def project_on_surface(subject, volume_file, colors_output_fname, surf_output_fname,
                       target_subject=None, threshold=2, overwrite_surf_data=False, overwrite_colors_file=True):
    if target_subject is None:
        target_subject = subject
    utils.make_dir(op.join(BLENDER_ROOT_DIR, subject, 'fmri'))
    for hemi in ['rh', 'lh']:
        print('project {} to {}'.format(volume_file, hemi))
        if not op.isfile(surf_output_fname.format(hemi=hemi)) or overwrite_surf_data:
            surf_data = fu.project_volume_data(volume_file, hemi, subject_id=subject, surf="pial", smooth_fwhm=3,
                target_subject=target_subject, output_fname=surf_output_fname.format(hemi=hemi))
            nans = np.sum(np.isnan(surf_data))
            if nans > 0:
                print('there are {} nans in {} surf data!'.format(nans, hemi))
            # np.save(surf_output_fname.format(hemi=hemi), surf_data)
        else:
            surf_data = np.load(surf_output_fname.format(hemi=hemi))
        if not op.isfile(colors_output_fname.format(hemi=hemi)) or overwrite_colors_file:
            print('Calulating the activaton colors for {}'.format(surf_output_fname))
            _save_fmri_colors(target_subject, hemi, surf_data, threshold, colors_output_fname.format(hemi=hemi))
        shutil.copyfile(colors_output_fname.format(hemi=hemi), op.join(BLENDER_ROOT_DIR, subject, 'fmri',
            op.basename(colors_output_fname.format(hemi=hemi))))


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


def copy_volume_to_blender(volume_fname_template, contrast='', overwrite_volume_mgz=True):
    if op.isfile(volume_fname_template.format(format='mgh')) and (not op.isfile(volume_fname_template.format(format='mgz')) or overwrite_volume_mgz):
        mri_convert(volume_fname_template, 'mgh', 'mgz')
    volume_fname = volume_fname_template.format(format='mgz')
    blender_volume_fname = op.basename(volume_fname) if contrast=='' else '{}.mgz'.format(contrast)
    shutil.copyfile(volume_fname, op.join(BLENDER_ROOT_DIR, subject, 'freeview', blender_volume_fname))
    return volume_fname


def project_volue_to_surface(subject, data_fol, threshold, volume_name, contrast, target_subject='',
                             overwrite_surf_data=True, overwrite_colors_file=True, overwrite_volume_mgz=True,
                             existing_format='nii.gz'):
    if target_subject == '':
        target_subject = subject
    volume_fname_template = op.join(data_fol, '{}.{}'.format(volume_name, '{format}'))
    # mri_convert_hemis(contrast_file_template, contrasts, existing_format=existing_format)
    copy_volume_to_blender(volume_fname_template, contrast, overwrite_volume_mgz)
    volume_fname = volume_fname_template.format(format=existing_format)
    target_subject_prefix = '_{}'.format(target_subject) if subject != target_subject else ''
    colors_output_fname = op.join(data_fol, 'fmri_{}{}_{}.npy'.format(volume_name, target_subject_prefix, '{hemi}'))
    surf_output_fname = op.join(data_fol, '{}{}_{}.mgz'.format(volume_name, target_subject_prefix, '{hemi}'))
        
    project_on_surface(subject, volume_fname, colors_output_fname, surf_output_fname,
                       target_subject, threshold, overwrite_surf_data=overwrite_surf_data,
                       overwrite_colors_file=overwrite_colors_file)
    # fu.transform_mni_to_subject('colin27', data_fol, volume_fname, '{}_{}'.format(target_subject, volume_fname))
    # load_images_file(surf_output_fname)


def calc_meg_activity_for_functional_rois(subject, meg_subject, atlas, task, contrast_name, contrast, inverse_method):
    fname_format, fname_format_cond, events_id, event_digit = meg.get_fname_format(task)
    raw_cleaning_method = 'tsss' # 'nTSSS'
    files_includes_cond = True
    meg.init_globals(meg_subject, subject, fname_format, fname_format_cond, files_includes_cond, raw_cleaning_method, contrast_name,
        SUBJECTS_MEG_DIR, task, SUBJECTS_DIR, BLENDER_ROOT_DIR)
    root_fol = op.join(SUBJECTS_DIR, subject, 'mmvt', 'fmri', 'functional_rois')
    labels_fol = op.join(root_fol, '{}_labels'.format(contrast))
    labels_output_fname = op.join(root_fol, '{}_labels_data_{}'.format(contrast, '{hemi}'))
    # src = meg.create_smooth_src(subject)
    for hemi in ['rh', 'lh']:
        meg.calc_labels_avg_per_condition(atlas, hemi, 'pial', events_id, labels_from_annot=False,
            labels_fol=labels_fol, stcs=None, inverse_method=inverse_method,
            labels_output_fname_template=labels_output_fname)


def copy_volumes(contrast_file_template):
    contrast_format = 'mgz'
    volume_type = 'mni305'
    for contrast in contrasts.keys():
        if '{contrast}' in contrast_file_template:
            contrast_file = contrast_file_template.format(contrast=contrast, hemi='{hemi}',
                                                          format=contrast_format)
            volume_file = contrast_file_template.format(contrast=contrast, hemi=volume_type, format='{format}')
        else:
            contrast_file = contrast_file_template.format(hemi='{hemi}', format=contrast_format)
            volume_file = contrast_file_template.format(hemi=volume_type, format='{format}')
        if not op.isfile(volume_file.format(format=contrast_format)):
            mri_convert(volume_file, 'nii.gz', contrast_format)
        volume_fname = volume_file.format(format=contrast_format)
        subject_volume_fname = op.join(volume_fol, '{}_{}'.format(subject, volume_name))
        if not op.isfile(subject_volume_fname):
            volume_fol, volume_name = op.split(volume_fname)
            fu.transform_mni_to_subject(subject, volume_fol, volume_name, '{}_{}'.format(subject, volume_name))
        blender_volume_fname = op.join(BLENDER_ROOT_DIR, subject, 'freeview', '{}.{}'.format(contrast, contrast_format))
        if not op.isfile(blender_volume_fname):
            print('copy {} to {}'.format(subject_volume_fname, blender_volume_fname))
            shutil.copyfile(subject_volume_fname, blender_volume_fname)


def main(subject, atlas, contrasts, contrast_file_template, t_val=2, surface_name='pial', contrast_format='mgz',
         existing_format='nii.gz', fmri_files_fol='', load_labels_from_annotation=True, volume_type='mni305', n_jobs=2):
    '''

    Parameters
    ----------
    subject: subject's name
    atlas: pacellation name
    contrasts: list of contrasts names
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
    # Check if the contrast is in mgz, and if not convert it to mgz
    mri_convert_hemis(contrast_file_template, contrasts, existing_format=existing_format)
    if contrasts is None:
        contrasts = ['group-avg']
    for contrast in contrasts:
        if '{contrast}' in contrast_file_template:
            contrast_file = contrast_file_template.format(contrast=contrast, hemi='{hemi}', format=contrast_format)
            volume_file = contrast_file_template.format(contrast=contrast, hemi=volume_type, format='{format}')
        else:
            contrast_file = contrast_file_template.format(hemi='{hemi}', format=contrast_format)
            volume_file = contrast_file_template.format(hemi=volume_type, format='{format}')
        copy_volume_to_blender(volume_file, contrast, overwrite_volume_mgz=True)
        for hemi in ['rh', 'lh']:
            # Save the contrast values with corresponding colors
            save_fmri_colors(subject, hemi, contrast, contrast_file.format(hemi=hemi), surface_name, threshold=2,
                             output_fol=fmri_files_fol)
        # Find the fMRI blobs (clusters of activation)
        find_clusters(subject, contrast, t_val, atlas, fmri_files_fol, load_labels_from_annotation, n_jobs)
        # Create functional rois out of the blobs
        create_functional_rois(subject, contrast)


if __name__ == '__main__':
    import argparse
    import sys
    from src.utils import args_utils as au

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-s', '--subject', help='subject name', required=True)
    parser.add_argument('-f', '--function', help='function name', required=False, default='all')
    parser.add_argument('-c', '--contrast', help='contrast name', required=True)
    parser.add_argument('-a', '--atlas', help='atlas name', required=False, default='laus250')
    parser.add_argument('-t', '--threshold', help='clustering threshold', required=False, default='2')
    parser.add_argument('-T', '--task', help='task', required=True)
    parser.add_argument('--volume_name', help='volume file name', required=False)
    parser.add_argument('--meg_subject', help='MEG subject name', required=False, default='')
    # args = vars(parser.parse_args())
    args = utils.Bag(au.parse_parser(parser))
    print(args)
    subject = args['subject']  #'colin27' #'fscopy' # 'mg78'
    os.environ['SUBJECT'] = subject
    threshold = float(args['threshold'])
    task = args['task'] # 'ARC' # 'MSIT' # 'ARC'
    contrast = args['contrast']  # 'arc_healthy'
    atlas = args['atlas'] # 'laus250'
    func = args['function']
    fol = op.join(FMRI_DIR, task, subject)

    contrast_name = 'interference'
    contrasts = {'non-interference-v-base': '-a 1', 'interference-v-base': '-a 2',
                 'non-interference-v-interference': '-a 1 -c 2', 'task.avg-v-base': '-a 1 -a 2'}
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

    volume_name = args.volume_name if args.volume_name != '' else args.volume_name
    if 'find_clusters' in func:
        find_clusters(subject, contrast, threshold, atlas)
    if 'main' in func:
        main(subject, atlas, None, contrast_file_template, t_val=14, surface_name='pial', existing_format='mgh')
    if 'project_volue_to_surface' in func:
        project_volue_to_surface(subject, fol, threshold, volume_name, contrast, existing_format='mgz')
    if 'calc_meg_activity' in func:
        meg_subject = args.meg_subject
        if meg_subject == '':
            print('You must set MEG subject (--meg_subject) to run calc_meg_activity function!')
        else:
            calc_meg_activity_for_functional_rois(
                subject, meg_subject, atlas, task, contrast_name, contrast, inverse_method)
    if 'copy_volumes' in func:
        copy_volumes(subject, contrast_file_template)

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

    print('finish!')



