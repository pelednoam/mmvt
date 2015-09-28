from surfer import Brain
from surfer import viz
from surfer import project_volume_data
import os, sys
import nibabel as nib
import mne
import numpy as np
import pickle
import math
import glob
import utils
import fsfast
from sklearn.neighbors import BallTree
import shutil

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


SUBJECTS_DIR = '/homes/5/npeled/space3/subjects'
# SUBJECTS_DIR = '/autofs/space/lilli_001/users/DARPA-MEG/freesurfs'
# SUBJECTS_DIR =  '/home/noam/subjects/mri'
# SUBJECT = 'ep001'
os.environ['SUBJECTS_DIR'] = SUBJECTS_DIR
ROOT_DIR = [f for f in ['/homes/5/npeled/space3/fMRI/MSIT', '/home/noam/fMRI/MSIT'] if os.path.isdir(f)][0]
BLENDER_DIR = '/homes/5/npeled/space3/visualization_blender'
FREE_SURFER_HOME = utils.get_exisiting_dir([os.environ.get('FREESURFER_HOME', ''),
    '/usr/local/freesurfer/stable5_3_0', '/home/noam/freesurfer'])

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
    print('fMRI constrast map vertices: {}'.format(len(scalar_data)))
    min, max = brain._get_display_range(scalar_data, min, max, sign)
    if sign not in ["abs", "pos", "neg"]:
        raise ValueError("Overlay sign must be 'abs', 'pos', or 'neg'")
    old = viz.OverlayData(scalar_data, brain.geo[hemi], min, max, sign)
    return old, brain


def save_fmri_colors(subject, hemi, fmri_file, surf_name, output_file, threshold=2):
    old, brain = get_hemi_data(subject, hemi, fmri_file.format(hemi), surf_name)
    x = old.mlab_data

    # Do some sanity checks
    verts, faces = utils.read_ply_file(os.path.join(SUBJECTS_DIR, subject, 'surf', '{}.pial.ply'.format(hemi)))
    print('{}.pial.ply vertices: {}'.format(hemi, verts.shape[0]))
    if verts.shape[0] != brain.geo[hemi].coords.shape[0]:
        raise Exception("Brain and ply objects doesn't have the same verices number!")
    if len(x) != verts.shape[0]:
        raise Exception("fMRI contrast map and the hemi doens't have the same vertices number!")

    colors = utils.arr_to_colors_two_colors_maps(x, cm_big='YlOrRd', cm_small='PuBu',
        threshold=threshold, default_val=1)
    colors = np.hstack((x.reshape((len(x), 1)), colors))
    np.save(output_file, colors)


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


def mri_convert(constrast_file_template, contrasts):
    for hemi in ['rh', 'lh']:
        for contrast in contrasts.keys():
            constrast_file_nii = constrast_file_template.format(hemi=hemi, contrast=contrast, format='nii.gz')
            constrast_file_mgz = constrast_file_template.format(hemi=hemi, contrast=contrast, format='mgz')
            try:
                utils.run_script('mri_convert {} {}'.format(constrast_file_nii, constrast_file_mgz))
            except:
                pass


def calculate_subcorticals_activity(volume_file, subcortical_codes_file='', aseg_stats_file_name='',
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
    aseg_fname = os.path.join(SUBJECTS_DIR, SUBJECT, 'mri', 'aseg.mgz')
    aseg = nib.load(aseg_fname)
    aseg_hdr = aseg.get_header()
    out_folder = os.path.join(SUBJECTS_DIR, SUBJECT, 'subcortical_fmri_activity')
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)
    sub_cortical_generator = utils.sub_cortical_voxels_generator(aseg, seg_labels, 5, False, FREE_SURFER_HOME)
    for pts, seg_name, seg_id in sub_cortical_generator:
        print(seg_name)
        verts, _ = utils.read_ply_file(os.path.join(SUBJECTS_DIR, SUBJECT, 'subcortical', '{}.ply'.format(seg_name)))
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
        np.save(os.path.join(out_folder, seg_name), verts_data)
        if do_plot:
            plot_points(verts, colors=verts_colors, fig_name=seg_name, ax=ax)
        # print(pts)
    utils.rmtree(os.path.join(BLENDER_SUBJECT_DIR, 'subcortical_fmri_activity'))
    shutil.copytree(out_folder, os.path.join(BLENDER_SUBJECT_DIR, 'subcortical_fmri_activity'))
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


def project_on_surface(subject, volume_file, reg_file):
    brain = Brain(subject, 'both', 'pial', curv=False, offscreen=False)
    for hemi in ['rh', 'lh']:
        zstat = project_volume_data(volume_file, hemi, reg_file, surf='pial')
        brain.add_overlay(zstat, hemi=hemi)

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


if __name__ == '__main__':
    SUBJECT = 'mg78'
    os.environ['SUBJECT'] = SUBJECT
    BLENDER_SUBJECT_DIR = os.path.join(BLENDER_DIR, SUBJECT)

    # SUBJECT = 'mg79'
    contrast_name='interference'
    contrasts={'non-interference-v-base': '-a 1', 'interference-v-base': '-a 2', 'non-interference-v-interference': '-a 1 -c 2', 'task.avg-v-base': '-a 1 -a 2'}
    constrast_file_template = os.path.join(ROOT_DIR, SUBJECT, 'bold', '{contrast_name}.sm05.{hemi}'.format(contrast_name=contrast_name, hemi='{hemi}'), '{contrast}', 'sig.{format}')
    TR = 1.75

    # show_fMRI_using_pysurfer(SUBJECT, '/homes/5/npeled/space3/fMRI/ECR/hc004/bold/congruence.sm05.lh/congruent-v-incongruent/sig.mgz', 'rh')

    # fsfast.run(SUBJECT, root_dir=ROOT_DIR, par_file = 'msit.par', contrast_name=contrast_name, tr=TR, contrasts=contrasts, print_only=False)
    # fsfast.plot_contrast(SUBJECT, ROOT_DIR, contrast_name, contrasts, hemi='rh')
    # mri_convert(constrast_file_template, contrasts)
    constrast_file=constrast_file_template.format(
        contrast='non-interference-v-interference', hemi='{hemi}', format='mgz')
    # show_fMRI_using_pysurfer(SUBJECT, input_file=constrast_file, hemi='lh')

    root = os.path.join('/autofs/space/franklin_003/users/npeled/fMRI/MSIT/pp003')
    volume_file = os.path.join(root, 'sig.anat.mgz')
    mask_file = os.path.join(root, 'VLPFC.mask.mgz')
    masked_file = os.path.join(root, 'sig.anat.masked.mgz')
    constrast_file = os.path.join(root, 'sig.{hemi}.mgz')
    constrast_masked_file = os.path.join(root, 'sig.masked.{hemi}.mgz')

    # for hemi in ['rh', 'lh']:
    #     save_fmri_colors(SUBJECT, hemi, constrast_masked_file.format(hemi=hemi), 'pial',
    #          os.path.join(BLENDER_SUBJECT_DIR, 'fmri_{}.npy'.format(hemi)),  threshold=2)
    # Show the fRMI in pysurfer
    # show_fMRI_using_pysurfer(SUBJECT, input_file=constrast_masked_file, hemi='both')

    # load_and_show_npy(SUBJECT, '/homes/5/npeled/space3/visualization_blender/mg79/fmri_lh.npy', 'lh')

    # mask_volume(volume_file, mask_file, masked_file)
    # project_on_surface(SUBJECT, volume_file, '/autofs/space/franklin_003/users/npeled/fMRI/MSIT/pp003/register.lta')
    # show_fMRI_using_pysurfer(SUBJECT, input_file='/autofs/space/franklin_003/users/npeled/fMRI/MSIT/pp003/sig.{hemi}.masked.mgz', hemi='both')
    # calculate_subcorticals_activity('/homes/5/npeled/space3/MSIT/mg78/bold/interference.sm05.mni305/non-interference-v-interference/sig.anat.mgh',
    #              '/autofs/space/franklin_003/users/npeled/MSIT/mg78/aseg_stats.csv')
    # calculate_subcorticals_activity('/home/noam/fMRI/MSIT/mg78/bold/interference.sm05.mni305/non-interference-v-interference/sig.anat.mgh',
    #              '/home/noam/fMRI/MSIT/mg78/aseg_stats.csv')

    constrast_file=constrast_file_template.format(
        contrast='non-interference-v-interference', hemi='mni305', format='mgz')
    # calculate_subcorticals_activity(volume_file, subcortical_codes_file=os.path.join(BLENDER_DIR, 'sub_cortical_codes.txt'),
    #     method='dist')

    # SPM_ROOT = '/homes/5/npeled/space3/spm_subjects'
    # for subject_fol in utils.get_subfolders(SPM_ROOT):
    #     subject = utils.namebase(subject_fol)
    #     print(subject)
    #     constrast_masked_file = os.path.join(subject_fol, '{}_VLPFC_{}.mgz'.format(subject, '{hemi}'))
    #     show_fMRI_using_pysurfer(SUBJECT, input_file=constrast_masked_file, hemi='rh')
    brain = Brain('fsaverage', 'both', "pial", curv=False, offscreen=False)

    print('finish!')


