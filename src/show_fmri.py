from scipy.stats._continuous_distns import foldcauchy_gen
from surfer import Brain
from surfer import viz

import os, sys
import nibabel as nib
import mne
import numpy as np
import pickle
import math
import glob
import utils

SUBJECTS_DIR = '/homes/5/npeled/space3/subjects'
SUBJECT = 'mg79'
os.environ['SUBJECT'] = SUBJECT
os.environ['SUBJECTS_DIR'] = SUBJECTS_DIR #'/autofs/space/lilli_001/users/DARPA-MEG/freesurfs' #

conds = ['congruent-v-base', 'incongruent-v-base',  'congruent-v-incongruent', 'task.avg-v-base']
x, xfs = {}, {}
show_fsaverage = False

MRI_FILE_RH_FS = '/homes/5/npeled/space3/ECR_fsaverage/hc001/bold/congruence.sm05.rh/congruent-v-base/sig.nii.gz'
MRI_FILE_LH_FS = '/homes/5/npeled/space3/ECR_fsaverage/hc001/bold/congruence.sm05.lh/congruent-v-base/sig.nii.gz'

fMRI_FILE = '/homes/5/npeled/space3/ECR/hc001/bold/congruence.sm05.{}/congruent-v-base/sig_mg79.mgz'

# x = nib.load('/homes/5/npeled/Desktop/sig_fsaverage.nii.gz')
# xfs = nib.load('/homes/5/npeled/Desktop/sig_subject.nii.gz')


def get_hemi_data(subject, hemi, source, surf_name='pial', name=None, sign="abs", min=None, max=None):
    brain = Brain(subject, hemi, surf_name, curv=False)
    hemi = brain._check_hemi(hemi)
    # load data here
    scalar_data, name = brain._read_scalar_data(source, hemi, name=name)
    min, max = brain._get_display_range(scalar_data, min, max, sign)
    if sign not in ["abs", "pos", "neg"]:
        raise ValueError("Overlay sign must be 'abs', 'pos', or 'neg'")
    old = viz.OverlayData(scalar_data, brain.geo[hemi], min, max, sign)
    return old.mlab_data, brain

# def get_hemi_colors(old, threshold=2):
#     colors = utils.arr_to_colors_two_colors_maps(old.mlab_data, 'YlOrRd', 'PuBu', threshold)
#     # colors = np.zeros((len(vals), 3))
#     # pos_colors = utils.arr_to_colors(-vals[vals>=threshold], colorsMap='YlOrRd')[:, :3]
#     # neg_colors = utils.arr_to_colors(vals[vals<=-threshold], colorsMap='PuBu')[:, :3]
#     # colors[vals>=threshold, :] = pos_colors #[vals>=threshold]
#     # colors[vals<=-threshold, :] = neg_colors #[vals<=-threshold]
#     return colors


def save_fmri_colors(subject, fmri_file, surf_name, output_file, threshold=0,
                     cm_big='YlOrRd', cm_small='PuBu', flip_cm_big=True, flip_cm_small=False,
                     norm_by_percentile=True, norm_percs=(2,98)):
    data = {}
    hemis = ['rh', 'lh']
    for hemi in hemis:
        data[hemi], _ = get_hemi_data(subject, hemi, fmri_file.format(hemi), surf_name)

        if norm_by_percentile:
            data_max = max([np.percentile(data[hemi], norm_percs[1]) for hemi in hemis])
            data_min = min([np.percentile(data[hemi], norm_percs[0]) for hemi in hemis])
        else:
            data_max = max([np.max(data[hemi]) for hemi in hemis])
            data_min = min([np.min(data[hemi]) for hemi in hemis])
        data_minmax = max(map(abs, [data_max, data_min]))

    for hemi in hemis:
        colors = utils.arr_to_colors_two_colors_maps(data[hemi], threshold=threshold,
            x_max=data_minmax,x_min = -data_minmax, cm_big=cm_big, cm_small=cm_small,
            default_val=1, flip_cm_big=flip_cm_big, flip_cm_small=flip_cm_small)
        data[hemi] = np.reshape(data[hemi], (len(data[hemi]), 1))
        colors = np.hstack((data[hemi], colors))
        np.save(output_file.format(hemi), colors)


def save_labels_from_annotation(subject, parc, surf_name, fol=''):
    brain = Brain(subject, 'both', surf_name, curv=False)
    labels = mne.read_labels_from_annot(subject, parc, 'both', surf_name)
    if fol=='':
        fol = os.path.join(os.environ['SUBJECTS_DIR'], os.environ['SUBJECT'], 'label', '{}_labels'.format(parc))
        if not os.path.isdir(fol):
            os.mkdir(fol)
    for ind, label in enumerate(labels):
        print('label {}/{}'.format(ind, len(labels)))
        label.save(os.path.join(fol, label.name))

def labels_to_annot(subject, fol, parc, overwrite=True):
    for hemi in ['lh', 'lr']:
        labels = []
        for label_file in glob.glob(os.path.join(fol, '*{}.label'.format(hemi))):
            label = mne.read_label(label_file)
            labels.append(label)
        print('write {} labels to annot file'.format(hemi))
        mne.write_labels_to_annot(labels, subject, parc, overwrite, hemi=hemi)

def morph_labels(subject_from, subject_to, parc, surf_name='pial', smooth=2, overwrite=True):
    '''
        mne_morph_labels --from fsaverage --to mg79 --labeldir /homes/5/npeled/space3/subjects/fsaverage/label/laus500_labels --smooth 5
    '''
    # brain = Brain(subject_from, 'both', surf_name, curv=False)
    labels = mne.read_labels_from_annot(subject_from, parc, 'both', surf_name)
    morphed_labels = []
    for ind, label in enumerate(labels):
        try:
            print('label {}/{}'.format(ind, len(labels)))
            label.values.fill(1.0)
            morphed_label = label.morph(subject_from, subject_to, smooth)
            morphed_labels.append(morphed_label)
        except:
            print('cant morph label {}'.format(label.name))
            print(sys.exc_info()[1])
    print('{} labels were morphed succefully.'.format(len(morphed_labels)))
    mne.write_labels_to_annot(morphed_labels, subject_to, parc, overwrite, hemi='both')


def load_brain_labels(subject, hemi, surf_name, parc):
    # brain = Brain(subject, hemi, surf_name, curv=False)
    labels = mne.read_labels_from_annot(subject, parc, hemi, surf_name)
    for label_ind, label in enumerate(labels):
        brain.add_label(label)
        print(label)

def create_annot_csv(subject, parc, hemi, source_file, surf_name):
    labels = mne.read_labels_from_annot(subject, parc, hemi, surf_name)
    old, brain = get_hemi_data(source_file, hemi,surf_name)
    colors = np.zeros((old.mlab_data.shape[0], 3)) #  arrToColors(old.mlab_data, colorsMap='RdBu_r')[:, :3]
    brain.toggle_toolbars(True)
    for label_ind, label in enumerate(labels):
    # label = labels[46]
        brain.add_label(label)
        print(label)
        # brain.remove_labels()
#         colors[label.vertices, :] = np.array(label.color[:3])
#     np.savetxt('{}_{}_{}.csv'.format(subject, parc, hemi), colors, delimiter=',', fmt='%.5f')

def create_annot_dic(subject, parc, hemi, surf_name, obj_positions):
    labels = mne.read_labels_from_annot(subject, parc, hemi, surf_name)
    for label in [labels[161]]:
        print(len(label.pos), len(obj_positions))
        for label_pos, obj_pos in zip(label.pos, obj_positions):
            label_pos = round_arr(label_pos*1000)
            obj_pos = round_arr(obj_pos)
            eq = np.all(label_pos==obj_pos)
            if (not eq):
                print(label_pos, obj_pos)
        # for ind, vert_ind in enumerate(label.vertices):
        #     label_pos = round_arr(label.pos[ind]*1000)
        #     obj_pos = round_arr(obj_positions[vert_ind])
        #     eq = np.all(label_pos==obj_pos)
        #     if (not eq):
        #         print(label_pos, obj_pos)
        # label_name = label.name.strip() # decode('utf-8')
        # np.savez(os.path.join('/homes/5/npeled/space3/MMVT/mg79/labels', label_name), vertices=label.vertices, color=label.color[:3])
        # annots[label_name] = label.vertices

def round_arr(arr, perc=2):
    perc_ = float(math.pow(10,perc))
    return [int(x*perc_)/perc_ for x in arr]

# '/homes/5/npeled/space3/MMVT/rh.pial.obj'
def read_obj(obj_file):
    obj_rh = obj_file
    with open(obj_rh, 'r') as f:
        pos = np.array([map(float, line.strip().split(' ')[1:]) for line in f.readlines() if line.strip().split(' ')[0] == 'v'])
    return pos

def open_dpv(dpv_file):
    with open(dpv_file, 'r') as f:
        pos = np.array([map(float, line.strip().split(' ')[1:]) for line in f.readlines()])
    return pos

def load_dpv(dpv_file, label_num=31):
    pos = open_dpv(dpv_file)
    lines = np.where(pos[:, 3]==label_num)[0]
    pos[lines, 0] = 1
    pos = pos[:, :3]
    np.savetxt('{}.csv'.format(label_num), pos)





# get_hemi_data(MRI_FILE_RH, 'rh', pos)
# load_dpv('/homes/5/npeled/space3/MMVT/rh.laus250.annot.dpv')
# obj = read_obj('/homes/5/npeled/space3/MMVT/rh.pial_rois/rh.pial_roi.0030.obj')
# create_annot_dic('mg79', 'laus250', 'rh', 'pial', obj)


def create_fmri_dpv():
    xr = nib.load(MRI_FILE_RH)
    # xl = nib.load(MRI_FILE_LH)
    pos = open_dpv('/homes/5/npeled/space3/MMVT/test/rh.pial.dpv')
    colors = np.squeeze(xr.dataobj)
    zeros_indices = np.abs(colors)<2
    colors[zeros_indices] = np.zeros((1))
    pos[:, 3] = colors
    pos = np.hstack((np.arange(pos.shape[0]).reshape((pos.shape[0], 1)), pos))
    np.savetxt('/homes/5/npeled/space3/MMVT/test/rh.pial.fmri.dpv', pos, fmt='%.5f', delimiter=' ')
    print('sdf')

def change_ply_colors(dpv_file, colors_values):
    colors = utils.arr_to_colors(colors_values)
    start_reading = False
    with open(dpv_file, 'r+') as f:
        for line in f.readlines():
            if (line.strip()=='end_header'):
                start_reading = True
                continue
            if (start_reading):
                vals = line.strip().split(' ')


def load_fmri(subject, input_file, hemi='both'):
    brain = Brain(subject, hemi, "pial", curv=False)
    brain.toggle_toolbars(True)
    if hemi=='both':
        for hemi in ['rh', 'lh']:
            brain.add_overlay(input_file.format(hemi), hemi=hemi)
    else:
        brain.add_overlay(input_file.format(hemi), hemi=hemi)


def morph_stc(subject_from, subject_to, stc_from_file):
    stc_from = mne.read_source_estimate(stc_from_file)
    vertices_to = [np.arange(10242), np.arange(10242)]
    stc_to = mne.morph_data(subject_from, subject_to, stc_from, n_jobs=4,
                        grade=vertices_to)
    stc_to.save('{}_{}.stc'.format(stc_from_file[:-4], subject_to))


if __name__ == '__main__':
    subject = 'mg79'
    # load_brain_labels('mg79','rh','pial','aparc')
    # load_brain_labels('fsaverage','rh','pial','laus500')

    # save_fmri_colors(subject, fMRI_FILE, 'pial',
    #         '/homes/5/npeled/space3/MMVT/mg79/fmri_{}.npy',  threshold=2)
    # Show the fRMI in pysurfer
    # load_fmri(subject, input_file=fMRI_FILE, hemi='both')

    # morph_labels('fsaverage', 'mg79', 'laus500', 'pial', smooth=2)
    # save_labels_from_annotation('fsaverage', 'laus500', 'pial')
    # labels_to_annot('mg79', '/homes/5/npeled/space3/subjects/mg79/label/laus500', 'laus500', True)
    # morph_stc('hc017', 'mg79', '/homes/5/npeled/space3/MMVT/mg79/MEG/hc017-C1-spm-rh.stc')
    print('finish!')



