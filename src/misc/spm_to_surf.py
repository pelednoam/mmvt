from surfer import Brain
from surfer import viz

import utils
import os
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

FS_SUBJECT = 'colin27'
mri_robust_register = 'mri_robust_register --mov {spm_brain_file} --dst {fs_brain_file} --lta {reg_file} --satit --vox2vox --mapmov {reg_spm_brain}'
mri_mask = 'mri_mask {spm_map} {spm_mask} {spm_map_masked}'
# mri_vol2surf = 'mri_vol2surf --mov {spm_map_masked} --hemi {hemi} --surf pial --reg {reg_file} --projfrac-avg 0 1 0.1 --surf-fwhm 3 --o {fs_hemi_map}'
mri_vol2surf = 'mri_vol2surf --mov {spm_mask} --hemi {hemi} --surf pial --reg {reg_file} --projfrac-avg 0 1 0.1 --surf-fwhm 3 --o {fs_hemi_map}'

SUBJECTS_DIR = '/homes/5/npeled/space3/subjects'
os.environ['SUBJECTS_DIR'] = SUBJECTS_DIR
FREE_SURFER_HOME = utils.get_exisiting_dir([os.environ.get('FREESURFER_HOME', ''),
    '/usr/local/freesurfer/stable5_3_0', '/home/noam/freesurfer'])

SPM_ROOT = '/homes/5/npeled/space3/spm_subjects'
SPM_BRAIN_TEMPLATE = 'w{subject}_MEMPRAGE_4e_1mm_iso_2.nii'
MAP_ROI = 'VLPFC'
SPM_MASK_TEMPLATE = '{}_{}_Mask.img'.format('{subject}', MAP_ROI)
FS_HEMI_MAP_TEMPLATE = '{}_{}_{}.mgz'.format('{subject}', MAP_ROI, '{hemi}')
FS_BRAIN_FILE = '$SUBJECTS_DIR/colin27/mri/orig.mgz'


def run(root_dir, spm_brain_file, fs_brain_file, reg_file, reg_spm_brain,
        spm_map, spm_mask, spm_map_masked, fs_hemi_map, fs_subject='colin27', print_only=True):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(root_dir)
    rs = utils.partial_run_script(locals(), print_only=print_only)
    # rs(mri_robust_register)
    # insert_subject_to_reg_file(reg_file, fs_subject)
    # rs(mri_mask)
    for hemi in ['rh', 'lh']:
        rs(mri_vol2surf, hemi=hemi, fs_hemi_map=fs_hemi_map.format(hemi=hemi))
    os.chdir(current_dir)


def insert_subject_to_reg_file(reg_file, subject):
    # check if subject exist
    subject_exist = False
    with open(reg_file, 'r') as f:
        lines = f.readlines()
        subject_exist = lines[-1] == 'subject {}'.format(subject)
    if not subject_exist:
        with open(reg_file, 'a') as f:
            f.write('subject {}'.format(subject))


def run_over_subjects(print_only=False, use_scaled_masks=False):
    for subject_fol in utils.get_subfolders(SPM_ROOT):
        subject = utils.namebase(subject_fol)
        spm_brain_file = SPM_BRAIN_TEMPLATE.format(subject=subject.upper())
        reg_file = '{}_register.lta'.format(subject)
        reg_spm_brain = '{}_reg.mgz'.format(subject)
        spm_map = os.path.basename(glob.glob(os.path.join(subject_fol, 'spmT_*.nii'))[0])
        spm_mask = SPM_MASK_TEMPLATE.format(subject=subject.upper())
        if use_scaled_masks:
            spm_mask_name, spm_mask_type =  os.path.splitext(spm_mask)
            spm_mask = '{}_scaled{}'.format(spm_mask_name, spm_mask_type)
        spm_map_masked = '{}_masked.mgz'.format(os.path.splitext(spm_map)[0])
        fs_hemi_map = FS_HEMI_MAP_TEMPLATE.format(subject=subject, hemi='{hemi}')
        run(subject_fol, spm_brain_file, FS_BRAIN_FILE, reg_file, reg_spm_brain, spm_map,
            spm_mask, spm_map_masked, fs_hemi_map, fs_subject=FS_SUBJECT, print_only=print_only)


def check_colors():
    subjects_folders = utils.get_subfolders(SPM_ROOT)
    good_subjects = ['pp002', 'pp003', 'pp004', 'pp005', 'pp006']
    subjects_folders = [os.path.join(SPM_ROOT, sub) for sub in good_subjects]
    subjects_colors = utils.get_spaced_colors(len(subjects_folders))
    # subjects_colors = utils.arr_to_colors(range(len(subjects_folders)), colors_map='Set1')
    plt.figure()
    for subject_fol, color in zip(subjects_folders, subjects_colors):
        subject = utils.namebase(subject_fol)
        plt.scatter([0], [0], label='{} {}'.format(subject, color), c=color)
    plt.legend()
    plt.show()


def check_all_colors():
    import matplotlib.colors
    colors = matplotlib.colors.cnames
    for color_name, color in colors.iteritems():
        plt.scatter([0], [0], label=color_name, c=color)
    plt.legend()
    plt.show()


def save_frmi_color_per_subject(out_file, threshold=2):
    # subjects_folders = utils.get_subfolders(SPM_ROOT)
    good_subjects = ['pp002', 'pp003', 'pp004', 'pp005', 'pp006']
    subjects_folders = [os.path.join(SPM_ROOT, sub) for sub in good_subjects]
    subjects_colors = utils.get_spaced_colors(len(subjects_folders))
    # subjects_colors = utils.arr_to_colors(range(len(subjects_folders)), colors_map='Set1')[:, :3]
    for hemi in ['rh', 'lh']:
        first = True
        all_colors = []
        for sub_id, (subject_fol, subject_color) in enumerate(zip(subjects_folders, subjects_colors)):
            subject = utils.namebase(subject_fol)
            print(hemi, subject)
            # if subject not in good_subjects:
            #     continue
            fs_hemi_map = os.path.join(subject_fol, FS_HEMI_MAP_TEMPLATE.format(subject=subject, hemi=hemi))
            old, brain = get_hemi_data(FS_SUBJECT, hemi, fs_hemi_map, 'pial')
            x = old.mlab_data
            brain.close()
            # x = nib.load(fs_hemi_map).get_data().squeeze()
            # plt.hist(x, bins=50)
            # plt.show()
            subject_colors = np.ones((len(x), 3))
            print(sum(x>threshold))
            # print(np.unique(x[np.where(x)]))
            subject_colors[x>threshold, :] = subject_color
            all_colors.append(subject_colors)
        all_colors = np.array(all_colors).mean(0)
        all_colors = np.hstack((np.ones((all_colors.shape[0], 1))*10, all_colors))
        np.save(out_file.format(hemi=hemi), all_colors)


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


def check_values():
    x = nib.load('/homes/5/npeled/space3/spm_subjects/pp005/PP005_VLPFC_Mask.img').get_data()
    x_rh = nib.load('/homes/5/npeled/space3/spm_subjects/pp005/pp005_VLPFC_rh.mgz').get_data()


def scale_masks(scale = 10):
    for subject_fol in utils.get_subfolders(SPM_ROOT):
        subject = utils.namebase(subject_fol)
        spm_mask = SPM_MASK_TEMPLATE.format(subject=subject.upper())
        spm_scaled_mask = os.path.join(subject_fol, '{}_scaled{}'.format(os.path.splitext(spm_mask)[0],os.path.splitext(spm_mask)[1]))
        img = nib.load(os.path.join(subject_fol, spm_mask))
        data = img.get_data()
        affine = img.get_affine()
        scaled_data = data * scale
        new_img = nib.Nifti1Image(scaled_data, affine)
        nib.save(new_img, spm_scaled_mask)


if __name__ == '__main__':
    # run_over_subjects(print_only=False, use_scaled_masks=True)
    save_frmi_color_per_subject(os.path.join(SPM_ROOT, 'fmri_{hemi}.npy'), threshold=0)
    # scale_masks()
    # check_colors()
    # check_all_colors()
    print('finish')


