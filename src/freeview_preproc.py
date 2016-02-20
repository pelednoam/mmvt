import os
import os.path as op
import numpy as np
import shutil
import time
import csv
from mne.label import _read_annot
from src import utils

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
FREE_SURFER_HOME = utils.get_link_dir(LINKS_DIR, 'freesurfer', 'FREESURFER_HOME')
BLENDER_ROOT_DIR = op.join(LINKS_DIR, 'mmvt')

APARC2ASEG = 'mri_aparc2aseg --s {subject} --annot {atlas} --o {atlas}+aseg.mgz'


def create_freeview_cmd(subject, atlas, bipolar, create_points_files=True, create_volume_file=False, way_points=False):
    electrodes_file = op.join(SUBJECTS_DIR, subject, 'electrodes', 'electrodes{}_positions.npz'.format(
        '_bipolar' if bipolar else ''))

    elecs = np.load(electrodes_file)
    if create_points_files:
        groups = set([name[:3] for name in elecs['names']])
        freeview_command = 'freeview -v T1.mgz:opacity=0.3 ' + \
            '{}+aseg.mgz:opacity=0.05:colormap=lut:lut={}ColorLUT.txt '.format(atlas, atlas) + \
            ('-w ' if way_points else '-c ')
        for group in groups:
            postfix = '.label' if way_points else '.dat'
            freeview_command = freeview_command + group + postfix + ' '
    print(freeview_command)


def create_lut_file_for_atlas(subject, atlas):
    # Read the subcortical segmentation from the freesurfer lut
    lut = utils.read_freesurfer_lookup_table(FREE_SURFER_HOME, get_colors=True)
    lut_new = [list(l) for l in lut if l[0] < 1000]
    for hemi, offset in zip(['lh', 'rh'], [1000, 2000]):
        if hemi == 'lh':
            lut_new.append([1000, 'ctx-lh-unknown', 25, 5,  25, 0])
        else:
            lut_new.append([2000, 'ctx-rh-unknown', 25,  5, 25,  0])
        _, ctab, names = _read_annot(op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot'.format(hemi, atlas)))
        for index, (label, cval) in enumerate(zip(names, ctab)):
            r,g,b,a, _ = cval
            lut_new.append([index + offset + 1, label, r, g, b, a])
    lut_new.sort(key=lambda x:x[0])
    # Add the values above 3000
    for l in [l for l in lut if l[0] >= 3000]:
        lut_new.append(l)
    new_lut_fname = op.join(SUBJECTS_DIR, subject, 'label', '{}ColorLUT.txt'.format(atlas))
    with open(new_lut_fname, 'w') as fp:
        csv_writer = csv.writer(fp, delimiter='\t')
        csv_writer.writerows(lut_new)
    # np.savetxt(new_lut_fname, lut_new, delimiter='\t', fmt="%s")
    utils.make_dir(op.join(BLENDER_ROOT_DIR, subject, 'freeview'))
    shutil.copyfile(new_lut_fname, op.join(BLENDER_ROOT_DIR, subject, 'freeview', '{}ColorLUT.txt'.format(atlas)))


def create_aparc_aseg_file(subject, atlas, print_only=False, overwrite=False, check_mgz_values=False):
    neccesary_files = {'surf': ['lh.white', 'rh.white'], 'mri': ['ribbon.mgz']}
    utils.check_for_necessary_files(neccesary_files, op.join(SUBJECTS_DIR, subject))
    rs = utils.partial_run_script(locals(), print_only=print_only)
    aparc_aseg_file = '{}+aseg.mgz'.format(atlas)
    mri_file_fol = op.join(SUBJECTS_DIR, subject, 'mri')
    mri_file = op.join(mri_file_fol, aparc_aseg_file)
    blender_file = op.join(BLENDER_ROOT_DIR, subject, 'freeview', aparc_aseg_file)
    if not op.isfile(blender_file) or overwrite:
        current_dir = op.dirname(op.realpath(__file__))
        os.chdir(mri_file_fol)
        now = time.time()
        rs(APARC2ASEG)
        if op.isfile(mri_file) and op.getmtime(mri_file) > now:
            shutil.copyfile(mri_file, blender_file)
            if check_mgz_values:
                import nibabel as nib
                vol = nib.load(op.join(BLENDER_ROOT_DIR, subject, 'freeview', '{}+aseg.mgz'.format(atlas)))
                vol_data = vol.get_data()
                vol_data = vol_data[np.where(vol_data)]
                data = vol_data.ravel()
                import matplotlib.pyplot as plt
                plt.hist(data, bins=100)
                plt.show()
        else:
            print('Failed to create {}'.format(mri_file))
        os.chdir(current_dir)


def main(subject, aparc_name, bipolar):
    # Create the files for freeview bridge
    create_freeview_cmd(subject, aparc_name, bipolar)
    create_aparc_aseg_file(subject, aparc_name, overwrite=True)
    create_lut_file_for_atlas(subject, aparc_name)


if __name__ == '__main__':
    subject = 'mg78'
    aparc_name = 'laus250'
    bipolar = False
    main(subject, aparc_name, bipolar)