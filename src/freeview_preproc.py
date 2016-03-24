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


def create_freeview_cmd(subject, atlas, bipolar, create_points_files=True, way_points=False):
    electrodes_file = op.join(SUBJECTS_DIR, subject, 'electrodes', 'electrodes{}_positions.npz'.format(
        '_bipolar' if bipolar else ''))
    blender_freeview_fol = op.join(BLENDER_ROOT_DIR, subject, 'freeview')
    freeview_command = 'freeview -v T1.mgz:opacity=0.3 ' + \
        '{}+aseg.mgz:opacity=0.05:colormap=lut:lut={}ColorLUT.txt '.format(atlas, atlas)
    if op.isfile(electrodes_file) and create_points_files:
        elecs = np.load(electrodes_file)
        groups = set([utils.elec_group(name, bipolar) for name in elecs['names']])
        freeview_command += '-w ' if way_points else '-c '
        postfix = '.label' if way_points else '.dat'
        for group in groups:
            freeview_command += group + postfix + ' '
    utils.make_dir(blender_freeview_fol)
    with open(op.join(blender_freeview_fol, 'run_freeview.sh'), 'w') as sh_file:
        sh_file.write(freeview_command)
    print(freeview_command)


# todo: fix duplications!
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
        names = [name.astype(str) for name in names]
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


def create_electrodes_points(subject, bipolar=False, create_points_files=True, create_volume_file=False,
                             way_points=False):
    electrodes_file = op.join(SUBJECTS_DIR, subject, 'electrodes',
        'electrodes{}_positions.npz'.format('_bipolar' if bipolar else ''))
    if not op.isfile(electrodes_file):
        return

    elecs = np.load(electrodes_file)
    elecs_pos, names = elecs['pos'], [name.astype(str) for name in elecs['names']]

    if create_points_files:
        groups = set([utils.elec_group(name, bipolar) for name in names])
        freeview_command = 'freeview -v T1.mgz:opacity=0.3 aparc+aseg.mgz:opacity=0.05:colormap=lut ' + \
            ('-w ' if way_points else '-c ')
        for group in groups:
            postfix = 'label' if way_points else 'dat'
            freeview_command = freeview_command + group + postfix + ' '
            group_pos = np.array([pos for name, pos in zip(names, elecs_pos) if utils.elec_group(name, bipolar) == group])
            file_name = '{}.{}'.format(group, postfix)
            with open(op.join(BLENDER_ROOT_DIR, subject, 'freeview', file_name), 'w') as fp:
                writer = csv.writer(fp, delimiter=' ')
                if way_points:
                    writer.writerow(['#!ascii label  , from subject  vox2ras=Scanner'])
                    writer.writerow([len(group_pos)])
                    points = np.hstack((np.ones((len(group_pos), 1)) * -1, group_pos, np.ones((len(group_pos), 1))))
                    writer.writerows(points)
                else:
                    writer.writerows(group_pos)
                    writer.writerow(['info'])
                    writer.writerow(['numpoints', len(group_pos)])
                    writer.writerow(['useRealRAS', '0'])

    if create_volume_file:
        import nibabel as nib
        from itertools import product
        sig = nib.load(op.join(BLENDER_ROOT_DIR, subject, 'freeview', 'T1.mgz'))
        sig_data = sig.get_data()
        sig_header = sig.get_header()
        electrodes_positions = np.load(electrodes_file)['pos']
        data = np.zeros((256, 256, 256), dtype=np.int16)
        # positions_ras = np.array(utils.to_ras(electrodes_positions, round_coo=True))
        elecs_pos = np.array(elecs_pos, dtype=np.int16)
        for pos_ras in elecs_pos:
            for x, y, z in product(*([[d+i for i in range(-5,6)] for d in pos_ras])):
                data[z,y,z] = 1
        img = nib.Nifti1Image(data, sig_header.get_affine(), sig_header)
        nib.save(img, op.join(BLENDER_ROOT_DIR, subject, 'freeview', 'electrodes.nii.gz'))


def copy_T1(subject):
    blender_T1 = op.join(BLENDER_ROOT_DIR, subject, 'freeview', 'T1.mgz')
    subject_T1 = op.join(SUBJECTS_DIR, subject, 'mri', 'T1.mgz')
    if not op.isfile(blender_T1):
        utils.copy_file(subject_T1, blender_T1)


def read_vox2ras0():
    import nibabel as nib
    from nibabel.affines import apply_affine
    mri = nib.load(op.join(SUBJECTS_DIR, subject, 'mri', 'T1.mgz'))
    mri_header = mri.get_header()
    ras_tkr2vox = np.linalg.inv(mri_header.get_vox2ras_tkr())
    vox2ras = mri_header.get_vox2ras()
    ras_rkr2ras = np.dot(ras_tkr2vox, vox2ras)
    print(np.dot([-22.37, 22.12, -11.70], ras_rkr2ras))
    print('sdf')


def main(subject, aparc_name, bipolar, overwrite_aseg_file=False, create_volume_file=False):
    # Create the files for freeview bridge
    create_freeview_cmd(subject, aparc_name, bipolar)
    create_electrodes_points(subject, bipolar, create_points_files=True, create_volume_file=create_volume_file, way_points=False)
    create_aparc_aseg_file(subject, aparc_name, overwrite=overwrite_aseg_file)
    create_lut_file_for_atlas(subject, aparc_name)
    copy_T1(subject)


if __name__ == '__main__':
    import sys
    subject = sys.argv[1] if len(sys.argv) > 1 else 'pp009'
    aparc_name = sys.argv[2] if len(sys.argv) > 2 else 'aparc.DKTatlas40'
    bipolar = False
    overwrite_aseg_file = False
    create_volume_file = True
    print('subject: {}, atlas: {}, bipolar: {}'.format(subject, aparc_name, bipolar))
    main(subject, aparc_name, bipolar, overwrite_aseg_file, create_volume_file)
    # create_electrodes_points(subject, bipolar, create_volume_file=False)
    # create_freeview_cmd(subject, aparc_name, bipolar)
    # create_lut_file_for_atlas(subject, aparc_name)
    print('finish!')