import os
import os.path as op
import numpy as np
import shutil
import time
import csv
from mne.label import _read_annot
from src.utils import utils


LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
FREE_SURFER_HOME = utils.get_link_dir(LINKS_DIR, 'freesurfer', 'FREESURFER_HOME')
BLENDER_ROOT_DIR = op.join(LINKS_DIR, 'mmvt')

APARC2ASEG = 'mri_aparc2aseg --s {subject} --annot {atlas} --o {atlas}+aseg.mgz'


def create_freeview_cmd(subject, args):#, atlas, bipolar, create_points_files=True, way_points=False):
    blender_freeview_fol = op.join(BLENDER_ROOT_DIR, subject, 'freeview')
    freeview_command = 'freeview -v T1.mgz:opacity=0.3 ' + \
        '{0}+aseg.mgz:opacity=0.05:colormap=lut:lut={0}ColorLUT.txt '.format(args.atlas)
    if args.elecs_names:
        groups = set([utils.elec_group(name, args.bipolar) for name in args.elecs_names])
        freeview_command += '-w ' if args.way_points else '-c '
        postfix = '.label' if args.way_points else '.dat'
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


def create_aparc_aseg_file(subject, args): #atlas, print_only=False, overwrite=False, check_mgz_values=False):
    neccesary_files = {'surf': ['lh.white', 'rh.white'], 'mri': ['ribbon.mgz']}
    utils.check_for_necessary_files(neccesary_files, op.join(SUBJECTS_DIR, subject))
    # The atlas var need to be in the locals for the APARC2ASEG call
    atlas = args.atlas
    rs = utils.partial_run_script(locals(), print_only=False)
    aparc_aseg_file = '{}+aseg.mgz'.format(args.atlas)
    mri_file_fol = op.join(SUBJECTS_DIR, subject, 'mri')
    mri_file = op.join(mri_file_fol, aparc_aseg_file)
    blender_file = op.join(BLENDER_ROOT_DIR, subject, 'freeview', aparc_aseg_file)
    if not op.isfile(blender_file) or args.overwrite_aseg_file:
        current_dir = op.dirname(op.realpath(__file__))
        os.chdir(mri_file_fol)
        now = time.time()
        rs(APARC2ASEG)
        if op.isfile(mri_file) and op.getmtime(mri_file) > now:
            shutil.copyfile(mri_file, blender_file)
        else:
            print('Failed to create {}'.format(mri_file))
        os.chdir(current_dir)


def check_mgz_values(atlas):
    import nibabel as nib
    vol = nib.load(op.join(BLENDER_ROOT_DIR, subject, 'freeview', '{}+aseg.mgz'.format(atlas)))
    vol_data = vol.get_data()
    vol_data = vol_data[np.where(vol_data)]
    data = vol_data.ravel()
    import matplotlib.pyplot as plt
    plt.hist(data, bins=100)
    plt.show()


def create_electrodes_points(subject, args): # bipolar=False, create_points_files=True, create_volume_file=False,
                             # way_points=False, electrodes_pos_fname=''):
    if args.elecs_names is None:
        return
    groups = set([utils.elec_group(name, args.bipolar) for name in args.elecs_names])
    freeview_command = 'freeview -v T1.mgz:opacity=0.3 aparc+aseg.mgz:opacity=0.05:colormap=lut ' + \
        ('-w ' if args.way_points else '-c ')
    for group in groups:
        postfix = 'label' if args.way_points else 'dat'
        freeview_command = freeview_command + group + postfix + ' '
        group_pos = np.array([pos for name, pos in zip(args.elecs_names, args.elecs_pos) if
                              utils.elec_group(name, args.bipolar) == group])
        file_name = '{}.{}'.format(group, postfix)
        with open(op.join(BLENDER_ROOT_DIR, subject, 'freeview', file_name), 'w') as fp:
            writer = csv.writer(fp, delimiter=' ')
            if args.way_points:
                writer.writerow(['#!ascii label  , from subject  vox2ras=Scanner'])
                writer.writerow([len(group_pos)])
                points = np.hstack((np.ones((len(group_pos), 1)) * -1, group_pos, np.ones((len(group_pos), 1))))
                writer.writerows(points)
            else:
                writer.writerows(group_pos)
                writer.writerow(['info'])
                writer.writerow(['numpoints', len(group_pos)])
                writer.writerow(['useRealRAS', '0'])

    if args.create_volume_file:
        import nibabel as nib
        from itertools import product
        sig = nib.load(op.join(BLENDER_ROOT_DIR, subject, 'freeview', 'T1.mgz'))
        sig_header = sig.get_header()
        data = np.zeros((256, 256, 256), dtype=np.int16)
        # positions_ras = np.array(utils.to_ras(electrodes_positions, round_coo=True))
        elecs_pos = np.array(args.elecs_pos, dtype=np.int16)
        for pos_ras in elecs_pos:
            for x, y, z in product(*([[d+i for i in range(-5,6)] for d in pos_ras])):
                data[z,y,z] = 1
        img = nib.Nifti1Image(data, sig_header.get_affine(), sig_header)
        nib.save(img, op.join(BLENDER_ROOT_DIR, subject, 'freeview', 'electrodes.nii.gz'))


def copy_T1(subject):
    for brain_file in ['T1.mgz', 'orig.mgz']:
        blender_brain_file = op.join(BLENDER_ROOT_DIR, subject, 'freeview', brain_file)
        subject_brain_file = op.join(SUBJECTS_DIR, subject, 'mri', brain_file)
        if not op.isfile(blender_brain_file):
            utils.copy_file(subject_brain_file, blender_brain_file)


def read_electrodes_pos(subject, args):
    electrodes_file = args.electrodes_pos_fname if args.electrodes_pos_fname != '' else op.join(
        SUBJECTS_DIR, subject, 'electrodes', 'electrodes{}_positions.npz'.format('_bipolar' if args.bipolar else ''))
    if op.isfile(electrodes_file):
        elecs = np.load(electrodes_file)
        elecs_pos, elecs_names = elecs['pos'], [name.astype(str) for name in elecs['names']]
        return elecs_pos, elecs_names
    else:
        return None, None

# def read_vox2ras0():
#     import nibabel as nib
#     from nibabel.affines import apply_affine
#     mri = nib.load(op.join(SUBJECTS_DIR, subject, 'mri', 'T1.mgz'))
#     mri_header = mri.get_header()
#     ras_tkr2vox = np.linalg.inv(mri_header.get_vox2ras_tkr())
#     vox2ras = mri_header.get_vox2ras()
#     ras_rkr2ras = np.dot(ras_tkr2vox, vox2ras)
#     print(np.dot([-22.37, 22.12, -11.70], ras_rkr2ras))
#     print('sdf')



def main(subject, args):
    # Create the files for freeview bridge
    utils.make_dir(op.join(BLENDER_ROOT_DIR, subject, 'freeview'))
    args.elecs_pos, args.elecs_names = read_electrodes_pos(subject, args)
    if 'all' in args.function or 'copy_T1' in args.function:
        copy_T1(subject)
    if 'all' in args.function or 'create_freeview_cmd' in args.function:
        create_freeview_cmd(subject, args)
    if 'all' in args.function or 'create_electrodes_points' in args.function:
        create_electrodes_points(subject, args)
    if 'all' in args.function or 'create_aparc_aseg_file' in args.function:
        create_aparc_aseg_file(subject, args)
    if 'all' in args.function or 'create_lut_file_for_atlas' in args.function:
        create_lut_file_for_atlas(subject, args.atlas)


if __name__ == '__main__':
    if os.environ.get('FREESURFER_HOME', '') == '':
        raise Exception('Source freesurfer and rerun')

    import argparse
    from src.utils import args_utils as au
    parser = argparse.ArgumentParser(description='MMVT freeview preprocessing')
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-a', '--atlas', help='atlas name', required=False, default='aparc.DKTatlas40')
    parser.add_argument('-b', '--bipolar', help='bipolar', required=False, default=0, type=bool)
    parser.add_argument('-f', '--function', help='function name', required=False, default='all', type=au.str_arr_type)
    parser.add_argument('--overwrite_aseg_file', help='overwrite_aseg_file', required=False, default=0, type=au.is_true)
    parser.add_argument('--create_volume_file', help='create_volume_file', required=False, default=1, type=au.is_true)
    parser.add_argument('--electrodes_pos_fname', help='electrodes_pos_fname', required=False, default='')
    parser.add_argument('--way_points', help='way_points', required=False, default=0, type=au.is_true)


    args = utils.Bag(au.parse_parser(parser))
    print(args)
    for subject in args.subject:
        main(subject, args)
    print('finish!')