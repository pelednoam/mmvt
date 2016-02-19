import glob
import os
import os.path as op
import shutil
import numpy as np
from setuptools.command.egg_info import overwrite_arg

from src import utils
from src import matlab_utils
import mne
import scipy.io as sio
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
FREE_SURFER_HOME = utils.get_link_dir(LINKS_DIR, 'freesurfer', 'FREESURFER_HOME')
BLENDER_ROOT_DIR = op.join(LINKS_DIR, 'mmvt')
CACH_SUBJECT_DIR = '/space/huygens/1/users/mia/subjects/{subject}_SurferOutput/'
SCAN_SUBJECTS_DIR = '/autofs/space/lilli_001/users/DARPA-MEG/freesurfs'
TASK_MSIT, TASK_ECR = range(2)
os.environ['SUBJECTS_DIR'] = SUBJECTS_DIR
BRAINDER_SCRIPTS_DIR = op.join(BLENDER_ROOT_DIR, 'brainder_scripts')
ASEG_TO_SRF = op.join(BRAINDER_SCRIPTS_DIR, 'aseg2srf -s "{}"') # -o {}'
HEMIS = ['rh', 'lh']

aparc2aseg = 'mri_aparc2aseg --s {subject} --annot {atlas} --o {atlas}+aseg.mgz'
STAT_AVG, STAT_DIFF = range(2)
STAT_NAME = {STAT_DIFF: 'diff', STAT_AVG: 'avg'}


def subcortical_segmentation(subject, overwrite_subcortical_objs=False):
    # !!! Can't run in an IDE! must run in the terminal after sourcing freesurfer !!!
    # Remember that you need to have write permissions on SUBJECTS_DIR!!!
    # You can create a local folder for the current subject. The files you need are:
    # mri/aseg.mgz, mri/norm.mgz'
    subject_dir = op.join(SUBJECTS_DIR, subject)
    script_fname = op.join(BRAINDER_SCRIPTS_DIR, 'aseg2srf')
    if not op.isfile(script_fname):
        raise Exception('The subcortical segmentation script is missing! {}'.format(script_fname))
    if not utils.is_exe(script_fname):
        utils.set_exe_permissions(script_fname)

    aseg_to_srf_output_fol = op.join(SUBJECTS_DIR, subject, 'ascii')
    function_output_fol = op.join(subject_dir, 'subcortical_objs')
    renamed_output_fol = op.join(subject_dir, 'subcortical')
    lookup = load_subcortical_lookup_table()
    obj_files = glob.glob(op.join(function_output_fol, '*.srf'))
    if len(obj_files) < len(lookup) or overwrite_subcortical_objs:
        utils.delete_folder_files(function_output_fol)
        utils.delete_folder_files(aseg_to_srf_output_fol)
        utils.delete_folder_files(renamed_output_fol)
        print('Trying to write into {}'.format(aseg_to_srf_output_fol))
        utils.run_script(ASEG_TO_SRF.format(subject, subject_dir))
        os.rename(aseg_to_srf_output_fol, function_output_fol)
    ply_files = glob.glob(op.join(renamed_output_fol, '*.ply'))
    if len(ply_files) < len(lookup) or overwrite_subcortical_objs:
        convert_and_rename_subcortical_files(function_output_fol, renamed_output_fol, lookup)


def load_subcortical_lookup_table():
    codes_file = op.join(SUBJECTS_DIR, 'sub_cortical_codes.txt')
    lookup = np.genfromtxt(codes_file, dtype=str, delimiter=',')
    lookup = {int(val):name for name, val in zip(lookup[:, 0], lookup[:, 1])}
    return lookup


def convert_and_rename_subcortical_files(fol, new_fol, lookup):
    obj_files = glob.glob(op.join(fol, '*.srf'))
    utils.delete_folder_files(new_fol)
    for obj_file in obj_files:
        num = int(op.basename(obj_file)[:-4].split('_')[-1])
        new_name = lookup.get(num, '')
        if new_name != '':
            utils.srf2ply(obj_file, op.join(new_fol, '{}.ply'.format(new_name)))
    blender_fol = op.join(BLENDER_SUBJECT_DIR, 'subcortical')
    if op.isdir(blender_fol):
        shutil.rmtree(blender_fol)
    shutil.copytree(new_fol, blender_fol)


def rename_cortical(fol, new_fol, codes_file):
    names = {}
    with open(codes_file, 'r') as f:
        for code, line in enumerate(f.readlines()):
            names[code+1] = line.strip()
    ply_files = glob.glob(op.join(fol, '*.ply'))
    utils.delete_folder_files(new_fol)
    for ply_file in ply_files:
        base_name = op.basename(ply_file)
        num = int(base_name.split('.')[-2])
        hemi = base_name.split('.')[0]
        name = names.get(num, num)
        new_name = '{}-{}'.format(name, hemi)
        shutil.copy(ply_file, op.join(new_fol, '{}.ply'.format(new_name)))


def freesurfer_surface_to_blender_surface(subject, hemi='both', overwrite=False):
    # Files needed:
    # surf/rh.pial, surf/lh.pial
    for hemi in utils.get_hemis(hemi):
        surf_name = op.join(SUBJECTS_DIR, subject, 'surf', '{}.pial'.format(hemi))
        surf_wavefront_name = '{}.asc'.format(surf_name)
        surf_new_name = '{}.srf'.format(surf_name)
        hemi_ply_fname = '{}.ply'.format(surf_name)
        if overwrite or not op.isfile(hemi_ply_fname):
            utils.run_script('mris_convert {} {}'.format(surf_name, surf_wavefront_name))
            print(surf_wavefront_name, surf_new_name)
            os.rename(surf_wavefront_name, surf_new_name)
            convert_hemis_srf_to_ply(hemi)


def convert_hemis_srf_to_ply(subject, hemi='both'):
    subject_dir = op.join(SUBJECTS_DIR, subject)
    for hemi in utils.get_hemis(hemi):
        ply_file = utils.srf2ply(op.join(subject_dir,'surf', '{}.pial.srf'.format(hemi)),
                                 op.join(subject_dir,'surf', '{}.pial.ply'.format(hemi)))
        shutil.copyfile(ply_file, op.join(BLENDER_SUBJECT_DIR, '{}.pial.ply'.format(hemi)))


def montage_to_npy(montage_file, output_file):
    sfp = mne.channels.read_montage(montage_file)
    np.savez(output_file, pos=np.array(sfp.pos), names=sfp.ch_names)


def electrodes_csv_to_npy(ras_file, output_file, bipolar=False, delimiter=','):
    data = np.genfromtxt(ras_file, dtype=str, delimiter=delimiter)
    # Check if the electrodes coordinates has a header
    try:
        header = data[0, 1:].astype(float)
    except:
        data = np.delete(data, (0), axis=0)

    pos = data[:, 1:].astype(float)
    if bipolar:
        names = []
        pos_biploar, pos_org = [], []
        for index in range(data.shape[0]-1):
            elc_group1, elc_num1 = elec_group_number(data[index, 0])
            elc_group2, elc_num12 = elec_group_number(data[index+1, 0])
            if elc_group1==elc_group2:
                names.append('{}-{}'.format(data[index+1, 0],data[index, 0]))
                pos_biploar.append(pos[index] + (pos[index+1]-pos[index])/2)
                pos_org.append([pos[index], pos[index+1]])
        pos = np.array(pos_biploar)
        pos_org = np.array(pos_org)
    else:
        names = data[1:, 0]
        pos_org = []
    if len(set(names))!=len(names):
        raise Exception('Duplicate electrodes names!')
    np.savez(output_file, pos=pos, names=names, pos_org=pos_org)


def elec_group_number(elec_name, bipolar=False):
    if bipolar:
        elec_name2, elec_name1 = elec_name.split('-')
        group, num1 = elec_group_number(elec_name1, False)
        _, num2 = elec_group_number(elec_name2, False)
        return group, num1, num2
    else:
        ind = np.where([int(s.isdigit()) for s in elec_name])[-1][0]
        num = int(elec_name[ind:])
        group = elec_name[:ind]
        return group, num


def read_electrodes(electrodes_file):
    elecs = np.load(electrodes_file)
    for (x, y, z), name in zip(elecs['pos'], elecs['names']):
        print(name, x, y, z)


def read_electrodes_data(elecs_data_dic, conditions, montage_file, output_file_name, from_t=0, to_t=None,
                         norm_by_percentile=True, norm_percs=(1,99)):
    for cond_id, (field, file_name) in enumerate(elecs_data_dic.iteritems()):
        d = sio.loadmat(file_name)
        if cond_id == 0:
            data = np.zeros((d[field].shape[0], to_t - from_t, 2))
        times = np.arange(0, to_t*2, 2)
        # todo: Need to do some interpulation for the MEG
        data[:, :, cond_id] = d[field][:, times]
        # time = d['Time']
    if norm_by_percentile:
        norm_val = max(map(abs, [np.percentile(data, norm_percs[ind]) for ind in [0,1]]))
    else:
        norm_val = max(map(abs, [np.max(data), np.min(data)]))
    data /= norm_val
    sfp = mne.channels.read_montage(montage_file)
    avg_data = np.mean(data, 2)
    colors = utils.mat_to_colors(avg_data, np.percentile(avg_data, 10), np.percentile(avg_data, 90), colorsMap='RdBu', flip_cm=True)
    np.savez(output_file_name, data=data, names=sfp.ch_names, conditions=conditions, colors=colors)


def read_electrodes_positions(subject, subject_dir, bipolar=False, copy_to_blender=True):
    electrodes_folder = op.join(subject_dir, 'electrodes')
    csv_file = op.join(electrodes_folder, '{}_RAS.csv'.format(subject))
    if not op.isfile(csv_file):
        print('No electrodes coordinates file! {}'.format(csv_file))
        return None

    output_file_name = 'electrodes{}_positions.npz'.format('_bipolar' if bipolar else '')
    output_file = op.join(subject_dir, 'electrodes', output_file_name)
    electrodes_csv_to_npy(csv_file, output_file, bipolar)
    if copy_to_blender:
        blender_file = op.join(BLENDER_SUBJECT_DIR, output_file_name)
        shutil.copyfile(output_file, blender_file)
    return output_file


def create_electrode_data_file(task, from_t, to_t, stat, conditions, subject_dir, bipolar, moving_average_win_size=0):
    input_file = op.join(subject_dir, 'electrodes', 'electrodes_data.mat')
    output_file = op.join(BLENDER_SUBJECT_DIR, 'electrodes{}_data_{}.npz'.format(
            '_bipolar' if bipolar else '', STAT_NAME[stat]))
    if task==TASK_ECR:
        read_electrodes_data_one_mat(input_file, conditions, stat, output_file,
            electrodeses_names_fiels='names', field_cond_template = '{}_ERP', from_t=from_t, to_t=to_t,
            moving_average_win_size=moving_average_win_size)# from_t=0, to_t=2500)
    elif task==TASK_MSIT:
        if bipolar:
            read_electrodes_data_one_mat(input_file, conditions, stat, output_file,
                electrodeses_names_fiels='electrodes_bipolar', field_cond_template = '{}_bipolar_evoked',
                from_t=from_t, to_t=to_t, moving_average_win_size=moving_average_win_size) #from_t=500, to_t=3000)
        else:
            read_electrodes_data_one_mat(input_file, conditions, stat, output_file,
                electrodeses_names_fiels='electrodes', field_cond_template = '{}_evoked',
                from_t=from_t, to_t=to_t, moving_average_win_size=moving_average_win_size) #from_t=500, to_t=3000)


def read_electrodes_data_one_mat(mat_file, conditions, stat, output_file_name, electrodeses_names_fiels,
        field_cond_template, from_t=0, to_t=None, norm_by_percentile=True, norm_percs=(3, 97), threshold=0,
        color_map='jet', cm_big='YlOrRd', cm_small='PuBu', flip_cm_big=False, flip_cm_small=True,
        moving_average_win_size=0):
    # load the matlab file
    d = sio.loadmat(mat_file)
    # get the labels names
    labels = d[electrodeses_names_fiels]
    #todo: change that!!!
    if len(labels) == 1:
        labels = [str(l[0]) for l in labels[0]]
    else:
        labels = [str(l[0][0]) for l in labels]
    # Loop for each condition
    for cond_id, cond_name in enumerate(conditions):
        field = field_cond_template.format(cond_name)
        # initialize the data matrix (electrodes_num x T x 2)
        if cond_id == 0:
            data = np.zeros((d[field].shape[0], to_t - from_t, 2))
        # times = np.arange(0, to_t*2, 2)
        # todo: Need to do some interpulation for the MEG
        cond_data = d[field] # [:, times]
        cond_data_downsample = utils.downsample_2d(cond_data, 2)
        data[:, :, cond_id] = cond_data_downsample[:, from_t:to_t]

    data = utils.normalize_data(data, norm_by_percentile, norm_percs)
    if stat == STAT_AVG:
        stat_data = np.squeeze(np.mean(data, axis=2))
    elif stat == STAT_DIFF:
        stat_data = np.squeeze(np.diff(data, axis=2))
    else:
        raise Exception('Wrong stat value!')

    if moving_average_win_size > 0:
        # data_mv[:, :, cond_id] = utils.downsample_2d(data[:, :, cond_id], moving_average_win_size)
        stat_data_mv = utils.moving_avg(stat_data, moving_average_win_size)

    colors = calc_colors(stat_data, norm_by_percentile, norm_percs, threshold, cm_big, cm_small,
                         flip_cm_big, flip_cm_small)
    colors_mv = calc_colors(stat_data_mv, norm_by_percentile, norm_percs, threshold, cm_big, cm_small,
                            flip_cm_big, flip_cm_small)
    # np.savez(output_file_name, data=data, names=labels, conditions=conditions, colors=colors)
    np.savez(output_file_name, data=data, stat=stat_data_mv, names=labels, conditions=conditions, colors=colors_mv)


def calc_colors(data, norm_by_percentile, norm_percs, threshold, cm_big, cm_small, flip_cm_big, flip_cm_small):
    data_max, data_min = utils.get_data_max_min(data, norm_by_percentile, norm_percs)
    data_minmax = max(map(abs, [data_max, data_min]))
    print('data minmax: {}'.format(data_minmax))
    colors = utils.mat_to_colors_two_colors_maps(data, threshold=threshold,
        x_max=data_minmax, x_min = -data_minmax, cm_big=cm_big, cm_small=cm_small,
        default_val=1, flip_cm_big=flip_cm_big, flip_cm_small=flip_cm_small)
    return colors
    # colors = utils.mat_to_colors(stat_data, -data_minmax, data_minmax, color_map)


def check_montage_and_electrodes_names(montage_file, electrodes_names_file):
    sfp = mne.channels.read_montage(montage_file)
    names = np.loadtxt(electrodes_names_file, dtype=np.str)
    names = set([str(e.strip()) for e in names])
    montage_names = set(sfp.ch_names)
    print(names-montage_names)
    print(montage_names-names)


def calc_faces_verts_dic(overwrite=False):
    ply_files = [op.join(subject_dir,'surf', '{}.pial.ply'.format(hemi)) for hemi in ['rh','lh']]
    out_files = [op.join(BLENDER_SUBJECT_DIR, 'faces_verts_{}.npy'.format(hemi)) for hemi in ['rh','lh']]
    subcortical_plys = glob.glob(op.join(BLENDER_SUBJECT_DIR, 'subcortical', '*.ply'))
    if len(subcortical_plys) > 0:
        faces_verts_dic_fnames = [op.join(BLENDER_SUBJECT_DIR, 'subcortical', '{}_faces_verts.npy'.format(
                utils.namebase(ply))) for ply in subcortical_plys]
        ply_files.extend(subcortical_plys)
        out_files.extend(faces_verts_dic_fnames)

    for ply_file, out_file in zip(ply_files, out_files):
        if not overwrite and op.isfile(out_file):
            print('{} already exist.'.format(out_file))
            continue
        # ply_file = op.join(subject_dir,'surf', '{}.pial.ply'.format(hemi))
        print('preparing a lookup table for {}'.format(ply_file))
        verts, faces = utils.read_ply_file(ply_file)
        _faces = faces.ravel()
        print('{}: verts: {}, faces: {}, faces ravel: {}'.format(ply_file, verts.shape[0], faces.shape[0], len(_faces)))
        faces_arg_sort = np.argsort(_faces)
        faces_sort = np.sort(_faces)
        faces_count = Counter(faces_sort)
        max_len = max([v for v in faces_count.values()])
        d_mat = np.ones((verts.shape[0], max_len)) * -1
        diff = np.diff(faces_sort)
        n = 0
        for ind, (k,v) in enumerate(zip(faces_sort, faces_arg_sort)):
            d_mat[k, n] = v
            n = 0 if ind<len(diff) and diff[ind] > 0 else n+1
        print('writing {}'.format(out_file))
        np.save(out_file, d_mat.astype(np.int))


def check_ply_files(ply_subject, ply_blender):
    for hemi in HEMIS:
        print('reading {}'.format(ply_subject.format(hemi)))
        verts1, faces1 = utils.read_ply_file(ply_subject.format(hemi))
        print('reading {}'.format(ply_blender.format(hemi)))
        verts2, faces2 = utils.read_ply_file(ply_blender.format(hemi))
        print('vertices: ply: {}, blender: {}'.format(verts1.shape[0], verts2.shape[0]))
        print('faces: ply: {}, blender: {}'.format(faces1.shape[0], faces2.shape[0]))
        ok = verts1.shape[0] == verts2.shape[0] and faces1.shape[0]==faces2.shape[0]
        if not ok:
            raise Exception('check_ply_files: ply files are not the same!')
    print('check_ply_files: ok')


def convert_perecelated_cortex(subject, aparc_name, hemi='both'):
    subject_dir = op.join(SUBJECTS_DIR, subject)
    for hemi in utils.get_hemis(hemi):
        utils.convert_srf_files_to_ply(op.join(subject_dir,'{}.pial.{}'.format(aparc_name, hemi)))
        utils.rmtree(op.join(subject_dir,'{}.pial.{}'.format(aparc_name, hemi)))
        rename_cortical(op.join(subject_dir,'{}.pial.{}'.format(aparc_name, hemi)),
                        op.join(subject_dir,'{}.pial.{}'.format(aparc_name, hemi)),
                        op.join(subject_dir,'label', '{}.{}.annot_names.txt'.format(hemi, aparc_name)))
        blender_fol = op.join(BLENDER_SUBJECT_DIR,'{}.pial.{}'.format(aparc_name, hemi))
        utils.rmtree(blender_fol)
        shutil.copytree(op.join(subject_dir,'{}.pial.{}'.format(aparc_name, hemi)),
                        op.join(BLENDER_SUBJECT_DIR,'{}.pial.{}'.format(aparc_name, hemi)))


def create_annotation_file_from_fsaverage(subject, aparc_name='aparc250', overwrite_annotation=False,
        overwrite_morphing=False, n_jobs=6):
    utils.morph_labels_from_fsaverage(subject, SUBJECTS_DIR, aparc_name, n_jobs=n_jobs, overwrite=overwrite_morphing)
    if not overwrite_annotation:
        subject_dir = op.join(SUBJECTS_DIR, subject)
        annotations_exist = np.all([op.isfile(op.join(subject_dir, 'label', '{}.{}.annot'.format(hemi,
            aparc_name))) for hemi in HEMIS])
    # Note that using the current mne version this code won't work, because of collissions between hemis
    # You need to change the mne.write_labels_to_annot code for that.
    if overwrite_annotation or not annotations_exist:
        utils.labels_to_annot(subject, SUBJECTS_DIR, aparc_name, overwrite=overwrite_annotation)


def parcelate_cortex(subject, aparc_name='aparc250', overwrite=False):
    labels_files = np.array([len(glob.glob(op.join(SUBJECTS_DIR, subject,'{}.pial.{}'.format(
        aparc_name, hemi), '*.ply'))) for hemi in HEMIS])
    if overwrite or np.any(labels_files == 0):
        matlab_command = op.join(BRAINDER_SCRIPTS_DIR, 'splitting_cortical.m')
        matlab_command = "'{}'".format(matlab_command)
        sio.savemat(op.join(BRAINDER_SCRIPTS_DIR, 'params.mat'),
            mdict={'subject': subject, 'aparc':aparc_name, 'subjects_dir': SUBJECTS_DIR,
                   'scripts_dir': BRAINDER_SCRIPTS_DIR, 'freesurfer_home': FREE_SURFER_HOME})
        cmd = 'matlab -nodisplay -nosplash -nodesktop -r "run({}); exit;"'.format(matlab_command)
        utils.run_script(cmd)
        # convert the  obj files to ply
        convert_perecelated_cortex(subject, aparc_name)
        save_matlab_labels_indices(subject, aparc_name)
    else:
        print('There are already labels ply files, rh:{}, lh:{}'.format(labels_files[0], labels_files[1]))


def save_matlab_labels_indices(subject, aparc_name):
    for hemi in HEMIS:
        matlab_fname = op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot_labels.m'.format(hemi, aparc_name))
        labels_dic = matlab_utils.matlab_cell_arrays_to_dict(matlab_fname)
        utils.save(labels_dic, op.join(BLENDER_ROOT_DIR, subject, 'labels_dic_{}_{}.pkl'.format(aparc_name, hemi)))


def create_electrodes_volume_file(electrodes_file, create_points_files=True, create_volume_file=False, way_points=False):
    import nibabel as nib
    from itertools import product
    import csv

    elecs = np.load(electrodes_file)
    elecs_pos, names = elecs['pos'], elecs['names']

    if create_points_files:
        groups = set([name[:3] for name in names])
        freeview_command = 'freeview -v T1.mgz:opacity=0.3 aparc+aseg.mgz:opacity=0.05:colormap=lut ' + \
            ('-w ' if way_points else '-c ')
        for group in groups:
            postfix = 'label' if way_points else 'dat'
            freeview_command = freeview_command + group + postfix + ' '
            group_pos = np.array([pos for name, pos in zip(names, elecs_pos) if name[:3]==group])
            file_name = '{}.{}'.format(group, postfix)
            with open(op.join(BLENDER_SUBJECT_DIR, 'freeview', file_name), 'w') as fp:
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
                    writer.writerow(['useRealRAS', '1'])

    if create_volume_file:
        sig = nib.load(op.join(BLENDER_SUBJECT_DIR, 'freeview', 'T1.mgz'))
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
        nib.save(img, op.join(BLENDER_SUBJECT_DIR, 'freeview', 'electrodes.nii.gz'))


def create_freeview_cmd(electrodes_file, atlas, create_points_files=True, create_volume_file=False, way_points=False):
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
    from mne.label import _read_annot
    import csv
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
    utils.make_dir(op.join(BLENDER_SUBJECT_DIR, 'freeview'))
    shutil.copyfile(new_lut_fname, op.join(BLENDER_SUBJECT_DIR, 'freeview', '{}ColorLUT.txt'.format(atlas)))


def create_aparc_aseg_file(subject, atlas, print_only=False, overwrite=False, check_mgz_values=False):
    import time
    neccesary_files = {'surf': ['lh.white', 'rh.white'], 'mri': ['ribbon.mgz']}
    utils.check_for_necessary_files(neccesary_files, op.join(SUBJECTS_DIR, subject))
    rs = utils.partial_run_script(locals(), print_only=print_only)
    aparc_aseg_file = '{}+aseg.mgz'.format(atlas)
    mri_file_fol = op.join(SUBJECTS_DIR, subject, 'mri')
    mri_file = op.join(mri_file_fol, aparc_aseg_file)
    blender_file = op.join(BLENDER_SUBJECT_DIR, 'freeview', aparc_aseg_file)
    if not op.isfile(blender_file) or overwrite:
        current_dir = op.dirname(op.realpath(__file__))
        os.chdir(mri_file_fol)
        now = time.time()
        rs(aparc2aseg)
        if op.isfile(mri_file) and op.getmtime(mri_file) > now:
            shutil.copyfile(mri_file, blender_file)
            if check_mgz_values:
                import nibabel as nib
                vol = nib.load(op.join(BLENDER_SUBJECT_DIR, 'freeview', '{}+aseg.mgz'.format(atlas)))
                vol_data = vol.get_data()
                vol_data = vol_data[np.where(vol_data)]
                data = vol_data.ravel()
                import matplotlib.pyplot as plt
                plt.hist(data, bins=100)
                plt.show()
        else:
            print('Failed to create {}'.format(mri_file))
        os.chdir(current_dir)


def main(bipolar, stat, neccesary_files, overwrite_annotation=False, overwrite_morphing_labels=False,
         overwrite_hemis_srf=False, overwrite_labels_ply_files=False):
    remote_subject_dir = op.join(SCAN_SUBJECTS_DIR, subject)
    # Files needed in the local subject folder
    # subcortical_to_surface: mri/aseg.mgz, mri/norm.mgz
    # freesurfer_surface_to_blender_surface: surf/rh.pial, surf/lh.pial
    # convert_srf_files_to_ply: surf/lh.sphere.reg, surf/rh.sphere.reg

    utils.prepare_local_subjects_folder(neccesary_files, subject, remote_subject_dir, SUBJECTS_DIR)
    # *) Create srf files for subcortical structures
    # !!! Should source freesurfer !!!
    # Remember that you need to have write permissions on SUBJECTS_DIR!!!
    subcortical_segmentation(subject)

    # *) Create annotation file from fsaverage
    create_annotation_file_from_fsaverage(subject, aparc_name, overwrite_annotation=overwrite_annotation,
        overwrite_morphing=overwrite_morphing_labels, n_jobs=1)
    # *) convert rh.pial and lh.pial to rh.pial.srf and lh.pial.srf
    freesurfer_surface_to_blender_surface(subject, overwrite=overwrite_hemis_srf)

    # *) Calls Matlab 'splitting_cortical.m' script
    parcelate_cortex(subject, aparc_name, overwrite_labels_ply_files)

    # *) Create a dictionary for verts and faces for both hemis
    calc_faces_verts_dic()

    # todo: Move these line to a different file
    # *) Read the electrodes data
    # electrodes_file = read_electrodes_positions(subject, subject_dir, bipolar=bipolar)
    # if electrodes_file:
    #     create_electrode_data_file(task, from_t_ind, to_t_ind, stat, conditions, subject_dir, bipolar)
    #     create_electrodes_volume_file(electrodes_file)
    # create_freeview_cmd(electrodes_file, aparc_name)
    # create_aparc_aseg_file(subject, aparc_name, overwrite=True)
    # create_lut_file_for_atlas(subject, aparc_name)

    check_ply_files(op.join(subject_dir,'surf', '{}.pial.ply'),
                    op.join(BLENDER_SUBJECT_DIR, '{}.pial.ply'))

    # misc
    # check_montage_and_electrodes_names('/homes/5/npeled/space3/ohad/mg79/mg79.sfp', '/homes/5/npeled/space3/inaivu/data/mg79_ieeg/angelique/electrode_names.txt')


if __name__ == '__main__':
    # ******************************************************************
    # Be sure that you have matlab installed on your machine,
    # and you can run it from the terminal by just calling 'matlab'
    # Some of the functions are using freesurfer, so if you want to
    # run main, you need to run it from the terminal (not from an IDE),
    # after sourcing freesurfer.
    # If you want to import the subcortical structures, you need to
    # download the folder 'brainder_scripts' from the git,
    # and put it under the main mmvt folder.
    # ******************************************************************

    # subject = 'colin27'
    subject = 'hc008'
    print('subject: {}'.format(subject))
    hemis = HEMIS
    subject_dir = op.join(SUBJECTS_DIR, subject)
    BLENDER_SUBJECT_DIR = op.join(BLENDER_ROOT_DIR, subject)
    if not op.isdir(BLENDER_SUBJECT_DIR):
        os.makedirs(BLENDER_SUBJECT_DIR)
    aparc_name = 'laus250' # 'aprc250'
    task = TASK_MSIT
    if task==TASK_ECR:
        conditions = ['happy', 'fear']
    elif task==TASK_MSIT:
        conditions = ['noninterference', 'interference']
    else:
        raise Exception('unknown task id!')
    from_t, to_t = -500, 2000
    from_t_ind, to_t_ind = 500, 3000
    remote_subject_dir = CACH_SUBJECT_DIR.format(subject=subject.upper())
    remote_subject_dir = op.join('/cluster/neuromind/tools/freesurfer', subject)
    neccesary_files = {'..': ['sub_cortical_codes.txt'], 'mri': ['aseg.mgz', 'norm.mgz', 'ribbon.mgz'],
        'surf': ['rh.pial', 'lh.pial', 'rh.sphere.reg', 'lh.sphere.reg', 'lh.white', 'rh.white']}

    bipolar = False
    stat = STAT_DIFF
    # main(bipolar, stat, neccesary_files)
    print('finish!')



