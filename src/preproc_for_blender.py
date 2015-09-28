import subprocess
import glob
import os
import shutil
import numpy as np
import utils
import mne
import scipy.io
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

# SRF_FOLDER = '/homes/5/npeled/space3/subjects/mg79/ascii'
SRF_FOLDER = '/homes/5/npeled/space3/ohad/rh.pial_rois'
# SRF_TO_OBJ = '/homes/5/npeled/space3/brainder_matlab_scripts/srf2obj {}.srf > {}.obj'
# OBJ_TO_SRF = '/homes/5/npeled/space3/brainder_matlab_scripts/obj2srf {}.obj > {}.srf'
SUBJECTS_DIR = '/homes/5/npeled/space3/subjects'
# PLY_HEADER = 'ply\nformat ascii 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nelement face {}\nproperty list uchar int vertex_index\nend_header\n'
SUBJECTS_DIR = [fol for fol in ['/homes/5/npeled/space3/subjects', '/home/noam/subjects/mri'] if os.path.isdir(fol)]
SUBJECTS_DIR = SUBJECTS_DIR[0] if len(SUBJECTS_DIR) > 0 else os.environ.get('SUBJECTS_DIR')
BLENDER_ROOT_DIR = '/homes/5/npeled/space3/visualization_blender'
CACH_SUBJECT_DIR = '/space/huygens/1/users/mia/subjects/{subject}_SurferOutput/'
SCAN_SUBJECTS_DIR = '/autofs/space/lilli_001/users/DARPA-MEG/freesurfs'
TASK_MSIT, TASK_ECR = range(2)
os.environ['SUBJECTS_DIR'] = SUBJECTS_DIR
ASEG_TO_SRF = os.path.join(BLENDER_ROOT_DIR, 'brainder_matlab_scripts', 'aseg2srf -s "{}"') # -o {}'

def convert_srf_files_to_ply(srf_folder):
    srf_files = glob.glob(os.path.join(srf_folder, '*.srf'))
    for srf_file in srf_files:
        srf2ply(srf_file, '{}.ply'.format(srf_file[:-4]))

def subcortical_segmentation(subject):
    # Should source freesurfer
    # Remember that you need to have write permissions on SUBJECTS_DIR!!!
    # You can create a local folder for the current subject. The files you need are:
    # mri/aseg.mgz, mri/norm.mgz
    subject_dir = os.path.join(SUBJECTS_DIR, subject)
    aseg_to_srf_output_fol = os.path.join(SUBJECTS_DIR, subject, 'ascii')
    function_output_fol = os.path.join(subject_dir, 'subcortical_objs')
    renamed_output_fol = os.path.join(subject_dir, 'subcortical')
    utils.rmtree(function_output_fol)
    utils.rmtree(aseg_to_srf_output_fol)
    utils.rmtree(renamed_output_fol)
    print('Trying to write into {}'.format(aseg_to_srf_output_fol))
    utils.run_script(ASEG_TO_SRF.format(subject, subject_dir))
    os.rename(aseg_to_srf_output_fol, function_output_fol)
    rename_sub_cortical(function_output_fol, renamed_output_fol,
        os.path.join(SUBJECTS_DIR, 'sub_cortical_codes.txt'))


def rename_sub_cortical(fol, new_fol, codes_file):
    # names = utils.read_sub_cortical_lookup_table(codes_file)
    lookup = np.genfromtxt(codes_file, dtype=str, delimiter=',')
    lookup = {int(val):name for name, val in zip(lookup[:, 0], lookup[:, 1])}
    obj_files = glob.glob(os.path.join(fol, '*.srf'))
    utils.delete_folder_files(new_fol)
    for obj_file in obj_files:
        num = int(os.path.basename(obj_file)[:-4].split('_')[-1])
        new_name = lookup.get(num, '')
        if new_name != '':
            srf2ply(obj_file, os.path.join(new_fol, '{}.ply'.format(new_name)))
    blender_fol = os.path.join(BLENDER_SUBJECT_DIR, 'subcortical')
    if os.path.isdir(blender_fol):
        shutil.rmtree(blender_fol)
    shutil.copytree(new_fol, blender_fol)


def rename_cortical(fol, new_fol, codes_file):
    names = {}
    with open(codes_file, 'r') as f:
        for code, line in enumerate(f.readlines()):
            names[code+1] = line.strip()
    ply_files = glob.glob(os.path.join(fol, '*.ply'))
    utils.delete_folder_files(new_fol)
    for ply_file in ply_files:
        base_name = os.path.basename(ply_file)
        num = int(base_name.split('.')[-2])
        hemi = base_name.split('.')[0]
        name = names.get(num, num)
        new_name = '{}-{}'.format(name, hemi)
        shutil.copy(ply_file, os.path.join(new_fol, '{}.ply'.format(new_name)))


def freesurfer_surface_to_blender_surface(subject, hemi='both'):
    # Files needed:
    # surf/rh.pial, surf/lh.pial
    for hemi in utils.get_hemis(hemi):
        surf_name = os.path.join(SUBJECTS_DIR, subject, 'surf', '{}.pial'.format(hemi))
        surf_wavefront_name = '{}.asc'.format(surf_name)
        utils.run_script('mris_convert {} {}'.format(surf_name, surf_wavefront_name))
        surf_new_name = '{}.srf'.format(surf_name)
        print(surf_wavefront_name, surf_new_name)
        os.rename(surf_wavefront_name, surf_new_name)
    # runScript(SRF_TO_OBJ.format(surf_name, surf_name))


def montage_to_npy(montage_file, output_file):
    sfp = mne.channels.read_montage(montage_file)
    np.savez(output_file, pos=np.array(sfp.pos), names=sfp.ch_names)


def electrodes_csv_to_npy(ras_file, output_file, delimiter=','):
    data = np.genfromtxt(ras_file, dtype=str, delimiter=delimiter)
    pos = data[1:, 1:].astype(float)
    names = data[1:, 0]
    # Should also check in the electrodes data file
    if len(set(names))!=len(names):
        raise Exeption('Duplicate electrodes names!')
    np.savez(output_file, pos=pos, names=names)
    return output_file

def read_electrodes(electrodes_file):
    elecs = np.load(electrodes_file)
    for (x, y, z), name in zip(elecs['pos'], elecs['names']):
        print(name, x, y, z)


def read_electrodes_data(elecs_data_dic, conditions, montage_file, output_file_name, from_t=0, to_t=None,
                         norm_by_percentile=True, norm_percs=(1,99)):
    for cond_id, (field, file_name) in enumerate(elecs_data_dic.iteritems()):
        d = scipy.io.loadmat(file_name)
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


def read_electrodes_positions():
    electrodes_folder = os.path.join(subject_dir, 'electrodes')
    # montage_to_npy('/homes/5/npeled/space3/ohad/mg79/mg79.sfp', '/homes/5/npeled/space3/ohad/mg79/electrodes_positions.npz')
    out_file = electrodes_csv_to_npy(os.path.join(electrodes_folder, '{}_RAS.csv'.format(subject)), os.path.join(subject_dir, 'electrodes', 'electrodes_positions.npz'))
    shutil.copyfile(out_file, os.path.join(BLENDER_SUBJECT_DIR, 'electrodes_positions.npz'))


def read_electrodes_data_one_mat(mat_file, conditions, output_file_name, electrodeses_names_fiels,
        field_cond_template, from_t=0, to_t=None, norm_by_percentile=True, norm_percs=(1,99)):
    d = scipy.io.loadmat(mat_file)
    labels = d[electrodeses_names_fiels]
    labels = [str(l[0][0]) for l in labels]
    for cond_id, cond_name in enumerate(conditions):
        field = field_cond_template.format(cond_name)
        if cond_id == 0:
            data = np.zeros((d[field].shape[0], to_t - from_t, 2))
        # times = np.arange(0, to_t*2, 2)
        # todo: Need to do some interpulation for the MEG
        cond_data = d[field] # [:, times]
        cond_data_downsample = utils.downsample_2d(cond_data, 2)
        data[:, :, cond_id] = cond_data_downsample[:, from_t:to_t]
    if norm_by_percentile:
        norm_val = max(map(abs, [np.percentile(data, norm_percs[ind]) for ind in [0,1]]))
    else:
        norm_val = max(map(abs, [np.max(data), np.min(data)]))
    data /= norm_val
    avg_data = np.mean(data, 2)
    colors = utils.mat_to_colors(avg_data, np.percentile(avg_data, 2), np.percentile(avg_data, 98), colorsMap='RdBu', flip_cm=True)
    np.savez(output_file_name, data=data, names=labels, conditions=conditions, colors=colors)


def check_montage_and_electrodes_names(montage_file, electrodes_names_file):
    sfp = mne.channels.read_montage(montage_file)
    names = np.loadtxt(electrodes_names_file, dtype=np.str)
    names = set([str(e.strip()) for e in names])
    montage_names = set(sfp.ch_names)
    print(names-montage_names)
    print(montage_names-names)


def srf2ply(srf_file, ply_file):
    print('convert {} to {}'.format(srf_file, ply_file))
    verts, faces, verts_num, faces_num = utils.read_srf_file(srf_file)
    utils.write_ply_file(verts, faces, ply_file)
    return ply_file
    # faces = np.hstack((np.ones((faces_num, 1)) * 3, faces))
    # with open(ply_file, 'w') as f:
    #     f.write(PLY_HEADER.format(verts_num, faces_num))
    #     np.savetxt(f, verts, '%.5f', ' ')
    #     np.savetxt(f, faces, '%d', ' ')


def obj2ply(obj_file, ply_file):
    # with open(obj_file, 'r') as f:
    #     lines = f.readlines()
    #     verts = np.array([[float(v) for v in l.strip().split(' ')[1:]] for l in lines if l[0]=='v'])
    #     faces = np.array([[int(v) for v in l.strip().split(' ')[1:]] for l in lines if l[0]=='f'])
    # faces -= 1
    verts, faces = utils.read_obj_file(obj_file)
    utils.write_ply_file(verts, faces, ply_file)
    # faces = np.hstack((np.ones((faces.shape[0], 1)) * 3, faces))
    # with open(ply_file, 'w') as f:
    #     f.write(PLY_HEADER.format(verts.shape[0], faces.shape[0]))
    #     np.savetxt(f, verts, '%.5f', ' ')
    #     np.savetxt(f, faces, '%d', ' ')


def calc_faces_verts_dic():
    for hemi in hemis:
        ply_file = os.path.join(subject_dir,'surf', '{}.pial.ply'.format(hemi))
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
        out_file = os.path.join(BLENDER_SUBJECT_DIR, 'faces_verts_{}'.format(hemi))
        print('writing {}'.format(out_file))
        np.save(out_file, d_mat.astype(np.int))


def check_ply_files(ply_subject, ply_blender):
    for hemi in ['rh', 'lh']:
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


def prepare_local_subjects_folder(subject, remote_subject_dir, local_subjects_dir):
    local_subject_dir = os.path.join(local_subjects_dir, subject)
    neccesary_files = {'mri': ['aseg.mgz', 'norm.mgz'], 'surf': ['rh.pial', 'lh.pial', 'rh.sphere.reg', 'lh.sphere.reg']}
    for fol, files in neccesary_files.iteritems():
        if not os.path.isdir(os.path.join(local_subject_dir, fol)):
            os.makedirs(os.path.join(local_subject_dir, fol))
        for file_name in files:
            try:
                shutil.copyfile(os.path.join(remote_subject_dir, fol, file_name),
                            os.path.join(local_subject_dir, fol, file_name))
            except:
                pass
    all_files_exists = True
    for fol, files in neccesary_files.iteritems():
        for file_name in files:
            if not os.path.isfile(os.path.join(local_subject_dir, fol, file_name)):
                print("The file {} doesn't exist in the local subjects folder!!!".format(file_name))
                all_files_exists = False
    if not all_files_exists:
        raise Exception('Not all files exist in the local subject folder!!!')


def convert_perecelated_cortex(subject_dir, aparc_name, hemi='both'):
    for hemi in utils.get_hemis(hemi):
        convert_srf_files_to_ply(os.path.join(subject_dir,'{}.pial.{}'.format(aparc_name, hemi)))
        utils.rmtree(os.path.join(subject_dir,'{}.pial.names.{}'.format(aparc_name, hemi)))
        rename_cortical(os.path.join(subject_dir,'{}.pial.{}'.format(aparc_name, hemi)),
                        os.path.join(subject_dir,'{}.pial.names.{}'.format(aparc_name, hemi)),
                        os.path.join(subject_dir,'label', '{}.{}.annot_names.txt'.format(hemi, aparc_name)))
        blender_fol = os.path.join(BLENDER_SUBJECT_DIR,'{}.pial.names.{}'.format(aparc_name, hemi))
        utils.rmtree(blender_fol)
        shutil.copytree(os.path.join(subject_dir,'{}.pial.names.{}'.format(aparc_name, hemi)),
                        os.path.join(BLENDER_SUBJECT_DIR,'{}.pial.names.{}'.format(aparc_name, hemi)))


def convert_cortex(subject_dir, hemi='both'):
    for hemi in utils.get_hemis(hemi):
        ply_file = srf2ply(os.path.join(subject_dir,'surf', '{}.pial.srf'.format(hemi)), os.path.join(subject_dir,'surf', '{}.pial.ply'.format(hemi)))
        shutil.copyfile(ply_file, os.path.join(BLENDER_SUBJECT_DIR, '{}.pial.ply'.format(hemi)))


def create_annotation_file_from_fsaverage(subject, aparc_name='aparc250', overwrite=True, n_jobs=6):
    utils.morph_labels_from_fsaverage(subject, SUBJECTS_DIR, aparc_name, n_jobs=n_jobs)
    utils.labels_to_annot(subject, SUBJECTS_DIR, aparc_name, overwrite=overwrite)


def parcelate_cortex(subject, aparc_name='aparc250'):
    matlab_command = os.path.join(BLENDER_ROOT_DIR, 'brainder_matlab_scripts', 'splitting_cortical.m')
    matlab_command = "'{}'".format(matlab_command)
    scipy.io.savemat(os.path.join(BLENDER_ROOT_DIR, 'brainder_matlab_scripts', 'params.mat'), mdict={'subject': subject, 'aparc':aparc_name})
    cmd = 'matlab -nodisplay -nosplash -nodesktop -r "run({}); exit;"'.format(matlab_command)
    utils.run_script(cmd)


if __name__ == '__main__':
    # subject = 'colin27'
    subject = 'hc004'
    hemis = ['rh', 'lh']
    subject_dir = os.path.join(SUBJECTS_DIR, subject)
    BLENDER_SUBJECT_DIR = os.path.join(BLENDER_ROOT_DIR, subject)
    if not os.path.isdir(BLENDER_SUBJECT_DIR):
        os.makedirs(BLENDER_SUBJECT_DIR)
    aparc_name = 'laus250' # 'aparc250'
    task = TASK_MSIT
    remote_subject_dir = CACH_SUBJECT_DIR.format(subject=subject.upper())
    remote_subject_dir = os.path.join('/cluster/neuromind/tools/freesurfer', subject)
    # remote_subject_dir = os.path.join(SCAN_SUBJECTS_DIR, subject)
    # Files needed in the local subject folder
    # subcortical_to_surface: mri/aseg.mgz, mri/norm.mgz
    # freesurfer_surface_to_blender_surface: surf/rh.pial, surf/lh.pial
    # convert_srf_files_to_ply: surf/lh.sphere.reg, surf/rh.sphere.reg

    # prepare_local_subjects_folder(subject, remote_subject_dir, SUBJECTS_DIR)
    # 1) Create srf files for subcortical structures
    # !!! Should source freesurfer !!!
    # Remember that you need to have write permissions on SUBJECTS_DIR!!!
    # subcortical_segmentation(subject)

    # 2) Create annotation file from fsaverage
    create_annotation_file_from_fsaverage(subject, aparc_name, overwrite=True, n_jobs=1)
    # 1) convert rh.pial and lh.pial to rh.pial.srf and lh.pial.srf
    freesurfer_surface_to_blender_surface(subject)
    # 2) convert the srf files to ply
    convert_cortex(subject_dir)

    # Calls Matlab 'splitting_cortical.m' script
    parcelate_cortex(subject, aparc_name)

    # 3) After running splitting_cortical_surface in Matlab, convert the  obj files to ply
    convert_perecelated_cortex(subject_dir, aparc_name)
    # 5) Read the montage file and save it as numpy
    # read_electrodes_positions()


    # 7) Create a dictionary for verts and faces for both hemis
    # calc_faces_verts_dic()
    # for subcortical_ply in glob.glob(os.path.join(BLENDER_SUBJECT_DIR, 'subcortical', '*.ply')):
    #     subcortical_name = utils.namebase(subcortical_ply)
    #     calc_faces_verts_dic(subcortical_ply,
    #         os.path.join(BLENDER_SUBJECT_DIR, 'subcortical', '{}_faces_verts'.format(subcortical_name)))


    # 6) Read the electrodes data
    # if task==TASK_ECR:
    #     read_electrodes_data_one_mat(os.path.join(subject_dir, 'electrodes', 'electrodes_data.mat'),
    #         ['happy', 'fear'], os.path.join(BLENDER_SUBJECT_DIR, 'electrodes_data.npz'),
    #         electrodeses_names_fiels='names', field_cond_template = '{}_ERP', from_t=0, to_t=2500)
    # elif task==TASK_MSIT:
    #     read_electrodes_data_one_mat(os.path.join(subject_dir, 'electrodes', 'electrodes_data.mat'),
    #         ['noninterference', 'interference'], os.path.join(BLENDER_SUBJECT_DIR, 'electrodes_data.npz'),
    #         electrodeses_names_fiels='electrodes', field_cond_template = '{}_evoked', from_t=500, to_t=3000)
    # else:
    #     read_electrodes_data({'HappyMatr': '/homes/5/npeled/space3/inaivu/data/mg79_ieeg/angelique/ERPAverageValuesHappyMatr.mat',
    #                           'FearMatr': '/homes/5/npeled/space3/inaivu/data/mg79_ieeg/angelique/ERPAverageValuesFearMatr.mat'},
    #                          ['Happy', 'Fear'],
    #                          '/homes/5/npeled/space3/ohad/mg79/mg79.sfp',
    #                          '/homes/5/npeled/space3/ohad/mg79/electrodes_data.npz', 0, 2500)

    # check_ply_files(os.path.join(subject_dir,'surf', '{}.pial.ply'),
    #                 os.path.join(BLENDER_SUBJECT_DIR, '{}.pial.ply'))

    # misc
    # check_montage_and_electrodes_names('/homes/5/npeled/space3/ohad/mg79/mg79.sfp', '/homes/5/npeled/space3/inaivu/data/mg79_ieeg/angelique/electrode_names.txt')
    print('finish!')

