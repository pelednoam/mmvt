import glob
import os
import os.path as op
import shutil
import numpy as np
import scipy.io as sio
from collections import Counter, defaultdict
import mne

from src import utils
from src import matlab_utils

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
FREE_SURFER_HOME = utils.get_link_dir(LINKS_DIR, 'freesurfer', 'FREESURFER_HOME')
BLENDER_ROOT_DIR = op.join(LINKS_DIR, 'mmvt')
os.environ['SUBJECTS_DIR'] = SUBJECTS_DIR
BRAINDER_SCRIPTS_DIR = op.join(utils.get_parent_fol(), 'brainder_scripts')
ASEG_TO_SRF = op.join(BRAINDER_SCRIPTS_DIR, 'aseg2srf -s "{}"') # -o {}'
HEMIS = ['rh', 'lh']


def subcortical_segmentation(subject, overwrite_subcortical_objs=False):
    # !!! Can't run in an IDE! must run in the terminal after sourcing freesurfer !!!
    # Remember that you need to have write permissions on SUBJECTS_DIR!!!
    # You can create a local folder for the current subject. The files you need are:
    # mri/aseg.mgz, mri/norm.mgz'
    script_fname = op.join(BRAINDER_SCRIPTS_DIR, 'aseg2srf')
    if not op.isfile(script_fname):
        raise Exception('The subcortical segmentation script is missing! {}'.format(script_fname))
    if not utils.is_exe(script_fname):
        utils.set_exe_permissions(script_fname)

    aseg_to_srf_output_fol = op.join(SUBJECTS_DIR, subject, 'ascii')
    function_output_fol = op.join(SUBJECTS_DIR, subject, 'subcortical_objs')
    renamed_output_fol = op.join(SUBJECTS_DIR, subject, 'subcortical')
    lookup = load_subcortical_lookup_table()
    obj_files = glob.glob(op.join(function_output_fol, '*.srf'))
    if len(obj_files) < len(lookup) or overwrite_subcortical_objs:
        utils.delete_folder_files(function_output_fol)
        utils.delete_folder_files(aseg_to_srf_output_fol)
        utils.delete_folder_files(renamed_output_fol)
        print('Trying to write into {}'.format(aseg_to_srf_output_fol))
        utils.run_script(ASEG_TO_SRF.format(subject, SUBJECTS_DIR, subject))
        os.rename(aseg_to_srf_output_fol, function_output_fol)
    ply_files = glob.glob(op.join(renamed_output_fol, '*.ply'))
    if len(ply_files) < len(lookup) or overwrite_subcortical_objs:
        convert_and_rename_subcortical_files(subject, function_output_fol, renamed_output_fol, lookup)


def load_subcortical_lookup_table():
    codes_file = op.join(SUBJECTS_DIR, 'sub_cortical_codes.txt')
    lookup = np.genfromtxt(codes_file, dtype=str, delimiter=',')
    lookup = {int(val):name for name, val in zip(lookup[:, 0], lookup[:, 1])}
    return lookup


def convert_and_rename_subcortical_files(subject, fol, new_fol, lookup):
    obj_files = glob.glob(op.join(fol, '*.srf'))
    utils.delete_folder_files(new_fol)
    for obj_file in obj_files:
        num = int(op.basename(obj_file)[:-4].split('_')[-1])
        new_name = lookup.get(num, '')
        if new_name != '':
            utils.srf2ply(obj_file, op.join(new_fol, '{}.ply'.format(new_name)))
    blender_fol = op.join(BLENDER_ROOT_DIR, subject, 'subcortical')
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
    for hemi in utils.get_hemis(hemi):
        ply_file = utils.srf2ply(op.join(SUBJECTS_DIR, subject,'surf', '{}.pial.srf'.format(hemi)),
                                 op.join(SUBJECTS_DIR, subject,'surf', '{}.pial.ply'.format(hemi)))
        shutil.copyfile(ply_file, op.join(BLENDER_ROOT_DIR, subject, '{}.pial.ply'.format(hemi)))


def calc_faces_verts_dic(subject, overwrite=False):
    ply_files = [op.join(SUBJECTS_DIR, subject,'surf', '{}.pial.ply'.format(hemi)) for hemi in HEMIS]
    out_files = [op.join(BLENDER_ROOT_DIR, subject, 'faces_verts_{}.npy'.format(hemi)) for hemi in HEMIS]
    subcortical_plys = glob.glob(op.join(BLENDER_ROOT_DIR, subject, 'subcortical', '*.ply'))
    if len(subcortical_plys) > 0:
        faces_verts_dic_fnames = [op.join(BLENDER_ROOT_DIR, subject, 'subcortical', '{}_faces_verts.npy'.format(
                utils.namebase(ply))) for ply in subcortical_plys]
        ply_files.extend(subcortical_plys)
        out_files.extend(faces_verts_dic_fnames)

    for ply_file, out_file in zip(ply_files, out_files):
        if not overwrite and op.isfile(out_file):
            print('{} already exist.'.format(out_file))
            continue
        # ply_file = op.join(SUBJECTS_DIR, subject,'surf', '{}.pial.ply'.format(hemi))
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
    for hemi in utils.get_hemis(hemi):
        utils.convert_srf_files_to_ply(op.join(SUBJECTS_DIR, subject,'{}.pial.{}'.format(aparc_name, hemi)))
        utils.rmtree(op.join(SUBJECTS_DIR, subject,'{}.pial.{}'.format(aparc_name, hemi)))
        rename_cortical(op.join(SUBJECTS_DIR, subject,'{}.pial.{}'.format(aparc_name, hemi)),
                        op.join(SUBJECTS_DIR, subject,'{}.pial.{}'.format(aparc_name, hemi)),
                        op.join(SUBJECTS_DIR, subject,'label', '{}.{}.annot_names.txt'.format(hemi, aparc_name)))
        blender_fol = op.join(BLENDER_ROOT_DIR, subject,'{}.pial.{}'.format(aparc_name, hemi))
        utils.rmtree(blender_fol)
        shutil.copytree(op.join(SUBJECTS_DIR, subject,'{}.pial.{}'.format(aparc_name, hemi)),
                        op.join(BLENDER_ROOT_DIR, subject,'{}.pial.{}'.format(aparc_name, hemi)))


def create_annotation_file_from_fsaverage(subject, aparc_name='aparc250', overwrite_annotation=False,
        overwrite_morphing=False, n_jobs=6):
    utils.morph_labels_from_fsaverage(subject, SUBJECTS_DIR, aparc_name, n_jobs=n_jobs, overwrite=overwrite_morphing)
    if not overwrite_annotation:
        annotations_exist = np.all([op.isfile(op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot'.format(hemi,
            aparc_name))) for hemi in HEMIS])
    # Note that using the current mne version this code won't work, because of collissions between hemis
    # You need to change the mne.write_labels_to_annot code for that.
    if overwrite_annotation or not annotations_exist:
        utils.labels_to_annot(subject, SUBJECTS_DIR, aparc_name, overwrite=overwrite_annotation)


def parcelate_cortex(subject, aparc_name, overwrite=False, only_save_labels_vertices=False):
    labels_files = np.array([len(glob.glob(op.join(SUBJECTS_DIR, subject,'{}.pial.{}'.format(
        aparc_name, hemi), '*.ply'))) for hemi in HEMIS])
    if overwrite or np.any(labels_files == 0):
        matlab_command = op.join(BRAINDER_SCRIPTS_DIR, 'splitting_cortical.m')
        matlab_command = "'{}'".format(matlab_command)
        sio.savemat(op.join(BRAINDER_SCRIPTS_DIR, 'params.mat'),
            mdict={'subject': subject, 'aparc':aparc_name, 'subjects_dir': SUBJECTS_DIR,
                   'scripts_dir': BRAINDER_SCRIPTS_DIR, 'freesurfer_home': FREE_SURFER_HOME})
        if not only_save_labels_vertices:
            cmd = 'matlab -nodisplay -nosplash -nodesktop -r "run({}); exit;"'.format(matlab_command)
            utils.run_script(cmd)
            # convert the  obj files to ply
            convert_perecelated_cortex(subject, aparc_name)
        save_matlab_labels_vertices(subject, aparc_name)
    else:
        print('There are already labels ply files, rh:{}, lh:{}'.format(labels_files[0], labels_files[1]))


def save_matlab_labels_vertices(subject, aparc_name):
    for hemi in HEMIS:
        matlab_fname = op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot_labels.m'.format(hemi, aparc_name))
        labels_dic = matlab_utils.matlab_cell_arrays_to_dict(matlab_fname)
        utils.save(labels_dic, op.join(BLENDER_ROOT_DIR, subject, 'labels_dic_{}_{}.pkl'.format(aparc_name, hemi)))


def save_labels_vertices(subject, aparc_name):
    labels_fnames = glob.glob(op.join(SUBJECTS_DIR, subject, 'label', aparc_name, '*.label'))
    labels_names, labels_vertices = defaultdict(list), defaultdict(list)
    for label_fname in labels_fnames:
        label = mne.read_label(label_fname)
        labels_names[label.hemi].append(label.name)
        labels_vertices[label.hemi].append(label.vertices)
    utils.save((labels_names, labels_vertices), op.join(BLENDER_ROOT_DIR, subject,
        'labels_vertices_{}.pkl'.format(aparc_name)))


def main(subject, aparc_name, neccesary_files, remote_subject_dir, overwrite_annotation=False,
         overwrite_morphing_labels=False, overwrite_hemis_srf=False, overwrite_labels_ply_files=False):
    # *) Prepare the local subject's folder
    local_subjects_dir = op.join(SUBJECTS_DIR, subject)
    utils.prepare_local_subjects_folder(neccesary_files, subject, remote_subject_dir, local_subjects_dir,
        print_traceback=False)
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
    calc_faces_verts_dic(subject)

    # *) Save the labels vertices for meg label plotting
    save_labels_vertices(subject, aparc_name)

    check_ply_files(op.join(SUBJECTS_DIR, subject, 'surf', '{}.pial.ply'),
                    op.join(BLENDER_ROOT_DIR, subject, '{}.pial.ply'))


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
    subject = 'mg78'
    aparc_name = 'laus250' # 'aprc250'
    print('subject: {}, atlas: {}'.format(subject, aparc_name))
    utils.make_dir(op.join(BLENDER_ROOT_DIR, subject))

    # Files needed in the local subject folder
    # subcortical_to_surface: mri/aseg.mgz, mri/norm.mgz
    # freesurfer_surface_to_blender_surface: surf/rh.pial, surf/lh.pial
    # convert_srf_files_to_ply: surf/lh.sphere.reg, surf/rh.sphere.reg
    neccesary_files = {'..': ['sub_cortical_codes.txt'], 'mri': ['aseg.mgz', 'norm.mgz', 'ribbon.mgz'],
        'surf': ['rh.pial', 'lh.pial', 'rh.sphere.reg', 'lh.sphere.reg', 'lh.white', 'rh.white']}

    # remote_subjects_dir = '/space/huygens/1/users/mia/subjects/{}_SurferOutput/'.format(subject.upper())
    # remote_subjects_dir = CACH_SUBJECT_DIR.format(subject=subject.upper())
    # remote_subjects_dir = op.join('/cluster/neuromind/tools/freesurfer', subject)
    remote_subject_dir = op.join('/autofs/space/lilli_001/users/DARPA-MEG/freesurfs', subject)
    # main(subject, aparc_name, neccesary_files, remote_subject_dir, overwrite_annotation=False, overwrite_morphing_labels=False,
    #      overwrite_hemis_srf=False, overwrite_labels_ply_files=False)
    save_labels_vertices(subject, aparc_name)
    print('finish!')