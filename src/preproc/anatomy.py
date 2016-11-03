import glob
import os
import os.path as op
import shutil
import traceback
from collections import Counter, defaultdict

import mne
import numpy as np
import scipy.io as sio
import nibabel as nib

from src.utils import labels_utils as lu
from src.utils import matlab_utils
from src.utils import utils
from src.utils import freesurfer_utils as fu
from src.mmvt_addon import colors_utils as cu


LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
FREE_SURFER_HOME = utils.get_link_dir(LINKS_DIR, 'freesurfer', 'FREESURFER_HOME')
MMVT_DIR = op.join(LINKS_DIR, 'mmvt')
os.environ['SUBJECTS_DIR'] = SUBJECTS_DIR
BRAINDER_SCRIPTS_DIR = op.join(utils.get_parent_fol(utils.get_parent_fol()), 'brainder_scripts')
ASEG_TO_SRF = op.join(BRAINDER_SCRIPTS_DIR, 'aseg2srf_cerebellum -s "{}"') # -o {}'
HEMIS = ['rh', 'lh']


def subcortical_segmentation_old(subject, overwrite_subcortical_objs=False):
    # Must source freesurfer. You need to have write permissions on SUBJECTS_DIR
    if not op.isfile(op.join(SUBJECTS_DIR, subject, 'mri', 'norm.mgz')):
        return False
    # script_fname = op.join(BRAINDER_SCRIPTS_DIR, 'aseg2srf')
    script_fname = op.join(BRAINDER_SCRIPTS_DIR, 'aseg2srf_cerebellum')
    if not op.isfile(script_fname):
        raise Exception('The subcortical segmentation script is missing! {}'.format(script_fname))
    if not utils.is_exe(script_fname):
        utils.set_exe_permissions(script_fname)

    # aseg2srf: For every subcortical region (8 10 11 12 13 16 17 18 26 47 49 50 51 52 53 54 58):
    # 1) mri_pretess: Changes region segmentation so that the neighbors of all voxels have a face in common
    # 2) mri_tessellate: Creates surface by tessellating
    # 3) mris_smooth: Smooth the new surface
    # 4) mris_convert: Convert the new surface into srf format

    function_output_fol = op.join(SUBJECTS_DIR, subject, 'mmvt', 'subcortical_objs')
    renamed_output_fol = op.join(SUBJECTS_DIR, subject, 'mmvt', 'subcortical')
    lookup = load_subcortical_lookup_table()
    obj_files = glob.glob(op.join(function_output_fol, '*.srf'))
    if len(obj_files) < len(lookup) or overwrite_subcortical_objs:
        # utils.delete_folder_files(function_output_fol)
        # # utils.delete_folder_files(aseg_to_srf_output_fol)
        # utils.delete_folder_files(renamed_output_fol)
        print('Trying to write into {}'.format(function_output_fol))
        utils.run_script(ASEG_TO_SRF.format(subject))
        # fu.
        # os.rename(aseg_to_srf_output_fol, function_output_fol)
    ply_files = glob.glob(op.join(renamed_output_fol, '*.ply'))
    if len(ply_files) < len(lookup) or overwrite_subcortical_objs:
        convert_and_rename_subcortical_files(subject, function_output_fol, renamed_output_fol, lookup)
    blender_dir = op.join(MMVT_DIR, subject, 'subcortical')
    if not op.isdir(blender_dir) or len(glob.glob(op.join(blender_dir, '*.ply'))) < len(ply_files):
        copy_subcorticals_to_blender(renamed_output_fol, subject)
    flag_ok = len(glob.glob(op.join(blender_dir, '*.ply'))) == len(lookup) and \
        len(glob.glob(op.join(blender_dir, '*.npz'))) == len(lookup)
    return flag_ok


def subcortical_segmentation(subject, overwrite_subcorticals=False):
    # 1) mri_pretess: Changes region segmentation so that the neighbors of all voxels have a face in common
    # 2) mri_tessellate: Creates surface by tessellating
    # 3) mris_smooth: Smooth the new surface
    # 4) mris_convert: Convert the new surface into srf format
    # Must source freesurfer. You need to have write permissions on SUBJECTS_DIR

    if not op.isfile(op.join(SUBJECTS_DIR, subject, 'mri', 'norm.mgz')):
        return False

    codes_file = op.join(MMVT_DIR, 'sub_cortical_codes.txt')
    subcortical_lookup = np.genfromtxt(codes_file, dtype=str, delimiter=',')
    # subcortical_lookup = np.array([['cerebellum_{}'.format(ind), ind] for ind in range(1, 18)])

    function_output_fol = op.join(SUBJECTS_DIR, subject, 'mmvt', 'subcortical_objs')
    # function_output_fol = op.join(SUBJECTS_DIR, subject, 'mmvt', 'subcortical_objs_Buckner2011_17')
    utils.make_dir(function_output_fol)
    renamed_output_fol = op.join(SUBJECTS_DIR, subject, 'mmvt', 'subcortical')
    # renamed_output_fol = op.join(SUBJECTS_DIR, subject, 'mmvt', 'subcortical_Buckner2011_17')
    utils.make_dir(renamed_output_fol)
    mask_fname = op.join(SUBJECTS_DIR, subject, 'mri', 'aseg.mgz')
    # mask_fname = op.join(SUBJECTS_DIR, subject, 'mri', 'Buckner2011_17Networks_MNI152_FreeSurferConformed1mm_TightMask.nii.gz')
    norm_fname = op.join(SUBJECTS_DIR, subject, 'mri', 'norm.mgz')
    lookup = load_subcortical_lookup_table()
    lookup = {int(val): name for name, val in zip(subcortical_lookup[:, 0], subcortical_lookup[:, 1])}
    obj_files = glob.glob(op.join(function_output_fol, '*.srf'))
    errors = []
    if len(obj_files) < len(lookup) or overwrite_subcorticals:
        # utils.delete_folder_files(function_output_fol)
        # # utils.delete_folder_files(aseg_to_srf_output_fol)
        # utils.delete_folder_files(renamed_output_fol)
        print('Trying to write into {}'.format(function_output_fol))
        for region_name, region_id in subcortical_lookup:
            ret = fu.aseg_to_srf(subject, SUBJECTS_DIR, function_output_fol, region_name, region_id, mask_fname, norm_fname,
                           overwrite_subcorticals)
            if not ret:
                errors.append(region_name)
        # utils.run_script(ASEG_TO_SRF.format(subject))
        # fu.
        # os.rename(aseg_to_srf_output_fol, function_output_fol)
    if len(errors) > 0:
        print('Errors: {}'.format(','.join(errors)))
    ply_files = glob.glob(op.join(renamed_output_fol, '*.ply'))
    if len(ply_files) < len(lookup) or overwrite_subcorticals:
        convert_and_rename_subcortical_files(subject, function_output_fol, renamed_output_fol, lookup)
    blender_dir = op.join(MMVT_DIR, subject, 'subcortical')
    if not op.isdir(blender_dir) or len(glob.glob(op.join(blender_dir, '*.ply'))) < len(ply_files):
        copy_subcorticals_to_blender(renamed_output_fol, subject)
    flag_ok = len(glob.glob(op.join(blender_dir, '*.ply'))) == len(lookup) and \
        len(glob.glob(op.join(blender_dir, '*.npz'))) == len(lookup)
    return flag_ok



def load_subcortical_lookup_table(fname='sub_cortical_codes.txt'):
    codes_file = op.join(MMVT_DIR, fname)
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
            verts, faces = utils.read_ply_file(op.join(new_fol, '{}.ply'.format(new_name)))
            np.savez(op.join(new_fol, '{}.npz'.format(new_name)), verts=verts, faces=faces)
    copy_subcorticals_to_blender(new_fol, subject)


def copy_subcorticals_to_blender(subcorticals_fol, subject):
    blender_fol = op.join(MMVT_DIR, subject, 'subcortical')
    if op.isdir(blender_fol):
        shutil.rmtree(blender_fol)
    shutil.copytree(subcorticals_fol, blender_fol)


def rename_cortical(lookup, fol, new_fol):
    ply_files = glob.glob(op.join(fol, '*.ply'))
    utils.delete_folder_files(new_fol)
    for ply_file in ply_files:
        base_name = op.basename(ply_file)
        num = int(base_name.split('.')[-2])
        hemi = base_name.split('.')[0]
        name = lookup[hemi].get(num, num)
        new_name = '{}-{}'.format(name, hemi)
        shutil.copy(ply_file, op.join(new_fol, '{}.ply'.format(new_name)))


def create_labels_lookup(subject, hemi, aparc_name):
    import mne.label
    annot_fname = op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot'.format(hemi, aparc_name))
    if not op.isfile(annot_fname):
        return {}
    annot, ctab, label_names = \
        mne.label._read_annot(annot_fname)
    lookup_key = 1
    lookup = {}
    for label_ind in range(len(label_names)):
        indices_num = len(np.where(annot == ctab[label_ind, 4])[0])
        if indices_num > 0 or label_names[label_ind].astype(str) == 'unknown':
            lookup[lookup_key] = label_names[label_ind].astype(str)
            lookup_key += 1
    return lookup


def freesurfer_surface_to_blender_surface(subject, hemi='both', overwrite=False):
    # verts, faces = {}, {}
    for hemi in utils.get_hemis(hemi):
        utils.make_dir(op.join(MMVT_DIR, subject, 'surf'))
        for surf_type in ['inflated', 'pial']:
            surf_name = op.join(SUBJECTS_DIR, subject, 'surf', '{}.{}'.format(hemi, surf_type))
            surf_wavefront_name = '{}.asc'.format(surf_name)
            surf_new_name = '{}.srf'.format(surf_name)
            hemi_ply_fname = '{}.ply'.format(surf_name)
            mmvt_hemi_ply_fname = op.join(MMVT_DIR, subject, 'surf', '{}.{}.ply'.format(hemi, surf_type))
            mmvt_hemi_npz_fname = op.join(MMVT_DIR, subject, 'surf', '{}.{}.npz'.format(hemi, surf_type))
            if overwrite or not op.isfile(mmvt_hemi_ply_fname) and not op.isfile(mmvt_hemi_npz_fname):
                print('{}: convert srf to asc'.format(hemi))
                utils.run_script('mris_convert {} {}'.format(surf_name, surf_wavefront_name))
                os.rename(surf_wavefront_name, surf_new_name)
                print('{}: convert asc to ply'.format(hemi))
                convert_hemis_srf_to_ply(subject, hemi, surf_type)
                # if surf_type == 'inflated':
                #     verts, faces = utils.read_ply_file(hemi_ply_fname)
                #     verts_offset = 5.5 if hemi == 'rh' else -5.5
                #     verts[:, 0] = verts[:, 0] + verts_offset
                #     utils.write_ply_file(verts, faces, '{}_offset.ply'.format(surf_name))
                if op.isfile(mmvt_hemi_ply_fname):
                    os.remove(mmvt_hemi_ply_fname)
                shutil.copy(hemi_ply_fname, mmvt_hemi_ply_fname)
            ply_fname = op.join(MMVT_DIR, subject, 'surf', '{}.{}.ply'.format(hemi, surf_type))
            if not op.isfile(mmvt_hemi_npz_fname):
                verts, faces = utils.read_ply_file(ply_fname)
                np.savez(mmvt_hemi_npz_fname, verts=verts, faces=faces)
                # verts[hemi], faces[hemi] = utils.read_ply_file(mmvt_hemi_npz_fname)
    # if not op.isfile(op.join(MMVT_DIR, subject, 'cortex.pial.npz')):
    #     faces['rh'] += np.max(faces['lh']) + 1
    #     verts_cortex = np.vstack((verts['lh'], verts['rh']))
    #     faces_cortex = np.vstack((faces['lh'], faces['rh']))
    #     utils.write_ply_file(verts_cortex, faces_cortex, op.join(MMVT_DIR, subject, 'cortex.pial.ply'))
    #     np.savez(op.join(MMVT_DIR, subject, 'cortex.pial.npz'), verts=verts_cortex, faces=faces_cortex)
    return utils.both_hemi_files_exist(op.join(MMVT_DIR, subject, 'surf', '{hemi}.pial.ply')) and \
           utils.both_hemi_files_exist(op.join(MMVT_DIR, subject, 'surf', '{hemi}.pial.npz')) and \
           utils.both_hemi_files_exist(op.join(MMVT_DIR, subject, 'surf', '{hemi}.inflated.ply')) and \
           utils.both_hemi_files_exist(op.join(MMVT_DIR, subject, 'surf', '{hemi}.inflated.npz'))
        # op.isfile(op.join(MMVT_DIR, subject, 'cortex.pial.ply')) and \
           # op.isfile(op.join(MMVT_DIR, subject, 'cortex.pial.npz'))


def convert_hemis_srf_to_ply(subject, hemi='both', surf_type='pial'):
    for hemi in utils.get_hemis(hemi):
        ply_file = utils.srf2ply(op.join(SUBJECTS_DIR, subject, 'surf', '{}.{}.srf'.format(hemi, surf_type)),
                                 op.join(SUBJECTS_DIR, subject, 'surf', '{}.{}.ply'.format(hemi, surf_type)))
        utils.make_dir(op.join(MMVT_DIR, subject))
        # shutil.copyfile(ply_file, op.join(MMVT_DIR, subject, 'surf', '{}.{}.ply'.format(hemi, surf_type)))


def save_hemis_curv(subject, atlas):
    out_curv_file = op.join(MMVT_DIR, subject, 'surf', '{hemi}.curv.npy')
    # out_border_file = op.join(MMVT_DIR, subject, 'surf', '{hemi}.curv.borders.npy')
    # if utils.both_hemi_files_exist(out_file):
    #     return True
    for hemi in utils.HEMIS:
        # Load in curvature values from the ?h.curv file.
        if not op.isfile(out_curv_file.format(hemi=hemi)):
            curv_path = op.join(SUBJECTS_DIR, subject, 'surf', '{}.curv'.format(hemi))
            curv = nib.freesurfer.read_morph_data(curv_path)
            bin_curv = np.array(curv > 0, np.int)
            np.save(out_curv_file.format(hemi=hemi), bin_curv)
        else:
            bin_curv = np.load(out_curv_file.format(hemi=hemi))
        labels_fol = op.join(MMVT_DIR, subject, 'surf', '{}_{}_curves'.format(atlas, hemi))
        utils.make_dir(labels_fol)
        labels = lu.read_labels(subject, SUBJECTS_DIR, atlas, hemi=hemi)
        for label in labels:
            labels_curv = bin_curv[label.vertices]
            np.save(op.join(labels_fol, '{}_curv.npy'.format(label.name)), labels_curv)
    return utils.both_hemi_files_exist(out_curv_file) # and utils.both_hemi_files_exist(out_border_file)


def calc_faces_verts_dic(subject, atlas, overwrite=False):
    # hemis_plus = HEMIS + ['cortex']
    ply_files = [op.join(MMVT_DIR, subject, 'surf', '{}.pial.npz'.format(hemi)) for hemi in utils.HEMIS]
    out_files = [op.join(MMVT_DIR, subject, 'faces_verts_{}.npy'.format(hemi)) for hemi in utils.HEMIS]
    subcortical_plys = glob.glob(op.join(MMVT_DIR, subject, 'subcortical', '*.ply'))
    errors = []
    if len(subcortical_plys) > 0:
        faces_verts_dic_fnames = [op.join(MMVT_DIR, subject, 'subcortical', '{}_faces_verts.npy'.format(
                utils.namebase(ply))) for ply in subcortical_plys]
        ply_files.extend(subcortical_plys)
        out_files.extend(faces_verts_dic_fnames)
    for hemi in utils.HEMIS:
        labels_plys = glob.glob(op.join(MMVT_DIR, subject, '{}.pial.{}'.format(atlas, hemi), '*.ply'))
        if len(labels_plys) > 0:
            faces_verts_dic_fnames = [op.join(MMVT_DIR, subject, '{}.pial.{}'.format(atlas, hemi), '{}_faces_verts.npy'.format(
                utils.namebase(ply))) for ply in labels_plys]
            ply_files.extend(labels_plys)
            out_files.extend(faces_verts_dic_fnames)

    for ply_file, out_file in zip(ply_files, out_files):
        if not overwrite and op.isfile(out_file):
            # print('{} already exist.'.format(out_file))
            continue
        # ply_file = op.join(SUBJECTS_DIR, subject,'surf', '{}.pial.ply'.format(hemi))
        # print('preparing a lookup table for {}'.format(ply_file))
        verts, faces = utils.read_ply_file(ply_file)
        _faces = faces.ravel()
        print('{}: verts: {}, faces: {}, faces ravel: {}'.format(utils.namebase(ply_file), verts.shape[0], faces.shape[0], len(_faces)))
        faces_arg_sort = np.argsort(_faces)
        faces_sort = np.sort(_faces)
        faces_count = Counter(faces_sort)
        max_len = max([v for v in faces_count.values()])
        lookup = np.ones((verts.shape[0], max_len)) * -1
        diff = np.diff(faces_sort)
        n = 0
        for ind, (k, v) in enumerate(zip(faces_sort, faces_arg_sort)):
            lookup[k, n] = v
            n = 0 if ind < len(diff) and diff[ind] > 0 else n+1
        # print('writing {}'.format(out_file))
        np.save(out_file, lookup.astype(np.int))
        print('{} max lookup val: {}'.format(utils.namebase(ply_file), int(np.max(lookup))))
        if len(_faces) != int(np.max(lookup)) + 1:
            errors[utils.namebase(ply_file)] = 'Wrong values in lookup table! ' + \
                'faces ravel: {}, max looup val: {}'.format(len(_faces), int(np.max(lookup)))
    if len(errors) > 0:
        for k, message in errors.items():
            print('{}: {}'.format(k, message))
    return len(errors) == 0


def check_ply_files(subject):
    ply_subject = op.join(SUBJECTS_DIR, subject, 'surf', '{}.pial.ply')
    npz_subject = op.join(SUBJECTS_DIR, subject, 'surf', '{}.pial.npz')
    ply_blender = op.join(MMVT_DIR, subject, 'surf', '{}.pial.ply')
    npz_blender = op.join(MMVT_DIR, subject, 'surf', '{}.pial.npz')
    ok = True
    for hemi in HEMIS:
        # print('reading {}'.format(ply_subject.format(hemi)))
        verts1, faces1 = utils.read_ply_file(ply_subject.format(hemi), npz_subject.format(hemi))
        # print('reading {}'.format(ply_blender.format(hemi)))
        verts2, faces2 = utils.read_ply_file(ply_blender.format(hemi), npz_blender.format(hemi))
        print('vertices: ply: {}, blender: {}'.format(verts1.shape[0], verts2.shape[0]))
        print('faces: ply: {}, blender: {}'.format(faces1.shape[0], faces2.shape[0]))
        ok = ok and verts1.shape[0] == verts2.shape[0] and faces1.shape[0]==faces2.shape[0]
    return ok


def convert_perecelated_cortex(subject, aparc_name, surf_type='pial', overwrite_ply_files=False, hemi='both'):
    lookup = {}
    for hemi in utils.get_hemis(hemi):
        lookup[hemi] = create_labels_lookup(subject, hemi, aparc_name)
        if len(lookup[hemi]) == 0:
            continue
        srf_fol = op.join(SUBJECTS_DIR, subject,'{}.{}.{}'.format(aparc_name, surf_type, hemi))
        ply_fol = op.join(SUBJECTS_DIR, subject,'{}_{}_{}_ply'.format(aparc_name, surf_type, hemi))
        blender_fol = op.join(MMVT_DIR, subject,'{}.{}.{}'.format(aparc_name, surf_type, hemi))
        utils.convert_srf_files_to_ply(srf_fol, overwrite_ply_files)
        rename_cortical(lookup, srf_fol, ply_fol)
        if surf_type == 'inflated':
            for ply_fname in glob.glob(op.join(ply_fol, '*.ply')):
                verts, faces = utils.read_ply_file(ply_fname)
                verts_offset = 5.5 if hemi == 'rh' else -5.5
                verts[:, 0] = verts[:, 0] + verts_offset
                utils.write_ply_file(verts, faces, ply_fname)
        utils.rmtree(blender_fol)
        shutil.copytree(ply_fol, blender_fol)
    return lookup


def create_annotation_from_fsaverage(subject, aparc_name='aparc250', fsaverage='fsaverage', remote_subject_dir='',
        overwrite_annotation=False, overwrite_morphing=False, do_solve_labels_collisions=False,
        morph_labels_from_fsaverage=True, fs_labels_fol='', n_jobs=6):
    annotations_exist = np.all([op.isfile(op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot'.format(hemi,
        aparc_name))) for hemi in HEMIS])
    if not annotations_exist:
        utils.make_dir(op.join(SUBJECTS_DIR, subject, 'label'))
        remote_annotations_exist = np.all([op.isfile(op.join(remote_subject_dir, 'label', '{}.{}.annot'.format(
            hemi, aparc_name))) for hemi in HEMIS])
        if remote_annotations_exist:
            for hemi in HEMIS:
                shutil.copy(op.join(remote_subject_dir, 'label', '{}.{}.annot'.format(hemi, aparc_name)),
                            op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot'.format(hemi, aparc_name)))
            return True
    existing_freesurfer_annotations = ['aparc.DKTatlas40.annot', 'aparc.annot', 'aparc.a2009s.annot']
    if '{}.annot'.format(aparc_name) in existing_freesurfer_annotations:
        morph_labels_from_fsaverage = False
        do_solve_labels_collisions = False
        if not annotations_exist:
            utils.make_dir(op.join(SUBJECTS_DIR, subject, 'label'))
            annotations_exist = fu.create_annotation_file(subject, aparc_name, subjects_dir=SUBJECTS_DIR, freesurfer_home=FREE_SURFER_HOME)
    if morph_labels_from_fsaverage:
        lu.morph_labels_from_fsaverage(subject, SUBJECTS_DIR, MMVT_DIR, aparc_name, n_jobs=n_jobs,
            fsaverage=fsaverage, overwrite=overwrite_morphing, fs_labels_fol=fs_labels_fol)
    if do_solve_labels_collisions:
        solve_labels_collisions(subject, aparc_name, fsaverage, n_jobs)
    # Note that using the current mne version this code won't work, because of collissions between hemis
    # You need to change the mne.write_labels_to_annot code for that.
    if overwrite_annotation or not annotations_exist:
        try:
            utils.labels_to_annot(subject, SUBJECTS_DIR, aparc_name, overwrite=overwrite_annotation)
        except:
            print("Can't write labels to annotation! Trying to solve labels collision")
            print(traceback.format_exc())
            solve_labels_collisions(subject, aparc_name, fsaverage, n_jobs)
        try:
            utils.labels_to_annot(subject, SUBJECTS_DIR, aparc_name, overwrite=overwrite_annotation)
        except:
            print("Can't write labels to annotation! Solving the labels collision didn't help...")
            print(traceback.format_exc())
    return utils.both_hemi_files_exist(op.join(
        SUBJECTS_DIR, subject, 'label', '{}.{}.annot'.format('{hemi}', aparc_name)))


def solve_labels_collisions(subject, aparc_name, fsaverage, n_jobs):
    backup_labels_fol = '{}_before_solve_collision'.format(aparc_name, fsaverage)
    lu.solve_labels_collision(subject, SUBJECTS_DIR, aparc_name, backup_labels_fol, n_jobs)
    lu.backup_annotation_files(subject, SUBJECTS_DIR, aparc_name)


def parcelate_cortex(subject, aparc_name, overwrite=False, overwrite_ply_files=False, minimum_labels_num=50):
    dont_do_anything = True
    ret = {'pial':True, 'inflated':True}
    for surface_type in ['pial', 'inflated']:
        files_exist = True
        for hemi in HEMIS:
            blender_labels_fol = op.join(MMVT_DIR, subject,'{}.{}.{}'.format(aparc_name, surface_type, hemi))
            files_exist = files_exist and op.isdir(blender_labels_fol) and \
                len(glob.glob(op.join(blender_labels_fol, '*.ply'))) > minimum_labels_num

        # if surface_type == 'inflated':
        if overwrite or not files_exist:
            dont_do_anything = False
            matlab_command = op.join(BRAINDER_SCRIPTS_DIR, 'splitting_cortical.m')
            matlab_command = "'{}'".format(matlab_command)
            sio.savemat(op.join(BRAINDER_SCRIPTS_DIR, 'params.mat'),
                mdict={'subject': subject, 'aparc':aparc_name, 'subjects_dir': SUBJECTS_DIR,
                       'scripts_dir': BRAINDER_SCRIPTS_DIR, 'freesurfer_home': FREE_SURFER_HOME,
                       'surface_type': surface_type})
            cmd = 'matlab -nodisplay -nosplash -nodesktop -r "run({}); exit;"'.format(matlab_command)
            utils.run_script(cmd)
            # convert the  obj files to ply
            lookup = convert_perecelated_cortex(subject, aparc_name, surface_type, overwrite_ply_files)
            matlab_labels_vertices = True
            if surface_type == 'pial':
                matlab_labels_vertices = save_matlab_labels_vertices(subject, aparc_name)

            labels_num = sum([len(lookup[hemi]) for hemi in HEMIS])
            labels_files_num = sum([len(glob.glob(op.join(MMVT_DIR, subject,'{}.{}.{}'.format(
                aparc_name, surface_type, hemi), '*.ply'))) for hemi in HEMIS])
            labels_dic_fname = op.join(MMVT_DIR, subject,'labels_dic_{}_{}.pkl'.format(aparc_name, hemi))
            print('labels_files_num == labels_num: {}'.format(labels_files_num == labels_num))
            print('isfile(labels_dic_fname): {}'.format(op.isfile(labels_dic_fname)))
            print('matlab_labels_vertices files: {}'.format(matlab_labels_vertices))
            ret[surface_type] = labels_files_num == labels_num and op.isfile(labels_dic_fname) and matlab_labels_vertices
    if dont_do_anything:
        return True
    else:
        return ret['pial'] and ret['inflated']


def save_matlab_labels_vertices(subject, aparc_name):
    for hemi in HEMIS:
        matlab_fname = op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot_labels.m'.format(hemi, aparc_name))
        if op.isfile(matlab_fname):
            labels_dic = matlab_utils.matlab_cell_arrays_to_dict(matlab_fname)
            utils.save(labels_dic, op.join(MMVT_DIR, subject, 'labels_dic_{}_{}.pkl'.format(aparc_name, hemi)))
        else:
            return False
    return True


def save_labels_vertices(subject, aparc_name):
    annot_fname_temp = op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot'.format('{hemi}', aparc_name))
    if not utils.hemi_files_exists(annot_fname_temp):
        pass
    labels_fnames = glob.glob(op.join(SUBJECTS_DIR, subject, 'label', aparc_name, '*.label'))
    if len(labels_fnames) > 0:
        labels = []
        for label_fname in labels_fnames:
            label = mne.read_label(label_fname)
            labels.append(label)
    else:
        # Read from the annotation file
        labels = utils.read_labels_from_annot(subject, aparc_name, SUBJECTS_DIR)
    labels_names, labels_vertices = defaultdict(list), defaultdict(list)
    for label in labels:
        labels_names[label.hemi].append(label.name)
        labels_vertices[label.hemi].append(label.vertices)
    output_fname = op.join(MMVT_DIR, subject, 'labels_vertices_{}.pkl'.format(aparc_name))
    utils.save((labels_names, labels_vertices), output_fname)
    return op.isfile(output_fname)


@utils.timeit
def create_spatial_connectivity(subject):
    try:
        verts_neighbors_fname = op.join(MMVT_DIR, subject, 'verts_neighbors_{hemi}.pkl')
        connectivity_fname = op.join(MMVT_DIR, subject, 'spatial_connectivity.pkl')
        if utils.both_hemi_files_exist(verts_neighbors_fname) and op.isfile(verts_neighbors_fname):
            return True
        connectivity_per_hemi = {}
        for hemi in utils.HEMIS:
            neighbors = defaultdict(list)
            d = np.load(op.join(MMVT_DIR, subject, 'surf', '{}.pial.npz'.format(hemi)))
            connectivity_per_hemi[hemi] = mne.spatial_tris_connectivity(d['faces'])
            rows, cols = connectivity_per_hemi[hemi].nonzero()
            for ind in range(len(rows)):
                neighbors[rows[ind]].append(cols[ind])
            utils.save(neighbors, verts_neighbors_fname.format(hemi=hemi))
        utils.save(connectivity_per_hemi, connectivity_fname)
        success = True
    except:
        print('Error in create_spatial_connectivity!')
        print(traceback.format_exc())
        success = False
    return success


#
# @utils.timeit
# def calc_verts_neighbors_lookup(subject):
#     import time
#     out_file = op.join(MMVT_DIR, subject, 'verts_neighbors_{hemi}.pkl')
#     if utils.both_hemi_files_exist(out_file):
#         return True
#     for hemi in utils.HEMIS:
#         neighbors = {}
#         verts, faces = utils.read_pial_npz(subject, MMVT_DIR, hemi)
#         now = time.time()
#         for vert_ind in range(verts.shape[0]):
#             utils.time_to_go(now, vert_ind, verts.shape[0], 1000)
#             neighbors[vert_ind] = set(faces[np.where(faces == vert_ind)[0]].ravel())
#         utils.save(neighbors, out_file.format(hemi=hemi))
#     return utils.both_hemi_files_exist(out_file)


def calc_labels_center_of_mass(subject, atlas, read_from_annotation=True, surf_name='pial', labels_fol='', labels=None):
    import csv
    # if (read_from_annotation):
    #     labels = mne.read_labels_from_annot(subject, atlas, 'both', surf_name, subjects_dir=SUBJECTS_DIR)
    #     if len(labels) == 0:
    #         print('No labels were found in {} annotation file!'.format(atlas))
    # else:
    #     labels = []
    #     if labels_fol == '':
    #         labels_fol = op.join(SUBJECTS_DIR, subject, 'label', atlas)
    #     for label_file in glob.glob(op.join(labels_fol, '*.label')):
    #         label = mne.read_label(label_file)
    #         labels.append(label)
    #     if len(labels) == 0:
    #         print('No labels were found in {}!'.format(labels_fol))
    labels = lu.read_labels(subject, SUBJECTS_DIR, atlas)
    if len(labels) > 0:
        center_of_mass = lu.calc_center_of_mass(labels)
        with open(op.join(SUBJECTS_DIR, subject, 'label', '{}_center_of_mass.csv'.format(atlas)), 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for label in labels:
                writer.writerow([label.name, *center_of_mass[label.name]])
        com_fname = op.join(SUBJECTS_DIR, subject, 'label', '{}_center_of_mass.pkl'.format(atlas))
        blend_fname = op.join(MMVT_DIR, subject, '{}_center_of_mass.pkl'.format(atlas))
        utils.save(center_of_mass, com_fname)
        shutil.copyfile(com_fname, blend_fname)
    return len(labels) > 0 and op.isfile(com_fname) and op.isfile(blend_fname)


def save_labels_coloring(subject, atlas, n_jobs=2):
    ret = False
    coloring_dir = op.join(MMVT_DIR, subject, 'coloring')
    utils.make_dir(coloring_dir)
    coloring_fname = op.join(coloring_dir, 'labels_{}_coloring.csv'.format(atlas))
    coloring_names_fname = op.join(coloring_dir, 'labels_{}_colors_names.csv'.format(atlas))
    try:
        labels = lu.read_labels(subject, SUBJECTS_DIR, atlas, n_jobs=n_jobs)
        colors_rgb_and_names = cu.get_distinct_colors_and_names()
        labels_colors_rgb, labels_colors_names = {}, {}
        for label in labels:
            label_inv_name = lu.get_label_hemi_invariant_name(label.name)
            if label_inv_name not in labels_colors_rgb:
                labels_colors_rgb[label_inv_name], labels_colors_names[label_inv_name] = next(colors_rgb_and_names)
        with open(coloring_fname, 'w') as colors_file, open(coloring_names_fname, 'w') as col_names_file:
            for label in labels:
                label_inv_name = lu.get_label_hemi_invariant_name(label.name)
                color_rgb = labels_colors_rgb[label_inv_name]
                color_name = labels_colors_names[label_inv_name]
                colors_file.write('{},{},{},{}\n'.format(label.name, *color_rgb))
                col_names_file.write('{},{}\n'.format(label.name, color_name))
        ret = op.isfile(coloring_fname)
    except:
        print('Error in save_labels_coloring!')
        print(traceback.format_exc())
    return ret

# def find_hemis_boarders(subject):
#     from scipy.spatial.distance import cdist
#     verts = {}
#     for hemi in utils.HEMIS:
#         ply_file = op.join(SUBJECTS_DIR)
#         verts[hemi], _ = utils.read_ply_file(op.join(SUBJECTS_DIR, subject, 'surf', '{}.pial.ply'.format(hemi)))
#     dists = cdist(verts['rh'], verts['lh'])


def prepare_local_subjects_folder(subject, args):
    return utils.prepare_local_subjects_folder(
        args.necessary_files, subject, args.remote_subject_dir, SUBJECTS_DIR,
        args.sftp, args.sftp_username, args.sftp_domain, args.sftp_password,
        args.overwrite_fs_files, args.print_traceback)


def main(subject, args):
    flags = dict()
    utils.make_dir(op.join(SUBJECTS_DIR, subject, 'mmvt'))
    args.remote_subject_dir = utils.build_remote_subject_dir(args.remote_subject_dir, subject)

    if utils.should_run(args, 'prepare_local_subjects_folder'):
        # *) Prepare the local subject's folder
        flags['prepare_local_subjects_folder'] = prepare_local_subjects_folder(subject, args)
        if not flags['prepare_local_subjects_folder'] and not args.ignore_missing:
            return flags

    if utils.should_run(args, 'freesurfer_surface_to_blender_surface'):
        # *) convert rh.pial and lh.pial to rh.pial.ply and lh.pial.ply
        flags['hemis'] = freesurfer_surface_to_blender_surface(subject, overwrite=args.overwrite_hemis_srf)


    if utils.should_run(args, 'create_annotation_from_fsaverage'):
        # *) Create annotation file from fsaverage
        flags['annot'] = create_annotation_from_fsaverage(
            subject, args.atlas, args.fsaverage, args.remote_subject_dir, args.overwrite_annotation, args.overwrite_morphing_labels,
            args.solve_labels_collisions, args.morph_labels_from_fsaverage, args.fs_labels_fol, args.n_jobs)

    if utils.should_run(args, 'parcelate_cortex'):
        # *) Calls Matlab 'splitting_cortical.m' script
        flags['parc_cortex'] = parcelate_cortex(
            subject, args.atlas, args.overwrite_labels_ply_files, args.overwrite_ply_files)

    if utils.should_run(args, 'subcortical_segmentation'):
        # *) Create srf files for subcortical structures
        # !!! Should source freesurfer !!!
        # Remember that you need to have write permissions on SUBJECTS_DIR!!!
        flags['subcortical'] = subcortical_segmentation(subject, args.overwrite_subcorticals)

    if utils.should_run(args, 'calc_faces_verts_dic'):
        # *) Create a dictionary for verts and faces for both hemis
        flags['faces_verts'] = calc_faces_verts_dic(subject, args.atlas, args.overwrite_faces_verts)

    if utils.should_run(args, 'save_labels_vertices'):
        # *) Save the labels vertices for meg label plotting
        flags['labels_vertices'] = save_labels_vertices(subject, args.atlas)

    # if utils.should_run(args, 'calc_verts_neighbors_lookup'):
    #     *) Calc the vertices neighbors lookup
        # flags['calc_verts_neighbors_lookup'] = calc_verts_neighbors_lookup(subject)

    if utils.should_run(args, 'save_hemis_curv'):
        # *) Save the hemis curvs for the inflated brain
        flags['save_hemis_curv'] = save_hemis_curv(subject, args.atlas)

    # if utils.should_run(args, 'calc_verts_neighbors_lookup'):
    #     flags['calc_verts_neighbors_lookup'] = calc_verts_neighbors_lookup(subject)

    if utils.should_run(args, 'create_spatial_connectivity'):
        # *) Create the subject's connectivity
        flags['connectivity'] = create_spatial_connectivity(subject)

    # if utils.should_run(args, 'check_ply_files'):
    #     # *) Check the pial surfaces
    #     flags['ply_files'] = check_ply_files(subject)

    if utils.should_run(args, 'calc_labels_center_of_mass'):
        # *) Calc the labels center of mass
        flags['center_of_mass'] = calc_labels_center_of_mass(subject, args.atlas, args.surf_name)

    if utils.should_run(args, 'save_labels_coloring'):
        # *) Save a coloring file for the atlas's labels
        flags['save_labels_coloring'] = save_labels_coloring(subject, args.atlas, args.n_jobs)

    # for flag_type, val in flags.items():
    #     print('{}: {}'.format(flag_type, val))
    return flags


def run_on_subjects(args):
    subjects_flags, subjects_errors = {}, {}
    args.sftp_password = utils.get_sftp_password(
        args.subject, SUBJECTS_DIR, args.necessary_files, args.sftp_username, args.overwrite_fs_files) \
        if args.sftp else ''
    for subject in args.subject:
        utils.make_dir(op.join(MMVT_DIR, subject, 'mmvt'))
        # os.chdir(op.join(SUBJECTS_DIR, subject, 'mmvt'))
        try:
            print('*******************************************')
            print('subject: {}, atlas: {}'.format(subject, args.atlas))
            print('*******************************************')
            flags = main(subject, args)
            subjects_flags[subject] = flags
        except:
            subjects_errors[subject] = traceback.format_exc()
            print('Error in subject {}'.format(subject))
            print(traceback.format_exc())

    errors = defaultdict(list)
    for subject, flags in subjects_flags.items():
        print('subject {}:'.format(subject))
        for flag_type, val in flags.items():
            print('{}: {}'.format(flag_type, val))
            if not val:
                errors[subject].append(flag_type)
    print('Errors:')
    for subject, error in errors.items():
        print('{}: {}'.format(subject, error))


def read_cmd_args(argv=None):
    import argparse
    from src.utils import args_utils as au
    parser = argparse.ArgumentParser(description='MMVT anatomy preprocessing')
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-a', '--atlas', help='atlas name', required=False, default='aparc.DKTatlas40')
    parser.add_argument('-f', '--function', help='functions to run', required=False, default='all', type=au.str_arr_type)
    parser.add_argument('--exclude', help='functions not to run', required=False, default='', type=au.str_arr_type)
    parser.add_argument('--ignore_missing', help='ignore missing files', required=False, default=0, type=au.is_true)
    parser.add_argument('--fsaverage', help='fsaverage', required=False, default='fsaverage')
    parser.add_argument('--remote_subject_dir', help='remote_subject_dir', required=False, default='')
    parser.add_argument('--surf_name', help='surf_name', required=False, default='pial')
    parser.add_argument('--overwrite', help='overwrite', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_subcorticals', help='overwrite', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_fs_files', help='overwrite freesurfer files', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_annotation', help='overwrite_annotation', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_morphing_labels', help='overwrite_morphing_labels', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_hemis_srf', help='overwrite_hemis_srf', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_labels_ply_files', help='overwrite_labels_ply_files', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_faces_verts', help='overwrite_faces_verts', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_ply_files', help='overwrite_ply_files', required=False, default=0, type=au.is_true)
    parser.add_argument('--solve_labels_collisions', help='solve_labels_collisions', required=False, default=0, type=au.is_true)
    parser.add_argument('--morph_labels_from_fsaverage', help='morph_labels_from_fsaverage', required=False, default=1, type=au.is_true)
    parser.add_argument('--fs_labels_fol', help='fs_labels_fol', required=False, default='')
    parser.add_argument('--sftp', help='copy subjects files over sftp', required=False, default=0, type=au.is_true)
    parser.add_argument('--sftp_username', help='sftp username', required=False, default='')
    parser.add_argument('--sftp_domain', help='sftp domain', required=False, default='')
    parser.add_argument('--print_traceback', help='print_traceback', required=False, default=1, type=au.is_true)
    parser.add_argument('--n_jobs', help='cpu num', required=False, default=-1)
    args = utils.Bag(au.parse_parser(parser, argv))
    args.necessary_files = {'mri': ['aseg.mgz', 'norm.mgz', 'ribbon.mgz', 'T1.mgz'],
        'surf': ['rh.pial', 'lh.pial', 'rh.inflated', 'lh.inflated', 'lh.curv', 'rh.curv', 'rh.sphere.reg', 'lh.sphere.reg', 'lh.white', 'rh.white', 'rh.smoothwm','lh.smoothwm'],
        'mri:transforms' : ['talairach.xfm']}
    args.n_jobs = utils.get_n_jobs(args.n_jobs)
    if args.overwrite:
        args.overwrite_annotation = True
        args.overwrite_morphing_labels = True
        args.overwrite_hemis_srf = True
        args.overwrite_labels_ply_files = True
        args.overwrite_faces_verts = True
        args.overwrite_fs_files = True
    print(args)
    return args


if __name__ == '__main__':
    # ******************************************************************
    # Be sure that you have matlab installed on your machine,
    # and you can run it from the terminal by just calling 'matlab'
    # Some of the functions are using freesurfer, so if you want to
    # run main, you need to source freesurfer.
    # ******************************************************************
    if os.environ.get('FREESURFER_HOME', '') == '':
        print('Source freesurfer and rerun')
    else:
        args = read_cmd_args()
        run_on_subjects(args)
        print('finish!')

    # fs_labels_fol = '/space/lilli/1/users/DARPA-Recons/fscopy/label/arc_april2016'
    # remote_subjects_dir = '/space/huygens/1/users/mia/subjects/{}_SurferOutput/'.format(subject.upper())
    # remote_subjects_dir = CACH_SUBJECT_DIR.format(subject=subject.upper())
    # remote_subjects_dir = op.join('/cluster/neuromind/tools/freesurfer', subject)
    # remote_subjects_dir = op.join('/autofs/space/lilli_001/users/DARPA-MEG/freesurfs')
    # subjects = ['mg78', 'mg82'] #set(utils.get_all_subjects(SUBJECTS_DIR, 'mg', '_')) - set(['mg96'])

