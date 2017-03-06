import glob
import os
import os.path as op
import shutil
import traceback
from collections import defaultdict

import mne
import numpy as np
import scipy.io as sio
import nibabel as nib

from src.utils import labels_utils as lu
from src.utils import matlab_utils
from src.utils import utils
from src.utils import freesurfer_utils as fu
from src.mmvt_addon import colors_utils as cu
from src.utils import args_utils as au
from src.utils import preproc_utils as pu


SUBJECTS_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()
BRAINDER_SCRIPTS_DIR = op.join(utils.get_parent_fol(utils.get_parent_fol()), 'brainder_scripts')
HEMIS = ['rh', 'lh']


def cerebellum_segmentation(subject, remote_subject_dir, args, model='Buckner2011_7Networks', subregions_num=7):
    # For cerebellum parcellation
    # http://www.freesurfer.net/fswiki/CerebellumParcellation_Buckner2011
    # First download the mask file and put it in the subject's mri folder
    # https://mail.nmr.mgh.harvard.edu/pipermail//freesurfer/2016-June/046380.html
    loose_tight = 'loose' if args.cerebellum_segmentation_loose else 'tight'
    bunker_atlas_fname = op.join(MMVT_DIR, 'templates', 'Buckner2011_atlas_{}_{}.nii.gz'.format(
        subregions_num, loose_tight))
    if not op.isfile(bunker_atlas_fname):
        print("Can't find Bunker atlas! Should be here: {}".format(bunker_atlas_fname))
        return False

    warp_buckner_atlas_fname = fu.warp_buckner_atlas_output_fname(subject, SUBJECTS_DIR,  subregions_num, loose_tight)
    if not op.isfile(warp_buckner_atlas_fname):
        prepare_subject_folder(subject, remote_subject_dir, args, {'mri:transforms' : ['talairach.m3z']})
        fu.warp_buckner_atlas(subject, SUBJECTS_DIR, bunker_atlas_fname, warp_buckner_atlas_fname)
    if not op.isfile(warp_buckner_atlas_fname):
        print('mask file does not exist! {}'.format(warp_buckner_atlas_fname))
        return False

    mask_data = nib.load(warp_buckner_atlas_fname).get_data()
    unique_values_num = len(np.unique(mask_data))
    if unique_values_num < subregions_num:
        print('subregions_num ({}) is bigger than the unique values num in the mask file ({})!'.format(
            subregions_num, unique_values_num))
        return False
    warp_buckner_hemis_atlas_fname = '{}_hemis.{}'.format(
        warp_buckner_atlas_fname.split('.')[0], '.'.join(warp_buckner_atlas_fname.split('.')[1:]))
    new_maks_fname = op.join(SUBJECTS_DIR, subject, 'mri', warp_buckner_hemis_atlas_fname)
    subregions_num = split_cerebellum_hemis(subject, warp_buckner_atlas_fname, new_maks_fname, subregions_num)
    subcortical_lookup = np.array([['{}_cerebellum_{}'.format(
        'right' if ind <= subregions_num/2 else 'left',  ind if ind <= subregions_num/2 else int(ind - subregions_num/2)), ind] for ind in range(1, subregions_num + 1)])
    lookup = {int(val): name for name, val in zip(subcortical_lookup[:, 0], subcortical_lookup[:, 1])}
    mmvt_subcorticals_fol_name = 'cerebellum'
    ret = subcortical_segmentation(subject, args.overwrite_subcorticals, model, lookup, warp_buckner_hemis_atlas_fname,
                                    mmvt_subcorticals_fol_name, subject)
    return ret


def split_cerebellum_hemis(subject, mask_fname, new_maks_name, subregions_num):
    import time
    if op.isfile(new_maks_name):
        return subregions_num * 2
    mask = nib.load(mask_fname)
    mask_data = mask.get_data()
    aseg_data = nib.load(op.join(SUBJECTS_DIR, subject, 'mri', 'aseg.mgz')).get_data()
    new_mask_data = np.zeros(mask_data.shape)
    lookup = utils.read_freesurfer_lookup_table(return_dict=True)
    inv_lookup = {name:val for val,name in lookup.items()}
    right_cerebral = inv_lookup['Right-Cerebellum-Cortex']
    left_cerebral = inv_lookup['Left-Cerebellum-Cortex']
    now = time.time()
    for seg_id in range(1, subregions_num + 1):
        utils.time_to_go(now, seg_id-1, subregions_num, 1)
        seg_inds = np.where(mask_data == seg_id)
        aseg_segs = aseg_data[seg_inds]
        seg_inds = np.vstack((seg_inds[0], seg_inds[1], seg_inds[2]))
        right_inds = seg_inds[:, np.where(aseg_segs == right_cerebral)[0]]
        left_inds = seg_inds[:, np.where(aseg_segs == left_cerebral)[0]]
        new_mask_data[right_inds[0], right_inds[1], right_inds[2]] = seg_id
        new_mask_data[left_inds[0], left_inds[1], left_inds[2]] = seg_id + subregions_num
    new_mask = nib.Nifti1Image(new_mask_data, affine=mask.get_affine())
    print('Saving new mask to {}'.format(new_maks_name))
    nib.save(new_mask, new_maks_name)
    return subregions_num * 2


def subcortical_segmentation(subject, overwrite_subcorticals=False, model='subcortical', lookup=None,
                             mask_name='aseg.mgz', mmvt_subcorticals_fol_name='subcortical',
                             template_subject='', norm_name='norm.mgz', overwrite=True):
    # 1) mri_pretess: Changes region segmentation so that the neighbors of all voxels have a face in common
    # 2) mri_tessellate: Creates surface by tessellating
    # 3) mris_smooth: Smooth the new surface
    # 4) mris_convert: Convert the new surface into srf format

    template_subject = subject if template_subject == '' else template_subject
    norm_fname = op.join(SUBJECTS_DIR, template_subject, 'mri', norm_name)
    if not op.isfile(norm_fname):
        print('norm file does not exist! {}'.format(norm_fname))
        return False

    mask_fname = op.join(SUBJECTS_DIR, template_subject, 'mri', mask_name)
    if not op.isfile(mask_fname):
        print('mask file does not exist! {}'.format(mask_fname))
        return False

    codes_file = op.join(MMVT_DIR, 'sub_cortical_codes.txt')
    if not op.isfile(codes_file):
        print('subcortical codes file does not exist! {}'.format(codes_file))
        return False

    # subcortical_lookup = np.genfromtxt(codes_file, dtype=str, delimiter=',')
    function_output_fol = op.join(SUBJECTS_DIR, subject, 'mmvt', '{}_objs'.format(model))
    utils.make_dir(function_output_fol)
    renamed_output_fol = op.join(SUBJECTS_DIR, subject, 'mmvt', model)
    utils.make_dir(renamed_output_fol)
    if lookup is None:
        lookup = load_subcortical_lookup_table()

    obj_files = glob.glob(op.join(function_output_fol, '*.srf'))
    errors = []
    if len(obj_files) < len(lookup) or overwrite_subcorticals:
        if overwrite:
            utils.delete_folder_files(function_output_fol)
            utils.delete_folder_files(renamed_output_fol)
        print('Trying to write into {}'.format(function_output_fol))
        for region_id in lookup.keys():
            if op.isfile(op.join(function_output_fol, '{}.srf'.format(region_id))):
                continue
            ret = fu.aseg_to_srf(subject, SUBJECTS_DIR, function_output_fol, region_id, mask_fname, norm_fname,
                           overwrite_subcorticals)
            if not ret:
                errors.append(lookup[region_id])
    if len(errors) > 0:
        print('Errors: {}'.format(','.join(errors)))
    ply_files = glob.glob(op.join(renamed_output_fol, '*.ply'))
    if len(ply_files) < len(lookup) or overwrite_subcorticals:
        convert_and_rename_subcortical_files(subject, function_output_fol, renamed_output_fol, lookup,
                                             mmvt_subcorticals_fol_name)
    blender_dir = op.join(MMVT_DIR, subject, mmvt_subcorticals_fol_name)
    if not op.isdir(blender_dir) or len(glob.glob(op.join(blender_dir, '*.ply'))) < len(ply_files) or overwrite_subcorticals:
        utils.delete_folder_files(blender_dir)
        copy_subcorticals_to_mmvt(renamed_output_fol, subject, mmvt_subcorticals_fol_name)
    flag_ok = len(glob.glob(op.join(blender_dir, '*.ply'))) >= len(lookup) and \
        len(glob.glob(op.join(blender_dir, '*.npz'))) >= len(lookup)
    return flag_ok



def load_subcortical_lookup_table(fname='sub_cortical_codes.txt'):
    codes_file = op.join(MMVT_DIR, fname)
    lookup = np.genfromtxt(codes_file, dtype=str, delimiter=',')
    lookup = {int(val):name for name, val in zip(lookup[:, 0], lookup[:, 1])}
    return lookup


def convert_and_rename_subcortical_files(subject, fol, new_fol, lookup, mmvt_subcorticals_fol_name='subcortical'):
    obj_files = glob.glob(op.join(fol, '*.srf'))
    utils.delete_folder_files(new_fol)
    for obj_file in obj_files:
        num = int(op.basename(obj_file)[:-4].split('_')[-1])
        new_name = lookup.get(num, '')
        if new_name != '':
            utils.srf2ply(obj_file, op.join(new_fol, '{}.ply'.format(new_name)))
            verts, faces = utils.read_ply_file(op.join(new_fol, '{}.ply'.format(new_name)))
            np.savez(op.join(new_fol, '{}.npz'.format(new_name)), verts=verts, faces=faces)
    copy_subcorticals_to_mmvt(new_fol, subject, mmvt_subcorticals_fol_name)


def copy_subcorticals_to_mmvt(subcorticals_fol, subject, mmvt_subcorticals_fol_name='subcortical'):
    blender_fol = op.join(MMVT_DIR, subject, mmvt_subcorticals_fol_name)
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
    for hemi in utils.get_hemis(hemi):
        utils.make_dir(op.join(MMVT_DIR, subject, 'surf'))
        for surf_type in ['inflated', 'pial']:
            surf_name = op.join(SUBJECTS_DIR, subject, 'surf', '{}.{}'.format(hemi, surf_type))
            surf_wavefront_name = '{}.asc'.format(surf_name)
            surf_new_name = '{}.srf'.format(surf_name)
            hemi_ply_fname = '{}.ply'.format(surf_name)
            mmvt_hemi_ply_fname = op.join(MMVT_DIR, subject, 'surf', '{}.{}.ply'.format(hemi, surf_type))
            mmvt_hemi_npz_fname = op.join(MMVT_DIR, subject, 'surf', '{}.{}.npz'.format(hemi, surf_type))
            if overwrite or not op.isfile(mmvt_hemi_ply_fname) or not op.isfile(mmvt_hemi_npz_fname) \
                    or not op.isfile(surf_new_name):
                print('mris_convert {} {}'.format(surf_name, surf_wavefront_name))
                utils.run_script('mris_convert {} {}'.format(surf_name, surf_wavefront_name))
                os.rename(surf_wavefront_name, surf_new_name)
                print('{} {}: convert asc to ply'.format(hemi, surf_type))
                convert_hemis_srf_to_ply(subject, hemi, surf_type)
                if op.isfile(mmvt_hemi_ply_fname):
                    os.remove(mmvt_hemi_ply_fname)
                shutil.copy(hemi_ply_fname, mmvt_hemi_ply_fname)
            if not op.isfile(mmvt_hemi_npz_fname):
                verts, faces = utils.read_ply_file(mmvt_hemi_ply_fname)
                np.savez(mmvt_hemi_npz_fname, verts=verts, faces=faces)
    return utils.both_hemi_files_exist(op.join(MMVT_DIR, subject, 'surf', '{hemi}.pial.ply')) and \
           utils.both_hemi_files_exist(op.join(MMVT_DIR, subject, 'surf', '{hemi}.pial.npz')) and \
           utils.both_hemi_files_exist(op.join(MMVT_DIR, subject, 'surf', '{hemi}.inflated.ply')) and \
           utils.both_hemi_files_exist(op.join(MMVT_DIR, subject, 'surf', '{hemi}.inflated.npz'))


def convert_hemis_srf_to_ply(subject, hemi='both', surf_type='pial'):
    for hemi in utils.get_hemis(hemi):
        ply_file = utils.srf2ply(op.join(SUBJECTS_DIR, subject, 'surf', '{}.{}.srf'.format(hemi, surf_type)),
                                 op.join(SUBJECTS_DIR, subject, 'surf', '{}.{}.ply'.format(hemi, surf_type)))
        # utils.make_dir(op.join(MMVT_DIR, subject))
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
            if op.isfile(curv_path):
                curv = nib.freesurfer.read_morph_data(curv_path)
                bin_curv = np.array(curv > 0, np.int)
                np.save(out_curv_file.format(hemi=hemi), bin_curv)
            else:
                print('{} is missing!'.format(curv_path))
                return False
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
    errors = {}
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
        errors = utils.calc_ply_faces_verts(verts, faces, out_file, overwrite, utils.namebase(ply_file), errors)
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


def create_annotation_from_template(subject, aparc_name='aparc250', fsaverage='fsaverage', remote_subject_dir='',
        overwrite_annotation=False, overwrite_morphing=False, do_solve_labels_collisions=False,
        morph_labels_from_fsaverage=True, fs_labels_fol='', n_jobs=6):
    annotations_exist = np.all([op.isfile(op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot'.format(hemi,
        aparc_name))) for hemi in HEMIS])
    if annotations_exist:
        return True
    else:
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
            annotations_exist = fu.create_annotation_file(subject, aparc_name, subjects_dir=SUBJECTS_DIR,
                                                          freesurfer_home=FREESURFER_HOME)
    if morph_labels_from_fsaverage:
        ret = lu.morph_labels_from_fsaverage(subject, SUBJECTS_DIR, MMVT_DIR, aparc_name, n_jobs=n_jobs,
            fsaverage=fsaverage, overwrite=overwrite_morphing, fs_labels_fol=fs_labels_fol)
        if not ret:
            return False
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
    utils.labels_to_annot(subject, SUBJECTS_DIR, aparc_name, overwrite=False)
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
                       'scripts_dir': BRAINDER_SCRIPTS_DIR, 'freesurfer_home': FREESURFER_HOME,
                       'surface_type': surface_type})
            cmd = 'matlab -nodisplay -nosplash -nodesktop -r "run({}); exit;"'.format(matlab_command)
            script_ret = utils.run_script(cmd)
            if script_ret == '':
                return False
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
    try:
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
    except:
        return False


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
    labels = lu.read_labels(subject, SUBJECTS_DIR, atlas)
    if len(labels) > 0:
        if np.all(labels[0].pos == 0):
            verts = {}
            for hemi in utils.HEMIS:
                verts[hemi], _ = utils.read_pial_npz(subject, MMVT_DIR, hemi)
            for label in labels:
                label.pos = verts[label.hemi][label.vertices]
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


# def save_labels_coloring(subject, atlas, n_jobs=2):
#     ret = False
#     coloring_dir = op.join(MMVT_DIR, subject, 'coloring')
#     utils.make_dir(coloring_dir)
#     coloring_fname = op.join(coloring_dir, 'labels_{}_coloring.csv'.format(atlas))
#     coloring_names_fname = op.join(coloring_dir, 'labels_{}_colors_names.txt'.format(atlas))
#     try:
#         labels = lu.read_labels(subject, SUBJECTS_DIR, atlas, n_jobs=n_jobs)
#         colors_rgb_and_names = cu.get_distinct_colors_and_names()
#         labels_colors_rgb, labels_colors_names = {}, {}
#         for label in labels:
#             label_inv_name = lu.get_label_hemi_invariant_name(label.name)
#             if label_inv_name not in labels_colors_rgb:
#                 labels_colors_rgb[label_inv_name], labels_colors_names[label_inv_name] = next(colors_rgb_and_names)
#         with open(coloring_fname, 'w') as colors_file, open(coloring_names_fname, 'w') as col_names_file:
#             for label in labels:
#                 label_inv_name = lu.get_label_hemi_invariant_name(label.name)
#                 color_rgb = labels_colors_rgb[label_inv_name]
#                 color_name = labels_colors_names[label_inv_name]
#                 colors_file.write('{},{},{},{}\n'.format(label.name, *color_rgb))
#                 col_names_file.write('{},{}\n'.format(label.name, color_name))
#         ret = op.isfile(coloring_fname)
#     except:
#         print('Error in save_labels_coloring!')
#         print(traceback.format_exc())
#     return ret


def save_cerebellum_coloring(subject):
    ret = False
    coloring_dir = op.join(MMVT_DIR, subject, 'coloring')
    utils.make_dir(coloring_dir)
    coloring_fname = op.join(coloring_dir, 'cerebellum_coloring.csv')
    lut_name = 'Buckner2011_17Networks_ColorLUT_new.txt'
    lut_fname = op.join(MMVT_DIR, 'templates', lut_name)
    if not op.isfile(lut_fname):
        lut_resources_fname = op.join(utils.get_resources_fol(), lut_name)
        if op.isfile(lut_resources_fname):
            shutil.copy(lut_resources_fname, lut_fname)
        else:
            print("The Buckner2011 17Networks Color LUT is missing! ({})".format(lut_fname))
            return False
    try:
        with open(coloring_fname, 'w') as colors_file, open(lut_fname, 'r') as lut_file:
            lut = lut_file.readlines()
            for ind, lut_line in zip(range(1, 34), lut[1:]):
                color_rgb = [float(x) / 255 for x in ' '.join(lut_line.split()).split(' ')[2:-1]]
                colors_file.write('{},{},{},{}\n'.format('cerebellum_{}'.format(ind), *color_rgb))
        ret = op.isfile(coloring_fname)
    except:
        print('Error in save_cerebellum_coloring!')
        print(traceback.format_exc())
    return ret


def transform_coordinates(subject, args):
    input_fname = op.join(MMVT_DIR, subject, 'cortical_points.npy')
    output_fname = op.join(MMVT_DIR, subject, 'cortical_points_{}'.format(args.trans_to_subject))
    try:
        if op.isfile(input_fname):
            points = np.genfromtxt(input_fname, dtype=np.float, delimiter=',')
            points_coords_to_subject = fu.transform_subject_to_subject_coordinates(
                subject, args.trans_to_subject, points, SUBJECTS_DIR)
            np.save(output_fname, points_coords_to_subject)
        else:
            print('transform_coordinates expecting coordinates file as input! ({})'.format(input_fname))
    except:
        print('Error in transform_coordinates!')
        print(traceback.format_exc())
    return op.isfile(output_fname)


# def find_hemis_boarders(subject):
#     from scipy.spatial.distance import cdist
#     verts = {}
#     for hemi in utils.HEMIS:
#         ply_file = op.join(SUBJECTS_DIR)
#         verts[hemi], _ = utils.read_ply_file(op.join(SUBJECTS_DIR, subject, 'surf', '{}.pial.ply'.format(hemi)))
#     dists = cdist(verts['rh'], verts['lh'])


def prepare_subject_folder(subject, remote_subject_dir, args, necessary_files=None):
    if necessary_files is None:
        necessary_files = args.necessary_files
    return utils.prepare_subject_folder(
        necessary_files, subject, remote_subject_dir, SUBJECTS_DIR,
        args.sftp, args.sftp_username, args.sftp_domain, args.sftp_password,
        args.overwrite_fs_files, args.print_traceback, args.sftp_port)


def main(subject, remote_subject_dir, args, flags):
    # from src.setup import create_fsaverage_link
    # create_fsaverage_link()
    utils.make_dir(op.join(SUBJECTS_DIR, subject, 'mmvt'))

    if utils.should_run(args, 'freesurfer_surface_to_blender_surface'):
        # *) convert rh.pial and lh.pial to rh.pial.ply and lh.pial.ply
        flags['hemis'] = freesurfer_surface_to_blender_surface(subject, overwrite=args.overwrite_hemis_srf)

    if utils.should_run(args, 'create_annotation_from_template'):
        # *) Create annotation file from fsaverage
        flags['create_annotation_from_template'] = create_annotation_from_template(
            subject, args.atlas, args.template_subject, remote_subject_dir, args.overwrite_annotation, args.overwrite_morphing_labels,
            args.solve_labels_collisions, args.morph_labels_from_fsaverage, args.fs_labels_fol, args.n_jobs)

    if utils.should_run(args, 'parcelate_cortex'):
        # *) Calls Matlab 'splitting_cortical.m' script
        flags['parcelate_cortex'] = parcelate_cortex(
            subject, args.atlas, args.overwrite_labels_ply_files, args.overwrite_ply_files)

    if utils.should_run(args, 'subcortical_segmentation'):
        # *) Create srf files for subcortical structures
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
        flags['save_labels_coloring'] = lu.create_atlas_coloring(subject, args.atlas, args.n_jobs)

    if 'cerebellum_segmentation' in args.function:
        flags['save_cerebellum_coloring'] = save_cerebellum_coloring(subject)
        flags['cerebellum_segmentation'] = cerebellum_segmentation(subject, remote_subject_dir, args)

    if 'transform_coordinates' in args.function:
        flags['transform_coordinates'] = transform_coordinates(subject, args)


    # for flag_type, val in flags.items():
    #     print('{}: {}'.format(flag_type, val))
    return flags


# def run_on_subjects(args):
#     subjects_flags, subjects_errors = {}, {}
#     args.sftp_password = utils.get_sftp_password(
#         args.subject, SUBJECTS_DIR, args.necessary_files, args.sftp_username, args.overwrite_fs_files) \
#         if args.sftp else ''
#     if '*' in args.subject:
#         args.subject = [utils.namebase(fol) for fol in glob.glob(op.join(SUBJECTS_DIR, args.subject))]
#     for subject in args.subject:
#         utils.make_dir(op.join(MMVT_DIR, subject, 'mmvt'))
#         # os.chdir(op.join(SUBJECTS_DIR, subject, 'mmvt'))
#         try:
#             flags = main(subject, args)
#             subjects_flags[subject] = flags
#         except:
#             subjects_errors[subject] = traceback.format_exc()
#             print('Error in subject {}'.format(subject))
#             print(traceback.format_exc())
#
#     errors = defaultdict(list)
#     for subject, flags in subjects_flags.items():
#         print('subject {}:'.format(subject))
#         for flag_type, val in flags.items():
#             print('{}: {}'.format(flag_type, val))
#             if not val:
#                 errors[subject].append(flag_type)
#     if len(errors) > 0:
#         print('Errors:')
#         for subject, error in errors.items():
#             print('{}: {}'.format(subject, error))


def read_cmd_args(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='MMVT anatomy preprocessing')
    parser.add_argument('--template_subject', help='template subject', required=False, default='fsaverage')
    parser.add_argument('--surf_name', help='surf_name', required=False, default='pial')
    parser.add_argument('--cerebellum_segmentation_loose', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--overwrite', help='overwrite', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_subcorticals', help='overwrite', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_annotation', help='overwrite_annotation', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_morphing_labels', help='overwrite_morphing_labels', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_hemis_srf', help='overwrite_hemis_srf', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_labels_ply_files', help='overwrite_labels_ply_files', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_faces_verts', help='overwrite_faces_verts', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_ply_files', help='overwrite_ply_files', required=False, default=0, type=au.is_true)
    parser.add_argument('--solve_labels_collisions', help='solve_labels_collisions', required=False, default=0, type=au.is_true)
    parser.add_argument('--morph_labels_from_fsaverage', help='morph_labels_from_fsaverage', required=False, default=1, type=au.is_true)
    parser.add_argument('--fs_labels_fol', help='fs_labels_fol', required=False, default='')
    parser.add_argument('--freesurfer', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--trans_to_subject', help='transform electrodes coords to this subject', required=False, default='')

    pu.add_common_args(parser)
    args = utils.Bag(au.parse_parser(parser, argv))
    args.necessary_files = {'mri': ['aseg.mgz', 'norm.mgz', 'ribbon.mgz', 'T1.mgz', 'orig.mgz'],
        'surf': ['rh.pial', 'lh.pial', 'rh.inflated', 'lh.inflated', 'lh.curv', 'rh.curv', 'rh.sphere.reg',
                 'lh.sphere.reg', 'lh.white', 'rh.white', 'rh.smoothwm','lh.smoothwm'],
        'mri:transforms' : ['talairach.xfm']}
        # 'label':['rh.{}.annot'.format(args.atlas), 'lh.{}.annot'.format(args.atlas)]}
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
    args = read_cmd_args()
    if os.environ.get('FREESURFER_HOME', '') == '' and args.freesurfer:
        print('Source freesurfer and rerun')
    else:
        pu.run_on_subjects(args, main)
        print('finish!')

    # fs_labels_fol = '/space/lilli/1/users/DARPA-Recons/fscopy/label/arc_april2016'
    # remote_subjects_dir = '/space/huygens/1/users/mia/subjects/{}_SurferOutput/'.format(subject.upper())
    # remote_subjects_dir = CACH_SUBJECT_DIR.format(subject=subject.upper())
    # remote_subjects_dir = op.join('/cluster/neuromind/tools/freesurfer', subject)
    # remote_subjects_dir = op.join('/autofs/space/lilli_001/users/DARPA-MEG/freesurfs')
    # subjects = ['mg78', 'mg82'] #set(utils.get_all_subjects(SUBJECTS_DIR, 'mg', '_')) - set(['mg96'])

