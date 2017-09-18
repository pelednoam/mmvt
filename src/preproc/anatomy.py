import glob
import os
import os.path as op
import shutil
import traceback
from collections import defaultdict
from tqdm import tqdm

import mne
import numpy as np
import scipy.io as sio
import nibabel as nib
import nibabel.freesurfer as nib_fs

from src.utils import labels_utils as lu
from src.utils import matlab_utils
from src.utils import utils
from src.utils import freesurfer_utils as fu
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
    ret = subcortical_segmentation(subject, args.overwrite_subcorticals, lookup, warp_buckner_hemis_atlas_fname,
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


@utils.check_for_freesurfer
@utils.timeit
def subcortical_segmentation(subject, overwrite_subcorticals=False, lookup=None,
                             mask_name='aseg.mgz', mmvt_subcorticals_fol_name='subcortical',
                             template_subject='', norm_name='norm.mgz', overwrite=True, n_jobs=6):
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
    # function_output_fol = op.join(MMVT_DIR, subject, '{}_objs'.format(model))
    # utils.make_dir(function_output_fol)
    mmvt_output_fol = op.join(MMVT_DIR, subject, mmvt_subcorticals_fol_name)
    utils.make_dir(mmvt_output_fol)
    if lookup is None:
        lookup = load_subcortical_lookup_table()

    ply_files = glob.glob(op.join(mmvt_output_fol, '*.ply'))
    npz_files = glob.glob(op.join(mmvt_output_fol, '*.npz'))
    errors = []
    if len(ply_files) < len(lookup) or len(npz_files) < len(lookup) or overwrite_subcorticals:
        if overwrite:
            utils.delete_folder_files(mmvt_output_fol)
        lookup_keys = [k for k in lookup.keys() if not op.isfile(op.join(mmvt_output_fol, '{}.ply'.format(
            lookup.get(k, ''))))]
        chunks = np.array_split(lookup_keys, n_jobs)
        params = [(subject, SUBJECTS_DIR, mmvt_output_fol, region_ids, lookup, mask_fname, norm_fname,
                   overwrite_subcorticals) for region_ids in chunks]
        errors = utils.run_parallel(_subcortical_segmentation_parallel, params, n_jobs)
        errors = utils.flat_list_of_lists(errors)
    if len(errors) > 0:
        print('Errors: {}'.format(','.join(errors)))
        return False
    flag_ok = len(glob.glob(op.join(mmvt_output_fol, '*.ply'))) >= len(lookup) and \
        len(glob.glob(op.join(mmvt_output_fol, '*.npz'))) >= len(lookup)
    return flag_ok


def _subcortical_segmentation_parallel(chunk):
    (subject, SUBJECTS_DIR, mmvt_output_fol, region_ids, lookup, mask_fname, norm_fname,
     overwrite_subcorticals) = chunk
    errors = []
    for region_id in region_ids:
        ret = fu.aseg_to_srf(
            subject, SUBJECTS_DIR, mmvt_output_fol, region_id, lookup, mask_fname, norm_fname,
            overwrite_subcorticals)
        if not ret:
            errors.append(lookup[region_id])
    return errors


def load_subcortical_lookup_table(fname='sub_cortical_codes.txt'):
    codes_file = op.join(MMVT_DIR, fname)
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
            verts, faces = utils.read_ply_file(op.join(new_fol, '{}.ply'.format(new_name)))
            np.savez(op.join(new_fol, '{}.npz'.format(new_name)), verts=verts, faces=faces)
    # copy_subcorticals_to_mmvt(new_fol, subject, mmvt_subcorticals_fol_name)


# def copy_subcorticals_to_mmvt(subcorticals_fol, subject, mmvt_subcorticals_fol_name='subcortical'):
#     blender_fol = op.join(MMVT_DIR, subject, mmvt_subcorticals_fol_name)
#     if op.isdir(blender_fol):
#         shutil.rmtree(blender_fol)
#     shutil.copytree(subcorticals_fol, blender_fol)


def create_surfaces(subject, hemi='both', overwrite=False):
    for hemi in utils.get_hemis(hemi):
        utils.make_dir(op.join(MMVT_DIR, subject, 'surf'))
        for surf_type in ['inflated', 'pial']:
            surf_name = op.join(SUBJECTS_DIR, subject, 'surf', '{}.{}'.format(hemi, surf_type))
            mmvt_hemi_ply_fname = op.join(MMVT_DIR, subject, 'surf', '{}.{}.ply'.format(hemi, surf_type))
            mmvt_hemi_npz_fname = op.join(MMVT_DIR, subject, 'surf', '{}.{}.npz'.format(hemi, surf_type))
            # mmvt_hemi_mat_fname = op.join(MMVT_DIR, subject, 'surf', '{}.{}.mat'.format(hemi, surf_type))
            if not op.isfile(mmvt_hemi_ply_fname) or overwrite:
                print('Reading {}'.format(surf_name))
                if op.isfile(mmvt_hemi_npz_fname):
                    verts, faces = utils.read_pial(subject, MMVT_DIR, hemi)
                else:
                    verts, faces = nib_fs.read_geometry(surf_name)
                if surf_type == 'inflated':
                    verts_offset = 55 if hemi == 'rh' else -55
                    verts[:, 0] = verts[:, 0] + verts_offset
                utils.write_ply_file(verts, faces, mmvt_hemi_ply_fname, True)
                # sio.savemat(mmvt_hemi_mat_fname, mdict={'verts': verts, 'faces': faces + 1})
    return all([utils.both_hemi_files_exist(op.join(MMVT_DIR, subject, 'surf', file_name)) for file_name in \
                ('{hemi}.pial.ply', '{hemi}.pial.npz', '{hemi}.inflated.ply', '{hemi}.inflated.npz')])


# def create_surfaces(subject, hemi='both', overwrite=False):
#     for hemi in utils.get_hemis(hemi):
#         utils.make_dir(op.join(MMVT_DIR, subject, 'surf'))
#         for surf_type in ['inflated', 'pial']:
#             surf_name = op.join(SUBJECTS_DIR, subject, 'surf', '{}.{}'.format(hemi, surf_type))
#             surf_wavefront_name = '{}.asc'.format(surf_name)
#             surf_new_name = '{}.srf'.format(surf_name)
#             hemi_ply_fname = '{}.ply'.format(surf_name)
#             hemi_npz_fname = '{}.ply'.format(surf_name)
#             mmvt_hemi_ply_fname = op.join(MMVT_DIR, subject, 'surf', '{}.{}.ply'.format(hemi, surf_type))
#             mmvt_hemi_npz_fname = op.join(MMVT_DIR, subject, 'surf', '{}.{}.npz'.format(hemi, surf_type))
#             if not op.isfile(surf_new_name) or overwrite:
#                 print('mris_convert {} {}'.format(surf_name, surf_wavefront_name))
#                 utils.run_script('mris_convert {} {}'.format(surf_name, surf_wavefront_name))
#                 os.rename(surf_wavefront_name, surf_new_name)
#             if not op.isfile(hemi_ply_fname)  or overwrite:
#                 print('{} {}: convert asc to ply'.format(hemi, surf_type))
#                 convert_hemis_srf_to_ply(subject, hemi, surf_type)
#                 if surf_type == 'inflated':
#                     verts, faces = utils.read_ply_file(hemi_ply_fname)
#                     verts[:, 0] += 55 if hemi == 'rh' else -55
#                     utils.write_ply_file(verts, faces, hemi_ply_fname)
#             if not op.isfile(mmvt_hemi_ply_fname) or overwrite:
#                 if op.isfile(mmvt_hemi_ply_fname):
#                     os.remove(mmvt_hemi_ply_fname)
#                 shutil.copy(hemi_ply_fname, mmvt_hemi_ply_fname)
#             if not op.isfile(mmvt_hemi_npz_fname) or overwrite:
#                 if op.isfile(mmvt_hemi_npz_fname):
#                     os.remove(mmvt_hemi_npz_fname)
#                 verts, faces = utils.read_ply_file(mmvt_hemi_ply_fname)
#                 np.savez(mmvt_hemi_npz_fname, verts=verts, faces=faces)
#             if not op.isfile(hemi_npz_fname) or overwrite:
#                 if op.isfile(hemi_npz_fname):
#                     os.remove(hemi_npz_fname)
#                 np.savez(hemi_npz_fname, verts=verts, faces=faces)
#     return utils.both_hemi_files_exist(op.join(MMVT_DIR, subject, 'surf', '{hemi}.pial.ply')) and \
#            utils.both_hemi_files_exist(op.join(MMVT_DIR, subject, 'surf', '{hemi}.pial.npz')) and \
#            utils.both_hemi_files_exist(op.join(MMVT_DIR, subject, 'surf', '{hemi}.inflated.ply')) and \
#            utils.both_hemi_files_exist(op.join(MMVT_DIR, subject, 'surf', '{hemi}.inflated.npz'))


# def convert_hemis_srf_to_ply(subject, hemi='both', surf_type='pial'):
#     for hemi in utils.get_hemis(hemi):
#         ply_file = utils.srf2ply(op.join(SUBJECTS_DIR, subject, 'surf', '{}.{}.srf'.format(hemi, surf_type)),
#                                  op.join(SUBJECTS_DIR, subject, 'surf', '{}.{}.ply'.format(hemi, surf_type)))
#         # utils.make_dir(op.join(MMVT_DIR, subject))
#         # shutil.copyfile(ply_file, op.join(MMVT_DIR, subject, 'surf', '{}.{}.ply'.format(hemi, surf_type)))


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


@utils.timeit
def convert_perecelated_cortex(subject, atlas, surf_type='pial', overwrite_ply_files=False, hemi='both'):
    lookup = {}
    for hemi in utils.get_hemis(hemi):
        lookup[hemi] = create_labels_lookup(subject, hemi, atlas)
        if len(lookup[hemi]) == 0:
            continue
        mat_fol = op.join(SUBJECTS_DIR, subject, '{}.{}.{}'.format(atlas, surf_type, hemi))
        ply_fol = op.join(SUBJECTS_DIR, subject, '{}_{}_{}_ply'.format(atlas, surf_type, hemi))
        utils.make_dir(op.join(MMVT_DIR, subject, 'labels'))
        blender_fol = op.join(MMVT_DIR, subject, 'labels', '{}.{}.{}'.format(atlas, surf_type, hemi))
        # utils.convert_mat_files_to_ply(mat_fol, overwrite_ply_files)
        # rename_cortical(lookup, mat_fol, ply_fol)
        # if surf_type == 'inflated':
        #     for ply_fname in glob.glob(op.join(ply_fol, '*.ply')):
        #         verts, faces = utils.read_ply_file(ply_fname)
        #         verts_offset = 55 if hemi == 'rh' else -55
        #         verts[:, 0] = verts[:, 0] + verts_offset
        #         utils.write_ply_file(verts, faces, ply_fname)
        # utils.rmtree(blender_fol)
        # shutil.copytree(ply_fol, blender_fol)
        # utils.rmtree(mat_fol)
        # utils.rmtree(ply_fol)
    return lookup


def create_annotation(subject, atlas='aparc250', fsaverage='fsaverage', remote_subject_dir='',
        overwrite_annotation=False, overwrite_morphing=False, do_solve_labels_collisions=False,
        morph_labels_from_fsaverage=True, fs_labels_fol='', save_annot_file=True, surf_type='inflated', n_jobs=6):
    annotation_fname_template = op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot'.format('{hemi}', atlas))
    annotations_exist = utils.both_hemi_files_exist(annotation_fname_template)
    if annotations_exist and not overwrite_annotation:
        print('The annotation file is already exist ({})'.format(annotation_fname_template))
        return True
    else:
        if len(glob.glob(op.join(SUBJECTS_DIR, subject, 'label', atlas, '*.label'))) > 0:
            if save_annot_file:
                labels_to_annot(subject, atlas, overwrite_annotation, surf_type, n_jobs)
            if not overwrite_annotation:
                return True
        utils.make_dir(op.join(SUBJECTS_DIR, subject, 'label'))
        remote_annotations_exist = np.all([op.isfile(op.join(remote_subject_dir, 'label', '{}.{}.annot'.format(
            hemi, atlas))) for hemi in HEMIS])
        if remote_annotations_exist:
            for hemi in HEMIS:
                shutil.copy(op.join(remote_subject_dir, 'label', '{}.{}.annot'.format(hemi, atlas)),
                            op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot'.format(hemi, atlas)))
            return True
    existing_freesurfer_annotations = ['aparc.DKTatlas40.annot', 'aparc.annot', 'aparc.a2009s.annot']
    if '{}.annot'.format(atlas) in existing_freesurfer_annotations:
        morph_labels_from_fsaverage = False
        do_solve_labels_collisions = False
        if not utils.both_hemi_files_exist(annotation_fname_template):
            utils.make_dir(op.join(SUBJECTS_DIR, subject, 'label'))
            annotations_exist = fu.create_annotation_file(
                subject, atlas, subjects_dir=SUBJECTS_DIR, freesurfer_home=FREESURFER_HOME)
    if morph_labels_from_fsaverage:
        ret = lu.morph_labels_from_fsaverage(subject, SUBJECTS_DIR, MMVT_DIR, atlas, n_jobs=n_jobs,
            fsaverage=fsaverage, overwrite=overwrite_morphing, fs_labels_fol=fs_labels_fol)
        if not ret:
            return False
    if do_solve_labels_collisions:
        solve_labels_collisions(subject, atlas, surf_type, n_jobs)
    if save_annot_file and (overwrite_annotation or not annotations_exist):
        labels_to_annot(subject, atlas, overwrite_annotation, surf_type, n_jobs)
    if save_annot_file:
        return utils.both_hemi_files_exist(op.join(
            SUBJECTS_DIR, subject, 'label', '{}.{}.annot'.format('{hemi}', atlas)))
    else:
        return len(glob.glob(op.join(SUBJECTS_DIR, subject, 'label', atlas, '*.label'))) > 0


def labels_to_annot(subject, atlas, overwrite_annotation=False, surf_type='inflated',
                    n_jobs=6):
    labels = []
    try:
        labels = lu.labels_to_annot(subject, SUBJECTS_DIR, atlas, overwrite=overwrite_annotation)
    except:
        print("Can't write labels to annotation! Trying to solve labels collision")
        # print(traceback.format_exc())
        solve_labels_collisions(subject, atlas, surf_type, n_jobs)
        try:
            labels = lu.labels_to_annot(subject, SUBJECTS_DIR, atlas, overwrite=overwrite_annotation)
        except:
            print("Can't write labels to annotation! Solving the labels collision didn't help...")
            print(traceback.format_exc())
    return labels


def solve_labels_collisions(subject, atlas, surf_type='inflated', n_jobs=6):
    backup_labels_fol = '{}_before_solve_collision'.format(atlas)
    lu.solve_labels_collision(subject, SUBJECTS_DIR, atlas, backup_labels_fol, surf_type, n_jobs)
    lu.backup_annotation_files(subject, SUBJECTS_DIR, atlas)


@utils.timeit
def parcelate_cortex(subject, atlas, overwrite=False, overwrite_annotation=False,
                     overwrite_vertices_labels_lookup=False, n_jobs=6):
    utils.make_dir(op.join(MMVT_DIR, subject, 'labels'))
    labels_to_annot(subject, atlas, overwrite_annotation, n_jobs=n_jobs)
    params = []

    for surface_type in ['pial', 'inflated']:
        files_exist = True
        vertices_labels_ids_lookup = lu.create_vertices_labels_lookup(
            subject, atlas, True, overwrite_vertices_labels_lookup)
        for hemi in HEMIS:
            blender_labels_fol = op.join(MMVT_DIR, subject, 'labels', '{}.{}.{}'.format(atlas, surface_type, hemi))
            labels = lu.read_labels(subject, SUBJECTS_DIR, atlas, hemi=hemi)
            files_exist = files_exist and op.isdir(blender_labels_fol) and \
                len(glob.glob(op.join(blender_labels_fol, '*.ply'))) >= len(labels)
            if overwrite or not files_exist:
                params.append((subject, atlas, hemi, surface_type, vertices_labels_ids_lookup[hemi]))

    if len(params) > 0:
        results = utils.run_parallel(_parcelate_cortex_parallel, params, njobs=n_jobs)
        return all(results)
    else:
        return True


def _parcelate_cortex_parallel(p):
    from src.preproc import parcelate_cortex
    subject, atlas, hemi, surface_type, vertices_labels_ids_lookup = p
    print('Parcelate the {} {} cortex'.format(hemi, surface_type))
    return parcelate_cortex.parcelate(subject, atlas, hemi, surface_type, vertices_labels_ids_lookup)


def save_matlab_labels_vertices(subject, atlas):
    for hemi in HEMIS:
        matlab_fname = op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot_labels.m'.format(hemi, atlas))
        if op.isfile(matlab_fname):
            labels_dic = matlab_utils.matlab_cell_arrays_to_dict(matlab_fname)
            utils.save(labels_dic, op.join(MMVT_DIR, subject, 'labels_dic_{}_{}.pkl'.format(atlas, hemi)))
        else:
            return False
    return True


def save_labels_vertices(subject, atlas):
    try:
        labels = lu.read_labels(subject, SUBJECTS_DIR, atlas, sorted_according_to_annot_file=True,
                                read_only_from_annot=True)
        if len(labels) == 0:
            labels = lu.read_labels(subject, SUBJECTS_DIR, atlas)
        labels_names, labels_vertices = defaultdict(list), defaultdict(list)
        for label in labels:
            labels_names[label.hemi].append(label.name)
            labels_vertices[label.hemi].append(label.vertices)
        output_fname = op.join(MMVT_DIR, subject, 'labels_vertices_{}.pkl'.format(atlas))
        utils.save((labels_names, labels_vertices), output_fname)
        return op.isfile(output_fname)
    except:
        return False


@utils.timeit
def create_spatial_connectivity(subject):
    try:
        verts_neighbors_fname = op.join(MMVT_DIR, subject, 'verts_neighbors_{hemi}.pkl')
        connectivity_fname = op.join(MMVT_DIR, subject, 'spatial_connectivity.pkl')
        if utils.both_hemi_files_exist(verts_neighbors_fname) and op.isfile(connectivity_fname):
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


@utils.tryit(False, False)
def calc_labeles_contours(subject, atlas, overwrite=True, verbose=False):
    output_fname = op.join(MMVT_DIR, subject, '{}_contours_{}.npz'.format(atlas, '{hemi}'))
    if utils.both_hemi_files_exist(output_fname) and not overwrite:
        return True
    verts_neighbors_fname = op.join(MMVT_DIR, subject, 'verts_neighbors_{hemi}.pkl')
    if not utils.both_hemi_files_exist(verts_neighbors_fname):
        print('calc_labeles_contours: You should first run create_spatial_connectivity')
        create_spatial_connectivity(subject)
        return calc_labeles_contours(subject, atlas, overwrite, verbose)
    vertices_labels_lookup = lu.create_vertices_labels_lookup(subject, atlas, False, overwrite)
    for hemi in utils.HEMIS:
        verts, _ = utils.read_pial(subject, MMVT_DIR, hemi)
        contours = np.zeros((len(verts)))
        vertices_neighbors = np.load(verts_neighbors_fname.format(hemi=hemi))
        # labels = lu.read_hemi_labels(subject, SUBJECTS_DIR, atlas, hemi)
        labels = lu.read_labels(subject, SUBJECTS_DIR, atlas, hemi=hemi)
        for label_ind, label in enumerate(labels):
            if verbose:
                label_nei = np.zeros((len(label.vertices)))
            for vert_ind, vert in enumerate(label.vertices):
                nei = set([vertices_labels_lookup[hemi].get(v, '') for v in vertices_neighbors[vert]]) - set([''])
                contours[vert] = label_ind + 1 if len(nei) > 1 else 0
                if verbose:
                    label_nei[vert_ind] = contours[vert]
            if verbose:
                print(label.name, len(np.where(label_nei)[0]) / len(verts))
        np.savez(output_fname.format(hemi=hemi), contours=contours, max=len(labels),
                 labels=[l.name for l in labels])
    return utils.both_hemi_files_exist(output_fname)

@utils.timeit
def create_verts_faces_lookup(subject):
    output_fname = op.join(MMVT_DIR, subject, 'faces_verts_lookup_{}.pkl'.format('{hemi}'))
    for hemi in utils.HEMIS:
        if op.isfile(output_fname.format(hemi=hemi)):
            continue
        verts, faces = utils.read_pial(subject, MMVT_DIR, hemi)
        lookup = defaultdict(list)
        for f_ind, f in tqdm(enumerate(faces)):
            for v_ind in f:
                lookup[v_ind].append(f_ind)
        utils.save(lookup, output_fname.format(hemi=hemi))


@utils.timeit
def calc_faces_contours(subject, atlas):
    create_verts_faces_lookup(subject)
    vertices_labels_lookup = lu.create_vertices_labels_lookup(subject, atlas)
    verts_neighbors_fname = op.join(MMVT_DIR, subject, 'verts_neighbors_{hemi}.pkl')
    contours_fname = op.join(MMVT_DIR, subject, '{}_contours_{}.npz'.format(atlas, '{hemi}'))
    # verts_faces_lookup_fname = op.join(MMVT_DIR, subject, 'faces_verts_{}.npy'.format('{hemi}'))
    verts_faces_lookup_fname = op.join(MMVT_DIR, subject, 'faces_verts_lookup_{}.pkl'.format('{hemi}'))
    output_fname = op.join(MMVT_DIR, subject, 'contours_faces_{}.pkl'.format(atlas))
    contours_faces = dict(rh=set(), lh=set())
    for hemi in utils.HEMIS:
        contours_dict = np.load(contours_fname.format(hemi=hemi))
        vertices_neighbors = np.load(verts_neighbors_fname.format(hemi=hemi))
        verts_faces_lookup = utils.load(verts_faces_lookup_fname.format(hemi=hemi))
        contours_vertices = np.where(contours_dict['contours'])[0]
        for vert in tqdm(contours_vertices):
            vert_label = vertices_labels_lookup[hemi].get(vert, '')
            vert_faces = verts_faces_lookup[vert]
            for vert_nei in vertices_neighbors[vert]:
                nei_label = vertices_labels_lookup[hemi].get(vert_nei, '')
                if vert_label != nei_label:
                    nei_faces = verts_faces_lookup[vert_nei]
                    common_faces = set(vert_faces) & set(nei_faces)
                    contours_faces[hemi] |= common_faces
    utils.save(contours_faces, output_fname)
    sio.savemat(op.join(MMVT_DIR, subject, 'contours_faces_{}.mat'.format(atlas)),
                mdict={hemi:np.array(list(contours_faces[hemi])) + 1 for hemi in utils.HEMIS})
    return op.isfile(output_fname)

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


def calc_labels_center_of_mass(subject, atlas):
    import csv
    labels = lu.read_labels(subject, SUBJECTS_DIR, atlas)
    if len(labels) > 0:
        if np.all(labels[0].pos == 0):
            verts = {}
            for hemi in utils.HEMIS:
                verts[hemi], _ = utils.read_pial(subject, MMVT_DIR, hemi)
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
            print('file saved at '+output_fname)
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


@utils.tryit()
def save_subject_orig_trans(subject):
    from src.utils import trans_utils as tu
    output_fname = op.join(MMVT_DIR, subject, 'orig_trans.npz')
    header = tu.get_subject_orig_header(subject, SUBJECTS_DIR)
    vox2ras_tkr = header.get_vox2ras_tkr()
    ras_tkr2vox = np.linalg.inv(vox2ras_tkr)
    vox2ras = header.get_vox2ras()
    ras2vox = np.linalg.inv(vox2ras)
    np.savez(output_fname, ras_tkr2vox=ras_tkr2vox, vox2ras_tkr=vox2ras_tkr, vox2ras=vox2ras, ras2vox=ras2vox)
    return op.isfile(output_fname)


def calc_3d_atlas(subject, atlas, overwrite_aseg_file=True):
    from src.preproc import freeview as fr
    aparc_ret = fr.create_aparc_aseg_file(subject, atlas, overwrite_aseg_file)
    lut_ret = fr.create_lut_file_for_atlas(subject, atlas)
    return aparc_ret and lut_ret


@utils.tryit()
def create_high_level_atlas(subject):
    if not utils.both_hemi_files_exist(op.join(SUBJECTS_DIR, subject, 'label', '{hemi}.aparc.DKTatlas40.annot')):
        fu.create_annotation_file(
            subject, 'aparc.DKTatlas40', subjects_dir=SUBJECTS_DIR, freesurfer_home=FREESURFER_HOME)
    look = create_labels_names_lookup(subject, 'aparc.DKTatlas40')
    csv_fname = op.join(MMVT_DIR, 'high_level_atlas.csv')
    if not op.isfile(csv_fname):
        print('No high_level_atlas.csv in MMVT_DIR!')
        return False
    labels = []
    for hemi in utils.HEMIS:
        for line in utils.csv_file_reader(csv_fname, ','):
            new_label = lu.join_labels('{}-{}'.format(line[0], hemi), (look['{}-{}'.format(l, hemi)] for l in line[1:]))
            labels.append(new_label)
    lu.labels_to_annot(subject, SUBJECTS_DIR, 'high.level.atlas', labels=labels, overwrite=True)
    return utils.both_hemi_files_exist(op.join(SUBJECTS_DIR, subject, 'label', '{hemi}.high.level.atlas.annot'))


def create_labels_names_lookup(subject, atlas):
    lookup = {}
    labels = lu.read_labels(subject, SUBJECTS_DIR, atlas)
    for label in labels:
        lookup[label.name] = label
    return lookup


def create_new_subject_blend_file(subject, atlas, overwrite_blend=False):
    # Create a file for the new subject
    atlas = utils.get_real_atlas_name(atlas, short_name=True)
    new_fname = op.join(MMVT_DIR, '{}_{}.blend'.format(subject, atlas))
    empty_subject_fname = op.join(MMVT_DIR, 'empty_subject.blend')
    if not op.isfile(empty_subject_fname):
        resources_dir = op.join(utils.get_parent_fol(levels=2), 'resources')
        shutil.copy(op.join(resources_dir, 'empty_subject.blend'), empty_subject_fname)
    if op.isfile(new_fname) and not overwrite_blend:
        overwrite = input('The file {} already exist, do you want to overwrite? '.format(new_fname))
        if au.is_true(overwrite):
           os.remove(new_fname)
           shutil.copy(op.join(MMVT_DIR, 'empty_subject.blend'), new_fname)
    else:
        shutil.copy(empty_subject_fname, new_fname)
    return op.isfile(new_fname)


def check_bem(subject, remote_subject_dir, args):
    from src.preproc import meg
    meg_args = meg.read_cmd_args(dict(subject=subject))
    meg_args.update(args)
    meg.init(subject, meg_args, remote_subject_dir=remote_subject_dir)
    args.remote_subject_dir = remote_subject_dir
    meg.check_bem(subject, meg_args)


def main(subject, remote_subject_dir, args, flags):
    # from src.setup import create_fsaverage_link
    # create_fsaveragge_link()
    # utils.make_dir(op.join(SUBJECTS_DIR, subject, 'mmvt'))

    if utils.should_run(args, 'create_surfaces'):
        # *) convert rh.pial and lh.pial to rh.pial.ply and lh.pial.ply
        flags['create_surfaces'] = create_surfaces(subject, overwrite=args.overwrite_hemis_ply)

    if utils.should_run(args, 'create_annotation'):
        # *) Create annotation file from fsaverage
        flags['create_annotation'] = create_annotation(
            subject, args.atlas, args.template_subject, remote_subject_dir, args.overwrite_annotation,
            args.overwrite_morphing_labels, args.solve_labels_collisions, args.morph_labels_from_fsaverage,
            args.fs_labels_fol, args.save_annot_file, args.solve_labels_collision_surf_type, args.n_jobs)

    if utils.should_run(args, 'parcelate_cortex'):
        flags['parcelate_cortex'] = parcelate_cortex(
            subject, args.atlas, args.overwrite_labels_ply_files, args.overwrite_annotation,
            args.overwrite_vertices_labels_lookup, args.n_jobs)

    if utils.should_run(args, 'subcortical_segmentation'):
        # *) Create srf files for subcortical structures
        flags['subcortical'] = subcortical_segmentation(subject, args.overwrite_subcorticals, n_jobs=args.n_jobs)

    if utils.should_run(args, 'calc_faces_verts_dic'):
        # *) Create a dictionary for verts and faces for both hemis
        flags['faces_verts'] = calc_faces_verts_dic(subject, args.atlas, args.overwrite_faces_verts)

    if utils.should_run(args, 'save_labels_vertices'):
        # *) Save the labels vertices for meg label plotting
        flags['labels_vertices'] = save_labels_vertices(subject, args.atlas)

    if utils.should_run(args, 'save_hemis_curv'):
        # *) Save the hemis curvs for the inflated brain
        flags['save_hemis_curv'] = save_hemis_curv(subject, args.atlas)

    if utils.should_run(args, 'create_high_level_atlas'):
        flags['create_high_level_atlas'] = create_high_level_atlas(subject)

    if utils.should_run(args, 'create_spatial_connectivity'):
        # *) Create the subject's connectivity
        flags['connectivity'] = create_spatial_connectivity(subject)

    if utils.should_run(args, 'calc_labeles_contours'):
        flags['calc_labeles_contours'] = calc_labeles_contours(subject, args.atlas)

    if utils.should_run(args, 'calc_labels_center_of_mass'):
        # *) Calc the labels center of mass
        flags['center_of_mass'] = calc_labels_center_of_mass(subject, args.atlas)

    if utils.should_run(args, 'save_labels_coloring'):
        # *) Save a coloring file for the atlas's labels
        flags['save_labels_coloring'] = lu.create_atlas_coloring(subject, args.atlas, args.n_jobs)

    if utils.should_run(args, 'save_subject_orig_trans'):
        flags['save_subject_orig_trans'] = save_subject_orig_trans(subject)

    if utils.should_run(args, 'calc_3d_atlas'):
        flags['calc_3d_atlas'] = calc_3d_atlas(subject, args.atlas, args.overwrite_aseg_file)

    if utils.should_run(args, 'create_new_subject_blend_file'):
        flags['create_new_subject_blend_file'] = create_new_subject_blend_file(
            subject, args.atlas, args.overwrite_blend)

    if 'cerebellum_segmentation' in args.function:
        flags['save_cerebellum_coloring'] = save_cerebellum_coloring(subject)
        flags['cerebellum_segmentation'] = cerebellum_segmentation(subject, remote_subject_dir, args)

    if 'transform_coordinates' in args.function:
        flags['transform_coordinates'] = transform_coordinates(subject, args)

    if 'grow_label' in args.function:
        flags['grow_label'] = lu.grow_label(
            subject, args.vertice_indice, args.hemi, args.label_name, args.label_r, args.n_jobs)

    if 'calc_faces_contours' in args.function:
        flags['calc_faces_contours'] = calc_faces_contours(
            subject, args.atlas)

    if 'check_bem' in args.function:
        flags['check_bem'] = check_bem(subject, remote_subject_dir, args)

    return flags


def read_cmd_args(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='MMVT anatomy preprocessing')
    parser.add_argument('--template_subject', help='template subject', required=False,
                        default='fsaverage6,fsaverage5,fsaverage,colin27', type=au.str_arr_type)
    parser.add_argument('--surf_name', help='surf_name', required=False, default='pial')
    parser.add_argument('--cerebellum_segmentation_loose', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--overwrite', help='overwrite', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_subcorticals', help='overwrite', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_annotation', help='overwrite_annotation', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_vertices_labels_lookup', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_morphing_labels', help='overwrite_morphing_labels', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_hemis_ply', help='overwrite_hemis_ply', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_labels_ply_files', help='overwrite_labels_ply_files', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_faces_verts', help='overwrite_faces_verts', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_ply_files', help='overwrite_ply_files', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_blend', help='overwrite_blend', required=False, default=0, type=au.is_true)
    parser.add_argument('--solve_labels_collisions', help='solve_labels_collisions', required=False, default=0, type=au.is_true)
    parser.add_argument('--morph_labels_from_fsaverage', help='morph_labels_from_fsaverage', required=False, default=1, type=au.is_true)
    parser.add_argument('--fs_labels_fol', help='fs_labels_fol', required=False, default='')
    parser.add_argument('--save_annot_file', help='save_annot_file', required=False, default=1, type=au.is_true)
    parser.add_argument('--freesurfer', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--trans_to_subject', help='transform electrodes coords to this subject', required=False, default='')
    parser.add_argument('--overwrite_aseg_file', help='overwrite_aseg_file', required=False, default=0, type=au.is_true)
    parser.add_argument('--no_fs', help='no_fs', required=False, default=0, type=au.is_true)
    parser.add_argument('--matlab_cmd', help='matlab cmd', required=False, default='matlab')
    parser.add_argument('--solve_labels_collision_surf_type', help='', required=False, default='inflated')

    parser.add_argument('--vertice_indice', help='', required=False, default=0, type=int)
    parser.add_argument('--label_name', help='', required=False, default='')
    parser.add_argument('--hemi', help='', required=False, default='')
    parser.add_argument('--label_r', help='', required=False, default='5', type=int)

    pu.add_common_args(parser)
    args = utils.Bag(au.parse_parser(parser, argv))
    existing_freesurfer_annotations = ['aparc.DKTatlas40', 'aparc', 'aparc.a2009s']
    args.necessary_files = {'mri': ['aseg.mgz', 'norm.mgz', 'ribbon.mgz', 'T1.mgz', 'orig.mgz'],
        'surf': ['rh.pial', 'lh.pial', 'rh.inflated', 'lh.inflated', 'lh.curv', 'rh.curv', 'rh.sphere.reg',
                 'lh.sphere.reg', 'rh.sphere', 'lh.sphere', 'lh.white', 'rh.white', 'rh.smoothwm','lh.smoothwm',
                 'lh.sphere.reg', 'rh.sphere.reg'],
        'mri:transforms' : ['talairach.xfm', 'talairach.m3z'],
        'label':['rh.{}.annot'.format(annot_name) for annot_name in existing_freesurfer_annotations] +
                ['lh.{}.annot'.format(annot_name) for annot_name in existing_freesurfer_annotations]}
    if args.overwrite:
        args.overwrite_annotation = True
        args.overwrite_morphing_labels = True
        args.overwrite_hemis_ply = True
        args.overwrite_labels_ply_files = True
        args.overwrite_faces_verts = True
        args.overwrite_fs_files = True
    if 'labeling' in args.function:
        args.function.extend(['create_annotation', 'save_labels_vertices', 'calc_labeles_contours',
                              'calc_labels_center_of_mass', 'save_labels_coloring'])
    # print(args)
    return args


if __name__ == '__main__':
    # ******************************************************************
    # Be sure that you have matlab installed on your machine,
    # and you can run it from the terminal by just calling 'matlab'
    # Some of the functions are using freesurfer, so if you want to
    # run main, you need to source freesurfer.
    # ******************************************************************
    args = read_cmd_args()
    # if not args.no_fs and os.environ.get('FREESURFER_HOME', '') == '' and args.freesurfer:
    #     print('Source freesurfer and rerun')
    # else:
    pu.run_on_subjects(args, main)
    print('finish!')

    # fs_labels_fol = '/space/lilli/1/users/DARPA-Recons/fscopy/label/arc_april2016'
    # remote_subjects_dir = '/space/huygens/1/users/mia/subjects/{}_SurferOutput/'.format(subject.upper())
    # remote_subjects_dir = CACH_SUBJECT_DIR.format(subject=subject.upper())
    # remote_subjects_dir = op.join('/cluster/neuromind/tools/freesurfer', subject)
    # remote_subjects_dir = op.join('/autofs/space/lilli_001/users/DARPA-MEG/freesurfs')
    # subjects = ['mg78', 'mg82'] #set(utils.get_all_subjects(SUBJECTS_DIR, 'mg', '_')) - set(['mg96'])

