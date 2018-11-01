import glob
import os
import os.path as op
import shutil
import traceback
from collections import defaultdict
from tqdm import tqdm
import csv
import copy

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
HEMIS = ['rh', 'lh']


def cerebellum_segmentation(subject, remote_subject_dir, args, subregions_num=7, model='Buckner2011_7Networks'):
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


def create_surfaces(subject, surfaces_types=('inflated', 'pial'), hemi='both', overwrite=False):
    for hemi in lu.get_hemis(hemi):
        utils.make_dir(op.join(MMVT_DIR, subject, 'surf'))
        for surf_type in surfaces_types:
            surf_name = op.join(SUBJECTS_DIR, subject, 'surf', '{}.{}'.format(hemi, surf_type))
            mmvt_hemi_ply_fname = op.join(MMVT_DIR, subject, 'surf', '{}.{}.ply'.format(hemi, surf_type))
            mmvt_hemi_npz_fname = op.join(MMVT_DIR, subject, 'surf', '{}.{}.npz'.format(hemi, surf_type))
            # mmvt_hemi_mat_fname = op.join(MMVT_DIR, subject, 'surf', '{}.{}.mat'.format(hemi, surf_type))
            if not op.isfile(mmvt_hemi_ply_fname) or overwrite:
                print('Reading {}'.format(surf_name))
                if op.isfile(surf_name):
                    verts, faces = nib_fs.read_geometry(surf_name)
                elif op.isfile(mmvt_hemi_npz_fname):
                    verts, faces = utils.read_pial(subject, MMVT_DIR, hemi)
                else:
                    if surf_type != 'dural':
                        raise Exception("Can't find the surface {}!".format(surf_name))
                    else:
                        try:
                            from src.misc.dural import create_dural
                            create_dural.create_dural_surface(subject, SUBJECTS_DIR)
                            verts, faces = utils.read_pial(subject, MMVT_DIR, hemi)
                        except:
                            verts, faces = None, None
                        # print('No dural surf! Run the following command from ielu folder')
                        # print('''python2 -c "from ielu import pipeline as pipe; pipe.create_dural_surface(subject='{}')"'''.format(subject))
                        # continue
                if surf_type == 'inflated':
                    verts_offset = 55 if hemi == 'rh' else -55
                    verts[:, 0] = verts[:, 0] + verts_offset
                if verts is not None:
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
        labels_plys = glob.glob(op.join(MMVT_DIR, subject, 'labels', '{}.pial.{}'.format(atlas, hemi), '*.ply'))
        if len(labels_plys) > 0:
            faces_verts_dic_fnames = [op.join(MMVT_DIR, subject, 'labels', '{}.pial.{}'.format(atlas, hemi), '{}_faces_verts.npy'.format(
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


@utils.tryit()
def load_bem_surfaces(subject, include_seghead=True, overwrite=False):
    watershed_files = ['brain_surface', 'inner_skull_surface', 'outer_skin_surface', 'outer_skull_surface']
    watershed_file_names = [op.join(SUBJECTS_DIR, subject, 'bem', 'watershed', '{}_{}'.format(subject, watershed_name))
                            for watershed_name in watershed_files]
    if not all([op.isfile(f) for f in watershed_file_names]):
        print('Not all the watershed files exist!')
        return False
    ret = True
    if include_seghead:
        seghead_fname = op.join(SUBJECTS_DIR, subject, 'surf', 'lh.seghead')
        if not op.isfile(seghead_fname):
            from src.utils import freesurfer_utils as fu
            fu.create_seghead(subject)
        if op.isfile(seghead_fname):
            watershed_file_names.append(seghead_fname)
            watershed_files.append('seghead')
        else:
            print('No seghead!')
    for surf_fname, watershed_name in zip(watershed_file_names, watershed_files):
        ply_fname = op.join(MMVT_DIR, subject, 'surf', '{}.ply'.format(watershed_name))
        verts, faces = nib_fs.read_geometry(surf_fname)
        if not op.isfile(ply_fname) or overwrite:
            utils.write_ply_file(verts, faces, ply_fname)
        faces_verts_fname = op.join(MMVT_DIR, subject, 'surf', '{}_faces_verts.npy'.format(watershed_name))
        if not op.isfile(faces_verts_fname):
            utils.calc_ply_faces_verts(verts, faces, faces_verts_fname, overwrite, watershed_name)
        ret = ret and op.isfile(ply_fname) and op.isfile(faces_verts_fname)
    return ret


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

#
# @utils.timeit
# def convert_perecelated_cortex(subject, atlas, surf_type='pial', overwrite_ply_files=False, hemi='both'):
#     lookup = {}
#     for hemi in lu.get_hemis(hemi):
#         lookup[hemi] = create_labels_lookup(subject, hemi, atlas)
#         if len(lookup[hemi]) == 0:
#             continue
#         mat_fol = op.join(SUBJECTS_DIR, subject, '{}.{}.{}'.format(atlas, surf_type, hemi))
#         ply_fol = op.join(SUBJECTS_DIR, subject, '{}_{}_{}_ply'.format(atlas, surf_type, hemi))
#         utils.make_dir(op.join(MMVT_DIR, subject, 'labels'))
#         blender_fol = op.join(MMVT_DIR, subject, 'labels', '{}.{}.{}'.format(atlas, surf_type, hemi))
#         # utils.convert_mat_files_to_ply(mat_fol, overwrite_ply_files)
#         # rename_cortical(lookup, mat_fol, ply_fol)
#         # if surf_type == 'inflated':
#         #     for ply_fname in glob.glob(op.join(ply_fol, '*.ply')):
#         #         verts, faces = utils.read_ply_file(ply_fname)
#         #         verts_offset = 55 if hemi == 'rh' else -55
#         #         verts[:, 0] = verts[:, 0] + verts_offset
#         #         utils.write_ply_file(verts, faces, ply_fname)
#         # utils.rmtree(blender_fol)
#         # shutil.copytree(ply_fol, blender_fol)
#         # utils.rmtree(mat_fol)
#         # utils.rmtree(ply_fol)
#     return lookup


def create_annotation(subject, atlas='aparc250', fsaverage='fsaverage', remote_subject_dir='',
        overwrite_annotation=False, overwrite_morphing=False, do_solve_labels_collisions=False,
        morph_labels_from_fsaverage=True, fs_labels_fol='', save_annot_file=True, surf_type='inflated',
        overwrite_vertices_labels_lookup=False, n_jobs=6):
    annotation_fname_template = op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot'.format('{hemi}', atlas))
    annotations_exist = utils.both_hemi_files_exist(annotation_fname_template)
    if annotations_exist and not overwrite_annotation:
        # lu.fix_unknown_labels(subject, atlas)
        # check the annot files:
        annot_ok = True
        for hemi in utils.HEMIS:
            try:
                labels = lu.read_labels_from_annot(annotation_fname_template.format(hemi=hemi))
            except:
                labels = []
            annot_ok = annot_ok and len(labels) > 1
        if annot_ok:
            print('The annotation file already exists ({})'.format(annotation_fname_template))
            return True

    labels_files = glob.glob(op.join(SUBJECTS_DIR, subject, 'label', atlas, '*.label'))
    # If there are only 2 files, most likely it's the unknowns
    if len(labels_files) <= 3:
        backup_labels_fol = op.join(SUBJECTS_DIR, subject, 'label', '{}_before_solve_collision'.format(atlas))
        labels_files = glob.glob(op.join(backup_labels_fol, '*.label')) if \
            op.isdir(backup_labels_fol) else []
    if save_annot_file and len(labels_files) > 3:
        annot_was_written = labels_to_annot(
            subject, atlas, overwrite_annotation, surf_type, overwrite_vertices_labels_lookup, labels_files,
            n_jobs=n_jobs)
        if annot_was_written:
            return True
    utils.make_dir(op.join(SUBJECTS_DIR, subject, 'label'))
    remote_annotations_exist = np.all([op.isfile(op.join(remote_subject_dir, 'label', '{}.{}.annot'.format(
        hemi, atlas))) for hemi in HEMIS])
    if remote_annotations_exist and not overwrite_annotation:
        for hemi in HEMIS:
            remote_fname = op.join(remote_subject_dir, 'label', '{}.{}.annot'.format(hemi, atlas))
            local_fname = op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot'.format(hemi, atlas))
            if remote_fname != local_fname:
                shutil.copy(remote_fname, local_fname)
        return True

    if fu.is_fs_atlas(atlas):
        morph_labels_from_fsaverage = False
        do_solve_labels_collisions = False
        save_annot_file = False
        if not utils.both_hemi_files_exist(annotation_fname_template): # or overwrite_annotation:
            utils.make_dir(op.join(SUBJECTS_DIR, subject, 'label'))
            annotations_exist = fu.create_annotation_file(
                subject, atlas, subjects_dir=SUBJECTS_DIR, freesurfer_home=FREESURFER_HOME)
    if morph_labels_from_fsaverage:
        ret = lu.morph_labels_from_fsaverage(subject, SUBJECTS_DIR, MMVT_DIR, atlas, n_jobs=n_jobs,
            fsaverage=fsaverage, overwrite=overwrite_morphing, fs_labels_fol=fs_labels_fol)
        if not ret:
            return False
    if do_solve_labels_collisions:
        solve_labels_collisions(subject, atlas, surf_type, overwrite_vertices_labels_lookup, n_jobs)
    if save_annot_file and (overwrite_annotation or not annotations_exist):
        labels_to_annot(subject, atlas, overwrite_annotation, surf_type, overwrite_vertices_labels_lookup,
                        n_jobs=n_jobs)
    if save_annot_file:
        return both_annot_files_exist(subject, atlas)
    else:
        return len(glob.glob(op.join(SUBJECTS_DIR, subject, 'label', atlas, '*.label'))) > 0 or \
               both_annot_files_exist(subject, atlas)


def both_annot_files_exist(subject, atlas):
    return utils.both_hemi_files_exist(op.join(
        SUBJECTS_DIR, subject, 'label', '{}.{}.annot'.format('{hemi}', atlas)))


def labels_to_annot(subject, atlas, overwrite_annotation=False, surf_type='inflated',
                    overwrite_vertices_labels_lookup=False, labels=(), n_jobs=6):
    if fu.is_fs_atlas(atlas) and both_annot_files_exist(subject, atlas):
        return True
    annot_was_created = lu.labels_to_annot(subject, SUBJECTS_DIR, atlas, labels=labels, overwrite=overwrite_annotation)
    if not annot_was_created:
        print("Can't write labels to annotation! Trying to solve labels collision")
        print(traceback.format_exc())
        ret = solve_labels_collisions(subject, atlas, surf_type, overwrite_vertices_labels_lookup, n_jobs)
        if not ret:
            print('An error occurred in solve_labels_collisions!')
            return False
        annot_was_created = lu.labels_to_annot(subject, SUBJECTS_DIR, atlas, overwrite=overwrite_annotation)
        if not annot_was_created:
            print("Can't write labels to annotation! Solving the labels collision didn't help...")
            return False
    return annot_was_created


def solve_labels_collisions(subject, atlas, surf_type='inflated', overwrite_vertices_labels_lookup=False, n_jobs=6):
    backup_labels_fol = '{}_before_solve_collision'.format(atlas)
    ret = lu.solve_labels_collision(subject, atlas, SUBJECTS_DIR, MMVT_DIR, backup_labels_fol,
                              overwrite_vertices_labels_lookup, surf_type, n_jobs)
    lu.backup_annotation_files(subject, SUBJECTS_DIR, atlas)
    labels_to_annot(subject, atlas, overwrite_annotation=False, surf_type=surf_type,
                    overwrite_vertices_labels_lookup=False, n_jobs=n_jobs)
    return ret


@utils.timeit
def parcelate_cortex(subject, atlas, overwrite=False, overwrite_annotation=False,
                     overwrite_vertices_labels_lookup=False, surf_type='inflated', n_jobs=6):
    utils.make_dir(op.join(MMVT_DIR, subject, 'labels'))
    labels_to_annot(subject, atlas, overwrite_annotation, surf_type, overwrite_vertices_labels_lookup,
                    n_jobs=n_jobs)
    vertices_labels_ids_lookup = lu.create_vertices_labels_lookup(
        subject, atlas, True, overwrite_vertices_labels_lookup)
    params = []
    for surface_type in ['pial', 'inflated']:
        files_exist = True
        for hemi in HEMIS:
            blender_labels_fol = op.join(MMVT_DIR, subject, 'labels', '{}.{}.{}'.format(atlas, surface_type, hemi))
            labels = lu.read_labels(subject, SUBJECTS_DIR, atlas, hemi=hemi)
            files_exist = files_exist and op.isdir(blender_labels_fol) and \
                len(glob.glob(op.join(blender_labels_fol, '*.ply'))) >= len(labels)
            if overwrite or not files_exist:
                params.append((subject, atlas, hemi, surface_type, vertices_labels_ids_lookup[hemi],
                               overwrite_vertices_labels_lookup))

    if len(params) > 0:
        if n_jobs > 1:
            results = utils.run_parallel(_parcelate_cortex_parallel, params, njobs=n_jobs)
        else:
            results = [_parcelate_cortex_parallel(p) for p in params]
        return all(results)
    else:
        return True


def _parcelate_cortex_parallel(p):
    from src.preproc import parcelate_cortex
    subject, atlas, hemi, surface_type, vertices_labels_ids_lookup, overwrite_vertices_labels_lookup = p
    print('Parcelate the {} {} cortex'.format(hemi, surface_type))
    return parcelate_cortex.parcelate(subject, atlas, hemi, surface_type, vertices_labels_ids_lookup,
                                      overwrite_vertices_labels_lookup)


def save_matlab_labels_vertices(subject, atlas):
    for hemi in HEMIS:
        matlab_fname = op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot_labels.m'.format(hemi, atlas))
        if op.isfile(matlab_fname):
            labels_dic = matlab_utils.matlab_cell_arrays_to_dict(matlab_fname)
            utils.save(labels_dic, op.join(MMVT_DIR, subject, 'labels_dic_{}_{}.pkl'.format(atlas, hemi)))
        else:
            return False
    return True


@utils.tryit()
def save_labels_vertices(subject, atlas, overwrite=False):
    output_fname = op.join(MMVT_DIR, subject, 'labels_vertices_{}.pkl'.format(atlas))
    if op.isfile(output_fname) and not overwrite:
        return True
    labels = lu.read_labels(subject, SUBJECTS_DIR, atlas, sorted_according_to_annot_file=True,
                            read_only_from_annot=True)
    if len(labels) == 0:
        labels = lu.read_labels(subject, SUBJECTS_DIR, atlas)
    labels_names, labels_vertices = defaultdict(list), defaultdict(list)
    for label in labels:
        labels_names[label.hemi].append(label.name)
        labels_vertices[label.hemi].append(label.vertices)
    utils.save((labels_names, labels_vertices), output_fname)
    return op.isfile(output_fname)


@utils.tryit()
def create_spatial_connectivity(subject, surf_types=('pial', 'dural'), overwrite=False):
    ret = True
    for surf in surf_types:
        verts_neighbors_fname = op.join(MMVT_DIR, subject, 'verts_neighbors{}_{}.pkl'.format(
            '' if surf == 'pial' else '_{}'.format(surf), '{hemi}'))
        connectivity_fname = op.join(MMVT_DIR, subject, 'spatial_connectivity{}.pkl'.format(
            '' if surf == 'pial' else '_{}'.format(surf)))
        if utils.both_hemi_files_exist(verts_neighbors_fname) and op.isfile(connectivity_fname) and not overwrite:
            continue
        connectivity_per_hemi = {}
        for hemi in utils.HEMIS:
            neighbors = defaultdict(list)
            pial_fname = op.join(MMVT_DIR, subject, 'surf', '{}.{}.ply'.format(hemi, surf))
            if not op.isfile(pial_fname):
                create_surfaces(subject)
            if not op.isfile(pial_fname):
                print('{} does not exist!'.format(pial_fname))
                continue
            _, faces = utils.read_pial(subject, MMVT_DIR, hemi, surface_type=surf)
            # if not op.isfile(conn_fname):
            #     print("Connectivity file doesn't exist! {}".format(conn_fname))
            #     continue
            # d = np.load(conn_fname)
            connectivity_per_hemi[hemi] = mne.spatial_tris_connectivity(faces)
            rows, cols = connectivity_per_hemi[hemi].nonzero()
            for ind in range(len(rows)):
                neighbors[rows[ind]].append(cols[ind])
            utils.save(neighbors, verts_neighbors_fname.format(hemi=hemi))
        utils.save(connectivity_per_hemi, connectivity_fname)
        ret = ret and op.isfile(connectivity_fname)
    return ret


def calc_three_rois_intersection(subject, rois, output_fol='', model_name='', atlas='aparc.DKTatlas', debug=False,
                                 overwrite=False):

    def get_vertices_between_labels(label1_contours_verts_inds, label2_contours_verts_inds, vertices_neighbros):
        vertices = []
        for vert_ind in label1_contours_verts_inds:
            for neighbor_vert in vertices_neighbros[vert_ind]:
                if neighbor_vert in label2_contours_verts_inds:
                    vertices.append(vert_ind)
        return set(vertices)

    if output_fol != '':
        output_fol = utils.make_dir(output_fol)
        if model_name == '':
            model_name = '{}_intersection.pkl'.format('_'.join(rois))
        output_fname = op.join(output_fol, model_name)
        if op.isfile(output_fname) and not overwrite:
            return utils.load(output_fname)

    verts_neighbors_fname = op.join(MMVT_DIR, subject, 'verts_neighbors_{hemi}.pkl')
    if not utils.both_hemi_files_exist(verts_neighbors_fname):
        print('calc_labeles_contours: You should first run create_spatial_connectivity')
        create_spatial_connectivity(subject)

    contours = op.join(MMVT_DIR, subject, 'labels', '{}_contours_{}.npz'.format(atlas, '{hemi}'))
    if not utils.both_hemi_files_exist(contours):
        calc_labeles_contours(subject, atlas)

    intesection_points = {}
    for hemi in utils.HEMIS:
        d = np.load(contours.format(hemi=hemi))
        hemi_contours = d['contours']
        surf, _ = utils.read_pial(subject, MMVT_DIR, hemi)
        vertices_neighbors = np.load(verts_neighbors_fname.format(hemi=hemi))
        labels = lu.read_labels(subject, SUBJECTS_DIR, atlas, hemi=hemi)
        labels_names = [label.name for label in labels]
        labels_contoures_inds = [set(np.where(hemi_contours == labels_names.index('{}-{}'.format(roi, hemi)) + 1)[0]) \
                                 for roi in rois]
        vertices_in_between = \
            (get_vertices_between_labels(labels_contoures_inds[0], labels_contoures_inds[1], vertices_neighbors) & \
            get_vertices_between_labels(labels_contoures_inds[0], labels_contoures_inds[2], vertices_neighbors)) | \
            (get_vertices_between_labels(labels_contoures_inds[1], labels_contoures_inds[2], vertices_neighbors) & \
            get_vertices_between_labels(labels_contoures_inds[1], labels_contoures_inds[0], vertices_neighbors)) | \
            (get_vertices_between_labels(labels_contoures_inds[2], labels_contoures_inds[1], vertices_neighbors) & \
            get_vertices_between_labels(labels_contoures_inds[2], labels_contoures_inds[0], vertices_neighbors))
        vertices_coords = np.array([surf[verts] for verts in vertices_in_between])
        # dists = cdist(vertices_coords, vertices_coords)
        intesection_points[hemi] = np.mean(vertices_coords, axis=0)
        if debug:
            prefix = '_'.join(rois)
            for vert in vertices_in_between:
                new_label_name = '{}_{}_{}'.format(prefix, vert, )
                new_label = lu.grow_label(subject, vert, hemi, new_label_name, 3, 4)
                utils.make_dir(op.join(MMVT_DIR, subject, 'labels'))
                new_label_fname = op.join(MMVT_DIR, subject, 'labels', '{}.label'.format(new_label_name))
                new_label.save(new_label_fname)

    if output_fol != '':
        utils.save(intesection_points, output_fname)
    return intesection_points


def calc_flat_patch_cut_vertices(subject, atlas='aparc.DKTatlas', overwrite=True):
    output_fname = op.join(MMVT_DIR, subject, 'flat_patch_cut_vertices.pkl')
    if op.isfile(output_fname) and not overwrite:
        return True
    neighbors_regions_for_cut =[('posteriorcingulate', 'isthmuscingulate'), ('posteriorcingulate', 'precuneus'),
                                ('paracentral', 'precuneus'), ('isthmuscingulate', 'parahippocampal'),
                                ('isthmuscingulate', 'lingual'), ('precuneus', 'lingual'),
                                ('pericalcarine', 'lingual'), ('lateralorbitofrontal', 'medialorbitofrontal'),
                                ('superiortemporal', 'entorhinal'), ('superiortemporal', 'inferiortemporal'),
                                ('caudalanteriorcingulate', 'posteriorcingulate'),
                                ('superiorfrontal', 'posteriorcingulate'), ('paracentral', 'superiorfrontal')]

    verts_neighbors_fname = op.join(MMVT_DIR, subject, 'verts_neighbors_{hemi}.pkl')
    if not utils.both_hemi_files_exist(verts_neighbors_fname):
        print('calc_labeles_contours: You should first run create_spatial_connectivity')
        create_spatial_connectivity(subject)
        # return calc_labeles_contours(subject, atlas, overwrite, verbose)
    # vertices_labels_lookup = lu.create_vertices_labels_lookup(subject, atlas, False, overwrite)
    bad_vertices = {}

    unknown_labels = lu.create_unknown_labels(subject, atlas)
    contours_tempalte = op.join(MMVT_DIR, subject, 'labels', '{}_contours_{}.npz'.format(atlas, '{hemi}'))
    if not utils.both_hemi_files_exist(contours_tempalte):
        calc_labeles_contours(subject, atlas)
    for hemi in utils.HEMIS:
        d = np.load(contours_tempalte.format(hemi=hemi))
        vertices_neighbors = np.load(verts_neighbors_fname.format(hemi=hemi))
        labels = lu.read_labels(subject, SUBJECTS_DIR, atlas, hemi=hemi)
        bad_vertices_hemi = []
        for regions_pair in neighbors_regions_for_cut:
            # print('Working on {} and {} '.format(regions_pair[0], regions_pair[1]))
            cur_bad_vertices = get_vertices_between_labels(
                hemi, regions_pair[0], regions_pair[1], labels, vertices_neighbors, d['contours'])
            # print('found {} bad vertices'.format(len(cur_bad_vertices)))
            bad_vertices_hemi.extend(cur_bad_vertices)

        # unknon_ind = [label.name for label in labels].index('unknown-{}'.format(hemi))


        # output_fname_seems_neighbors = op.join(MMVT_DIR, subject, 'neighbros_of_seems_{}.npz'.format('{hemi}'))
        # get_neighbros_of_seems(np.arange(0, len(vertices_neighbors)), bad_vertices_hemi, vertices_neighbors,
        #                        output_fname_seems_neighbors, hemi)

        bad_vertices_hemi.extend(unknown_labels[hemi].vertices)
        bad_vertices[hemi] = bad_vertices_hemi

    utils.save(bad_vertices, output_fname)
    print(output_fname)

        # for label_ind, label in enumerate(labels):
        #     if verbbad_verticesose:
        #         label_nei = np.zeros((len(label.vertices)))
        #     for vert_ind, vert in enumerate(label.vertices):
        #         nei = set([vertices_labels_lookup[hemi].get(v, '') for v in vertices_neighbors[vert]]) - set([''])
        #         contours[vert] = label_ind + 1 if len(nei) > 1 else 0
        #         if verbose:
        #             label_nei[vert_ind] = contours[vert]
        #     if verbose:
        #         print(label.name, len(np.where(label_nei)[0]) / len(verts))
        # np.savez(output_fname.format(hemi=hemi), contours=contours, max=len(labels),
        #          labels=[l.name for l in labels])
    return op.isfile(output_fname)


def get_vertices_between_labels(hemi, label1, label2, labels, vertices_neighbros, contours):
    vertices_between_labels = []
    label1_code = [label.name for label in labels].index('{}-{}'.format(label1, hemi))+1
    label2_code = [label.name for label in labels].index('{}-{}'.format(label2, hemi))+1

    label1_contours_verts_inds = np.where(contours == label1_code)
    label2_contours_verts_inds = np.where(contours == label2_code)

    for vert_ind in label1_contours_verts_inds[0]:
        for neighbor_vert in vertices_neighbros[vert_ind]:
            if neighbor_vert in label2_contours_verts_inds[0]:
                vertices_between_labels.append(vert_ind)
                break
    if len(vertices_between_labels)==0:
        print('empty vertices_between_labels for pair {} and {}'.format(label1, label2))
    return vertices_between_labels


# def get_neighbros_of_seems(all_verts,seem_verts,vertices_neighbors,output_fname,hemi):
#     not_seem_verts = set(all_verts)-set(seem_verts)
#
#     seems_neighbor_verts = []
#     for vert in seem_verts:
#         for vert_neighbor in vertices_neighbors[vert]:
#             seems_neighbor_verts.extend(vert_neighbor)
#
#     d = np.load(op.join(MMVT_DIR, subject, '{}_contours_{}.npz'.format(atlas, hemi)))
#     np.savez(output_fname.format(hemi=hemi), seems_neighbor_verts=seems_neighbor_verts)
#     return list(set(seems_neighbor_verts))


@utils.check_for_freesurfer
def create_flat_brain(subject, print_only=False, overwrite=False, n_jobs=2):
    patch_fname_template = op.join(SUBJECTS_DIR, subject, 'surf', '{}.inflated.patch'.format('{hemi}'))
    if not utils.both_hemi_files_exist(patch_fname_template) or overwrite:
        flat_patch_cut_vertices_fname = op.join(MMVT_DIR, subject, 'flat_patch_cut_vertices.pkl')
        calc_flat_patch_cut_vertices(subject)
        flat_patch_cut_vertices = utils.load(flat_patch_cut_vertices_fname)
        for hemi in utils.HEMIS:
            patch_fname = patch_fname_template.format(hemi=hemi)
            if op.isfile(patch_fname) and not overwrite:
                continue
            inf_verts, _ = nib.freesurfer.read_geometry(
                op.join(SUBJECTS_DIR, subject, 'surf', '{}.inflated'.format(hemi)))
            flat_patch_cut_vertices_hemi = set(flat_patch_cut_vertices[hemi])
            fu.write_patch(patch_fname, [(ind, v) for ind, v in enumerate(inf_verts)
                                         if ind not in flat_patch_cut_vertices_hemi])
    params = [(subject, hemi, overwrite, print_only) for hemi in utils.HEMIS]
    results = utils.run_parallel(_flat_brain_parallel, params, n_jobs)
    return all(results)


def _flat_brain_parallel(p):
    subject, hemi, overwrite, print_only = p
    flat_patch_fname = fu.get_flat_patch_fname(subject, hemi, SUBJECTS_DIR)
    if not op.isfile(flat_patch_fname) or overwrite:
        flat_patch_fname = fu.flat_brain(subject, hemi, SUBJECTS_DIR, print_only)
    return write_flat_brain_patch(subject, hemi, flat_patch_fname)


def write_flat_brain_patch(subject, hemi, flat_patch_fname):
    ply_fname = op.join(MMVT_DIR, subject, 'surf', '{}.flat.pial.ply'.format(hemi))
    flat_verts, flat_faces = fu.read_patch(
        subject, hemi, SUBJECTS_DIR, surface_type='inflated', patch_fname=flat_patch_fname)
    bad_vertices = np.setdiff1d(np.arange(len(flat_verts)), flat_faces)
    valid_vertices = np.unique(flat_faces)

    # flat_verts *= 0.1
    flat_verts_norm = utils.remove_mean_columnwise(flat_verts, valid_vertices)
    flat_verts_norm[bad_vertices] = 0
    # for vert_ind in list(bad_vertices):
    #     flat_verts_norm[vert_ind] = (0.0, 0.0, 0.0)

    # flat_verts = np.roll(flat_verts, -1, 1)
    flat_verts = flat_verts[:, [1, 2, 0]]
    flat_verts[:, 0] *= -0.5 #* (10 if hemi == 'rh' else -10)
    flat_verts[:, 1] = 0 #100 if hemi == 'rh' else -100
    flat_verts[:, 2] *= -0.5


    # for vert in cur_obj.data.vertices:
        # shapekey.data[vert.index].co = (flat_verts[vert.index, 1] * -10 + 200 * flatmap_orientation, 0, flat_verts[vert.index, 0] * -10)
        # shapekey.data[vert.index].co = (flat_verts_norm[vert.index, 1] * -5, 0, flat_verts_norm[vert.index, 0] * -5)

    return utils.write_ply_file(flat_verts, flat_faces, ply_fname, True)


@utils.tryit(False, False)
def calc_labeles_contours(subject, atlas, overwrite=True, verbose=False):
    utils.make_dir(op.join(MMVT_DIR, subject, 'labels'))
    output_fname = op.join(MMVT_DIR, subject, 'labels', '{}_contours_{}.npz'.format(atlas, '{hemi}'))
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
        labels = [l for l in labels if 'unknown' not in l.name]
        for label_ind, label in enumerate(labels):
            if verbose:
                label_nei = np.zeros((len(label.vertices)))
            for vert_ind, vert in enumerate(label.vertices):
                if vert >= len(verts):
                    continue
                nei = set([vertices_labels_lookup[hemi].get(v, '') for v in vertices_neighbors[vert]]) - set([''])
                contours[vert] = label_ind + 1 if len(nei) > 1 else 0
                if verbose:
                    label_nei[vert_ind] = contours[vert]
            if verbose:
                print(label.name, len(np.where(label_nei)[0]) / len(verts))
        if utils.both_hemi_files_exist(op.join(SUBJECTS_DIR, subject, 'surf', '{hemi}.sphere')):
            centers = [l.center_of_mass(restrict_vertices=True) for l in labels]
        else:
            centers = None
        np.savez(output_fname.format(hemi=hemi), contours=contours, max=len(labels),
                 labels=[l.name for l in labels], centers=centers)
    return utils.both_hemi_files_exist(output_fname)


@utils.timeit
def create_verts_faces_lookup(subject, surface_type='pial'):
    output_fname = op.join(MMVT_DIR, subject, 'faces_verts_lookup_{}{}.pkl'.format(
        '{hemi}', '' if surface_type == 'pial' else '{}_'.format(surface_type)))
    if utils.both_hemi_files_exist(output_fname):
        return {hemi:utils.load(output_fname.format(hemi=hemi)) for hemi in utils.HEMIS}

    for hemi in utils.HEMIS:
        if op.isfile(output_fname.format(hemi=hemi)):
            continue
        verts, faces = utils.read_pial(subject, MMVT_DIR, hemi, surface_type)
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
    contours_fname = op.join(MMVT_DIR, subject, 'labels', '{}_contours_{}.npz'.format(atlas, '{hemi}'))
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


def calc_labels_center_of_mass(subject, atlas, overwrite=False):
    com_fname = op.join(SUBJECTS_DIR, subject, 'label', '{}_center_of_mass.pkl'.format(atlas))
    if op.isfile(com_fname) and not overwrite:
        return True
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
    output_fname_template = op.join(MMVT_DIR, subject, '{}_trans.npz')
    for image_name in ['T1.mgz', 'T2.mgz']:
        header = tu.get_subject_mri_header(subject, SUBJECTS_DIR, image_name)
        if header is None:
            continue
        output_fname = output_fname_template.format(utils.namebase(image_name.lower()))
        ras_tkr2vox, vox2ras_tkr, vox2ras, ras2vox = get_trans_functions(header)
        print('save_subject_orig_trans: saving {}'.format(output_fname))
        np.savez(output_fname, ras_tkr2vox=ras_tkr2vox, vox2ras_tkr=vox2ras_tkr, vox2ras=vox2ras, ras2vox=ras2vox)
    return op.isfile(output_fname_template.format('t1'))


def get_trans_functions(header):
    vox2ras_tkr = header.get_vox2ras_tkr()
    ras_tkr2vox = np.linalg.inv(vox2ras_tkr)
    vox2ras = header.get_vox2ras()
    ras2vox = np.linalg.inv(vox2ras)
    return ras_tkr2vox, vox2ras_tkr, vox2ras, ras2vox


def calc_3d_atlas(subject, atlas, overwrite_aseg_file=True):
    from src.preproc import freeview as fr
    aparc_ret = fr.create_aparc_aseg_file(subject, atlas, overwrite_aseg_file)
    lut_ret = fr.create_lut_file_for_atlas(subject, atlas)
    return aparc_ret and lut_ret


@utils.tryit()
def create_high_level_atlas(subject, high_level_atlas_name='high.level.atlas', base_atlas='aparc.DKTatlas',
                            overwrite=False):
    if not utils.both_hemi_files_exist(op.join(SUBJECTS_DIR, subject, 'label', '{hemi}.aparc.DKTatlas.annot')):
        fu.create_annotation_file(
            subject, base_atlas, subjects_dir=SUBJECTS_DIR, freesurfer_home=FREESURFER_HOME)
    look = create_labels_names_lookup(subject, base_atlas)
    if len(look) == 0:
        return False
    csv_fname = op.join(MMVT_DIR, '{}.csv'.format(high_level_atlas_name))
    if not op.isfile(csv_fname):
        csv_fname = op.join(MMVT_DIR, '{}.csv'.format(high_level_atlas_name.replace('.', '_')))
        if not op.isfile(csv_fname):
            print('No {}.csv in {}!'.format(high_level_atlas_name, MMVT_DIR))
            return False
    labels = []
    try:
        for hemi in utils.HEMIS:
            for line in utils.csv_file_reader(csv_fname, ','):
                if len(line) == 0:
                    continue
                elif len(line) > 1:
                    new_label = lu.join_labels('{}-{}'.format(line[0], hemi), (look['{}-{}'.format(l, hemi)] for l in line[1:]))
                else:
                    new_label = look['{}-{}'.format(line[0], hemi)]
                labels.append(new_label)
    except:
        return False
    lu.labels_to_annot(subject, SUBJECTS_DIR, high_level_atlas_name, labels=labels, overwrite=True)
    save_labels_vertices(subject, high_level_atlas_name, overwrite)
    create_spatial_connectivity(subject, ['pial'], overwrite)
    calc_labeles_contours(subject, high_level_atlas_name, overwrite)
    calc_labels_center_of_mass(subject, high_level_atlas_name, overwrite)
    return utils.both_hemi_files_exist(op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot'.format(
        '{hemi}', high_level_atlas_name)))


def create_labels_names_lookup(subject, atlas):
    lookup = {}
    labels = lu.read_labels(subject, SUBJECTS_DIR, atlas)
    for label in labels:
        lookup[label.name] = label
    return lookup


def create_new_subject_blend_file(subject, atlas, overwrite_blend=False, ask_if_overwrite_blend=True):
    from src.mmvt_addon.scripts import create_new_subject as mmvt_script
    # Create a file for the new subject
    args = mmvt_script.create_new_subject(subject, atlas, overwrite_blend)
    if args is not None:
        utils.waits_for_file(args.log_fname)
    atlas = utils.get_real_atlas_name(atlas, short_name=True)
    new_fname = op.join(MMVT_DIR, '{}_{}.blend'.format(subject, atlas))
    return op.isfile(new_fname)
    # empty_subject_fname = op.join(MMVT_DIR, 'empty_subject.blend')
    # if not op.isfile(empty_subject_fname):
    #     resources_dir = op.join(utils.get_parent_fol(levels=2), 'resources')
    #     shutil.copy(op.join(resources_dir, 'empty_subject.blend'), empty_subject_fname)
    # if op.isfile(new_fname) and not overwrite_blend:
    #     if ask_if_overwrite_blend:
    #         overwrite = input('The file {} already exist, do you want to overwrite? '.format(new_fname))
    #         if au.is_true(overwrite):
    #            os.remove(new_fname)
    #            shutil.copy(op.join(MMVT_DIR, 'empty_subject.blend'), new_fname)
    # else:
    #     shutil.copy(empty_subject_fname, new_fname)


def check_bem(subject, remote_subject_dir, recreate_src_spacing, recreate_bem_solution=False, args={}):
    from src.preproc import meg
    meg_args = meg.read_cmd_args(dict(subject=subject))
    meg_args.update(args)
    meg.init(subject, meg_args, remote_subject_dir=remote_subject_dir)
    # args.remote_subject_dir = remote_subject_dir
    bem_exist, _ = meg.check_bem(subject, recreate_src_spacing, remote_subject_dir, recreate_bem_solution, meg_args)
    return bem_exist


def full_extent(ax, pad=0.0):
    from matplotlib.transforms import Bbox
    """Get the full extent of an axes, including axes labels, tick labels, and titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = [ax, *ax.texts]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)

# @utils.ignore_warnings
def create_slices(subject, xyz, modality='mri', header=None, data=None):
    import matplotlib
    try:
        # Force matplotlib to not use any Xwindows backend.
        matplotlib.use('Agg')
    except:
        pass
    import matplotlib.pyplot as plt
    from nibabel.orientations import axcodes2ornt, aff2axcodes
    from nibabel.affines import voxel_sizes

    """ Function to display row of image slices """
    if modality == 'mri':
        fname = op.join(MMVT_DIR, subject, 'freeview', 'T1.mgz')
        if not op.isfile(fname):
            subjects_fname = op.join(SUBJECTS_DIR, subject, 'mri', 'T1.mgz')
            if op.isfile(subjects_fname):
                shutil.copy(subjects_fname, fname)
            else:
                print("Can't find subject's T1.mgz!")
                return False
    elif modality == 'ct':
        fname = op.join(MMVT_DIR, subject, 'freeview', 'ct.mgz')
        if not op.isfile(fname):
            subjects_fname = op.join(SUBJECTS_DIR, subject, 'mri', 'ct.mgz')
            if op.isfile(subjects_fname):
                shutil.copy(subjects_fname, fname)
            else:
                print("Can't find subject's CT! ({})".format(fname))
                return False
    else:
        print('create_slices: The modality {} is not supported!')
        return False
    if header is None or data is None:
        header = nib.load(fname)
        data = header.get_data()
    affine = np.array(header.affine, float)
    images_fol = op.join(MMVT_DIR, subject, 'figures', 'slices')
    utils.make_dir(images_fol)
    images_names = []

    percentiles_fname = op.join(MMVT_DIR, subject, 'freeview', '{}_1_99_precentiles.npy'.format(modality))
    if not op.isfile(percentiles_fname):
        clim = np.percentile(data, (1, 99))
        np.save(percentiles_fname, clim)
    else:
        clim = np.load(percentiles_fname)
    codes = axcodes2ornt(aff2axcodes(affine))
    order = np.argsort([c[0] for c in codes])
    flips = np.array([c[1] < 0 for c in codes])[order]
    flips[0] = not flips[0]
    sizes = [data.shape[order] for order in order]
    scalers = voxel_sizes(affine)
    x, y, z = xyz #.split(',')
    coordinates = np.rint(np.array([x, y, z])[order]).astype(int)
    # print('Creating slices for {}'.format(coordinates))

    r = [scalers[order[2]] / scalers[order[1]],
         scalers[order[2]] / scalers[order[0]],
         scalers[order[1]] / scalers[order[0]]]

    if utils.all_items_equall(sizes):
        fig = plt.figure()
        # fig.set_size_inches(1. * sizes[xax] / sizes[yax], 1, forward=False)
        fig.set_size_inches(1., 1, forward=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)


    # plt.style.use('dark_background')
    # fig, axes = plt.subplots(2, 2)
    # fig.set_size_inches((8, 8), forward=True)
    # axes = [axes[0, 0], axes[0, 1], axes[1, 0]]

    crosshairs = [dict()] * 3
    verts, horizs = [None] * 3, [None] * 3
    for ii, xax, yax in zip([0, 1, 2], [1, 0, 0], [2, 2, 1]):
        verts[ii] = np.array([[0] * 2, [-0.5, sizes[yax] - 0.5]]).T
        horizs[ii] = np.array([[-0.5, sizes[xax] - 0.5], [0] * 2]).T
    for ii, xax, yax in zip([0, 1, 2], [1, 0, 0], [2, 2, 1]):
        loc = coordinates[ii]
        if flips[ii]:
            loc = sizes[ii] - loc
        loc = [loc] * 2
        if ii == 0:
            verts[2][:, 0] = loc
            verts[1][:, 0] = loc
        elif ii == 1:
            horizs[2][:, 1] = loc
            verts[0][:, 0] = loc
        else:  # ii == 2
            horizs[1][:, 1] = loc
            horizs[0][:, 1] = loc

    for ii, xax, yax, ratio, prespective, label in zip(
            [0, 1, 2], [1, 0, 0], [2, 2, 1], r, ['sagital', 'coronal', 'axial'], ('SAIP', 'SLIR', 'ALPR')):
        # if not utils.all_items_equall(sizes):
        #     fig = plt.figure()
        #     fig.set_size_inches(1. * sizes[xax] / sizes[yax], 1, forward=True)
        #     ax = plt.Axes(fig, [0., 0., 1., 1.])
        #     # ax = plt.subplot(111)
        #     ax.set_axis_off()
        #     fig.add_axes(ax)
        # ax = axes[ii]
        d = get_image_data(data, order, flips, ii, coordinates)
        if d is None:
            continue
        if modality == 'ct':
            d[np.where(d == 0)] = -200
        ax.imshow(
            d, vmin=clim[0], vmax=clim[1], aspect=1,
            cmap='gray', interpolation='nearest', origin='lower')
        lims = [0, sizes[xax], 0, sizes[yax]]

        ln1, = ax.plot(horizs[ii].T[0], horizs[ii].T[1], color=(0, 1, 0), linestyle='-', linewidth=0.2)
        ln2, = ax.plot(verts[ii].T[0], verts[ii].T[1], color=(0, 1, 0), linestyle='-', linewidth=0.2)

        print('hline y={} vline x={}'.format(horizs[ii][0, 1], verts[ii][0, 0]))
        # ax.axhline(y=horizs[ii][0, 1], color='r', linestyle='-')
        # ax.axvline(x=verts[ii][0, 0], color='r', linestyle='-')

        # bump = 0.01
        # poss = [[lims[1] / 2., lims[3]],
        #         [(1 + bump) * lims[1], lims[3] / 2.],
        #         [lims[1] / 2., 0],
        #         [lims[0] - bump * lims[1], lims[3] / 2.]]
        # anchors = [['center', 'bottom'], ['left', 'center'],
        #            ['center', 'top'], ['right', 'center']]
        # for pos, anchor, lab in zip(poss, anchors, label):
        #     ax.text(pos[0], pos[1], lab, color='white',
        #             horizontalalignment=anchor[0],
        #             verticalalignment=anchor[1])

        ax.axis(lims)
        ax.set_aspect(ratio)
        ax.patch.set_visible(False)
        ax.set_frame_on(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.axes.get_xaxis().set_visible(False)
        # ax.set_facecolor('black')

        # extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        # extent = full_extent(ax).transformed(fig.dpi_scale_trans.inverted())

        image_fname = op.join(images_fol, '{}_{}.png'.format(modality, prespective))
        # print('Saving {}'.format(image_fname))
        plt.savefig(image_fname, dpi=sizes[xax]) # bbox_inches=extent
        ln1.remove()
        ln2.remove()
        images_names.append(image_fname)
    plt.close()
    with open(op.join(images_fol, '{}_slices.txt'.format(modality)), 'w') as f:
        f.write('Slices created for {}'.format(coordinates))
    return all([op.isfile(img) for img in images_names])


def get_image_data(image_data, order, flips, ii, pos):
    try:
        data = np.rollaxis(image_data, axis=order[ii])[pos[ii]]  # [data_idx] # [pos[ii]]
    except:
        return None
    xax = [1, 0, 0][ii]
    yax = [2, 2, 1][ii]
    if order[xax] < order[yax]:
        data = data.T
    if flips[xax]:
        data = data[:, ::-1]
    if flips[yax]:
        data = data[::-1]
    return data


def get_data_and_header(subject, image_name):
    # print('Loading header and data for {}, {}'.format(subject, modality))
    utils.make_dir(op.join(MMVT_DIR, subject, 'freeview'))
    fname = op.join(MMVT_DIR, subject, 'freeview', image_name)
    if not op.isfile(fname):
        subjects_fname = op.join(SUBJECTS_DIR, subject, 'mri', image_name)
        if op.isfile(subjects_fname):
            shutil.copy(subjects_fname, fname)
        else:
            print("Can't find subject's {}!".format(image_name))
            return None, None
    header = nib.load(fname)
    data = header.get_data()
    return data, header


def save_images_data_and_header(subject):
    modalities = {'T1.mgz':'mri', 'T2.mgz':'t2'}
    for image_name in modalities.keys():
        data, header = get_data_and_header(subject, image_name)
        if data is None or header is None:
            continue
        affine = header.affine
        precentiles = np.percentile(data, (1, 99))
        colors_ratio = 256 / (precentiles[1] - precentiles[0])
        output_fname = op.join(MMVT_DIR, subject, 'freeview', '{}_data.npz'.format(modalities[image_name]))
        if not op.isfile(output_fname):
            print('save_images_data_and_header: saving {}'.format(output_fname))
            np.savez(output_fname, data=data, affine=affine, precentiles=precentiles, colors_ratio=colors_ratio)
    return op.isfile(op.join(MMVT_DIR, subject, 'freeview', 'mri_data.npz'))


def create_pial_volume_mask(subject, overwrite=True):
    pial_output_fname = op.join(MMVT_DIR, subject, 'freeview', 'pial_vol_mask.npy')
    dural_output_fname = op.join(MMVT_DIR, subject, 'freeview', 'dural_vol_mask.npy')
    if op.isfile(pial_output_fname) and op.isfile(dural_output_fname) and not overwrite:
        print('The files are already exist! Use --overwrite 1 to overwrite')
        return True
    pial_verts = utils.load_surf(subject, MMVT_DIR, SUBJECTS_DIR)
    dural_verts, _ = fu.read_surface(subject, SUBJECTS_DIR, 'dural')
    t1_data, t1_header = get_data_and_header(subject, 'T1.mgz')
    if t1_header is None:
        return False
    ras_tkr2vox = np.linalg.inv(t1_header.get_header().get_vox2ras_tkr())
    pial_vol = np.zeros(t1_data.shape, dtype=np.uint8)
    dural_vol = np.zeros(t1_data.shape, dtype=np.uint8)
    for hemi in utils.HEMIS:
        hemi_pial_voxels = np.rint(utils.apply_trans(ras_tkr2vox, pial_verts[hemi])).astype(int)
        for vox in tqdm(hemi_pial_voxels):
            pial_vol[tuple(vox)] = 1
        if dural_verts is not None:
            hemi_dural_voxels = np.rint(utils.apply_trans(ras_tkr2vox, dural_verts[hemi])).astype(int)
            for vox in tqdm(hemi_dural_voxels):
                dural_vol[tuple(vox)] = 1
    print('{:.2f}% voxels are pial'.format(len(np.where(pial_vol)[0])/(t1_data.shape[0] * t1_data.shape[1])))
    np.save(pial_output_fname, pial_vol)
    if dural_verts is not None:
        print('{:.2f}% voxels are dural'.format(len(np.where(dural_vol)[0]) / (t1_data.shape[0] * t1_data.shape[1])))
        np.save(dural_output_fname, dural_vol)
    return op.isfile(pial_output_fname) and op.isfile(dural_output_fname)


def create_skull_surfaces(subject, surfaces_fol_name='bem', verts_in_ras=True):
    skull_fol = op.join(MMVT_DIR, subject, 'skull')
    utils.make_dir(skull_fol)
    errors = {}
    vertices_faces = defaultdict(list)
    t1_header = nib.load(op.join(SUBJECTS_DIR, subject, 'mri', 'T1.mgz')).header
    for skull_surf in ['inner_skull', 'outer_skull']:
        ply_fname = op.join(skull_fol, '{}.ply'.format(skull_surf))
        surf_fname = op.join(SUBJECTS_DIR, subject, surfaces_fol_name, '{}.surf'.format(skull_surf))
        if op.isfile(surf_fname):
            verts, faces = nib_fs.read_geometry(surf_fname)
            if verts_in_ras:
                vox = utils.apply_trans(np.linalg.inv(t1_header.get_vox2ras()), verts)
                verts = utils.apply_trans(t1_header.get_vox2ras_tkr(), vox)
            utils.write_ply_file(verts, faces, ply_fname, True)
            faces_verts_fname = op.join(skull_fol, 'faces_verts_{}.npy'.format(skull_surf))
            errors = utils.calc_ply_faces_verts(
                verts, faces, faces_verts_fname, False, utils.namebase(ply_fname), errors)
            for face_ind, face in enumerate(faces):
                for vert in face:
                    vertices_faces[vert].append(face_ind)
            utils.save(vertices_faces, op.join(skull_fol, '{}_vertices_faces.pkl'.format(skull_surf)))
        else:
            print('No {} surface in bem folder'.format(subject))
            return False

    if len(errors) > 0:
        for k, message in errors.items():
            print('{}: {}'.format(k, message))

    return all([op.isfile(op.join(skull_fol, '{}.ply'.format(skull_surf))) and \
                op.isfile(op.join(skull_fol, 'faces_verts_{}.npy'.format(skull_surf))) \
                for skull_surf in ['inner_skull', 'outer_skull']])


def copy_sphere_reg_files(subject):
    # If the user is planning to plot the stc file, it needs also the ?h.sphere.reg files
    tempalte = op.join(SUBJECTS_DIR, subject, 'surf', '{}.sphere.reg'.format('{hemi}'))
    if utils.both_hemi_files_exist(tempalte):
        for hemi in utils.HEMIS:
            mmvt_fname = op.join(MMVT_DIR, subject, 'surf', '{}.sphere.reg'.format(hemi))
            utils.make_dir(op.join(MMVT_DIR, subject, 'surf'))
            if not op.isfile(mmvt_fname):
                shutil.copy(tempalte.format(hemi=hemi), mmvt_fname)
    else:
        print("No ?h.sphere.reg files! You won't be able to plot stc files")


def check_labels(subject, atlas):
    return lu.check_labels(subject, atlas, SUBJECTS_DIR, MMVT_DIR)


def morph_labels_from_fsaverage(subject, atlas, fsaverage, overwrite_morphing, fs_labels_fol, n_jobs):
    return lu.morph_labels_from_fsaverage(
        subject, SUBJECTS_DIR, MMVT_DIR, atlas, fsaverage=fsaverage, overwrite=overwrite_morphing,
        fs_labels_fol=fs_labels_fol, n_jobs=n_jobs)


def call_main(args):
    pu.run_on_subjects(args, main)


def main(subject, remote_subject_dir, org_args, flags):
    args = utils.Bag({k: copy.deepcopy(org_args[k]) for k in org_args.keys()})
    copy_sphere_reg_files(subject)

    if utils.should_run(args, 'create_surfaces'):
        # *) convert rh.pial and lh.pial to rh.pial.ply and lh.pial.ply
        flags['create_surfaces'] = create_surfaces(subject, args.surf_type, overwrite=args.overwrite_hemis_ply)

    if utils.should_run(args, 'create_annotation'):
        # *) Create annotation file from fsaverage
        flags['create_annotation'] = create_annotation(
            subject, args.atlas, args.template_subject, remote_subject_dir, args.overwrite_annotation,
            args.overwrite_morphing_labels, args.solve_labels_collisions, args.morph_labels_from_fsaverage,
            args.fs_labels_fol, args.save_annot_file, args.solve_labels_collision_surf_type,
            args.overwrite_vertices_labels_lookup, args.n_jobs)

    if utils.should_run(args, 'parcelate_cortex'):
        flags['parcelate_cortex'] = parcelate_cortex(
            subject, args.atlas, args.overwrite_labels_ply_files, args.overwrite_annotation,
            args.overwrite_vertices_labels_lookup, args.solve_labels_collision_surf_type, args.n_jobs)

    if utils.should_run(args, 'subcortical_segmentation'):
        # *) Create srf files for subcortical structures
        flags['subcortical'] = subcortical_segmentation(subject, args.overwrite_subcorticals, n_jobs=args.n_jobs)

    if utils.should_run(args, 'calc_faces_verts_dic'):
        # *) Create a dictionary for verts and faces for both hemis
        flags['faces_verts'] = calc_faces_verts_dic(subject, args.atlas, args.overwrite_faces_verts)

    if utils.should_run(args, 'save_labels_vertices'):
        # *) Save the labels vertices for labels plotting
        flags['labels_vertices'] = save_labels_vertices(subject, args.atlas)

    if utils.should_run(args, 'save_hemis_curv'):
        # *) Save the hemis curvs for the inflated brain
        flags['save_hemis_curv'] = save_hemis_curv(subject, args.atlas)

    if utils.should_run(args, 'create_high_level_atlas'):
        flags['create_high_level_atlas'] = create_high_level_atlas(subject, args.high_level_atlas_name)

    if utils.should_run(args, 'create_spatial_connectivity'):
        # *) Create the subject's connectivity
        flags['connectivity'] = create_spatial_connectivity(subject)

    if utils.should_run(args, 'calc_labeles_contours'):
        flags['calc_labeles_contours'] = calc_labeles_contours(subject, args.atlas, args.overwrite_labels_contours)

    if utils.should_run(args, 'calc_labels_center_of_mass'):
        # *) Calc the labels center of mass
        flags['center_of_mass'] = calc_labels_center_of_mass(subject, args.atlas)

    if utils.should_run(args, 'save_labels_coloring'):
        # *) Save a coloring file for the atlas's labels
        flags['save_labels_coloring'] = lu.create_atlas_coloring(subject, args.atlas, args.n_jobs)

    if utils.should_run(args, 'save_subject_orig_trans'):
        flags['save_subject_orig_trans'] = save_subject_orig_trans(subject)

    if utils.should_run(args, 'save_images_data_and_header'):
        flags['save_images_data_and_header'] = save_images_data_and_header(subject)

    if utils.should_run(args, 'create_pial_volume_mask'):
        flags['create_pial_volume_mask'] = create_pial_volume_mask(subject, args.overwrite)

    if utils.should_run(args, 'create_new_subject_blend_file'):
        flags['create_new_subject_blend_file'] = create_new_subject_blend_file(
            subject, args.atlas, args.overwrite_blend, args.ask_if_overwrite_blend)

    if 'cerebellum_segmentation' in args.function:
        flags['save_cerebellum_coloring'] = save_cerebellum_coloring(subject)
        flags['cerebellum_segmentation'] = cerebellum_segmentation(
            subject, remote_subject_dir, args, args.cerebellum_subregions_num)

    if 'transform_coordinates' in args.function:
        flags['transform_coordinates'] = transform_coordinates(subject, args)

    if 'grow_label' in args.function:
        flags['grow_label'] = lu.grow_label(
            subject, args.vertice_indice, args.hemi, args.label_name, args.label_r, args.n_jobs)

    if 'calc_faces_contours' in args.function:
        flags['calc_faces_contours'] = calc_faces_contours(
            subject, args.atlas)

    if 'check_bem' in args.function:
        flags['check_bem'] = check_bem(
            subject, remote_subject_dir, args.recreate_src_spacing, args.recreate_bem_solution, args)

    if 'create_flat_brain' in args.function:
        flags['create_flat_brain'] = create_flat_brain(subject, args.print_only, args.overwrite_flat_surf, args.n_jobs)

    if 'create_slices' in args.function:
        flags['create_slices'] = create_slices(subject, args.slice_xyz, args.slices_modality)

    if 'calc_3d_atlas' in args.function:
        flags['calc_3d_atlas'] = calc_3d_atlas(subject, args.atlas, args.overwrite_aseg_file)

    if 'create_skull_surfaces' in args.function:
        flags['create_skull_surfaces'] = create_skull_surfaces(subject, args.skull_surfaces_fol_name)

    if 'check_labels' in args.function:
        flags['check_labels'] = check_labels(subject, args.atlas)

    if 'load_bem_surfaces' in args.function:
        flags['load_bem_surfaces'] = load_bem_surfaces(subject, True, args.overwrite)

    if 'morph_labels_from_fsaverage' in args.function:
        flags['morph_labels_from_fsaverage'] = morph_labels_from_fsaverage(
            subject, args.atlas, args.template_subject, args.overwrite_morphing_labels, args.fs_labels_fol,
            args.n_jobs)

    if 'solve_labels_collisions' in args.function:
        flags['morph_labels_from_fsaverage'] = solve_labels_collisions(
            subject, args.atlas, args.solve_labels_collision_surf_type, args.overwrite_vertices_labels_lookup,
            args.n_jobs)

    return flags


def read_cmd_args(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='MMVT anatomy preprocessing')
    parser.add_argument('--template_subject', help='template subject', required=False,
                        default='fsaverage6,fsaverage5,fsaverage,colin27', type=au.str_arr_type)
    parser.add_argument('--surf_name', help='surf_name', required=False, default='pial')
    parser.add_argument('--surf_type', required=False, default='inflated,pial,dural', type=au.str_arr_type)
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
    parser.add_argument('--overwrite_labels_contours', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_blend', help='overwrite_blend', required=False, default=0, type=au.is_true)
    parser.add_argument('--ask_if_overwrite_blend', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--overwrite_flat_surf', help='overwrite_flat_surf', required=False, default=0, type=au.is_true)
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
    parser.add_argument('--cerebellum_subregions_num', help='', required=False, default=7, type=int)
    parser.add_argument('--print_only', help='', required=False, default=False)

    parser.add_argument('--vertice_indice', help='', required=False, default=0, type=int)
    parser.add_argument('--label_name', help='', required=False, default='')
    parser.add_argument('--hemi', help='', required=False, default='')
    parser.add_argument('--label_r', help='', required=False, default='5', type=int)
    parser.add_argument('--high_level_atlas_name', help='', required=False, default='high.level.atlas')

    parser.add_argument('--slice_xyz', help='', required=False, default='166,118,113', type=au.int_arr_type)
    parser.add_argument('--slices_modality', help='', required=False, default='mri')
    parser.add_argument('--skull_surfaces_fol_name', help='', required=False, default='bem')
    parser.add_argument('--recreate_bem_solution', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--recreate_src_spacing', help='', required=False, default='oct6')


    pu.add_common_args(parser)
    args = utils.Bag(au.parse_parser(parser, argv))
    existing_freesurfer_annotations = ['aparc.DKTatlas', 'aparc', 'aparc.a2009s']
    args.necessary_files = {'mri': ['aseg.mgz', 'norm.mgz', 'ribbon.mgz', 'T1.mgz', 'orig.mgz', 'brain.mgz'],
        'surf': ['rh.pial', 'lh.pial', 'rh.inflated', 'lh.inflated', 'lh.curv', 'rh.curv', 'rh.sphere.reg',
                 'lh.sphere.reg', 'rh.sphere', 'lh.sphere', 'lh.white', 'rh.white', 'rh.smoothwm','lh.smoothwm',
                 'lh.sphere.reg', 'rh.sphere.reg'],
        'mri:transforms' :['talairach.xfm', 'talairach.m3z'],
        'label': ['rh.{}.annot'.format(annot_name) for annot_name in existing_freesurfer_annotations] +
                 ['lh.{}.annot'.format(annot_name) for annot_name in existing_freesurfer_annotations]}
    if args.overwrite:
        args.overwrite_annotation = True
        args.overwrite_morphing_labels = True
        args.overwrite_vertices_labels_lookup = True
        args.overwrite_hemis_ply = True
        args.overwrite_labels_ply_files = True
        args.overwrite_faces_verts = True
        args.overwrite_fs_files = True
    if 'labeling' in args.function:
        args.function.extend(['create_annotation', 'save_labels_vertices', 'calc_labeles_contours',
                              'calc_labels_center_of_mass', 'save_labels_coloring'])
    if 'create_dural' in args.function and len(args.function) == 1:
        args.function = ['create_surfaces', 'create_pial_volume_mask', 'create_spatial_connectivity']
        args.surf_name = 'dural'
    # python -m src.preproc.anatomy -s nmr00479 -f create_surfaces,create_pial_volume_mask,create_spatial_connectivity --surf_name dural
    # print(args)
    return args


if __name__ == '__main__':
    args = read_cmd_args()
    pu.run_on_subjects(args, main)
    print('finish!')

