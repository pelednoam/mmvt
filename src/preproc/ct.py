import os
import os.path as op
import glob
import shutil
import nibabel as nib
import numpy as np

from src.utils import utils
from src.utils import preproc_utils as pu
from src.utils import args_utils as au
from src.utils import freesurfer_utils as fu
from src.utils import ct_utils as ctu
from src.preproc import anatomy as anat
from src.mmvt_addon import slicer

SUBJECTS_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()


def convert_ct_to_mgz(subject, ct_raw_input_fol, ct_fol='', output_name='ct_org.mgz', overwrite=False, print_only=False,
                      ask_before=False):
    if not op.isdir(ct_fol):
        ct_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'ct'))
    if op.isfile(op.join(ct_fol, 'ct_reg_to_mr.mgz')) and not overwrite:
        return True
    ct_fname = utils.select_one_file(glob.glob(op.join(ct_fol, '*.mgz')))
    if op.isfile(ct_fname):
        if utils.namebase(ct_fname) != 'ct_org':
            utils.make_link(ct_fname, op.join(ct_fol, 'ct_org.mgz'))
        return True
    output_fname = op.join(ct_fol, output_name)
    if op.isfile(output_fname):
        if not overwrite:
            return True
        else:
            os.remove(output_fname)
    if op.isfile(op.join(SUBJECTS_DIR, subject, 'ct', 'ct.nii.gz')):
        ct_files = [op.join(SUBJECTS_DIR, subject, 'ct', 'ct.nii.gz')]
    elif op.isfile(op.join(SUBJECTS_DIR, subject, 'mri', 'ct.nii.gz')):
        ct_files = [op.join(SUBJECTS_DIR, subject, 'mri', 'ct.nii.gz')]
    else:
        if not op.isdir(ct_raw_input_fol):
            print('{} does not exist!'.format(ct_fol))
            return False
        ct_files = glob.glob(op.join(ct_raw_input_fol, '*.dcm'))
        if len(ct_files) == 0:
            sub_folders = [d for d in glob.glob(op.join(ct_raw_input_fol, '*')) if op.isdir(d)]
            if len(sub_folders) == 0:
                print('Cannot find CT files in {}!'.format(ct_raw_input_fol))
                return False
            fol = utils.select_one_file(sub_folders, '', 'CT', is_dir=True)
            ct_files = glob.glob(op.join(fol, '*.dcm'))
            if len(ct_files) == 0:
                print('Cannot find CT files in {}!'.format(fol))
                return False
        ct_files.sort(key=op.getmtime)
        if ask_before:
            ret = input('convert {} to {}? '.format(ct_files[0], output_fname))
            if not au.is_true(ret):
                return False
    fu.mri_convert(ct_files[0], output_fname, print_only=print_only)
    return True if print_only else op.isfile(output_fname)


def register_to_mr(subject, ct_fol='', ct_name='', nnv_ct_name='', register_ct_name='', threshold=-200,
                   cost_function='nmi', overwrite=False, print_only=False):
    if op.isfile(op.join(SUBJECTS_DIR, subject, 'ct', ct_name)):
        shutil.copy(op.join(SUBJECTS_DIR, subject, 'ct', ct_name), op.join(MMVT_DIR, subject, 'ct', ct_name))
    if not op.isdir(ct_fol):
        ct_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'ct'))
    if op.isfile(op.join(ct_fol, 'ct_reg_to_mr.mgz')) and not overwrite:
        return True
    if ct_name == '':
        ct_name = 'ct_org.mgz'
    if nnv_ct_name == '':
        nnv_ct_name = 'ct_no_large_negative_values.mgz'
    if register_ct_name == '':
        register_ct_name = 'ct_reg_to_mr.mgz'
    print('Removing large negative values: {} -> {}'.format(op.join(ct_fol, ct_name), op.join(ct_fol, nnv_ct_name)))
    if not print_only:
        ctu.remove_large_negative_values_from_ct(
            op.join(ct_fol, ct_name), op.join(ct_fol, nnv_ct_name), threshold, overwrite)
    ctu.register_ct_to_mr_using_mutual_information(
        subject, SUBJECTS_DIR, op.join(ct_fol, nnv_ct_name), op.join(ct_fol, register_ct_name), lta_name='',
        overwrite=overwrite, cost_function=cost_function, print_only=print_only)
    t1_fname = op.join(SUBJECTS_DIR, subject, 'mri', 'T1.mgz')
    print('freeview -v {} {}'.format(t1_fname, op.join(ct_fol, register_ct_name)))
    return True if print_only else op.isfile(op.join(ct_fol, register_ct_name))


def save_subject_ct_trans(subject, ct_name='ct_reg_to_mr.mgz', overwrite=False):
    output_fname = op.join(MMVT_DIR, subject, 'ct', 'ct_trans.npz')
    if op.isfile(output_fname) and not overwrite:
        return True
    ct_fname, ct_exist = utils.locating_file(ct_name, ['*.mgz', '*.nii', '*.nii.gz'], op.join(MMVT_DIR, subject, 'ct'))
    # ct_fname = op.join(MMVT_DIR, subject, 'ct', ct_name)
    if not ct_exist:# op.isfile(ct_fname):
        # subjects_ct_fname = op.join(SUBJECTS_DIR, subject, 'mri', ct_name)
        ct_fname, ct_exist = utils.locating_file(
            ct_name, ['*.mgz', '*.nii', '*.nii.gz'], op.join(SUBJECTS_DIR, subject, 'mri'))
        if ct_exist: #op.isfile(subjects_ct_fname):
            utils.make_dir(op.join(MMVT_DIR, subject, 'ct'))
            ct_fname = utils.copy(ct_fname, op.join(MMVT_DIR, subject, 'ct'))
        else:
            print("Can't find subject's CT! ({})".format(ct_fname))
            return False
    if ct_fname != op.join(MMVT_DIR, subject, 'ct', ct_name):
        utils.make_link(ct_fname, op.join(MMVT_DIR, subject, 'ct', ct_name))
    print('save_subject_ct_trans: loading {}'.format(ct_fname))
    header = nib.load(ct_fname).header
    ras_tkr2vox, vox2ras_tkr, vox2ras, ras2vox = anat.get_trans_functions(header)
    print('save_subject_ct_trans: Saving {}'.format(output_fname))
    np.savez(output_fname, ras_tkr2vox=ras_tkr2vox, vox2ras_tkr=vox2ras_tkr, vox2ras=vox2ras, ras2vox=ras2vox)
    return op.isfile(output_fname)


def merge_t1_with_ct(subject, ct_threshold=None, ct_name='ct_reg_to_mr.mgz', overwrite=True):
    output_fname = op.join(MMVT_DIR, subject, 'ct', 't1_ct.mgz')
    if op.isfile(output_fname) and not overwrite:
        return True
    t1 = nib.load(op.join(SUBJECTS_DIR, subject, 'mri', 'T1.mgz'))
    t1_data = t1.get_data()
    ct_data = nib.load(op.join(MMVT_DIR, subject, 'ct', ct_name)).get_data()
    if ct_threshold is None:
        ct_threshold = np.percentile(ct_data, 99)
    ct_trans = utils.Bag(np.load(op.join(MMVT_DIR, subject, 'ct', 'ct_trans.npz')))
    t1_trans = utils.Bag(np.load(op.join(MMVT_DIR, subject, 't1_trans.npz')))
    print('Finding all voxels above {}'.format(ct_threshold))
    ct_indices = np.where(ct_data > ct_threshold)
    ct_voxels = np.array(ct_indices).T
    ct_ras_coordinates = apply_trans(ct_trans.vox2ras, ct_voxels)
    t1_voxels = np.rint(apply_trans(t1_trans.ras2vox, ct_ras_coordinates)).astype(int)
    t1_data[(t1_voxels.T[0], t1_voxels.T[1], t1_voxels.T[2])] = ct_data[(ct_voxels.T[0], ct_voxels.T[1], ct_voxels.T[2])]

    t1_ct_mask = np.zeros(t1_data.shape, dtype=np.int8)
    t1_ct_mask[(t1_voxels.T[0], t1_voxels.T[1], t1_voxels.T[2])] = 1
    np.save(op.join(MMVT_DIR, subject, 'ct', 't1_ct_mask.npy'), t1_ct_mask)

    img = nib.Nifti1Image(t1_data, t1.affine)
    nib.save(img, output_fname)
    save_images_data_and_header(subject, ct_name=output_fname, output_name='t1_ct_data', overwrite=True)
    np.savez(op.join(MMVT_DIR, subject, 'ct', 't1_ct_trans.npz'), ras_tkr2vox=t1_trans.ras_tkr2vox,
             vox2ras_tkr=t1_trans.vox2ras_tkr, vox2ras=t1_trans.vox2ras, ras2vox=t1_trans.ras2vox)
    return op.isfile(output_fname)


def apply_trans(trans, points):
    return np.array([np.dot(trans, np.append(p, 1))[:3] for p in points])


def save_images_data_and_header(subject, ct_name='ct_reg_to_mr.mgz', output_name='ct_data', no_negatives=True,
                                overwrite=False):
    ret = True
    data, header = ctu.get_data_and_header(subject, MMVT_DIR, SUBJECTS_DIR, ct_name)
    if data is None or header is None:
        return False
    affine = header.affine
    precentiles = np.percentile(data, (1, 99))
    if no_negatives:
        precentiles[0] = max(precentiles[0], 0)
    colors_ratio = 256 / (precentiles[1] - precentiles[0])
    output_fname = op.join(MMVT_DIR, subject, 'ct', '{}.npz'.format(output_name))
    if not op.isfile(output_fname) or overwrite:
        print('save_images_data_and_header: saving to {}'.format(output_fname))
        np.savez(output_fname, data=data, affine=affine, precentiles=precentiles, colors_ratio=colors_ratio)
    ret = ret and op.isfile(output_fname)
    return ret


def find_electrodes(subject, n_components, n_groups, ct_name='', brain_mask_fname='', output_fol=None,
                    clustering_method='knn', max_iters=5, cylinder_error_radius=3, min_elcs_for_lead=4,
                    max_dist_between_electrodes=20, min_cylinders_ang=0.1,
                    thresholds=(99, 99.9, 99.95, 99.99, 99.995, 99.999), min_joined_items_num=1,
                    min_distance_beteen_electrodes=2, overwrite=False, debug=False):
    from src.misc.dell import find_electrodes_in_ct
    if n_components <= 0 or n_groups <= 0:
        print('Both n_components and n_groups should be > 0!')
        return False
    if ct_name == '':
        ct_name = 'ct_reg_to_mr.mgz'
    ct_fname = op.join(MMVT_DIR, subject, 'ct', ct_name)
    if brain_mask_fname == '':
        brain_mask_fname = op.join(SUBJECTS_DIR, subject, 'mri', 'brain.mgz')
    if output_fol is None:
        output_fol = utils.make_dir(op.join(
            MMVT_DIR, subject, 'ct', 'finding_electrodes_in_ct', utils.rand_letters(5)))
    electrodes, groups, groups_hemis = find_electrodes_in_ct.find_depth_electrodes_in_ct(
        subject, ct_fname, brain_mask_fname, n_components, n_groups, output_fol, clustering_method,
        max_iters, cylinder_error_radius, min_elcs_for_lead, max_dist_between_electrodes,
        min_cylinders_ang, thresholds, min_joined_items_num, min_distance_beteen_electrodes, overwrite, debug)
    if output_fol == '':
        return all(x is not None for x in [electrodes, groups, groups_hemis])
    else:
        return op.isfile(op.join(output_fol, 'objects.pkl'))


def save_electrode_ct_pics(subject, voxel, elc_name='', pixels_around_voxel=30, interactive=True, states=None, fig_fname=''):
    fol = utils.make_dir(op.join(MMVT_DIR, subject, 'ct', 'figures'))
    if fig_fname == '' and not interactive:
        fig_fname = op.join(fol, '{}_{}.jpg'.format(voxel, pixels_around_voxel))
    if states is None:
        states = {}
        for modality in ['mri', 'ct']:
            states[modality] = slicer.init(None, modality=modality, subject=subject, mmvt_dir=MMVT_DIR)
    slicer.plot_slices(None, voxel, states, 'ct', interactive, pixels_around_voxel, fig_fname=fig_fname, elc_name=elc_name)
    if fig_fname != '':
        return op.isfile(fig_fname)
    else:
        return True


def save_electrodes_group_ct_pics(subject, voxels, group_name='', electrodes_names='', pixels_around_voxel=30):
    states = {}
    for modality in ['mri', 'ct']:
        states[modality] = slicer.init(None, modality=modality, subject=subject, mmvt_dir=MMVT_DIR)
    if electrodes_names == '':
        electrodes_names = [''] * len(voxels)
    elif group_name == '':
        group_name = '{}-{}'.format(electrodes_names[0], electrodes_names[-1])
    if group_name == '':
        print("Both group_name and electrodes_names can't be empty!")
        return False
    fol = utils.make_dir(op.join(MMVT_DIR, subject, 'ct', 'figures', group_name))
    ret = True
    for ind, (voxel, elc_name) in enumerate(zip(voxels, electrodes_names)):
        elc_name = voxel if elc_name == '' else elc_name
        fig_fname = op.join(fol, '{}_{}_{}.jpg'.format(ind, elc_name, pixels_around_voxel))
        ret = ret and save_electrode_ct_pics(subject, voxel, elc_name, pixels_around_voxel, False, states, fig_fname)
    return ret


def isotropization(subject, ct_fname, ct_fol, new_image_fname='', isotropization_type=1, iso_vector_override=None):
    if not op.isdir(ct_fol):
        ct_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'ct'))
    if new_image_fname == '':
        new_image_fname = 'iso_ct.{}'.format(utils.file_type(ct_fname))
    ct_fname = op.join(ct_fol, ct_fname)
    iso_img = ctu.isotropization(
        ct_fname, isotropization_type=isotropization_type, iso_vector_override=iso_vector_override)
    nib.save(iso_img, op.join(ct_fol, new_image_fname))


def main(subject, remote_subject_dir, args, flags):

    if utils.should_run(args, 'convert_ct_to_mgz'):
        flags['convert_ct_to_mgz'] = convert_ct_to_mgz(
            subject, args.ct_raw_input_fol, args.ct_fol, args.ct_org_name, args.overwrite, args.print_only,
            args.ask_before)

    if utils.should_run(args, 'register_to_mr'):
        flags['register_to_mr'] = register_to_mr(
            subject, args.ct_fol, args.ct_org_name, args.nnv_ct_name, args.register_ct_name, args.negative_threshold,
            args.register_cost_function, args.overwrite, args.print_only)

    if utils.should_run(args, 'save_subject_ct_trans'):
        flags['save_subject_ct_trans'] = save_subject_ct_trans(subject, args.register_ct_name, args.overwrite)

    if utils.should_run(args, 'save_images_data_and_header'):
        flags['save_images_data_and_header'] = save_images_data_and_header(
            subject, args.register_ct_name, 'ct_data', args.no_negatives, args.overwrite)

    if 'find_electrodes' in args.function:
        flags['find_electrodes'] = find_electrodes(
            subject, args.n_components, args.n_groups, args.register_ct_name, args.brain_mask_fname,
            args.output_fol, args.clustering_method, args.max_iters, args.cylinder_error_radius,
            args.min_elcs_for_lead, args.max_dist_between_electrodes, args.min_cylinders_ang, args.ct_thresholds,
            args.min_joined_items_num, args.min_distance_beteen_electrodes, args.overwrite, args.debug)

    if 'isotropization' in args.function:
        flags['isotropization'] = isotropization(
            subject, args.ct_org_name, args.ct_fol, isotropization_type=args.isotropization_type,
            iso_vector_override=None)

    if 'save_electrode_ct_pics' in args.function:
        flags['save_electrode_ct_pics'] = save_electrode_ct_pics(
            subject, args.voxel, args.elc_name, args.pixels_around_voxel, args.interactive, fig_fname=args.fig_name)

    if 'save_electrodes_group_ct_pics' in args.function:
        flags['save_electrodes_group_ct_pics'] = save_electrodes_group_ct_pics(
            subject, args.voxels, args.group_name, args.electrodes_names, args.pixels_around_voxel)

    if 'merge_t1_with_ct' in args.function:
        flags['merge_t1_with_ct'] = merge_t1_with_ct(subject, args.ct_threshold)

    return flags


def read_cmd_args(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='MMVT CT preprocessing')
    parser.add_argument('--ct_raw_input_fol', help='', required=False, default='')
    parser.add_argument('--ct_fol', help='', required=False, default='')
    parser.add_argument('--ct_org_name', help='', required=False, default='ct_org.mgz')
    parser.add_argument('--nnv_ct_name', help='', required=False, default='ct_no_large_negative_values.mgz')
    parser.add_argument('--register_ct_name', help='', required=False, default='ct_reg_to_mr.mgz')
    parser.add_argument('--negative_threshold', help='', required=False, default=-200, type=int)
    parser.add_argument('--no_negatives', help='', required=False, default=True, type=au.is_true)
    parser.add_argument('--register_cost_function', help='', required=False, default='nmi')
    parser.add_argument('--ct_threshold', help='', required=False, type=float)
    parser.add_argument('--overwrite', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--print_only', help='', required=False, default=False, type=au.is_true)
    parser.add_argument('--ask_before', help='', required=False, default=False, type=au.is_true)
    parser.add_argument('--debug', help='', required=False, default=False, type=au.is_true)

    # find_electrodes:
    parser.add_argument('--n_components', help='', required=False, default=0, type=int)
    parser.add_argument('--n_groups', help='', required=False, default=0, type=int)
    parser.add_argument('--brain_mask_fname', help='', required=False, default='')
    parser.add_argument('--output_fol', help='', required=False, default=None, type=au.str_or_none)
    parser.add_argument('--clustering_method', help='', required=False, default='knn')
    parser.add_argument('--max_iters', help='', required=False, default=5, type=int)
    parser.add_argument('--cylinder_error_radius', help='', required=False, default=3, type=float)
    parser.add_argument('--min_elcs_for_lead', help='', required=False, default=4, type=int)
    parser.add_argument('--max_dist_between_electrodes', help='', required=False, default=20, type=float)
    parser.add_argument('--min_cylinders_ang', help='', required=False, default=0.1, type=float)
    parser.add_argument('--ct_thresholds', help='', required=False, default='99,99.9,99.95,99.99,99.995,99.999',
                        type=au.float_arr_type)
    parser.add_argument('--min_joined_items_num', help='', required=False, default=1, type=int)
    parser.add_argument('--min_distance_beteen_electrodes', help='', required=False, default=2, type=float)

    # Dell
    parser.add_argument('--voxels', help='', required=False, default='', type=au.list_of_int_lists_type)
    parser.add_argument('--voxel', help='', required=False, default='', type=au.int_arr_type) # 102,99,131
    parser.add_argument('--pixels_around_voxel', help='', required=False, default=30, type=int)
    parser.add_argument('--interactive', help='', required=False, default=0, type=au.is_true)
    parser.add_argument('--electrodes_names', help='', required=False, default='', type=au.str_arr_type)
    parser.add_argument('--group_name', help='', required=False, default='')
    parser.add_argument('--elc_name', help='', required=False, default='')
    parser.add_argument('--fig_name', help='', required=False, default='')
    parser.add_argument('--isotropization_type', help='', required=False, default=1, type=int)

    pu.add_common_args(parser)
    args = utils.Bag(au.parse_parser(parser, argv))
    pu.set_default_folders(args)
    args.necessary_files = {'mri': ['brain.mgz']}

    global SUBJECTS_DIR, MMVT_DIR
    SUBJECTS_DIR = args.mri_dir
    MMVT_DIR = args.mmvt_dir
    print('Subjects dir: {}'.format(SUBJECTS_DIR))
    print('MMVT dir: {}'.format(MMVT_DIR))
    return args


if __name__ == '__main__':
    args = read_cmd_args()
    pu.run_on_subjects(args, main)
    print('finish!')

