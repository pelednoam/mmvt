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

SUBJECTS_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()


def convert_ct_to_mgz(subject, ct_raw_input_fol, ct_fol='', output_name='ct_org.mgz', overwrite=False, print_only=False,
                      ask_before=False):
    if not op.isdir(ct_fol):
        ct_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'ct'))
    output_fname = op.join(ct_fol, output_name)
    if op.isfile(output_fname):
        if not overwrite:
            return True
        else:
            os.remove(output_fname)
    if not op.isdir(ct_raw_input_fol):
        print(f'{ct_fol} does not exist!')
        return False
    ct_files = glob.glob(op.join(ct_raw_input_fol, '*.dcm'))
    if len(ct_files) == 0:
        sub_folders = [d for d in glob.glob(op.join(ct_raw_input_fol, '*')) if op.isdir(d)]
        if len(sub_folders) == 0:
            print(f'Cannot find CT files in {ct_raw_input_fol}!')
            return False
        fol = utils.select_one_file(sub_folders, '', 'CT', is_dir=True)
        ct_files = glob.glob(op.join(fol, '*.dcm'))
        if len(ct_files) == 0:
            print(f'Cannot find CT files in {fol}!')
            return False
    ct_files.sort(key=op.getmtime)
    if ask_before:
        ret = input(f'convert {ct_files[0]} to {output_fname}? ')
        if not au.is_true(ret):
            return False
    fu.mri_convert(ct_files[0], output_fname, print_only=print_only)
    return True if print_only else op.isfile(output_fname)


def register_to_mr(subject, ct_fol='', ct_name='', nnv_ct_name='', register_ct_name='', threshold=-200,
                   cost_function='nmi', overwrite=False, print_only=False):
    if not op.isdir(ct_fol):
        ct_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'ct'))
    if ct_name == '':
        ct_name = 'ct_org.mgz'
    if nnv_ct_name == '':
        nnv_ct_name = 'ct_no_large_negative_values.mgz'
    if register_ct_name == '':
        register_ct_name = 'ct_reg_to_mr.mgz'
    if print_only:
        print(f'Removign large negative values: {op.join(ct_fol, ct_name)} -> {op.join(ct_fol, nnv_ct_name)}')
    else:
        ctu.remove_large_negative_values_from_ct(
            op.join(ct_fol, ct_name), op.join(ct_fol, nnv_ct_name), threshold, overwrite)
    ctu.register_ct_to_mr_using_mutual_information(
        subject, SUBJECTS_DIR, op.join(ct_fol, nnv_ct_name), op.join(ct_fol, register_ct_name), lta_name='',
        overwrite=overwrite, cost_function=cost_function, print_only=print_only)
    t1_fname = op.join(SUBJECTS_DIR, subject, 'mri', 'T1.mgz')
    print(f'freeview -v {t1_fname} {op.join(ct_fol, register_ct_name)}')
    return True if print_only else op.isfile(op.join(ct_fol, register_ct_name))


def save_subject_ct_trans(subject, ct_name='ct_reg_to_mr.mgz'):
    output_fname = op.join(MMVT_DIR, subject, 'ct', 'ct_trans.npz')
    ct_fname = op.join(MMVT_DIR, subject, 'ct', ct_name)
    if not op.isfile(ct_fname):
        subjects_ct_fname = op.join(SUBJECTS_DIR, subject, 'mri', ct_name)
        if op.isfile(subjects_ct_fname):
            shutil.copy(subjects_ct_fname, ct_fname)
        else:
            print("Can't find subject's CT! ({})".format(ct_fname))
            return False
    header = nib.load(ct_fname).header
    ras_tkr2vox, vox2ras_tkr, vox2ras, ras2vox = anat.get_trans_functions(header)
    np.savez(output_fname, ras_tkr2vox=ras_tkr2vox, vox2ras_tkr=vox2ras_tkr, vox2ras=vox2ras, ras2vox=ras2vox)
    return op.isfile(output_fname)


def save_images_data_and_header(subject, ct_name='ct_reg_to_mr.mgz'):
    ret = True
    data, header = ctu.get_data_and_header(subject, MMVT_DIR, SUBJECTS_DIR, ct_name)
    if data is None or header is None:
        return False
    affine = header.affine
    precentiles = np.percentile(data, (1, 99))
    colors_ratio = 256 / (precentiles[1] - precentiles[0])
    output_fname = op.join(MMVT_DIR, subject, 'ct', 'ct_data.npz')
    if not op.isfile(output_fname):
        np.savez(output_fname, data=data, affine=affine, precentiles=precentiles, colors_ratio=colors_ratio)
    ret = ret and op.isfile(output_fname)
    return ret


def find_electrodes(subject, n_components, n_groups, ct_name='', brain_mask_fname='', output_fol=None,
                    clustering_method='knn', max_iters=5, cylinder_error_radius=3, min_elcs_for_lead=4,
                    max_dist_between_electrodes=20, min_cylinders_ang=0.1,
                    thresholds=(99, 99.9, 99.95, 99.99, 99.995, 99.999), overwrite=False, debug=False):
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
        min_cylinders_ang, thresholds, overwrite, debug)
    if output_fol == '':
        return all(x is not None for x in [electrodes, groups, groups_hemis])
    else:
        return op.isfile(op.join(output_fol, 'objects.pkl'))


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
        flags['save_subject_ct_trans'] = save_subject_ct_trans(subject, args.register_ct_name)

    if utils.should_run(args, 'save_images_data_and_header'):
        flags['save_images_data_and_header'] = save_images_data_and_header(subject, args.register_ct_name)

    if 'find_electrodes' in args.function:
        flags['find_electrodes'] = find_electrodes(
            subject, args.n_components, args.n_groups, args.register_ct_name, args.brain_mask_fname,
            args.output_fol, args.clustering_method, args.max_iters, args.cylinder_error_radius,
            args.min_elcs_for_lead, args.max_dist_between_electrodes, args.min_cylinders_ang, args.ct_thresholds,
            args.overwrite, args.debug)

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
    parser.add_argument('--register_cost_function', help='', required=False, default='nmi')
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

    pu.add_common_args(parser)
    args = utils.Bag(au.parse_parser(parser, argv))
    args.necessary_files = {'mri': ['brain.mgz']}
    return args


if __name__ == '__main__':
    args = read_cmd_args()
    pu.run_on_subjects(args, main)
    print('finish!')
