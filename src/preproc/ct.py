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


def convert_ct_to_mgz(subject, ct_raw_input_fol, ct_fol='', output_name='ct_org.mgz', overwrite=False):
    if not op.isdir(ct_fol):
        ct_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'ct'))
    output_fname = op.join(ct_fol, output_name)
    if op.isfile(output_fname) and not overwrite:
        return True
    if not op.isdir(ct_raw_input_fol):
        print(f'{ct_fol} does not exist!')
        return
    ct_files = glob.glob(op.join(ct_raw_input_fol, '*.dcm'))
    ct_files.sort(key=op.getmtime)
    fu.mri_convert(ct_files[0], output_fname)
    return op.isfile(output_fname)


def register_to_mr(subject, ct_fol='', ct_name='', nnv_ct_name='', register_ct_name='', threshold=-200,
                   cost_function='nmi', overwrite=False):
    if not op.isdir(ct_fol):
        ct_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'ct'))
    if ct_name == '':
        ct_name = 'ct_org.mgz'
    if nnv_ct_name == '':
        nnv_ct_name = 'ct_no_large_negative_values.mgz'
    if register_ct_name == '':
        register_ct_name = 'ct_reg_to_mr.mgz'
    ctu.remove_large_negative_values_from_ct(
        op.join(ct_fol, ct_name), op.join(ct_fol, nnv_ct_name) , threshold, overwrite)
    ctu.register_ct_to_mr_using_mutual_information(
        subject, SUBJECTS_DIR, op.join(ct_fol, nnv_ct_name), op.join(ct_fol, register_ct_name), lta_name='',
        overwrite=overwrite, cost_function=cost_function)
    t1_fname = op.join(SUBJECTS_DIR, subject, 'mri', 'T1.mgz')
    print(f'freeview -v {t1_fname} {op.join(ct_fol, register_ct_name)}')
    return op.isfile(op.join(ct_fol, register_ct_name))


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


def main(subject, remote_subject_dir, args, flags):
    if utils.should_run(args, 'convert_ct_to_mgz'):
        flags['convert_ct_to_mgz'] = convert_ct_to_mgz(
            subject, args.ct_raw_input_fol, args.ct_fol, args.ct_org_name, args.overwrite)

    if utils.should_run(args, 'register_to_mr'):
        flags['register_to_mr'] = register_to_mr(
            subject, args.ct_fol, args.ct_org_name, args.nnv_ct_name, args.register_ct_name, args.negative_threshold,
            args.register_cost_function, args.overwrite)

    if utils.should_run(args, 'save_subject_ct_trans'):
        flags['save_subject_ct_trans'] = save_subject_ct_trans(subject, args.register_ct_name)

    if utils.should_run(args, 'save_images_data_and_header'):
        flags['save_images_data_and_header'] = save_images_data_and_header(subject, args.register_ct_name)


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
    pu.add_common_args(parser)
    args = utils.Bag(au.parse_parser(parser, argv))
    args.necessary_files = {'mri': ['rawavg.mgz']}
    if args.ct_fol == '':
        for sub in args.subject:
            args.ct_fol = utils.make_dir(op.join(MMVT_DIR, sub, 'ct'))
    return args


if __name__ == '__main__':
    args = read_cmd_args()
    pu.run_on_subjects(args, main)
    print('finish!')
