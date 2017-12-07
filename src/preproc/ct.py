import os.path as op
import glob

from src.utils import utils
from src.utils import preproc_utils as pu
from src.utils import args_utils as au
from src.utils import freesurfer_utils as fu

SUBJECTS_MRI_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()


def convert_ct_to_mgz(subject, args):
    if not op.isdir(args.ct_fol):
        args.ct_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'ct'))
    output_fname = op.join(args.ct_fol, f'{subject}.mgz')
    if op.isfile(output_fname) and not args.overwrite_ct:
        return True
    if not op.isdir(args.ct_raw_input_fol):
        print(f'{args.ct_fol} does not exist!')
        return
    ct_files = glob.glob(op.join(args.ct_raw_input_fol, '*.dcm'))
    ct_files.sort(key=op.getmtime)
    fu.mri_convert(ct_files[0], output_fname)
    return op.isfile(output_fname)


def main(subject, remote_subject_dir, args, flags):
    if utils.should_run(args, 'convert_ct_to_mgz'):
        flags['convert_ct_to_mgz'] = convert_ct_to_mgz(subject, args)
    return flags


def read_cmd_args(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='MMVT CT preprocessing')
    parser.add_argument('--ct_raw_input_fol', help='', required=False, default='')
    parser.add_argument('--ct_fol', help='', required=False, default='')
    parser.add_argument('--overwrite_ct', help='', required=False, default=0, type=au.is_true)
    pu.add_common_args(parser)
    args = utils.Bag(au.parse_parser(parser, argv))
    if args.ct_fol == '':
        for sub in args.subject:
            args.ct_fol = utils.make_dir(op.join(MMVT_DIR, sub, 'ct'))
    return args


if __name__ == '__main__':
    args = read_cmd_args()
    pu.run_on_subjects(args, main)
    print('finish!')
