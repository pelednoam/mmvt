import argparse
from src.preproc import fMRI_preproc as fmri
from src.utils import utils
from src.utils import args_utils as au


def load_rest_to_colin():
    args = fmri.read_cmd_args(['-s', subject])
    args.task = 'REST'
    args.function = 'project_volume_to_surface,find_clusters'
    args.contrast = 'rest'
    args.volume_name = 'spmT_0001'
    fmri.main(subject, mri_subject, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=False, type=au.str_arr_type, default='colin27')
    parser.add_argument('-f', '--function', help='function name', required=True)
    args = utils.Bag(au.parse_parser(parser))
    if not args.mri_subject:
        args.mri_subject = args.subject
    for subject, mri_subject in zip(args.subject, args.mri_subject):
        locals()[args.function](subject, mri_subject)