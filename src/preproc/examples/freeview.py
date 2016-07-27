import argparse
import os.path as op
from src.preproc import freeview_preproc as fp
from src.utils import utils
from src.utils import args_utils as au


def darpa(subject, args):
    args = fp.read_cmd_args(['-s', subject, '-b', str(args.bipolar)])
    args.remote_subject_dir = op.join('/space/huygens/1/users/kara/{}_SurferOutput/'.format(subject.upper()))
    fp.run_on_subjects(subject, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-b', '--bipolar', help='bipolar', required=True, type=au.is_true)
    parser.add_argument('-f', '--function', help='function name', required=True)
    args = utils.Bag(au.parse_parser(parser))
    for subject in args.subject:
        locals()[args.function](subject, args)