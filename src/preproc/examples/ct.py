import os.path as op
import argparse
from src.utils import utils
from src.utils import args_utils as au
from src.utils import preproc_utils as pu
from src.preproc import ct


def convert_ct_from_kara(args):
    for subject in args.subject:
        darpa_subject = subject[:2].upper() + subject[2:]
        ct_raw_input_fol = op.join('/space/huygens/1/users/kara', f'{darpa_subject}_CT')
        args = ct.read_cmd_args(utils.Bag(
            subject=args.subject,
            function='convert_ct_to_mgz',
            ct_raw_input_fol=ct_raw_input_fol))
        pu.run_on_subjects(args, ct.main)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-f', '--function', help='function name', required=True)
    args = utils.Bag(au.parse_parser(parser))
    # for subject in args.subject:
    locals()[args.function](args)
