import argparse
from src.preproc import meg_preproc as meg
from src.utils import utils
from src.utils import args_utils as au


def calc_msit_evoked(subject, mri_subject):
    # -s ep001 -m mg78 -a laus250 -t MSIT --contrast interference --files_includes_cond 1 --read_events_from_file 1 --calc_epochs_from_raw 1 -f calc_evoked
    args = meg.read_cmd_args(['-s', subject, '-m', mri_subject])
    args.task = 'MSIT'
    args.atlas = 'laus250'
    args.function = 'calc_evoked'
    args.constrast = 'interference'
    args.files_includes_cond  = True
    args.read_events_from_file = True
    args.calc_epochs_from_raw = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-m', '--mri_subject', help='mri subject name', required=False, default=None,
                        type=au.str_arr_type)
    parser.add_argument('-f', '--function', help='function name', required=True)
    args = utils.Bag(au.parse_parser(parser))
    if not args.mri_subject:
        args.mri_subject = args.subject
    for subject, mri_subject in zip(args.subject, args.mri_subject):
        locals()[args.function](subject, mri_subject)