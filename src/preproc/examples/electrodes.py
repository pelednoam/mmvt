import os.path as op
import argparse
from src.preproc import electrodes_preproc as elecs_preproc
from src.utils import utils
from src.utils import args_utils as au


def read_electrodes_coordiantes_from_specific_xlsx_sheet(subject, bipolar):
    args = elecs_preproc.read_cmd_args(['-s', subject, '-b', str(bipolar)])
    args.ras_xls_sheet_name = 'RAS_Snapped'
    elecs_preproc.main(subject, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-b', '--bipolar', help='bipolar', required=True, type=au.is_true)
    parser.add_argument('-f', '--function', help='function name', required=True)
    args = utils.Bag(au.parse_parser(parser))
    for subject in args.subject:
        locals()[args.function](subject, args.bipolar)