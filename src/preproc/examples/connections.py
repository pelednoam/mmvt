import argparse
from src.preproc import connections as con
from src.utils import utils
from src.utils import args_utils as au


def example1(subject):
    # subject = 'fsaverage5c'
    args = con.read_cmd_args(['-s', subject])
    args.atlas = 'laus125'
    args.mat_fname = '/cluster/neuromind/npeled/linda/figure_file1.mat'
    args.mat_field = 'figure_file'
    args.windows = 1
    args.stat = con.STAT_DIFF
    args.threshold_percentile = 99
    args.conditions = ['task', 'rest']
    con.save_rois_connectivity(subject, args)


def example2(subject):
    # subject = 'fsaverage5c'
    args = con.read_cmd_args(['-s', subject])
    args.atlas = 'laus125'
    args.mat_fname = '/cluster/neuromind/npeled/linda/figure_file3.mat'
    args.mat_field = 'matrix_cognition_ratio_fpn_dmn'
    args.windows = 1
    args.threshold = 0
    args.norm_by_percentile = False
    args.symetric_colors = False
    args.color_map = 'YlOrRd'
    args.data_min = 0.15
    args.data_max = 0.6
    con.save_rois_connectivity(subject, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-f', '--function', help='function name', required=True)
    args = utils.Bag(au.parse_parser(parser))
    for subject in args.subject:
        locals()[args.function](subject)