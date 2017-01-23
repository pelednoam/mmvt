import argparse
from src.preproc import connections as con
from src.utils import utils
from src.utils import args_utils as au
from src.utils import preproc_utils as pu


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


def calc_electrodes_con(args):
    # -s mg78 -a laus250 -f save_electrodes_coh --threshold_percentile 95 -c interference,non-interference
    args = con.read_cmd_args(utils.Bag(
        subject=args.subject,
        atlas='laus250',
        function='save_electrodes_coh',
        threshold_percentile=95,
        conditions='interference,non-interference'))
    pu.run_on_subjects(args, con.main)


def calc_fmri_connectivity(args):
    '-s hc029 -a laus125 -f calc_lables_connectivity --connectivity_modality fmri --windows_length 20 --windows_shift 3'
    args = con.read_cmd_args(utils.Bag(
        subject=args.subject,
        atlas='laus125',
        function='calc_lables_connectivity',
        connectivity_modality='fmri',
        connectivity_method='corr,cv',
        windows_length=20,
        windows_shift=3
    ))
    pu.run_on_subjects(args, con.main)


def calc_meg_connectivity(args):
    args = con.read_cmd_args(utils.Bag(
        subject=args.subject,
        atlas='laus125',
        function='calc_lables_connectivity',
        connectivity_modality='meg',
        connectivity_method='wpli2_debiased,cv',
        windows_length=500,
        windows_shift=100,
        sfreq=1000.0,
        fmin=5,
        fmax=100
    ))
    pu.run_on_subjects(args, con.main)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-f', '--function', help='function name', required=True)
    args = utils.Bag(au.parse_parser(parser))
    locals()[args.function](args)