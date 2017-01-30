import argparse
import os.path as op
from src.preproc import connections as con
from src.utils import utils
from src.utils import args_utils as au
from src.utils import preproc_utils as pu

LINKS_DIR = utils.get_links_dir()
ELECTRODES_DIR = utils.get_link_dir(LINKS_DIR, 'electrodes')


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
        connectivity_method='pli,cv',
        windows_length=500,
        windows_shift=100,
        sfreq=1000.0,
        fmin=10,
        fmax=100
    ))
    pu.run_on_subjects(args, con.main)


def calc_electrodes_connectivity(args):
    import scipy.io as sio
    import math
    import numpy as np
    import src.mmvt_addon.mmvt_utils as mu
    for subject in args.subject:
        data_fname = op.join(ELECTRODES_DIR, subject, 'data.npy')
        if not op.isfile(data_fname):
            d = sio.loadmat(op.join(ELECTRODES_DIR, subject, 'data.mat'))
            data_mat = d['data']
            electrodes_num = data_mat.shape[1]
            labels = []
            for label_mat in d['labels']:
                group, elc1, elc2 = mu.elec_group_number('{1}-{0}'.format(*label_mat.split(' ')), True)
                labels.append('{0}{1}-{0}{2}'.format(group, elc2, elc1))
            np.save(op.join(ELECTRODES_DIR, subject, 'electrodes.npy'), labels)
            data = data_mat.swapaxes(0,1).reshape(electrodes_num, -1)
            E, T = data.shape
            windows_length = 1000
            windows_shift = 200
            windows_nun = math.floor((T - windows_length) / windows_shift + 1)
            data_winodws = np.zeros((windows_nun, E, windows_length))
            for w in range(windows_nun):
                data_winodws[w] = data[:, w * windows_shift:w * windows_shift + windows_length]
            np.save(data_fname, data_winodws)

    args = con.read_cmd_args(utils.Bag(
        subject=args.subject,
        function='save_electrodes_connectivity',
        connectivity_modality='electrodes',
        connectivity_method='pli,cv',
        windows_length=1000,
        windows_shift=200,
        sfreq=2000.0,
        n_jobs=args.n_jobs
        # fmin=10,
        # fmax=100
    ))
    pu.run_on_subjects(args, con.main)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-f', '--function', help='function name', required=True)
    parser.add_argument('--n_jobs', help='cpu num', required=False, default=-1)
    args = utils.Bag(au.parse_parser(parser))
    locals()[args.function](args)