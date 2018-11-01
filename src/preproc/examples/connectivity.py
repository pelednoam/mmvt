import argparse
import os.path as op
from src.preproc import connectivity as con
from src.utils import utils
from src.utils import args_utils as au
from src.utils import preproc_utils as pu

LINKS_DIR = utils.get_links_dir()
ELECTRODES_DIR = utils.get_link_dir(LINKS_DIR, 'electrodes')
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')


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
        function='calc_electrodes_coh',
        threshold_percentile=95,
        conditions='interference,non-interference'))
    pu.run_on_subjects(args, con.main)


def calc_fmri_connectivity(args):
    '-s hc029 -a laus125 -f calc_lables_connectivity --connectivity_modality fmri connectivity_method=corr,cv--windows_length 20 --windows_shift 3'
    args = con.read_cmd_args(dict(
        subject=args.subject,
        atlas='laus125',
        function='calc_lables_connectivity',
        connectivity_modality='fmri',
        # connectivity_method='mi_vec,cv',
        connectivity_method='corr,cv',
        # labels_extract_mode='pca_2,pca_4,pca_8', #mean,pca,
        labels_extract_mode='mean',
        windows_length=34, #20,
        windows_shift=4, #3,
        save_mmvt_connectivity=False,
        calc_subs_connectivity=False,
        recalc_connectivity=True,
        n_jobs=args.n_jobs
    ))
    con.call_main(args)


def calc_fmri_static_connectivity(args):
    args = con.read_cmd_args(dict(
        subject=args.subject,
        atlas='laus125',# 'yao17'
        function='calc_lables_connectivity',
        connectivity_modality='fmri',
        connectivity_method='corr',
        labels_extract_mode='mean',
        identifier='',#'freesurfer',#'hesheng', #''linda',#
        save_mmvt_connectivity=False,
        calc_subs_connectivity=False,
        recalc_connectivity=True,
        n_jobs=args.n_jobs
    ))
    con.call_main(args)


def calc_meg_connectivity(args):
    args = con.read_cmd_args(utils.Bag(
        subject=args.subject,
        atlas='laus125',
        function='calc_lables_connectivity',
        connectivity_modality='meg',
        connectivity_method='pli',
        windows_length=500,
        windows_shift=100,
        # sfreq=1000.0,
        # fmin=10,
        # fmax=100
        # recalc_connectivity=True,
        max_windows_num=100,
        recalc_connectivity=True,
        n_jobs=args.n_jobs
    ))
    con.call_main(args)


def calc_meg_gamma_connectivity(args):
    args = con.read_cmd_args(utils.Bag(
        subject=args.subject,
        # atlas='laus125',
        function='calc_lables_connectivity',
        connectivity_modality='meg',
        connectivity_method='pli',
        windows_length=100,
        windows_shift=10,
        # sfreq=1000.0,
        fmin=30,
        fmax=55,
        recalc_connectivity=True,
        # max_windows_num=100,
        n_jobs=args.n_jobs
    ))
    con.call_main(args)


def calc_electrodes_rest_connectivity(args):
    args = con.read_cmd_args(utils.Bag(
        subject=args.subject,
        function='calc_electrodes_rest_connectivity',
        connectivity_modality='electrodes',
        connectivity_method='pli,cv',
        windows_length=1000,
        windows_shift=200,
        sfreq=2000.0,
        fmin=8,
        fmax=13,
        # max_windows_num=500,
        n_jobs=args.n_jobs,
    ))
    pu.run_on_subjects(args, con.main)


def calc_seed_corr(args):
    args = con.read_cmd_args(dict(
        function='calc_seed_corr',
        subject=args.subject, atlas=args.atlas,
        identifier='freesurfer', labels_regex='precentral-rh',#''post*cingulate*rh',
        seed_label_name='post_cingulate_rh', seed_label_r=5, overwrite_seed_data=True,
        n_jobs=args.n_jobs
    ))
    con.call_main(args)


def calc_electrodes_rest_connectivity_from_matlab(args):

    def read_matlab_and_split_into_windows(args):
        import scipy.io as sio
        import math
        import numpy as np
        import src.mmvt_addon.mmvt_utils as mu
        for subject in args.subject:
            data_fname = op.join(ELECTRODES_DIR, subject, 'electrodes_data_diff_data.npy')
            if not op.isfile(data_fname):
                d = sio.loadmat(op.join(ELECTRODES_DIR, subject, 'data.mat'))
                data_mat = d['data']
                electrodes_num = data_mat.shape[1]
                labels = []
                for label_mat in d['labels']:
                    group, elc1, elc2 = mu.elec_group_number('{1}-{0}'.format(*label_mat.split(' ')), True)
                    labels.append('{0}{1}-{0}{2}'.format(group, elc2, elc1))
                np.save(op.join(ELECTRODES_DIR, subject, 'electrodes.npy'), labels)
                data = data_mat.swapaxes(0, 1).reshape(electrodes_num, -1)
                E, T = data.shape
                windows_length = 1000
                windows_shift = 200
                windows_nun = math.floor((T - windows_length) / windows_shift + 1)
                data_winodws = np.zeros((windows_nun, E, windows_length))
                for w in range(windows_nun):
                    data_winodws[w] = data[:, w * windows_shift:w * windows_shift + windows_length]
                np.save(data_fname, data_winodws)

    read_matlab_and_split_into_windows(args)
    args = con.read_cmd_args(utils.Bag(
        subject=args.subject,
        function='calc_electrodes_rest_connectivity',
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


def load_connectivity_results(args):
    # from src.utils import matlab_utils as matu
    # mat_file = matu.load_mat_to_bag(op.join(MMVT_DIR, args.subject[0], 'connectivity', 'corr.mat'))
    # conn = mat_file.mAdjMat
    # labels = matu.matlab_cell_str_to_list(mat_file.mLabels)
    args = con.read_cmd_args(dict(
        function='calc_rois_matlab_connectivity',
        subject=args.subject, atlas=args.atlas,
        mat_fname=op.join(MMVT_DIR, args.subject[0], 'connectivity', 'corr.mat'),
        mat_field='mAdjMat', sorted_labels_names_field='mLabelsCort',
        n_jobs=args.n_jobs
    ))
    con.call_main(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-a', '--atlas', help='atlas name', required=False, default='aparc.DKTatlas')
    parser.add_argument('-f', '--function', help='function name', required=True)
    parser.add_argument('--n_jobs', help='cpu num', required=False, default=-1)
    args = utils.Bag(au.parse_parser(parser))
    locals()[args.function](args)