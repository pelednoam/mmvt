import os.path as op
import scipy.io as sio
import argparse
import numpy as np
from src.preproc import electrodes as elecs
from src.utils import utils
from src.utils import args_utils as au
from src.utils import preproc_utils as pu
from src.utils import matlab_utils as mu


def read_electrodes_coordiantes_from_specific_xlsx_sheet(subject, bipolar):
    args = elecs.read_cmd_args(['-s', subject, '-b', str(bipolar)])
    args.ras_xls_sheet_name = 'RAS_Snapped'
    elecs.main(subject, args)


def save_msit_single_trials_data(subject, bipolar):
    args = elecs.read_cmd_args(['-s', subject, '-b', str(bipolar)])
    args.task = 'MSIT'
    args.function = 'create_electrode_data_file'
    args.input_matlab_fname = 'electrodes_data_trials.mat'
    args.electrodes_names_field = 'electrodes'
    args.field_cond_template = '{}'
    elecs.main(subject, args)


def load_edf_data_seizure(args):
    args = elecs.read_cmd_args(utils.Bag(
        subject=args.subject,
        atlas='laus250',
        function='create_raw_data_for_blender',
        task='seizure',
        bipolar=False,
        raw_fname='Bakhamis_Amal_1.edf',
        start_time='00:01:34',
        # seizure_onset='00:03:33',
        seizure_onset='00:03:28',
        seizure_end='00:03:50',
        baseline_onset='00:01:34',
        baseline_end='00:03:11',
        lower_freq_filter=0.5,
        upper_freq_filter=70,
        power_line_notch_widths=5,
        ref_elec='CII',
        normalize_data=False,
        calc_zscore=False,
        factor=1000
    ))
    pu.run_on_subjects(args, elecs.main)


def load_edf_data_rest(args):
    args = elecs.read_cmd_args(utils.Bag(
        subject=args.subject,
        function='create_raw_data_for_blender',
        task='rest',
        bipolar=False,
        remove_power_line_noise=True,
        raw_fname='MG102_d3_Fri.edf',
        # rest_onset_time='6:50:00',
        # end_time='7:05:00',
        normalize_data=False,
        preload=False
    ))
    pu.run_on_subjects(args, elecs.main)


def get_electrodes_file_from_server(args):
    args = elecs.read_cmd_args(utils.Bag(
        subject=args.subject,
        function='prepare_subject_folder',
        sftp=True,
        sftp_username='npeled',
        sftp_domain='door.nmr.mgh.harvard.edu',
        remote_subject_dir='/space/thibault/1/users/npeled/subjects/{subject}'))
    # This line causes sometimes the sftp to hang, not sure why...
    args.sftp_password = utils.ask_for_sftp_password(args.sftp_username)
    for subject in args.subject:
        upper_subject = subject[:2].upper() + subject[2:]
        args.necessary_files['electrodes'] = \
            ['{}_RAS.{}'.format(upper_subject, file_type) for file_type in ['csv', 'xls', 'xlsx']] + \
            ['{}_RAS.{}'.format(subject, file_type) for file_type in ['csv', 'xls', 'xlsx']]
        pu.run_on_subjects(args, elecs.main)


def load_electrodes_matlab_file(args):
    from src.preproc import stim
    subject = args.subject[0]
    mat_fname = op.join(elecs.ELECTRODES_DIR, subject, 'MG106_LVF45_continuous.mat')
    d = utils.Bag(dict(**sio.loadmat(mat_fname)))
    labels = mu.matlab_cell_str_to_list(d.Label)
    fs = d.fs[0][0]
    # times: 62000 -66000
    data = d.data[:, 62000:66000]
    args.stim_channel = 'LVF04-LVF05'
    bipolar = '-' in args.stim_channel
    elecs.convert_electrodes_coordinates_file_to_npy(subject, bipolar=False)
    stim.create_stim_electrodes_positions(subject, args, labels)
    data_fname = op.join(elecs.MMVT_DIR, subject, 'electrodes', 'electrodes{}_data.npy'.format(
        '_bipolar' if bipolar else ''))
    meta_fname = op.join(elecs.MMVT_DIR, subject, 'electrodes', 'electrodes{}_meta_data.npz'.format(
        '_bipolar' if bipolar else ''))
    C, T = data.shape
    data = data.reshape((C, T, 1))
    np.save(data_fname, data)
    np.savez(meta_fname, names=labels, conditions=['rest'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-b', '--bipolar', help='bipolar', required=False, type=au.is_true)
    parser.add_argument('-f', '--function', help='function name', required=False)
    args = utils.Bag(au.parse_parser(parser))
    locals()[args.function](args)