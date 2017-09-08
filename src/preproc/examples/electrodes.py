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


def load_edf_data_seizure_2(args):
    args = elecs.read_cmd_args(utils.Bag(
        subject=args.subject,
        atlas='laus125',
        function='create_raw_data_for_blender',
        task='seizure',
        bipolar=args.bipolar,
        raw_fname='/cluster/neuromind/npeled/taha/dm04002705/edf/DMphaseIISz_TG.edf',
        start_time='00:00:00',
        seizure_onset='00:01:30',
        seizure_end='00:02:00',
        baseline_onset='00:00:00',
        baseline_end='00:01:00',
        lower_freq_filter=0.5,
        upper_freq_filter=70,
        power_line_notch_widths=5,
        ref_elec='PST1',
        normalize_data=False,
        calc_zscore=False,
        factor=1000,
        channels_names_mismatches='LFO=LIF'
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


def load_electrodes_matlab_stim_file2(args):
    from src.preproc import stim
    args = elecs.read_cmd_args(utils.Bag(
        subject=args.subject,
        bipolar=True
    ))
    args = pu.add_default_args(args, {'error_radius': 3, 'elec_length': 4, 'file_frefix': ''})

    subject = args.subject[0]
    mat_fname = op.join(elecs.ELECTRODES_DIR, subject, 'MG106_HGP_sess2_3.mat')
    d = mu.load_mat_to_bag(mat_fname)
    labels = mu.matlab_cell_str_to_list(d.Label)
    labels = [l.replace(' ', '-') for l in labels]
    labels = ['{0}{1}-{0}{2}'.format(*utils.elec_group_number(l, bipolar=True)) for l in labels]
    time = d.time2.squeeze()
    data = d.Data2
    data = data.reshape((*data.shape, 1))

    args.stim_channel = 'sess2'
    args.bipolar = '-' in labels[0]
    elecs.convert_electrodes_coordinates_file_to_npy(subject, bipolar=False)
    output_file = stim.create_stim_electrodes_positions(subject, args, labels)
    data_fname = op.join(elecs.MMVT_DIR, subject, 'electrodes', 'electrodes{}_data.npy'.format(
        '_bipolar' if args.bipolar else ''))
    meta_fname = op.join(elecs.MMVT_DIR, subject, 'electrodes', 'electrodes{}_meta_data.npz'.format(
        '_bipolar' if args.bipolar else ''))
    np.save(data_fname, data)
    np.savez(meta_fname, names=labels, conditions=['rest'], time=time)
    print('Saving data in {} and {}'.format(data_fname, meta_fname))


def load_electrodes_matlab_stim_file(args):
    from src.preproc import stim
    args = elecs.read_cmd_args(utils.Bag(
        subject=args.subject,
        bipolar = True
    ))
    args = pu.add_default_args(args, {'error_radius': 3, 'elec_length': 4, 'file_frefix': ''})

    subject = args.subject[0]
    mat_fname = op.join(elecs.ELECTRODES_DIR, subject, 'MG106_LVF56_continuous.mat')
    d = mu.load_mat_to_bag(mat_fname)
    labels = mu.matlab_cell_str_to_list(d.Label)
    fs = d.fs[0][0]
    T = d.data.shape[1]
    # t1, t2 = 62000, 66000
    t1, t2 = T - 5*fs, T # Take last 5 seconds
    data = d.data[:, t1:t2]

    bad_electrodes = ['ROF04-05', 'ROF05-06', 'LPF14-NT']
    labels, data = elecs.remove_bad_channels(labels, data, bad_electrodes)
    data = data.reshape((*data.shape, 1))

    args.stim_channel = 'LVF05-LVF06'
    args.bipolar = '-' in args.stim_channel
    elecs.convert_electrodes_coordinates_file_to_npy(subject, bipolar=False)
    output_file = stim.create_stim_electrodes_positions(subject, args, labels)
    # First you need to run find_rois in electrodes_rois project:
    # python -m src.find_rois -s mg106 -b 1 --pos_fname '$MMVT/mg106/electrodes/electrodes_bipolar_stim_LVF04-LVF05_positions.npz' --output_postfix _stim_LVF04-LVF05
    # stim.set_labels_colors(subject, args, stim_dict=dict(labels=labels, data=data, conditions=['rest']))
    data_fname = op.join(elecs.MMVT_DIR, subject, 'electrodes', 'electrodes{}_data.npy'.format(
        '_bipolar' if args.bipolar else ''))
    meta_fname = op.join(elecs.MMVT_DIR, subject, 'electrodes', 'electrodes{}_meta_data.npz'.format(
        '_bipolar' if args.bipolar else ''))
    np.save(data_fname, data)
    np.savez(meta_fname, names=labels, conditions=['rest'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-b', '--bipolar', help='bipolar', required=False, type=au.is_true)
    parser.add_argument('-f', '--function', help='function name', required=False)
    args = utils.Bag(au.parse_parser(parser))
    locals()[args.function](args)