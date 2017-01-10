import os.path as op
import argparse
from src.preproc import electrodes as elecs
from src.utils import utils
from src.utils import args_utils as au
from src.utils import preproc_utils as pu


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


def load_edf_data():
    '-s mg80b -t seizure -a laus250 -b 0 -f create_raw_data_for_blender --raw_fname seizure_1.edf --start_time 17:25:02 --seizure_time 17:25:48 --window_length 10 --baseline_delta 10 --seizure_onset_time 5 --ref_elec REF2'
    pass


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-b', '--bipolar', help='bipolar', required=False, type=au.is_true)
    parser.add_argument('-f', '--function', help='function name', required=False)
    args = utils.Bag(au.parse_parser(parser))
    locals()[args.function](args)