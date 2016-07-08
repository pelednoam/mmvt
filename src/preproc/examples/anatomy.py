import argparse
from src.preproc import anatomy_preproc as anat
from src.utils import utils
from src.utils import args_utils as au


def get_subject_files_using_sftp(subject):
    args = anat.read_cmd_args(['-s', subject])
    args.sftp = True
    args.sftp_username = 'npeled'
    args.sftp_domain = 'door.nmr.mgh.harvard.edu'
    args.remote_subjects_dir = '/autofs/cluster/neuromind/npeled/subjects'
    anat.run_on_subjects(args)


def get_subject_files_from_server(subject):
    args = anat.read_cmd_args(['-s', subject])
    args.remote_subjects_dir = '/autofs/cluster/neuromind/npeled/subjects'
    anat.run_on_subjects(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-f', '--function', help='function name', required=True)
    args = utils.Bag(au.parse_parser(parser))
    for subject in args.subject:
        locals()[args.function](subject)