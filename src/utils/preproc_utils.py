import os
import os.path as op
from collections import defaultdict
import glob
import traceback

from src.utils import utils
from src.utils import args_utils as au

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
MMVT_DIR = op.join(LINKS_DIR, 'mmvt')


def run_on_subjects(args, main_func, subjects_itr=None, subject_func=None):
    if subjects_itr is None:
        subjects_itr = args.subject
    subjects_flags, subjects_errors = {}, {}
    args.n_jobs = utils.get_n_jobs(args.n_jobs)
    if args.necessary_files == '':
        args.necessary_files = dict()
    if 'sftp_password' not in args or args.sftp_password == '':
        args.sftp_password = utils.get_sftp_password(
            args.subject, SUBJECTS_DIR, args.necessary_files, args.sftp_username, args.overwrite_fs_files) \
            if args.sftp else ''
    if '*' in args.subject:
        args.subject = [utils.namebase(fol) for fol in glob.glob(op.join(SUBJECTS_DIR, args.subject))]
    os.environ['SUBJECTS_DIR'] = SUBJECTS_DIR
    for tup in subjects_itr:
        subject = get_subject(tup, subject_func)
        utils.make_dir(op.join(MMVT_DIR, subject, 'mmvt'))
        remote_subject_dir = utils.build_remote_subject_dir(args.remote_subject_dir, subject)
        print('****************************************************************')
        print('subject: {}, atlas: {}'.format(subject, args.atlas))
        print('remote dir: {}'.format(remote_subject_dir))
        print('****************************************************************')
        os.environ['SUBJECT'] = subject
        flags = dict()
        try:
            # if utils.should_run(args, 'prepare_subject_folder'):
            # I think we always want to run this
            # *) Prepare the local subject's folder
            flags['prepare_subject_folder'] = prepare_subject_folder(
                subject, remote_subject_dir, args)
            if not flags['prepare_subject_folder'] and not args.ignore_missing:
                ans = input('Do you wish to continue (y/n)? ')
                if not au.is_true(ans):
                    continue

            flags = main_func(tup, remote_subject_dir, args, flags)
            subjects_flags[subject] = flags
        except:
            subjects_errors[subject] = traceback.format_exc()
            print('Error in subject {}'.format(subject))
            print(traceback.format_exc())

    errors = defaultdict(list)
    ret = True
    for subject, flags in subjects_flags.items():
        print('subject {}:'.format(subject))
        for flag_type, val in flags.items():
            print('{}: {}'.format(flag_type, val))
            if not val:
                errors[subject].append(flag_type)
    if len(errors) > 0:
        ret = False
        print('Errors:')
        for subject, error in errors.items():
            print('{}: {}'.format(subject, error))
    return ret


def prepare_subject_folder(subject, remote_subject_dir, args, necessary_files=None):
    if necessary_files is None:
        necessary_files = args.necessary_files
    return utils.prepare_subject_folder(
        necessary_files, subject, remote_subject_dir, SUBJECTS_DIR,
        args.sftp, args.sftp_username, args.sftp_domain, args.sftp_password,
        args.overwrite_fs_files, args.print_traceback, args.sftp_port)


def get_subject(tup, subject_func):
    if subject_func is None:
        if isinstance(tup, str):
            subject = tup
        else:
            raise Exception('subject_func is None, but tup is not str!')
    else:
        subject = subject_func(tup)
    return subject


def add_common_args(parser):
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-a', '--atlas', help='atlas name', required=False, default='aparc.DKTatlas40')
    parser.add_argument('-f', '--function', help='function name', required=False, default='all')
    parser.add_argument('--exclude', help='functions not to run', required=False, default='', type=au.str_arr_type)
    parser.add_argument('--n_jobs', help='cpu num', required=False, default=-1)

    # Prepare subject dir
    parser.add_argument('--necessary_files', help='necessary_files', required=False, default='')
    parser.add_argument('--remote_subject_dir', help='remote_subject_dir', required=False, default='')
    parser.add_argument('--ignore_missing', help='ignore missing files', required=False, default=0, type=au.is_true)
    parser.add_argument('--overwrite_fs_files', help='overwrite freesurfer files', required=False, default=0,
                        type=au.is_true)
    parser.add_argument('--sftp', help='copy subjects files over sftp', required=False, default=0, type=au.is_true)
    parser.add_argument('--sftp_username', help='sftp username', required=False, default='')
    parser.add_argument('--sftp_domain', help='sftp domain', required=False, default='')
    parser.add_argument('--sftp_port', help='sftp port', required=False, default=22, type=int)
    parser.add_argument('--print_traceback', help='print_traceback', required=False, default=1, type=au.is_true)


def check_freesurfer():
    if os.environ.get('FREESURFER_HOME', '') == '':
        raise Exception('Source freesurfer and rerun')


def get_links():
    links_dir = utils.get_links_dir()
    subjects_dir = utils.get_link_dir(links_dir, 'subjects', 'SUBJECTS_DIR')
    freesurfer_home = utils.get_link_dir(links_dir, 'freesurfer', 'FREESURFER_HOME')
    mmvt_dir = utils.get_link_dir(links_dir, 'mmvt')
    return subjects_dir, mmvt_dir, freesurfer_home


def tryit_ret_bool(func):
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
            retval = True
        except:
            print('Error in {}!'.format(func.__name__))
            print(traceback.format_exc())
            retval = False
        return retval

    return wrapper
