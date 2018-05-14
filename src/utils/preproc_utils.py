import os
import os.path as op
from collections import defaultdict
import glob
import traceback
import shutil
import collections
import logging

from src.utils import utils
from src.utils import args_utils as au

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
MMVT_DIR = op.join(LINKS_DIR, 'mmvt')


def decode_subjects(subjects):
    for sub in subjects:
        if '*' in sub:
            subjects.remove(sub)
            subjects.extend([utils.namebase(fol) for fol in glob.glob(op.join(SUBJECTS_DIR, sub))])
        elif 'file:' in sub:
            subjects.remove(sub)
            subjects.extend(utils.read_list_from_file(sub[len('file:'):]))
    return subjects


def init_args(args):
    args.n_jobs = utils.get_n_jobs(args.n_jobs)
    if args.necessary_files == '':
        args.necessary_files = dict()
    args.subject = decode_subjects(args.subject)
    if 'sftp_password' not in args or args.sftp_password == '':
        args.sftp_password = utils.get_sftp_password(
            args.subject, SUBJECTS_DIR, args.necessary_files, args.sftp_username, args.overwrite_fs_files) \
            if args.sftp else ''
    set_default_args(args)
    args.atlas = utils.get_real_atlas_name(args.atlas)
    os.environ['SUBJECTS_DIR'] = SUBJECTS_DIR
    return args


def run_on_subjects(args, main_func, subjects_itr=None, subject_func=None):
    if subjects_itr is None:
        subjects_itr = args.subject
    subjects_flags, subjects_errors = {}, {}
    args = init_args(args)
    for tup in subjects_itr:
        subject = get_subject(tup, subject_func)
        utils.make_dir(op.join(MMVT_DIR, subject, 'mmvt'))
        remote_subject_dir = utils.build_remote_subject_dir(args.remote_subject_dir, subject)
        logging.info(args)
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
            flags['prepare_subject_folder'] = True

            flags = main_func(tup, remote_subject_dir, args, flags)
            subjects_flags[subject] = flags
        except:
            subjects_errors[subject] = traceback.format_exc()
            print('Error in subject {}'.format(subject))
            print(traceback.format_exc())

    errors = defaultdict(list)
    ret = True
    good_subjects, bad_subjects = [], []
    logs_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'logs'))
    logging.basicConfig(filename=op.join(logs_fol, 'preproc.log'), level=logging.DEBUG)
    for subject, flags in subjects_flags.items():
        print('subject {}:'.format(subject))
        logging.info('subject {}:'.format(subject))
        for flag_type, val in flags.items():
            print('{}: {}'.format(flag_type, val))
            logging.info('{}: {}'.format(flag_type, val))
            if not val:
                errors[subject].append(flag_type)
    if len(errors) > 0:
        ret = False
        print('Errors:')
        logging.info('Errors:')
        for subject, error in errors.items():
            print('{}: {}'.format(subject, error))
            logging.info('{}: {}'.format(subject, error))
    for subject in subjects_flags.keys():
        if len(errors[subject]) == 0:
            good_subjects.append(subject)
        else:
            bad_subjects.append(subject)
    print('Good subjects:\n {}'.format(good_subjects))
    logging.info('Good subjects:\n {}'.format(good_subjects))
    print('Bad subjects:\n {}'.format(bad_subjects))
    logging.info('Good subjects:\n {}'.format(good_subjects))
    utils.write_list_to_file(good_subjects, op.join(utils.get_logs_fol(), 'good_subjects.txt'))
    utils.write_list_to_file(bad_subjects, op.join(utils.get_logs_fol(), 'bad_subjects.txt'))
    return ret


def set_default_args(args, ini_name='default_args.ini'):
    settings = utils.read_config_ini(MMVT_DIR, ini_name)
    if settings is not None:
        import inspect
        module_name = ''
        for frm in inspect.stack():
            if 'src/preproc' in frm.filename:
                module_name = utils.namebase(frm.filename)
                break
        if module_name != '' and module_name in settings.sections():
            for args_key in args.keys():
                settings_val = settings[module_name].get(args_key, '')
                if settings_val != '':
                    print('{}: setting {} to {}'.format(ini_name, args_key, settings_val))
                    args[args_key] = settings_val


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


def add_default_args(args, default_args):
    for def_key, def_val in default_args.items():
        if def_key not in args:
            args[def_key] = def_val
    return args


def add_common_args(parser):
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-a', '--atlas', help='atlas name', required=False, default='aparc.DKTatlas40')
    parser.add_argument('-f', '--function', help='function name', required=False, default='all', type=au.str_arr_type)
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
    parser.add_argument('--sftp_password', help='sftp port', required=False, default='')
    parser.add_argument('--print_traceback', help='print_traceback', required=False, default=1, type=au.is_true)

    # global folders
    parser.add_argument('--meg_dir', required=False, default='')
    parser.add_argument('--mri_dir', required=False, default='')
    parser.add_argument('--mmvt_dir', required=False, default='')


def set_default_folders(args):
    if args.mri_dir == '':
        args.mri_dir = SUBJECTS_DIR
    if args.mmvt_dir == '':
        args.mmvt_dir = MMVT_DIR


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
            retval = func(*args, **kwargs)
        except:
            print('Error in {}!'.format(func.__name__))
            print(traceback.format_exc())
            retval = False
        return retval

    return wrapper


def backup_folder(subject, folder_name, backup_suffix='_backup'):
    '''
    :param folder_name: one of the modalities undert MMVT_DIR/subject:
     fmri / meg / eeg / electrodes / connectivity
    :return: 
    '''
    folder_names = ['fmri','meg','eeg','electrodes', 'connectivity']
    if folder_name not in folder_names:
        print('folder name should be on of: {}'.format(folder_names))
        return
    source_dir = op.join(MMVT_DIR, subject, folder_name)
    if not op.isdir(source_dir):
        print('{} does not exist!'.format(source_dir))
        return
    target_dir = op.join(MMVT_DIR, subject, '{}{}'.format(folder_name, backup_suffix))
    if op.isdir(target_dir):
        print('{} already exist!'.format(target_dir))
        return
    print('backup {} to {}'.format(source_dir, target_dir))
    shutil.copytree(source_dir, target_dir)


def check_func_output(ret):
    if isinstance(ret, collections.Iterable):
        return ret[0], ret[1]
    else:
        return None