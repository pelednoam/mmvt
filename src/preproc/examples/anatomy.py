import os.path as op
import argparse
from src.preproc import anatomy as anat
from src.utils import utils
from src.utils import args_utils as au


def get_subject_files_using_sftp(subject, args):
    args = anat.read_cmd_args(['-s', subject, '--sftp_username', args.sftp_username, '--sftp_domain', args.sftp_domain])
    args.sftp = True
    anat.run_on_subjects(args)


def get_subject_files_from_server(subject, args):
    args = anat.read_cmd_args(['-s', subject])
    args.remote_subject_dir = op.join('/autofs/cluster/neuromind/npeled/subjects', subject)
    anat.run_on_subjects(args)


def prepare_subject_folder_from_franklin(subject, args):
    args = anat.read_cmd_args(['-s', subject])
    args.remote_subject_dir = op.join('/autofs/space/franklin_003/users/npeled/subjects_old/{}'.format(subject))
    args.function = 'prepare_local_subjects_folder'
    anat.run_on_subjects(args)


def prepare_subject_folder_from_huygens(subject, args):
    args = anat.read_cmd_args(['-s', subject])
    subject = subject[:2].upper() + subject[2:]
    args.remote_subject_dir = op.join('/space/huygens/1/users/kara/{}_SurferOutput/'.format(subject))
    args.function = 'prepare_local_subjects_folder'
    anat.run_on_subjects(args)


def darpa(subject, args):
    args = anat.read_cmd_args(['-s', subject, '-a', args.atlas])
    subject = subject[:2].upper() + subject[2:]
    args.remote_subject_dir = op.join('/space/huygens/1/users/kara/{}_SurferOutput/'.format(subject))
    anat.run_on_subjects(args)


def add_parcellation(subject, args):
    args = anat.read_cmd_args(['-s', subject, '-a', args.atlas])
    args.function = 'create_annotation_from_template,parcelate_cortex,calc_faces_verts_dic,' + \
        'save_labels_vertices,save_hemis_curv,calc_labels_center_of_mass,save_labels_coloring'
    anat.run_on_subjects(args)


def get_subject_files_using_sftp_from_ohad(subject, args):
    args = anat.read_cmd_args(['-s', subject,'-a', args.atlas])
    args.sftp = True
    args.sftp_username = 'ohadfel'
    args.sftp_domain = '127.0.0.1'
    args.sftp_port = 3333
    args.sftp_subject_dir = '/media/ohadfel/New_Volume/subs/{}'.format(subject)
    args.function = 'prepare_local_subjects_folder'
    anat.run_on_subjects(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-a', '--atlas', help='atlas name', required=False, default='aparc.DKTatlas40')
    parser.add_argument('-u', '--sftp_username', help='sftp username', required=False, default='npeled')
    parser.add_argument('-d', '--sftp_domain', help='sftp domain', required=False, default='door.nmr.mgh.harvard.edu')
    parser.add_argument('--remote_subject_dir', help='remote_subjects_dir', required=False,
                        default='/autofs/cluster/neuromind/npeled/subjects')
    parser.add_argument('-f', '--function', help='function name', required=True)
    args = utils.Bag(au.parse_parser(parser))
    for subject in args.subject:
        locals()[args.function](subject, args)