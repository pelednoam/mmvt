import os.path as op
import argparse
import shutil

from src.preproc import anatomy as anat
from src.utils import utils
from src.utils import args_utils as au
from src.utils import preproc_utils as pu
# from gooey import Gooey

SUBJECTS_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()


def get_subject_files_using_sftp(args):
    for subject in args.subject:
        args = anat.read_cmd_args(dict(
            subject=subject,
            atlas=args.atlas,
            sftp_username=args.sftp_username,
            sftp_domain=args.sftp_domain,
            sftp=True,
            remote_subject_dir=args.remote_subject_dir,
            function='prepare_subject_folder'
        ))
        pu.run_on_subjects(args, anat.main)


def get_subject_files_from_mad(args):
    for subject in args.subject:
        args = anat.read_cmd_args(dict(
            subject=subject,
            atlas=args.atlas,
            remote_subject_dir='/mnt/cashlab/Original Data/MG/{subject}/{subject}_Notes_and_Images/{subject}_SurferOutput',
            function='prepare_subject_folder'
        ))
        pu.run_on_subjects(args, anat.main)


@utils.check_for_freesurfer
def create_annot_from_mad(args):
    remote_subject_dir_template = '/mnt/cashlab/Original Data/MG/{subject}/{subject}_Notes_and_Images/{subject}_SurferOutput'
    for subject in args.subject:
        remote_subject_dir = remote_subject_dir_template.format(subject=subject)
        if utils.both_hemi_files_exist(op.join(remote_subject_dir, 'label', '{hemi}.aparc.DKTatlas.annot')):
            print('{} has already both annot files!'.format(subject))
            continue
        args = anat.read_cmd_args(dict(
            subject=subject.lower(),
            atlas=args.atlas,
            remote_subject_dir=remote_subject_dir_template,
            function='create_annotation',
            ignore_missing=True,
        ))
        pu.run_on_subjects(args, anat.main)
        if not utils.both_hemi_files_exist(op.join(SUBJECTS_DIR, subject.lower(), 'label', '{hemi}.aparc.DKTatlas.annot')):
            print('Couldn\'t create annot files for {}!'.format(subject))
            continue
        local_annot_fol = utils.make_dir(op.join(SUBJECTS_DIR, 'annot_files', subject.lower()))
        for hemi in utils.HEMIS:
            local_annot_fname = op.join(SUBJECTS_DIR, subject.lower(), 'label', '{}.aparc.DKTatlas.annot'.format(hemi))
            remote_annot_fname = op.join(remote_subject_dir, 'label', '{}.aparc.DKTatlas.annot'.format(hemi))
            local_temp_annot_fname = op.join(local_annot_fol, '{}.aparc.DKTatlas.annot'.format(hemi))
            if not op.isfile(remote_annot_fname):
                if op.isfile(local_annot_fname):
                    shutil.copyfile(local_annot_fname, local_temp_annot_fname)
                else:
                    print('Can\'t copy {} for {}, it doesn\'t exist!'.format(local_annot_fname, subject))


# def get_subject_files_from_server(subject, args):
#     args = anat.read_cmd_args(['-s', subject, '-a', args.atlas])
#     args.remote_subject_dir = op.join('/autofs/cluster/neuromind/npeled/subjects', subject)
#     pu.run_on_subjects(args, anat.main)


def prepare_subject_folder_from_franklin(subject, args):
    args = anat.read_cmd_args(['-s', subject, '-a', args.atlas])
    args.remote_subject_dir = op.join('/autofs/space/franklin_003/users/npeled/subjects_old/{}'.format(subject))
    args.function = 'prepare_subject_folder'
    pu.run_on_subjects(args, anat.main)


def darpa(args):
    for subject in args.subject:
        darpa_subject = subject[:2].upper() + subject[2:]
        args = anat.read_cmd_args(utils.Bag(
            subject=subject,
            remote_subject_dir=op.join('/space/huygens/1/users/kara/{}_SurferOutput/'.format(darpa_subject))
        ))
        pu.run_on_subjects(args, anat.main)


def darpa_prep(args):
    for subject in args.subject:
        darpa_subject = subject[:2].upper() + subject[2:]
        args = anat.read_cmd_args(utils.Bag(
            subject=subject,
            function='prepare_subject_folder',
            remote_subject_dir=op.join('/space/huygens/1/users/kara/{}_SurferOutput/'.format(darpa_subject))
        ))
        pu.run_on_subjects(args, anat.main)


def darpa_prep_angelique(args):
    import glob
    for subject in args.subject:
        darpa_subject = subject[:2].upper() + subject[2:]
        root = op.join('/homes/5/npeled/space1/Angelique/recon-alls', darpa_subject)
        recon_all_dirs = glob.glob(op.join(root, '**', '*SurferOutput*'), recursive=True)
        if len(recon_all_dirs) == 0:
            print("Can't find the recon-all folder for {}!".format(subject))
            continue
        args = anat.read_cmd_args(utils.Bag(
            subject=subject,
            function='prepare_subject_folder',
            remote_subject_dir=recon_all_dirs[0]
        ))
        pu.run_on_subjects(args, anat.main)



def darpa_sftp(args):
    for subject in args.subject:
        darpa_subject = subject[:2].upper() + subject[2:]
        args = anat.read_cmd_args(utils.Bag(
            subject=subject,
            remote_subject_dir=op.join('/space/huygens/1/users/kara/{}_SurferOutput/'.format(darpa_subject)),
            sftp=True,
            sftp_username='npeled',
            sftp_domain='door.nmr.mgh.harvard.edu',
        ))
        pu.run_on_subjects(args, anat.main)


def darpa_prep_huygens(subject, args):
    args = anat.read_cmd_args(['-s', subject, '-a', args.atlas])
    subject = subject[:2].upper() + subject[2:]
    args.remote_subject_dir = op.join('/space/huygens/1/users/kara/{}_SurferOutput/'.format(subject))
    args.function = 'prepare_subject_folder'
    pu.run_on_subjects(args, anat.main)


def darpa_prep_lili(subject, args):
    args = anat.read_cmd_args(['-s', subject, '-a', args.atlas, '--sftp_username', args.sftp_username, '--sftp_domain', args.sftp_domain])
    args.remote_subject_dir = op.join('/autofs/space/lilli_001/users/DARPA-Recons', subject)
    args.sftp = True
    args.function = 'prepare_subject_folder'
    pu.run_on_subjects(args, anat.main)


def add_parcellation(subject, args):
    args = anat.read_cmd_args(['-s', subject, '-a', args.atlas])
    args.function = 'create_annotation_from_template,parcelate_cortex,calc_faces_verts_dic,' + \
        'save_labels_vertices,save_hemis_curv,calc_labels_center_of_mass,save_labels_coloring'
    pu.run_on_subjects(args, anat.main)


def get_subject_files_using_sftp_from_ohad(subject, args):
    args = anat.read_cmd_args(['-s', subject,'-a', args.atlas])
    args.sftp = True
    args.sftp_username = 'ohadfel'
    args.sftp_domain = '127.0.0.1'
    args.sftp_port = 4444
    args.sftp_subject_dir = '/media/ohadfel/New_Volume/subs/{}'.format(subject)
    args.remote_subject_dir = '/media/ohadfel/New_Volume/subs/{}'.format(subject)
    args.function = 'prepare_subject_folder'
    pu.run_on_subjects(args, anat.main)


def get_subject_files_from_server(args):
    args = anat.read_cmd_args(dict(
        subject=args.subject,
        atlas=args.atlas,
        function='prepare_subject_folder',
        sftp=True,
        sftp_username='npeled',
        sftp_domain='door.nmr.mgh.harvard.edu',
        remote_subject_dir='/space/thibault/1/users/npeled/subjects/{subject}'))
    pu.run_on_subjects(args, anat.main)


# https://github.com/chriskiehl/Gooey
# @Gooey
def main():
    import collections
    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-a', '--atlas', help='atlas name', required=False, default='aparc.DKTatlas')
    parser.add_argument('-u', '--sftp_username', help='sftp username', required=False, default='npeled')
    parser.add_argument('-d', '--sftp_domain', help='sftp domain', required=False, default='door.nmr.mgh.harvard.edu')
    parser.add_argument('--remote_subject_dir', help='remote_subjects_dir', required=False,
                        default='/space/thibault/1/users/npeled/subjects/{subject}')
    parser.add_argument('-f', '--function', help='function name', required=True,
                        choices=[f_name for f_name, f in globals().items() if isinstance(f, collections.Callable)
                                 if f_name not in ['Gooey', 'main']])
    args = utils.Bag(au.parse_parser(parser))
    # for subject in args.subject:
    globals()[args.function](args)


if __name__ == '__main__':
    main()