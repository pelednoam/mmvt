import argparse
from src.preproc import fMRI as fmri
from src.utils import utils
from src.utils import args_utils as au
from src.utils import preproc_utils as pu


def get_subject_files_using_sftp(args):
    for subject in args.subject:
        args = fmri.read_cmd_args(dict(
            subject=subject,
            atlas=args.atlas,
            sftp_username=args.sftp_username,
            sftp_domain=args.sftp_domain,
            sftp=True,
            remote_subject_dir=args.remote_subject_dir,
            function='prepare_subject_folder'
        ))
        pu.run_on_subjects(args, fmri.main)


def load_rest_to_colin():
    args = fmri.read_cmd_args(['-s', subject])
    args.task = 'REST'
    args.function = 'project_volume_to_surface,find_clusters'
    args.contrast = 'rest'
    args.volume_name = 'spmT_0001'
    fmri.main(subject, mri_subject, args)


def asdf():
    '-s pp009 -a laus250 -f project_volume_to_surface -t ARC --volume_name pp009_ARC_PPI_highrisk_L_VLPFC --contrast highrisk'
    '-s pp009 -a laus250 -f calc_fmri_min_max -t ARC --volume_name pp009_ARC_PPI_highrisk_L_VLPFC --contrast highrisk'
    pass

def fsfast():
    args = fmri.read_cmd_args(['-s', subject])
    args.task = 'MSIT'
    args.function = 'fmri_pipeline'
    args.contrast_name = 'interference'
    args.atlas = 'laus250'
    fmri.main(subject, mri_subject, args)


def pet():
    args = fmri.read_cmd_args(['-s', subject])
    args.threshold = 0
    args.is_pet = True
    args.symetric_colors = False
    args.atlas = 'laus250'
    fmri.main(subject, mri_subject, args)
    '-s s02 --threshold 0 --is_pet 1 --symetric_colors 0 --overwrite_surf_data 1 --remote_subject_dir /local_mount/space/thibault/1/users/npeled/artur/recon_tese/{subject}'


def analyze_resting_state(args):
    '-s subject-name -a atlas-name -f analyze_resting_state --fmri_file_template {subject}*{morph_to_subject}.{hemi}.{format}  --morph_labels_to_subject fsaverage'
    args = fmri.read_cmd_args(dict(
        subject=args.subject,
        atlas=args.atlas,
        function='analyze_resting_state',
        fmri_file_template='*rest*.{hemi}*.{format}',
        rest_template='fsaverage5',
        morph_labels_from_subject='fsaverage5c',
        labels_extract_mode='mean'
    ))
    pu.run_on_subjects(args, fmri.main)


def clean_resting_state(args):
    'python -m src.preproc.fMRI -s nmr00474,nmr00502,nmr00515,nmr00603,nmr00609,nmr00626,nmr00629,nmr00650,nmr00657,nmr00669,nmr00674,nmr00681,nmr00683,nmr00692,nmr00698,nmr00710 -a laus125 -f clean_resting_state_data --rest_template fsaverage5 --fmri_file_template "f.nii*" --remote_subject_dir "/space/franklin/1/users/sx424/mem_flex/subjects/{subject}"'
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=False, type=au.str_arr_type, default='colin27')
    parser.add_argument('-a', '--atlas', help='atlas name', required=False, default='aparc.DKTatlas40')
    parser.add_argument('-u', '--sftp_username', help='sftp username', required=False, default='npeled')
    parser.add_argument('-d', '--sftp_domain', help='sftp domain', required=False, default='door.nmr.mgh.harvard.edu')
    parser.add_argument('--remote_subject_dir', help='remote_subjects_dir', required=False,
                        default='/space/thibault/1/users/npeled/subjects/{subject}')
    parser.add_argument('-f', '--function', help='function name', required=True)
    args = utils.Bag(au.parse_parser(parser))
    locals()[args.function](args)