import argparse
import shutil
import os.path as op
import glob
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



def fmri_msit_pipeline(args):
    '-s pp009 -a laus250 -f fmri_pipeline -t MSIT --contrast_template "*Interference*"'
    for subject in args.subject:
        args = fmri.read_cmd_args(dict(
            subject=subject,
            atlas=args.atlas,
            function='fmri_pipeline',
            task='MSIT',
            contrast_template='*Interference*'
        ))
        pu.run_on_subjects(args, fmri.main)


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


def analyze_4d_data(args):
    # '-s subject-name -a atlas-name -f analyze_4d_data --fmri_file_template {subject}*{morph_to_subject}.{hemi}.{format}  --morph_labels_to_subject fsaverage'
    # '-f analyze_4d_data -a laus125 -s "file:/homes/5/npeled/space1/Documents/memory_task/subjects.txt"'
    args = fmri.read_cmd_args(dict(
        subject=args.subject,
        atlas=args.atlas,
        function='analyze_4d_data',
        # fmri_file_template='*rest*.{hemi}*.{format}',
        # fmri_file_template='rest.sm6.fsaverage6.{hemi}.mgz',
        fmri_file_template='rest_linda.sm6.{subject}.{hemi}.mgz',
        # template_brain='fsaverage5',
        # template_brain='fsaverage6',
        labels_extract_mode='pca,pca_2,pca_4,pca_8,pca_16'
    ))
    pu.run_on_subjects(args, fmri.main)


def clean_4d_data(args):
    '''
    python -m src.preproc.fMRI -s nmr00474,nmr00502,nmr00515,nmr00603,nmr00609,nmr00626,nmr00629,nmr00650,nmr00657,nmr00669,nmr00674,nmr00681,nmr00683,nmr00692,nmr00698,nmr00710
        -a laus125 -f clean_resting_state_data --template_brain fsaverage5 --fmri_file_template "f.nii*" --remote_subject_dir "/space/franklin/1/users/sx424/mem_flex/subjects/{subject}"'
    '''
    args = fmri.read_cmd_args(dict(
        subject=args.subject,
        atlas=args.atlas,
        function='clean_4d_data',
        fmri_file_template='rest.nii*',
        fsd='rest_linda'
        # template_brain='fsaverage5',
    ))
    pu.run_on_subjects(args, fmri.main)


def get_subjects_files(args):
    ''' -f get_subjects_files -s "file:/homes/5/npeled/space1/Documents/memory_task/subjects.txt" '''
    subjects = pu.decode_subjects(args.subject)
    for subject in subjects:
        subject_fol = op.join(fmri.FMRI_DIR, subject)
        # data_fol = '/cluster/neuromind/douw/scans/adults/{}/surf'.format(subject)
        data_fol = '/cluster/neuromind/douw/scans/adults/{}/bold'.format(subject)
        # template = '*.{}_bld*_rest_reorient_skip_faln_mc_g1000000000_bpss_resid_fsaverage6_sm6.nii.gz'.format(subject)
        template = '{}_*_rest.nii'.format(subject)
        files = glob.glob(op.join(data_fol, '**', template), recursive=True)
        # if len(files) % 2 == 0:
        if len(files) >= 1:
            utils.make_dir(subject_fol)
            for fname in files:
                # hemi = 'rh' if 'rh' in fname.split(op.sep)[-1] else 'lh'
                # output_fname = op.join(subject_fol, 'rest.sm6.fsaverage6.{}.mgz'.format(hemi))
                output_fname = op.join(subject_fol, 'rest.nii')
                if not op.isfile(output_fname):
                    shutil.copy(fname, output_fname)
                else:
                    print('{} already exist!'.format(output_fname))
        else:
            print("Couldn't find the files for {}!".format(subject))


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