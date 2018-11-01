import argparse
import shutil
import os.path as op
import os
import glob
from src.preproc import fMRI as fmri
from src.utils import utils
from src.utils import args_utils as au
from src.utils import preproc_utils as pu
from src.utils import freesurfer_utils as fu

LINKS_DIR = utils.get_links_dir()
FMRI_DIR = utils.get_link_dir(utils.get_links_dir(), 'fMRI')
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')


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
        # fmri_file_template='fmcpr.up.sm6.{subject}.{hemi}.nii.gz',
        fmri_file_template='{subject}_bld???_rest_reorient_skip_faln_mc_g1000000000_bpss_resid_{hemi}.mgz',
        # template_brain='fsaverage5',
        # template_brain='fsaverage6',
        # labels_extract_mode='mean,pca,pca_2,pca_4,pca_8',
        labels_extract_mode='mean',
        overwrite_labels_data=True
    ))
    pu.run_on_subjects(args, fmri.main)


def project_volume_to_surface(args):
    args = fmri.read_cmd_args(dict(
        subject=args.subject,
        function='project_volume_to_surface',
        fmri_file_template='f*.gz',
        sftp_username=args.sftp_username,
        sftp_domain=args.sftp_domain,
        sftp=True,
        remote_subject_dir=args.remote_subject_dir,
    ))
    pu.run_on_subjects(args, fmri.main)


def load_labels_ts(args):
    args = fmri.read_cmd_args(dict(
        subject=args.subject,
        atlas=args.atlas,
        function='load_labels_ts',
        labels_order_fname=op.join(fmri.MMVT_DIR, 'labels_order', 'linda_laus125_order.txt'),
        labels_extract_mode='mean',
        excluded_labels='corpuscallosum,unknown',
        labels_indices_to_remove_from_data='0,4,113,117',
        st_template='{subject}_{atlas}_mri.txt',
        backup_existing_files=False,
        pick_the_first_one=True
    ))
    pu.run_on_subjects(args, fmri.main)


def calc_labels_mean_freesurfer(args):
    '''
    python -m src.preproc.fMRI -a laus125 -f calc_labels_mean_freesurfer --fmri_file_template "{hemi}.{subject}_bld014_rest_reorient_skip_faln_mc_g1000000000_bpss_resid_fsaverage6_sm6_fsaverage5*.mgz" --excluded_labels corpuscallosum,unknown --overwrite_labels_data 1 --remote_fmri_dir "/autofs/cluster/scratch/tuesday/noam/DataProcessed_memory/{subject}/surf" -s 'nmr00506','nmr00599','nmr00515','nmr00692','nmr00657','nmr00609','nmr00468','nmr00629','nmr00681','nmr00643','nmr00448','nmr00650','nmr00674','nmr00669','nmr00603','nmr00710','nmr00683','nmr00640','nmr00634','nmr00502','nmr00698'
    '''
    args = fmri.read_cmd_args(dict(
        subject=args.subject,
        atlas=args.atlas,
        function='calc_labels_mean_freesurfer',
        fmri_file_template='{hemi}.{subject}_bld???_rest_reorient_skip_faln_mc_g1000000000_bpss_resid_fsaverage6_sm6_fsaverage5*.mgz',
        excluded_labels='corpuscallosum,unknown',
        remote_fmri_dir='/autofs/cluster/neuromind/douw/scans/adults/{subject}/surf',
        overwrite_labels_data=True,
        sftp_username=args.sftp_username,
        sftp_domain=args.sftp_domain,
        sftp=True,
        remote_subject_dir=args.remote_subject_dir,
    ))
    pu.run_on_subjects(args, fmri.main)


def save_dynamic_activity_map(args):
    args = fmri.read_cmd_args(dict(
        subject=args.subject,
        atlas=args.atlas,
        function='save_dynamic_activity_map',
        fmri_file_template='fmcpr.up.sm6.{subject}.{hemi}.*',
        overwrite_activity_data=True
    ))
    pu.run_on_subjects(args, fmri.main)


def calc_subcorticals_activity(args):
    args = fmri.read_cmd_args(dict(
        subject=args.subject,
        function='calc_subcorticals_activity',
        # fmri_file_template='rest*',
        fmri_file_template='fmcpr.sm6.mni305.2mm.*',
        labels_extract_mode='mean', #,pca,pca_2,pca_4,pca_8',
        overwrite_subs_data=True
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


def create_nii_from_npy(args):
    import numpy as np
    import nibabel as nib
    contrast_name = 'non-interference-v-interference'
    subject = 'mg78_old'
    fmri_subject = 'mg78'
    T1 = nib.load(op.join(SUBJECTS_DIR, subject, 'mri', 'T1.mgz'))
    affine = T1.affine
    utils.make_dir(op.join(MMVT_DIR, subject, 'fmri'))
    for hemi in utils.HEMIS:
        surf_fname = op.join(MMVT_DIR, fmri_subject, 'fmri', 'fmri_{}_{}.npy'.format(contrast_name, hemi))
        data = np.load(surf_fname)
        data = data.reshape((data.shape[0], 1, 1))
        output_fname = op.join(MMVT_DIR, fmri_subject, 'fmri', '{}_{}.mgz'.format(contrast_name, hemi))
        nib.save(nib.Nifti1Image(data, affine), output_fname)
        print('Data was saved to {}'.format(output_fname))


def morph_fmri(args):
    morph_from, morph_to = 'mg78', 'colin27'
    nii_template = 'non-interference-v-interference_{hemi}.mgz'
    from src.utils import freesurfer_utils as fu
    utils.make_dir(op.join(MMVT_DIR, morph_to, 'fmri'))
    for hemi in utils.HEMIS:
        fu.surf2surf(
            morph_from, morph_to, hemi, op.join(MMVT_DIR, morph_from, 'fmri', nii_template.format(hemi=hemi)),
            op.join(MMVT_DIR, morph_to, 'fmri', nii_template.format(hemi=hemi)))


def project_and_calc_clusters(args):
    if not op.isdir(args.root_fol):
        print('You should first set args.root_fol!')
        return False
    img_files = [f for f in glob.glob(op.join(args.root_fol, '*.img')) if op.isfile(f)]
    for img_fname in img_files:
        mgz_fname = fu.mri_convert_to(img_fname, 'mgz', overwrite=False)
        if ' ' in utils.namebase(mgz_fname):
            mgz_new_fname = op.join(
                utils.get_parent_fol(mgz_fname),
                utils.namebase_with_ext(mgz_fname).replace(' ', '_').replace(',', '').lower())
            os.rename(mgz_fname, mgz_new_fname)
    nii_files = [f for f in glob.glob(op.join(args.root_fol, '*'))
                 if op.isfile(f) and utils.file_type(f) in ('nii', 'nii.gz', 'mgz')]
    for fname in nii_files:
        fmri_args = fmri.read_cmd_args(dict(
            subject=args.subject,
            function='project_volume_to_surface,find_clusters',
            fmri_file_template=fname,
            threshold=args.cluster_threshold
        ))
        pu.run_on_subjects(fmri_args, fmri.main)


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
    parser.add_argument('-a', '--atlas', help='atlas name', required=False, default='aparc.DKTatlas')
    parser.add_argument('-u', '--sftp_username', help='sftp username', required=False, default='npeled')
    parser.add_argument('-d', '--sftp_domain', help='sftp domain', required=False, default='door.nmr.mgh.harvard.edu')
    parser.add_argument('--remote_subject_dir', help='remote_subjects_dir', required=False,
                        default='/space/thibault/1/users/npeled/subjects/{subject}')
    parser.add_argument('-f', '--function', help='function name', required=True)
    parser.add_argument('-r', '--root_fol', help='root folder', required=False, default='')
    parser.add_argument('--cluster_threshold', required=False, default=2, type=float)
    args = utils.Bag(au.parse_parser(parser))
    locals()[args.function](args)