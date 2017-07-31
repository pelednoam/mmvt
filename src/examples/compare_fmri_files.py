import os
import os.path as op
import glob
import shutil
import nibabel as nib

from src.utils import utils
from src.utils import labels_utils as lu
from src.utils import freesurfer_utils as fu
from src.utils import preproc_utils as pu
from src.utils import args_utils as au
from src.preproc import fMRI as fmri

hesheng_surf_fol = '/autofs/cluster/scratch/tuesday/noam/DataProcessed_memory/{subject}/surf/'
hesheng_vol_fol = '/autofs/cluster/scratch/tuesday/noam/DataProcessed_memory/{subject}/bold/*/'
hesheng_vol_template = '{subject}_bld???_rest_reorient_skip_faln_mc_g1000000000_bpss_resid.nii.gz'
hesheng_template = '?h.{subject}_bld???_rest_reorient_skip_faln_mc_g1000000000_bpss_resid_fsaverage6_sm6_fsaverage5.nii.gz'
linda_fol = '/autofs/cluster/neuromind/douw/scans/adults/{subject}/bold/*/'
linda_vol_template = '{subject}_bld???_rest_reorient_skip_faln_mc_g1000000000_bpss_resid.nii.gz'
linda_template = '{subject}_bld???_rest_reorient_skip_faln_mc_g1000000000_bpss_resid.nii.gz'
linda_template_npy = '{subject}_bld???_rest_reorient_skip_faln_mc_g1000000000_bpss_resid*.npy'


def check_original_files_dim(subject):
    hesheng_vol_fnames = glob.glob(op.join(
        hesheng_vol_fol.format(subject=subject), hesheng_vol_template.format(subject=subject)))
    hesheng_org_vol_fnames = glob.glob(op.join(
        hesheng_vol_fol.format(subject=subject), '{subject}_bld???_rest_reorient*.nii.gz'.format(subject=subject)))
    for fname in hesheng_vol_fnames + hesheng_org_vol_fnames:
        print(fname, nib.load(fname).get_data().shape)
    hesheng_fnames = glob.glob(op.join(
        hesheng_surf_fol.format(subject=subject), hesheng_template.format(subject=subject)))
    for fname in hesheng_fnames:
        print(fname, nib.load(fname).get_data().shape)
    linda_volume_fnames = glob.glob(op.join(
        linda_fol.format(subject=subject), linda_vol_template.format(subject=subject)))
    for fname in linda_volume_fnames:
        print(fname, nib.load(fname).get_data().shape)


def compare_linda_and_hesheng_files(subject):
    # nmr00502,nmr00515,nmr00603,nmr00609,nmr00629,nmr00650,nmr00657,nmr00669,nmr00674,nmr00681,nmr00683,nmr00692,nmr00698,nmr00710
    subject_fol = op.join(fmri.MMVT_DIR, subject, 'fmri')
    if not (utils.both_hemi_files_exist(op.join(subject_fol, 'fmri_hesheng_{hemi}.npy')) and
            op.isfile(op.join(subject_fol, 'hesheng_minmax.pkl'))):
        # Copy and rename Hesheng's files
        hesheng_fnames = glob.glob(op.join(
            hesheng_surf_fol.format(subject=subject), hesheng_template.format(subject=subject)))
        for fname in hesheng_fnames:
            hemi = lu.get_label_hemi_invariant_name(utils.namebase(fname))
            target_file = op.join(fmri.FMRI_DIR, subject, 'hesheng_{}.nii.gz'.format(hemi))
            mgz_target_file = utils.change_fname_extension(target_file, 'mgz')
            if not op.isfile(mgz_target_file):
                shutil.copy(fname, target_file)
                fu.nii_gz_to_mgz(target_file)
                os.remove(target_file)
        # Load Hesheng's files
        args = fmri.read_cmd_args(dict(
            subject=subject, atlas='laus125', function='load_surf_files', overwrite_surf_data=True,
            fmri_file_template='hesheng_{hemi}.mgz'))
        pu.run_on_subjects(args, fmri.main)
    # Check for Linda's output fname
    if not utils.both_hemi_files_exist(op.join(fmri.MMVT_DIR, subject, 'fmri', 'fmri_linda_{}.npy'.format('{hemi}'))) \
            and not op.isfile(op.join(fmri.MMVT_DIR, subject, 'fmri', 'linda_minmax.pkl')):
        # Find Linda's files
        linda_volume_fnames = glob.glob(op.join(
            linda_fol.format(subject=subject), linda_vol_template.format(subject=subject)))
        linda_volume_folder = utils.get_parent_fol(linda_volume_fnames[0])
        # project linda files on the surface
        args = fmri.read_cmd_args(dict(
            subject=subject, function='project_volume_to_surface', remote_fmri_dir=linda_volume_folder,
            fmri_file_template=linda_vol_template.format(subject=subject)))
        pu.run_on_subjects(args, fmri.main)
        # rename Linda's files
        linda_fnames = glob.glob(op.join(fmri.MMVT_DIR, subject, 'fmri', 'fmri_{}'.format(
            linda_template_npy.format(subject=subject))))
        for fname in linda_fnames:
            hemi = lu.get_label_hemi(utils.namebase(fname))
            target_file = op.join(fmri.MMVT_DIR, subject, 'fmri', 'fmri_linda_{}.npy'.format(hemi))
            if not op.isfile(target_file):
                os.rename(fname, target_file)
        # rename minmax file
        linda_minmax_name = '{}.pkl'.format(utils.namebase(glob.glob(op.join(
            fmri.MMVT_DIR, subject, 'fmri', '{}_minmax.pkl'.format(
                utils.namebase(linda_vol_template.format(subject=subject)))))[0]))
        os.rename(op.join(fmri.MMVT_DIR, subject, 'fmri', linda_minmax_name),
                  op.join(fmri.MMVT_DIR, subject, 'fmri', 'linda_minmax.pkl'))
        # delete mgz files
        mgz_files = glob.glob(op.join(fmri.MMVT_DIR, subject, 'fmri', 'fmri_{}_?h.mgz'.format(
            utils.namebase(linda_template.format(subject=subject)))))
        for mgz_file in mgz_files:
            os.remove(mgz_file)
    # Calc diff
    args = fmri.read_cmd_args(dict(
        subject=subject, function='calc_files_diff', fmri_file_template='*linda_{hemi}*,*hesheng_{hemi}'))
    pu.run_on_subjects(args, fmri.main)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=True)
    args = utils.Bag(au.parse_parser(parser))
    # check_original_files_dim(args.subject)
    compare_linda_and_hesheng_files(args.subject)