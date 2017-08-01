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
from src.preproc import connectivity as con


hesheng_root = '/autofs/cluster/scratch/tuesday/noam/DataProcessed_memory/{subject}'
hesheng_surf_fol = op.join(hesheng_root, 'surf')
hesheng_vol_fol = op.join(hesheng_root, 'bold', '*')
hesheng_vol_template = '{subject}_bld???_rest_reorient_skip_faln_mc_g1000000000_bpss_resid.nii.gz'
hesheng_template = '?h.{subject}_bld???_rest_reorient_skip_faln_mc_g1000000000_bpss_resid_fsaverage6_sm6_fsaverage5.nii.gz'

linda_fol = '/autofs/cluster/neuromind/douw/scans/adults/{subject}/bold/*/'
linda_surf_fol = op.join(fmri.FMRI_DIR, '{subject}')
linda_vol_template = '{subject}_bld???_rest_reorient_skip_faln_mc_g1000000000_bpss_resid.nii.gz'
linda_hemi_template = '{}_{}.mgz'.format(linda_vol_template[:-len('.nii.gz')], '{hemi}')
linda_template_npy = '{subject}_bld???_rest_reorient_skip_faln_mc_g1000000000_bpss_resid*.npy'

fs_surf_fol = linda_surf_fol
fs_surf_template = 'rest_linda.sm6.{subject}.{hemi}.mgz'

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


def calc_hesheng_surf(subject, atlas):
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
            subject=subject, atlas=atlas, function='load_surf_files', overwrite_surf_data=True,
            fmri_file_template='hesheng_{hemi}.mgz'))
        pu.run_on_subjects(args, fmri.main)


def calc_linda_surf(subject, atlas):
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
            fmri_file_template=linda_vol_template.format(subject=subject), overwrite_surf_data=True))
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
            utils.namebase(linda_vol_template.format(subject=subject)))))
        for mgz_file in mgz_files:
            os.remove(mgz_file)


def calc_freesurfer_surf(subject, atlas):
    # Clean
    args = fmri.read_cmd_args(dict(
        subject=subject, atlas=atlas, function='clean_4d_data', fmri_file_template='rest.nii*', fsd='rest_linda',
        overwrite_4d_preproc=False))
    pu.run_on_subjects(args, fmri.main)
    # save the surf files
    args = fmri.read_cmd_args(dict(
        subject=subject, atlas=atlas, function='load_surf_files', overwrite_surf_data=True,
        fmri_file_template=fs_surf_template.format(subject=subject, hemi='{hemi}')))
    pu.run_on_subjects(args, fmri.main)
    # Renaming the files
    root_fol = op.join(fmri.MMVT_DIR, subject, 'fmri')
    for hemi in utils.HEMIS:
        os.rename(op.join(root_fol, 'fmri_rest_linda.sm6.{}.{}.npy'.format(subject, hemi)),
                  op.join(root_fol, 'fmri_freesurfer_{}.npy'.format(hemi)))
    os.rename(op.join(root_fol, 'rest_linda.sm6.{}_minmax.pkl'.format(subject)),
              op.join(root_fol, 'freesurfer_minmax.pkl'))


def calc_diff(subject, fmri_file_template='*linda_{hemi}*,*hesheng_{hemi}'):
    # Calc diff
    args = fmri.read_cmd_args(dict(
        subject=subject, function='calc_files_diff', fmri_file_template=fmri_file_template))
    pu.run_on_subjects(args, fmri.main)


def compare_connectivity(subject, atlas, n_jobs=6):
    for name, fol, template in zip(['hesheng', 'linda', 'freesurfer'],
                                   [hesheng_surf_fol, linda_surf_fol, fs_surf_fol],
                                   [hesheng_template, linda_hemi_template, fs_surf_template]):
        output_fname_template = op.join(
            fmri.MMVT_DIR, subject, 'fmri', '{}_labels_data_laus125_mean_{}.npz'.format(name, '{hemi}'))
        if not utils.both_hemi_files_exist(output_fname_template):
            args = fmri.read_cmd_args(dict(
                subject=subject, atlas=atlas, function='analyze_4d_data', fmri_file_template=template, remote_fmri_dir=fol,
                labels_extract_mode='mean', overwrite_labels_data=False))
            pu.run_on_subjects(args, fmri.main)
            for hemi in utils.HEMIS:
                os.rename(op.join(fmri.MMVT_DIR, subject, 'fmri', 'labels_data_laus125_mean_{}.npz'.format(hemi)),
                          output_fname_template.format(hemi=hemi))

        args = con.read_cmd_args(dict(
            subject=subject, atlas='laus125', function='calc_lables_connectivity',
            connectivity_modality='fmri', connectivity_method='corr,cv', labels_extract_mode='mean',
            windows_length=34, windows_shift=4, save_mmvt_connectivity=False, calc_subs_connectivity=False,
            labels_name=name, recalc_connectivity=True, n_jobs=n_jobs))
        pu.run_on_subjects(args, con.main)
        conn_fol = op.join(con.MMVT_DIR, subject, 'connectivity')
        coloring_fol = op.join(con.MMVT_DIR, subject, 'coloring')
        os.rename(op.join(conn_fol, 'fmri_corr.npy'), op.join(conn_fol, 'fmri_corr_{}.npy'.format(name)))
        os.rename(op.join(conn_fol, 'fmri_corr_cv_mean.npz'), op.join(conn_fol, 'mri_corr_cv_mean_{}.npz'.format(name)))
        os.rename(op.join(conn_fol, 'fmri_corr_cv_mean_mean.npz'), op.join(conn_fol, 'fmri_corr_cv_mean_mean_{}.npz'.format(name)))
        os.rename(op.join(coloring_fol, 'fmri_corr_cv_mean.csv'), op.join(coloring_fol, 'fmri_corr_cv_mean_{}.csv'.format(name)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=True)
    parser.add_argument('-a', '--atlas', help='atlas', required=False, default='laus125')
    parser.add_argument('--n_jobs', help='n_jobs', required=False, default=6, type=int)
    args = utils.Bag(au.parse_parser(parser))

    # nmr00502,nmr00515,nmr00603,nmr00609,nmr00629,nmr00650,nmr00657,nmr00669,nmr00674,nmr00681,nmr00683,nmr00692,nmr00698,nmr00710
    # check_original_files_dim(args.subject)
    # calc_hesheng_surf(args.subject, args.atlas)
    # calc_linda_surf(args.subject, args.atlas)
    # calc_freesurfer_surf(args.subject, args.atlas)
    # calc_diff(args.subject)
    # calc_diff(args.subject, fmri_file_template='*fmri_freesurfer_{hemi}*,*fmri_hesheng_{hemi}')
    compare_connectivity(args.subject, args.atlas, args.n_jobs)