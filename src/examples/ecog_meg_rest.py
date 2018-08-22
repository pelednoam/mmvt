import os.path as op
import glob

from src.utils import utils
from src.preproc import meg as meg


def meg_rest_ana(subject, inv_method='MNE', em='mean_flip', atlas='electrodes_labels', remote_subject_dir='',
                 meg_remote_dir='', raw_fname='', empty_fname='', cor_fname='', overwrite=False, n_jobs=-1):
    meg_args = meg.read_cmd_args(dict(
        subject=subject, mri_subject=subject,
        task='rest', inverse_method=inv_method, extract_mode=em, atlas=atlas,
        remote_subject_meg_dir=meg_remote_dir,
        remote_subject_dir=remote_subject_dir,
        raw_fname=raw_fname,
        empty_fname=empty_fname,
        cor_fname=cor_fname,
        function='make_forward_solution,calc_inverse_operator,calc_stc,' + # calc_epochs,calc_evokes
                 'calc_labels_avg_per_condition,calc_labels_min_max',
        # cor_fname=cors[task].format(subject=subject),
        use_demi_events=True,
        windows_length=10000,
        windows_shift=5000,
        using_auto_reject=False,
        reject=False,
        use_empty_room_for_noise_cov=True,
        read_only_from_annot=False,
        # pick_ori='normal',
        overwrite_evoked=overwrite,
        overwrite_fwd=overwrite,
        overwrite_inv=overwrite,
        overwrite_stc=overwrite,
        overwrite_labels_data=overwrite,
        n_jobs=n_jobs
    ))
    ret = meg.call_main(meg_args)


if __name__ == '__main__':
    remote_subject_dir = '/autofs/space/megraid_clinical/MEG-MRI/seder/freesurfer/nmr01209'
    meg_remote_dir = '/autofs/space/megraid_clinical/MEG/epilepsy/subj_6213848/171127'
    raw_fname = glob.glob(op.join(meg_remote_dir, '*_??_raw.fif'))[0]
    cor_fname = op.join(remote_subject_dir, 'mri', 'T1-neuromag', 'sets', 'COR-naoro-171130.fif') # Can be found automatically
    if not op.isfile(raw_fname):
        raise Exception('No raw file was chosen!')
    empty_fname = op.join(meg_remote_dir, 'empty_room_raw.fif')
    meg_rest_ana('nmr01209', remote_subject_dir=remote_subject_dir, meg_remote_dir=meg_remote_dir,
                 raw_fname=raw_fname, empty_fname=empty_fname, cor_fname=cor_fname)