import os.path as op
import glob

from src.utils import utils
from src.preproc import meg as meg
from src.preproc import electrodes


def create_electrodes_labels(subject, bipolar=False, labels_fol_name='electrodes_labels',
        label_r=5, overwrite=False, n_jobs=-1):
    return electrodes.create_labels_around_electrodes(
        subject, bipolar, labels_fol_name, label_r, overwrite, n_jobs)


def meg_preproc(subject, inv_method='MNE', em='mean_flip', atlas='electrodes_labels', remote_subject_dir='',
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
        use_demi_events=True,
        windows_length=10000,
        windows_shift=5000,
        using_auto_reject=False,
        reject=False,
        use_empty_room_for_noise_cov=True,
        read_only_from_annot=False,
        overwrite_evoked=overwrite,
        overwrite_fwd=overwrite,
        overwrite_inv=overwrite,
        overwrite_stc=overwrite,
        overwrite_labels_data=overwrite,
        n_jobs=n_jobs
    ))
    return meg.call_main(meg_args)


def calc_electrodes_labels_power_spectrum(subject, atlas, inv_method, em, overwrite=False, n_jobs=-1):
    meg_args = meg.read_cmd_args(dict(
        subject=subject, mri_subject=subject,
        task='rest', inverse_method=inv_method, extract_mode=em, atlas=atlas,
        function='calc_labels_power_spectrum',
        overwrite_labels_power_spectrum=overwrite,
        n_jobs=n_jobs
    ))
    return meg.call_main(meg_args)


if __name__ == '__main__':
    import argparse
    from src.utils import args_utils as au
    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=False, default='nmr01209')
    parser.add_argument('-f', '--function', help='function name', required=False, default='meg_preproc')
    parser.add_argument('--n_jobs', help='cpu num', required=False, default=-1)
    args = utils.Bag(au.parse_parser(parser))

    remote_subject_dir = [d for d in [
        '/autofs/space/megraid_clinical/MEG-MRI/seder/freesurfer/{}'.format(args.subject),
        '/home/npeled/subjects/{}'.format(args.subject)] if op.isdir(d)][0]
    meg_remote_dir = [d for d in [
        '/autofs/space/megraid_clinical/MEG/epilepsy/subj_6213848/171127',
        '/home/npeled/meg/{}'.format(args.subject)] if op.isdir(d)][0]
    raw_fnames = glob.glob(op.join(meg_remote_dir, '*_??_raw.fif'))
    raw_fname = raw_fnames[0] if len(raw_fnames) > 0 else ''
    cor_fname = op.join(remote_subject_dir, 'mri', 'T1-neuromag', 'sets', 'COR-naoro-171130.fif') # Can be found automatically
    empty_fname = op.join(meg_remote_dir, 'empty_room_raw.fif')
    inv_method, em = 'MNE', 'mean_flip'
    overwrite_meg, overwrite_electrodes_labels, overwrite_labels_power_spectrum = False, False, False

    bipolar = False
    labels_fol_name = atlas = 'electrodes_labels'
    label_r = 5

    if args.function == 'create_electrodes_labels':
        create_electrodes_labels(
            args.subject, bipolar, labels_fol_name, label_r, overwrite_electrodes_labels, args.n_jobs)
    elif args.function == 'meg_preproc':
        meg_preproc(
            args.subject, inv_method, em, atlas, remote_subject_dir, meg_remote_dir,raw_fname, empty_fname,
            cor_fname, overwrite_meg, args.n_jobs)
    elif args.function == 'calc_electrodes_labels_power_spectrum':
        calc_electrodes_labels_power_spectrum(
            args.subject, atlas, inv_method, em, overwrite_labels_power_spectrum, args.n_jobs)