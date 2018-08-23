import os.path as op
import glob
import numpy as np
import scipy.interpolate

from src.utils import utils
from src.preproc import meg as meg
from src.preproc import electrodes

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
MMVT_DIR = op.join(LINKS_DIR, 'mmvt')


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


def calc_electrodes_power_spectrum(subject, edf_name, overwrite=False):
    elecs_args = electrodes.read_cmd_args(utils.Bag(
        subject=subject,
        function='create_raw_data_from_edf,calc_epochs_power_spectrum',
        task='rest',
        bipolar=False,
        remove_power_line_noise=True,
        raw_fname='{}.edf'.format(edf_name),
        normalize_data=False,
        preload=True,
        windows_length=10, # s
        windows_shift=5,
        # epoches_nun=1000,
        overwrite_epochs_power_spectrum=overwrite
    ))
    electrodes.call_main(elecs_args)


def combine_meg_and_electrodes_power_spectrum(subject, inv_method='MNE', em='mean_flip'):
    meg_ps_dict = utils.Bag(np.load(op.join(MMVT_DIR, subject, 'meg', 'rest_{}_{}_power_spectrum.npz'.format(
        inv_method, em))))
    elecs_ps_dict = utils.Bag(np.load(op.join(MMVT_DIR, subject, 'electrodes', 'power_spectrum.npz'.format(
        inv_method, em))))

    meg_ps = meg_ps_dict.power_spectrum.squeeze().mean(axis = 0)
    elecs_ps = elecs_ps_dict.power_spectrum.squeeze().mean(axis=0)
    meg_func = scipy.interpolate.interp1d(meg_ps_dict.frequencies, meg_ps, kind='cubic')
    elecs_func = scipy.interpolate.interp1d(elecs_ps_dict.frequencies, elecs_ps, kind='cubic')

    min_freq = int(max([min(meg_ps_dict.frequencies), min(elecs_ps_dict.frequencies)]))
    max_freq = int(min([max(meg_ps_dict.frequencies), max(elecs_ps_dict.frequencies)]))
    freqs_num = max_freq - min_freq + 1
    frequencies = np.linspace(min_freq, max_freq, num=freqs_num, endpoint=True)

    meg_ps_inter = meg_func(frequencies)
    elecs_ps_inter = elecs_func(frequencies)
    plot_results(meg_ps_dict, elecs_ps_dict, frequencies, meg_ps, meg_ps_inter, elecs_ps, elecs_ps_inter)


def plot_results(meg_ps_dict, elecs_ps_dict, frequencies, meg_ps, meg_ps_inter, elecs_ps, elecs_ps_inter):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(meg_ps_dict.frequencies, meg_ps.T)
    plt.title('Original MEG PS')
    plt.figure()
    plt.plot(frequencies, meg_ps_inter.T)
    plt.title('Interpolate MEG PS')
    plt.figure()
    plt.plot(elecs_ps_dict.frequencies, elecs_ps.T)
    plt.title('Original Electrodes PS')
    plt.figure()
    plt.plot(frequencies, elecs_ps_inter.T)
    plt.title('Interpolate Electrodes PS')
    plt.show()


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

    edf_name = 'SDohaseIIday2'

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
    elif args.function == 'calc_electrodes_power_spectrum':
        calc_electrodes_power_spectrum(args.subject, edf_name)
    elif args.function == 'combine_meg_and_electrodes_power_spectrum':
        combine_meg_and_electrodes_power_spectrum(args.subject, inv_method, em)