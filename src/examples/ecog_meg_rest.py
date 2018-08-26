import os.path as op
import glob
import numpy as np
import scipy.interpolate

from src.utils import utils
from src.preproc import meg as meg
from src.preproc import electrodes
from src.utils import labels_utils as lu

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
MMVT_DIR = op.join(LINKS_DIR, 'mmvt')


def create_electrodes_labels(subject, bipolar=False, labels_fol_name='electrodes_labels',
        label_r=5, overwrite=False, n_jobs=-1):
    return electrodes.create_labels_around_electrodes(
        subject, bipolar, labels_fol_name, label_r, overwrite, n_jobs)


def create_atlas_coloring(subject, labels_fol_name='electrodes_labels', n_jobs=-1):
    return lu.create_atlas_coloring(subject, labels_fol_name, n_jobs)


def meg_remove_artifcats(subject, raw_fname):
    meg_args = meg.read_cmd_args(dict(
        subject=subject, mri_subject=subject,
        function='remove_artifacts',
        raw_fname=raw_fname,
        overwrite_ica=True
    ))
    return meg.call_main(meg_args)


def meg_preproc(subject, inv_method='MNE', em='mean_flip', atlas='electrodes_labels', remote_subject_dir='',
                meg_remote_dir='', empty_fname='', cor_fname='', overwrite=False, n_jobs=-1):
    meg_args = meg.read_cmd_args(dict(
        subject=subject, mri_subject=subject,
        task='rest', inverse_method=inv_method, extract_mode=em, atlas=atlas,
        remote_subject_meg_dir=meg_remote_dir,
        remote_subject_dir=remote_subject_dir,
        empty_fname=empty_fname,
        cor_fname=cor_fname,
        function='calc_epochs,calc_evokes,make_forward_solution,calc_inverse_operator',
        use_demi_events=True,
        windows_length=10000,
        windows_shift=5000,
        # power_line_notch_widths=5,
        using_auto_reject=False,
        # reject=False,
        use_empty_room_for_noise_cov=True,
        read_only_from_annot=False,
        overwrite_epochs=overwrite,
        overwrite_evoked=overwrite,
        n_jobs=n_jobs
    ))
    return meg.call_main(meg_args)


def calc_meg_power_spectrum(subject, atlas, inv_method, em, overwrite=False, n_jobs=-1):
    meg_args = meg.read_cmd_args(dict(
        subject=subject, mri_subject=subject,
        task='rest', inverse_method=inv_method, extract_mode=em, atlas=atlas,
        function='calc_labels_power_spectrum',
        max_epochs_num=100,
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
        overwrite_power_spectrum=overwrite
    ))
    electrodes.call_main(elecs_args)


def combine_meg_and_electrodes_power_spectrum(subject, inv_method='MNE', em='mean_flip', low_freq=None, high_freq=None,
                                              do_plot=True, overwrite=False):
    # https://martinos.org/mne/dev/generated/mne.time_frequency.psd_array_welch.html
    output_fname = op.join(MMVT_DIR, subject, 'electrodes', 'electrodes_data_power_spectrum_comparison.npz')
    # if op.isfile(output_fname) and not overwrite:
    #     return True

    meg_ps_dict = utils.Bag(
        np.load(op.join(MMVT_DIR, subject, 'meg', 'rest_{}_{}_power_spectrum.npz'.format(inv_method, em))))
    elecs_ps_dict = utils.Bag(
        np.load(op.join(MMVT_DIR, subject, 'electrodes', 'power_spectrum.npz'.format(inv_method, em))))
    # Power Spectral Density (dB)
    meg_ps = 10 * np.log10(meg_ps_dict.power_spectrum.squeeze().mean(axis=0))
    elecs_ps = 10 * np.log10(elecs_ps_dict.power_spectrum.squeeze().mean(axis=0))
    meg_func = scipy.interpolate.interp1d(meg_ps_dict.frequencies, meg_ps)#, kind='cubic')
    elecs_func = scipy.interpolate.interp1d(elecs_ps_dict.frequencies, elecs_ps)#, kind='cubic')

    if low_freq is None:
        low_freq = int(max([min(meg_ps_dict.frequencies), min(elecs_ps_dict.frequencies)]))
    if high_freq is None:
        high_freq = int(min([max(meg_ps_dict.frequencies), max(elecs_ps_dict.frequencies)]))
    freqs_num = high_freq - low_freq + 1
    frequencies = np.linspace(low_freq, high_freq, num=freqs_num, endpoint=True)

    meg_ps_inter = meg_func(frequencies)
    meg_ps_inter -= np.mean(meg_ps_inter)
    elecs_ps_inter = elecs_func(frequencies)
    elecs_ps_inter -= np.mean(elecs_ps_inter)
    if do_plot:
        plot_results(meg_ps_dict, elecs_ps_dict, frequencies, meg_ps, meg_ps_inter, elecs_ps, elecs_ps_inter)

    electrodes_meta_fname = op.join(MMVT_DIR, subject, 'electrodes', 'electrodes_meta_data.npz')
    elecs_dict = utils.Bag(np.load(electrodes_meta_fname))
    labels = elecs_dict.names

    data = np.zeros((len(labels), len(frequencies), 2))
    data[:, :, 0] = elecs_ps_inter
    data[:, :, 1] = meg_ps_inter
    np.savez(output_fname, data=data, names=labels, conditions=['grid_rest', 'meg_rest'])


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
    args.n_jobs = utils.get_n_jobs(args.n_jobs)

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
    overwrite_meg, overwrite_electrodes_labels = True, False
    overwrite_labels_power_spectrum, overwrite_power_spectrum = True, True
    bipolar = False
    labels_fol_name = atlas = 'electrodes_labels'
    label_r = 5

    edf_name = 'SDohaseIIday2'
    low_freq, high_freq = 1, 40

    if args.function == 'create_electrodes_labels':
        create_electrodes_labels(
            args.subject, bipolar, labels_fol_name, label_r, overwrite_electrodes_labels, args.n_jobs)
    elif args.function == 'create_atlas_coloring':
        create_atlas_coloring(args.subject, labels_fol_name, args.n_jobs)
    elif args.function == 'meg_remove_artifcats':
        meg_remove_artifcats(args.subject, raw_fname)
    elif args.function == 'meg_preproc':
        meg_preproc(
            args.subject, inv_method, em, atlas, remote_subject_dir, meg_remote_dir, empty_fname,
            cor_fname, overwrite_meg, args.n_jobs)
    elif args.function == 'calc_meg_power_spectrum':
        calc_meg_power_spectrum(
            args.subject, atlas, inv_method, em, overwrite_labels_power_spectrum, args.n_jobs)
    elif args.function == 'calc_electrodes_power_spectrum':
        calc_electrodes_power_spectrum(args.subject, edf_name, overwrite_power_spectrum)
    elif args.function == 'combine_meg_and_electrodes_power_spectrum':
        combine_meg_and_electrodes_power_spectrum(args.subject, inv_method, em, low_freq, high_freq)