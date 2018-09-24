import os.path as op
import glob
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import warnings

from src.utils import utils
from src.preproc import meg as meg
from src.preproc import electrodes
from src.utils import labels_utils as lu

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
MMVT_DIR = op.join(LINKS_DIR, 'mmvt')
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')


def create_electrodes_labels(subject, bipolar=False, labels_fol_name='electrodes_labels',
        label_r=5, snap=False, sigma=1, overwrite=False, n_jobs=-1):
    return electrodes.create_labels_around_electrodes(
        subject, bipolar, labels_fol_name, label_r, snap, sigma, overwrite, n_jobs)


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
                meg_remote_dir='', empty_fname='', cor_fname='', use_demi_events=True, calc_labels_avg=False,
                overwrite=False, n_jobs=-1):
    functions = 'calc_epochs,calc_evokes,make_forward_solution,calc_inverse_operator'
    if calc_labels_avg:
        functions += ',calc_stc,calc_labels_avg_per_condition'
    meg_args = meg.read_cmd_args(dict(
        subject=subject, mri_subject=subject,
        task='rest', inverse_method=inv_method, extract_mode=em, atlas=atlas,
        remote_subject_meg_dir=meg_remote_dir,
        remote_subject_dir=remote_subject_dir,
        empty_fname=empty_fname,
        cor_fname=cor_fname,
        function=functions,
        use_demi_events=use_demi_events,
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
        pick_ori='normal',  # very important for calculation of the power spectrum
        # max_epochs_num=20,
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
        # epochs_num=20,
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
    meg_ps = 10 * np.log10(meg_ps_dict.power_spectrum.squeeze())
    mask = np.where(meg_ps_dict.frequencies > 8)[0]
    np.argmax(np.sum(meg_ps[:, :, mask], axis=(1, 2)))

    plot_power_spectrum(meg_ps, meg_ps_dict.frequencies, 'MEG')
    meg_ps = meg_ps.mean(axis=0)
    elecs_ps = 10 * np.log10(elecs_ps_dict.power_spectrum.squeeze())
    plot_power_spectrum(elecs_ps, elecs_ps_dict.frequencies, 'electrodes')
    elecs_ps = elecs_ps.mean(axis=0)
    meg_func = scipy.interpolate.interp1d(meg_ps_dict.frequencies, meg_ps)
    elecs_func = scipy.interpolate.interp1d(elecs_ps_dict.frequencies, elecs_ps)

    low_freq = int(max([min(meg_ps_dict.frequencies), min(elecs_ps_dict.frequencies), low_freq]))
    high_freq = int(min([max(meg_ps_dict.frequencies), max(elecs_ps_dict.frequencies), high_freq]))
    freqs_num = high_freq - low_freq + 1
    frequencies = np.linspace(low_freq, high_freq, num=freqs_num * 10, endpoint=True)

    meg_ps_inter = meg_func(frequencies)
    meg_ps_inter = (meg_ps_inter - np.mean(meg_ps_inter)) / np.std(meg_ps_inter)
    elecs_ps_inter = elecs_func(frequencies)
    elecs_ps_inter = (elecs_ps_inter - np.mean(elecs_ps_inter)) / np.std(elecs_ps_inter)

    plot_all_results(meg_ps_inter, elecs_ps_inter, frequencies)

    electrodes_meta_fname = op.join(MMVT_DIR, subject, 'electrodes', 'electrodes_meta_data.npz')
    elecs_dict = utils.Bag(np.load(electrodes_meta_fname))
    labels = elecs_dict.names

    data = np.zeros((len(labels), len(frequencies), 2))
    data[:, :, 0] = elecs_ps_inter
    data[:, :, 1] = meg_ps_inter
    np.savez(output_fname, data=data, names=labels, conditions=['grid_rest', 'meg_rest'])

    if do_plot:
        plot_results(meg_ps_dict, elecs_ps_dict, frequencies, meg_ps, meg_ps_inter, elecs_ps, elecs_ps_inter)


def plot_power_spectrum(psds, freqs, title):
    f, ax = plt.subplots()
    psds_mean = psds.mean(0)
    psds_std = psds.std(0)
    for ps_mean, ps_std in zip(psds_mean, psds_std):
        ax.plot(freqs, ps_mean, color='k')
        ax.fill_between(freqs, ps_mean - ps_std, ps_mean + ps_std, color='k', alpha=.5)
    ax.set(title='{} Multitaper PSD'.format(title), xlabel='Frequency',
           ylabel='Power Spectral Density (dB)')
    plt.show()


def plot_all_results(meg_ps_inter, elecs_ps_inter, frequencies):
    fig, ax = plt.subplots(nrows=8, ncols=8, sharex='col', sharey='row')
    ind = 0
    for row in ax:
        for col in row:
            col.plot(frequencies, meg_ps_inter[ind], 'b')
            col.plot(frequencies, elecs_ps_inter[ind], 'r')
            col.set_xlim([0, 60])
            ind += 1
    plt.show()


def plot_results(meg_ps_dict, elecs_ps_dict, frequencies, meg_ps, meg_ps_inter, elecs_ps, elecs_ps_inter):
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


def compare_ps_from_epochs_and_from_time_series(subject):
    ps1 = np.load(op.join(MMVT_DIR, subject, 'meg', 'rest_dSPM_mean_flip_power_spectrum_from_epochs.npz'))['power_spectrum'].mean(axis=0).squeeze()
    ps2 = 10 * np.log10(np.load(op.join(MMVT_DIR, subject, 'meg', 'rest_dSPM_mean_flip_power_spectrum.npz'))['power_spectrum'].mean(axis=0).squeeze())
    plt.figure()
    plt.plot(ps1.T)
    plt.title('power spectrum from epochs')
    plt.xlim([0, 100])
    plt.figure()
    plt.plot(ps2.T)
    plt.title('power spectrum from time series')
    plt.xlim([0, 100])
    plt.show()


def check_mmvt_file(subject):
    input_fname = op.join(MMVT_DIR, subject, 'electrodes', 'electrodes_data_power_spectrum_comparison.npz')
    d = utils.Bag(np.load(input_fname))
    plt.figure()
    plt.plot(d.data[:, :, 0].T)
    plt.title(d.conditions[0])
    plt.figure()
    plt.plot(d.data[:, :, 1].T)
    plt.title(d.conditions[1])
    plt.show()



def main(args):
    remote_subject_dir = [d for d in [
        '/autofs/space/megraid_clinical/MEG-MRI/seder/freesurfer/{}'.format(args.subject),
        '/home/npeled/subjects/{}'.format(args.subject),
        op.join(SUBJECTS_DIR, args.subject)] if op.isdir(d)][0]
    meg_remote_dir = [d for d in [
        '/autofs/space/megraid_clinical/MEG/epilepsy/subj_6213848/171127',
        '/home/npeled/meg/{}'.format(args.subject),
        op.join(MEG_DIR, args.subject)] if op.isdir(d)][0]
    raw_fnames = glob.glob(op.join(meg_remote_dir, '*_??_raw.fif'))
    raw_fname =  '' #utils.select_one_file(raw_fnames) # raw_fnames[0] if len(raw_fnames) > 0 else ''
    cor_fname = '' #op.join(remote_subject_dir, 'mri', 'T1-neuromag', 'sets', 'COR-naoro-171130.fif') # Can be found automatically
    empty_fname = op.join(meg_remote_dir, 'empty_room_raw.fif')
    inv_method = 'dSPM' # 'MNE'
    em = 'mean_flip'
    overwrite_meg, overwrite_electrodes_labels = True, False
    overwrite_labels_power_spectrum, overwrite_power_spectrum = True, True
    bipolar = False
    labels_fol_name = atlas = 'electrodes_labels'
    label_r = 5
    snap = True
    sigma = 3
    use_demi_events = False
    calc_labels_avg = True

    edf_name = 'SDohaseIIday2'
    low_freq, high_freq = 1, 100

    if args.function == 'create_electrodes_labels':
        create_electrodes_labels(
            args.subject, bipolar, labels_fol_name, label_r, snap, sigma, overwrite_electrodes_labels, args.n_jobs)
    elif args.function == 'create_atlas_coloring':
        create_atlas_coloring(args.subject, labels_fol_name, args.n_jobs)
    elif args.function == 'meg_remove_artifcats':
        meg_remove_artifcats(args.subject, raw_fname)
    elif args.function == 'meg_preproc':
        meg_preproc(
            args.subject, inv_method, em, atlas, remote_subject_dir, meg_remote_dir, empty_fname,
            cor_fname, use_demi_events, calc_labels_avg, overwrite_meg, args.n_jobs)
    elif args.function == 'calc_meg_power_spectrum':
        calc_meg_power_spectrum(
            args.subject, atlas, inv_method, em, overwrite_labels_power_spectrum, args.n_jobs)
    elif args.function == 'calc_electrodes_power_spectrum':
        calc_electrodes_power_spectrum(args.subject, edf_name, overwrite_power_spectrum)
    elif args.function == 'combine_meg_and_electrodes_power_spectrum':
        combine_meg_and_electrodes_power_spectrum(args.subject, inv_method, em, low_freq, high_freq)
    elif args.function == 'check_mmvt_file':
        check_mmvt_file(args.subject)
    elif args.function == 'compare_ps_from_epochs_and_from_time_series':
        compare_ps_from_epochs_and_from_time_series(args.subject)


if __name__ == '__main__':
    import argparse
    from src.utils import args_utils as au
    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=False, default='')
    parser.add_argument('-f', '--function', help='function name', required=False, default='')
    parser.add_argument('--n_jobs', help='cpu num', required=False, default=-1)
    args = utils.Bag(au.parse_parser(parser))
    args.n_jobs = utils.get_n_jobs(args.n_jobs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main(args)
