import matplotlib.pyplot as plt
import numpy as np
import time
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

import mne
from mne import io
from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator, source_band_induced_power


def timeit(func):
    def wrapper(*args, **kwargs):
        now = time.time()
        retval = func(*args, **kwargs)
        print('{} took {:.5f}s'.format(func.__name__, time.time() - now))
        return retval
    return wrapper


def init_data():
    data_path = sample.data_path()
    raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
    fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
    tmin, tmax, event_id = -0.2, 0.5, 1

    # Setup for reading the raw data
    raw = io.read_raw_fif(raw_fname)
    events = mne.find_events(raw, stim_channel='STI 014')
    inverse_operator = read_inverse_operator(fname_inv)

    # Setting the label
    label = mne.read_label(data_path + '/MEG/sample/labels/Aud-lh.label')

    include = []
    raw.info['bads'] += ['MEG 2443', 'EEG 053']  # bads + 2 more

    # picks MEG gradiometers
    picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=True,
                           stim=False, include=include, exclude='bads')

    # Load condition 1
    event_id = 1
    events = events[:10]  # take 10 events to keep the computation time low
    # Use linear detrend to reduce any edge artifacts
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        baseline=(None, 0), reject=dict(grad=4000e-13, eog=150e-6),
                        preload=True, detrend=1)
    return epochs, inverse_operator, label


@timeit
def calc_morlet_cwt(epochs, inverse_operator, label, bands, inverse_method, lambda2, pick_ori, n_cycles):
    powers, time_ax = None, None
    bands_frqs = [[v[0], v[1]] for v in bands.values()]
    bands_names = bands.keys()
    stcs = mne.minimum_norm.apply_inverse_epochs(
        epochs, inverse_operator, lambda2, inverse_method, label, pick_ori=pick_ori, return_generator=True)
    ws = [(mne.time_frequency.morlet(
        epochs.info['sfreq'], freqs, n_cycles=n_cycles, zero_mean=False)) for freqs in bands_frqs]
    for stc_ind, stc in enumerate(stcs):
        if powers is None:
            powers = np.empty((len(bands), len(epochs), stc.shape[1]))
            times = stc.times
        for band_ind in range(len(bands_names)):
            this_tfr = mne.time_frequency.tfr.cwt(
                stc.data, ws[band_ind], use_fft=False)
            powers[band_ind, stc_ind] = (this_tfr * this_tfr.conj()).real.mean((0, 1)) # avg over label vertices and band's freqs
    return powers, times


def plot_morlet_cwt_results(times, bands_power, bands_names):
    plt.figure()
    for band_ind, band_name in enumerate(bands_names):
        plt.plot(times, bands_power[band_ind].mean(0), label=band_name)
    plt.xlabel('Time (ms)')
    plt.ylabel('Power')
    plt.legend()
    plt.title('morlet & cwt')


@timeit
def calc_source_band_induced_power(epochs, inverse_operator, label, bands, n_cycles):
    # Compute a source estimate per frequency band
    return source_band_induced_power(epochs, inverse_operator, bands, label=label, n_cycles=n_cycles,
                                     use_fft=False, n_jobs=1)


def plot_source_band_induced_power(stcs):
    plt.figure()
    plt.plot(stcs['alpha'].times, stcs['alpha'].data.mean(axis=0), label='Alpha')
    plt.plot(stcs['beta'].times, stcs['beta'].data.mean(axis=0), label='Beta')
    plt.xlabel('Time (ms)')
    plt.ylabel('Power')
    plt.legend()
    plt.title('Mean source induced power')


@timeit
def calc_source_psd_epochs(epochs, inverse_operator, inverse_method, label, bands, lambda2, bandwidth):
    powers = np.empty((len(bands.keys()), len(epochs)))
    for band_ind, (fmin, fmax) in enumerate(bands.values()):
        # with warnings.catch_warnings():
        stcs = mne.minimum_norm.compute_source_psd_epochs(
            epochs, inverse_operator, lambda2=lambda2, method=inverse_method, fmin=fmin, fmax=fmax,
            bandwidth=bandwidth, label=label, return_generator=True)
        for stc_ind, stc in enumerate(stcs):
            powers[band_ind, stc_ind] = stc.data.mean()
    return powers


if __name__ == '__main__':
    snr = 3.0
    lambda2 = 1.0 / snr ** 2
    inverse_method = 'dSPM'
    pick_ori = 'normal'
    n_cycles = 2
    bandwidth = 1.
    bands = dict(alpha=[9, 11], beta=[18, 22])

    epochs, inverse_operator, label = init_data()
    powers, times = calc_morlet_cwt(epochs, inverse_operator, label, bands, inverse_method, lambda2, pick_ori, n_cycles)
    plot_morlet_cwt_results(times, powers, bands.keys())
    #
    stcs = calc_source_band_induced_power(epochs, inverse_operator, label, bands, n_cycles)
    plot_source_band_induced_power(stcs)

    psd_powers = calc_source_psd_epochs(epochs, inverse_operator, inverse_method, label, bands, lambda2, bandwidth)
    plt.show()

