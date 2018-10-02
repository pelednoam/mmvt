import matplotlib.pyplot as plt
import numpy as np

import mne
from mne import io
from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator, source_band_induced_power


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


def calc_label_src_vertices(src, label):
    vertno = [s['vertno'] for s in src]
    nvert = [len(vn) for vn in vertno]
    if label.hemi == 'lh':
        this_vertno = np.intersect1d(vertno[0], label.vertices)
        label_src_vertices = np.searchsorted(vertno[0], this_vertno)
    elif label.hemi == 'rh':
        this_vertno = np.intersect1d(vertno[1], label.vertices)
        label_src_vertices = nvert[0] + np.searchsorted(vertno[1], this_vertno)
    return label_src_vertices


def calc_morlet_cwt(epochs, inverse_operator, label, bands, inverse_method, lambda2, pick_ori, n_cycles):
    powers, time_ax = None, None
    bands_frqs = [[v[0], v[1]] for v in bands.values()]
    bands_names = bands.keys()
    label_src_indices = calc_label_src_vertices(inverse_operator['src'], label)
    stcs = mne.minimum_norm.apply_inverse_epochs(
        epochs, inverse_operator, lambda2, inverse_method, pick_ori=pick_ori, return_generator=True)
    for stc_ind, stc in enumerate(stcs):
        if powers is None:
            powers = np.empty((len(bands), len(epochs), stc.shape[1]))
            time_ax = stc.times
        for band_ind, (freqs, band_name) in enumerate(zip(bands_frqs, bands_names)):
            Ws = mne.time_frequency.morlet(
                epochs.info['sfreq'], freqs, n_cycles=n_cycles, zero_mean=False)
            this_tfr = mne.time_frequency.tfr.cwt(
                stc.data[label_src_indices], Ws, use_fft=False)
            powers[band_ind, stc_ind] = (this_tfr * this_tfr.conj()).real.mean((0, 1)) # avg over label vertices and band's freqs

    plt.figure()
    for band_ind, band_name in enumerate(bands_names):
        plt.plot(time_ax, powers[band_ind].mean(0), label=band_name)
    plt.xlabel('Time (ms)')
    plt.ylabel('Power')
    plt.legend()
    plt.title('morlet & cwt')


def calc_source_band_induced_power(epochs, inverse_operator, label, bands, n_cycles):
    # Compute a source estimate per frequency band
    stcs = source_band_induced_power(epochs, inverse_operator, bands, label=label, n_cycles=n_cycles,
                                     use_fft=False, n_jobs=1)

    plt.figure()
    plt.plot(stcs['alpha'].times, stcs['alpha'].data.mean(axis=0), label='Alpha')
    plt.plot(stcs['beta'].times, stcs['beta'].data.mean(axis=0), label='Beta')
    plt.xlabel('Time (ms)')
    plt.ylabel('Power')
    plt.legend()
    plt.title('Mean source induced power')


if __name__ == '__main__':
    snr = 3.0
    lambda2 = 1.0 / snr ** 2
    inverse_method = 'dSPM'
    pick_ori = 'normal'
    n_cycles = 2
    bands = dict(alpha=[9, 11], beta=[18, 22])

    epochs, inverse_operator, label = init_data()
    calc_morlet_cwt(epochs, inverse_operator, label, bands, inverse_method, lambda2, pick_ori, n_cycles)
    calc_source_band_induced_power(epochs, inverse_operator, label, bands, n_cycles)
    plt.show()
