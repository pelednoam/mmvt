import os.path as op
from mne.filter import notch_filter
import numpy as np
import matplotlib.pyplot as plt
import glob


def run(x, fs, n_jobs):
    plt.figure()
    plt.plot(x.T)
    plt.figure()
    plt.psd(x, Fs=fs)
    x2 = notch_filter(x, fs, np.arange(60, 241, 60), notch_widths=5, filter_length='auto', copy=True, n_jobs=n_jobs)
    plt.figure()
    plt.plot(x2.T)
    plt.figure()
    plt.psd(x2, Fs=fs)
    plt.show()


if __name__ == '__main__':
    root_fol = '/homes/5/npeled/space1/mmvt/Niles/electrodes/streaming/2017-04-12-2'
    data_fname =  op.join(root_fol, 'streaming_data_12-33-10.npy')
    fs = 1000  # Hz
    data = []
    for fname in sorted(glob.glob(op.join(root_fol, '*.npy'))):
        x = np.load(fname)
        data = x if data == [] else np.hstack((data, x))
    # x = np.load(data_fname)
    run(data[0], fs, 4)
    print('asdf')