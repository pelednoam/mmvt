import mne
import numpy as np
import os.path as op

# Noam's healthy control subject
subjects_dir = '/autofs/cluster/fusion/data/john-noam/'
subject = 'hc029'
bem_fname = '/autofs/cluster/fusion/data/john-noam/hc029/bem/hc029-5120-5120-5120-bem-sol.fif'
trans_fname = '/autofs/cluster/fusion/data/john-noam/hc029/mri/T1-neuromag/sets/COR.fif'

bem = mne.read_bem_solution(bem_fname)

src = mne.setup_source_space(subject, spacing='ico5',
                             subjects_dir=subjects_dir,
                             add_dist=False)

raw_fname = '/autofs/cluster/fusion/data/john-noam/hc029/MEG/hc029_tsss-rest-raw.fif'

fwd = mne.make_forward_solution(info=raw_fname, trans=trans_fname, src=src, bem=bem, meg=True, eeg=False, n_jobs=2)

cov = '/autofs/cluster/fusion/data/john-noam/hc029/MEG/hc029_tsss-rest-noise-cov.fif'

labels_dir = '/autofs/cluster/fusion/data/john-noam/hc029/label/laus500/'

cort_labels_fname = 'superiorfrontal_22-lh.label'
deep_labels_fname = 'caudalanteriorcingulate_2-lh.label'

cort_labels = mne.read_label(op.join(labels_dir, cort_labels_fname))
deep_labels = mne.read_label(op.join(labels_dir, deep_labels_fname))

times = np.linspace(0, 1, 1500)
fd = 211
fc = 37


def data_cort(times):
    return 1e-7 * np.cos(times * 2 * np.pi * fc)


def data_deep(times):
    return 1e-7 * np.cos(times * 2 * np.pi * fd)  # np.ones(len(times))


stc_cort = mne.simulation.simulate_sparse_stc(fwd['src'], n_dipoles=1, times=times,
                               random_state=42, labels=cort_labels, data_fun=data_cort)  # cortical label