import scipy.io as sio
import matplotlib.pyplot as plt
import os.path as op
import numpy as np
from src.utils import utils

LINKS_DIR = utils.get_links_dir()
BLENDER_ROOT_DIR = op.join(LINKS_DIR, 'mmvt')


def read_psd_mat_file(subject, psd_fname, stim_channel):
    x = sio.loadmat(psd_fname)
    labels = get_labels(x)
    psd = x['Psd']
    time = x['Tpsd']
    del x
    time = time.reshape((time.shape[1]))
    freqs = [(0, 4), (4, 8), (8, 15), (15, 30), (30, 55), (65, 100)]
    # plt.plot(time, psd[0, 0, :])
    # plt.show()
    out_fol = op.join(BLENDER_ROOT_DIR, subject, 'stim')
    utils.make_dir(out_fol)
    np.savez(op.join(out_fol, 'psd_{}'.format(stim_channel)), labels=labels, psd=psd, time=time, freqs=freqs)


def get_labels(x):
    labels = x['Label']
    labels = labels.reshape((len(labels)))
    fix_labels = []
    for label in labels:
        label = label[0]
        elecs = label.split('-')
        group, elc1_name = utils.elec_group_number(elecs[0], False)
        _, elc2_name = utils.elec_group_number(elecs[1], False)
        fix_labels.append('{0}{2}-{0}{1}'.format(group, elc1_name, elc2_name))
    return fix_labels


if __name__ == '__main__':
    root_fol = '/cluster/neuromind/npeled/Ishita/'
    psd_fnames = ['MG99_Psd_stim_LVF34_2mA.mat', 'MG99_Psd_stim_LVF56_2mA.mat']
    stim_channels = ['LVF4-LVF3', 'LVF6-LVF5']
    subject = 'mg99'
    for psd_fname, stim_channel in zip(psd_fnames, stim_channels):
        read_psd_mat_file(subject, op.join(root_fol, psd_fname), stim_channel)