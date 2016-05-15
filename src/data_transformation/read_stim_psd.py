import scipy.io as sio
import matplotlib.pyplot as plt
import os.path as op
import numpy as np
from src.utils import utils

LINKS_DIR = utils.get_links_dir()
BLENDER_ROOT_DIR = op.join(LINKS_DIR, 'mmvt')


def read_psd_mat_file(subject, psd_fname, stim_channel):
    x = utils.read_mat_file_into_bag(psd_fname)
    labels = get_labels(x)
    data = x.psd if 'psd' in x else x.Psd
    if 'time' in x:
        time = x.time.reshape((x.time.shape[1]))
    else:
        time = None
    freqs = [(0, 4), (4, 8), (8, 15), (15, 30), (30, 55), (65, 100)]
    # plt.plot(time, psd[0, 0, :])
    # plt.show()
    out_fol = op.join(BLENDER_ROOT_DIR, subject, 'electrodes')
    utils.make_dir(out_fol)
    np.savez(op.join(out_fol, 'psd_{}'.format(stim_channel)), labels=labels, psd=data, time=time, freqs=freqs)


def read_new_psd_mat_file(subject, psd_fname, stim_channel, labels):
    x = utils.read_mat_file_into_bag(psd_fname)
    psd = None
    freqs = [(0, 4), (4, 8), (8, 15), (15, 30), (30, 55), (65, 100)]
    F = len(freqs)
    for ind, field in enumerate(['P{}'.format(ind) for ind in range(1, F + 1)]):
        if psd is None:
            T, L = x[field].shape
            psd = np.zeros((F, T, L))
        psd[ind, :, :] = x[field]
    del x
    time = range(psd.shape[1])
    # plt.plot(time, psd[0, 0, :])
    # plt.show()
    out_fol = op.join(BLENDER_ROOT_DIR, subject, 'electrodes')
    utils.make_dir(out_fol)
    np.savez(op.join(out_fol, 'psd_{}'.format(stim_channel)), labels=labels, psd=psd, time=time, freqs=freqs)


def get_labels(x=None, mat_fname=''):
    if mat_fname != '':
        x = utils.read_mat_file_into_bag(mat_fname)
    if x is None:
        raise Exception('x is None!')
    labels = x['Label']
    labels = labels.reshape((len(labels)))
    fix_labels = []
    for label in labels:
        label = label[0]
        elecs = label.split('-')
        group, elc1_name = utils.elec_group_number(elecs[0], False)
        _, elc2_name = utils.elec_group_number(elecs[1], False)
        fix_labels.append('{0}{2}-{0}{1}'.format(group, elc1_name, elc2_name))
    return sorted(fix_labels, key=utils.natural_keys)


if __name__ == '__main__':
    subject = 'mg99'
    # root_fol = '/cluster/neuromind/npeled/Ishita/'
    root_fol = op.join(BLENDER_ROOT_DIR, subject, 'electrodes', 'mat_files')
    # psd_fnames = ['MG99_Philbert_stim_LVF34_2mA.mat', 'MG99_Philbert_stim_LVF56_2mA.mat']
    psd_fnames = ['MG99_Psd_stim_LVF34_2mA.mat', 'MG99_Psd_stim_LVF56_2mA.mat']
    labels_fnames = ['MG99_Psd_stim_LVF34_2mA.mat', 'MG99_Psd_stim_LVF56_2mA.mat']
    stim_channels = ['LVF4-LVF3', 'LVF6-LVF5']
    for psd_fname, labels_fname, stim_channel in zip(psd_fnames, labels_fnames, stim_channels):
        # labels = get_labels(op.join(root_fol, labels_fname))
        read_psd_mat_file(subject, op.join(root_fol, psd_fname), stim_channel)