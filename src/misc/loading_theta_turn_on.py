import os.path as op
import numpy as np
from src.utils import utils
from src.utils import matlab_utils as mu

MMVT_DIR = utils.get_link_dir(utils.get_links_dir(), 'mmvt')


def load_file(fname):
    d = mu.load_mat_to_bag(fname)
    channel_list = []
    for c in mu.matlab_cell_str_to_list(d.channel_list):
        elecs = ['{}{}'.format(*utils.elec_group_number(e)) for e in c.split(' ')]
        channel_list.append('{1}-{0}'.format(*elecs))
    fs = d.fs[0][0]
    time = d.T.squeeze()
    stim = np.array(d.stim, dtype=np.int)
    theta = d.Theta
    return channel_list, fs, time, stim, theta

def create_mmvt_file(subject, stim_channel, channel_list, fs, time, stim, theta, downsample_ratio=10):
    theta -= np.min(theta)
    theta /= np.max(theta)
    theta *= -1
    stim = utils.downsample(stim.squeeze(), downsample_ratio)
    stim[stim >= 0.5] = 1
    stim[stim < 0.5] = 0
    theta = utils.downsample_2d(theta.T, downsample_ratio)
    data = np.vstack((stim, theta))
    data = data[:, :, np.newaxis]
    channel_list = [stim_channel] + channel_list
    meta_data_fname = op.join(MMVT_DIR, subject, 'electrodes', 'electrodes_bipolar_data.npz')
    np.savez(meta_data_fname, data=data, names=channel_list, conditions=['on'], dt=1/fs)


if __name__ == '__main__':
    subject = 'mg118'
    stim_channel = 'LMF2-LMF1'
    root = '/autofs/space/thibault_001/users/npeled/Documents/darpa_year4_meeting'
    fname = op.join(root, 'MG118_Theta_TurnOn.mat')
    downsample_ratio = 10

    channel_list, fs, time, stim, theta = load_file(fname)
    create_mmvt_file(subject, stim_channel, channel_list, fs, time, stim, theta, downsample_ratio)