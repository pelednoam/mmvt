import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from src.utils import utils
from src.utils import matlab_utils as mu


MMVT_DIR = utils.get_link_dir(utils.get_links_dir(), 'mmvt')


def load_file(fname):
    d = mu.load_mat_to_bag(fname)
    channel_list = []
    for c in mu.matlab_cell_str_to_list(d.channel_list):
        elecs = ['{}{}'.format(*utils.elec_group_number(e)) for e in c.split(' ')]
        channel_list.append('{1}-{0}'.format(*elecs))
        # channel_list.extend(elecs)
    fs = d.fs[0][0]
    time = d.T.squeeze()
    stim = np.array(d.stim, dtype=np.int)
    theta = d.Theta
    return channel_list, fs, time, stim, theta


def create_mmvt_file(subject, stim_channel, channel_list, fs, time, stim, theta, downsample_ratio=10,
                     downsample_using_mean=True, smooth_win_len=101):
    theta = utils.downsample_2d(theta.T, downsample_ratio, downsample_using_mean)
    theta_smooth = np.zeros((theta.shape[0], theta.shape[1]))
    for k in range(theta.shape[0]):
        theta_smooth[k] = scipy.signal.savgol_filter(theta[k], smooth_win_len, 5)
        theta_smooth[k] -= np.min(theta_smooth[k])
        theta_smooth[k] /= np.max(theta_smooth[k])
    theta_smooth *= -1

    plt.plot(theta_smooth.T)
    plt.show()
    stim = utils.downsample(stim.squeeze(), downsample_ratio)
    stim[stim >= 0.5] = 1
    stim[stim < 0.5] = 0

    data = np.vstack((stim, theta_smooth))
    data = data[:, :, np.newaxis]
    channel_list = [stim_channel] + channel_list
    meta_data_fname = op.join(MMVT_DIR, subject, 'electrodes', 'electrodes_bipolar_data.npz')
    np.savez(meta_data_fname, data=data, names=channel_list, conditions=['on'], dt=1/fs)


if __name__ == '__main__':
    subject = 'mg118'
    stim_channel = 'LMF2-LMF1'
    home = [d for d in ['/home/npeled', '/autofs/space/thibault_001/users/npeled/'] if op.isdir(d)][0]
    fname = op.join(home, 'Documents', 'darpa_year4_meeting', 'MG118_Theta_TurnOn.mat')
    downsample_ratio = 100
    downsample_using_mean = False

    channel_list, fs, time, stim, theta = load_file(fname)
    create_mmvt_file(subject, stim_channel, channel_list, fs, time, stim, theta, downsample_ratio,
                     downsample_using_mean)