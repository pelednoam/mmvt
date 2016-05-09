import numpy as np
import os
import os.path as op
from src.utils import utils

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
FREE_SURFER_HOME = utils.get_link_dir(LINKS_DIR, 'freesurfer', 'FREESURFER_HOME')
BLENDER_ROOT_DIR = op.join(LINKS_DIR, 'mmvt')
os.environ['SUBJECTS_DIR'] = SUBJECTS_DIR


def load_stim_file(subject, args):
    stim = np.load(op.join(BLENDER_ROOT_DIR, subject, 'stim', 'psd_{}.npz'.format(args.stim_channel)))
    labels, psd, time, freqs = (stim[k] for k in ['labels', 'psd', 'time', 'freqs'])
    bipolar = '-' in labels[0]
    data = None
    freqs_dim = psd.shape.index(len(freqs))
    time_dim = psd.shape.index(len(time))
    colors = None
    conditions = []
    for freq_ind, (freq_from, freq_to) in enumerate(freqs):
        if data is None:
            # initialize the data matrix (electrodes_num x T x freqs_num)
            data = np.zeros((len(labels), len(time), len(freqs)))
        psd_slice = get_psd_freq_slice(psd, freq_ind, freqs_dim, time_dim)
        data[:, :, freq_ind] = psd_slice
        data_min, data_max = utils.check_min_max(psd_slice)
        if colors is None:
            colors = np.zeros((*data.shape, 3))
        colors[:, :, freq_ind] = utils.mat_to_colors(psd_slice, data_min, data_max, colorsMap=args.colors_map)
        conditions.append('{}-{}'.format(freq_from, freq_to))
    output_fname = op.join(BLENDER_ROOT_DIR, subject, 'electrodes', 'stim_electrodes{}_data.npz'.format(
            '_bipolar' if bipolar else ''))
    np.savez(output_fname, data=data, names=labels, conditions=conditions, colors=colors)


def get_psd_freq_slice(psd, freq_ind, freqs_dim, time_dim):
    if freqs_dim == 0:
        psd_slice = psd[freq_ind, :, :]
        if time_dim == 1:
            psd_slice = psd_slice.T
    elif freqs_dim == 1:
        psd_slice = psd[:, freq_ind, :]
        if time_dim == 0:
            psd_slice = psd_slice.T
    else:
        psd_slice = psd[:, :, freq_ind]
        if time_dim == 0:
            psd_slice = psd_slice.T
    return psd_slice


def main(subject, args, n_jobs):
    utils.make_dir(op.join(BLENDER_ROOT_DIR, subject, 'electrodes'))
    if 'all' in args.function or 'load_stim_file' in args.function:
        load_stim_file(subject, args)


if __name__ == '__main__':
    import argparse
    from src.utils import args_utils as au
    parser = argparse.ArgumentParser(description='MMVT stim preprocessing')
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-a', '--atlas', help='atlas name', required=False, default='aparc.DKTatlas40')
    parser.add_argument('--stim_channel', help='stim channel', required=True)
    parser.add_argument('--colors_map', help='activity colors map', required=False, default='OrRd')
    parser.add_argument('-f', '--function', help='function name', required=False, default='all', type=au.str_arr_type)
    parser.add_argument('--n_jobs', help='cpu num', required=False, default=-1)
    args = utils.Bag(au.parse_parser(parser))
    print(args)
    n_jobs = utils.get_n_jobs(args.n_jobs)

    for subject in args.subject:
        main(subject, args, n_jobs)

    print('finish!')