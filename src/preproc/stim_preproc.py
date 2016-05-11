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
    stim = np.load(op.join(BLENDER_ROOT_DIR, subject, 'electrodes', '{}_{}.npz'.format(
        args.file_frefix, args.stim_channel)))
    labels, psd, time, freqs = (stim[k] for k in ['labels', 'psd', 'time', 'freqs'])
    bipolar = '-' in labels[0]
    data = None
    freqs_dim = psd.shape.index(len(freqs))
    time_dim = psd.shape.index(len(time))
    if args.downsample > 1:
        time = utils.downsample(time, args.downsample)
    colors = None
    conditions = []
    for freq_ind, (freq_from, freq_to) in enumerate(freqs):
        if data is None:
            # initialize the data matrix (electrodes_num x T x freqs_num)
            data = np.zeros((len(labels), len(time), len(freqs)))
        psd_slice = get_psd_freq_slice(psd, freq_ind, freqs_dim, time_dim)
        if args.downsample > 1:
            psd_slice = utils.downsample(psd_slice, args.downsample)
        data[:, :, freq_ind] = psd_slice
        data_min, data_max = utils.check_min_max(psd_slice, norm_percs=args.norm_percs)
        if colors is None:
            colors = np.zeros((*data.shape, 3))
        colors[:, :, freq_ind] = utils.mat_to_colors(psd_slice, data_min, data_max, colorsMap=args.colors_map)
        conditions.append('{}-{}Hz'.format(freq_from, freq_to))
    output_fname = op.join(BLENDER_ROOT_DIR, subject, 'electrodes', 'stim_electrodes_{}{}_{}.npz'.format(
            args.file_frefix, '_bipolar' if bipolar else '', args.stim_channel))
    np.savez(output_fname, data=data, names=labels, conditions=conditions, colors=colors)
    return dict(data=data, labels=labels, conditions=conditions, colors=colors)


def create_stim_electrodes_positions(stim_labels=None):
    if stim_labels is None:
        stim = np.load(op.join(BLENDER_ROOT_DIR, subject, 'electrodes', '{}{}.npz'.format(
            args.file_frefix, args.stim_channel)))
        bipolar = '-' in stim['labels'][0]
        stim_data = np.load(op.join(BLENDER_ROOT_DIR, subject, 'electrodes', 'stim_electrodes_{}{}_{}.npz'.format(
            args.file_frefix, '_bipolar' if bipolar else '', args.stim_channel)))
        stim_labels = stim_data['names']
    else:
        bipolar = '-' in stim_labels[0]

    output_file_name = 'electrodes{}_stim_{}_positions.npz'.format('_bipolar' if bipolar else '', args.stim_channel)
    output_file = op.join(BLENDER_ROOT_DIR, subject, 'electrodes', output_file_name)
    if op.isfile(output_file):
        return

    f = np.load(op.join(BLENDER_ROOT_DIR, subject, 'electrodes', 'electrodes_positions.npz'))
    org_pos, org_labels = f['pos'], f['names']
    if bipolar:
        pos = []
        for stim_label in stim_labels:
            group, num1, num2 = utils.elec_group_number(stim_label, True)
            stim_label1 = '{}{}'.format(group, num1)
            stim_label2 = '{}{}'.format(group, num2)
            label_ind1 = np.where(org_labels == stim_label1)[0]
            label_ind2 = np.where(org_labels == stim_label2)[0]
            elc_pos = org_pos[label_ind1] + (org_pos[label_ind2] - org_pos[label_ind1]) / 2
            pos.append(elc_pos.reshape((3)))
        pos = np.array(pos)
    else:
        pos = [pos for (pos, label) in zip(org_pos, org_labels) if label in stim_labels]

    np.savez(output_file, pos=pos, names=stim_labels)


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
    stim_data = None
    if 'all' in args.function or 'load_stim_file' in args.function:
        stim_data = load_stim_file(subject, args)
    if 'all' in args.function or 'create_stim_electrodes_positions' in args.function:
        labels = stim_data['labels'] if stim_data else None
        create_stim_electrodes_positions(labels)

if __name__ == '__main__':
    import argparse
    from src.utils import args_utils as au
    parser = argparse.ArgumentParser(description='MMVT stim preprocessing')
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-a', '--atlas', help='atlas name', required=False, default='aparc.DKTatlas40')
    parser.add_argument('--stim_channel', help='stim channel', required=True)
    parser.add_argument('--colors_map', help='activity colors map', required=False, default='OrRd')
    parser.add_argument('--norm_percs', help='normalization percerntiles', required=False, type=au.int_arr_type)
    parser.add_argument('--downsample', help='downsample', required=False, type=int, default=1)
    parser.add_argument('--file_frefix', help='file_frefix', required=False, default='psd_')
    parser.add_argument('-f', '--function', help='function name', required=False, default='all', type=au.str_arr_type)
    parser.add_argument('--n_jobs', help='cpu num', required=False, default=-1)
    args = utils.Bag(au.parse_parser(parser))
    print(args)
    n_jobs = utils.get_n_jobs(args.n_jobs)

    for subject in args.subject:
        main(subject, args, n_jobs)

    print('finish!')