import numpy as np
import os
import os.path as op
from collections import defaultdict
from src.utils import utils
from src.utils import labels_utils as lu

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
FREE_SURFER_HOME = utils.get_link_dir(LINKS_DIR, 'freesurfer', 'FREESURFER_HOME')
BLENDER_ROOT_DIR = op.join(LINKS_DIR, 'mmvt')
os.environ['SUBJECTS_DIR'] = SUBJECTS_DIR


def load_stim_file(subject, args):
    stim_fname = op.join(BLENDER_ROOT_DIR, subject, 'electrodes', '{}{}.npz'.format(
        args.file_frefix, args.stim_channel))
    stim = np.load(stim_fname)
    labels, psd, time, freqs = (stim[k] for k in ['labels', 'psd', 'time', 'freqs'])
    bipolar = '-' in labels[0]
    data = None
    freqs_dim = psd.shape.index(len(freqs))
    labels_dim = psd.shape.index(len(labels))
    if time.ndim > 0:
        time_dim = psd.shape.index(len(time))
    else:
        time_dim = next(iter(set(range(3)) - set([freqs_dim, labels_dim])))
    T, L, F = psd.shape[time_dim], psd.shape[labels_dim], psd.shape[freqs_dim]
    if args.downsample > 1:
        time = utils.downsample(time, args.downsample)
    colors = None
    conditions = []
    for freq_ind, (freq_from, freq_to) in enumerate(freqs):
        if data is None:
            # initialize the data matrix (electrodes_num x T x freqs_num)
            data = np.zeros((L, T, F))
        psd_slice = get_psd_freq_slice(psd, freq_ind, freqs_dim, time_dim)
        if args.downsample > 1:
            psd_slice = utils.downsample(psd_slice, args.downsample)
        data[:, :, freq_ind] = psd_slice
        data_min, data_max = utils.calc_min_max(psd_slice, norm_percs=args.norm_percs)
        if colors is None:
            colors = np.zeros((*data.shape, 3))
        for elec_ind, elec_name in enumerate(labels):
            elec_group = utils.elec_group(elec_name, bipolar)
            # if elec_group in ['LVF', 'RMT']:
            #     colors[elec_ind, :, freq_ind] = utils.mat_to_colors(psd_slice[elec_ind], data_min, data_max, colorsMap='BuGn')
            # else:
            colors[elec_ind, :, freq_ind] = utils.mat_to_colors(psd_slice[elec_ind], data_min, data_max, colorsMap=args.colors_map)
        conditions.append('{}-{}Hz'.format(freq_from, freq_to))
    output_fname = op.join(BLENDER_ROOT_DIR, subject, 'electrodes', 'stim_electrodes_{}{}_{}.npz'.format(
            args.file_frefix, 'bipolar' if bipolar else '', args.stim_channel))
    print('Saving {}'.format(output_fname))
    np.savez(output_fname, data=data, names=labels, conditions=conditions, colors=colors)
    return dict(data=data, labels=labels, conditions=conditions, colors=colors)


def get_stim_labels(subject, args, stim_labels):
    if stim_labels is None:
        stim = np.load(op.join(BLENDER_ROOT_DIR, subject, 'electrodes', '{}{}.npz'.format(
            args.file_frefix, args.stim_channel)))
        bipolar = '-' in stim['labels'][0]
        stim_data = np.load(op.join(BLENDER_ROOT_DIR, subject, 'electrodes', 'stim_electrodes_{}{}_{}.npz'.format(
            args.file_frefix, 'bipolar' if bipolar else '', args.stim_channel)))
        stim_labels = stim_data['names']
    else:
        bipolar = '-' in stim_labels[0]
    return stim_labels, bipolar


def create_stim_electrodes_positions(subject, args, stim_labels=None):
    from src.preproc import electrodes as ep

    stim_labels, bipolar = get_stim_labels(subject, args, stim_labels)
    output_file_name = 'electrodes{}_stim_{}_positions.npz'.format('_bipolar' if bipolar else '', args.stim_channel)
    output_file = op.join(BLENDER_ROOT_DIR, subject, 'electrodes', output_file_name)
    # if op.isfile(output_file):
    #     return

    f = np.load(op.join(BLENDER_ROOT_DIR, subject, 'electrodes', 'electrodes_positions.npz'))
    org_pos, org_labels = f['pos'], f['names']
    if bipolar:
        pos, dists = [], []
        for stim_label in stim_labels:
            group, num1, num2 = utils.elec_group_number(stim_label, True)
            stim_label1 = '{}{}'.format(group, num1)
            stim_label2 = '{}{}'.format(group, num2)
            label_ind1 = np.where(org_labels == stim_label1)[0]
            label_ind2 = np.where(org_labels == stim_label2)[0]
            elc_pos = org_pos[label_ind1] + (org_pos[label_ind2] - org_pos[label_ind1]) / 2
            pos.append(elc_pos.reshape((3)))
            dist = np.linalg.norm(org_pos[label_ind2] - org_pos[label_ind1])
            dists.append(dist)
        pos = np.array(pos)
    else:
        pos = [pos for (pos, label) in zip(org_pos, org_labels) if label in stim_labels]
        dists = [np.linalg.norm(p2 - p1) for p1, p2 in zip(pos[:-1], pos[1:])]

    elcs_oris = calc_ori(stim_labels, bipolar, pos)
    electrodes_types = ep.calc_electrodes_type(stim_labels, dists, bipolar)
    np.savez(output_file, pos=pos, names=stim_labels, dists=dists, electrodes_types=electrodes_types,
             electrodes_oris=elcs_oris, bipolar=bipolar)


def calc_ori(stim_labels, bipolar, pos):
    elcs_oris = np.zeros((len(stim_labels), 3))
    for ind in range(len(stim_labels) - 1):
        # todo: really need to change elec_group_number to return always 2 values
        if bipolar:
            group1, _, _ = utils.elec_group_number(stim_labels[ind], bipolar)
            group2, _, _ = utils.elec_group_number(stim_labels[ind + 1], bipolar)
        else:
            group1, _ = utils.elec_group_number(stim_labels[ind], bipolar)
            group2, _ = utils.elec_group_number(stim_labels[ind + 1], bipolar)
        ori = 1 if group1 == group2 else -1
        elc_pos = pos[ind]
        next_pos = pos[ind + ori]
        dist = np.linalg.norm(next_pos - elc_pos)
        elcs_oris[ind] = ori * (next_pos - elc_pos) / dist
    elcs_oris[-1] = -1 * (pos[-2] - pos[-1]) / np.linalg.norm(pos[-2] - pos[-1])
    return elcs_oris


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


def set_labels_colors(subject, args, stim_dict=None):
    stim_labels = stim_dict['labels'] if stim_dict else None
    stim_labels, bipolar = get_stim_labels(subject, args, stim_labels)
    if stim_dict is None:
        stim_data_fname = op.join(BLENDER_ROOT_DIR, subject, 'electrodes', 'stim_electrodes_{}{}_{}.npz'.format(
                args.file_frefix, 'bipolar' if bipolar else '', args.stim_channel))
        stim_dict = np.load(stim_data_fname)
    stim_data = stim_dict['data']
    electrode_labeling_fname = op.join(BLENDER_ROOT_DIR, subject, 'electrodes',
            '{}_{}_electrodes_cigar_r_{}_l_{}{}_stim_{}.pkl'.format(subject, args.atlas, args.error_radius,
            args.elec_length, '_bipolar' if bipolar else '', args.stim_channel))
    elecs_labeling, electrode_labeling_fname = utils.get_electrodes_labeling(
        subject, BLENDER_ROOT_DIR, args.atlas, bipolar, args.error_radius, args.elec_length,
        other_fname=electrode_labeling_fname)
    if elecs_labeling is None:
        print('No electrodes labeling file!')
        return
    electrodes = [e['name'] for e in elecs_labeling]
    if len(set(stim_labels) - set(electrodes)) > 0:
        print("The electrodes labeling isn't calculated for all the stim electrodes!")
        print(set(stim_labels) - set(electrodes))
        return
    labels_elecs_lookup, subcortical_elecs_lookup = create_labels_electordes_lookup(
        subject, args.atlas, elecs_labeling, args.n_jobs)
    labels_data, labels_colors, data_labels_names = {}, {}, {}
    for hemi in utils.HEMIS:
        labels_data[hemi], labels_colors[hemi], data_labels_names[hemi] = \
            calc_labels_data(labels_elecs_lookup[hemi], stim_data, stim_labels, hemi)
        output_fname = op.join(BLENDER_ROOT_DIR, subject, 'electrodes', 'stim_labels_{}{}_{}-{}.npz'.format(
                args.file_frefix, 'bipolar' if bipolar else '', args.stim_channel, hemi))
        np.savez(output_fname, data=labels_data[hemi], names=data_labels_names[hemi],
                 conditions=stim_dict['conditions'], colors=labels_colors[hemi])

    subcortical_data, subcortical_colors, data_subcortical_names = calc_labels_data(
        subcortical_elecs_lookup, stim_data, stim_labels)
    output_fname = op.join(BLENDER_ROOT_DIR, subject, 'electrodes', 'stim_subcortical_{}{}_{}.npz'.format(
            args.file_frefix, 'bipolar' if bipolar else '', args.stim_channel))
    np.savez(output_fname, data=subcortical_data, names=data_subcortical_names,
             conditions=stim_dict['conditions'], colors=subcortical_colors)


def calc_labels_data(elecs_lookup, stim_data, stim_labels, hemi=None):
    labels_names = list(elecs_lookup.keys())
    labels_data = np.zeros((len(labels_names), stim_data.shape[1], stim_data.shape[2]))
    colors = np.zeros((*labels_data.shape, 3))
    labels_data_names = []
    label_ind = 0
    for label_name, electordes_data in elecs_lookup.items():
        if not hemi is None:
            if lu.get_hemi_from_name(label_name) != hemi:
                continue
        labels_data_names.append(label_name)
        for elec_name, elec_prob in electordes_data:
            elec_inds = np.where(stim_labels == elec_name)[0]
            if len(elec_inds) > 0:
                elec_data = stim_data[elec_inds[0], :, :] * elec_prob
                labels_data[label_ind, :, :] += elec_data
        label_ind += 1
    # Calc colors for each freq
    for freq_id in range(labels_data.shape[2]):
        data_min, data_max = utils.calc_min_max(labels_data[:, :, freq_id], norm_percs=args.norm_percs)
        colors[:, :, freq_id] = utils.mat_to_colors(
            labels_data[:, :, freq_id], data_min, data_max, colorsMap=args.colors_map)
    return labels_data, colors, labels_data_names


def create_labels_electordes_lookup(subject, atlas, elecs_labeling, n_jobs):
    # delim, pos = lu.get_hemi_delim_and_pos(elecs_labeling[0]['cortical_rois'][0])
    # atlas_labels = lu.get_atlas_labels_names(subject, atlas, delim, pos, n_jobs)
    labels_elecs_lookup = dict(rh=defaultdict(list), lh=defaultdict(list))
    subcortical_elecs_lookup = defaultdict(list)
    # for label_name in atlas_labels['rh'] + atlas_labels['lh']:
    for elec_labeling in elecs_labeling:
        for label_name, label_prob in zip(elec_labeling['cortical_rois'], elec_labeling['cortical_probs']):
            # label_ind = elec_labeling['cortical_rois'].index(label_name)
            # label_prob = elec_labeling['cortical_probs'][label_ind]
            hemi = lu.get_hemi_from_name(label_name)
            labels_elecs_lookup[hemi][label_name].append((elec_labeling['name'], label_prob))
        for label_name, label_prob in zip(elec_labeling['subcortical_rois'], elec_labeling['subcortical_probs']):
            subcortical_elecs_lookup[label_name].append((elec_labeling['name'], label_prob))
    return labels_elecs_lookup, subcortical_elecs_lookup


def main(subject, args):
    utils.make_dir(op.join(BLENDER_ROOT_DIR, subject, 'electrodes'))
    stim_data = None
    if 'all' in args.function or 'load_stim_file' in args.function:
        stim_data = load_stim_file(subject, args)
    if 'all' in args.function or 'create_stim_electrodes_positions' in args.function:
        labels = stim_data['labels'] if stim_data else None
        create_stim_electrodes_positions(subject, args, labels)
    if 'all' in args.function or 'set_labels_colors' in args.function:
        set_labels_colors(subject, args, stim_data)


if __name__ == '__main__':
    import argparse
    from src.utils import args_utils as au
    parser = argparse.ArgumentParser(description='MMVT stim preprocessing')
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-a', '--atlas', help='atlas name', required=False, default='aparc.DKTatlas40')
    parser.add_argument('-f', '--function', help='function name', required=False, default='all', type=au.str_arr_type)
    parser.add_argument('--stim_channel', help='stim channel', required=True)
    parser.add_argument('--colors_map', help='activity colors map', required=False, default='OrRd')
    parser.add_argument('--norm_percs', help='normalization percerntiles', required=False, type=au.int_arr_type, default='1,95')
    parser.add_argument('--downsample', help='downsample', required=False, type=int, default=1)
    parser.add_argument('--file_frefix', help='file_frefix', required=False, default='psd_')
    parser.add_argument('--error_radius', help='error_radius', required=False, default=3, type=int)
    parser.add_argument('--elec_length', help='elec_length', required=False, default=4, type=int)
    parser.add_argument('--n_jobs', help='cpu num', required=False, default=1)
    args = utils.Bag(au.parse_parser(parser))
    args.n_jobs = utils.get_n_jobs(args.n_jobs)
    print(args)

    for subject in args.subject:
        main(subject, args)

    print('finish!')