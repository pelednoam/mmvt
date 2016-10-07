import numpy as np
from functools import partial
import os
import os.path as op
import shutil
import mne
import scipy.io as sio
import scipy
import csv
from collections import defaultdict, OrderedDict, Iterable
import matplotlib.pyplot as plt

from src.utils import utils
from src.mmvt_addon import colors_utils as cu
from src.utils import matlab_utils as mu

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
FREE_SURFER_HOME = utils.get_link_dir(LINKS_DIR, 'freesurfer', 'FREESURFER_HOME')
BLENDER_ROOT_DIR = op.join(LINKS_DIR, 'mmvt')
ELECTRODES_DIR = op.join(LINKS_DIR, 'electrodes')

HEMIS = utils.HEMIS
STAT_AVG, STAT_DIFF = range(2)
STAT_NAME = {STAT_DIFF: 'diff', STAT_AVG: 'avg'}
DEPTH, GRID = range(2)


def montage_to_npy(montage_file, output_file):
    sfp = mne.channels.read_montage(montage_file)
    np.savez(output_file, pos=np.array(sfp.pos), names=sfp.ch_names)


def electrodes_csv_to_npy(ras_file, output_file, bipolar=False, delimiter=','):
    data = np.genfromtxt(ras_file, dtype=str, delimiter=delimiter)
    data = fix_str_items_in_csv(data)
    # Check if the electrodes coordinates has a header
    try:
        header = data[0, 1:].astype(float)
    except:
        data = np.delete(data, (0), axis=0)

    electrodes_types = grid_or_depth(data)
    # pos = data[:, 1:].astype(float)
    if bipolar:
        # names = []
        # pos_biploar, pos_org = [], []
        # for index in range(data.shape[0]-1):
        #     elc_group1, elc_num1 = utils.elec_group_number(data[index, 0])
        #     elc_group2, elc_num12 = utils.elec_group_number(data[index+1, 0])
        #     if elc_group1==elc_group2:
        #         names.append('{}-{}'.format(data[index+1, 0],data[index, 0]))
        #         pos_biploar.append(pos[index] + (pos[index+1]-pos[index])/2)
        #         pos_org.append([pos[index], pos[index+1]])
        # pos = np.array(pos_biploar)
        # pos_org = np.array(pos_org)
        depth_data = data[electrodes_types == DEPTH, :]
        pos = depth_data[:, 1:4].astype(float)
        pos_depth, names_depth, dists_depth, pos_org = [], [], [], []
        for index in range(depth_data.shape[0]-1):
            elc1_name = depth_data[index, 0].strip()
            elc2_name = depth_data[index + 1, 0].strip()
            elc_group1, elc_num1 = utils.elec_group_number(elc1_name)
            elc_group2, elc_num12 = utils.elec_group_number(elc2_name)
            if elc_group1 == elc_group2:
                elec_name = '{}-{}'.format(elc2_name, elc1_name)
                names_depth.append(elec_name)
                pos_depth.append(pos[index] + (pos[index+1]-pos[index])/2)
                pos_org.append([pos[index], pos[index+1]])
        # There is no point in calculating bipolar for grid electrodes
        grid_data = data[electrodes_types == GRID, :]
        names_grid, pos_grid = get_names_dists_non_bipolar_electrodes(grid_data)
        names = np.concatenate((names_depth, names_grid))
        pos = utils.vstack(pos_depth, pos_grid)
        # No need in pos_org for grid electordes
        pos_org.extend([() for _ in pos_grid])
    else:
        names, pos = get_names_dists_non_bipolar_electrodes(data)
        pos_org = []
    if len(set(names)) != len(names):
        raise Exception('Duplicate electrodes names!')
    if pos.shape[0] != len(names):
        raise Exception('pos dim ({}) != names dim ({})'.format(pos.shape[0], len(names)))
    # print(np.hstack((names.reshape((len(names), 1)), pos)))
    np.savez(output_file, pos=pos, names=names, pos_org=pos_org)


def get_names_dists_non_bipolar_electrodes(data):
    names = data[:, 0]
    pos = data[:, 1:4].astype(float)
    return names, pos


def grid_or_depth(data):
    pos = data[:, 1:].astype(float)
    dists = defaultdict(list)
    group_type = {}
    electrodes_group_type = [None] * pos.shape[0]
    for index in range(data.shape[0] - 1):
        elc_group1, _ = utils.elec_group_number(data[index, 0])
        elc_group2, _ = utils.elec_group_number(data[index + 1, 0])
        if elc_group1 == elc_group2:
            dists[elc_group1].append(np.linalg.norm(pos[index + 1] - pos[index]))
    for group, group_dists in dists.items():
        #todo: not sure this is the best way to check it. Strip with 1xN will be mistaken as a depth
        if np.max(group_dists) > 2 * np.median(group_dists):
            group_type[group] = GRID
        else:
            group_type[group] = DEPTH
    for index in range(data.shape[0]):
        elc_group, _ = utils.elec_group_number(data[index, 0])
        electrodes_group_type[index] = group_type.get(elc_group, DEPTH)
    return np.array(electrodes_group_type)


def read_electrodes_file(subject, bipolar):
    electrodes_fname = 'electrodes{}_positions.npz'.format('_bipolar' if bipolar else '')
    electrodes_fname = op.join(SUBJECTS_DIR, subject, 'electrodes', electrodes_fname)
    if not op.isfile(electrodes_fname):
        convert_electrodes_coordinates_file_to_npy(subject, bipolar, True)
    d = np.load(electrodes_fname)
    return d['names'], d['pos']


def save_electrodes_file(subject, bipolar, elecs_names, elecs_coordinates, fname_postfix):
    output_fname = 'electrodes{}_positions{}.npz'.format('_bipolar' if bipolar else '', fname_postfix)
    output_fname = op.join(SUBJECTS_DIR, subject, 'electrodes', output_fname)
    np.savez(output_fname, pos=elecs_coordinates, names=elecs_names, pos_org=[])
    return output_fname


def fix_str_items_in_csv(csv):
    lines = []
    for line in csv:
        fix_line = list(map(lambda x: str(x).replace('"', ''), line))
        if not np.all([len(v)==0 for v in fix_line[1:]]):
            lines.append(fix_line)
    return np.array(lines)


def read_electrodes(electrodes_file):
    elecs = np.load(electrodes_file)
    for (x, y, z), name in zip(elecs['pos'], elecs['names']):
        print(name, x, y, z)


def read_electrodes_data(elecs_data_dic, conditions, montage_file, output_file_name, from_t=0, to_t=None,
                         norm_by_percentile=True, norm_percs=(1,99)):
    for cond_id, (field, file_name) in enumerate(elecs_data_dic.iteritems()):
        d = sio.loadmat(file_name)
        if cond_id == 0:
            data = np.zeros((d[field].shape[0], to_t - from_t, 2))
        times = np.arange(0, to_t*2, 2)
        # todo: Need to do some interpulation for the MEG
        data[:, :, cond_id] = d[field][:, times]
        # time = d['Time']
    if norm_by_percentile:
        norm_val = max(map(abs, [np.percentile(data, norm_percs[ind]) for ind in [0,1]]))
    else:
        norm_val = max(map(abs, [np.max(data), np.min(data)]))
    data /= norm_val
    sfp = mne.channels.read_montage(montage_file)
    avg_data = np.mean(data, 2)
    colors = utils.mat_to_colors(avg_data, np.percentile(avg_data, 10), np.percentile(avg_data, 90), colorsMap='RdBu', flip_cm=True)
    np.savez(output_file_name, data=data, names=sfp.ch_names, conditions=conditions, colors=colors)


def convert_electrodes_coordinates_file_to_npy(subject, bipolar=False, copy_to_blender=True, ras_xls_sheet_name=''):
    rename_and_convert_electrodes_file(subject, ras_xls_sheet_name)
    electrodes_folder = op.join(SUBJECTS_DIR, subject, 'electrodes')
    csv_file = op.join(electrodes_folder, '{}_RAS.csv'.format(subject))
    if not op.isfile(csv_file):
        print('No electrodes coordinates file! {}'.format(csv_file))
        return None

    output_file_name = 'electrodes{}_positions.npz'.format('_bipolar' if bipolar else '')
    output_file = op.join(SUBJECTS_DIR, subject, 'electrodes', output_file_name)
    electrodes_csv_to_npy(csv_file, output_file, bipolar)
    if copy_to_blender:
        blender_file = op.join(BLENDER_ROOT_DIR, subject, 'electrodes', output_file_name)
        shutil.copyfile(output_file, blender_file)
    return output_file


def rename_and_convert_electrodes_file(subject, ras_xls_sheet_name=''):
    subject_elec_fname_no_ras_pattern = op.join(SUBJECTS_DIR, subject, 'electrodes', '{subject}.{postfix}')
    subject_elec_fname_pattern = op.join(SUBJECTS_DIR, subject, 'electrodes', '{subject}_RAS.{postfix}')
    subject_elec_fname_csv = subject_elec_fname_pattern.format(subject=subject, postfix='csv')
    subject_elec_fname_xlsx = subject_elec_fname_pattern.format(subject=subject, postfix='xlsx')

    utils.rename_files([subject_elec_fname_no_ras_pattern.format(subject=subject, postfix='xlsx'),
                        subject_elec_fname_no_ras_pattern.format(subject=subject.upper(), postfix='xlsx'),
                        subject_elec_fname_no_ras_pattern.format(subject=subject, postfix='xls'),
                        subject_elec_fname_no_ras_pattern.format(subject=subject.upper(), postfix='xls')],
                       subject_elec_fname_pattern.format(subject=subject, postfix='xlsx'))
    utils.rename_files([subject_elec_fname_pattern.format(subject=subject.upper(), postfix='csv')],
                       subject_elec_fname_csv)
    utils.rename_files([subject_elec_fname_pattern.format(subject=subject.upper(), postfix='xlsx')],
                       subject_elec_fname_xlsx)
    if op.isfile(subject_elec_fname_xlsx) and \
                    (not op.isfile(subject_elec_fname_csv) or op.getsize(subject_elec_fname_csv) == 0):
        utils.csv_from_excel(subject_elec_fname_xlsx, subject_elec_fname_csv, ras_xls_sheet_name)


def create_electrode_data_file(subject, task, from_t, to_t, stat, conditions, bipolar,
                               electrodes_names_field, moving_average_win_size=0, input_fname='',
                               input_type='ERP', field_cond_template=''):
    if input_fname == '':
        input_fname = 'data.mat'
    input_file = utils.get_file_if_exist(
        [op.join(SUBJECTS_DIR, subject, 'electrodes', input_fname),
         op.join(ELECTRODES_DIR, subject, task, input_fname),
         op.join(SUBJECTS_DIR, subject, 'electrodes', 'evo.mat'),
         op.join(ELECTRODES_DIR, subject, task, 'evo.mat')])
    if not input_file is None:
        input_type = utils.namebase(input_file) if input_type == '' else input_type
    else:
        print('No electrodes data file!!!')
        return
    output_file = op.join(BLENDER_ROOT_DIR, subject, 'electrodes', 'electrodes{}_data_{}.npz'.format(
            '_bipolar' if bipolar else '', STAT_NAME[stat]))
    if field_cond_template == '':
        if input_type.lower() == 'evo':
            field_cond_template = 'AvgERP_{}'
        elif input_type.lower() == 'erp':
            field_cond_template = '{}_ERP'

    #     d = utils.Bag(**sio.loadmat(input_file))
        # pass
    # else:
    if task == 'ECR':
        read_electrodes_data_one_mat(input_file, conditions, stat, output_file, bipolar,
            electrodes_names_field, field_cond_template = field_cond_template, from_t=from_t, to_t=to_t,
            moving_average_win_size=moving_average_win_size)# from_t=0, to_t=2500)
    elif task == 'MSIT':
        if bipolar:
            read_electrodes_data_one_mat(input_file, conditions, stat, output_file, bipolar,
                electrodes_names_field, field_cond_template=field_cond_template, # '{}_bipolar_evoked',
                from_t=from_t, to_t=to_t, moving_average_win_size=moving_average_win_size) #from_t=500, to_t=3000)
        else:
            read_electrodes_data_one_mat(input_file, conditions, stat, output_file, bipolar,
                electrodes_names_field, field_cond_template = '{}_evoked',
                from_t=from_t, to_t=to_t, moving_average_win_size=moving_average_win_size) #from_t=500, to_t=3000)


# def calc_colors(data, norm_by_percentile, norm_percs, threshold, cm_big, cm_small, flip_cm_big, flip_cm_small):
#     data_max, data_min = utils.get_data_max_min(data, norm_by_percentile, norm_percs)
#     data_minmax = max(map(abs, [data_max, data_min]))
#     print('data minmax: {}'.format(data_minmax))
#     colors = utils.mat_to_colors_two_colors_maps(data, threshold=threshold,
#         x_max=data_minmax, x_min = -data_minmax, cm_big=cm_big, cm_small=cm_small,
#         default_val=1, flip_cm_big=flip_cm_big, flip_cm_small=flip_cm_small)
#     return colors
    # colors = utils.mat_to_colors(stat_data, -data_minmax, data_minmax, color_map)


def check_montage_and_electrodes_names(montage_file, electrodes_names_file):
    sfp = mne.channels.read_montage(montage_file)
    names = np.loadtxt(electrodes_names_file, dtype=np.str)
    names = set([str(e.strip()) for e in names])
    montage_names = set(sfp.ch_names)
    print(names - montage_names)
    print(montage_names - names)


def bipolarize_data(data, labels):
    bipolar_electrodes = []
    if isinstance(data, dict):
        single_trials = True
        bipolar_data = {}
        for key in data.keys():
            bipolar_data[key] = np.zeros(data[key].shape)
    else:
        single_trials = False
        bipolar_electrodes_num = calc_bipolar_electrodes_number(labels)
        bipolar_data = np.zeros((bipolar_electrodes_num, data.shape[1], data.shape[2]))
    bipolar_data_index = 0
    for index in range(len(labels) - 1):
        elc1_name = labels[index].strip()
        elc2_name = labels[index + 1].strip()
        elc_group1, _ = utils.elec_group_number(elc1_name)
        elc_group2, _ = utils.elec_group_number(elc2_name)
        if elc_group1 == elc_group2:
            elec_name = '{}-{}'.format(elc2_name, elc1_name)
            bipolar_electrodes.append(elec_name)
            if single_trials:
                for key in data.keys():
                    bipolar_data[key][:, bipolar_data_index, :] = (data[key][:, index, :] + data[key][:, index + 1, :]) / 2.
            else:
                bipolar_data[bipolar_data_index, :, :] = (data[index, :, :] + data[index + 1, :, :]) / 2.
            bipolar_data_index += 1
    if single_trials:
        for key in data.keys():
            bipolar_data[key] = scipy.delete(bipolar_data[key], range(bipolar_data_index, len(labels)), 1)
    return bipolar_data, bipolar_electrodes


def calc_bipolar_electrodes_number(labels):
    bipolar_data_index = 0
    for index in range(len(labels) - 1):
        elc1_name = labels[index].strip()
        elc2_name = labels[index + 1].strip()
        elc_group1, _ = utils.elec_group_number(elc1_name)
        elc_group2, _ = utils.elec_group_number(elc2_name)
        if elc_group1 == elc_group2:
            bipolar_data_index += 1
    return bipolar_data_index


def read_electrodes_data_one_mat(mat_file, conditions, stat, output_file_name, bipolar, electrodes_names_field,
        field_cond_template, from_t=0, to_t=None, norm_by_percentile=True, norm_percs=(3, 97), threshold=0,
        color_map='jet', cm_big='YlOrRd', cm_small='PuBu', flip_cm_big=False, flip_cm_small=True,
        moving_average_win_size=0, downsample=2):
    # load the matlab file
    d = sio.loadmat(mat_file)
    # get the labels names
    if electrodes_names_field in d:
        labels = d[electrodes_names_field]
        labels = mu.matlab_cell_str_to_list(labels)
        if bipolar and '-' in labels[0]:
            labels = fix_bipolar_labels(labels)
        # else:
        #     #todo: change that!!!
        #     if len(labels) == 1:
        #         labels = [str(l[0]) for l in labels[0]]
        #     else:
        #         labels = [str(l[0][0]) for l in labels]
    else:
        raise Exception('electrodes_names_field not in the matlab file!')
    # Loop for each condition
    for cond_id, cond_name in enumerate(conditions):
        field = field_cond_template.format(cond_name)
        if field not in d:
            field = field_cond_template.format(cond_name.title())
        if field not in d:
            print('{} not in the mat file!'.format(cond_name))
            continue
        # todo: downsample suppose to be a cmdline paramter
        # dims: samples * electrodes * time (single triale) or electrodes * time (evoked)
        if to_t == 0 or to_t == -1:
            to_t = int(d[field].shape[-1] / downsample)
        # initialize the data matrix (electrodes_num x T x 2)
        # times = np.arange(0, to_t*2, 2)
        cond_data = d[field] # [:, times]
        if cond_id == 0:
            single_trials = cond_data.ndim == 3
            data = {} if single_trials else np.zeros((cond_data.shape[0], to_t - from_t, 2))
        if single_trials:
            # Different number of trials per condition, can't save it in a matrix
            # data[cond_name] = np.zeros((cond_data.shape[0], cond_data.shape[1], to_t - from_t))
            cond_data_downsample = utils.downsample_3d(cond_data, downsample)
            data[cond_name] = cond_data_downsample[:, :, from_t:to_t]
        else:
            cond_data_downsample = utils.downsample_2d(cond_data, downsample)
            data[:, :, cond_id] = cond_data_downsample[:, from_t:to_t]

    if bipolar:
        data, labels = bipolarize_data(data, labels)

    if single_trials:
        output_fname = op.join(BLENDER_ROOT_DIR, subject, 'electrodes', 'electrodes{}_data_st.npz'.format(
            '_bipolar' if bipolar else ''))
        data_conds = [(data[key], key) for key in data.keys()]
        data = [d[0] for d in data_conds]
        conditions = [d[1] for d in data_conds]
        np.savez(output_fname, data=data, names=labels, conditions=conditions)
    else:
        data = utils.normalize_data(data, norm_by_percentile, norm_percs)
        stat_data = calc_stat_data(data, stat)
        calc_colors = partial(
            utils.mat_to_colors_two_colors_maps, threshold=threshold, cm_big=cm_big, cm_small=cm_small, default_val=1,
            flip_cm_big=flip_cm_big, flip_cm_small=flip_cm_small, min_is_abs_max=True, norm_percs=norm_percs)
        if moving_average_win_size > 0:
            # data_mv[:, :, cond_id] = utils.downsample_2d(data[:, :, cond_id], moving_average_win_size)
            stat_data_mv = utils.moving_avg(stat_data, moving_average_win_size)
            colors_mv = calc_colors(stat_data_mv)
            np.savez(output_file_name, data=data, stat=stat_data_mv, names=labels, conditions=conditions, colors=colors_mv)
        else:
            colors = calc_colors(stat_data)
            np.savez(output_file_name, data=data, names=labels, conditions=conditions, colors=colors)


def fix_bipolar_labels(labels):
    ret = []
    for label in labels:
        elc1, elc2 = str(label).split(' ')
        group, num1 = utils.elec_group_number(elc1, False)
        _, num2 = utils.elec_group_number(elc2, False)
        ret.append('{}{}-{}{}'.format(group, num2, group, num1))
    return ret


def calc_stat_data(data, stat):
    if stat == STAT_AVG:
        stat_data = np.squeeze(np.mean(data, axis=2))
    elif stat == STAT_DIFF:
        stat_data = np.squeeze(np.diff(data, axis=2))
    else:
        raise Exception('Wrong stat value!')
    return stat_data


def find_first_electrode_per_group(electrodes, positions, bipolar=False):
    groups = OrderedDict()  # defaultdict(list)
    first_electrodes = OrderedDict()
    for elc, pos in zip(electrodes, positions):
        elc_group = utils.elec_group(elc, bipolar)
        if elc_group not in groups:
            groups[elc_group] = []
        groups[elc_group].append((elc, pos))
    first_pos = np.empty((len(groups), 3))
    for ind, (group, group_electrodes) in enumerate(groups.items()):
        first_electrode = sorted(group_electrodes)[0]
        first_pos[ind, :] = first_electrode[1]
        first_electrodes[group] = first_electrode[0]
    return first_electrodes, first_pos, groups


def get_groups_pos(electrodes, positions, bipolar):
    groups = defaultdict(list)
    for elc, pos in zip(electrodes, positions):
        elc_group = utils.elec_group(elc, bipolar)
        groups[elc_group].append(pos)
    return groups


def find_groups_hemi(electrodes, transformed_positions, bipolar):
    groups = get_groups_pos(electrodes, transformed_positions, bipolar)
    groups_hemi = {}
    for group, positions in groups.items():
        trans_pos = np.array(positions)
        hemi = 'rh' if sum(trans_pos[:, 1] < 0) > 0 else 'lh'
        groups_hemi[group] = hemi
    return groups_hemi


def sort_groups(first_electrodes, transformed_first_pos, groups_hemi, bipolar):
    sorted_groups = {}
    for hemi in ['rh', 'lh']:
        # groups_pos = sorted([(pos[0], group) for (group, elc), pos in zip(
        #     first_electrodes.items(), transformed_first_pos) if groups_hemi[utils.elec_group(elc, bipolar)] == hemi])
        groups_pos = []
        for (group, elc), pos in zip(first_electrodes.items(), transformed_first_pos):
            group_hemi = groups_hemi[utils.elec_group(elc, bipolar)]
            if group_hemi == hemi:
                groups_pos.append((pos[0], group))
        groups_pos = sorted(groups_pos)
        sorted_groups[hemi] = [group_pos[1] for group_pos in groups_pos]
    return sorted_groups


def sort_electrodes_groups(subject, bipolar, do_plot=True):
    from sklearn.decomposition import PCA
    electrodes, pos = read_electrodes_file(subject, bipolar)
    first_electrodes, first_pos, elc_pos_groups = find_first_electrode_per_group(electrodes, pos, bipolar)
    pca = PCA(n_components=2)
    pca.fit(first_pos)
    transformed_pos = pca.transform(pos)
    # transformed_pos_3d = PCA(n_components=3).fit(first_pos).transform(pos)
    transformed_first_pos = pca.transform(first_pos)
    groups_hemi = find_groups_hemi(electrodes, transformed_pos, bipolar)
    sorted_groups = sort_groups(first_electrodes, transformed_first_pos, groups_hemi, bipolar)
    print(sorted_groups)
    utils.save(sorted_groups, op.join(BLENDER_ROOT_DIR, subject, 'electrodes', 'sorted_groups.pkl'))
    if do_plot:
        # utils.plot_3d_scatter(pos, names=electrodes.tolist(), labels=first_electrodes.values())
        # electrodes_3d_scatter_plot(pos, first_pos)
        first_electrodes_names = list(first_electrodes.values())
        utils.plot_2d_scatter(transformed_first_pos, names=first_electrodes_names)
        # utils.plot_2d_scatter(transformed_pos, names=electrodes.tolist(), labels=first_electrodes_names)


def read_edf(edf_fname, from_t, to_t):
    import mne.io
    edf_raw = mne.io.read_raw_edf(edf_fname, preload=True)
    edf_raw.notch_filter(np.arange(60, 241, 60))
    dt = (edf_raw.times[1] - edf_raw.times[0])
    hz = int(1/ dt)
    T = edf_raw.times[-1] # sec
    live_channels = find_live_channels(edf_raw, hz)

    ylim = [-0.0015, 0.0015]
    from_t = 17
    window = to_t - from_t
    # plot_window(edf_raw, live_channels, t_start, window, hz, ylim)
    # plot_all_windows(edf_raw, live_channels, T, hz, window, edf_fname, ylim)

    data, times = edf_raw[:, int(from_t*hz):int(from_t*hz) + hz * window]
    # plot_power(data[0], dt)
    # edf_raw.plot(None, 1, 20, 20)
    print('sdf')


def create_raw_data_for_blender(subject, edf_name, conds, bipolar=False, norm_by_percentile=True,
                                norm_percs=(3, 97), threshold=0, cm_big='YlOrRd', cm_small='PuBu', flip_cm_big=False,
                                flip_cm_small=True, moving_average_win_size=0, stat = STAT_DIFF, do_plot=False):
    import mne.io
    edf_fname = op.join(ELECTRODES_DIR, subject, edf_name)
    if not op.isfile(edf_fname):
        raise Exception('The EDF file cannot be found in {}!'.format(edf_fname))
    edf_raw = mne.io.read_raw_edf(edf_fname, preload=True)
    edf_raw.notch_filter(np.arange(60, 241, 60))
    dt = (edf_raw.times[1] - edf_raw.times[0])
    hz = int(1/ dt)
    # T = edf_raw.times[-1] # sec
    data_channels, _ = read_electrodes_file(subject, bipolar)
    # live_channels = find_live_channels(edf_raw, hz)
    # no_stim_channels = get_data_channels(edf_raw)
    channels_tup = [(ind, ch) for ind, ch in enumerate(edf_raw.ch_names) if ch in data_channels]
    channels_indices = [c[0] for c in channels_tup]
    labels = [c[1] for c in channels_tup]
    for cond_id, cond in enumerate(conds):
        cond_data, times = edf_raw[channels_indices, int(cond['from_t']*hz):int(cond['to_t']*hz)]
        if cond_id == 0:
            data = np.zeros((cond_data.shape[0], cond_data.shape[1], len(conds)))
        data[:, :, cond_id] = cond_data

    conditions = [c['name'] for c in conds]
    data = utils.normalize_data(data, norm_by_percentile=False)
    stat_data = calc_stat_data(data, STAT_DIFF)
    output_fname = op.join(BLENDER_ROOT_DIR, subject, 'electrodes', 'electrodes{}_data_{}.npz'.format(
        '_bipolar' if bipolar else '', STAT_NAME[stat]))
    calc_colors = partial(
        utils.mat_to_colors_two_colors_maps, threshold=threshold, cm_big=cm_big, cm_small=cm_small, default_val=1,
        flip_cm_big=flip_cm_big, flip_cm_small=flip_cm_small, min_is_abs_max=True, norm_percs=norm_percs)
    if moving_average_win_size > 0:
        stat_data_mv = utils.moving_avg(stat_data, moving_average_win_size)
        colors_mv = calc_colors(stat_data_mv)
        np.savez(output_fname, data=data, stat=stat_data_mv, names=labels, conditions=conditions, colors=colors_mv,
                 times=times)
    else:
        colors = calc_colors(stat_data)
        np.savez(output_fname, data=data, names=labels, conditions=conditions, colors=colors, times=times)

    # if do_plot:
    #     figs_fol = op.join(SUBJECTS_DIR, subject, 'electrodes', 'stat_figs')
    #     utils.make_dir(figs_fol)
    #     plot_stat_data(data, conds, labels, figs_fol)


def plot_stat_data(data, conds, channels, figs_fol):
    plt.plot((data[:, :, 0] - data[:, :, 1]).T)
    plt.savefig(op.join(figs_fol, 'diff.jpg'))
    for ch in range(data.shape[0]):
        plt.figure()
        plt.plot(data[ch, :, 0], label=conds[0]['name'])
        plt.plot(data[ch, :, 1], label=conds[1]['name'])
        plt.plot(data[ch, :, 0] - data[ch, :, 1], label='diff')
        plt.legend()
        plt.title(channels[ch])
        plt.savefig(op.join(figs_fol, '{}.jpg'.format(channels[ch])))
        plt.close()


def get_data_channels(edf_raw):
    return [ind for ind, ch in enumerate(edf_raw.ch_names) if 'STI' not in ch and 'REF' not in ch]


def plot_window(edf_raw, live_channels, t_start, window, hz, ylim=[-0.0015, 0.0015]):
    data, times = edf_raw[:, int(t_start*hz):int(t_start*hz) + hz * window]
    data = data[live_channels, :]
    plt.figure()
    plt.plot(times, data.T)
    plt.xlim([t_start, t_start + window])
    plt.ylim(ylim)


def plot_all_windows(edf_raw, live_channels, T, hz, window, edf_fname, ylim):
    pics_fol = op.join(op.split(edf_fname)[0], 'pics')
    utils.make_dir(pics_fol)
    for t_start in np.arange(0, T-window, window):
        plot_window(edf_raw, live_channels, t_start, window, hz, ylim)
        print('plotting {}-{}'.format(t_start, t_start+window))
        plt.savefig(op.join(pics_fol, '{}-{}.jpg'.format(t_start, t_start+window)))
        plt.close()


def find_live_channels(edf_raw, hz, threshold=1e-6):
    t_start = 0
    secs = 50
    data, times = edf_raw[:, int(t_start*hz):int(t_start*hz) + hz * secs]
    live_channels = np.where(abs(np.sum(np.diff(data, 1), 1)) > threshold)[0]
    print('live channels num: {}'.format(len(live_channels)))
    return live_channels


def plot_power(data, time_step):
    ps = np.abs(np.fft.fft(data))**2
    freqs = np.fft.fftfreq(data.size, time_step)
    idx = np.argsort(freqs)
    plt.plot(freqs[idx], ps[idx])
    plt.show()


def electrodes_2d_scatter_plot(pos):
    plt.scatter(pos[:, 0], pos[:, 1])
    plt.show()


def electrodes_3d_scatter_plot(pos, pos2=None):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2])
    if not pos2 is None:
        ax.scatter(pos2[:, 0], pos2[:, 1], pos2[:, 2], color='r')
    plt.show()


def create_electrodes_labeling_coloring(subject, bipolar, atlas, good_channels=None, error_radius=3, elec_length=4,
        p_threshold=0.05, legend_name='', coloring_fname=''):
    elecs_names, elecs_coords = read_electrodes_file(subject, bipolar)
    elecs_probs, electrode_labeling_fname = utils.get_electrodes_labeling(
        subject, BLENDER_ROOT_DIR, atlas, bipolar, error_radius, elec_length)
    if elecs_probs is None:
        print('No electrodes labeling file!')
        return
    if electrode_labeling_fname != op.join(BLENDER_ROOT_DIR, subject, 'electrodes',
            op.basename(electrode_labeling_fname)):
        shutil.copy(electrode_labeling_fname, op.join(BLENDER_ROOT_DIR, subject, 'electrodes',
            op.basename(electrode_labeling_fname)))
    most_probable_rois = get_most_probable_rois(elecs_probs, p_threshold, good_channels)
    rois_colors_rgbs, rois_colors_names = get_rois_colors(subject, atlas, most_probable_rois)
    save_rois_colors_legend(subject, rois_colors_rgbs, bipolar, legend_name)
    utils.make_dir(op.join(BLENDER_ROOT_DIR, subject, 'coloring'))
    if coloring_fname == '':
        coloring_fname = 'electrodes{}_coloring.csv'.format('_bipolar' if bipolar else '')
    coloring_fol = op.join(BLENDER_ROOT_DIR, subject, 'coloring')
    coloring_fname =  op.join(coloring_fol, coloring_fname)
    colors_names_fname = op.join(coloring_fol, 'electrodes{}_colors_names.txt'.format('_bipolar' if bipolar else ''))
    elec_names_rois_colors = defaultdict(list)
    with open(coloring_fname, 'w') as colors_rgbs_file, open(colors_names_fname, 'w') as colors_names_file:
        # colors_csv_writer = csv.writer(colors_csv_file, delimiter=',')
        for elec_name, elec_probs in zip(elecs_names, elecs_probs):
            assert(elec_name == elec_probs['name'])
            if not good_channels is None and elec_name not in good_channels:
                continue
            roi = get_most_probable_roi([*elec_probs['cortical_probs'], *elec_probs['subcortical_probs']],
                [*elec_probs['cortical_rois'], *elec_probs['subcortical_rois']], p_threshold)
            if roi != '':
                inv_roi = utils.get_hemi_indifferent_roi(roi)
                colors_rgbs_file.write('{},{},{},{}\n'.format(elec_name, *rois_colors_rgbs[inv_roi]))
                colors_names_file.write('{},{},{}\n'.format(elec_name, inv_roi, rois_colors_names[inv_roi]))
                elec_names_rois_colors[inv_roi].append(elec_name)
                # colors_csv_writer.writerow([elec_name, *color_rgb])
    with open(op.join(coloring_fol, 'electrodes_report.txt'), 'w') as elecs_report_file:
        for inv_roi, electrodes_names in elec_names_rois_colors.items():
            elecs_report_file.write('{},{},{}\n'.format(inv_roi, rois_colors_names[inv_roi], ','.join(electrodes_names)))


def get_most_probable_rois(elecs_probs, p_threshold, good_channels=None):
    if not good_channels is None:
        elecs_probs = list(filter(lambda e:e['name'] in good_channels, elecs_probs))
    probable_rois = set([get_most_probable_roi([*elec['cortical_probs'], *elec['subcortical_probs']],
        [*elec['cortical_rois'], *elec['subcortical_rois']], p_threshold) for elec in elecs_probs])
    if '' in probable_rois:
        probable_rois.remove('')
    return utils.get_hemi_indifferent_rois(probable_rois)


def get_most_probable_roi(probs, rois, p_threshold):
    probs_rois = sorted([(p, r) for p, r in zip(probs, rois)])[::-1]
    if len(probs_rois) == 0:
        roi = ''
    elif len(probs_rois) == 1 and 'white' in probs_rois[0][1].lower():
        roi = probs_rois[0][1]
    elif 'white' in probs_rois[0][1].lower():
        roi = probs_rois[1][1] if probs_rois[1][0] > p_threshold else probs_rois[0][1]
    else:
        roi = probs_rois[0][1]
    return roi


def get_rois_colors(subject, atlas, rois):
    not_white_rois = set(filter(lambda r:'white' not in r.lower(), rois))
    white_rois = rois - not_white_rois
    not_white_rois = sorted(list(not_white_rois))
    colors = cu.get_distinct_colors_and_names()
    lables_colors_rgbs_fname = op.join(BLENDER_ROOT_DIR, subject, 'coloring', 'labels_{}_coloring.csv'.format(atlas))
    lables_colors_names_fname = op.join(BLENDER_ROOT_DIR, subject, 'coloring', 'labels_{}_colors_names.csv'.format(atlas))
    labels_colors_exist = op.isfile(lables_colors_rgbs_fname) and op.isfile(lables_colors_names_fname)
    rois_colors_rgbs, rois_colors_names = OrderedDict(), OrderedDict()
    if not labels_colors_exist:
        print('No labels coloring file!')
    else:
        labels_colors_rgbs = np.genfromtxt(lables_colors_rgbs_fname, dtype=str, delimiter=',')
        labels_colors_names = np.genfromtxt(lables_colors_names_fname, dtype=str, delimiter=',')
    for roi in not_white_rois:
        # todo: set the labels colors to be the same in both hemis
        if labels_colors_exist:
            roi_inds = np.where(labels_colors_rgbs[:, 0] == '{}-rh'.format(roi))[0]
            if len(roi_inds) > 0:
                color_rgb = labels_colors_rgbs[roi_inds][0, 1:].tolist()
                color_name = labels_colors_names[roi_inds][0, 1]
            else:
                color_rgb, color_name = next(colors)
        else:
            color_rgb, color_name = next(colors)
        rois_colors_rgbs[roi], rois_colors_names[roi] = color_rgb, color_name
    for white_roi in white_rois:
        rois_colors_rgbs[white_roi], rois_colors_names[white_roi] = cu.name_to_rgb('white').tolist(), 'white'
    return rois_colors_rgbs, rois_colors_names


def save_rois_colors_legend(subject, rois_colors, bipolar, legend_name=''):
    from matplotlib import pylab
    if legend_name == '':
        legend_name = 'electrodes{}_coloring_legend.jpg'.format('_bipolar' if bipolar else '')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    figlegend = pylab.figure()
    dots, labels = [], []
    for roi, color in rois_colors.items():
        dots.append(ax.scatter([0],[0], c=color))
        labels.append(roi)
    figlegend.legend(dots, labels, 'center')
    figlegend.savefig(op.join(BLENDER_ROOT_DIR, subject, 'coloring', legend_name))


def transform_electrodes_to_mni(subject, args):
    from src.utils import freesurfer_utils as fu
    elecs_names, elecs_coords = read_electrodes_file(subject, args.bipolar)
    elecs_coords_mni = fu.transform_subject_to_mni_coordinates(subject, elecs_coords, SUBJECTS_DIR)
    save_electrodes_coords(elecs_names, elecs_coords_mni, args.good_channels, args.bad_channels)


def save_electrodes_coords(elecs_names, elecs_coords_mni, good_channels=None, bad_channels=None):
    good_elecs_names, good_elecs_coords_mni = [], []
    for elec_name, elec_coord_min in zip(elecs_names, elecs_coords_mni):
        if (not good_channels or elec_name in good_channels) and (not bad_channels or elec_name not in bad_channels):
            good_elecs_names.append(elec_name)
            good_elecs_coords_mni.append(elec_coord_min)
    good_elecs_coords_mni = np.array(good_elecs_coords_mni)
    electrodes_mni_fname = save_electrodes_file(subject, args.bipolar, good_elecs_names, good_elecs_coords_mni, '_mni')
    output_file_name = op.split(electrodes_mni_fname)[1]
    utils.make_dir(op.join(BLENDER_ROOT_DIR, 'colin27', 'electrodes'))
    blender_file = op.join(BLENDER_ROOT_DIR, 'colin27', 'electrodes', output_file_name.replace('_mni', ''))
    shutil.copyfile(electrodes_mni_fname, blender_file)


def set_args(args):
    # todo: fix this part!
    # if args.to_t.isnumeric() and args.from_t.isnumeric():
    #     args.from_t, args.to_t, args.indices_shift = int(args['from_t']), int(args['to_t']), int(args['indices_shift'])
    #     args.from_t_ind, args.to_t_ind = args.from_t + args.indices_shift, args.to_t + args.indices_shift
    # elif ',' in args.to_t and ',' in args.from_t and ',' in args.conditions:
    #     args.to_t = list(map(int, args.to_t.split(',')))
    #     args.from_t = list(map(int, args.to_t.split(',')))
    #     assert (len(args.to_t) == len(args.from_t) == len(args.conditions))
    # else:
    #     print('No from_t, to_t and conditions!')

    if args.task == 'ECR':
        args.conditions = ['congruent', 'incongruent'] # ['happy', 'fearful'] # ['happy', 'fear']
    elif args.task == 'MSIT':
        args.conditions = ['noninterference', 'interference']
    elif args.task == 'seizure':
        args.conditions = [dict(name='baseline', from_t=args.from_t[0], to_t=args.to_t[0]),
                      dict(name='seizure', from_t=args.from_t[1], to_t=args.to_t[1])]
        # conditions = [dict(name='baseline', from_t=12, to_t=16), dict(name='seizure', from_t=from_t, to_t=20)]
    else:
        if isinstance(args.from_t, Iterable) and isinstance(args.to_t, Iterable):
            args.conditions = [dict(name=cond_name, from_t=from_t, to_t=to_t) for (cond_name, from_t, to_t) in
                          zip(args.conditions, args.from_t, args.to_t)]
    return args


def main(subject, args):
    utils.make_dir(op.join(BLENDER_ROOT_DIR, subject, 'electrodes'))
    utils.make_dir(op.join(BLENDER_ROOT_DIR, subject, 'coloring'))
    utils.make_dir(op.join(BLENDER_ROOT_DIR, subject))
    args = set_args(args)

    if utils.should_run(args, 'convert_electrodes_file_to_npy'):
        convert_electrodes_coordinates_file_to_npy(
            subject, bipolar=args.bipolar, ras_xls_sheet_name=args.ras_xls_sheet_name)

    if utils.should_run(args, 'sort_electrodes_groups'):
        sort_electrodes_groups(subject, args.bipolar, do_plot=args.do_plot)

    if utils.should_run(args, 'create_electrode_data_file') and not args.task is None:
        create_electrode_data_file(subject, args.task, args.from_t_ind, args.to_t_ind, args.stat, args.conditions,
                                   args.bipolar, args.electrodes_names_field, args.moving_average_window_size,
                                   args.input_matlab_fname, args.input_type, args.field_cond_template)

    if utils.should_run(args, 'create_electrodes_labeling_coloring'):
        create_electrodes_labeling_coloring(subject, args.bipolar, args.atlas)

    if utils.should_run(args, 'transform_electrodes_to_mni'):
        transform_electrodes_to_mni(subject, args)

    if 'show_image' in args.function:
        legend_name = 'electrodes{}_coloring_legend.jpg'.format('_bipolar' if args.bipolar else '')
        utils.show_image(op.join(BLENDER_ROOT_DIR, subject, 'coloring', legend_name))

    if 'create_raw_data_for_blender' in args.function:# and not args.task is None:
        create_raw_data_for_blender(subject, args.raw_fname, args.conditions, do_plot=args.do_plot)

    # check_montage_and_electrodes_names('/homes/5/npeled/space3/MMVT/mg79/mg79.sfp', '/homes/5/npeled/space3/inaivu/data/mg79_ieeg/angelique/electrode_names.txt')


def read_cmd_args(argv=None):
    from src.utils import args_utils as au
    import argparse
    parser = argparse.ArgumentParser(description='MMVT electrodes preprocessing')
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-t', '--task', help='task', required=False)
    parser.add_argument('-a', '--atlas', help='atlas name', required=False, default='aparc.DKTatlas40')
    parser.add_argument('-b', '--bipolar', help='bipolar', required=False, default=0, type=au.is_true)
    parser.add_argument('-f', '--function', help='function name', required=False, default='all', type=au.str_arr_type)
    parser.add_argument('--exclude', help='functions not to run', required=False, default='', type=au.str_arr_type)
    parser.add_argument('--good_channels', help='good channels', required=False, default='', type=au.str_arr_type)
    parser.add_argument('--bad_channels', help='bad channels', required=False, default='', type=au.str_arr_type)
    parser.add_argument('--from_t', help='from_t', required=False, default=0, type=float) # was -500
    parser.add_argument('--to_t', help='to_t', required=False, default=0, type=float) # was 2000
    parser.add_argument('--from_t_ind', help='from_t_ind', required=False, default=0, type=int) # was -500
    parser.add_argument('--to_t_ind', help='to_t_ind', required=False, default=-1, type=int) # was 2000
    parser.add_argument('--indices_shift', help='indices_shift', required=False, default=0, type=int) # was 1000
    parser.add_argument('--conditions', help='conditions', required=False, default='')
    parser.add_argument('--raw_fname', help='raw fname', required=False, default='')
    parser.add_argument('--stat', help='stat', required=False, default=STAT_DIFF, type=int)
    parser.add_argument('--electrodes_names_field', help='electrodes_names_field', required=False, default='names')
    parser.add_argument('--moving_average_window_size', help='', required=False, default=0, type=int)
    parser.add_argument('--field_cond_template', help='', required=False, default='{}')
    parser.add_argument('--input_type', help='', required=False, default='ERP')
    parser.add_argument('--input_matlab_fname', help='', required=False, default='')
    parser.add_argument('--ras_xls_sheet_name', help='ras_xls_sheet_name', required=False, default='')
    parser.add_argument('--do_plot', help='do plot', required=False, default=0, type=au.is_true)
    args = utils.Bag(au.parse_parser(parser, argv))
    print(args)
    return args


if __name__ == '__main__':
    args = read_cmd_args()
    for subject in args.subject:
        print('********* {} ***********'.format(subject))
        main(subject, args)
    print('finish!')
