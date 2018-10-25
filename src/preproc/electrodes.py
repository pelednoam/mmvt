import numpy as np
from functools import partial
import os.path as op
import os
import shutil
import scipy.io as sio
import scipy
from collections import defaultdict, OrderedDict, Iterable
import matplotlib.pyplot as plt
from datetime import datetime
import traceback
import glob
import math
import time
import mne
from mne.filter import notch_filter
import mne.io
import nibabel as nib
from scipy.spatial.distance import cdist

from src.utils import utils
from src.mmvt_addon import colors_utils as cu
from src.utils import matlab_utils as mu
from src.utils import preproc_utils as pu
from src.utils import geometry_utils as gu
from src.utils import labels_utils as lu
from src.utils import args_utils as au

SUBJECTS_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()
ELECTRODES_DIR = utils.get_link_dir(utils.get_links_dir(), 'electrodes')
HEMIS = utils.HEMIS
STAT_AVG, STAT_DIFF = range(2)
STAT_NAME = {STAT_DIFF: 'diff', STAT_AVG: 'avg'}
DEPTH, GRID = range(2)


def montage_to_npy(montage_file, output_file):
    sfp = mne.channels.read_montage(montage_file)
    np.savez(output_file, pos=np.array(sfp.pos), names=sfp.ch_names)


def electrodes_csv_to_npy(ras_file, output_file, bipolar=False, delimiter=',', electrodes_type=None):
    data = np.genfromtxt(ras_file, dtype=str, delimiter=delimiter)
    if data.shape[1] == 5 and np.all(data[:, 4] == ''):
        data = np.delete(data, (4), axis=1)
    data = fix_str_items_in_csv(data)
    # Check if the electrodes coordinates has a header
    try:
        header = data[0, 1:4].astype(float)
    except:
        data = np.delete(data, (0), axis=0)

    electrodes_types = grid_or_depth(data, electrodes_type)
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
        try:
            names = np.concatenate((names_depth, names_grid))
        except:
            print('Can\'t concatenate names_depth: {} and names_grid: {}'.format(names_depth, names_grid))
            return None, None
        pos = utils.vstack(pos_depth, pos_grid)
        # No need in pos_org for grid electordes
        pos_org.extend([() for _ in pos_grid])
    else:
        names, pos = get_names_dists_non_bipolar_electrodes(data)
        # fix for a bug in ielu
        names = [n.replace('elec_unsorted', '') for n in names]
        pos_org = []
    if len(set(names)) != len(names):
        raise Exception('Duplicate electrodes names!')
    if pos.shape[0] != len(names):
        raise Exception('pos dim ({}) != names dim ({})'.format(pos.shape[0], len(names)))
    # print(np.hstack((names.reshape((len(names), 1)), pos)))
    np.savez(output_file, pos=pos, names=names, pos_org=pos_org)
    return pos, names


def get_names_dists_non_bipolar_electrodes(data):
    names = data[:, 0]
    pos = data[:, 1:4].astype(float)
    return names, pos


def calc_electrodes_types(labels, pos, electrodes_type=None):
    group_type = {}
    dists = defaultdict(list)
    electrodes_group_type = [None] * len(pos)

    if electrodes_type is not None:
        print('All the electrodes are {}'.format('grid' if electrodes_type == GRID else 'depth'))
        for index in range(len(labels)):
            electrodes_group_type[index] = electrodes_type
        return np.array(electrodes_group_type)

    for index in range(len(labels) - 1):
        elc_group1, _ = utils.elec_group_number(labels[index])
        elc_group2, _ = utils.elec_group_number(labels[index + 1])
        if elc_group1 == elc_group2:
            dists[elc_group1].append(np.linalg.norm(pos[index + 1] - pos[index]))
    for group, group_dists in dists.items():
        # todo: not sure this is the best way to check it. Strip with 1xN will be mistaken as a depth
        if np.max(group_dists) > 2 * np.median(group_dists):
            group_type[group] = GRID
        else:
            group_type[group] = DEPTH
        print('{} is {}'.format(group, 'depth' if group_type[group] == 0 else 'grid'))
    if not utils.all_items_equall(list(group_type.values())):
        ret = input('Do you want to reset the types manually (y/n)? ')
        if au.is_true(ret):
            for group in dists.keys():
                man_group_type = None
                while man_group_type not in [0, 1]:
                    man_group_type = input('{} is depth (0) or grid/strip (1)? ')
                    group_type[group] = man_group_type
    for index in range(len(labels)):
        elc_group, _ = utils.elec_group_number(labels[index])
        electrodes_group_type[index] = group_type.get(elc_group, DEPTH)
    return np.array(electrodes_group_type)


def grid_or_depth(data, electrodes_type=None):
    if data.shape[1] == 5:
        electrodes_group_type = [None] * data.shape[0]
        for ind, elc_type in enumerate(data[:, 4]):
            electrodes_group_type[ind] = GRID if elc_type in ['grid', 'strip'] else DEPTH
    else:
        pos = data[:, 1:4].astype(float)
        return calc_electrodes_types(data[:, 0], pos, electrodes_type)


def read_electrodes_file(subject, bipolar, postfix='', snap=False, electrodes_type=None):
    electrodes_fname = op.join(MMVT_DIR, subject, 'electrodes', 'electrodes{}{}_positions{}.npz'.format(
        '_bipolar' if bipolar else '', '_snap' if snap else '',
        '_{}'.format(postfix) if postfix != '' else ''))
    # electrodes_fname = op.join(MMVT_DIR, subject, 'electrodes', electrodes_fname)
    if not op.isfile(electrodes_fname):
        print('{}: No npz file, trying to read xls file'.format(subject))
        try:
            convert_electrodes_pos(subject, bipolar, electrodes_type=electrodes_type)
        except:
            print(traceback.format_exc())
    if not op.isfile(electrodes_fname):
        print("Can't find {} electrodes file (bipolar={})".format(subject, bipolar))
        return [], []
    else:
        try:
            print('Loading {}'.format(electrodes_fname))
            d = np.load(electrodes_fname)
            # fix for a bug in ielu
            names = [n.replace('elec_unsorted', '') for n in d['names']]
            return names, d['pos']
        except:
            os.remove(electrodes_fname)
            return read_electrodes_file(subject, bipolar, postfix, electrodes_type=electrodes_type)


def save_electrodes_file(subject, bipolar, elecs_names, elecs_coordinates, fname_postfix):
    output_fname = 'electrodes{}_positions{}.npz'.format('_bipolar' if bipolar else '', fname_postfix)
    output_fname = op.join(MMVT_DIR, subject, 'electrodes', output_fname)
    np.savez(output_fname, pos=elecs_coordinates, names=elecs_names, pos_org=[])
    return output_fname


def fix_str_items_in_csv(csv):
    lines = []
    for line in csv:
        fix_line = list(map(lambda x: str(x).replace('"', ''), line))
        if not np.all([len(v) == 0 for v in fix_line[1:]]) and np.all([utils.is_float(x) for x in fix_line[1:4]]):
            lines.append(fix_line)
        else:
            print('csv: ignoring the following line: {}'.format(line))
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


def calc_dist_mat(subject, bipolar=False, snap=False):
    from scipy.spatial import distance

    pos_fname = 'electrodes{}_{}positions.npz'.format('_bipolar' if bipolar else '', 'snap_' if snap else '')
    pos_fname = op.join(MMVT_DIR, subject, 'electrodes', pos_fname)
    output_fname = 'electrodes{}_{}dists.npy'.format('_bipolar' if bipolar else '', 'snap_' if snap else '')
    output_fname = op.join(MMVT_DIR, subject, 'electrodes', output_fname)

    if not op.isfile(pos_fname):
        return False
    d = np.load(pos_fname)
    pos = d['pos']
    x = np.zeros((len(pos), len(pos)))
    for i in range(len(pos)):
        for j in range(len(pos)):
            x[i,j] = np.linalg.norm(pos[i]- pos[j]) # np.sqrt((pos[i]- pos[j])**2)
            # assert(x[i,j]==np.linalg.norm(pos[i]- pos[j]))
    # x = distance.cdist(pos, pos, 'euclidean')
    np.save(output_fname, x)
    return op.isfile(output_fname)


def convert_electrodes_pos(
        subject, bipolar=False, ras_xls_sheet_name='', snaps=[True, False], electrodes_type=None):
    rename_and_convert_electrodes_file(subject, ras_xls_sheet_name)
    electrodes_folder = op.join(MMVT_DIR, subject, 'electrodes')
    utils.make_dir(electrodes_folder)

    file_found = False
    for snap in snaps:
        elc_file_name = '{}{}_RAS.csv'.format(subject, '_snap' if snap else '')
        csv_file = op.join(electrodes_folder, elc_file_name)
        subjects_elecs_fname = op.join(SUBJECTS_DIR, subject, 'electrodes', elc_file_name)
        if not op.isfile(csv_file):
            if op.isfile(subjects_elecs_fname):
                shutil.copy(subjects_elecs_fname, op.join(electrodes_folder, elc_file_name))
            else:
                print("Can't find {}!".format(elc_file_name))
                continue
        elif op.isfile(subjects_elecs_fname) and utils.file_is_newer(subjects_elecs_fname, csv_file):
            shutil.copy(subjects_elecs_fname, op.join(electrodes_folder, elc_file_name))
        file_found = True
        output_file_name = 'electrodes{}_{}positions.npz'.format('_bipolar' if bipolar else '', 'snap_' if snap else '')
        output_file = op.join(MMVT_DIR, subject, 'electrodes', output_file_name)
        pos, names = electrodes_csv_to_npy(csv_file, output_file, bipolar, electrodes_type=electrodes_type)
        if pos is None or names is None:
            return False, None, None
        # if copy_to_blender:
        #     blender_file = op.join(MMVT_DIR, subject, 'electrodes', output_file_name)
        #     shutil.copyfile(output_file, blender_file)
    if not file_found:
        print('No electrodes coordinates file!')
        return False, None, None
    return op.isfile(output_file), pos, names


def rename_and_convert_electrodes_file(subject, ras_xls_sheet_name=''):
    for root_fol in [MMVT_DIR, SUBJECTS_DIR]:
        subject_elec_fname_no_ras_pattern = op.join(root_fol, subject, 'electrodes', '{subject}.{postfix}')
        subject_elec_fname_pattern = op.join(root_fol, subject, 'electrodes', '{subject}_RAS.{postfix}')
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


@pu.tryit_ret_bool
def create_electrode_data_file(subject, task, from_t, to_t, stat, conditions, bipolar,
                               electrodes_names_field, moving_average_win_size=0, input_fname='',
                               input_type='ERP', field_cond_template=''):
    if input_fname == '':
        input_fname = 'data.mat'
    input_file = utils.get_file_if_exist(
        [op.join(MMVT_DIR, subject, 'electrodes', input_fname),
         op.join(ELECTRODES_DIR, subject, task, input_fname),
         op.join(MMVT_DIR, subject, 'electrodes', 'evo.mat'),
         op.join(ELECTRODES_DIR, subject, task, 'evo.mat')])
    if not input_file is None:
        input_type = utils.namebase(input_file) if input_type == '' else input_type
    else:
        print('No electrodes data file!!!')
        return
    output_file = op.join(MMVT_DIR, subject, 'electrodes', 'electrodes{}_data_{}.npz'.format(
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
        read_electrodes_data_one_mat(subject, input_file, conditions, stat, output_file, bipolar,
            electrodes_names_field, field_cond_template = field_cond_template, from_t=from_t, to_t=to_t,
            moving_average_win_size=moving_average_win_size)# from_t=0, to_t=2500)
    elif task == 'MSIT':
        if bipolar:
            read_electrodes_data_one_mat(subject, input_file, conditions, stat, output_file, bipolar,
                electrodes_names_field, field_cond_template=field_cond_template, # '{}_bipolar_evoked',
                from_t=from_t, to_t=to_t, moving_average_win_size=moving_average_win_size) #from_t=500, to_t=3000)
        else:
            read_electrodes_data_one_mat(subject, input_file, conditions, stat, output_file, bipolar,
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


def read_electrodes_data_one_mat(subject, mat_file, conditions, stat, output_file_name, bipolar, electrodes_names_field,
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
        output_fname = op.join(MMVT_DIR, subject, 'electrodes', 'electrodes{}_data_st.npz'.format(
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
    if data.shape[2] == 1:
        stat_data = data
    elif stat == STAT_AVG:
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
        hemi = 'lh' if sum(trans_pos[:, 1] < 0) > 0 else 'rh'
        groups_hemi[group] = hemi
    return groups_hemi


# def sort_groups(first_electrodes, transformed_first_pos, groups_hemi, bipolar):
#     sorted_groups = {}
#     for hemi in ['rh', 'lh']:
#         # groups_pos = sorted([(pos[0], group) for (group, elc), pos in zip(
#         #     first_electrodes.items(), transformed_first_pos) if groups_hemi[utils.elec_group(elc, bipolar)] == hemi])
#         groups_pos = []
#         for (group, elc), pos in zip(first_electrodes.items(), transformed_first_pos):
#             group_hemi = groups_hemi[utils.elec_group(elc, bipolar)]
#             if group_hemi == hemi:
#                 groups_pos.append((pos[0], group))
#         groups_pos = sorted(groups_pos)
#         sorted_groups[hemi] = [group_pos[1] for group_pos in groups_pos]
#     return sorted_groups


@pu.tryit_ret_bool
def find_electrodes_hemis(subject, bipolar, sigma=0, manual=False, electrodes_type=None):
    from collections import Counter

    elcs_groups = {}
    groups = defaultdict(list)
    sorted_groups = dict(rh=[], lh=[])

    electrodes, electrodes_t1_tkreg = read_electrodes_file(subject, bipolar, electrodes_type=electrodes_type)
    if manual:
        groups = list(set([utils.elec_group(elc, bipolar) for elc in electrodes]))
        for group in groups:
            hem = input("{}: r/l? ".format(group))
            while hem not in ['r', 'l']:
                hem = input('r/l? ')
            hemi = '{}h'.format(hem)
            sorted_groups[hemi].append(group)
        utils.save(sorted_groups, op.join(MMVT_DIR, subject, 'electrodes', 'sorted_groups.pkl'))
        return True

    in_dural = check_if_electrodes_inside_the_dura(subject, electrodes_t1_tkreg, sigma)
    if in_dural is None:
        return False
    hemis = {elc_name:'rh' if in_dural['rh'][ind] else 'lh' if in_dural['lh'][ind] else 'un' for ind, elc_name in
             enumerate(electrodes)}
    for elc_name in electrodes:
        elc_group = utils.elec_group(elc_name, bipolar)
        elcs_groups[elc_name] = elc_group
        groups[elc_group].append(elc_name)
    groups_hemis = {group:Counter([hemis[elc] for elc in electrodes]).most_common()[0][0]
                    for group, electrodes in groups.items()}
    for group, group_hemi in groups_hemis.items():
        if group_hemi == 'un':
            hem = input("Couldn't detect the hemisphere for {}. r/l? ".format(group))
            while hem not in ['r', 'l']:
                hem = input('r/l? ')
            groups_hemis[group] = '{}h'.format(hem)
    for hemi in utils.HEMIS:
        sorted_groups[hemi] = [group for group, group_hemi in groups_hemis.items() if group_hemi == hemi]
    print(sorted_groups)
    utils.save(sorted_groups, op.join(MMVT_DIR, subject, 'electrodes', 'sorted_groups.pkl'))
    return True


def get_electrodes_hemis(subject, bipolar=False, sigma=0, manual=False, electrodes_type=None):
    fname = op.join(MMVT_DIR, subject, 'electrodes', 'sorted_groups.pkl')
    if op.isfile(fname):
        return utils.load(fname)
    else:
        find_electrodes_hemis(subject, bipolar, sigma=sigma, manual=manual, electrodes_type=electrodes_type)
        return utils.load(fname) if op.isfile(fname) else None


def get_group_hemi(group, electrodes_hemis):
    return 'lh' if group in electrodes_hemis['lh'] else 'rh' if group in electrodes_hemis['rh'] else 'uh'


def check_if_electrodes_inside_the_dura(subject, electrodes_t1_tkreg, sigma):

    if not utils.both_hemi_files_exist(op.join(SUBJECTS_DIR, subject, 'surf', '{hemi}.dural')):
        print('check_if_electrodes_inside_the_dura: No dura surface!')
        return None

    in_dural = {}
    dural_verts, _, dural_normals = gu.get_dural_surface(op.join(SUBJECTS_DIR, subject), do_calc_normals=True)
    if dural_verts is None:
        from src.misc.dural import create_dural
        create_dural.create_dural_surface(subject, SUBJECTS_DIR)
        dural_verts, _, dural_normals = gu.get_dural_surface(op.join(SUBJECTS_DIR, subject), do_calc_normals=True)
    if dural_verts is None:
        print('No Dural surface!!!')
        return False
    for hemi in ['rh', 'lh']:
        dists = cdist(electrodes_t1_tkreg, dural_verts[hemi])
        close_verts_indices = np.argmin(dists, axis=1)
        in_dural[hemi] = [gu.point_in_mesh(u, dural_verts[hemi][vert_ind], dural_normals[hemi][vert_ind], sigma)
                          for u, vert_ind in zip(electrodes_t1_tkreg, close_verts_indices)]
    return in_dural


@utils.tryit()
def check_how_many_electrodes_inside_the_dura(subject, sigma=0, bipolar=False, electrodes_type=None):
    if bipolar:
        print('This function is only for monopolar electrodes')
        return False
    output_fname = op.join(MMVT_DIR, subject, 'electrodes', 'how_many_inside_dura.csv')
    if op.isfile(output_fname):
        os.remove(output_fname)
    electrodes, electrodes_t1_tkreg = read_electrodes_file(
        subject, bipolar, electrodes_type=electrodes_type)
    in_dural_hemis = check_if_electrodes_inside_the_dura(subject, electrodes_t1_tkreg, sigma)
    if in_dural_hemis is None:
        return False
    num_inside_dura = defaultdict(int)
    electrodes_num = defaultdict(int)
    groups_inside = defaultdict(dict)
    for elc_ind, elc_name in enumerate(electrodes):
        elc_group, elc_num = utils.elec_group_number(elc_name, bipolar)
        in_dural = in_dural_hemis['rh'][elc_ind] or in_dural_hemis['lh'][elc_ind]
        groups_inside[elc_group][elc_num] = in_dural
        electrodes_num[elc_group] += 1
    for elc_ind, elc_name in enumerate(electrodes):
        elc_group, elc_num = utils.elec_group_number(elc_name, bipolar)
        in_dural = groups_inside[elc_group][elc_num]
        if in_dural:
            num_inside_dura[elc_group] += 1
        else: # Check if the elc is inside the brain
            if any([groups_inside[elc_group][k] for k in range(elc_num + 1,  electrodes_num[elc_group])]):
                num_inside_dura[elc_group] += 1
    print('Saving results to {}'.format(output_fname))
    with open(output_fname, 'w') as output_file:
        for group, in_dura_num in num_inside_dura.items():
            output_str = '{}, {}, {}'.format(group, in_dura_num, electrodes_num[group])
            output_file.write('{}\n'.format(output_str))
            print(output_str)
    return op.isfile(output_fname)


# @pu.tryit_ret_bool
# def sort_electrodes_groups(subject, bipolar, all_in_hemi='', ask_for_hemis=False, do_plot=True):
#     from sklearn.decomposition import PCA
#     electrodes, pos = read_electrodes_file(subject, bipolar)
#     first_electrodes, first_pos, elc_pos_groups = find_first_electrode_per_group(electrodes, pos, bipolar)
#     if all_in_hemi in ['rh', 'lh']:
#         sorted_groups = {}
#         sorted_groups[all_in_hemi] = list(elc_pos_groups.keys())
#         sorted_groups[utils.other_hemi(all_in_hemi)] = []
#     else:
#         pca = PCA(n_components=2)
#         pca.fit(first_pos)
#         if pca.explained_variance_.shape[0] == 1 or ask_for_hemis:
#             # Can't find hemis via PAC, just ask the user
#             sorted_groups = dict(rh=[], lh=[])
#             print("Please set the hemi (r/l) for the following electrodes' groups:")
#             for group in elc_pos_groups.keys():
#                 inp = input('{}: '.format(group))
#                 while inp not in ['r', 'l']:
#                     print('Wrong hemi value!')
#                     inp = input('{}: '.format(group))
#                 hemi = '{}h'.format(inp)
#                 sorted_groups[hemi].append(group)
#         else:
#             transformed_pos = pca.transform(pos)
#             # transformed_pos_3d = PCA(n_components=3).fit(first_pos).transform(pos)
#             transformed_first_pos = pca.transform(first_pos)
#             groups_hemi = find_groups_hemi(electrodes, transformed_pos, bipolar)
#             sorted_groups = sort_groups(first_electrodes, transformed_first_pos, groups_hemi, bipolar)
#     print(sorted_groups)
#     utils.save(sorted_groups, op.join(MMVT_DIR, subject, 'electrodes', 'sorted_groups.pkl'))
#     if do_plot:
#         # utils.plot_3d_scatter(pos, names=electrodes.tolist(), labels=first_electrodes.values())
#         # electrodes_3d_scatter_plot(pos, first_pos)
#         first_electrodes_names = list(first_electrodes.values())
#         utils.plot_2d_scatter(transformed_first_pos, names=first_electrodes_names)
#         # utils.plot_2d_scatter(transformed_pos, names=electrodes.tolist(), labels=first_electrodes_names)


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


def fix_mismatches(edf_raw, channels_names_mismatches):
    for mismatch in channels_names_mismatches:
        if '=' not in mismatch:
            continue
        gr1, gr2 = mismatch.split('=')
        inds1 = np.array([i for i, l in enumerate(edf_raw.ch_names) if l.startswith(gr1)])
        inds2 = np.array([i for i, l in enumerate(edf_raw.ch_names) if l.startswith(gr2)])
        for ind, ch_ind in enumerate(inds1):
            edf_raw.ch_names[ch_ind] = edf_raw.info['chs'][ch_ind]['ch_name'] = '{}{}'.format(gr2, ind + 1)
        for ind, ch_ind in enumerate(inds2):
            edf_raw.ch_names[ch_ind] = edf_raw.info['chs'][ch_ind]['ch_name'] = '{}{}'.format(gr1, ind + 1)


def plot_electrodes(subject, edf_raw, data_channels, ref_ind=-1):
    # inds = np.array([i for i, l in enumerate(edf_raw.ch_names) if l.startswith('LMF')])
    # data_channels = data_channels[inds]
    N = len(data_channels)
    data, times = edf_raw[:N]
    if ref_ind > -1:
        ref_data, _ = edf_raw[ref_ind]
        data -= np.tile(ref_data, (data.shape[0], 1))
    amp = np.diff(utils.calc_min_max(data, norm_percs=[1, 99]))[0] * 2
    amp_diff = np.arange(0, amp * N, amp)
    amp_diff = np.tile(amp_diff, (data.shape[1], 1)).T
    plt.plot(times, (data + amp_diff).T)
    plt.savefig(op.join(MMVT_DIR, subject, 'figures', 'electrodes.png'))
    plt.legend(data_channels)
    plt.show()


def fix_channles_names(edf_raw, data_channels):
    channels_groups = set([utils.elec_group(l) for l in data_channels])
    ch_groups = set([utils.elec_group(l) for l in edf_raw.ch_names])
    fix_dict = {g: [c for c in channels_groups if g.lower() == c.lower()][0] for g in ch_groups if
        g.lower() in [c.lower() for c in channels_groups]}
    fix_dict = {k: v for k, v in fix_dict.items() if k != v}
    if len(fix_dict) == 0:
        return
    for ch_ind, c in enumerate(edf_raw.ch_names):
        gr, num = utils.elec_group_number(c)
        if gr not in fix_dict:
            continue
        new_name = '{}{}'.format(fix_dict[gr], num)
        edf_raw.ch_names[ch_ind] = edf_raw.info['chs'][ch_ind]['ch_name'] = new_name


def create_raw_data_from_edf(subject, args, stat=STAT_DIFF, electrodes_type=None, do_plot=False, overwrite=False):
    fol = op.join(MMVT_DIR, subject, 'electrodes')
    meta_fname_exist = len(glob.glob(op.join(fol, 'electrodes_meta_data*.npz'))) > 0
    data_fname_exist = len(glob.glob(op.join(fol, 'electrodes_data*.npy'))) > 0
    if meta_fname_exist and data_fname_exist and not overwrite:
        return True

    edf_fname, _ = utils.locating_file(args.raw_fname, '*.edf', op.join(ELECTRODES_DIR, subject))
    if not op.isfile(edf_fname):
        raise Exception('The EDF file cannot be found in {}!'.format(edf_fname))
    edf_raw = mne.io.read_raw_edf(edf_fname, preload=args.preload)

    # edf_raw.plot(n_channels=1)
    fs = float(edf_raw.info['sfreq'])
    if args.preload and args.remove_power_line_noise:
        power_line_freqs = np.arange(args.power_line_freq, args.power_line_freq * 4 + 1, args.power_line_freq)
        power_line_freqs = [f for f in power_line_freqs if f < fs / 2] # must be less than the Nyquist frequency
        edf_raw.notch_filter(power_line_freqs, notch_widths=args.power_line_notch_widths)
    if not (args.lower_freq_filter is None and args.upper_freq_filter is None):
        edf_raw.filter(args.lower_freq_filter, args.upper_freq_filter)
    dt = (edf_raw.times[1] - edf_raw.times[0])
    hz = int(1/ dt)
    # T = edf_raw.times[-1] # sec
    data_channels, all_pos = read_electrodes_file(subject, bipolar=False, electrodes_type=electrodes_type)
    fix_mismatches(edf_raw, args.channels_names_mismatches)
    fix_channles_names(edf_raw, data_channels)
    # live_channels = find_live_channels(edf_raw, hz)
    # no_stim_channels = get_data_channels(edf_raw)
    # data_channels_lower = [l.lower() for l in data_channels]
    # ch_names_lower = [l.lower() for l in edf_raw.ch_names]

    channels_tup = [(ind, ch) for ind, ch in enumerate(edf_raw.ch_names) if ch in data_channels]
    channels_indices = [c[0] for c in channels_tup]
    labels = [c[1] for c in channels_tup]
    conditions = [c['name'] for c in args.conditions]
    # pos = all_pos[np.array(channels_indices)]
    #
    # from scipy.spatial import distance
    # dists = distance.cdist(pos, pos, 'euclidean')
    # dists_output_fname = 'electrodes{}_dists.npy'.format('_bipolar' if args.bipolar else '')
    # dists_output_fname = op.join(MMVT_DIR, subject, 'electrodes', dists_output_fname)
    # np.save(dists_output_fname, dists)

    ref_ind = edf_raw.ch_names.index(args.ref_elec) if args.ref_elec != '' else -1
    if do_plot:
        plot_electrodes(subject, edf_raw, data_channels, ref_ind)

    data = None
    cond_id = 0
    baseline = None
    for cond in args.conditions:
        if 'from_t' in cond:
            cond_data, times = edf_raw[channels_indices, int(cond['from_t']*hz):int(cond['to_t']*hz)]
        else:
            cond_data, times = edf_raw[channels_indices, :]
        if args.ref_elec != '':
            ref_data, _ = edf_raw[ref_ind, int(cond['from_t'] * hz):int(cond['to_t'] * hz)]
            cond_data -= np.tile(ref_data, (cond_data.shape[0], 1))
        if not args.preload and args.remove_power_line_noise:
            notch_filter(cond_data, fs, np.arange(60, 241, 60), notch_widths=args.power_line_notch_widths,
                         copy=False, n_jobs=args.n_jobs)
        if cond['name'] != 'baseline':
            C, T = cond_data.shape
            if data is None:
                conds_num = len(args.conditions)
                if 'baseline' in conditions:
                    conds_num = conds_num -1
                data = np.zeros((C, T, conds_num))
        if cond['name'] == 'baseline':
            baseline_fname = op.join(fol, 'electrodes{}_baseline.npy'.format('_bipolar' if args.bipolar else ''))
            baseline = cond_data
            np.save(baseline_fname, cond_data)
            baseline_mean = np.mean(cond_data, 1)
        else:
            # import rdp
            # t = range(cond_data.shape[1])
            # x = rdp.rdp(list(zip(t, cond_data[0])), epsilon=0.2)
            data[:, :, cond_id] = cond_data
            cond_id = cond_id + 1

    if do_plot:
        plt.psd(data, Fs=hz)
        plt.show()
    if 'baseline' in conditions and args.remove_baseline:
        for c in range(data.shape[2]):
            data[:, :, c] -= np.tile(baseline_mean, (T, 1)).T
            if args.calc_zscore:
                data[:, :, c] /= np.tile(baseline.std(axis=1), (T, 1)).T
        conditions.remove('baseline')
    if args.normalize_data:
        data = utils.normalize_data(data, norm_by_percentile=False)
    data *= args.factor
    stat_data = calc_stat_data(data, STAT_DIFF)

    if args.moving_average_win_size > 0:
        output_fname = op.join(
            fol, 'electrodes_data_{}.npz'.format('_{}'.format(STAT_NAME[stat] if len(conditions) > 1 else '')))
        stat_data_mv = utils.moving_avg(stat_data, args.moving_average_win_size)
        np.savez(output_fname, data=data, stat=stat_data_mv, names=labels, conditions=conditions, times=times) # colors=colors_mv
        if args.bipolar:
            return data_electrodes_to_bipolar(subject, electrodes_type)
        else:
            return op.isfile(output_fname)
    else:
        meta_fname = op.join(fol, 'electrodes_meta_data{}.npz'.format(
            '_{}'.format(STAT_NAME[stat]) if len(conditions) > 1 else ''))
        data_fname = op.join(fol, 'electrodes_data{}.npy'.format(
            '_{}'.format(STAT_NAME[stat]) if len(conditions) > 1 else ''))
        np.savez(meta_fname, names=labels, conditions=conditions, times=times)
        np.save(data_fname, data)
        # return op.isfile(data_fname) and op.isfile(meta_fname) and data_electrodes_to_bipolar(subject)
        if args.bipolar:
            return data_electrodes_to_bipolar(subject, electrodes_type)
        else:
            return op.isfile(data_fname) and op.isfile(meta_fname)


def data_electrodes_to_bipolar(subject, electrodes_type=None):
    fol = op.join(MMVT_DIR, subject, 'electrodes')
    meta_data = np.load(op.join(fol, 'electrodes_meta_data.npz'))
    data = np.load(op.join(fol, 'electrodes_data.npy'))
    channels, conditions, times = meta_data['names'], meta_data['conditions'], meta_data['times']
    _, _, bipolar_channels = convert_electrodes_pos(subject, bipolar=True, electrodes_type=electrodes_type)
    data_bipolar = np.zeros((len(bipolar_channels), data.shape[1], len(conditions)))
    for cond_ind in range(len(conditions)):
        for ind, bipolar_channel in enumerate(bipolar_channels):
            ind1, ind2 = (np.where(channels == cha)[0][0] for cha in bipolar_channel.split('-'))
            data_bipolar[ind, :, cond_ind] = data[ind2, :, cond_ind] - data[ind1, :, cond_ind]
    meta_fname = op.join(fol, 'electrodes_bipolar_meta_data.npz')
    data_fname = op.join(fol, 'electrodes_bipolar_data.npy')
    np.savez(meta_fname, names=bipolar_channels, conditions=conditions, times=times)
    np.save(data_fname, data_bipolar)
    return op.isfile(meta_fname) and op.isfile(data_fname)


def calc_seizure_times(start_time, seizure_onset, seizure_end, baseline_onset, baseline_end,
                       time_format='%H:%M:%S'):
    start_time = datetime.strptime(start_time, time_format)
    seizure_time = datetime.strptime(seizure_onset, time_format) - start_time
    seizure_start_t = seizure_time.seconds # - seizure_onset_time
    seizure_end_t = (datetime.strptime(seizure_end, time_format) - start_time).seconds
    baseline_start_t = (datetime.strptime(baseline_onset, time_format) - start_time).seconds
    baseline_end_t = (datetime.strptime(baseline_end, time_format) - start_time).seconds
    return seizure_start_t, seizure_end_t, baseline_start_t, baseline_end_t


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


def get_data_and_meta(subject):
    fol = op.join(MMVT_DIR, subject, 'electrodes')
    meta_fnames = glob.glob(op.join(fol, 'electrodes_meta_data*.npz'))
    data_fnames = glob.glob(op.join(fol, 'electrodes_data*.npy'))
    if len(meta_fnames) != 1 and len(data_fnames) != 1:
        print('Couldn\'t find the data and met')
        return None, None
    else:
        # names=labels, conditions=conditions, times=times
        meta_data = utils.Bag(np.load(meta_fnames[0])) 
        data = np.load(data_fnames[0])
        return data, meta_data
    
    
def calc_epochs_power_spectrum(subject, windows_length, windows_shift, epochs_num=-1, overwrite=False, n_jobs=-1):
    output_fname = op.join(MMVT_DIR, subject, 'electrodes', 'power_spectrum.npz')
    if op.isfile(output_fname) and not overwrite:
        return True
    
    data, meta_data = get_data_and_meta(subject)
    if data is None or meta_data is None:
        return False

    power_spectrum = None
    data = data.squeeze()
    sfreq = 1 / (meta_data.times[1] - meta_data.times[0])
    T = data.shape[1] / sfreq #meta_data.times[-1] - meta_data.times[0]
    electrodes_num = data.shape[0]
    if epochs_num == -1:
        epochs_num = math.floor((T - windows_length) / windows_shift + 1)
    demi_epochs = np.zeros((epochs_num, 2), dtype=np.uint32)
    for win_ind in range(epochs_num):
        demi_epochs[win_ind] = [int(win_ind * windows_shift * sfreq),
                                int(sfreq * (win_ind * windows_shift + windows_length))]
    now = time.time()
    fmin, fmax = 0., 100.
    bandwidth = 2.  # bandwidth of the windows in Hz

    for epoch_ind, demi_epoch in enumerate(demi_epochs):
        if epoch_ind == epochs_num:
            break
        utils.time_to_go(now, epoch_ind, len(demi_epochs), runs_num_to_print=10)
        psds, freqs = mne.time_frequency.psd_array_multitaper(
            data[:, demi_epoch[0]:demi_epoch[1]], sfreq, fmin, fmax, bandwidth, n_jobs=n_jobs)
        if power_spectrum is None:
            power_spectrum = np.empty((epochs_num, electrodes_num, len(freqs)))
        power_spectrum[epoch_ind] = psds
    np.savez(output_fname, power_spectrum=power_spectrum, frequencies=freqs)
    return op.isfile(output_fname)


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


def get_electrodes_groups(subject, bipolar, electrodes_type=None):
    electrodes, _ = read_electrodes_file(subject, bipolar, electrodes_type=electrodes_type)
    groups = set()
    for elc in electrodes:
        groups.add(utils.elec_group(elc, bipolar))
    return electrodes, groups


@pu.tryit_ret_bool
def create_electrodes_groups_coloring(subject, bipolar, electrodes_type=None, coloring_fname=''):
    if coloring_fname == '':
        coloring_fname = 'electrodes_{}groups_coloring.csv'.format('bipolar_' if bipolar else '')
    electrodes, groups = get_electrodes_groups(subject, bipolar, electrodes_type=electrodes_type)
    colors_rgb_and_names = cu.get_distinct_colors_and_names(len(groups) - 1, boynton=True)
    group_colors = dict()
    coloring_fname = op.join(MMVT_DIR, subject, 'coloring', coloring_fname)
    coloring_names_fname = op.join(MMVT_DIR, subject, 'coloring', 'electrodes_groups_coloring_names.txt')
    with open(coloring_names_fname, 'w') as colors_names_file:
        for group, (color_rgb, color_name) in zip(groups, colors_rgb_and_names):
            if 'ref' in group.lower():
                continue
            group_colors[group] = color_rgb
            colors_names_file.write('{},{}\n'.format(group, color_name))
    with open(coloring_fname, 'w') as colors_file:
        for elc in electrodes:
            if 'ref' in elc.lower():
                continue
            elc_group = utils.elec_group(elc, bipolar)
            colors_file.write('{},{},{},{}\n'.format(elc, *group_colors[elc_group]))
    return op.isfile(coloring_fname)


def get_electrodes_labeling(subject, blender_root, atlas, bipolar=False, error_radius=3, elec_length=4,
                            electrodes_type=None, other_fname='', overwrite_ela=False):
    if other_fname == '':
        # We remove the 'all_rois' and 'stretch' for the name!
        electrode_labeling_fname = op.join(blender_root, subject, 'electrodes',
            '{}_{}_electrodes_cigar_r_{}_l_{}{}.pkl'.format(subject, atlas, error_radius, elec_length,
            '_bipolar' if bipolar else ''))
    else:
        electrode_labeling_fname = other_fname
    if not op.isfile(electrode_labeling_fname):
        run_ela(subject, atlas, bipolar, overwrite_ela, error_radius, elec_length, electrodes_type)
    if op.isfile(electrode_labeling_fname):
        labeling = utils.load(electrode_labeling_fname)
        return labeling, electrode_labeling_fname
    else:
        print("Can't find the electrodes' labeling file in {}!".format(electrode_labeling_fname))
        return None, None


@pu.tryit_ret_bool
def create_electrodes_labeling_coloring(subject, bipolar, atlas, good_channels=None, error_radius=3, elec_length=4,
        overwrite_ela=False, p_threshold=0.05, legend_name='', coloring_fname='', electrodes_type=None):
    # elecs_names, elecs_coords = read_electrodes_file(subject, bipolar)
    elecs_probs, electrode_labeling_fname = get_electrodes_labeling(
        subject, MMVT_DIR, atlas, bipolar, error_radius, elec_length, electrodes_type=electrodes_type,
        overwrite_ela=overwrite_ela)
    if elecs_probs is None:
        print('No electrodes labeling file!')
        return
    if electrode_labeling_fname != op.join(MMVT_DIR, subject, 'electrodes',
            op.basename(electrode_labeling_fname)):
        shutil.copy(electrode_labeling_fname, op.join(MMVT_DIR, subject, 'electrodes',
            op.basename(electrode_labeling_fname)))
    most_probable_rois = get_most_probable_rois(elecs_probs, p_threshold, good_channels)
    rois_colors_rgbs, rois_colors_names = get_rois_colors(subject, atlas, most_probable_rois)
    # save_rois_colors_legend(subject, rois_colors_rgbs, bipolar, legend_name)
    utils.make_dir(op.join(MMVT_DIR, subject, 'coloring'))
    if coloring_fname == '':
        coloring_fname = 'electrodes{}_{}_coloring.csv'.format('_bipolar' if bipolar else '', atlas)
    coloring_fol = op.join(MMVT_DIR, subject, 'coloring')
    coloring_fname =  op.join(coloring_fol, coloring_fname)
    colors_names_fname = op.join(coloring_fol, 'electrodes{}_colors_names.txt'.format('_bipolar' if bipolar else ''))
    elec_names_rois_colors = defaultdict(list)
    with open(coloring_fname, 'w') as colors_rgbs_file, open(colors_names_fname, 'w') as colors_names_file:
        # colors_csv_writer = csv.writer(colors_csv_file, delimiter=',')
        for elec_probs in elecs_probs:
            elec_name = elec_probs['name']
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
    return op.isfile(op.join(coloring_fol, 'electrodes_report.txt'))


def remove_bad_channels(labels, data, bad_channels):
    bad_indices = [labels.index(bad) for bad in bad_channels]
    for bad_electrode in bad_channels:
        labels.remove(bad_electrode)
    data = np.delete(data, bad_indices, axis=0)
    return labels, data


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
    rois_colors_rgbs, rois_colors_names = OrderedDict(), OrderedDict()

    # lables_colors_rgbs_fname = op.join(MMVT_DIR, subject, 'coloring', 'labels_{}_coloring.csv'.format(atlas))
    # lables_colors_names_fname = op.join(MMVT_DIR, subject, 'coloring', 'labels_{}_colors_names.txt'.format(atlas))
    # labels_colors_exist = op.isfile(lables_colors_rgbs_fname) # and op.isfile(lables_colors_names_fname)
    # if not labels_colors_exist:
    #     print('No labels coloring file!')
    # else:
    #     labels_colors_rgbs = np.genfromtxt(lables_colors_rgbs_fname, dtype=str, delimiter=',')
    #     # labels_colors_names = np.genfromtxt(lables_colors_names_fname, dtype=str, delimiter=',')
    for roi in not_white_rois:
        # if labels_colors_exist:
        #     roi_inds = np.where(labels_colors_rgbs[:, 0] == '{}-rh'.format(roi))[0]
        #     if len(roi_inds) > 0:
        #         color_rgb = labels_colors_rgbs[roi_inds][0, 1:].tolist()
        #         # color_name = labels_colors_names[roi_inds][0, 1]
        #     else:
        #         color_rgb, color_name = next(colors)
        # else:
        color_rgb, color_name = next(colors)
        rois_colors_rgbs[roi], rois_colors_names[roi] = color_rgb, color_name
    for white_roi in white_rois:
        rois_colors_rgbs[white_roi], rois_colors_names[white_roi] = cu.name_to_rgb('white').tolist(), 'white'
    return rois_colors_rgbs, rois_colors_names


@utils.tryit()
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
    figlegend.savefig(op.join(MMVT_DIR, subject, 'coloring', legend_name))


def transform_electrodes_to_mni(subject, args):
    from src.utils import freesurfer_utils as fu
    elecs_names, elecs_coords = read_electrodes_file(subject, args.bipolar, electrodes_type=args.electrodes_type)
    elecs_coords_mni = fu.transform_subject_to_mni_coordinates(subject, elecs_coords, SUBJECTS_DIR)
    if elecs_coords_mni is not None:
        save_electrodes_coords(subject, elecs_names, elecs_coords_mni, args.good_channels, args.bad_channels, '_mni')
        return True
    else:
        return False


def transform_electrodes_to_subject(subject, args):
    from src.utils import freesurfer_utils as fu
    elecs_names, elecs_coords = read_electrodes_file(subject, args.bipolar, 'mni', electrodes_type=args.electrodes_type)
    elecs_coords_to_subject = fu.transform_subject_to_subject_coordinates(
        args.trans_from_subject, subject, elecs_coords, SUBJECTS_DIR)
    electrodes_fname = save_electrodes_coords(
        subject, elecs_names, elecs_coords_to_subject, args.good_channels, args.bad_channels,
        '_from_{}'.format(args.trans_from_subject))
    print('Writing {}'.format(electrodes_fname))
    csv_fname = op.join(MMVT_DIR, subject, 'electrodes', '{}_RAS_from_{}.csv'.format(subject, args.trans_from_subject))
    print('Save also to csv {}'.format(csv_fname))
    csv_data = np.hstack((np.array(elecs_names).reshape((len(elecs_names), 1)),
                          elecs_coords_to_subject)).astype(np.str)
    np.savetxt(csv_fname, csv_data, fmt='%s', delimiter=',')
    return op.isfile(electrodes_fname)


def save_electrodes_coords(subject, elecs_names, elecs_coords, good_channels=None, bad_channels=None, fname_postfix=''):
    good_elecs_names, good_elecs_coords = [], []
    for elec_name, elec_coord in zip(elecs_names, elecs_coords):
        if (not good_channels or elec_name in good_channels) and (not bad_channels or elec_name not in bad_channels):
            good_elecs_names.append(elec_name)
            good_elecs_coords.append(elec_coord)
    good_elecs_coords = np.array(good_elecs_coords)
    electrodes_fname = save_electrodes_file(
        subject, args.bipolar, good_elecs_names, good_elecs_coords, fname_postfix)
    if 'mni' in fname_postfix:
        output_file_name = op.split(electrodes_fname)[1]
        utils.make_dir(op.join(MMVT_DIR, 'colin27', 'electrodes'))
        blender_file = op.join(MMVT_DIR, 'colin27', 'electrodes', output_file_name.replace(fname_postfix, ''))
        shutil.copyfile(electrodes_fname, blender_file)
    return electrodes_fname


def snap_electrodes_to_dural(subject, snap_all=False, overwrite_snap=False, electrodes_type=None):
    from src.utils import args_utils as au
    # todo: of over all the electrodes, check the groups, and run the snap in a loop only for the grids
    groups_pos_dict = defaultdict(list)
    all_names, all_pos = read_electrodes_file(subject, False, electrodes_type=electrodes_type)
    for elc_name, elc_pos in zip(all_names, all_pos):
        group = utils.elec_group(elc_name, False)
        groups_pos_dict[group].append(elc_pos)
    snap_ret = True
    for group in groups_pos_dict.keys():
        pos = np.array(groups_pos_dict[group])
        if not snap_all:
            do_snap = au.is_true(input('Do you want to snap {}? '.format(group)))
        else:
            do_snap = True
        if do_snap:
            snap_ret = snap_ret and snap_electrodes_to_surface(
                subject, pos, group, SUBJECTS_DIR, overwrite=overwrite_snap)
    if snap_ret:
        read_snapped_electrodes(subject, electrodes_type, overwrite_snap)
    return snap_ret


def read_snapped_electrodes(subject, electrodes_type=None, overwrite=False):
    output_fname = op.join(MMVT_DIR, subject, 'electrodes', 'electrodes_snap_positions.npz')
    if op.isfile(output_fname) and not overwrite:
        return True
    groups_names_dict = defaultdict(list)
    all_names, all_pos = read_electrodes_file(subject, False, electrodes_type=electrodes_type)
    for elc_name in all_names:
        group = utils.elec_group(elc_name, False)
        groups_names_dict[group].append(elc_name)
    snap_names, snap_pos = [], None
    for group_name in groups_names_dict.keys():
        snap_fname = op.join(MMVT_DIR, subject, 'electrodes', '{}_snap_electrodes.npz'.format(group_name))
        if not op.isfile(snap_fname):
            print('The snap electrodes for group {} couldn\'t be found! {}'.format(group_name, snap_fname))
            continue
        group_snap_dict = np.load(snap_fname)
        snap_to_dura_pos = group_snap_dict['snapped_electrodes']
        # snap_to_pial_pos = group_snap_dict['snapped_electrodes_pial']
        snap_names.extend(groups_names_dict[group_name])
        snap_pos = snap_to_dura_pos if snap_pos is None else np.concatenate((snap_pos, snap_to_dura_pos))
    np.savez(output_fname, pos=snap_pos, names=snap_names)
    return op.isfile(output_fname)


def snap_electrodes_to_surface(subject, elecs_pos, grid_name, subjects_dir,
                               max_steps=40000, giveup_steps=10000,
                               init_temp=1e-3, temperature_exponent=1,
                               deformation_constant=1., overwrite=False):
    '''
    Transforms electrodes from surface space to positions on the surface
    using a simulated annealing "snapping" algorithm which minimizes an
    objective energy function as in Dykstra et al. 2012

    Parameters
    ----------
    electrodes : List(Electrode)
        List of electrodes with the surf_coords attribute filled. Caller is
        responsible for filtering these into grids if desired.
    subjects_dir : Str | None
        The freesurfer subjects_dir. If this is None, it is assumed to be the
        $SUBJECTS_DIR environment variable. Needed to access the dural
        surface.
    subject : Str | None
        The freesurfer subject. If this is None, it is assumed to be the
        $SUBJECT environment variable. Needed to access the dural surface.
    max_steps : Int
        The maximum number of steps for the Simulated Annealing algorithm.
        Adding more steps usually causes the algorithm to take longer. The
        default value is 40000. max_steps can be smaller than giveup_steps,
        in which case giveup_steps is ignored
    giveup_steps : Int
        The number of steps after which, with no change of objective function,
        the algorithm gives up. A higher value may cause the algorithm to
        take longer. The default value is 10000.
    init_temp : Float
        The initial annealing temperature. Default value 1e-3
    temperature_exponent : Float
        The exponentially determined temperature when making random changes.
        The value is Texp0 = 1 - Texp/H where H is max_steps
    deformation_constant : Float
        A constant to weight the deformation term of the energy cost. When 1,
        the deformation and displacement are weighted equally. When less than
        1, there is assumed to be considerable deformation and the spring
        condition is weighted more highly than the deformation condition.

    There is no return value. The 'snap_coords' attribute will be used to
    store the snapped locations of the electrodes
    '''

    fol = utils.make_dir(op.join(MMVT_DIR, subject, 'electrodes'))
    output_fname = op.join(fol, '{}_snap_electrodes.npz'.format(grid_name))
    if op.isfile(output_fname) and not overwrite:
        print('The output file {} is already exist! To overwrite use --overwrite_snap 1'.format(output_fname))
        return True
    print('Snapping {} electrodes'.format(grid_name))

    n = elecs_pos.shape[0]
    e_init = np.array(elecs_pos)
    snapped_electrodes = np.zeros(elecs_pos.shape)

    # first set the alpha parameter exactly as described in Dykstra 2012.
    # this parameter controls which electrodes have virtual springs connected.
    # this may not matter but doing it is fast and safe
    alpha = np.zeros((n, n))
    init_dist = cdist(e_init, e_init)

    neighbors = []

    k_nei = np.min([n, 6])
    for i in range(n):
        neighbor_vec = init_dist[:, i]
        # take 5 highest neighbors
        h5, = np.where(np.logical_and(neighbor_vec < np.sort(neighbor_vec)[k_nei - 1],
                                      neighbor_vec != 0))

        neighbors.append(h5)

    neighbors = np.squeeze(neighbors)

    # get distances from each neighbor pairing
    neighbor_dists = []
    for i in range(n):
        neighbor_dists.append(init_dist[i, neighbors[i]])

    neighbor_dists = np.hstack(neighbor_dists)

    # collect distance into histogram of resolution 0.2
    max = np.max(np.around(neighbor_dists))
    min = np.min(np.around(neighbor_dists))

    hist, _ = np.histogram(neighbor_dists, bins=int((max - min) / 2), range=(min, max))

    fundist = np.argmax(hist) * 2 + min + 1

    # apply fundist to alpha matrix
    alpha_tweak = 1.75

    for i in range(n):
        neighbor_vec = init_dist[:, i]
        neighbor_vec[i] = np.inf

        neighbors = np.where(neighbor_vec < fundist * alpha_tweak)

        if len(neighbors) > 5:
            neighbors = np.where(neighbor_vec < np.sort(neighbor_vec)[5])

        if len(neighbors) == 0:
            closest = np.argmin(neighbors)
            neighbors = np.where(neighbor_vec < closest * alpha_tweak)

        alpha[i, neighbors] = 1

        for j in range(i):
            if alpha[j, i] == 1:
                alpha[i, j] = 1
            if alpha[i, j] == 1:
                alpha[j, i] = 1

    # alpha is set, now do the annealing
    def energycost(e_new, e_old, alpha):
        n = len(alpha)

        dist_new = cdist(e_new, e_new)
        dist_old = cdist(e_old, e_old)

        H = 0
        for i in range(n):
            H += deformation_constant * float(cdist([e_new[i]], [e_old[i]]))
            for j in range(i):
                H += alpha[i, j] * (dist_new[i, j] - dist_old[i, j]) ** 2
        return H

    # load the dural surface locations
    ply_template = op.join(MMVT_DIR, subject, 'surf', '{hemi}.dural.ply')
    fs_template = op.join(SUBJECTS_DIR, subject, 'surf', '{hemi}.dural')
    if utils.both_hemi_files_exist(ply_template):
        lh_dura, _ = utils.read_ply_file(ply_template.format(hemi='lh'))
        rh_dura, _ = utils.read_ply_file(ply_template.format(hemi='rh'))
    elif utils.both_hemi_files_exist(fs_template):
        lh_dura, _ = nib.freesurfer.read_geometry(fs_template.format(hemi='lh'))
        rh_dura, _ = nib.freesurfer.read_geometry(fs_template.format(hemi='rh'))
    else:
        print('No dura can be found!')
        return False
    dura = np.vstack((lh_dura, rh_dura))

    max_deformation = 3
    deformation_choice = 50

    # adjust annealing parameters
    # H determines maximal number of steps
    H = max_steps
    # Texp determines the steepness of temperateure gradient
    Texp = 1 - temperature_exponent / H
    # T0 sets the initial temperature and scales the energy term
    T0 = init_temp
    # Hbrk sets a break point for the annealing
    Hbrk = giveup_steps

    h = 0;
    hcnt = 0
    lowcost = mincost = 1e6

    # start e-init as greedy snap to surface
    e_snapgreedy = dura[np.argmin(cdist(dura, e_init), axis=0)]

    e = np.array(e_snapgreedy).copy()
    emin = np.array(e_snapgreedy).copy()

    # the annealing schedule continues until the maximum number of moves
    while h < H:
        h += 1
        hcnt += 1
        # terminate if no moves have been made for a long time
        if hcnt > Hbrk:
            break

        # current temperature
        T = T0 * (Texp ** h)

        # select a random electrode
        e1 = np.random.randint(n)
        # transpose it with a *nearby* point on the surface

        # find distances from this point to all points on the surface
        dists = np.squeeze(cdist(dura, [e[e1]]))
        # take a distance within the minimum 5X

        mindist = np.sort(dists)[deformation_choice]
        candidate_verts, = np.where(dists < mindist * max_deformation)
        choice_vert = candidate_verts[np.random.randint(len(candidate_verts))]
        e_tmp = e.copy()
        e_tmp[e1] = dura[choice_vert]

        cost = energycost(e_tmp, e_init, alpha)

        if cost < lowcost or np.random.random() < np.exp(-(cost - lowcost) / T):
            e = e_tmp
            lowcost = cost

            if cost < mincost:
                emin = e
                mincost = cost
                print('step %i ... current lowest cost = %f' % (h, mincost))
                hcnt = 0

            if mincost == 0:
                break
        if h % 200 == 0:
            print('%s %s: step %i ... final lowest cost = %f' % (subject, grid_name, h, mincost))

    # return the emin coordinates
    for ind, loc in enumerate(emin):
        snapped_electrodes[ind] = loc

    lh_pia, _ = nib.freesurfer.read_geometry(op.join(subjects_dir, subject, 'surf', 'lh.pial'))
    rh_pia, _ = nib.freesurfer.read_geometry(op.join(subjects_dir, subject, 'surf', 'rh.pial'))
    pia = np.vstack((lh_pia, rh_pia))
    e_pia = np.argmin(cdist(pia, emin), axis=0)

    snapped_electrodes_pial = np.zeros(snapped_electrodes.shape)
    for ind, soln in enumerate(e_pia):
        snapped_electrodes_pial[ind] = pia[soln]

    np.savez(output_fname, snapped_electrodes=snapped_electrodes, snapped_electrodes_pial=snapped_electrodes_pial)
    print('The snap electrodes were saved to {}'.format(output_fname))
    return op.isfile(output_fname)


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
        if args.start_time != '':
            seizure_start_t, seizure_end_t, baseline_start_t, baseline_end_t = \
                calc_seizure_times(args.start_time, args.seizure_onset, args.seizure_end,
                                   args.baseline_onset, args.baseline_end, args.time_format)
            args.from_t = [baseline_start_t, seizure_start_t]
            args.to_t = [baseline_end_t, seizure_end_t]
        args.conditions = [dict(name='baseline', from_t=args.from_t[0], to_t=args.to_t[0]),
                      dict(name='seizure', from_t=args.from_t[1], to_t=args.to_t[1])]
        print(args.conditions)
        # conditions = [dict(name='baseline', from_t=12, to_t=16), dict(name='seizure', from_t=from_t, to_t=20)]
    elif args.task == 'rest':
        if args.rest_onset_time != '':
            start_time = datetime.strptime(args.start_time, args.time_format)
            rest_onset_time = datetime.strptime(args.rest_onset_time, args.time_format)
            args.from_t = [rest_onset_time - start_time]
            args.to_t = [datetime.strptime(args.end_time, args.time_format) - start_time]
            args.conditions = [dict(name='rest', from_t=args.from_t[0], to_t=args.to_t[0])]
        else:
            args.conditions = [dict(name='rest')]
    else:
        if isinstance(args.from_t, Iterable) and isinstance(args.to_t, Iterable):
            args.conditions = [dict(name=cond_name, from_t=from_t, to_t=to_t) for (cond_name, from_t, to_t) in
                          zip(args.conditions, args.from_t, args.to_t)]
    return args


def get_ras_file(subject, args):
    local_elecs_fol = utils.make_dir(op.join(SUBJECTS_DIR, subject, 'electrodes'))
    local_fname = op.join(local_elecs_fol, '{}_RAS.xlsx'.format(subject))
    if args.remote_ras_fol != '' and not op.isfile(local_fname):
        remote_ras_fol = utils.build_remote_subject_dir(args.remote_ras_fol, subject)
        remote_fnames = glob.glob(op.join(remote_ras_fol, '{}*RAS*.xlsx'.format(subject.upper())))
        # print('glob.glob({}):'.format(op.join(remote_ras_fol, '{}*RAS*.xlsx'.format(subject.upper()))))
        # print(remote_fnames)
        remote_fname = utils.select_one_file(remote_fnames)
        # remote_fname = op.join(remote_ras_fol, '{}_RAS.xlsx'.format(subject))
        if op.isfile(remote_fname):
            shutil.copyfile(remote_fname, local_fname)
    return op.isfile(local_fname)


def run_ela(subject, atlas, bipolar, overwrite=False, elc_r=3, elc_len=4, electrodes_type=None, n_jobs=-1):
    mmvt_code_fol = utils.get_mmvt_code_root()
    ela_code_fol = op.join(utils.get_parent_fol(mmvt_code_fol), 'electrodes_rois')
    if not op.isdir(ela_code_fol) or not op.isfile(op.join(ela_code_fol, 'find_rois', 'find_rois.py')):
        print("Can't find ELA folder!")
        return

    output_name = '{}_{}_electrodes_cigar_r_{}_l_{}{}.pkl'.format(
        subject, atlas, elc_r, elc_len, '_bipolar' if bipolar else '')
    output_fname = op.join(ela_code_fol, 'electrodes', output_name)
    mmvt_ela_fname = op.join(MMVT_DIR, subject, 'electrodes', output_name)
    if (op.isfile(output_fname) or op.isfile(mmvt_ela_fname)) and not overwrite:
        if not op.isfile(mmvt_ela_fname) and op.isfile(output_fname):
            shutil.copyfile(output_fname, mmvt_ela_fname)
        print('The model for {}, {} is already exist ({})'.format(subject, atlas, mmvt_ela_fname))
        return True

    import importlib
    import sys
    if ela_code_fol not in sys.path:
        sys.path.append(ela_code_fol)
    from find_rois import find_rois
    importlib.reload(find_rois)
    cmd_args = ['-s', subject, '-a', atlas, '-b', str(bipolar), '--n_jobs', str(n_jobs)]
    if electrodes_type is not None:
        cmd_args.extend(['--electrodes_type', electrodes_type])
    args = find_rois.get_args(cmd_args)
    find_rois.run_for_all_subjects(args)
    if not op.isfile(output_fname):
        return False
    else:
        shutil.copyfile(output_fname, mmvt_ela_fname)
        return True


def create_labels_around_electrodes(subject, bipolar=False, labels_fol_name='electrodes_labels',
        label_r=5, snap=False, sigma=1, electrodes_type=None, overwrite=False, n_jobs=4):
    names, pos = read_electrodes_file(subject, bipolar, snap=snap, electrodes_type=electrodes_type)
    labels_fol = utils.make_dir(op.join(SUBJECTS_DIR, subject, 'label', labels_fol_name))
    if op.isdir(labels_fol) and overwrite:
        shutil.rmtree(labels_fol)
    verts = {}
    for hemi in utils.HEMIS:
        verts[hemi], _ = utils.read_pial(subject, MMVT_DIR, hemi)
    electrodes_hemis = get_electrodes_hemis(subject, sigma=sigma, electrodes_type=electrodes_type)
    for elc_name, elc_pos in zip(names, pos):
        group = utils.elec_group(elc_name, bipolar)
        elc_hemi = get_group_hemi(group, electrodes_hemis)
        # label_name = '{}_label'.format(elc_name)
        new_label_fname = op.join(labels_fol, '{}-{}.label'.format(elc_name, elc_hemi))
        if not overwrite and op.isfile(new_label_fname):
            continue
        dists = cdist([elc_pos], verts[elc_hemi]).squeeze()
        elc_vert = np.argmin(dists)
        print('vert {} is located {} from {}'.format(elc_vert, dists[elc_vert], elc_name))
        lu.grow_label(subject, elc_vert, elc_hemi, elc_name, label_r, n_jobs, labels_fol)
    # ret = lu.labels_to_annot(subject, SUBJECTS_DIR, new_atlas_name, labels_fol, print_error=True)
    ret = lu.create_atlas_coloring(subject, labels_fol_name, n_jobs)
    print('Done with create_labels_around_electrodes')
    return ret


def call_main(args):
    return pu.run_on_subjects(args, main)


def main(subject, remote_subject_dir, args, flags):
    utils.make_dir(op.join(ELECTRODES_DIR, subject))
    utils.make_dir(op.join(MMVT_DIR, subject))
    utils.make_dir(op.join(MMVT_DIR, subject, 'electrodes'))
    utils.make_dir(op.join(MMVT_DIR, subject, 'coloring'))
    args = set_args(args)

    if utils.should_run(args, 'get_ras_file'):
        flags['get_ras_file'] = get_ras_file(subject, args)

    if utils.should_run(args, 'convert_electrodes_pos'):
        flags['convert_electrodes_pos'], _, _ = convert_electrodes_pos(
            subject, bipolar=args.bipolar, ras_xls_sheet_name=args.ras_xls_sheet_name,
            electrodes_type=args.electrodes_type)

    if utils.should_run(args, 'calc_dist_mat'):
        flags['calc_dist_mat'] = calc_dist_mat(subject, bipolar=args.bipolar)

    # if utils.should_run(args, 'sort_electrodes_groups'):
    #     flags['sort_electrodes_groups'] = sort_electrodes_groups(
    #         subject, args.bipolar, args.all_in_hemi, args.ask_for_hemis, do_plot=args.do_plot)

    if utils.should_run(args, 'find_electrodes_hemis'):
        flags['find_electrodes_hemis'] = find_electrodes_hemis(
            subject, args.bipolar, args.sigma, args.find_hemis_manual, args.electrodes_type)

    if utils.should_run(args, 'create_electrode_data_file') and not args.task is None:
        flags['create_electrode_data_file'] = create_electrode_data_file(
            subject, args.task, args.from_t_ind, args.to_t_ind, args.stat, args.conditions,
            args.bipolar, args.electrodes_names_field, args.moving_average_win_size,
            args.input_matlab_fname, args.input_type, args.field_cond_template)

    if utils.should_run(args, 'create_electrodes_labeling_coloring'):
        flags['create_electrodes_labeling_coloring'] = create_electrodes_labeling_coloring(
            subject, args.bipolar, args.atlas, overwrite_ela=args.overwrite_ela,
            electrodes_type=args.electrodes_type)

    if utils.should_run(args, 'create_electrodes_groups_coloring'):
        flags['create_electrodes_groups_coloring'] = create_electrodes_groups_coloring(
            subject, args.bipolar, args.electrodes_type, args.electrodes_groups_coloring_fname)

    # if utils.should_run(args, 'transform_electrodes_to_mni'):
    #     flags['transform_electrodes_to_mni'] = transform_electrodes_to_mni(
    #         subject, args)

    if 'transform_electrodes_to_subject' in args.function:
        flags['transform_electrodes_to_subject'] = transform_electrodes_to_subject(subject, args)

    if 'show_image' in args.function:
        legend_name = 'electrodes{}_coloring_legend.jpg'.format('_bipolar' if args.bipolar else '')
        flags['show_image'] = utils.show_image(op.join(MMVT_DIR, subject, 'coloring', legend_name))

    if 'create_raw_data_from_edf' in args.function:# and not args.task is None:
        flags['create_raw_data_from_edf'] = create_raw_data_from_edf(
            subject, args, overwrite=args.overwrite_raw_data, electrodes_type=args.electrodes_type)

    if 'electrodes_inside_the_dura' in args.function:
        flags['electrodes_inside_the_dura'] = check_how_many_electrodes_inside_the_dura(
            subject, args.sigma, args.bipolar, args.electrodes_type)

    if 'run_ela' in args.function:
        flags['run_ela'] = run_ela(
            subject, args.atlas, args.bipolar, args.overwrite_ela, args.error_radius, args.elc_length,
            args.electrodes_type, args.n_jobs)

    if 'create_labels_around_electrodes' in args.function:
        flags['create_labels_around_electrodes'] = create_labels_around_electrodes(
            subject, args.bipolar, args.electrodes_labels_fol_name, args.electrodes_label_r, args.snap,
            args.electrodes_types, args.overwrite_electrodes_labels, args.n_jobs)

    if 'calc_epochs_power_spectrum' in args.function:
        flags['calc_epochs_power_spectrum'] = calc_epochs_power_spectrum(
            subject, args.windows_length, args.windows_shift, args.epochs_num, args.overwrite_power_spectrum,
            args.n_jobs)

    if 'snap_electrodes_to_dural' in args.function:
        flags['snap_electrodes_to_dural'] = snap_electrodes_to_dural(
            subject, args.snap_all, args.overwrite_snap, args.electrodes_type)

    if 'read_snapped_electrodes' in args.function:
        flags['read_snapped_electrodes'] = read_snapped_electrodes(
            subject, args.electrodes_type, args.overwrite_snap)

    return flags


def read_cmd_args(argv=None):
    from src.utils import args_utils as au
    import argparse
    parser = argparse.ArgumentParser(description='MMVT electrodes preprocessing')
    parser.add_argument('-b', '--bipolar', help='bipolar', required=False, default=0, type=au.is_true)
    parser.add_argument('-t', '--task', help='task', required=False)
    parser.add_argument('--remote_ras_fol', help='remote_ras_fol', required=False, default='')
    parser.add_argument('--good_channels', help='good channels', required=False, default='', type=au.str_arr_type)
    parser.add_argument('--bad_channels', help='bad channels', required=False, default='', type=au.str_arr_type)
    parser.add_argument('--from_t', help='from_t', required=False, default='0', type=au.float_arr_type)# float) # was -500
    parser.add_argument('--to_t', help='to_t', required=False, default='0', type=au.float_arr_type) # was 2000
    parser.add_argument('--from_t_ind', help='from_t_ind', required=False, default=0, type=int) # was -500
    parser.add_argument('--to_t_ind', help='to_t_ind', required=False, default=-1, type=int) # was 2000
    parser.add_argument('--indices_shift', help='indices_shift', required=False, default=0, type=int) # was 1000
    parser.add_argument('--conditions', help='conditions', required=False, default='')
    parser.add_argument('--raw_fname', help='raw fname', required=False, default='no_raw_fname_given')
    parser.add_argument('--ref_elec', help='referece electrode', required=False, default='')
    parser.add_argument('--stat', help='stat', required=False, default=STAT_DIFF, type=int)
    parser.add_argument('--electrodes_names_field', help='electrodes_names_field', required=False, default='names')
    parser.add_argument('--moving_average_win_size', help='', required=False, default=0, type=int)
    parser.add_argument('--field_cond_template', help='', required=False, default='{}')
    parser.add_argument('--input_type', help='', required=False, default='ERP')
    parser.add_argument('--input_matlab_fname', help='', required=False, default='')
    parser.add_argument('--normalize_data', help='normalize_data', required=False, default=1, type=au.is_true)
    parser.add_argument('--preload', help='preload', required=False, default=1, type=au.is_true)
    parser.add_argument('--sigma', help='surf sigma', required=False, default=0, type=float)
    parser.add_argument('--find_hemis_manual', required=False, default=0, type=au.is_true)

    parser.add_argument('--start_time', help='', required=False, default='0:00:00')
    parser.add_argument('--end_time', help='', required=False, default='')
    parser.add_argument('--rest_onset_time', help='', required=False, default='')
    parser.add_argument('--seizure_onset', help='', required=False, default='')
    parser.add_argument('--seizure_end', help='', required=False, default='')
    parser.add_argument('--baseline_onset', help='', required=False, default='')
    parser.add_argument('--baseline_end', help='', required=False, default='')
    parser.add_argument('--time_format', help='', required=False, default='%H:%M:%S')
    parser.add_argument('--remove_power_line_noise', help='remove power line noise', required=False, default=1, type=au.is_true)
    parser.add_argument('--power_line_freq', help='power line freq', required=False, default=60, type=int)
    parser.add_argument('--power_line_notch_widths', help='notch_widths', required=False, default=None, type=au.float_or_none)
    parser.add_argument('--lower_freq_filter', help='', required=False, default=0, type=float)
    parser.add_argument('--upper_freq_filter', help='', required=False, default=0, type=float)
    parser.add_argument('--remove_baseline', help='remove_baseline', required=False, default=1, type=au.is_true)
    parser.add_argument('--factor', help='', required=False, default=1, type=float)
    parser.add_argument('--calc_zscore', help='calc_zscore', required=False, default=0, type=au.is_true)
    parser.add_argument('--channels_names_mismatches', required=False, default='', type=au.str_arr_type)
    parser.add_argument('--overwrite_raw_data', required=False, default=0, type=au.is_true)
    parser.add_argument('--grid_to_snap', required=False, default='G')
    parser.add_argument('--snap', required=False, default=0, type=au.is_true)
    parser.add_argument('--snap_all', required=False, default=1, type=au.is_true)
    parser.add_argument('--overwrite_snap', required=False, default=0, type=au.is_true)

    parser.add_argument('--electrodes_groups_coloring_fname', help='', required=False, default='electrodes_groups_coloring.csv')
    parser.add_argument('--ras_xls_sheet_name', help='ras_xls_sheet_name', required=False, default='')
    parser.add_argument('--all_in_hemi', help='', required=False, default='')
    parser.add_argument('--ask_for_hemis', help='', required=False, default=False, type=au.is_int)
    parser.add_argument('--do_plot', help='do plot', required=False, default=0, type=au.is_true)
    parser.add_argument('--trans_to_subject', help='transform electrodes coords to this subject', required=False, default='')
    parser.add_argument('--trans_from_subject', help='transform electrodes coords from this subject', required=False,
                        default='colin27')
    parser.add_argument('--overwrite_ela', required=False, default=0, type=au.is_true)
    parser.add_argument('--error_radius', help='error radius', required=False, default=3)
    parser.add_argument('--elc_length', help='elc length', required=False, default=4)

    parser.add_argument('--electrodes_labels_fol_name', required=False, default='electrodes_labels')
    parser.add_argument('--electrodes_label_r', required=False, default=5)
    parser.add_argument('--overwrite_electrodes_labels', help='', required=False, default=False, type=au.is_int)

    parser.add_argument('--windows_length', help='windows length', required=False, default=1000, type=int)
    parser.add_argument('--windows_shift', help='windows shift', required=False, default=500, type=int)
    parser.add_argument('--epochs_num', help='epoches nun', required=False, default=-1, type=int)
    parser.add_argument('--overwrite_power_spectrum', help='', required=False, default=False, type=au.is_true)
    parser.add_argument('--electrodes_type', help='', required=False, default=None)

    pu.add_common_args(parser)
    args = utils.Bag(au.parse_parser(parser, argv))
    args.ignore_missing = True
    for field in ['from_t', 'to_t']:
        if len(args[field]) == 1:
            args[field] = args[field][0]
    args.necessary_files = {'electrodes': ['{subject}_RAS.csv']}
    if args.lower_freq_filter == 0:
        args.lower_freq_filter = None
    if args.upper_freq_filter == 0:
        args.upper_freq_filter = None
    if args.bipolar:
        args.ref_elec = ''
    if args.overwrite_ela and 'run_ela' not in args.function:
        args.function.append('run_ela')
    # print(args)
    return args


if __name__ == '__main__':
    args = read_cmd_args()
    pu.run_on_subjects(args, main)
    print('finish!')
