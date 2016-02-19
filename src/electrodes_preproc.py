import numpy as np
import os.path as op
import shutil
import mne
import scipy.io as sio
import nibabel as nib
from itertools import product
import csv
from src import utils


LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
FREE_SURFER_HOME = utils.get_link_dir(LINKS_DIR, 'freesurfer', 'FREESURFER_HOME')
BLENDER_ROOT_DIR = op.join(LINKS_DIR, 'mmvt')

TASK_MSIT, TASK_ECR = range(2)
HEMIS = ['rh', 'lh']
STAT_AVG, STAT_DIFF = range(2)
STAT_NAME = {STAT_DIFF: 'diff', STAT_AVG: 'avg'}


def montage_to_npy(montage_file, output_file):
    sfp = mne.channels.read_montage(montage_file)
    np.savez(output_file, pos=np.array(sfp.pos), names=sfp.ch_names)


def electrodes_csv_to_npy(ras_file, output_file, bipolar=False, delimiter=','):
    data = np.genfromtxt(ras_file, dtype=str, delimiter=delimiter)
    # Check if the electrodes coordinates has a header
    try:
        header = data[0, 1:].astype(float)
    except:
        data = np.delete(data, (0), axis=0)

    pos = data[:, 1:].astype(float)
    if bipolar:
        names = []
        pos_biploar, pos_org = [], []
        for index in range(data.shape[0]-1):
            elc_group1, elc_num1 = elec_group_number(data[index, 0])
            elc_group2, elc_num12 = elec_group_number(data[index+1, 0])
            if elc_group1==elc_group2:
                names.append('{}-{}'.format(data[index+1, 0],data[index, 0]))
                pos_biploar.append(pos[index] + (pos[index+1]-pos[index])/2)
                pos_org.append([pos[index], pos[index+1]])
        pos = np.array(pos_biploar)
        pos_org = np.array(pos_org)
    else:
        names = data[1:, 0]
        pos_org = []
    if len(set(names))!=len(names):
        raise Exception('Duplicate electrodes names!')
    np.savez(output_file, pos=pos, names=names, pos_org=pos_org)


def elec_group_number(elec_name, bipolar=False):
    if bipolar:
        elec_name2, elec_name1 = elec_name.split('-')
        group, num1 = elec_group_number(elec_name1, False)
        _, num2 = elec_group_number(elec_name2, False)
        return group, num1, num2
    else:
        ind = np.where([int(s.isdigit()) for s in elec_name])[-1][0]
        num = int(elec_name[ind:])
        group = elec_name[:ind]
        return group, num


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


def read_electrodes_positions(subject, bipolar=False, copy_to_blender=True):
    electrodes_folder = op.join(SUBJECTS_DIR, subject, 'electrodes')
    csv_file = op.join(electrodes_folder, '{}_RAS.csv'.format(subject))
    if not op.isfile(csv_file):
        print('No electrodes coordinates file! {}'.format(csv_file))
        return None

    output_file_name = 'electrodes{}_positions.npz'.format('_bipolar' if bipolar else '')
    output_file = op.join(SUBJECTS_DIR, subject, 'electrodes', output_file_name)
    electrodes_csv_to_npy(csv_file, output_file, bipolar)
    if copy_to_blender:
        blender_file = op.join(BLENDER_ROOT_DIR, subject, output_file_name)
        shutil.copyfile(output_file, blender_file)
    return output_file


def create_electrode_data_file(subject, task, from_t, to_t, stat, conditions, bipolar, moving_average_win_size=0):
    input_file = op.join(SUBJECTS_DIR, subject, 'electrodes', 'electrodes_data.mat')
    output_file = op.join(BLENDER_ROOT_DIR, subject, 'electrodes{}_data_{}.npz'.format(
            '_bipolar' if bipolar else '', STAT_NAME[stat]))
    if task==TASK_ECR:
        read_electrodes_data_one_mat(input_file, conditions, stat, output_file,
            electrodeses_names_fiels='names', field_cond_template = '{}_ERP', from_t=from_t, to_t=to_t,
            moving_average_win_size=moving_average_win_size)# from_t=0, to_t=2500)
    elif task==TASK_MSIT:
        if bipolar:
            read_electrodes_data_one_mat(input_file, conditions, stat, output_file,
                electrodeses_names_fiels='electrodes_bipolar', field_cond_template = '{}_bipolar_evoked',
                from_t=from_t, to_t=to_t, moving_average_win_size=moving_average_win_size) #from_t=500, to_t=3000)
        else:
            read_electrodes_data_one_mat(input_file, conditions, stat, output_file,
                electrodeses_names_fiels='electrodes', field_cond_template = '{}_evoked',
                from_t=from_t, to_t=to_t, moving_average_win_size=moving_average_win_size) #from_t=500, to_t=3000)


def calc_colors(data, norm_by_percentile, norm_percs, threshold, cm_big, cm_small, flip_cm_big, flip_cm_small):
    data_max, data_min = utils.get_data_max_min(data, norm_by_percentile, norm_percs)
    data_minmax = max(map(abs, [data_max, data_min]))
    print('data minmax: {}'.format(data_minmax))
    colors = utils.mat_to_colors_two_colors_maps(data, threshold=threshold,
        x_max=data_minmax, x_min = -data_minmax, cm_big=cm_big, cm_small=cm_small,
        default_val=1, flip_cm_big=flip_cm_big, flip_cm_small=flip_cm_small)
    return colors
    # colors = utils.mat_to_colors(stat_data, -data_minmax, data_minmax, color_map)


def check_montage_and_electrodes_names(montage_file, electrodes_names_file):
    sfp = mne.channels.read_montage(montage_file)
    names = np.loadtxt(electrodes_names_file, dtype=np.str)
    names = set([str(e.strip()) for e in names])
    montage_names = set(sfp.ch_names)
    print(names-montage_names)
    print(montage_names-names)


def read_electrodes_data_one_mat(mat_file, conditions, stat, output_file_name, electrodeses_names_fiels,
        field_cond_template, from_t=0, to_t=None, norm_by_percentile=True, norm_percs=(3, 97), threshold=0,
        color_map='jet', cm_big='YlOrRd', cm_small='PuBu', flip_cm_big=False, flip_cm_small=True,
        moving_average_win_size=0):
    # load the matlab file
    d = sio.loadmat(mat_file)
    # get the labels names
    labels = d[electrodeses_names_fiels]
    #todo: change that!!!
    if len(labels) == 1:
        labels = [str(l[0]) for l in labels[0]]
    else:
        labels = [str(l[0][0]) for l in labels]
    # Loop for each condition
    for cond_id, cond_name in enumerate(conditions):
        field = field_cond_template.format(cond_name)
        # initialize the data matrix (electrodes_num x T x 2)
        if cond_id == 0:
            data = np.zeros((d[field].shape[0], to_t - from_t, 2))
        # times = np.arange(0, to_t*2, 2)
        # todo: Need to do some interpulation for the MEG
        cond_data = d[field] # [:, times]
        cond_data_downsample = utils.downsample_2d(cond_data, 2)
        data[:, :, cond_id] = cond_data_downsample[:, from_t:to_t]

    data = utils.normalize_data(data, norm_by_percentile, norm_percs)
    if stat == STAT_AVG:
        stat_data = np.squeeze(np.mean(data, axis=2))
    elif stat == STAT_DIFF:
        stat_data = np.squeeze(np.diff(data, axis=2))
    else:
        raise Exception('Wrong stat value!')

    if moving_average_win_size > 0:
        # data_mv[:, :, cond_id] = utils.downsample_2d(data[:, :, cond_id], moving_average_win_size)
        stat_data_mv = utils.moving_avg(stat_data, moving_average_win_size)

    colors = calc_colors(stat_data, norm_by_percentile, norm_percs, threshold, cm_big, cm_small,
                         flip_cm_big, flip_cm_small)
    colors_mv = calc_colors(stat_data_mv, norm_by_percentile, norm_percs, threshold, cm_big, cm_small,
                            flip_cm_big, flip_cm_small)
    # np.savez(output_file_name, data=data, names=labels, conditions=conditions, colors=colors)
    np.savez(output_file_name, data=data, stat=stat_data_mv, names=labels, conditions=conditions, colors=colors_mv)


def create_electrodes_volume_file(subject, electrodes_file, create_points_files=True, create_volume_file=False, way_points=False):
    elecs = np.load(electrodes_file)
    elecs_pos, names = elecs['pos'], elecs['names']

    if create_points_files:
        groups = set([name[:3] for name in names])
        freeview_command = 'freeview -v T1.mgz:opacity=0.3 aparc+aseg.mgz:opacity=0.05:colormap=lut ' + \
            ('-w ' if way_points else '-c ')
        for group in groups:
            postfix = 'label' if way_points else 'dat'
            freeview_command = freeview_command + group + postfix + ' '
            group_pos = np.array([pos for name, pos in zip(names, elecs_pos) if name[:3]==group])
            file_name = '{}.{}'.format(group, postfix)
            with open(op.join(BLENDER_ROOT_DIR, subject, 'freeview', file_name), 'w') as fp:
                writer = csv.writer(fp, delimiter=' ')
                if way_points:
                    writer.writerow(['#!ascii label  , from subject  vox2ras=Scanner'])
                    writer.writerow([len(group_pos)])
                    points = np.hstack((np.ones((len(group_pos), 1)) * -1, group_pos, np.ones((len(group_pos), 1))))
                    writer.writerows(points)
                else:
                    writer.writerows(group_pos)
                    writer.writerow(['info'])
                    writer.writerow(['numpoints', len(group_pos)])
                    writer.writerow(['useRealRAS', '1'])

    if create_volume_file:
        sig = nib.load(op.join(BLENDER_ROOT_DIR, subject, 'freeview', 'T1.mgz'))
        sig_data = sig.get_data()
        sig_header = sig.get_header()
        electrodes_positions = np.load(electrodes_file)['pos']
        data = np.zeros((256, 256, 256), dtype=np.int16)
        # positions_ras = np.array(utils.to_ras(electrodes_positions, round_coo=True))
        elecs_pos = np.array(elecs_pos, dtype=np.int16)
        for pos_ras in elecs_pos:
            for x, y, z in product(*([[d+i for i in range(-5,6)] for d in pos_ras])):
                data[z,y,z] = 1
        img = nib.Nifti1Image(data, sig_header.get_affine(), sig_header)
        nib.save(img, op.join(BLENDER_ROOT_DIR, subject, 'freeview', 'electrodes.nii.gz'))


def main(subject, bipolar, conditions, task, from_t_ind, to_t_ind, stat):
    # *) Read the electrodes data
    electrodes_file = read_electrodes_positions(subject, bipolar=bipolar)
    if electrodes_file:
        create_electrode_data_file(subject, task, from_t_ind, to_t_ind, stat, conditions, bipolar)
        create_electrodes_volume_file(subject, electrodes_file)
    # misc
    # check_montage_and_electrodes_names('/homes/5/npeled/space3/ohad/mg79/mg79.sfp', '/homes/5/npeled/space3/inaivu/data/mg79_ieeg/angelique/electrode_names.txt')


if __name__ == '__main__':
    subject = 'hc008'
    print('subject: {}'.format(subject))
    utils.make_dir(op.join(BLENDER_ROOT_DIR, subject))
    task = TASK_MSIT
    if task==TASK_ECR:
        conditions = ['happy', 'fear']
    elif task==TASK_MSIT:
        conditions = ['noninterference', 'interference']
    else:
        raise Exception('unknown task id!')
    from_t, to_t = -500, 2000
    from_t_ind, to_t_ind = 500, 3000
    bipolar = False
    stat = STAT_DIFF

    main(subject, bipolar, conditions, task, from_t_ind, to_t_ind, stat)
    print('finish!')
