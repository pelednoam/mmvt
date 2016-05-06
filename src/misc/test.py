import csv
import os.path as op
import shutil
from collections import defaultdict

import numpy as np

from src.utils import utils
from src.mmvt_addon.mmvt_utils import natural_keys
from src.preproc import electrodes_preproc as elec_pre
from src.utils import freesurfer_utils as fu

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
BLENDER_ROOT_DIR = op.join(LINKS_DIR, 'mmvt')
OUTPUT_DIR = '/cluster/neuromind/npeled/Documents/darpa_electrodes_csvs'
OUTPUT_DIR = '/home/noam/Documents/darpa_electrodes_csvs'


def prepare_darpa_csv(subject, bipolar, atlas, good_channels=None, groups_ordering=None, error_radius=3, elec_length=4, p_threshold=0.05):
    elecs_names, elecs_coords = elec_pre.read_electrodes_file(subject, bipolar)
    elecs_coords_mni = fu.transform_subject_to_mni_coordinates(subject, elecs_coords, SUBJECTS_DIR)
    save_electrodes_coords(elecs_names, elecs_coords_mni, good_channels)
    elecs_coords_mni_dic = {elec_name:elec_coord for (elec_name,elec_coord) in zip(elecs_names, elecs_coords_mni)}
    elecs_probs = utils.get_electrodes_labeling(subject, atlas, bipolar, error_radius, elec_length)
    assert(len(elecs_names) == len(elecs_coords_mni) == len(elecs_probs))
    most_probable_rois = elec_pre.get_most_probable_rois(elecs_probs, p_threshold, good_channels)
    rois_colors = elec_pre.get_rois_colors(most_probable_rois)
    elec_pre.save_rois_colors_legend(subject, rois_colors, bipolar)
    utils.make_dir(op.join(BLENDER_ROOT_DIR, 'colin27', 'coloring'))
    results = defaultdict(list)
    for elec_name, elec_probs in zip(elecs_names, elecs_probs):
        assert(elec_name == elec_probs['name'])
        if not good_channels is None and elec_name not in good_channels:
            continue
        group = get_elec_group(elec_name, bipolar)
        roi = elec_pre.get_most_probable_roi([*elec_probs['cortical_probs'], *elec_probs['subcortical_probs']],
            [*elec_probs['cortical_rois'], *elec_probs['subcortical_rois']], p_threshold)
        color = rois_colors[utils.get_hemi_indifferent_roi(roi)]
        results[group].append(dict(name=elec_name, roi=roi, color=color))
    with open(op.join(OUTPUT_DIR, '{}_electrodes_info.csv'.format(subject)), 'w') as csv_file, \
        open(op.join(BLENDER_ROOT_DIR, 'colin27', 'coloring','electrodes.csv'.format(subject)), 'w') as colors_csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        colors_csv_writer = csv.writer(colors_csv_file, delimiter=',')
        elec_ind = 0
        for group in groups_ordering:
            group_res = sorted(results[group], key=lambda x:natural_keys(x['name']))
            for res in group_res:
                csv_writer.writerow([elec_ind, res['name'], *elecs_coords_mni_dic[res['name']], res['roi'], *res['color']])
                colors_csv_writer.writerow([elec_name, *color])
                elec_ind += 1


def save_electrodes_coords(elecs_names, elecs_coords_mni, good_channels=None):
    good_elecs_names, good_elecs_coords_mni = [], []
    for elec_name, elec_coord_min in zip(elecs_names, elecs_coords_mni):
        if good_channels is None or elec_name in good_channels:
            good_elecs_names.append(elec_name)
            good_elecs_coords_mni.append(elec_coord_min)
    good_elecs_coords_mni = np.array(good_elecs_coords_mni)
    electrodes_mni_fname = elec_pre.save_electrodes_file(subject, bipolar, good_elecs_names, good_elecs_coords_mni, '_mni')
    output_file_name = op.split(electrodes_mni_fname)[1]
    blender_file = op.join(BLENDER_ROOT_DIR, 'colin27', 'electrodes', output_file_name.replace('_mni', ''))
    shutil.copyfile(electrodes_mni_fname, blender_file)


def get_good_channels():
    channels = read_channels_from_csv(op.join(OUTPUT_DIR, 'MG96MSITnostimChannelPairNamesBank1.csv'))
    channels |= read_channels_from_csv(op.join(OUTPUT_DIR, 'MG96MSITnostimChannelPairNamesBank2.csv'))
    print('good electdeos num: {}'.format(len(channels)))
    print('good electrodes, rh:')
    print([e for e in channels if e[0]=='R'])
    print('good electrodes, lh:')
    print([e for e in channels if e[0]=='L'])
    return set(channels)


def read_channels_from_csv(csv_fname):
    from src.mmvt_addon.mmvt_utils import csv_file_reader
    channels = set()
    for line in csv_file_reader(csv_fname):
        elecs_names = []
        ind = line.index('')
        for elc_name in [''.join(line[:ind]), ''.join(line[ind + 1:])]:
            if '0' in elc_name and elc_name[-1] != '0':
                elc_name = elc_name.replace('0', '')
            elecs_names.append(elc_name)
        channels.add('{}-{}'.format(elecs_names[1], elecs_names[0]))
    return channels

def get_groups_ordering():
    groups_ordering = np.genfromtxt(op.join(OUTPUT_DIR, 'groups_order.txt'), dtype=str)
    return groups_ordering


def get_elec_group(elec_name, bipolar):
    g = elec_group_number(elec_name, bipolar)
    return g[0]


def elec_group_number(elec_name, bipolar=False):
    import re
    if bipolar:
        elec_name2, elec_name1 = elec_name.split('-')
        group, num1 = elec_group_number(elec_name1, False)
        _, num2 = elec_group_number(elec_name2, False)
        return group, num1, num2
    else:
        elec_name = elec_name.strip()
        num = int(re.sub('\D', ',', elec_name).split(',')[-1])
        group = elec_name[:elec_name.rfind(str(num))]
        return group, num


if __name__ == '__main__':
    subject = 'mg96'
    bipolar = True
    atlas = 'aparc.DKTatlas40'
    error_radius, elec_length = 3, 4
    good_channels = get_good_channels()
    groups_ordering = get_groups_ordering()
    prepare_darpa_csv(subject, bipolar, atlas, good_channels, groups_ordering, error_radius, elec_length)
