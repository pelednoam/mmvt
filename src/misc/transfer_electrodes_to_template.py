import os.path as op
import numpy as np
from collections import defaultdict

from src.utils import utils
from src.utils import preproc_utils as pu
from src.utils import freesurfer_utils as fu

SUBJECTS_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()


def transfer_electrodes_to_template_system(electrodes, template_system):
    teamplte_electrodes = defaultdict(list)
    for subject in electrodes.keys():
        for elc_name, coords in electrodes[subject]:
            if template_system == 'ras':
                template_cords = fu.transform_subject_to_ras_coordinates(subject, coords, SUBJECTS_DIR)
            elif tempalte_system == 'mni':
                template_cords = fu.transform_subject_to_mni_coordinates(subject, coords, SUBJECTS_DIR)
            else:
                raise Exception('Wrong template system! ({})'.format(tempalte_system))
            if template_cords is not None:
                teamplte_electrodes[subject].append((elc_name, template_cords))
    return teamplte_electrodes


def read_csv_file(csv_fname, save_as_bipolar):
    csv_lines = np.genfromtxt(csv_fname,  dtype=str, delimiter=',', skip_header=1)
    electrodes = defaultdict(list)
    for line in csv_lines:
        subject = line[0].lower()
        elecs_names = [e.strip() for e in line[1:3].tolist()]
        elecs_groups = [utils.elec_group(e) for e in elecs_names]
        if elecs_groups[0] == elecs_groups[1]:
            elec_name = f'{elecs_names[1]}-{elecs_names[0]}'
        else:
            print(f'The electrodes has different groups! {elecs_groups[0]}, {elecs_groups[1]}')
            continue
        coords1 = line[3:6].astype(np.float)
        coords2 = line[6:9].astype(np.float)
        if save_as_bipolar:
            coords = (coords1 + (coords2 - coords1) / 2)
            electrodes[subject].append((elec_name, coords))
        else:
            electrodes[subject].append((elecs_names[0], coords1))
            electrodes[subject].append((elecs_names[1], coords2))
    return electrodes


def save_mni_electrodes_to_template(mni_electrodes, bipolar, tempalte_system='mni', prefix='', postfix=''):
    output_fname = '{}electrodes{}_positions.npz'.format(prefix, '_bipolar' if bipolar else '', postfix)
    if tempalte_system == 'ras':
        fol = utils.make_dir(op.join(MMVT_DIR, 'fsaverage5', 'electrodes'))
    elif tempalte_system == 'mni':
        fol = utils.make_dir(op.join(MMVT_DIR, 'colin27', 'electrodes'))
    else:
        raise Exception('Wrong template system! ({})'.format(tempalte_system))
    output_fname = op.join(fol, output_fname)
    elecs_coordinates = np.array(utils.flat_list_of_lists(
        [[e[1] for e in mni_electrodes[subject]] for subject in mni_electrodes.keys()]))
    elecs_names = utils.flat_list_of_lists(
        [['{}_{}'.format(subject, e[0]) for e in mni_electrodes[subject]] for subject in mni_electrodes.keys()])
    np.savez(output_fname, pos=elecs_coordinates, names=elecs_names, pos_org=[])
    print(f'Electrodes were saved to {output_fname}')


if __name__ == '__main__':
    root = '/homes/5/npeled/space1/Angelique/misc'
    csv_name = 'StimLocationsPatientList.csv'
    save_as_bipolar = False
    tempalte_system = 'mni'
    electrodes = read_csv_file(op.join(root, csv_name), save_as_bipolar)
    mni_electrodes = transfer_electrodes_to_template_system(electrodes, tempalte_system)
    save_mni_electrodes_to_template(mni_electrodes, save_as_bipolar, tempalte_system, 'stim_')