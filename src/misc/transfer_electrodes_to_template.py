import os.path as op
import numpy as np
from collections import defaultdict

from src.utils import utils
from src.utils import preproc_utils as pu
from src.utils import freesurfer_utils as fu

SUBJECTS_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()


mri_robust_register = 'mri_robust_register --mov {subjects_dir}/{subject_from}/mri/T1.mgz --dst {subjects_dir}/{subject_to}/mri/T1.mgz --lta {subjects_dir}/{subject_from}/mri/t1_to_{subject_to}.lta --satit --mapmov {subjects_dir}/{subject_from}/mri/T1_to_{subject_to}.mgz --cost nmi'


def register_to_template(subjects, template_system, subjects_dir, vox2vox=False, print_only=False):
    subject_to = 'fsaverage5' if template_system == 'ras' else 'colin27' if template_system == 'mni' else template_system
    for subject_from in subjects:
        rs = utils.partial_run_script(locals(), print_only=print_only)
        cmd = mri_robust_register
        if vox2vox:
            cmd += ' --vox2vox'
        rs(mri_robust_register)


def transfer_electrodes_to_template_system(electrodes, template_system):
    teamplte_electrodes = defaultdict(list)
    for subject in electrodes.keys():
        for elc_name, coords in electrodes[subject]:
            if template_system == 'ras':
                template_cords = fu.transform_subject_to_ras_coordinates(subject, coords, SUBJECTS_DIR)
            elif template_system == 'mni':
                template_cords = fu.transform_subject_to_mni_coordinates(subject, coords, SUBJECTS_DIR)
            else:
                template_cords = fu.transform_subject_to_subject_coordinates(
                    subject, template_system, coords, SUBJECTS_DIR)
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


def save_template_electrodes_to_template(template_electrodes, bipolar, template_system='mni', prefix='', postfix=''):
    output_fname = '{}electrodes{}_positions.npz'.format(prefix, '_bipolar' if bipolar else '', postfix)
    template = 'fsaverage5' if template_system == 'ras' else 'colin27' if template_system == 'mni' else template_system
    fol = utils.make_dir(op.join(MMVT_DIR, template, 'electrodes'))
    output_fname = op.join(fol, output_fname)
    elecs_coordinates = np.array(utils.flat_list_of_lists(
        [[e[1] for e in template_electrodes[subject]] for subject in template_electrodes.keys()]))
    elecs_names = utils.flat_list_of_lists(
        [['{}_{}'.format(subject, e[0]) for e in template_electrodes[subject]] for subject in template_electrodes.keys()])
    np.savez(output_fname, pos=elecs_coordinates, names=elecs_names, pos_org=[])
    print(f'Electrodes were saved to {output_fname}')


def export_into_csv(template_electrodes, template_system, prefix=''):
    import csv
    template = 'fsaverage5' if template_system == 'ras' else 'colin27' if template_system == 'mni' else template_system
    fol = utils.make_dir(op.join(MMVT_DIR, template, 'electrodes'))
    csv_fname = op.join(fol, '{}{}_RAS.csv'.format(prefix, template))
    print('Writing csv file to {}'.format(csv_fname))
    with open(csv_fname, 'w') as csv_file:
        wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        wr.writerow(['Electrode Name','R','A','S'])
        for subject in template_electrodes.keys():
            for elc_name, elc_coords in template_electrodes[subject]:
                wr.writerow(['{}_{}'.format(subject, elc_name), *['{:.2f}'.format(x) for x in elc_coords]])



if __name__ == '__main__':
    root = '/homes/5/npeled/space1/Angelique/misc'
    csv_name = 'StimLocationsPatientList.csv'
    save_as_bipolar = False
    template_system = 'mni' # 'hc029' #''mni'
    electrodes = read_csv_file(op.join(root, csv_name), save_as_bipolar)
    register_to_template(electrodes.keys(), template_system, SUBJECTS_DIR, vox2vox=True, print_only=True)
    # template_electrodes = transfer_electrodes_to_template_system(electrodes, template_system)
    # save_template_electrodes_to_template(template_electrodes, save_as_bipolar, template_system, 'stim_')
    # export_into_csv(template_electrodes, template_system, 'stim_')