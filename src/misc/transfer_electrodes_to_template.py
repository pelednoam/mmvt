import os.path as op
import numpy as np
from collections import defaultdict
import nibabel as nib
import glob

from src.utils import utils
from src.utils import preproc_utils as pu
from src.utils import freesurfer_utils as fu

SUBJECTS_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()


mri_robust_register = 'mri_robust_register --mov {subjects_dir}/{subject_from}/mri/T1.mgz --dst {subjects_dir}/{subject_to}/mri/T1.mgz --lta {subjects_dir}/{subject_from}/mri/{lta_name}.lta --satit --mapmov {subjects_dir}/{subject_from}/mri/T1_to_{subject_to}.mgz --cost nmi'


def register_to_template(subjects, template_system, subjects_dir, vox2vox=False, print_only=False):
    subject_to = 'fsaverage5' if template_system == 'ras' else 'colin27' if template_system == 'mni' else template_system
    for subject_from in subjects:
        cmd = mri_robust_register
        lta_name = 't1_to_{}'.format(subject_to)
        if vox2vox:
            cmd += ' --vox2vox'
            lta_name += '_vox2vox'
        rs = utils.partial_run_script(locals(), print_only=print_only)
        rs(cmd)


def apply_trans(trans, points):
    if isinstance(points, list):
        points = np.array(points)
    ndim = points.ndim
    if ndim == 1:
        points = [points]
    points = np.hstack((points, np.ones((len(points), 1))))
    points = np.dot(trans, points.T).T
    points = points[:, :3]
    if ndim == 1:
        points = points[0]
    return points


def lta_transfer_ras2ras(subject, coords, return_trans=False):
    lta_fname = op.join(SUBJECTS_DIR, subject, 'mri', 't1_to_{}.lta'.format(template_system))
    if not op.isfile(lta_fname):
        return None
    lta = fu.read_lta_file(lta_fname)
    lta[np.isclose(lta, np.zeros(lta.shape))] = 0
    subject_header = nib.load(op.join(SUBJECTS_DIR, subject, 'mri', 'T1.mgz')).get_header()
    template_header = nib.load(op.join(SUBJECTS_DIR, template_system, 'mri', 'T1.mgz')).get_header()
    vox = apply_trans(np.linalg.inv(subject_header.get_vox2ras_tkr()), coords)
    ras = apply_trans(subject_header.get_vox2ras(), vox)
    template_ras = apply_trans(lta, ras)
    template_vox = apply_trans(template_header.get_ras2vox(), template_ras)
    template_cords = apply_trans(template_header.get_vox2ras_tkr(), template_vox)
    if return_trans:
        return template_cords, lta
    else:
        return template_cords


def lta_transfer_ras2ras(subject, coords, return_trans=False):
    lta_fname = op.join(SUBJECTS_DIR, subject, 'mri', 't1_to_{}.lta'.format(template_system))
    if not op.isfile(lta_fname):
        return None
    lta = fu.read_lta_file(lta_fname)
    lta[np.isclose(lta, np.zeros(lta.shape))] = 0
    subject_header = nib.load(op.join(SUBJECTS_DIR, subject, 'mri', 'T1.mgz')).get_header()
    template_header = nib.load(op.join(SUBJECTS_DIR, template_system, 'mri', 'T1.mgz')).get_header()
    vox = apply_trans(np.linalg.inv(subject_header.get_vox2ras_tkr()), coords)
    ras = apply_trans(subject_header.get_vox2ras(), vox)
    template_ras = apply_trans(lta, ras)
    template_vox = apply_trans(template_header.get_ras2vox(), template_ras)
    template_cords = apply_trans(template_header.get_vox2ras_tkr(), template_vox)
    if return_trans:
        return template_cords, lta
    else:
        return template_cords



def lta_transfer_vox2vox(subject, coords):
    lta_fname = op.join(SUBJECTS_DIR, subject, 'mri', 't1_to_{}_vox2vox.lta'.format(template_system))
    if not op.isfile(lta_fname):
        return None
    lta = fu.read_lta_file(lta_fname)
    subject_header = nib.load(op.join(SUBJECTS_DIR, subject, 'mri', 'T1.mgz')).get_header()
    template_header = nib.load(op.join(SUBJECTS_DIR, template_system, 'mri', 'T1.mgz')).get_header()
    vox = apply_trans(np.linalg.inv(subject_header.get_vox2ras_tkr()), coords)
    template_vox = apply_trans(lta, vox)
    template_cords = apply_trans(template_header.get_vox2ras_tkr(), template_vox)
    return template_cords


def transfer_electrodes_to_template_system(electrodes, template_system, use_mri_robust_lta=False, vox2vox=False):
    teamplte_electrodes = defaultdict(list)
    for subject in electrodes.keys():
        # if subject != 'mg101':
        #     continue
        for elc_name, coords in electrodes[subject]:
            if use_mri_robust_lta:
                if vox2vox:
                    template_cords = lta_transfer_vox2vox(subject, coords)
                else:
                    template_cords = lta_transfer_ras2ras(subject, coords)
            else:
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


def compare_electrodes_labeling(electrodes, template_system, atlas='aparc.DKTatlas40'):
    template = 'fsaverage5' if template_system == 'ras' else 'colin27' if template_system == 'mni' else template_system
    template_elab_files = glob.glob(op.join(
        MMVT_DIR, template, 'electrodes', f'{template}_{atlas}_electrodes_cigar_r_3_l_4.pkl'))
    if len(template_elab_files) == 0:
        print(f'No electrodes labeling file for {template}!')
        return
    elab_template = utils.load(template_elab_files[0])
    errors = defaultdict(list)
    for subject in electrodes.keys():
        elab_files = glob.glob(op.join(
            MMVT_DIR, subject, 'electrodes', f'{subject}_{atlas}_electrodes_cigar_r_3_l_4.pkl'))
        if len(elab_files) == 0:
            errors[subject].append(f'No electrodes labeling file for {subject}!')
            print(f'No electrodes labeling file for {subject}!')
            continue
        errors[subject] = []
        electrodes_names = [e[0] for e in electrodes[subject]]
        elab = utils.load(elab_files[0])
        elab = [e for e in elab if e['name'] in electrodes_names]
        for elc in electrodes_names:
            no_erros = True
            elc_labeling = [e for e in elab if e['name'] == elc][0]
            elc_labeling_template = [e for e in elab_template if e['name'] == f'{subject}_{elc}'][0]
            for roi, prob in zip(elc_labeling['cortical_rois'], elc_labeling['cortical_probs']):
                no_erros = no_erros and compare_rois_and_probs(
                    subject, template, elc, roi, prob, elc_labeling['cortical_rois'],
                    elc_labeling_template['cortical_rois'], elc_labeling_template['cortical_probs'])
            for roi, prob in zip(elc_labeling['subcortical_rois'], elc_labeling['subcortical_probs']):
                no_erros = no_erros and compare_rois_and_probs(
                    subject, template, elc, roi, prob, elc_labeling['subcortical_rois'],
                    elc_labeling_template['subcortical_rois'], elc_labeling_template['subcortical_probs'])
            if no_erros:
                print(f'{subject},{elc},Good!')
    # print(errors)


def compare_rois_and_probs(subject, template, elc, roi, prob, elc_labeling_rois, elc_labeling_template_rois,
                           elc_labeling_template_rois_probs):
    no_erros = True
    if roi not in elc_labeling_template_rois:
        if prob > 0.05:
            err = f'{subject},{elc},{roi} ({prob}) not in {template}'
            print(err)
            no_erros = False
    else:
        roi_ind = elc_labeling_template_rois.index(roi)
        template_roi_prob = elc_labeling_template_rois_probs[roi_ind]
        if abs(prob - template_roi_prob) > 0.05:
            err = f'{subject},{elc},{roi} prob ({prob} != {template} prob ({template_roi_prob})'
            print(err)
            no_erros = False
    for roi, prob in zip(elc_labeling_template_rois, elc_labeling_template_rois_probs):
        if roi not in elc_labeling_rois and prob > 0.05:
            err = f'{subject},{elc},{roi} ({prob}) only in {template}'
            print(err)
            no_erros = False
    return no_erros


def sanity_check():
    subject = 'mg101'
    template_system = 'hc029'
    # mg101 RMF3 to hc029
    tk_ras = [7.3, 37.9, 59]
    ras = [6.08, 73.07, 17.80]
    vox = [121, 69, 166]
    template_tk_ras_true = np.array([6.18, 52.26, 21.46])
    template_vox_true = np.array([122, 107, 180])

    template_tk_ras, trans = fu.transform_subject_to_subject_coordinates(
        subject, template_system, tk_ras, SUBJECTS_DIR, return_trans=True)
    template_tk_ras2 = apply_trans(trans, tk_ras)
    assert (all(np.isclose(template_tk_ras, template_tk_ras2, rtol=1e-3)))
    lta = fu.read_lta_file(op.join(SUBJECTS_DIR, subject, 'mri', 't1_to_{}.lta'.format(template_system)))
    lta[np.isclose(trans, np.zeros(lta.shape))] = 0
    print(lta-trans)
    # template_lta_tk_ras = lta_transfer_ras2ras(subject, tk_ras)
    # assert(all(np.isclose(template_tk_ras_true, template_tk_ras, rtol=1e-3)))

    subject = 'mg112'
    # ROF1 - 2
    tk_ras = [2.00, 29.00, 8.00]
    ras = [-6.88, 50.40, -7.31]
    template_tk_ras = fu.transform_subject_to_subject_coordinates(
        subject, template_system, tk_ras, SUBJECTS_DIR)
    print('asdf')

if __name__ == '__main__':
    roots = ['/home/npeled/Documents/', '/homes/5/npeled/space1/Angelique/misc']
    root = [d for d in roots if op.isdir(d)][0]
    csv_name = 'StimLocationsPatientList.csv'
    save_as_bipolar = False
    template_system = 'hc029' #''mni'
    atlas = 'aparc.DKTatlas40'

    # electrodes = read_csv_file(op.join(root, csv_name), save_as_bipolar)
    # print(','.join(electrodes.keys()))
    # register_to_template(electrodes.keys(), template_system, SUBJECTS_DIR, vox2vox=True, print_only=False)
    # template_electrodes = transfer_electrodes_to_template_system(electrodes, template_system)
    # save_template_electrodes_to_template(template_electrodes, save_as_bipolar, template_system, 'stim_')
    # export_into_csv(template_electrodes, template_system, 'stim_')
    # compare_electrodes_labeling(electrodes, template_system, atlas)

    sanity_check()

    # mri_cvs_data_copy
    # mri_cvs_check
    # mri_cvs_register
    '''
    mri_vol2vol --mov $folder/w_MniNick.nii --targ $folder/MniNick.nii - --o 
        $folder/outvol.mgz --m3z $folder/final_CVSmorph_toSynthF.m3z --noDefM3zPath 
        --no-save-reg.
    '''
    print('finish')