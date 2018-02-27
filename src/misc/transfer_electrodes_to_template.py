import os.path as op
import numpy as np
from collections import defaultdict
import nibabel as nib
import glob
from scipy.spatial.distance import cdist
import csv
import shutil

from src.utils import utils
from src.utils import preproc_utils as pu
from src.utils import freesurfer_utils as fu

SUBJECTS_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()


mri_robust_register = 'mri_robust_register --mov {subjects_dir}/{subject_from}/mri/T1.mgz --dst {subjects_dir}/{subject_to}/mri/T1.mgz --lta {subjects_dir}/{subject_from}/mri/{lta_name}.lta --satit --mapmov {subjects_dir}/{subject_from}/mri/T1_to_{subject_to}.mgz --cost nmi'
mri_cvs_register = 'mri_cvs_register --mov {subject_from} --template {subject_to} ' + \
                   '--outdir {subjects_dir}/{subject_from}/mri_cvs_register_to_{subject_to} --nocleanup' # --step3'
mri_vol2vol = 'mri_vol2vol --mov {subjects_dir}/{subject}/mri/T1.mgz ' + \
    '--o {subjects_dir}/{subject}/mri/T1_to_colin_csv_register.mgz --m3z ' + \
    '{subjects_dir}/{subject}/mri_cvs_register_to_colin27/final_CVSmorph_tocolin27.m3z ' + \
    '--noDefM3zPath --no-save-reg --targ {subjects_dir}/colin27/mri/T1.mgz'
mri_elcs2elcs = 'mri_vol2vol --mov {subjects_dir}/{subject}/electrodes/{elcs_file_name} ' + \
    '--o {subjects_dir}/{subject}/electrodes/{output_name}_to_colin27.nii.gz --m3z ' + \
    '{subjects_dir}/{subject}/mri_cvs_register_to_colin27/final_CVSmorph_tocolin27.m3z ' + \
    '--noDefM3zPath --no-save-reg --targ {subjects_dir}/colin27/mri/T1.mgz'
applyMorph = 'applyMorph --template {subjects_dir}/{subject_to}/mri/orig.mgz ' \
             '--transform {subjects_dir}/{subject_from}/mri_cvs_register_to_{subject_to}/' + \
             'combined_to{subject_to}_elreg_afteraseg-norm.tm3d ' + \
             'point_list {subjects_dir}/{subject_from}/electrodes/stim_electrodes.txt ' + \
             '{subjects_dir}/{subject_from}/electrodes/stim_electrodes_to_{subject_to}.txt a'


def robust_register_to_template(subjects, template_system, subjects_dir, vox2vox=False, print_only=False):
    subject_to = 'fsaverage5' if template_system == 'ras' else 'colin27' if template_system == 'mni' else template_system
    for subject_from in subjects:
        cmd = mri_robust_register
        lta_name = 't1_to_{}'.format(subject_to)
        if vox2vox:
            cmd += ' --vox2vox'
            lta_name += '_vox2vox'
        rs = utils.partial_run_script(locals(), print_only=print_only)
        rs(cmd)


def cvs_register_to_template(subjects, template_system, subjects_dir, overwrite=False, print_only=False, n_jobs=1):
    subject_to = 'fsaverage5' if template_system == 'ras' else 'colin27' if template_system == 'mni' else template_system
    subjects = [s for s in subjects if s != subject_to and (overwrite or not op.isfile(op.join(
                subjects_dir, s, f'mri_cvs_register_to_{subject_to}', f'final_CVSmorph_to{subject_to}.m3z')))]
    indices = np.array_split(np.arange(len(subjects)), n_jobs)
    chunks = [([subjects[ind] for ind in chunk_indices], subject_to, subjects_dir, overwrite, print_only)
              for chunk_indices in indices]
    utils.run_parallel(_mri_cvs_register_parallel, chunks, n_jobs)


def _mri_cvs_register_parallel(p):
    subjects, subject_to, subjects_dir, overwrite, print_only = p
    for subject_from in subjects:
        if overwrite and not print_only:
            utils.delete_folder_files(op.join(subjects_dir, subject_from, 'mri_cvs_register_to_{}'.format(subject_to)))
        rs = utils.partial_run_script(locals(), print_only=print_only)
        rs(mri_cvs_register)


def morph_t1(subjects, template_system, subjects_dir, print_only=False):
    subject_to = 'fsaverage5' if template_system == 'ras' else 'colin27' if template_system == 'mni' else template_system
    for subject in subjects:
        if not op.isfile(op.join(subjects_dir, subject, 'mri_cvs_register_to_colin27', 'final_CVSmorph_tocolin27.m3z')):
            print(f'The m3z morph matrix does not exist for subject {subject}!')
            continue
        output_fname = op.join(subjects_dir, subject, 'mri', 'T1_to_colin_csv_register.mgz')
        if not op.isfile(output_fname):
            rs = utils.partial_run_script(locals(), print_only=print_only)
            rs(mri_vol2vol)
        print(f'freeview -v {subjects_dir}/colin27/mri/T1.mgz {subjects_dir}/{subject}/mri/T1_to_colin_csv_register.mgz')


def morph_electrodes(electrodes, template_system, subjects_dir, mmvt_dir, overwrite=False, print_only=False, n_jobs=4):
    subject_to = 'fsaverage5' if template_system == 'ras' else 'colin27' if template_system == 'mni' else template_system

    subjects = list(electrodes.keys())
    indices = np.array_split(np.arange(len(subjects)), n_jobs)
    chunks = [([subjects[ind] for ind in chunk_indices], subject_to, subjects_dir, overwrite, print_only)
              for chunk_indices in indices]
    utils.run_parallel(_morph_electrodes_parallel, chunks, n_jobs)


def _morph_electrodes_parallel(p):
    subjects, subject_to, subjects_dir, overwrite, print_only = p
    bad_subjects, good_subjects = [], []
    for subject_from in subjects:
        output_fname = op.join(subjects_dir, subject_from, 'electrodes', f'stim_electrodes_to_{subject_to}.txt')
        if op.isfile(output_fname) and not overwrite:
            continue
        rs = utils.partial_run_script(locals(), print_only=print_only)
        rs(applyMorph)
        ret = op.isfile(output_fname)
        if not ret:
            bad_subjects.append(subject_from)
        else:
            good_subjects.append(subject_from)

    print('good subjects: {}'.format(good_subjects))
    print('bad subjects: {}'.format(bad_subjects))


def read_morphed_electrodes(electrodes, template_system, subjects_dir, mmvt_dir, overwrite=False):
    subject_to = 'fsaverage5' if template_system == 'ras' else 'colin27' if template_system == 'mni' else template_system
    output_fname = op.join(mmvt_dir, subject_to, 'electrodes', 'template_electrodes.pkl')
    if op.isfile(output_fname) and not overwrite:
        return
    t1_header = nib.load(op.join(subjects_dir, subject_to, 'mri', 'T1.mgz')).header
    trans = t1_header.get_vox2ras_tkr()
    template_electrodes = defaultdict(list)
    bad_subjects, good_subjects = [], []
    for subject in electrodes.keys():
        if subject == subject_to:
            continue
        input_fname = op.join(subjects_dir, subject, 'electrodes', f'stim_electrodes_to_{subject_to}.txt')
        if not op.isfile(input_fname):
            bad_subjects.append(subject)
            continue
        print('Reading {} ({})'.format(input_fname, utils.file_modification_time(input_fname)))
        vox = np.genfromtxt(input_fname,  dtype=np.float, delimiter=' ')
        tkregs = apply_trans(trans, vox)
        for tkreg, (elc_name, _) in zip(tkregs, electrodes[subject]):
            template_electrodes[subject].append((f'{subject}_{elc_name}', tkreg))
        good_subjects.append(subject)
    utils.save(template_electrodes, output_fname)
    print('read_morphed_electrodes: {}'.format(op.isfile(output_fname)))
    print('good subjects: {}'.format(good_subjects))
    print('bad subjects: {}'.format(bad_subjects))


def morph_electrodes_volume(electrodes, template_system, subjects_dir, mmvt_dir, overwrite=False, print_only=False):
    subject_to = 'fsaverage5' if template_system == 'ras' else 'colin27' if template_system == 'mni' else template_system
    template_electrodes = defaultdict(list)
    header = nib.load(op.join(subjects_dir, subject_to, 'mri', 'T1.mgz')).header
    for subject in electrodes.keys():
        if not op.isfile(op.join(subjects_dir, subject, 'mri_cvs_register_to_colin27', 'final_CVSmorph_tocolin27.m3z')):
            # print(f'The m3z morph matrix does not exist for subject {subject}!')
            continue
        electrodes_fname = op.join(subjects_dir, subject, 'electrodes', 'stim_electrodes.nii.gz')
        if not op.isfile(electrodes_fname):
            # print(f"Can't find volumetric electrodes file for {subject}")
            continue
        for stim_file in glob.glob(op.join(subjects_dir, subject, 'electrodes', 'stim_????.nii.gz')):
            elcs_file_name = utils.namebase_with_ext(stim_file)
            output_name = utils.namebase(stim_file)
            output_fname = op.join(subjects_dir, subject, 'electrodes', f'{output_name}_to_colin27.nii.gz')
            if not op.isfile(output_fname) or overwrite:
                rs = utils.partial_run_script(locals(), print_only=print_only)
                rs(mri_elcs2elcs)
        for morphed_fname in glob.glob(op.join(subjects_dir, subject, 'electrodes', 'stim_????_to_colin27.nii.gz')):
            print(f'Loading {morphed_fname}')
            x = nib.load(morphed_fname).get_data()
            inds = np.array(np.where(x>0)).T
            vol = inds[np.argmax([x[tuple(ind)] for ind in inds])]
            tkreg = fu.apply_trans(header.get_vox2ras_tkr(), vol)[0]
            elc_name = utils.namebase(morphed_fname).split('_')[1]
            template_electrodes[subject].append((f'{subject}_{elc_name}', tkreg))
            # utils.plot_3d_scatter(inds, names=[x[tuple(ind)] for ind in inds])
            # print(subject, utils.namebase(morphed_fname), len(inds))
        # morphed_output_fname = op.join(subjects_dir, subject, 'electrodes', 'stim_electrodes_to_colin27.nii.gz')
        # if not op.isfile(morphed_output_fname):
        #     elcs_file_name = 'stim_electrodes.nii.gz'
        #     rs = utils.partial_run_script(locals(), print_only=print_only)
        #     rs(mri_elcs2elcs)
        # if not op.isfile(morphed_output_fname):
        #     print('Error in morphing the electrodes volumetric file!')
        #     continue
        # elecs = get_tkreg_from_volume(subject, electrodes_fname)
        # tkreg, pairs, dists = get_electrodes_from_morphed_volume(template_system, morphed_output_fname, len(elecs), subjects_dir, 0)
        # print([dists[p[0],p[1]] for p in pairs])
        # utils.plot_3d_scatter(tkreg, names=range(len(tkreg)), labels_indices=range(len(tkreg)), title=subject)
        # print(f'{subject} has {len(elecs)} electrodes')
        # print(f'{subject} after morphing as {len(tkreg)} electrodes:')
        # print(f'freeview -v {subjects_dir}/{subject}/electrodes/stim_electrodes.nii.gz {subjects_dir}/colin27/mri/T1.mgz')
        # for ind, pair in enumerate(pairs):
        #     pair_name = chr(ord('A') + ind)
        #     for elc_ind in range(2):
        #         template_electrodes[subject].append((f'{subject}_{pair_name}{elc_ind + 1}', tkreg[pair[elc_ind]]))
    utils.save(template_electrodes, op.join(mmvt_dir, subject_to, 'electrodes', 'template_electrodes.pkl'))
    return template_electrodes


def get_electrodes_from_morphed_volume(template_system, morphed_output_fname, electrodes_num, subjects_dir, threshold=0):
    from src.misc.dell import find_electrodes_in_ct
    template = 'fsaverage5' if template_system == 'ras' else 'colin27' if template_system == 'mni' else template_system

    morphed_data = nib.load(morphed_output_fname).get_data()
    header = nib.load(op.join(subjects_dir, template, 'mri', 'T1.mgz'))
    opt_voxels = np.array(np.where(morphed_data > threshold)).T
    vol, _ = find_electrodes_in_ct.clustering(opt_voxels, morphed_data, electrodes_num, clustering_method='knn')
    tkreg = fu.apply_trans(header.header.get_vox2ras_tkr(), vol)
    dists = cdist(tkreg, tkreg)
    inds = np.where((dists < 7) & (dists > 3))
    pairs = list(set([tuple(sorted([inds[0][k], inds[1][k]])) for k in range(len(inds[0]))]))
    return tkreg, pairs, dists


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


def read_all_electrodes(subjects, bipolar):
    from src.preproc import electrodes as elc_pre
    electrodes = defaultdict(list)
    bads, goods = [], []
    for subject in subjects:
        names, pos = elc_pre.read_electrodes_file(subject, bipolar)
        if len(names) == 0:
            bads.append(subject)
        else:
            goods.append(subject)
        for elec_name, coords in zip(names, pos):
            electrodes[subject].append((elec_name, coords))
    print('bads: {}'.format(bads))
    print('goods: {}'.format(goods))
    return electrodes


def save_template_electrodes_to_template(template_electrodes, bipolar, mmvt_dir, template_system='mni', prefix='', postfix=''):
    output_fname = '{}electrodes{}_positions.npz'.format(prefix, '_bipolar' if bipolar else '', postfix)
    template = 'fsaverage5' if template_system == 'ras' else 'colin27' if template_system == 'mni' else template_system
    if template_electrodes is None:
        input_fname = op.join(mmvt_dir, template, 'electrodes', 'template_electrodes.pkl')
        print('Reading {} ({})'.format(input_fname, utils.file_modification_time(input_fname)))
        template_electrodes = utils.load(input_fname)
    fol = utils.make_dir(op.join(MMVT_DIR, template, 'electrodes'))
    output_fname = op.join(fol, output_fname)
    elecs_coordinates = np.array(utils.flat_list_of_lists(
        [[e[1] for e in template_electrodes[subject]] for subject in template_electrodes.keys()]))
    elecs_names = utils.flat_list_of_lists(
        [[e[0] for e in template_electrodes[subject]] for subject in template_electrodes.keys()])
    np.savez(output_fname, pos=elecs_coordinates, names=elecs_names, pos_org=[])
    print(f'Electrodes were saved to {output_fname}')


def export_into_csv(template_system, mmvt_dir, prefix=''):
    template = 'fsaverage5' if template_system == 'ras' else 'colin27' if template_system == 'mni' else template_system
    electrodes_dict = utils.Bag(np.load(op.join(mmvt_dir, template, 'electrodes', 'stim_electrodes_positions.npz')))
    fol = utils.make_dir(op.join(MMVT_DIR, template, 'electrodes'))
    csv_fname = op.join(fol, '{}{}_RAS.csv'.format(prefix, template))
    print('Writing csv file to {}'.format(csv_fname))
    with open(csv_fname, 'w') as csv_file:
        wr = csv.writer(csv_file, quoting=csv.QUOTE_NONE)
        wr.writerow(['Electrode Name','R','A','S'])
        for elc_name, elc_coords in zip(electrodes_dict.names, electrodes_dict.pos):
            wr.writerow([elc_name, *['{:.2f}'.format(x) for x in elc_coords.squeeze()]])
    fol = utils.make_dir(op.join(SUBJECTS_DIR, template, 'electrodes'))
    csv_fname2 = op.join(fol, utils.namebase_with_ext(csv_fname))
    shutil.copy(csv_fname, csv_fname2)
    print('export_into_csv: {}'.format(op.isfile(csv_fname) and op.isfile(csv_fname2)))


def compare_electrodes_labeling(electrodes, template_system, atlas='aparc.DKTatlas40'):
    template = 'fsaverage5' if template_system == 'ras' else 'colin27' if template_system == 'mni' else template_system
    template_elab_files = glob.glob(op.join(
        MMVT_DIR, template, 'electrodes', f'{template}_{atlas}_electrodes_cigar_r_3_l_4.pkl'))
    if len(template_elab_files) == 0:
        print(f'No electrodes labeling file for {template}!')
        return
    elab_template = utils.load(template_elab_files[0])
    errors = ''
    for subject in electrodes.keys():
        elab_files = glob.glob(op.join(
            MMVT_DIR, subject, 'electrodes', f'{subject}_{atlas}_electrodes_cigar_r_3_l_4.pkl'))
        if len(elab_files) == 0:
            print(f'No electrodes labeling file for {subject}!')
            continue
        electrodes_names = [e[0] for e in electrodes[subject]]
        elab = utils.load(elab_files[0])
        elab = [e for e in elab if e['name'] in electrodes_names]
        for elc in electrodes_names:
            no_errors = True
            elc_labeling = [e for e in elab if e['name'] == elc][0]
            elc_labeling_template = [e for e in elab_template if e['name'] == f'{subject}_{elc}'][0]
            for roi, prob in zip(elc_labeling['cortical_rois'], elc_labeling['cortical_probs']):
                no_err, err = compare_rois_and_probs(
                    subject, template, elc, roi, prob, elc_labeling['cortical_rois'],
                    elc_labeling_template['cortical_rois'], elc_labeling_template['cortical_probs'])
                no_errors = no_errors and no_err
                if err != '':
                    errors += err + '\n'
            for roi, prob in zip(elc_labeling['subcortical_rois'], elc_labeling['subcortical_probs']):
                no_err, err = compare_rois_and_probs(
                    subject, template, elc, roi, prob, elc_labeling['subcortical_rois'],
                    elc_labeling_template['subcortical_rois'], elc_labeling_template['subcortical_probs'])
                no_errors = no_errors and no_err
                if err != '':
                    errors += err + '\n'
            if no_errors:
                print(f'{subject},{elc},Good!')
                errors += f'{subject},{elc},Good!\n'
    with open(op.join(MMVT_DIR, template, 'electrodes', 'trans_errors.txt'), "w") as text_file:
        print(errors, file=text_file)
    # print(errors)


def compare_rois_and_probs(subject, template, elc, roi, prob, elc_labeling_rois, elc_labeling_template_rois,
                           elc_labeling_template_rois_probs):
    no_errors = True
    err = ''
    if roi not in elc_labeling_template_rois:
        if prob > 0.05:
            err = f'{subject},{elc},{roi} ({prob}) not in {template}'
            print(err)
            no_errors = False
    else:
        roi_ind = elc_labeling_template_rois.index(roi)
        template_roi_prob = elc_labeling_template_rois_probs[roi_ind]
        if abs(prob - template_roi_prob) > 0.05:
            err = f'{subject},{elc},{roi} prob ({prob} != {template} prob ({template_roi_prob})'
            print(err)
            no_errors = False
    for roi, prob in zip(elc_labeling_template_rois, elc_labeling_template_rois_probs):
        if roi not in elc_labeling_rois and prob > 0.05:
            err = f'{subject},{elc},{roi} ({prob}) only in {template}'
            print(err)
            no_errors = False
    return no_errors, err


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


def prepare_files(subjects, template_system):
    template = 'fsaverage5' if template_system == 'ras' else 'colin27' if template_system == 'mni' else template_system
    # mri_cvs_check --mov {subject} --template {template}
    necessary_files = {'surf': ['lh.inflated', 'rh.inflated', 'lh.pial', 'rh.pial', 'rh.white', 'lh.white',
                                'lh.smoothwm', 'rh.smoothwm', 'rh.sulc', 'lh.sulc', 'lh.sphere', 'rh.sphere',
                                'lh.inflated.K', 'rh.inflated.K', 'lh.inflated.H', 'rh.inflated.H'],
                       'label': ['lh.aparc.annot', 'rh.aparc.annot']}
    subjects = list(subjects) + [template]
    martinos_subjects = {'mg96':'ep007', 'mg78': 'ep001', 'ep001': 'ep001'}
    goods, bads = [], []
    for subject in subjects:
        files_exist = utils.check_if_all_necessary_files_exist(
            subject, necessary_files, op.join(SUBJECTS_DIR, subject), trace=True)
        if files_exist:
            goods.append(subject)
            continue
        darpa_subject = subject[:2].upper() + subject[2:]
        fols = glob.glob(op.join(
            f'/homes/5/npeled/space1/Angelique/recon-alls/{darpa_subject}/', '**', f'{darpa_subject}_SurferOutput'),
            recursive=True)
        remote_subject_dir = fols[0] if len(fols) == 1 else ''
        files_exist = get_subject_files(subject, necessary_files, remote_subject_dir)
        if not files_exist:
            remote_subject_dir = op.join('/space/huygens/1/users/kara', f'{darpa_subject}_SurferOutput')
            files_exist = get_subject_files(subject, necessary_files, remote_subject_dir)
        if not files_exist and subject in martinos_subjects.keys():
            remote_subject_dir = op.join(
                '/autofs/space/lilli_001/users/DARPA-Recons/', martinos_subjects[subject])
            files_exist = get_subject_files(subject, necessary_files, remote_subject_dir)
        if files_exist:
            goods.append(subject)
        else:
            bads.append(subject)
    print(f'goods: {goods}')
    print(f'bads: {bads}')
    return goods, bads


def get_subject_files(subject, necessary_files, remote_subject_dir):
    if not op.isdir(remote_subject_dir):
        return False
    return utils.prepare_subject_folder(
        necessary_files, subject, remote_subject_dir, SUBJECTS_DIR, print_traceback=False)


def create_electrodes_files(electrodes, subjects_dir, overwrite=False):
    for subject in electrodes.keys():
        t1_header = nib.load(op.join(subjects_dir, subject, 'mri', 'T1.mgz')).header
        trans = np.linalg.inv(t1_header.get_vox2ras_tkr())
        fol = utils.make_dir(op.join(subjects_dir, subject, 'electrodes'))
        csv_fname = op.join(fol, 'stim_electrodes.txt')
        if op.isfile(csv_fname) and not overwrite:
            continue
        with open(csv_fname, 'w') as csv_file:
            wr = csv.writer(csv_file, quoting=csv.QUOTE_NONE, delimiter=' ')
            for _, coords in electrodes[subject]:
                vox = np.rint(apply_trans(trans, coords)).astype(int)
                wr.writerow([*['{:.2f}'.format(x) for x in vox]])


def create_volume_with_electrodes(electrodes, subjects_dir, merge_to_pairs=True, overwrite=False):
    for subject in electrodes.keys():
        if not op.isfile(op.join(SUBJECTS_DIR, subject, 'mri', 'T1.mgz')):
            print(f'No T1 file for {subject}')
            continue
        t1_header = nib.load(op.join(SUBJECTS_DIR, subject, 'mri', 'T1.mgz')).header
        fol = utils.make_dir(op.join(subjects_dir, subject, 'electrodes'))
        output_fname = op.join(fol, 'stim_electrodes.nii.gz')
        data = np.zeros((256, 256, 256), dtype=np.int16)
        if merge_to_pairs:
            groups = defaultdict(list)
            for elc_name, coords in electrodes[subject]:
                groups[utils.elec_group(elc_name)].append(elc_name, coords)
            for group in groups.keys():
                pair_data = np.zeros((256, 256, 256), dtype=np.int16)
                for elc_name, coords in groups[group]:
                    vox = tkreg_to_vox(t1_header, coords)
                    pair_data[tuple(vox)] = 1000
                pair_output_fname = op.join(fol, f'stim_{elc_name}.nii.gz')
                if not op.isfile(elc_output_fname) or overwrite:
                    pass
        else:
            for elc_name, coords in electrodes[subject]:
                vox = tkreg_to_vox(t1_header, coords)
                # data[tuple(vox)] = 1000
                elc_output_fname = op.join(fol, f'stim_{elc_name}.nii.gz')
                if not op.isfile(elc_output_fname) or overwrite:
                    elc_data = np.zeros((256, 256, 256), dtype=np.int16)
                    elc_data[tuple(vox)] = 1000
                    elc_img = nib.Nifti1Image(elc_data, t1_header.get_affine(), t1_header)
                    print(f'Saving {elc_output_fname}')
                    nib.save(elc_img, elc_output_fname)
        # if not op.isfile(output_fname) or overwrite:
        #     img = nib.Nifti1Image(data, t1_header.get_affine(), t1_header)
        #     print(f'Saving {output_fname}')
        #     nib.save(img, output_fname)
        # tkreg = get_tkreg_from_volume(subject, output_fname)


def tkreg_to_vox(t1_header, tkreg):
    return np.rint(fu.apply_trans(np.linalg.inv(t1_header.get_vox2ras_tkr()), tkreg)).astype(int)[0]


def get_tkreg_from_volume(subject, data_fname):
    if not op.isfile(op.join(SUBJECTS_DIR, subject, 'mri', 'T1.mgz')):
        print(f'No T1 file for {subject}')
        return None
    t1_header = nib.load(op.join(SUBJECTS_DIR, subject, 'mri', 'T1.mgz')).header
    data = nib.load(data_fname).get_data()
    indices = np.where(data > 0)
    vox = np.array(indices).T
    print(vox)
    tkreg = fu.apply_trans(t1_header.get_vox2ras_tkr(), vox)
    return tkreg


if __name__ == '__main__':
    roots = ['/home/npeled/Documents/stim_locations', '/homes/5/npeled/space1/Angelique/misc']
    root = [d for d in roots if op.isdir(d)][0]
    csv_name = 'StimLocationsPatientList.csv'
    save_as_bipolar = False
    template_system = 'mni' # hc029
    atlas = 'aparc.DKTatlas40'
    bipolar = False

    # electrodes = read_csv_file(op.join(root, csv_name), save_as_bipolar)
    # subjects = electrodes.keys()
    subjects = ['ep001']
    # electrodes = read_all_electrodes(subjects, bipolar)
    good_subjects, bad_subjects = prepare_files(subjects, template_system)
    cvs_register_to_template(good_subjects, template_system, SUBJECTS_DIR, n_jobs=4, print_only=False, overwrite=False)
    # create_electrodes_files(electrodes, SUBJECTS_DIR, True)
    # morph_electrodes(electrodes, template_system, SUBJECTS_DIR, MMVT_DIR, overwrite=True, n_jobs=4)
    # read_morphed_electrodes(electrodes, template_system, SUBJECTS_DIR, MMVT_DIR, overwrite=True)
    # save_template_electrodes_to_template(None, save_as_bipolar, MMVT_DIR, template_system, 'stim_')
    # export_into_csv(template_system, MMVT_DIR, 'stim_')


    # compare_electrodes_labeling(electrodes, template_system, atlas)




    # print(','.join(electrodes.keys()))
    # good_subjects = ['mg96']
    # cvs_register_to_template(good_subjects, template_system, SUBJECTS_DIR, n_jobs=4, print_only=False, overwrite=False) #
    # template_electrodes = transfer_electrodes_to_template_system(electrodes, template_system)
    # save_template_electrodes_to_template(template_electrodes, save_as_bipolar, template_system, 'stim_')

    # create_volume_with_electrodes(electrodes, SUBJECTS_DIR, merge_to_pairs=True, False)
    # morph_t1(electrodes.keys(), template_system, SUBJECTS_DIR)

    # export_into_csv(template_system, MMVT_DIR, 'stim_')

    # sanity_check()

    # mri_cvs_data_copy
    # mri_cvs_check
    # mri_cvs_register
    # cvs_register_to_template(['mg105'], template_system, SUBJECTS_DIR)
    '''
    mri_cvs_register --mov mg112 --template colin27 c
    mri_vol2vol --mov {subjects_dir}/mg112/mri/T1.mgz --o {subjects_dir}/mg112/mri/T1_to_colin_csv_register.mgz --m3z
     {subjects_dir}/mg112/mri_cvs_register/final_CVSmorph_tocolin27.m3z --noDefM3zPath --no-save-reg --targ {subjects_dir}/colin27/mri/T1.mgz

    '''
    print('finish')