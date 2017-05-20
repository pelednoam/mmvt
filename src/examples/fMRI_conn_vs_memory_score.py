# import matplotlib
# matplotlib.use('svg')
import matplotlib.pyplot as plt
import numpy as np
import pandas
from scipy.stats import mannwhitneyu
import os.path as op

from src.utils import preproc_utils as pu
from src.utils import utils

SUBJECTS_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()

only_left = False #only L TLE patients
fast_TR = False #include 568 ms TR
root_paths = ['/homes/5/npeled/space1/Documents/memory_task', '/home/npeled/Documents/memory_task/']
root_path = [p for p in root_paths if op.isdir(p)][0]


def get_inds(only_left, TR, fast_TR, to_use, laterality):
    if only_left:
        if not fast_TR:
            inds = np.where(np.logical_and(np.logical_not(to_use == 'No'), laterality == 'L',TR > 1))[0]
        else:
            inds = np.where(np.logical_and(np.logical_not(to_use == 'No'), laterality == 'L'))[0]
    else:
        if not fast_TR:
            inds = np.where(np.logical_and(np.logical_not(to_use == 'No'), np.in1d(laterality,['L', 'R']),TR > 1))[0]
        else:
            inds = np.where(np.logical_and(np.logical_not(to_use == 'No'), np.in1d(laterality,['L', 'R'])))[0]
    return inds


def read_scoring():
    scoring_fname = op.join(root_path, 'neuropsych_scores.npz')
    if not op.isfile(scoring_fname):
        scoring_xls_fname = '/cluster/neuromind/sx424/subject_info/StufflebeamLabDataba_DATA_LABELS_2017-01-27_1132.xlsx'
        neuropsych_scores = pandas.read_excel(scoring_xls_fname, sheetname='Necessary scores', header=None, skiprows={0})
        subjects_master = np.array(neuropsych_scores.loc[:,0].astype(str))
        laterality = np.array(neuropsych_scores.loc[:, 1].astype(str))
        to_use = np.array(neuropsych_scores.loc[:, 2].astype(str))
        TR = np.array(neuropsych_scores.loc[:, 3].astype(float))
        values = np.array(neuropsych_scores.loc[:,4:].astype(float))
        np.savez(op.join(root_path, 'neuropsych_scores.npy'), laterality=laterality, to_use=to_use, TR=TR, values=values,
                 subjects_master=subjects_master)
        return laterality, to_use, TR, values, subjects_master
    else:
        d = np.load(scoring_fname)
        return d['laterality'], d['to_use'], d['TR'], d['values'], d['subjects_master']


def find_good_inds(all_subjects, only_left, TR, fast_TR, to_use, laterality):
    scoring_inds = get_inds(only_left, TR, fast_TR, to_use, laterality)
    _, subjects_inds = find_subjects_with_data(all_subjects)
    bad_indices, labels = check_subjects_labels(all_subjects)
    inds = list(set(scoring_inds) & set(subjects_inds) - set(bad_indices))
    print('{}/{} good subjects'.format(len(inds), len(all_subjects)))
    return all_subjects[inds], inds, labels


def calc_disturbed_preserved_inds(inds, values):
    master_grouping = (np.sum((values <= 5).astype(int), axis=1) > 0).astype(int)
    subject_groups = master_grouping[inds]
    #disturbed = 1, preserved = 0
    disturbed_inds = np.where(subject_groups == 1)[0]
    preserved_inds = np.where(subject_groups == 0)[0]
    return disturbed_inds, preserved_inds


def find_subjects_with_data(all_subjects):
    pcs = [1, 2, 4, 8]
    subjects, subjects_inds = [], []
    for subject_ind, subject in enumerate(all_subjects):
        fol = op.join(MMVT_DIR, subject, 'connectivity')
        if not op.isdir(fol):
            print('No connectivity folder for {}'.format(subject))
            continue
        all_files_exist = all([op.isfile(op.join(
            fol, 'fmri_mi_vec_cv_mean_pca{}.npy'.format('' if pc == 1 else '_{}'.format(pc)))) for pc in pcs])
        if all_files_exist:
            subjects.append(subject)
            subjects_inds.append(subject_ind)
        else:
            print('Not all pcs results for {}!'.format(subject))
    print('{}/{} subjects with data were found'.format(len(subjects), len(all_subjects)))
    return subjects, subjects_inds


def get_subjects_dFC(subjects):
    pcs = [1, 2, 4, 8]
    dFC_res = {pc:None for pc in pcs}
    std_mean_res = {pc:None for pc in pcs}
    stat_conn_res = {pc:None for pc in pcs}
    for subject_ind, subject in enumerate(subjects):
        fol = op.join(MMVT_DIR, subject, 'connectivity')
        for pc in pcs:
            fname = op.join(fol, 'fmri_mi_vec_cv_mean_pca{}.npz'.format('' if pc == 1 else '_{}'.format(pc)))
            d = np.load(fname)
            dFC = d['dFC']
            std_mean = d['std_mean']
            stat_conn = d['stat_conn']
            if dFC_res[pc] is None:
                dFC_res[pc] = np.zeros((len(subjects), *dFC.shape))
            if std_mean_res[pc] is None:
                std_mean_res[pc] = np.zeros((len(subjects), *std_mean.shape))
            if stat_conn_res[pc] is None:
                stat_conn_res[pc] = np.zeros((len(subjects), *stat_conn.shape))
            dFC_res[pc][subject_ind] = dFC
            std_mean_res[pc][subject_ind] = std_mean
            stat_conn_res[pc][subject_ind] = stat_conn

    return dFC_res, std_mean_res, stat_conn_res


def switch_laterality(res, subjects, subject_lateralities):
    rois_inds = find_labels_inds(labels)
    for pc in res.keys():
        for s in subjects:
            if subject_lateralities[s] == 'L':
                res[pc][rois_inds, s] = stat_conn[ROIs_L].T
            else:
                res[pc][rois_inds, s] = dFC_unnorm[ROIs_R].T



def stat_test(res, disturbed_inds, preserved_inds):
    mann_whitney_results = {pc:None for pc in res.keys()}
    for pc, dFCs in res.items():
        subjects_num, labels_num = dFCs.shape
        for label_ind in range(labels_num):
            test_res = mannwhitneyu(dFCs[disturbed_inds, label_ind], dFCs[preserved_inds, label_ind])
            if mann_whitney_results[pc] is None:
                mann_whitney_results[pc] = np.zeros(labels_num)
            mann_whitney_results[pc][label_ind] = test_res.pvalue
    return mann_whitney_results


def check_subjects_labels(subjects):
    _labels = np.load(op.join(MMVT_DIR, subjects[0], 'connectivity', 'labels_names.npy'))
    bad_indices = []
    for sub_ind, subject in enumerate(subjects):
        labels_fname = op.join(MMVT_DIR, subject, 'connectivity', 'labels_names.npy')
        if not op.isfile(labels_fname):
            bad_indices.append(sub_ind)
            continue
        labels = np.load(labels_fname)
        all_eq = np.array_equal(labels, _labels)
        if not all_eq:
            print('{} labels are not equal!'.format(subject))
            bad_indices.append(sub_ind)
    if not all_eq:
        print('Not all the subjects labels are equall!!')
    return bad_indices, _labels


def get_rois_pvals(all_stat_results, labels, rois_inds):
    for res_type, stat_results in all_stat_results.items():
        pcs = sorted(list(stat_results.keys()))
        pvals = {}
        for pc in pcs:
            pvals[pc] = stat_results[pc][rois_inds]
        print(pvals)
        plt.figure()
        width = 0.2
        x = np.arange(len(rois_inds))
        for ind, pc in enumerate(pcs):
            plt.bar(x + ind * width, stat_results[pc][rois_inds], width, label=str(pc))
        plt.plot((0, len(rois_inds)), (0.05, 0.05), 'r--')
        plt.xticks(x, labels[rois_inds], rotation='vertical')
        plt.legend()
        plt.title(res_type)
        plt.show()


def find_sig_results(all_stat_results, labels):
    for res_type, stat_results in all_stat_results.items():
        pcs = sorted(list(stat_results.keys()))
        sig_inds = []
        for pc in pcs:
            sig_inds.extend(np.where(stat_results[pc] < 0.05)[0])
        sig_inds = sorted(list(set(sig_inds)))
        for sig_ind in sig_inds:
            print(labels[sig_ind], [(pc, stat_results[pc][sig_ind]) for pc in pcs])
        plt.figure()
        width = 0.2
        x = np.arange(len(sig_inds))
        for ind, pc in enumerate(pcs):
            plt.bar(x + ind * width, stat_results[pc][sig_inds], width, label=str(pc))
        plt.plot((0, len(sig_inds)), (0.05, 0.05), 'r--')
        plt.xticks(x, labels[sig_inds], rotation='vertical')
        plt.legend()
        plt.title(res_type)
        plt.show()


def check_labels():
    from src.utils import labels_utils as lu

    labels = np.load(op.join(root_path, 'labels_names.npy'))
    labels = lu.read_labels('fsaverage', SUBJECTS_DIR, 'laus125', only_names=True, sorted_according_to_annot_file=True)
    # labels = np.array(labels)
    print([(ind, l) for ind, l in enumerate(labels) if l.startswith('unk')])
    print([(ind, l) for ind, l in enumerate(labels) if l.startswith('corp')])
    remove_ids = np.array([1, 5, 114, 118]) - 1
    print('asdf')


def find_labels_inds(labels):
    laus125_labels_lh = [line.rstrip() for line in open(op.join(root_path, 'fmri_laus125_lh.txt'))]
    laus125_labels_rh = [line.rstrip() for line in open(op.join(root_path, 'fmri_laus125_rh.txt'))]
    laus125_labels_lh = [s + '-lh' for s in laus125_labels_lh]
    laus125_labels_rh = [s + '-rh' for s in laus125_labels_rh]
    laus125_labels = np.array(laus125_labels_lh + laus125_labels_rh)
    rois = laus125_labels[np.array([4, 8, 115, 119])]
    labels_rois_inds = np.array([np.where(labels==l)[0][0] for l in rois])
    print(labels[labels_rois_inds])
    return labels_rois_inds


def calc_mann_whitney_results():
    mann_whitney_results_fname = op.join(root_path, 'mann_whitney_results.pkl')
    good_subjects_fname = op.join(root_path, 'good_subjects.npz')
    ana_results_fname = op.join(root_path, 'ana_results.pkl')
    if True: #not op.isfile(ana_results_fname) or not op.isfile(good_subjects_fname):
        laterality, to_use, TR, values, all_subjects = read_scoring()
        good_subjects, good_subjects_inds, labels = find_good_inds(
            all_subjects, only_left, TR, fast_TR, to_use, laterality)
        disturbed_inds, preserved_inds = calc_disturbed_preserved_inds(good_subjects_inds, values)
        dFC_res, std_mean_res, stat_conn_res = get_subjects_dFC(good_subjects)
        utils.save((dFC_res, std_mean_res, stat_conn_res, disturbed_inds, preserved_inds, good_subjects, labels),
                   op.join(root_path, 'ana_results.pkl'))
    else:
        (dFC_res, std_mean_res, stat_conn_res, disturbed_inds, preserved_inds, good_subjects, labels, laterality) = \
            utils.load(ana_results_fname)
    if True: #not op.isfile(mann_whitney_results_fname):
        mann_whitney_results = {}
        for res, res_name in zip([dFC_res, std_mean_res], ['dFC_res', 'std_mean_res']): # stat_conn_res
            switch_laterality(res, good_subjects, laterality)
            mann_whitney_results[res_name] = stat_test(res, disturbed_inds, preserved_inds)
        utils.save(mann_whitney_results, mann_whitney_results_fname)
        np.savez(good_subjects_fname, good_subjects=good_subjects, labels=labels)
    else:
        mann_whitney_results = utils.load(mann_whitney_results_fname)
        d = np.load(good_subjects_fname)
        good_subjects = d['good_subjects']
        labels = d['labels']
    return mann_whitney_results, good_subjects, labels


if __name__ == '__main__':
    mann_whitney_results, good_subjects, labels = calc_mann_whitney_results()
    rois_inds = find_labels_inds(labels)
    get_rois_pvals(mann_whitney_results, labels, rois_inds)
    # plot_stat_results(mann_whitney_results)
    # find_sig_results(mann_whitney_results, labels)
    print('Wooohooo!')