# import matplotlib
# matplotlib.use('svg')
import matplotlib.pyplot as plt
import numpy as np
import pandas
from scipy.stats import mannwhitneyu
import os.path as op

from src.utils import utils


def get_links():
    links_dir = utils.get_links_dir()
    subjects_dir = utils.get_link_dir(links_dir, 'subjects', 'SUBJECTS_DIR')
    freesurfer_home = utils.get_link_dir(links_dir, 'freesurfer', 'FREESURFER_HOME')
    mmvt_dir = utils.get_link_dir(links_dir, 'mmvt')
    return subjects_dir, mmvt_dir, freesurfer_home

SUBJECTS_DIR, MMVT_DIR, FREESURFER_HOME = get_links()

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


def get_linda_subjects():
    return ['nmr00474', 'nmr00502', 'nmr00515', 'nmr00603', 'nmr00609', 'nmr00626',
        'nmr00629', 'nmr00650', 'nmr00657', 'nmr00669', 'nmr00674', 'nmr00681', 'nmr00683',
        'nmr00692', 'nmr00698', 'nmr00710']


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
        # if only_linda:
        #     subject_list = get_linda_subjects()
        #     inds = np.where(np.in1d(subjects_master, subject_list))[0]
        #     subject_list = subjects_master[inds]
        #     master_grouping = (np.sum((values <= 5).astype(int), axis=1) > 0).astype(int)
        #     subject_groups = master_grouping[inds]
        np.savez(op.join(root_path, 'neuropsych_scores.npy'), laterality=laterality, to_use=to_use, TR=TR, values=values,
                 subjects_master=subjects_master)
        return laterality, to_use, TR, values, subjects_master
    else:
        d = np.load(scoring_fname)
        return d['laterality'], d['to_use'], d['TR'], d['values'], d['subjects_master']


def find_good_subjects_indices(all_subjects):
    good_subjects = utils.read_list_from_file(op.join(root_path, 'good_subjects.txt'))
    return [ind for ind in range(len(all_subjects)) if all_subjects[ind] in good_subjects]


def find_linda_subjects_indices(all_subjects):
    linda_subjects = get_linda_subjects()
    return [np.where(sub==all_subjects)[0][0] for sub in linda_subjects]


def find_good_inds(all_subjects, only_left, TR, fast_TR, to_use, laterality):
    scoring_inds = get_inds(only_left, TR, fast_TR, to_use, laterality)
    _, subjects_inds = find_subjects_with_data(all_subjects)
    bad_indices, labels = check_subjects_labels(all_subjects)
    good_subjects_inds = find_good_subjects_indices(all_subjects)
    linda_subjects_inds = find_linda_subjects_indices(all_subjects)
    inds = list(set(scoring_inds) & set(subjects_inds) & set(good_subjects_inds) & set(linda_subjects_inds) - set(bad_indices))
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


def get_subjects_fmri_conn(subjects):
    labels = None
    conn_stds = []
    for subject_ind, subject in enumerate(subjects):
        fol = op.join(MMVT_DIR, subject, 'connectivity')
        if labels is None:
            labels = np.load(op.join(fol, 'labels_names.npy'))
            labeals_inds = find_labels_inds(labels)
            subs_inds = [np.where(labels=='Right-Hippocampus')[0][0], np.where(labels=='Left-Hippocampus')[0][0]]
            inds = np.concatenate((labeals_inds, subs_inds))
        d = np.load(op.join(fol, 'fmri_corr_cv_mean.npz'))
        conn_std = d['conn_std']
        # conn_std = conn_std[inds][:, inds]
        conn_stds.append(conn_std)
    conn_stds = np.array(conn_stds)
    np.save(op.join(root_path, 'conn_stds.npy'), conn_stds)


def get_subjects_dFC(subjects):
    pcs = ['mean', 1, 2, 4, 8]
    dFC_res = {pc:None for pc in pcs}
    std_mean_res = {pc:None for pc in pcs}
    stat_conn_res = {pc:None for pc in pcs}
    for subject_ind, subject in enumerate(subjects):
        fol = op.join(MMVT_DIR, subject, 'connectivity')
        for pc in pcs:
            if pc == 'mean':
                fname = op.join(fol, 'fmri_corr_cv_mean_mean.npz')
            else:
                fname = op.join(fol, 'fmri_mi_vec_cv_mean_pca{}.npz'.format('' if pc == 1 else '_{}'.format(pc)))
            print('Loading {} ({})'.format(fname, utils.file_modification_time(fname)))
            if not op.isfile(fname):
                print('{} not exist!'.format(fname))
                continue
            d = np.load(fname)
            dFC = d['dFC']
            std_mean = d['std_mean']
            # stat_conn = d['stat_conn']
            if dFC_res[pc] is None:
                dFC_res[pc] = np.zeros((len(subjects), *dFC.shape))
            if std_mean_res[pc] is None:
                std_mean_res[pc] = np.zeros((len(subjects), *std_mean.shape))
            # if stat_conn_res[pc] is None:
            #     stat_conn_res[pc] = np.zeros((len(subjects), *stat_conn.shape))
            dFC_res[pc][subject_ind] = dFC
            std_mean_res[pc][subject_ind] = std_mean
            # stat_conn_res[pc][subject_ind] = stat_conn

    return dFC_res, std_mean_res, stat_conn_res


def switch_laterality(res, subjects, labels, subject_lateralities):
    labels = np.load(op.join(root_path, 'labels_names.npy'))
    corr_stds = np.load(op.join(root_path, 'conn_stds.npy'))
    rois_inds = find_labels_inds(labels)
    subs_L = np.where(labels == 'Left-Hippocampus')[0][0]
    subs_R = np.where(labels=='Right-Hippocampus')[0][0]
    ROIs_L = rois_inds[0:2] #, subs_L, subs_R] # rois_inds
    ROIs_R = rois_inds[2:4] # , subs_L, subs_R]# np.concaten
    # rois_inds = [rois_inds[0], subs_L, subs_R]
    rois_res = {}
    # new_corr_stds = np.zeros((len(subjects), 2, 2))
    for pc in res.keys():
        if res[pc] is None:
            print('res[{}] is None!'.format(pc))
            continue
        rois_res[pc] = np.zeros((len(subjects), len(ROIs_L)))
        for s_ind, s in enumerate(subjects):
            if subject_lateralities[s_ind] == 'L':
                rois_res[pc][s_ind] = res[pc][s_ind, ROIs_R]
                # new_corr_stds[s_ind] = corr_stds[s_ind, np.array([ROIs_R, ROIs_L]), np.array([subs_R, subs_L])]
            else:
                rois_res[pc][s_ind] = res[pc][s_ind, ROIs_L]
                # new_corr_stds[s_ind] = corr_stds[s_ind, np.array([ROIs_R, ROIs_L]) , np.array([subs_R, subs_L])]
    # rois_inds = [rois_inds[1]]
    return rois_res, rois_inds


def run_stat(res, disturbed_inds, preserved_inds):
    mann_whitney_results = {pc:None for pc in res.keys()}
    for pc, dFCs in res.items():
        subjects_num, labels_num = dFCs.shape
        for label_ind in range(labels_num):
            test_res = mannwhitneyu(dFCs[disturbed_inds, label_ind], dFCs[preserved_inds, label_ind])
            if mann_whitney_results[pc] is None:
                mann_whitney_results[pc] = np.zeros(labels_num)
            mann_whitney_results[pc][label_ind] = test_res.pvalue
    return mann_whitney_results


def check_subjects_labels(subjects, check_labels_indices=True):
    for subject in subjects:
        labels_fname = op.join(MMVT_DIR, subject, 'connectivity', 'labels_names.npy')
        if op.isfile(labels_fname):
            _labels = np.load(labels_fname)
            break
    # _labels = np.load(op.join(MMVT_DIR, subjects[0], 'connectivity', 'labels_names.npy'))
    if not check_labels_indices:
        return [], _labels
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


def take_only_linda_subjects(subjects, disturbed_inds, preserved_inds, laterality):
    linda_subjects = ['nmr00474', 'nmr00502', 'nmr00515', 'nmr00603', 'nmr00609', 'nmr00626',
        'nmr00629', 'nmr00650', 'nmr00657', 'nmr00669', 'nmr00674', 'nmr00681', 'nmr00683',
        'nmr00692', 'nmr00698', 'nmr00710']
    indices, new_disturbed_inds, new_preserved_inds = [], [], []
    new_ind = -1
    for s_ind, s in enumerate(subjects):
        if s in linda_subjects:
            indices.append(linda_subjects.index(s))
            new_ind += 1
            if s_ind in disturbed_inds:
                new_disturbed_inds.append(new_ind)
            if s_ind in preserved_inds:
                new_preserved_inds.append(new_ind)
    indices = np.array(indices)
    return subjects[indices], new_disturbed_inds, new_preserved_inds, laterality[indices]


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


def plot_bar(corr_stds, disturbed_inds, preserved_inds):
    f, axs = plt.subplots(2, 2, sharey=True, sharex=True)
    axs = axs.ravel()
    indices = [[0,0],[0,1],[1,0],[1,1]]
    for ind, ax in enumerate(axs):
        i, j = indices[ind]
        ax.scatter(np.ones((len(preserved_inds))) * 0.3, corr_stds[preserved_inds, i, j])
        ax.scatter(np.ones((len(disturbed_inds))) * 0.7, corr_stds[disturbed_inds, i, j])
        ax.set_xticks([.3, .7])
        ax.set_xlim([0, 1])
        ax.set_xticklabels(['preserved', 'disturbed'], rotation=30)
    # plt.set_title('{}{}'.format(pc, ' PCs' if pc != 'mean' else ''))
    plt.show()
    print('asfd')


def plot_comparisson_bars(res, res_name, labels, disturbed_inds, preserved_inds, mann_whitney_res):
    # plot_bar(corr_stds, disturbed_inds, preserved_inds)
    x = np.arange(len(res.keys()))
    from collections import defaultdict
    x1, x2 = defaultdict(list), defaultdict(list)
    width = 0.35
    pcs = ['mean', 1, 2, 4, 8]
    # for pc in pcs:
    #     x1 = (np.mean(res[pc][disturbed_inds, :], 0))
    #     x2.extend(np.mean(res[pc][preserved_inds, :], 0))
    #     # for label_ind, label in enumerate(labels):
    #     #     x1[label_ind].append(np.mean(res[pc][disturbed_inds, label_ind]))
    #     #     x2[label_ind].append(np.mean(res[pc][preserved_inds, label_ind]))
    for label_ind in range(res[pcs[0]].shape[1]):
        f, axs = plt.subplots(1, len(pcs), sharey=True)
        for ind, (pc, ax) in enumerate(zip(pcs, axs)):
            if not pc in res:
                continue
            ax.scatter(np.ones((len(preserved_inds))) * 0.3, res[pc][preserved_inds, label_ind])
            ax.scatter(np.ones((len(disturbed_inds))) * 0.7, res[pc][disturbed_inds, label_ind])
            ax.set_xticks([.3, .7])
            ax.set_xlim([0, 1])
            ax.set_xticklabels(['preserved', 'disturbed'], rotation=30)
            ax.set_title('{}{} ({:.2f})'.format(pc, ' PCs' if pc != 'mean' else '', mann_whitney_res[pc][label_ind]))
        fig = plt.gcf()
        # plt.legend(['preserved', 'disturbed'])
        fig.suptitle('{} - {}'.format(res_name, '{} Memory and cPCC flexibility'.format(labels[label_ind])))
    plt.show()
    # plt.savefig(op.join(root_path, '{}-{}.png'.format(res_name, 'Memory and cPCC flexibility')))
        # plt.figure()
        # plt.bar(x, x1[label_ind], width, label='disturbed')
        # plt.bar(x + width, x2[label_ind], width, label='preserved')
        # plt.xticks(x, ['{}{}'.format(pc, ' PCs' if pc != 'mean' else '') for pc in pcs])
        # plt.legend()
        # plt.title('{} - {}'.format(res_name, label))
        # plt.savefig(op.join(root_path, '{}-{}.png'.format(res_name, label)))
    # plt.show()
    print('done with plotting!')


def check_labels():
    from src.utils import labels_utils as lu

    labels = np.load(op.join(root_path, 'labels_names.npy'))
    labels = lu.read_labels('fsaverage', SUBJECTS_DIR, 'laus125', only_names=True, sorted_according_to_annot_file=True)
    # labels = np.array(labels)
    print([(ind, l) for ind, l in enumerate(labels) if l.startswith('unk')])
    print([(ind, l) for ind, l in enumerate(labels) if l.startswith('corp')])
    remove_ids = np.array([1, 5, 114, 118]) - 1
    print('asdf')


def get_labels_order():
    laus125_labels_lh = [line.rstrip() for line in open(op.join(root_path, 'fmri_laus125_lh.txt'))]
    laus125_labels_rh = [line.rstrip() for line in open(op.join(root_path, 'fmri_laus125_rh.txt'))]
    laus125_labels_lh = [s + '-lh' for s in laus125_labels_lh]
    laus125_labels_rh = [s + '-rh' for s in laus125_labels_rh]
    laus125_labels = np.array(laus125_labels_lh + laus125_labels_rh)
    utils.write_list_to_file(laus125_labels, op.join(root_path, 'linda_laus125_order.txt'))
    return laus125_labels


def find_labels_inds(labels):
    laus125_labels= get_labels_order()
    rois = laus125_labels[np.array([4, 8, 115, 119])]
    labels_rois_inds = np.array([np.where(labels==l)[0][0] for l in rois])
    print(labels[labels_rois_inds])
    return labels_rois_inds


def calc_ana(overwrite=False, only_linda=False):
    good_subjects_fname = op.join(root_path, 'good_subjects.npz')
    ana_results_fname = op.join(root_path, 'ana_results.pkl')
    if not op.isfile(ana_results_fname) or not op.isfile(good_subjects_fname) or overwrite:
        laterality, to_use, TR, values, all_subjects = read_scoring()
        if only_linda:
            subject_list = get_linda_subjects()
            inds = np.where(np.in1d(all_subjects, subject_list))[0]
            good_subjects = all_subjects[inds]
            master_grouping = (np.sum((values <= 5).astype(int), axis=1) > 0).astype(int)
            subject_groups = master_grouping[inds]
            disturbed_inds = np.array(np.where(subject_groups == 1)[0])
            preserved_inds = np.array(np.where(subject_groups == 0)[0])
            laterality = ['L'] * len(good_subjects)
            bad_indices, labels = check_subjects_labels(good_subjects, check_labels_indices=False)
        else:
            good_subjects, good_subjects_inds, labels = find_good_inds(
                all_subjects, only_left, TR, fast_TR, to_use, laterality)
            disturbed_inds, preserved_inds = calc_disturbed_preserved_inds(good_subjects_inds, values)
        dFC_res, std_mean_res, stat_conn_res = get_subjects_dFC(good_subjects)
        utils.save(
            (dFC_res, std_mean_res, stat_conn_res, disturbed_inds, preserved_inds, good_subjects, labels, laterality),
            op.join(root_path, 'ana_results.pkl'))
    else:
        (dFC_res, std_mean_res, stat_conn_res, disturbed_inds, preserved_inds, good_subjects, labels, laterality) = \
            utils.load(ana_results_fname)
    return dFC_res, std_mean_res, stat_conn_res, disturbed_inds, preserved_inds, good_subjects, labels, laterality


def calc_mann_whitney_results(dFC_res, std_mean_res, stat_conn_res, disturbed_inds, preserved_inds, good_subjects,
                              labels, laterality, switch=True):
    good_subjects_fname = op.join(root_path, 'good_subjects.npz')
    mann_whitney_results_fname = op.join(root_path, 'mann_whitney_results.pkl')
    # good_subjects, disturbed_inds, preserved_inds, laterality = take_only_linda_subjects(
    #     good_subjects, disturbed_inds, preserved_inds, laterality)
    if True: # op.isfile(mann_whitney_results_fname):
        mann_whitney_results = {}
        res, res_name = std_mean_res, 'std_mean_res'
        # for res, res_name in zip([dFC_res, std_mean_res], ['dFC_res', 'std_mean_res']): # stat_conn_res
        # for res, res_name in zip([std_mean_res], ['std_mean_res']):  # stat_conn_res
        # if switch:
        #     res, rois_inds = switch_laterality(res, good_subjects, labels, laterality)
        # else:
        #     rois_inds = find_labels_inds(labels)
        rois_inds = np.array([4, 8, 115, 119])
        mann_whitney_results[res_name] = run_stat(res, disturbed_inds, preserved_inds)
        print(mann_whitney_results[res_name])
        plot_comparisson_bars(res, res_name, labels[rois_inds], disturbed_inds, preserved_inds, mann_whitney_results[res_name])
        utils.save(mann_whitney_results, mann_whitney_results_fname)
        np.savez(good_subjects_fname, good_subjects=good_subjects, labels=labels)
    else:
        mann_whitney_results = utils.load(mann_whitney_results_fname)
        d = np.load(good_subjects_fname)
        good_subjects = d['good_subjects']
        labels = d['labels']
    return mann_whitney_results, good_subjects, labels


def run_sandya_code():
    import scipy as spy
    san_path = '/home/npeled/Documents/memory_task/code/'
    ROI_values_dyn = np.load(op.join(san_path, 'ROI_values_dyn102.npy'))
    subject_groups = np.load(op.join(san_path, 'subject_groups.npy'))

    # disturbed = 1, preserved = 0
    disturbed_inds = np.array(np.where(subject_groups == 1))
    preserved_inds = np.array(np.where(subject_groups == 0))

    # left is 0, right is 1
    laterality = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Separate by contralateral and ipsilateral - ipsi is left, contra is right
    # Mann-Whitney tests between groups - p<0.0125 - bonferroni correction for 4 ROIs
    # spy.mannwhitneyu(x,y)
    ROI_values_dyn_disturbed = np.squeeze(ROI_values_dyn[:, disturbed_inds])
    ROI_values_dyn_preserved = np.squeeze(ROI_values_dyn[:, preserved_inds])
    mann_whitney_results_dyn = np.zeros((ROI_values_dyn.shape[0], 2), 'float')

    for i in range(ROI_values_dyn.shape[0]):
        mann_whitney_results_dyn[i, :] = spy.stats.mannwhitneyu(ROI_values_dyn_disturbed[i, :],
                                                                ROI_values_dyn_preserved[i, :]).pvalue

    print(mann_whitney_results_dyn)

if __name__ == '__main__':
    # get_labels_order()
    # run_sandya_code()
    ana_res = calc_ana(True, only_linda=True)
    # get_subjects_fmri_conn(ana_res[5])
    mann_whitney_results, good_subjects, labels = calc_mann_whitney_results(*ana_res)
    # rois_inds = find_labels_inds(labels)
    # get_rois_pvals(mann_whitney_results, labels, range(4))
    # plot_stat_results(mann_whitney_results)
    # find_sig_results(mann_whitney_results, labels)
    print('Wooohooo!')