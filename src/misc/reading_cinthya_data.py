from collections import defaultdict
import scipy.stats
import numpy as np
import os.path as op


def read_xlsx(xlsx_fname):
    import xlrd
    wb = xlrd.open_workbook(xlsx_fname)
    sh = wb.sheets()[0]
    labels = sh.row_values(0)[4:]
    for i in range(0, len(labels), 2):
        labels[i + 1] = f'{labels[i]}-rh'
        labels[i] = f'{labels[i]}-lh'
    all_data = defaultdict(lambda: defaultdict(list))
    for rownum in range(2, sh.nrows):
        row_data = sh.row_values(rownum)
        if row_data[0] == '':
            break
        data = row_data[4:]
        if row_data[1] == '':
            continue
        subject_type = int(row_data[1])
        for val, label in zip(data, labels):
            all_data[label][subject_type].append(val)
    return all_data


def ttest(data, root_fol, output_name, two_tailed_test=True, alpha=0.05, is_greater=True):
    labels = sorted(list(data.keys()))
    subjects_types = sorted(list(data[labels[0]]))
    ttest_stats, ttest_labels, welch_stats, welch_labels, labels_means = [], [], [] ,[], []
    for ind, label in enumerate(labels):
        labels_means.append(np.mean(data[label][1] + data[label][2]))
        data1, data2 = only_floats(data[label][subjects_types[0]]), only_floats(data[label][subjects_types[1]])
        # two-tailed p-value
        t, pval = scipy.stats.ttest_ind(data1, data2, equal_var=True)
        if is_significant(pval, t, two_tailed_test, alpha, is_greater):
            ttest_stats.append(pval)
            ttest_labels.append(label)
        t, pval = scipy.stats.ttest_ind(data1, data2, equal_var=False)
        if is_significant(pval, t, two_tailed_test, alpha, is_greater):
            welch_stats.append(pval)
            welch_labels.append(label)
    title = output_name.replace('_', ' ')
    np.savez(op.join(root_fol, '{}_mean.npz'.format(output_name)), names=labels,
             atlas='aparc.DKTatlas', data=np.array(labels_means), title=title,
             data_min=np.min(labels_means), data_max=np.max(labels_means), cmap='YlOrRd')
    print('{} ttest: {} significant labels were found'.format(title, len(ttest_stats)))
    print('{} welch: {} significant labels were found'.format(title, len(ttest_stats)))
    np.savez(op.join(root_fol, '{}_ttest.npz'.format(output_name)), names=np.array(ttest_labels),
             atlas='aparc.DKTatlas', data=np.array(ttest_stats), title='{} ttest'.format(title),
             data_min=0, data_max=0.05, cmap='RdOrYl')
    np.savez(op.join(root_fol, '{}_welch.npz'.format(output_name)), names=np.array(welch_labels),
             atlas='aparc.DKTatlas', data=np.array(welch_stats), title='{} welch'.format(title),
             data_min=0, data_max=0.05, cmap='RdOrYl')
    return ttest_stats, welch_stats


def is_significant(pval, t, two_tailed_test, alpha=0.05, is_greater=True):
    if two_tailed_test:
        return pval < alpha
    else:
        if is_greater:
            return pval / 2 < alpha and t > 0
        else:
            return pval / 2 < alpha and t < 0


def is_float(x):
    try:
        float(x)
        return True
    except:
        return False


def only_floats(arr):
    return [x for x in arr if is_float(x)]


if __name__ == '__main__':
    root_fol = [f for f in ['/homes/5/npeled/space1/Cinthya/', '/home/npeled/Documents/Cinthya/'] if op.isdir(f)][0]

    data = read_xlsx(op.join(root_fol, 'COLBOS_CT_Database.xlsx'))
    ttest(data, root_fol, 'cortical_thickness')

    # data = read_xlsx(op.join(root_fol, 'COLBOS_Vol_DB.xlsx'))
    # ttest(data, root_fol, 'cortical_volume')
