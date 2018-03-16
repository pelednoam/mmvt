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


def ttest(data, root_fol):
    labels = sorted(list(data.keys()))
    subjects_types = sorted(list(data[labels[0]]))
    ttest_stats, welch_stats = np.zeros((len(labels))), np.zeros((len(labels)))
    for ind, label in enumerate(labels):
        data1, data2 = data[label][subjects_types[0]], data[label][subjects_types[1]]
        # two-tailed p-value
        _, ttest_stats[ind] = scipy.stats.ttest_ind(data1, data2, equal_var=True)
        _, welch_stats[ind] = scipy.stats.ttest_ind(data1, data2, equal_var=False)
    print(ttest_stats)
    print(welch_stats)
    np.savez(op.join(root_fol, 'cortical_thickness_ttest.npz'), names=labels, data=ttest_stats,
             title='cortical thickness ttest', data_min=0, data_max=1, cmap='YlOrRd')
    np.savez(op.join(root_fol, 'cortical_thickness_welch.npz'), names=labels, data=welch_stats,
             title='cortical thickness welch', data_min=0, data_max=1, cmap='YlOrRd')


if __name__ == '__main__':
    root_fol = [f for f in ['/homes/5/npeled/space1/Cinthya/', '/home/npeled/Documents/Cinthya/'] if op.isdir(f)][0]
    data = read_xlsx(op.join(root_fol, 'COLBOS_CT_Database.xlsx'))
    ttest(data, root_fol)
