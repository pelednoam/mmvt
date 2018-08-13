from pyexcel_ods import get_data
import os.path as op
import glob
import numpy as np


def read_subject_results(subject, root):
    sub_results_fname = op.join(root, '{}_electrodes_num.ods'.format(subject))
    data = get_data(sub_results_fname)['Sheet1']
    all_extra, groups_num, elecs_num, all_hits = 0, 0, 0, 0
    all_hits_prob = []
    if isinstance(data[0][0], str):
        del data[0]
    for line in data:
        if len(line) == 0 or line[0] == '':
            continue
        try:
            elecs_inside = int(line[3])
            found_by_mmvt = int(line[4])
            if len(line) > 5:
                non_real_electrodes = int(line[5])
                found_by_mmvt -= non_real_electrodes
            elecs_num += elecs_inside
            groups_num += 1
            hits = found_by_mmvt if found_by_mmvt <= elecs_inside else elecs_inside
            all_hits += hits
            extra = max(found_by_mmvt - elecs_inside, 0)
            all_extra += extra
            hits_prob = (hits / elecs_inside) * 100
            all_hits_prob.append(hits_prob)
            # print('{}, {:.2f}% found, {} extra'.format(line[1], hits_prob, extra))
        except:
            # print('Error with {}'.format(line))
            continue
    if elecs_num == 0 or groups_num == 0:
        # print('{}: No electrodes/groups!'.format(subject))
        return False, 0, 0, 0, 0, 0
    else:
        found = sum(all_hits_prob) / groups_num
        print('{}: {:.2f}% found, {:.2f}% extra'.format(subject, found, extra))
        return True, found, all_hits, all_extra, groups_num, elecs_num


def read_missing_extra_groups(root):
    sub_results_fname = op.join(root, 'missing-extra_groups.ods')
    data = get_data(sub_results_fname)['Sheet1']
    extra_groups, missing_groups = [], []
    if isinstance(data[0][1], str):
        del data[0]
    for line in data:
        if len(line) == 0 or line[0] == '':
            continue
        if len(line) > 1 and line[1] != '':
            extra_groups.append(line[1])
        if len(line) > 2 and line[2] != '':
            missing_groups.append(line[2])
    return extra_groups, missing_groups


def run(root):
    good_subjects = 0
    subjects = [op.basename(f).split('_')[0] for f in glob.glob(op.join(root, '*_electrodes_num.ods'))]
    all_found, all_hits, all_extra, all_extra_probs, all_groups, all_electrodes = \
        [], [], [], [], [], []
    for s in subjects:
        success, found, hits, extra, groups_num, elecs_num = read_subject_results(s, root)
        if success:
            all_found.append(found)
            all_hits.append(hits)
            all_extra.append(extra)
            all_extra_probs.append((extra / elecs_num) * 100)
            all_groups.append(groups_num)
            all_electrodes.append(elecs_num)
            good_subjects += 1
    all_groups_num, all_electrodes_num = sum(all_groups), sum(all_electrodes)
    print('{} good subjects'.format(good_subjects))
    print('On average: {:.2f} groups ({} std), {:.2f} electrodes ({} std)'.format(
        np.mean(all_groups), np.std(all_groups), np.mean(all_electrodes), np.std(all_electrodes)))
    print('Total: {:.2f}% found ({:.2f} std) ({}/{}), {:.2f} extra ({:.2f} std, {} total ({:.2f})'.format(
        np.mean(all_found), np.std(all_found), sum(all_hits), all_electrodes_num, np.mean(all_extra), np.std(all_extra),
        sum(all_extra), (sum(all_extra) / all_electrodes_num) * 100))
    extra_groups, missing_groups = read_missing_extra_groups(root)
    # print(extra_groups, missing_groups)
    extra_groups_num = sum(extra_groups)
    extra_groups_prob = (extra_groups_num / (all_groups_num + extra_groups_num)) * 100
    print('Extra groups were found: {:.2f}% ({}/{})'.format(
        extra_groups_prob, extra_groups_num, all_groups_num + extra_groups_num))


if __name__ == '__main__':
    root = [d for d in ['/cluster/neuromind/Natalia/electrodes',
                        '/home/npeled/Documents/finding_electrodes_in_ct/electrodes']
            if op.isdir(d)][0]
    run(root)