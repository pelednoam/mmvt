#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 16:34:16 2017

@author: sx424
"""

# Repeating Linda's analysis

import numpy as np
import numpy.ma as ma
import scipy as spy
import scipy.io as sio
from scipy import signal
import os
import os.path as op
import math as m
import matplotlib.pyplot as plt
import pandas as p
import flexibility_LDrec

wl = 102

os.chdir('/cluster/neuromind/sx424/mem_flex')
laus125_labels_lh_sandya = [line.rstrip() for line in open('fmri_laus125_lh.txt')]
laus125_labels_rh_sandya = [line.rstrip() for line in open('fmri_laus125_rh.txt')]
laus125_labels_lh_sandya = [s + '-lh' for s in laus125_labels_lh_sandya]
laus125_labels_rh_sandya = [s + '-rh' for s in laus125_labels_rh_sandya]
laus125_labels_sandya = laus125_labels_lh_sandya + laus125_labels_rh_sandya

# Find the ROIs of interest - PCC's and Hipp
# 4 - entorhinal_1-lh (Hipp)
# 8 - isthmuscingulate_1-lh (PCC)

# 115 - entorhinal_1-rh (Hipp)
# 119 - isthmuscingulate_1-rh (PCC)

ROIs = np.array([4, 8, 115, 119])

# Running Linda's patients to check
os.chdir('/cluster/neuromind/sx424/mem_flex')
subject_list = ['nmr00474', 'nmr00502', 'nmr00515', 'nmr00603', 'nmr00609', 'nmr00626', 'nmr00629', 'nmr00650',
                'nmr00657', 'nmr00669', 'nmr00674', 'nmr00681', 'nmr00683', 'nmr00692', 'nmr00698', 'nmr00710']

# Get neuropsych scores from spreadsheet
neuropsych_scores2 = p.read_excel(
    '/cluster/neuromind/sx424/subject_info/StufflebeamLabDataba_DATA_LABELS_2017-01-27_1132.xlsx',
    sheetname='Necessary scores', header=None, skiprows={0})
subjects_master = neuropsych_scores2.loc[:, 0]
laterality = neuropsych_scores2.loc[:, 1]
values = neuropsych_scores2.loc[:, 4:].astype(float)
master_grouping = (np.sum((values <= 5).astype(int), axis=1) > 0).astype(int);
# disturbed = 1, preserved = 0
inds = np.where(np.in1d(subjects_master, subject_list))[0]
subject_list = subjects_master[inds]
subject_groups = master_grouping[inds]

ROI_values_dyn = np.zeros((ROIs.shape[0], len(subject_list)), 'float')
ROI_values_stat = np.zeros((ROIs.shape[0], len(subject_list)), 'float')
for s in range(len(subject_list)):
    subject = subject_list[s];
    #    path = 'subjects/' + subject + '/rest/laus125_001.txt'
    path = '/cluster/neuromind/douw/scans/patients_mri_epochs_final/laus125/' + subject + '_laus125_mri.txt'
    laus125_raw = np.loadtxt(path)
    conn, avg_conn, std_conn, cv_conn, dFC, dFC_unnorm, stat_conn = flexibility_LDrec.flexibility_LDrec(laus125_raw.T,
                                                                                                        False, 3, wl,
                                                                                                        12)

    # Extract the relevant dFC values and save
    ROI_values_dyn[:, s] = dFC_unnorm[ROIs].T
    ROI_values_stat[:, s] = stat_conn[ROIs].T

    # Save everything back in relevant folder
    savepath = 'from_Linda_data/' + subject + '/'
    np.save(savepath + 'conn' + str(wl) + '.npy', conn)
    np.save(savepath + 'avg_conn' + str(wl) + '.npy', avg_conn)
    np.save(savepath + 'std_conn' + str(wl) + '.npy', std_conn)
    np.save(savepath + 'cv_conn' + str(wl) + '.npy', cv_conn)
    np.save(savepath + 'dFC' + str(wl) + '.npy', dFC)
    np.save(savepath + 'dFC_unnorm' + str(wl) + '.npy', dFC_unnorm)
    np.save(savepath + 'stat_conn' + str(wl) + '.npy', stat_conn)

np.save('from_Linda_data/ROI_values_dyn' + str(wl) + '.npy', ROI_values_dyn)
np.save('from_Linda_data/ROI_values_stat' + str(wl) + '.npy', ROI_values_stat)

# load the dFC values
# Should already be there


# Extract subjects in the order you want - subject_list order
# Get order of inds from subject_list and then just pull those from subjects_master to get vector in order (subject_groups)

# Enter matrix with percentile scores on relevant memory tests
# neuropsych_scores = np.array([[30,67,float('NaN'),99,79,float('NaN'),float('NaN'),37,75,50,84,1,16],
#                              [14,50,float('NaN'),1,2,float('NaN'),float('NaN'),16,25,50,50,50,37],
#                              [7,float('NaN'),float('NaN'),5,37,float('NaN'),float('NaN'),63,50,16,25,75,50],
#                              [float('NaN'),float('NaN'),50,float('NaN'),float('NaN'),50,84,float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN')],
#                              [float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),27,73,float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN')],
#                              [6,42,float('NaN'),63,18,float('NaN'),float('NaN'),16,25,25,25,25,50],
#                              [21,50,float('NaN'),55,18,float('NaN'),float('NaN'),37,37,37,63,9,9],
#                              [float('NaN'),float('NaN'),16,float('NaN'),float('NaN'),76,73,float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN')],
#                              [2,50,float('NaN'),45,79,float('NaN'),float('NaN'),75,75,16,25,91,84],
#                              [23,50,float('NaN'),3,18,float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN')],
#                              [1,17,float('NaN'),73,58,float('NaN'),float('NaN'),float('NaN'),float('NaN'),1,5,float('NaN'),float('NaN')],
#                              [float('NaN'),float('NaN'),1,float('NaN'),float('NaN'),1,1,5,1,float('NaN'),float('NaN'),float('NaN'),float('NaN')],
#                              [float('NaN'),float('NaN'),8,float('NaN'),float('NaN'),12,1,75,50,float('NaN'),float('NaN'),float('NaN'),float('NaN')],
#                              [1,0,float('NaN'),1,7,float('NaN'),float('NaN'),16,2,float('NaN'),float('NaN'),2,2],
#                              [float('NaN'),float('NaN'),25,float('NaN'),float('NaN'),1,1,16,25,float('NaN'),float('NaN'),float('NaN'),float('NaN')],
#                              [95,75,float('NaN'),45,79,float('NaN'),float('NaN'),37,9,50,50,84,84]])

# Pt vector based on if any are less than 5th percentile
# condition = (neuropsych_scores<=5)*1.0
# subject_groups = (condition.sum(axis=1)>0)*1.0
# disturbed = 1, preserved = 0
disturbed_inds = np.array(np.where(subject_groups == 1))
preserved_inds = np.array(np.where(subject_groups == 0))

# Get left right for each patient
# left is 0, right is 1
laterality = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Separate by contralateral and ipsilateral - ipsi is left, contra is right

# Mann-Whitney tests between groups - p<0.0125 - bonferroni correction for 4 ROIs
# spy.mannwhitneyu(x,y)
ROI_values_dyn_disturbed = np.squeeze(ROI_values_dyn[:, disturbed_inds])
ROI_values_stat_disturbed = np.squeeze(ROI_values_stat[:, disturbed_inds])
ROI_values_dyn_preserved = np.squeeze(ROI_values_dyn[:, preserved_inds])
ROI_values_stat_preserved = np.squeeze(ROI_values_stat[:, preserved_inds])

mann_whitney_results_dyn = np.zeros((ROI_values_dyn.shape[0], 2), 'float')
mann_whitney_results_stat = np.zeros((ROI_values_stat.shape[0], 2), 'float')

# Summary: preserved mean, preserved std, disturbed mean, disturbed std
ROI_values_dyn_summary = np.vstack((np.mean(ROI_values_dyn_preserved, axis=1), np.std(ROI_values_dyn_preserved, axis=1),
                                    np.mean(ROI_values_dyn_disturbed, axis=1),
                                    np.std(ROI_values_dyn_disturbed, axis=1))).T
ROI_values_stat_summary = np.vstack((np.mean(ROI_values_stat_preserved, axis=1),
                                     np.std(ROI_values_stat_preserved, axis=1),
                                     np.mean(ROI_values_stat_disturbed, axis=1),
                                     np.std(ROI_values_stat_disturbed, axis=1))).T

for i in range(ROI_values_dyn.shape[0]):
    mann_whitney_results_dyn[i, :] = spy.stats.mannwhitneyu(ROI_values_dyn_disturbed[i, :],
                                                            ROI_values_dyn_preserved[i, :])
    mann_whitney_results_stat[i, :] = spy.stats.mannwhitneyu(ROI_values_stat_disturbed[i, :],
                                                             ROI_values_stat_preserved[i, :])