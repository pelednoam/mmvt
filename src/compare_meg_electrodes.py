import os
import numpy as np
from collections import defaultdict
from preproc_for_blender import SUBJECTS_DIR, BLENDER_ROOT_DIR
from meg_preproc import FREE_SURFER_HOME, SUBJECTS_MEG_DIR, TASKS, TASK_ECR, TASK_MSIT
import utils

def compare(subject, region, electrode, inverse_methods, task, conditions):
    d = np.load(os.path.join(BLENDER_SUBJECT_DIR, 'electrodes_data.npz'))
    data, names, elecs_conditions = d['data'], d['names'], d['conditions']
    index = np.where(names==electrode)[0][0]
    elec_data = data[index]

    sub_cortical, _ = utils.get_numeric_index_to_label(region, None, FREE_SURFER_HOME)
    subject_meg_fol = os.path.join(SUBJECTS_MEG_DIR, TASKS[task], subject)

    all_vertices = False
    activity = defaultdict(dict)
    for inverse_method in inverse_methods:
        for cond in conditions:
            meg_data_file_name = '{}-{}-{}{}.npy'.format(cond, sub_cortical, inverse_method, '-all-vertices' if all_vertices else '')
            activity[inverse_method][cond] = np.load(os.path.join(subject_meg_fol, 'subcorticals', meg_data_file_name))

    print("sdf")

if __name__ == '__main__':
    subject = 'mg78'
    subject_ep = 'ep001'
    task = TASK_MSIT
    subject_dir = os.path.join(SUBJECTS_DIR, subject)
    BLENDER_SUBJECT_DIR = os.path.join(BLENDER_ROOT_DIR, subject)

    conditions = ['interference', 'neutral']
    inverse_methods = ['dSPM', 'lcmv', 'dics']
    electrode = 'LAT1'
    region = 'Left-Hippocampus'
    compare(subject_ep, region, electrode, inverse_methods, task, conditions)