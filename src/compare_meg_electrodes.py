import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from preproc_for_blender import SUBJECTS_DIR, BLENDER_ROOT_DIR
from meg_preproc import FREE_SURFER_HOME, SUBJECTS_MEG_DIR, TASKS, TASK_ECR, TASK_MSIT
import utils


def load_data(subject, region, electrode, inverse_methods, task, meg_conditions, meg_elec_conditions_translate,
              all_vertices=False, do_plot=False, normalize_data=True):
    d = np.load(os.path.join(BLENDER_SUBJECT_DIR, 'electrodes{}_data.npz'.format('_bipolar' if bipolar else '')))
    data, names, elecs_conditions = d['data'], d['names'], d['conditions']
    index = np.where(names==electrode)[0][0]
    elec_data_mat = data[index]
    elec_data = {}
    for cond in meg_conditions:
        ind = meg_elec_conditions_translate[cond]
        elec_data[cond] = elec_data_mat[:, ind]
        if normalize_data:
            elec_data[cond] = elec_data[cond] * 1.0/max(elec_data[cond])
        if do_plot:
            plt.figure()
            plt.plot(elec_data[cond])
            plt.title('{}-{}'.format(cond,electrode))
            plt.show()

    sub_cortical, _ = utils.get_numeric_index_to_label(region, None, FREE_SURFER_HOME)
    subject_meg_fol = os.path.join(SUBJECTS_MEG_DIR, TASKS[task], subject)

    meg_data = defaultdict(dict)
    for inverse_method in inverse_methods:
        for cond in meg_conditions:
            meg_data_file_name = '{}-{}-{}{}.npy'.format(cond, sub_cortical, inverse_method, '-all-vertices' if all_vertices else '')
            meg_data[cond][inverse_method] = np.load(os.path.join(subject_meg_fol, 'subcorticals', meg_data_file_name))
            if normalize_data:
                if all_vertices:
                    for vert in range(meg_data[cond][inverse_method].shape[0]):
                        meg_data[cond][inverse_method][vert] = \
                            meg_data[cond][inverse_method][vert] * 1.0/max(meg_data[cond][inverse_method][vert])
                else:
                    meg_data[cond][inverse_method] = meg_data[cond][inverse_method] * 1.0/max(meg_data[cond][inverse_method])
            if do_plot:
                plt.figure()
                plt.plot(meg_data[cond][inverse_method])
                plt.title('{}-{}'.format(cond,inverse_method))
                plt.show()
    return meg_data, elec_data


def plot_activation(meg_data_all, elec_data, electrode, from_t, to_t, all_vertices=False):
    xaxis = range(from_t, to_t)
    T = len(xaxis)
    for cond in meg_data_all.keys():
        f, axs = plt.subplots(len(meg_data_all[cond].keys()), sharex=True)#, sharey=True)
        for ind, ((inverse_method, meg_data), ax) in enumerate(zip(meg_data_all[cond].iteritems(), axs)):
            if all_vertices:
                for vert in range(meg_data.shape[0]):
                    if vert == 0:
                        ax.plot(xaxis, meg_data[vert, :T], label=inverse_method, color='r')
                    else:
                        ax.plot(xaxis, meg_data[vert, :T], color='r')
                ax.plot(xaxis, np.mean(meg_data[:, :T], 0), label=inverse_method, color='k')
            else:
                ax.plot(xaxis, meg_data[:T], label=inverse_method, color='r')
            ax.plot(xaxis, elec_data[cond][:T], label=electrode, color='b')
            ax.axvline(x=0, linestyle='--', color='k')
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.legend()
            ax.set_title('{}-{}'.format(cond, inverse_method))
    plt.show()


if __name__ == '__main__':
    subject = 'mg78'
    subject_ep = 'ep001'
    task = TASK_MSIT
    subject_dir = os.path.join(SUBJECTS_DIR, subject)
    BLENDER_SUBJECT_DIR = os.path.join(BLENDER_ROOT_DIR, subject)

    meg_conditions = ['interference', 'neutral']
    elec_conditions = ['noninterference', 'interference']
    meg_elec_conditions_translate = {'interference':1, 'neutral':0}
    from_t, to_t = -500, 2000
    all_vertices = False
    bipolar = False

    inverse_methods = ['dSPM', 'MNE', 'sLORETA'] #, 'dics', 'lcmv', 'rap_music']
    electrode = 'LAT2-LAT1' if bipolar else 'LAT3'
    region = 'Left-Hippocampus'
    meg_data, elec_data = load_data(subject_ep, region, electrode, inverse_methods, task, meg_conditions,
        meg_elec_conditions_translate, do_plot=False, all_vertices=all_vertices)
    plot_activation(meg_data, elec_data, electrode, from_t, to_t, all_vertices=all_vertices)