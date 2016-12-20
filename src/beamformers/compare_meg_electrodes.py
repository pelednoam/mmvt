import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from preproc_for_blender import SUBJECTS_DIR, BLENDER_ROOT_DIR
from meg_preproc import FREESURFER_HOME, SUBJECTS_MEG_DIR, TASKS, TASK_ECR, TASK_MSIT
import utils


def load_all_electrodes_data(root_fol, bipolar):
    d = np.load(os.path.join(root_fol, 'electrodes{}_data.npz'.format('_bipolar' if bipolar else '')))
    data, names, elecs_conditions = d['data'], d['names'], d['conditions']
    return data, names, elecs_conditions


def load_electrode_data(root_fol, electrode, bipolar, meg_conditions, meg_elec_conditions_translate,
        from_t=None, to_t=None, normalize_data=True, positive=False, do_plot=False):
    data, names, elecs_conditions = load_all_electrodes_data(root_fol, bipolar)
    if electrode not in names:
        raise Exception('{} not in {}'.format(electrode, os.path.join(root_fol, 'electrodes{}_data.npz'.format('_bipolar' if bipolar else ''))))
    index = np.where(names==electrode)[0][0]
    elec_data_mat = data[index]
    elec_data = {}
    for cond in meg_conditions:
        ind = meg_elec_conditions_translate[cond]
        data = elec_data_mat[:, ind]
        if not from_t is None and not to_t is None:
            data = data[from_t: to_t]
        if positive:
            data = data - min(data)
        if normalize_data:
            data = data * 1.0/max(data)
        elec_data[cond] = data
        if do_plot:
            plt.figure()
            plt.plot(elec_data[cond])
            plt.title('{}-{}'.format(cond,electrode))
            plt.show()
    return elec_data


def load_meg_data(subject, region, inverse_methods, task, normalize_data=True, all_vertices=False, do_plot=False):
    sub_cortical, _ = utils.get_numeric_index_to_label(region, None, FREESURFER_HOME)
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
    return meg_data



def plot_activation(meg_data_all, elec_data, electrode, from_t, to_t, inverse_methods, all_vertices=False):
    xaxis = range(from_t, to_t)
    T = len(xaxis)
    for cond in meg_data_all.keys():
        f, axs = plt.subplots(len(meg_data_all[cond].keys()), sharex=True)#, sharey=True)
        for ind, ((inverse_method), ax) in enumerate(zip(inverse_methods, axs)):
            meg_data = meg_data_all[cond][inverse_method]
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
            plt.setp(ax.get_legend().get_texts(), fontsize='20')
            ax.set_title('{}-{}'.format(cond, inverse_method), fontsize=20)
            plt.xlabel('Time(ms)', fontsize=20)
    plt.show()


def plot_activation_one_plot(meg_data_all, elec_data, electrode, from_t, to_t):
    xaxis = range(from_t, to_t)
    T = len(xaxis)
    # colors = utils.get_spaced_colors(len(meg_data_all[meg_data_all.keys()[0]].keys()))
    colors = ['g', 'r', 'c', 'm', 'y', 'b']
    for cond in meg_data_all.keys():
        plt.figure()
        for ind, (inverse_method, meg_data) in enumerate(meg_data_all[cond].iteritems()):
            plt.plot(xaxis, meg_data[:T], label=inverse_method, color=colors[ind])
        plt.plot(xaxis, elec_data[cond][:T], label='electrode', color='k')
        plt.axvline(x=0, linestyle='--', color='k')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    subject = 'mg78'
    subject_ep = 'ep001'
    task = TASK_MSIT
    subject_dir = os.path.join(SUBJECTS_DIR, subject)
    root_fol = os.path.join(BLENDER_ROOT_DIR, subject)

    meg_conditions = ['interference', 'neutral']
    elec_conditions = ['noninterference', 'interference']
    meg_elec_conditions_translate = {'interference':1, 'neutral':0}
    from_t, to_t = -500, 2000
    all_vertices = False
    bipolar = False

    inverse_methods = ['dSPM', 'MNE', 'sLORETA', 'dics', 'lcmv', 'rap_music']
    inverse_methods = ['sLORETA']
    electrode = 'LAT3-LAT2' if bipolar else 'LAT3'
    region = electrode #'Left-Hippocampus'
    normalize_data = True
    do_plot = False
    elec_data = load_electrode_data(root_fol, electrode, bipolar, meg_conditions, meg_elec_conditions_translate, normalize_data, do_plot)
    meg_data = load_meg_data(subject, region, inverse_methods, task, normalize_data, all_vertices, do_plot)
    plot_activation(meg_data, elec_data, electrode, from_t, to_t, inverse_methods, all_vertices=all_vertices)
    # plot_activation_one_plot(meg_data, elec_data, electrode, from_t, to_t)