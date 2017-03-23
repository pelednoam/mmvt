import os.path as op
import numpy as np
from src.utils.make_movie import create_movie
from src.utils import preproc_utils as pu
from src.utils import color_maps_utils as cmu

SUBJECTS_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()


def mg78_electrodes_coherence_meg(dpi, bitrate, pics_type, show_first_pic, n_jobs):
    fol = '/home/noam/Pictures/mmvt/movie1'
    fol2 = '/home/noam/Pictures/mmvt/movie2'
    data_to_show_in_graph = ('electrodes', 'coherence')
    video_fname = 'mg78_elecs_coh_meg.mp4'
    cb_title = 'MEG dSPM difference'
    time_range = range(2500)
    xticks = range(-500, 2500, 500)
    ylim = ()
    ylabels = []
    xticklabels = []
    xlabel = ''
    cb_data_type = 'meg'
    fps = 10
    cb_min_max_eq = True
    color_map = 'jet'
    create_movie(time_range, xticks, fol, dpi, fps, video_fname, cb_data_type, data_to_show_in_graph, cb_title,
        cb_min_max_eq, color_map, bitrate, fol2, ylim, ylabels, xticklabels, xlabel, pics_type, show_first_pic, n_jobs)


def fsaverage_ttest(dpi, bitrate, pics_type, show_first_pic, n_jobs):
    fol = '/home/noam/Pictures/mmvt/fsaverage'
    fol2 = ''
    data_to_show_in_graph = ('meg')
    video_fname = 'fsaverage_meg_ttest.mp4'
    cb_title = 'MEG t values'
    time_range = range(1000)
    xticks = range(0, 1000, 100)
    ylim = ()
    ylabels = ['MEG t-values']
    xticklabels = []
    xlabel = ''
    cb_data_type = 'meg'
    fps = 10
    cb_min_max_eq = True
    color_map = 'jet'
    create_movie(time_range, xticks, fol, dpi, fps, video_fname, cb_data_type, data_to_show_in_graph, cb_title,
        cb_min_max_eq, color_map, bitrate, fol2, ylim, ylabels, xticklabels, xlabel, pics_type, show_first_pic, n_jobs)


def meg_labels(dpi, bitrate, pics_type, show_first_pic, n_jobs):
    fol = '/home/noam/Pictures/mmvt/movie1'
    fol2 = ''
    data_to_show_in_graph = ('meg_labels')
    video_fname = 'mg78_labels_demo.mp4'
    cb_title = 'MEG activity'
    time_range = range(2500)
    xticks = range(-500, 2500, 500)
    ylim = ()
    ylabels = ['MEG activity']
    xticklabels = []
    xlabel = ''
    cb_data_type = 'meg_labels'
    fps = 10
    cb_min_max_eq = True
    color_map = 'jet'
    create_movie(time_range, xticks, fol, dpi, fps, video_fname, cb_data_type, data_to_show_in_graph, cb_title,
        cb_min_max_eq, color_map, bitrate, fol2, ylim, ylabels, xticklabels, xlabel, pics_type, show_first_pic, n_jobs)


def meg(dpi, bitrate, pics_type, show_first_pic, n_jobs):
    fol = '/cluster/neuromind/npeled/mmvt/ESZC25/movies/movie2'
    fol2 = ''
    data_to_show_in_graph = ('meg')
    video_fname = 'ESZC25.mp4'
    cb_title = 'MEG activity'
    time_range = range(0, 300)
    xticks = range(0, 300, 50)
    ylim = ()
    ylabels = ['MEG activity']
    xticklabels = []
    xlabel = ''
    cb_data_type = 'meg'
    fps = 10
    cb_min_max_eq = True
    color_map = 'jet'
    create_movie(time_range, xticks, fol, dpi, fps, video_fname, cb_data_type, data_to_show_in_graph, cb_title,
        cb_min_max_eq, color_map, bitrate, fol2, ylim, ylabels, xticklabels, xlabel, pics_type, show_first_pic, n_jobs)


def pp009_vs_healthy_coherence(dpi, bitrate, pics_type, show_first_pic, n_jobs):
    # todo: this example doesn't work
    fol = '/home/noam/Videos/mmvt/meg_con/healthy'
    fol2 = '/home/noam/Videos/mmvt/meg_con/pp009'
    data_to_show_in_graph = ('coherence', 'coherence2')
    video_fname = 'pp009_healthy_meg_coh.mp4'
    cb_title = ''
    ms_before_stimuli, labels_time_dt = 0, 1
    time_range = range(11)
    ylabels = ['Healthy', 'pp009']
    ylim = ()
    # xticklabels = ['', 'Risk onset', '', '', '', 'Reward onset', '', '', '', 'Shock?', '']
    xticklabels = ['Risk onset','Reward onset','Shock?']
    xticks = range(3)
    xlabel = ''
    cb_data_type = ''
    fps = 10
    # duplicate_frames(fol, 30)
    cb_min_max_eq = True
    color_map = 'jet'
    create_movie(time_range, xticks, fol, dpi, fps, video_fname, cb_data_type, data_to_show_in_graph, cb_title,
        cb_min_max_eq, color_map, bitrate, fol2, ylim, ylabels, xticklabels, xlabel, pics_type, show_first_pic, n_jobs)


def mg99_stim(dpi, bitrate, pics_type, show_first_pic, n_jobs):
    fol = '/home/noam/Pictures/mmvt/mg99/lvf4-3_1'
    fol2 = ''
    data_to_show_in_graph = 'stim'
    video_fname = 'mg99_LVF4-3_stim.mp4'
    cb_title = 'Electrodes PSD'
    ylabels = ['Electrodes PSD']
    time_range = np.arange(-1, 1.5, 0.01)
    xticks = [-1, -0.5, 0, 0.5, 1]
    xticklabels = [(-1, 'stim onset'), (0, 'end of stim')]
    ylim = (0, 500)
    xlabel = 'Time(s)'
    cb_data_type = 'stim'
    cb_min_max_eq = False
    color_map = 'OrRd'
    fps = 10
    create_movie(time_range, xticks, fol, dpi, fps, video_fname, cb_data_type, data_to_show_in_graph, cb_title,
        cb_min_max_eq, color_map, bitrate, fol2, ylim, ylabels, xticklabels, xlabel, pics_type, show_first_pic, n_jobs)


def mg99_stim_srouces(dpi, bitrate, pics_type, show_first_pic, n_jobs):
    root = '/home/noam/Pictures/mmvt/mg99'
    fol = op.join(root, 'mg99_stim_lvf4-3_laus250_1_1')
    fol2 = op.join(root, 'mg99_stim_lvf4-3_laus250_1_2')
    data_to_show_in_graph = ['stim']#, 'stim_sources']
    video_fname = 'mg99_LVF4-3_stim_srouces.mp4'
    cb_title = 'Electrodes PSD'
    ylabels = ['Electrodes PSD']
    time_range = np.arange(-1, 2, 0.01)
    xticks = [-1.5, -1, -0.5, 0, 0.5, 1]
    xticklabels = [(-1, 'stim onset'), (0, 'end of stim')]
    ylim = (0, 10)
    xlabel = 'Time(s)'
    cb_data_type = 'stim'
    cb_min_max_eq = False
    color_map = 'OrRd'
    fps = 10
    create_movie(time_range, xticks, fol, dpi, fps, video_fname, cb_data_type, data_to_show_in_graph, cb_title,
        cb_min_max_eq, color_map, bitrate, fol2, ylim, ylabels, xticklabels, xlabel, pics_type, show_first_pic, n_jobs)


def fMRI_4D(dpi, bitrate, pics_type, show_first_pic, n_jobs):
    fol = op.join(MMVT_DIR, 'nmr00698' ,'movies', 'fmri')
    fol2 = ''
    data_to_show_in_graph = ('fmri')
    video_fname = 'nmr00698_fmri_4D.mp4'
    cb_title = 'fMRI'
    time_range = range(0, 161)
    xticks = range(0, 161, 20)
    ylim = ()
    ylabels = ['Labels mean intensity']
    xticklabels = [(xt, str(int(xt*3/60))) for xt in xticks][1:]
    xlabel = 'Time (m)'
    cb_data_type = 'fmri'
    fps = 5
    cb_min_max_eq = True
    color_map = 'BuPu_YlOrRd'
    show_animation = False
    overwrite = True
    create_movie(time_range, xticks, fol, dpi, fps, video_fname, cb_data_type, data_to_show_in_graph, cb_title,
        cb_min_max_eq, color_map, bitrate, fol2, ylim, ylabels, xticklabels, xlabel, pics_type, show_first_pic,
        show_animation, overwrite, n_jobs)


def fmri_cor_connectivity(dpi, bitrate, pics_type, show_first_pic, n_jobs):
    fol = op.join(MMVT_DIR, 'nmr00698' ,'movies', 'labels_connectivity')
    fol2 = ''
    data_to_show_in_graph = ('labels_connectivity')
    video_fname = 'nmr00698_fmri_cor.mp4'
    cb_title = 'fMRI corr'
    time_range = range(0, 161)
    xticks = range(0, 161, 20)
    ylim = ()
    ylabels = ['Labels corr']
    xticklabels = [(xt, str(int(xt*3/60))) for xt in xticks][1:]
    xlabel = 'Time (m)'
    cb_data_type = 'labels_connectivity'
    fps = 5
    cb_min_max_eq = True
    color_map = 'jet'
    show_animation = True
    overwrite = True
    create_movie(time_range, xticks, fol, dpi, fps, video_fname, cb_data_type, data_to_show_in_graph, cb_title,
        cb_min_max_eq, color_map, bitrate, fol2, ylim, ylabels, xticklabels, xlabel, pics_type, show_first_pic,
        show_animation, overwrite, n_jobs)


if __name__ == '__main__':
    dpi = 100
    bitrate = 5000
    pics_type = 'png'
    show_first_pic = False
    n_jobs = 1

    # mg99_stim(dpi, bitrate, pics_type, show_first_pic, n_jobs)
    # mg99_stim_srouces(dpi, bitrate, pics_type, show_first_pic, n_jobs)
    fmri_cor_connectivity(dpi, bitrate, pics_type, show_first_pic, n_jobs)