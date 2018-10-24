import os.path as op
import numpy as np
from itertools import product
import shutil
import os
import os
import time
import glob
from src.utils import utils
from src.utils import labels_utils as lu
from src.preproc import anatomy as anat
from src.preproc import meg as meg
from src.preproc import connectivity
from collections import defaultdict
import warnings
import matplotlib.pyplot as plt

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')


def prepare_files(args):
    # todo: should look in the dict for files locations
    ret = {}
    for subject in args.subject:
        ret[subject] = True
        for task in args.tasks:
            fol = utils.make_dir(op.join(MEG_DIR, task, subject))
            local_epo_fname = op.join(fol, args.epo_template.format(subject=subject, task=task))
            local_raw_fname = op.join(fol, '{}_{}-raw.fif'.format(subject, task))
            if not args.overwrite and (op.islink(local_epo_fname) or op.isfile(local_epo_fname)) and \
                    (op.islink(local_raw_fname) or op.isfile(local_raw_fname)):
                continue

            if op.islink(local_epo_fname) or op.isfile(local_epo_fname) and args.overwrite_local_files:
                os.remove(local_epo_fname)
            if not op.islink(local_epo_fname) and not op.isfile(local_epo_fname):
                remote_epo_fname = op.join(args.meg_dir, subject, args.epo_template.format(subject=subject, task=task))
                if not op.isfile(remote_epo_fname):
                    print('{} does not exist!'.format(remote_epo_fname))
                    ret[subject] = False
                    continue
                print('Creating a link {} -> {}'.format(remote_epo_fname, local_epo_fname))
                utils.make_link(remote_epo_fname, local_epo_fname)

            if op.islink(local_raw_fname) or op.isfile(local_raw_fname) and args.overwrite_local_files:
                os.remove(local_raw_fname)
            if not op.islink(local_raw_fname) and not op.isfile(local_raw_fname):
                remote_raw_fname = op.join(
                    utils.get_parent_fol(args.meg_dir), 'raw_preprocessed', subject,
                    args.raw_template.format(subject=subject, task=task))
                if not op.isfile(remote_raw_fname):
                    print('{} does not exist!'.format(remote_raw_fname))
                    ret[subject] = False
                    continue
                print('Creating a link {} -> {}'.format(remote_raw_fname, local_raw_fname))
                utils.make_link(remote_raw_fname, local_raw_fname)
        ret[subject] = ret[subject] and (op.isfile(local_epo_fname) or op.islink(local_epo_fname)) and \
                       (op.isfile(local_raw_fname) or op.islink(local_raw_fname))
    print('Good subjects:')
    print([s for s, r in ret.items() if r])
    print('Bad subjects:')
    print([s for s, r in ret.items() if not r])


def anatomy_preproc(args, subject=''):
    args = anat.read_cmd_args(dict(
        subject=args.subject if subject == '' else subject,
        remote_subject_dir='/autofs/space/lilli_001/users/DARPA-Recons/{subject}',
        # high_level_atlas_name='darpa-atlas',
        # function='create_annotation,create_high_level_atlas',
        function='create_annotation',
        overwrite_fs_files=args.overwrite,
        atlas='laus125',
        ignore_missing=False
    ))
    anat.call_main(args)


def get_empty_fnames(subject, tasks, args, overwrite=False):
    utils.make_dir(op.join(MEG_DIR, subject))
    utils.make_link(op.join(args.remote_subject_dir.format(subject=subject), 'bem'),
                    op.join(MEG_DIR, subject, 'bem'), overwrite=overwrite)
    for task in tasks:
        utils.make_dir(op.join(MEG_DIR, task, subject))
        utils.make_link(op.join(MEG_DIR, subject, 'bem'), op.join(MEG_DIR, task, subject, 'bem'), overwrite=overwrite)
    utils.make_link(op.join(MEG_DIR, subject, 'bem'), op.join(SUBJECTS_DIR, subject, 'bem'), overwrite=overwrite)

    remote_meg_fol = op.join(args.remote_meg_dir, subject)
    csv_fname = op.join(remote_meg_fol, 'cfg.txt')
    empty_fnames, cors, days = '', '', ''

    if not op.isfile(csv_fname):
        print('No cfg file!')
        return '', '', ''
    days, empty_fnames, cors = {}, {}, {}
    for line in utils.csv_file_reader(csv_fname, ' '):
        for task in tasks:
            if line[4].lower() == task.lower():
                days[task] = line[2]
    # print(days)
    for line in utils.csv_file_reader(csv_fname, ' '):
        if line[4] == 'empty':
            for task in tasks:
                empty_fnames[task] = op.join(MEG_DIR, task, subject, '{}_empty_raw.fif'.format(subject))
                if op.isfile(empty_fnames[task]):
                    continue
                task_day = days[task]
                if line[2] == task_day:
                    empty_fname = op.join(remote_meg_fol, line[0].zfill(3), line[-1])
                    if not op.isfile(empty_fname):
                        raise Exception('empty file does not exist! {}'.format(empty_fname[task]))
                    utils.make_link(empty_fname, empty_fnames[task])
    cor_dir = op.join(args.remote_subject_dir.format(subject=subject), 'mri', 'T1-neuromag', 'sets')
    for task in tasks:
        if op.isfile(op.join(cor_dir, 'COR-{}-{}.fif'.format(subject, task.lower()))):
            cors[task] = op.join(cor_dir, 'COR-{}-{}.fif'.format('{subject}', task.lower()))
        elif op.isfile(op.join(cor_dir, 'COR-{}-day{}.fif'.format(subject, days[task]))):
            cors[task] = op.join(cor_dir, 'COR-{}-day{}.fif'.format('{subject}', days[task]))
    return empty_fnames, cors, days

#
# def calc_meg_epochs(args):
#     empty_fnames, cors, days = get_empty_fnames(args.subject[0], args.tasks, args)
#     times = (-2, 4)
#     for task in args.tasks:
#         args = meg.read_cmd_args(dict(
#             subject=args.subject, mri_subject=args.subject,
#             task=task,
#             remote_subject_dir='/autofs/space/lilli_001/users/DARPA-Recons/{subject}',
#             get_task_defaults=False,
#             fname_format='{}_{}_nTSSS-ica-raw'.format('{subject}', task.lower()),
#             empty_fname=empty_fnames[task],
#             function='calc_epochs,calc_evokes',
#             conditions=task.lower(),
#             data_per_task=True,
#             normalize_data=False,
#             t_min=times[0], t_max=times[1],
#             read_events_from_file=False, stim_channels='STI001',
#             use_empty_room_for_noise_cov=True,
#             n_jobs=args.n_jobs
#         ))
#         meg.call_main(args)


def meg_preproc_evoked(args):
    inv_method, em, atlas= 'dSPM', 'mean_flip', args.atlas
    # bands = dict(theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55], high_gamma=[65, 200])
    times = (-2, 4)
    subjects_with_error = []
    good_subjects = get_good_subjects(args)
    args.subject = good_subjects
    prepare_files(args)

    for subject in good_subjects:
        args.subject = subject
        empty_fnames, cors, days = get_empty_fnames(subject, args.tasks, args)
        input_fol = utils.make_dir(op.join(MEG_DIR, subject, 'labels_induced_power'))
        for task in args.tasks:

            # output_fname = op.join(
            #     MMVT_DIR, subject, 'meg', '{}_{}_{}_power_spectrum.npz'.format(task.lower(), inv_method, em))
            # if op.isfile(output_fname) and args.check_file_modification_time:
            #     file_mod_time = utils.file_modification_time_struct(output_fname)
            #     if file_mod_time.tm_year >= 2018 and (file_mod_time.tm_mon == 9 and file_mod_time.tm_mday >= 21) or \
            #             (file_mod_time.tm_mon > 9):
            #         print('{} already exist!'.format(output_fname))
            #         continue

            remote_epo_fname = op.join(args.meg_dir, subject, args.epo_template.format(subject=subject, task=task))
            local_epo_fname = op.join(MEG_DIR, task, subject, args.epo_template.format(subject=subject, task=task))
            if not op.isfile(local_epo_fname) and not op.isfile(remote_epo_fname):
                print('Can\'t find {}!'.format(local_epo_fname))
                continue
            if not op.isfile(local_epo_fname):
                utils.make_link(remote_epo_fname, local_epo_fname)

            meg_args = meg.read_cmd_args(dict(
                subject=args.subject, mri_subject=args.subject,
                task=task, inverse_method=inv_method, extract_mode=em, atlas=atlas,
                remote_subject_dir=args.remote_subject_dir, # Needed for finding COR
                get_task_defaults=False,
                fname_format=args.epo_template.format(subject=subject, task=task)[:-len('-epo.fif')],
                raw_fname=op.join(MEG_DIR, task, subject, args.raw_template.format(subject=subject, task=task)),
                epo_fname=local_epo_fname,
                empty_fname=empty_fnames[task] if empty_fnames != '' else '',
                function='make_forward_solution,calc_inverse_operator,calc_stc,calc_labels_avg_per_condition,calc_labels_min_max',
                conditions=task.lower(),
                cor_fname=cors[task].format(subject=subject) if cors != '' else '',
                average_per_event=False,
                data_per_task=True,
                pick_ori='normal', # very important for calculation of the power spectrum
                ica_overwrite_raw=False,
                normalize_data=False,
                t_min=times[0], t_max=times[1],
                read_events_from_file=False, stim_channels='STI001',
                use_empty_room_for_noise_cov=True,
                read_only_from_annot=False,
                # pick_ori='normal',
                check_for_channels_inconsistency = args.check_for_channels_inconsistency,
                overwrite_labels_power_spectrum = args.overwrite_labels_power_spectrum,
                overwrite_evoked=True,#args.overwrite,
                overwrite_fwd=args.overwrite,
                overwrite_inv=args.overwrite,
                overwrite_stc=True,#args.overwrite,
                overwrite_labels_data=True,#args.overwrite,
                n_jobs=args.n_jobs
            ))
            ret = meg.call_main(meg_args)
            # output_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'labels', 'labels_data'))
            # join_res_fol = utils.make_dir(op.join(utils.get_parent_fol(MMVT_DIR), 'msit-ecr', subject))
            # for res_fname in glob.glob(op.join(output_fol, '{}_labels_{}_{}_*_power.npz'.format(
            #         task.lower(), inv_method, em))):
            #     shutil.copyfile(res_fname, op.join(join_res_fol, utils.namebase_with_ext(res_fname)))
            if not ret:
                if args.throw:
                    raise Exception("errors!")
                else:
                    subjects_with_error.append(subject)


    good_subjects = [s for s in good_subjects if
           op.isfile(op.join(MMVT_DIR, subject, 'meg',
                             'labels_data_msit_{}_{}_{}_minmax.npz'.format(atlas, inv_method, em))) and
           op.isfile(op.join(MMVT_DIR, subject, 'meg',
                             'labels_data_ecr_{}_{}_{}_minmax.npz'.format(atlas, inv_method, em)))]
    print('Good subjects:')
    print(good_subjects)
    print('subjects_with_error:')
    print(subjects_with_error)


def meg_preproc_power(args):
    inv_method, em, atlas = 'dSPM', 'mean_flip', args.atlas
    # bands = dict(theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55], high_gamma=[65, 200])
    times = (-2, 4)
    subjects_with_error = []
    good_subjects = get_good_subjects(args)
    args.subject = good_subjects
    prepare_files(args)

    for subject in good_subjects:
        args.subject = subject
        empty_fnames, cors, days = get_empty_fnames(subject, args.tasks, args)
        input_fol = utils.make_dir(op.join(MEG_DIR, subject, 'labels_induced_power'))
        for task in args.tasks:

            # output_fname = op.join(
            #     MMVT_DIR, subject, 'meg', '{}_{}_{}_power_spectrum.npz'.format(task.lower(), inv_method, em))
            # if op.isfile(output_fname) and args.check_file_modification_time:
            #     file_mod_time = utils.file_modification_time_struct(output_fname)
            #     if file_mod_time.tm_year >= 2018 and (file_mod_time.tm_mon == 9 and file_mod_time.tm_mday >= 21) or \
            #             (file_mod_time.tm_mon > 9):
            #         print('{} already exist!'.format(output_fname))
            #         continue

            if not args.overwrite_output_files:
                output_fnames = glob.glob(
                    op.join(input_fol, '{}_*_{}_{}_{}_induced_power.npz'.format(task.lower(), atlas, inv_method, em)))
                overwrite = False
                for output_fname in output_fnames:
                    file_mod_time = utils.file_modification_time_struct(output_fname)
                    if file_mod_time.tm_year < 2018 or (file_mod_time.tm_mon == 10 and file_mod_time.tm_mday < 23) or \
                            (file_mod_time.tm_mon < 10):
                        overwrite = True

                if len(output_fnames) == 28:
                    print('{} has already all the results for {}'.format(subject, task))
                    continue

            remote_epo_fname = op.join(args.meg_dir, subject, args.epo_template.format(subject=subject, task=task))
            local_epo_fname = op.join(MEG_DIR, task, subject, args.epo_template.format(subject=subject, task=task))
            if not op.isfile(local_epo_fname) and not op.isfile(remote_epo_fname):
                print('Can\'t find {}!'.format(local_epo_fname))
                continue
            if not op.isfile(local_epo_fname):
                utils.make_link(remote_epo_fname, local_epo_fname)

            meg_args = meg.read_cmd_args(dict(
                subject=args.subject, mri_subject=args.subject,
                task=task, inverse_method=inv_method, extract_mode=em, atlas=atlas,
                # meg_dir=args.meg_dir,
                remote_subject_dir=args.remote_subject_dir, # Needed for finding COR
                get_task_defaults=False,
                fname_format=args.epo_template.format(subject=subject, task=task)[:-len('-epo.fif')],
                raw_fname=op.join(MEG_DIR, task, subject, args.raw_template.format(subject=subject, task=task)),
                epo_fname=local_epo_fname,
                empty_fname=empty_fnames[task] if empty_fnames != '' else '',
                function='make_forward_solution,calc_inverse_operator,calc_labels_induced_power',#,
                conditions=task.lower(),
                cor_fname=cors[task].format(subject=subject) if cors != '' else '',
                average_per_event=False,
                data_per_task=True,
                pick_ori='normal', # very important for calculation of the power spectrum
                # fmin=4, fmax=120, bandwidth=2.0,
                max_epochs_num=args.max_epochs_num,
                ica_overwrite_raw=False,
                normalize_data=False,
                fwd_recreate_source_space=True,
                t_min=times[0], t_max=times[1],
                read_events_from_file=False, stim_channels='STI001',
                use_empty_room_for_noise_cov=True,
                read_only_from_annot=False,
                # pick_ori='normal',
                overwrite_labels_induced_power=args.overwrite_output_files,
                overwrite_evoked=args.overwrite,
                overwrite_fwd=args.overwrite,
                overwrite_inv=args.overwrite,
                overwrite_stc=args.overwrite,
                overwrite_labels_data=args.overwrite,
                n_jobs=args.n_jobs
            ))
            ret = meg.call_main(meg_args)
            output_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'labels', 'labels_data'))
            join_res_fol = utils.make_dir(op.join(utils.get_parent_fol(MMVT_DIR), 'msit-ecr', subject))
            for res_fname in glob.glob(op.join(output_fol, '{}_labels_{}_{}_*_power.npz'.format(
                    task.lower(), inv_method, em))):
                shutil.copyfile(res_fname, op.join(join_res_fol, utils.namebase_with_ext(res_fname)))
            if not ret:
                if args.throw:
                    raise Exception("errors!")
                else:
                    subjects_with_error.append(subject)


    good_subjects = [s for s in good_subjects if
           op.isfile(op.join(MMVT_DIR, subject, 'meg',
                             'labels_data_msit_{}_{}_{}_minmax.npz'.format(atlas, inv_method, em))) and
           op.isfile(op.join(MMVT_DIR, subject, 'meg',
                             'labels_data_ecr_{}_{}_{}_minmax.npz'.format(atlas, inv_method, em)))]
    print('Good subjects:')
    print(good_subjects)
    print('subjects_with_error:')
    print(subjects_with_error)


# def calc_source_band_induced_power(args):
#     inv_method, em, atlas= 'MNE', 'mean_flip', 'darpa-atlas'
#     bands = dict(theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55], high_gamma=[65, 200])
#     times = (-2, 4)
#     subjects_with_error = []
#     good_subjects = get_good_subjects(args)
#     args.subject = good_subjects
#     prepare_files(args)
#     done_subjects = []
#
#     for subject in good_subjects:
#         args.subject = subject
#         empty_fnames, cors, days = get_empty_fnames(subject, args.tasks, args)
#         for task in args.tasks:
#             fol = utils.make_dir(op.join(MEG_DIR, task, subject, 'induced_power'))
#             output_fnames = glob.glob(op.join(fol, '{}*induced_power*.stc'.format(task)))
#             # If another thread is working on this subject / task, continue to another subject / task
#             # if len(output_fnames) > 0:
#             #     done_subjects.append(subject)
#             #     continue
#             meg_args = meg.read_cmd_args(dict(
#                 subject=args.subject, mri_subject=args.subject,
#                 task=task, inverse_method=inv_method, extract_mode=em, atlas=atlas,
#                 # meg_dir=args.meg_dir,
#                 remote_subject_dir=args.remote_subject_dir, # Needed for finding COR
#                 get_task_defaults=False,
#                 fname_format='{}_{}_Onset'.format('{subject}', task),
#                 raw_fname=op.join(MEG_DIR, task, subject, '{}_{}-raw.fif'.format(subject, task)),
#                 epo_fname=op.join(MEG_DIR, task, subject, '{}_{}_meg_Onset-epo.fif'.format(subject, task)),
#                 function='calc_stc',
#                 calc_source_band_induced_power=True,
#                 calc_inducde_power_per_label=True,
#                 induced_power_normalize_proj=True,
#                 overwrite_stc=args.overwrite,
#                 conditions=task.lower(),
#                 cor_fname=cors[task].format(subject=subject),
#                 data_per_task=True,
#                 n_jobs=args.n_jobs
#             ))
#             ret = meg.call_main(meg_args)
#             if not ret:
#                 if args.throw:
#                     raise Exception("errors!")
#                 else:
#                     subjects_with_error.append(subject)
#
#     print('#done_subjects: {}'.format(len(set(done_subjects))))
#     good_subjects = [s for s in good_subjects if
#            op.isfile(op.join(MMVT_DIR, subject, 'meg',
#                              'labels_data_msit_{}_{}_{}_minmax.npz'.format(atlas, inv_method, em))) and
#            op.isfile(op.join(MMVT_DIR, subject, 'meg',
#                              'labels_data_ecr_{}_{}_{}_minmax.npz'.format(atlas, inv_method, em)))]
#     print('Good subjects:')
#     print(good_subjects)
#     print('subjects_with_error:')
#     print(subjects_with_error)


def post_meg_preproc(args):
    inv_method, em, atlas = 'dSPM', 'mean_flip', args.atlas
    bands = dict(theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55], high_gamma=[65, 200])
    norm_times = (500, 2500)
    do_plot = False

    subjects = args.subject
    res_fol = utils.make_dir(op.join(utils.get_parent_fol(MMVT_DIR), 'msit-ecr'))
    subjects_with_results = {}
    labels = lu.read_labels(subjects[0], SUBJECTS_DIR, atlas)
    labels_names = [l.name for l in labels]
    labels_num = len(labels_names)
    epochs_max_num = 50
    template_brain = 'colin27'

    now = time.time()
    bands_power_mmvt_all = []
    for subject_ind, subject in enumerate(subjects):
        utils.time_to_go(now, subject_ind, len(subjects), runs_num_to_print=1)
        subjects_with_results[subject] = {}
        input_fol = utils.make_dir(op.join(MEG_DIR, subject, 'labels_induced_power'))
        plots_fol = utils.make_dir(op.join(input_fol, 'plots'))
        args.subject = subject
        bands_power_mmvt = {'rh':{}, 'lh':{}}
        for task_ind, task in enumerate(args.tasks):
            task = task.lower()
            input_fnames = glob.glob(op.join(input_fol, '{}_*_{}_{}_induced_power.npz'.format(task, inv_method, em)))
            if len(input_fnames) < 1:# labels_num:
                print('No enough files for {} {}!'.format(subject, task))
                subjects_with_results[subject][task] = False
                continue
            # input_dname = ecr_caudalanteriorcingulate-lh_dSPM_mean_flip_induced_power
            # if not do_plot:
            #     continue
            bands_power = np.empty((len(bands), labels_num, epochs_max_num))
            for input_fname in input_fnames:
                d = utils.Bag(np.load(input_fname)) # label_name, atlas, data
                # label_power = np.empty((len(bands), epochs_num, T)) (5, 50, 3501)
                label_power, label_name = d.data, d.label_name
                # for band_ind in range(len(bands)):
                #     label_power[band_ind] /= label_power[band_ind][:, norm_times[0]:norm_times[1]].mean()
                label_ind = labels_names.index(label_name)
                hemi = labels[label_ind].hemi
                for band_ind, band in enumerate(bands.keys()):
                    label_power_norm = label_power[band_ind][:, norm_times[0]:norm_times[1]].mean(axis=1)[:epochs_max_num]
                    if len(label_power_norm) != epochs_max_num:
                        print('{} does have {} epochs!'.format(input_fname, len(label_power_norm)))
                        break
                    bands_power[band_ind, label_ind] = label_power_norm
                    if band not in bands_power_mmvt[hemi]:
                        bands_power_mmvt[hemi][band] = np.empty((len(labels_names), label_power[band_ind].shape[1], 1, len(args.tasks)))
                    bands_power_mmvt[hemi][band][label_ind, :, 0, task_ind] = label_power[band_ind].mean(axis=0)
                fig_fname = op.join(plots_fol, 'power_{}_{}.jpg'.format(label_name, task))
                if do_plot: # not op.isfile(fig_fname) and
                    times = np.arange(0, label_power.shape[2]) if 'times' not in d else d.times
                    plot_label_power(label_power, times, label_name, bands, task, fig_fname)
            for band_ind, band in enumerate(bands.keys()):
                power_fname = op.join(
                    res_fol, subject, '{}_labels_{}_{}_{}_power.npz'.format(task.lower(), inv_method, em, band))
                np.savez(power_fname, data=np.array(bands_power[band_ind]), names=labels_names)
            subjects_with_results[subject][task] = True


        if all(subjects_with_results[subject].values()):
            bands_power_mmvt_all.append(bands_power_mmvt)
        else:
            print('{} does not have both tasks data!'.format(subject))

    labels_data_template = op.join(MMVT_DIR, template_brain, 'meg', 'labels_data_power_{}_{}_{}_{}_{}.npz')  # task, atlas, extract_method, hemi
    for hemi in utils.HEMIS:
        for band_ind, band in enumerate(bands.keys()):
            power = np.array([x[hemi][band] for x in bands_power_mmvt_all]).mean(axis=0)
            labels_output_fname = meg.get_labels_data_fname(
                labels_data_template, inv_method, band, atlas, em, hemi)
            utils.make_dir(utils.get_parent_fol(labels_output_fname))
            np.savez(labels_output_fname, data=power, names=labels_names, conditions=args.tasks)


    have_all = len([subject for subject, results in subjects_with_results.items() if all(results.values())])
    print('{}/{} with all files'.format(have_all, len(subjects)))
    print(subjects_with_results)


def plot_label_power(power, times, label, bands, task, fig_fname):
    # plt.figure()
    f, axs = plt.subplots(5, 1, sharex=True)
    for band_ind, (band_name, ax) in enumerate(zip(bands.keys(), axs)):
        power_mean = power[band_ind].mean(0)
        power_std = power[band_ind].std(0)
        ax.plot(times, power_mean)
        ax.fill_between(times, power_mean - power_std, power_mean + power_std, alpha=.5)
        ax.set_title(band_name)
    print('Saving {}'.format(fig_fname))
    plt.savefig(fig_fname)
    plt.close()


def calc_meg_connectivity(args):
    inv_method, em = 'dSPM', 'mean_flip'
    prepare_files(args)
    good_subjects = get_good_subjects(args)
    args.subject = good_subjects

    for subject in good_subjects:
        args.subject = subject
        for task in args.tasks:

            output_fname = op.join(
                MMVT_DIR, subject, 'connectivity', '{}_{}_coh_cwt_morlet.npz'.format(task.lower(), em))
            if op.isfile(output_fname):
                file_mod_time = utils.file_modification_time_struct(output_fname)
                if file_mod_time.tm_year >= 2018 and (file_mod_time.tm_mon == 10 and file_mod_time.tm_mday >= 10) or \
                        (file_mod_time.tm_mon > 10):
                    print('{} already exist!'.format(output_fname))
                    continue

            remote_epo_fname = op.join(args.meg_dir, subject, args.epo_template.format(subject=subject, task=task))
            local_epo_fname = op.join(MEG_DIR, task, subject, args.epo_template.format(subject=subject, task=task))
            if not op.isfile(local_epo_fname):
                utils.make_link(remote_epo_fname, local_epo_fname)

            con_args = meg.read_cmd_args(utils.Bag(
                subject=args.subject, mri_subject=args.subject,
                task=task, inverse_method=inv_method, extract_mode=em, atlas=args.atlas,
                # meg_dir=args.meg_dir,
                remote_subject_dir=args.remote_subject_dir,  # Needed for finding COR
                get_task_defaults=False,
                data_per_task=True,
                fname_format=args.epo_template.format(subject=subject, task=task)[:-len('-epo.fif')],
                raw_fname=op.join(MEG_DIR, task, subject, args.raw_template.format(subject=subject, task=task)),
                epo_fname=local_epo_fname,
                # empty_fname=empty_fnames[task],
                function='calc_labels_connectivity',
                conditions=task.lower(),
                overwrite_connectivity=True,#args.overwrite_connectivity,
                # cor_fname=cors[task].format(subject=subject),
                # ica_overwrite_raw=False,
                # normalize_data=False,
                # t_min=times[0], t_max=times[1],
                # read_events_from_file=False, stim_channels='STI001',
                # use_empty_room_for_noise_cov=True,
                # read_only_from_annot=False,
                # pick_ori='normal',
                # overwrite_evoked=args.overwrite,
                # overwrite_inv=args.overwrite,
                # overwrite_stc=args.overwrite,
                # overwrite_labels_data=args.overwrite,
                n_jobs=args.n_jobs
            ))
            meg.call_main(con_args)


def post_analysis(args):
    import matplotlib.pyplot as plt
    from collections import defaultdict

    inv_method, em = 'dSPM', 'mean_flip'
    res_fol = utils.make_dir(op.join(utils.get_parent_fol(MMVT_DIR), 'msit-ecr'))
    plot_fol = utils.make_dir(op.join(res_fol, 'plots'))
    bands = dict(theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55], high_gamma=[65, 200])
    data_dic = np.load(op.join(res_fol, 'data_dictionary.npz'))
    meta_data = data_dic['noam_dict'].tolist()
    # brain_overall_res_fname = op.join(res_fol, 'brain_overall_res.npz')
    msit_subjects = set(meta_data[0]['MSIT'].keys())
    ecr_subjects = set(meta_data[1]['ECR'].keys())
    subjects_with_data = defaultdict(list)
    mean_evo = {group_id:defaultdict(list) for group_id in range(2)}
    mean_power_power_emotion_reactivit = {group_id: {} for group_id in range(2)}
    power_emotion_reactivit = {group_id: {} for group_id in range(2)}

    mean_power_power_task = {}
    power_task = {}
    for task in args.tasks:
        mean_power_power_task[task] = defaultdict(list)
        power_task[task] = {band: None for band in bands.keys()}
    for subject in args.subject:
        if not op.isdir(op.join(res_fol, subject)):
            print('No folder data for {}'.format(subject))
            continue
        for task in args.tasks:
            # mean_fname = op.join(res_fol, subject, '{}_{}_mean.npz'.format(task.lower(), args.atlas))
            # if op.isfile(mean_fname):
            #     d = utils.Bag(np.load(mean_fname))
            #     mean_evo[group_id][task].append(d.data.mean())
            for band in bands.keys():
                if power_task[task][band] is None:
                    power_task[task][band] = defaultdict(list)
                power_fname = op.join(
                    res_fol, subject, '{}_labels_{}_{}_{}_power.npz'.format(task.lower(), inv_method, em, band))
                if op.isfile(power_fname):
                    d = utils.Bag(np.load(power_fname))
                    mean_power_power_task[task][band].append(d.data.mean())
                    for label_id, label in enumerate(d.names):
                        power_task[task][band][label].append(d.data[label_id].mean())

    # for group_id in range(2):
    #     for task in args.tasks:
    #         mean_power_power_emotion_reactivit[group_id][task] = defaultdict(list)
    #         power_emotion_reactivit[group_id][task] = {band: None for band in bands.keys()}
    #     for subject in meta_data[group_id]['ECR'].keys():
    #         if not op.isdir(op.join(res_fol, subject)):
    #             print('No folder data for {}'.format(subject))
    #             continue
    #         for task in args.tasks:
    #             mean_fname = op.join(res_fol, subject, '{}_{}_mean.npz'.format(task.lower(), args.atlas))
    #             if op.isfile(mean_fname):
    #                 d = utils.Bag(np.load(mean_fname))
    #                 mean_evo[group_id][task].append(d.data.mean())
    #             for band in bands.keys():
    #                 if power_emotion_reactivit[group_id][task][band] is None:
    #                     power_emotion_reactivit[group_id][task][band] = defaultdict(list)
    #                 power_fname = op.join(
    #                     res_fol, subject, '{}_labels_{}_{}_{}_power.npz'.format(task.lower(), inv_method, em, band))
    #                 if op.isfile(power_fname):
    #                     d = utils.Bag(np.load(power_fname))
    #                     mean_power_power_emotion_reactivit[group_id][task][band].append(d.data.mean())
    #                     for label_id, label in enumerate(d.names):
    #                         power_emotion_reactivit[group_id][task][band][label].append(d.data[label_id].mean())

    do_plot = False
    percentile = 90
    alpha = 0.05
    for band in bands.keys():

        x = [np.array(mean_power_power_task[task][band]) for task in args.tasks]
        x[0] = x[0][x[0] < np.percentile(x[0], percentile)]
        x[1] = x[1][x[1] < np.percentile(x[1], percentile)]
        sig = ttest(x[0], x[1], title='MSIT vs ECR band {}'.format(band), alpha=alpha, always_print=False)
        if do_plot or sig:
            f, (ax1, ax2) = plt.subplots(2, 1)
            ax1.hist(x[0], bins=80)
            ax1.set_title('{} {}'.format(band, args.tasks[0]))
            ax2.hist(x[1], bins=80)
            ax2.set_title('{} {}'.format(band, args.tasks[1]))
            # plt.title('{} mean power'.format(band))
            plt.show()
            # plt.savefig(op.join(plot_fol, '{}_group_{}.jpg'.format(band, group_id)))

        for label_id, label in enumerate(d.names):
            x = [np.array(power_task[task][band][label]) for task in args.tasks]
            x[0] = x[0][x[0] < np.percentile(x[0], percentile)]
            x[1] = x[1][x[1] < np.percentile(x[1], percentile)]
            sig = ttest(x[0], x[1], alpha=alpha, title='band {} label {}'.format(band, label))
            if do_plot or sig:
                f, (ax1, ax2) = plt.subplots(2, 1)
                ax1.hist(x[0], bins=80)
                ax2.hist(x[1], bins=80)
                plt.title('{} mean power'.format(band))
                plt.show()
                # plt.savefig(op.join(plot_fol, '{}_group_{}.jpg'.format(band, group_id)))


        continue
        for group_id in range(2): #, ax in zip(range(2), [ax1, ax2]):
            # subjects_with_data[group_id] = np.array(subjects_with_data[group_id])
            # print()
            x = [np.array(mean_power_power_emotion_reactivit[group_id][task][band]) for task in args.tasks]
            # x = [_x[_x < np.percentile(_x, 90)] for _x in x]
            x[0] = x[0][x[0] < np.percentile(x[0], percentile)]
            x[1] = x[1][x[1] < np.percentile(x[1], percentile)]
            print('band {}, group {}, {} for {}, {} for {}'.format(
                band, group_id, len(x[0]), args.tasks[0], len(x[1]), args.tasks[1]))
            ttest(x[0], x[1], title='group {} band {}'.format(group_id, band), alpha=alpha)
            if do_plot:
                f, (ax1, ax2) = plt.subplots(2, 1)
                ax1.hist(x[0], bins=80)
                ax2.hist(x[1], bins=80)
                plt.title('{} mean power'.format(band))
                plt.savefig(op.join(plot_fol, '{}_group_{}.jpg'.format(band, group_id)))

            for label_id, label in enumerate(d.names):
                x = [np.array(power_emotion_reactivit[group_id][task][band][label]) for task in args.tasks]
                # x = [_x[_x < np.percentile(_x, 90)] for _x in x]
                x[0] = x[0][x[0] < np.percentile(x[0], percentile)]
                x[1] = x[1][x[1] < np.percentile(x[1], percentile)]
                ttest(x[0], x[1], alpha=alpha, title='group {} band {} label {}'.format(group_id, band, label))
                if do_plot:
                    f, (ax1, ax2) = plt.subplots(2, 1)
                    ax1.hist(x[0], bins=80)
                    ax2.hist(x[1], bins=80)
                    plt.title('{} {} power'.format(band, label))
                    plt.savefig(op.join(plot_fol, '{}_group_{}_label_{}.jpg'.format(band, group_id, label)))
        # ax.set_title('group {}'.format(group_id))
        # ax.bar(np.arange(2), [np.mean(mean_power[group_id][task][band]) for task in args.tasks])
        # ax.set_xticklabels(args.tasks, rotation=30)
        # fig.suptitle('{} mean_power'.format(band))
        # plt.show()


def ttest(x1, x2, two_tailed_test=True, alpha=0.1, is_greater=True, title='', always_print=False):
    import scipy.stats
    t, pval = scipy.stats.ttest_ind(x1, x2, equal_var=True)
    sig = is_significant(pval, t, two_tailed_test, alpha, is_greater)
    welch_t, welch_pval = scipy.stats.ttest_ind(x1, x2, equal_var=False)
    welch_sig = is_significant(pval, t, two_tailed_test, alpha, is_greater)
    if sig or welch_sig or always_print:
        print('{}: {:.2f}+-{:.2f}, {:.2f}+-{:.2f}'.format(title, np.mean(x1), np.std(x1), np.mean(x2), np.std(x2)))
        print('test: pval: {:.4f} sig: {}. welch: pval: {:.4f} sig: {}'.format(pval, sig, welch_pval, welch_sig))
    return sig or welch_sig


def is_significant(pval, t, two_tailed_test, alpha=0.05, is_greater=True):
    if two_tailed_test:
        return pval < alpha
    else:
        if is_greater:
            return pval / 2 < alpha and t > 0
        else:
            return pval / 2 < alpha and t < 0

    # for subject, task, band in product(args.subject, tasks, bands.keys()):
    #     mean_power_fol = op.join(MMVT_DIR, subject, 'labels', 'labels_data')
    #     d = utils.Bag(np.load(op.join(mean_power_fol, '{}_mean_power_{}.npz'.format(task.lower(), band))))
    #     for label_name, label_data in zip(d.names, d.data):
    #         hemi = lu.get_label_hemi(label_name)
    #         plt.figure()
    #         plt.axhline(label_data * 1e5, color='r', linestyle='--')
    #         plt.show()
    #         plt.close()
    #         print('asdf')
    #


def get_good_subjects(args, check_dict=False):
    if check_dict:
        data_dict_fname = op.join(args.remote_root_dir, 'data_dictionary.npz')
        if not op.isfile(data_dict_fname):
            ret = input('No data dict, do you want to continue? (y/n)')
            if not au.is_true(ret):
                return
            msit_ecr_subjects = args.subject
        else:
            data_dic = np.load(op.join(args.remote_root_dir, 'data_dictionary.npz'))
            meta_data = data_dic['noam_dict'].tolist()
            msit_subjects = set(meta_data[0]['MSIT'].keys()) | set(meta_data[1]['MSIT'].keys())
            ecr_subjects = set(meta_data[0]['ECR'].keys()) | set(meta_data[1]['ECR'].keys())
            msit_ecr_subjects = msit_subjects.intersection(ecr_subjects)
    else:
        msit_ecr_subjects = set()

    if not args.check_files:
        good_subjects = args.subject
    else:
        good_subjects = []
        for subject in args.subject:
            # if subject == 'pp009':
            #     continue
            if check_dict and subject not in msit_ecr_subjects:
                print('*** {} not in the meta data!'.format(subject))
                continue
            if not op.isdir(args.remote_subject_dir.format(subject=subject)) and not op.isdir(op.join(SUBJECTS_DIR, subject)):
                print('*** {}: No recon-all files!'.format(subject))
                continue
            if args.anatomy_preproc:
                anatomy_preproc(args, subject)
            if not utils.both_hemi_files_exist(op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot'.format('{hemi}', args.atlas))):
                anatomy_preproc(args, subject)
            if not utils.both_hemi_files_exist(op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot'.format('{hemi}', args.atlas))):
                print('*** Can\'t find the atlas {}!'.format(args.atlas))
                continue
            empty_fnames, cors, days = get_empty_fnames(subject, args.tasks, args)
            if empty_fnames == '' or cors == '' or days == '':
                print('{}: Error with get_empty_fnames!'.format(subject))
            if any([task not in cors for task in args.tasks]) and  args.check_cor:
                print('*** {}: one of the tasks does not have a cor transformation matrix!'.format(subject))
                print(cors)
                continue
            files_exist = \
                op.isfile(op.join(args.meg_dir.format(task='ECR'), subject, args.epo_template.format(subject=subject, task='ECR'))) and \
                op.isfile(op.join(args.meg_dir.format(task='MSIT'), subject, args.epo_template.format(subject=subject, task='MSIT')))
            if not files_exist and args.check_for_both_files:
                print('**** {} doesn\'t have both MSIT and ECR files!'.format(subject))
                continue
            # for task in args.tasks:
            #     print('{}: empty: {}, cor: {}'.format(subject, empty_fnames[task], cors[task].format(subject=subject)))
            good_subjects.append(subject)
        print('Good subjects: ({}):'.format(len(good_subjects)))
        print(good_subjects)
        bad_subjects = set(args.subject) - set(good_subjects)
        print('Bad subjects: ({}):'.format(len(bad_subjects)))
        print(bad_subjects)
    return good_subjects


if __name__ == '__main__':
    import argparse
    from src.utils import args_utils as au
    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-f', '--function', help='function name', required=False, default='meg_preproc_power')
    parser.add_argument('-a', '--atlas', help='atlas name', required=False, default='laus125') #darpa-atlas')
    parser.add_argument('-t', '--tasks', help='tasks', required=False, default='MSIT,ECR', type=au.str_arr_type)
    parser.add_argument('--overwrite', required=False, default=False, type=au.is_true)
    parser.add_argument('--overwrite_output_files', required=False, default=False, type=au.is_true)
    parser.add_argument('--overwrite_local_files', required=False, default=False, type=au.is_true)
    parser.add_argument('--overwrite_connectivity', required=False, default=False, type=au.is_true)
    parser.add_argument('--overwrite_labels_power_spectrum', required=False, default=False, type=au.is_true)
    parser.add_argument('--throw', required=False, default=False, type=au.is_true)
    parser.add_argument('--anatomy_preproc', required=False, default=True, type=au.is_true)
    parser.add_argument('--check_files', required=False, default=True, type=au.is_true)
    parser.add_argument('--check_file_modification_time', required=False, default=False, type=au.is_true)
    parser.add_argument('--check_cor', required=False, default=True, type=au.is_true)
    parser.add_argument('--check_for_both_files', required=False, default=True, type=au.is_true)
    parser.add_argument('--check_for_channels_inconsistency', required=False, default=1, type=au.is_true)
    parser.add_argument('--max_epochs_num', required=False, default=0, type=int)

    parser.add_argument('--remote_root_dir', required=False,
                        default='/autofs/space/karima_001/users/alex/MSIT_ECR_Preprocesing_for_Noam/')
    meg_dirs = ['/home/npeled/meg/{task}',
                '/autofs/space/karima_001/users/alex/MSIT_ECR_Preprocesing_for_Noam/epochs']
    meg_dir = [d for d in meg_dirs if op.isdir(d.format(task='MSIT'))][0]
    parser.add_argument('--meg_dir', required=False, default=meg_dir)
                        # default='/autofs/space/karima_001/users/alex/MSIT_ECR_Preprocesing_for_Noam/raw_preprocessed')
    remote_subject_dirs = ['/autofs/space/lilli_001/users/DARPA-Recons/',
                           '/home/npeled/subjects']
    remote_subject_dir = [op.join(d, '{subject}') for d in remote_subject_dirs if op.isdir(d)][0]
    parser.add_argument('--remote_subject_dir', required=False, default=remote_subject_dir)
    parser.add_argument('--remote_meg_dir', required=False,
                        default='/autofs/space/lilli_003/users/DARPA-TRANSFER/meg')
    parser.add_argument('--epo_template', required=False, default='{subject}_{task}_meg_Onset_ar-epo.fif')
    parser.add_argument('--raw_template', required=False, default='{subject}_{task}_meg_ica-raw.fif')
    parser.add_argument('--n_jobs', help='cpu num', required=False, default=-1)
    args = utils.Bag(au.parse_parser(parser))

    if args.subject[0] == 'all':
        args.subject = utils.shuffle(
            [utils.namebase(d) for d in glob.glob(op.join(args.meg_dir, '*')) if op.isdir(d) and
             op.isfile(op.join(d, args.epo_template.format(subject=utils.namebase(d), task='ECR'))) and
             op.isfile(op.join(d, args.epo_template.format(subject=utils.namebase(d), task='MSIT')))])
        print('{} subjects were found with both tasks!'.format(len(args.subject)))
        print(sorted(args.subject))
    elif '*' in args.subject[0]:
        args.subject = utils.shuffle(
            [utils.namebase(d) for d in glob.glob(op.join(args.meg_dir, args.subject[0])) if op.isdir(d) and
             op.isfile(op.join(d, args.epo_template.format(subject=utils.namebase(d), task='ECR'))) and
             op.isfile(op.join(d, args.epo_template.format(subject=utils.namebase(d), task='MSIT')))])
        print('{} subjects were found with both tasks:'.format(len(args.subject)))
        print(sorted(args.subject))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        locals()[args.function](args)


