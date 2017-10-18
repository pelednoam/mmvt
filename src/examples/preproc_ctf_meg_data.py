import matplotlib.pyplot as plt
import os.path as op
import glob
import numpy as np
import mne

from src.preproc import meg
from src.utils import utils
from src.utils import preproc_utils as pu

LINKS_DIR = utils.get_links_dir()
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')


def analyze(subject, raw_files_template, inverse_method, conditions, sessions, args):
    overwrite_raw = False
    only_examine_ica = False
    plot_evoked = False
    calc_stc_per_session = True
    look_for_ica_eog_file = True
    filter_raw_data = True
    raw_data_filter_freqs = (int(args.l_freq), int(args.h_freq))
    eog_inds_fname = op.join(MEG_DIR, subject, 'ica_eog_comps.txt')
    # args.noise_cov_fname = op.join(MEG_DIR, subject, 'noise-cov.fif')
    freqs_str = '-{}-{}'.format(raw_data_filter_freqs[0], raw_data_filter_freqs[1]) if filter_raw_data else ''
    if op.isfile(eog_inds_fname):
        all_eog_inds = np.genfromtxt(eog_inds_fname, dtype=np.str, delimiter=',', autostrip=True)
    else:
        if look_for_ica_eog_file:
            raise Exception("Can't find the ICA eog file! {}".format(eog_inds_fname))
        all_eog_inds = []
    for cond, cond_key in conditions.items():
        args.inv_fname = op.join(MEG_DIR, subject, '{}-inv.fif'.format(cond))
        args.fwd_fname = op.join(MEG_DIR, subject, '{}-fwd.fif'.format(cond))
        args.evo_fname = op.join(MEG_DIR, subject, '{}{}-ave.fif'.format(cond, freqs_str))
        meg.calc_fwd_inv_wrapper(subject, cond, args)
        raw_files_cond = raw_files_template.format(cond=cond)
        raw_files = glob.glob(raw_files_cond)
        args.conditions = condition = {cond:cond_key}
        for ctf_raw_data in raw_files:
            calc_per_session(
                subject, condition, ctf_raw_data, inverse_method, args, all_eog_inds, eog_channel, calc_stc_per_session,
                only_examine_ica, overwrite_raw, plot_evoked, filter_raw_data, raw_data_filter_freqs)
        combine_evokes(subject, cond, sessions, filter_raw_data, raw_data_filter_freqs)

    for session in sessions:
        args.evo_fname = op.join(MEG_DIR, subject, '{}-session{}{}-ave.fif'.format('{cond}', session, freqs_str))
        args.inv_fname = op.join(MEG_DIR, subject, '{}-session{}-inv.fif'.format('{cond}', session))
        args.stc_template = op.join(MEG_DIR, subject, '{}-session{}-{}{}.stc'.format('{cond}', session, '{method}', freqs_str))
        args.labels_data_template = op.join(
            MEG_DIR, subject, 'labels_data_session' + session + freqs_str + '_{}_{}_{}.npz')
        stc_hemi_template = meg.get_stc_hemi_template(args.stc_template)
        meg.calc_stc_diff_both_hemis(conditions, stc_hemi_template, inverse_method)
        meg.calc_labels_avg_per_condition_wrapper(subject, conditions, args.atlas, inverse_method, None, args)

    args.evo_fname = op.join(MEG_DIR, subject, '{}{}-ave.fif'.format('{cond}', freqs_str))
    args.inv_fname = op.join(MEG_DIR, subject, '{cond}-inv.fif')
    args.stc_template = op.join(MEG_DIR, subject, '{}-{}{}.stc'.format('{cond}', '{method}', freqs_str))
    args.labels_data_template = op.join(MEG_DIR, subject, 'labels_data' + freqs_str + '_{}_{}_{}.npz')
    _, stcs_conds, _ = meg.calc_stc_per_condition_wrapper(subject, conditions, inverse_method, args)
    meg.calc_labels_avg_per_condition_wrapper(subject, conditions, args.atlas, inverse_method, stcs_conds, args)


def dipole_fit(conditions, filter_raw_data, raw_data_filter_freqs, extract_mode='mean_flip', inverse_method='MNE'):
    dipoles_times = [(-0.3, 0.5)]
    freqs_str = '-{}-{}'.format(raw_data_filter_freqs[0], raw_data_filter_freqs[1]) if filter_raw_data else ''
    evo_fname = op.join(MEG_DIR, subject, '{}{}-ave.fif'.format('{cond}', freqs_str))
    inv_fname = op.join(MEG_DIR, subject, '{cond}-inv.fif')
    stc_hemi_template = op.join(MEG_DIR, subject, '{}-{}{}-{}.stc'.format('{cond}', inverse_method, freqs_str, '{hemi}'))

    plt.figure()
    sfreq = 625
    time = np.arange(-2, 2 + 1 / sfreq, 1 / sfreq)
    for cond in conditions.keys():
        dipoles_names =['peak_{}_motor'.format(cond)]
        noise_cov_fname = op.join(MEG_DIR, subject, 'right-session1-noise-cov.fif')
        cond_evo_fname = evo_fname.format(cond=cond)
        dipole_vert, label_data, label_fname, dipole_hemi = \
            meg.find_dipole_cortical_locations(args.atlas, cond, dipoles_names[0], grow_label=True, label_r=5,
            inv_fname=inv_fname, stc_hemi_template=stc_hemi_template, extract_mode=extract_mode[0], n_jobs=6)
        plt.plot(time, label_data, label='{} index {}'.format(cond, dipole_hemi))
    plt.legend()
    plt.xlim([-1.5, 0.5])
    plt.savefig(op.join(MEG_DIR, subject, 'figures', 'motor_{}.png'.format(args.inverse_method[0])))
        # meg.dipoles_fit(dipoles_times, dipoles_names, evokes=None, evo_fname=cond_evo_fname, mask_roi='precentral', do_plot=False,
        #                 noise_cov_fname=noise_cov_fname, min_dist=5., use_meg=True, use_eeg=False, n_jobs=6)


def calc_per_session(subject, condition, ctf_raw_data, inverse_method, args, all_eog_inds, eog_channel,
                     calc_stc_per_session, only_examine_ica, overwrite_raw, plot_evoked, filter_raw_data,
                     raw_data_filter_freqs):
    session = ctf_raw_data[-4]
    cond = list(condition.keys())[0]
    freqs_str = '-{}-{}'.format(raw_data_filter_freqs[0], raw_data_filter_freqs[1]) if filter_raw_data else ''
    args.raw_fname = op.join(MEG_DIR, subject, '{}-session{}-raw.fif'.format(cond, session))
    new_raw_no_filter_fname = op.join(MEG_DIR, subject, '{}-session{}-ica-raw.fif'.format(cond, session))
    new_raw_fname = op.join(MEG_DIR, subject, '{}-session{}{}-ica-raw.fif'.format(cond, session, freqs_str))
    args.epo_fname = op.join(MEG_DIR, subject, '{}-session{}{}-epo.fif'.format(cond, session, freqs_str))
    args.evo_fname = op.join(MEG_DIR, subject, '{}-session{}{}-ave.fif'.format(cond, session, freqs_str))
    args.inv_fname = op.join(MEG_DIR, subject, '{}-session{}-inv.fif'.format(cond, session))
    args.fwd_fname = op.join(MEG_DIR, subject, '{}-session{}-fwd.fif'.format(cond, session))
    args.noise_cov_fname = op.join(MEG_DIR, subject, '{}-session{}-noise-cov.fif'.format(cond, session))
    args.stc_template = op.join(MEG_DIR, subject, '{cond}-session' + session + '-{method}' + freqs_str + '.stc')
    stc_hemi_template = meg.get_stc_hemi_template(args.stc_template)
    if check_if_all_done(new_raw_fname, cond, inverse_method, calc_stc_per_session, stc_hemi_template,
                         args.labels_data_template, args):
        return
    ica_fname = op.join(MEG_DIR, subject, '{}-session{}-ica.fif'.format(cond, session))
    if len(all_eog_inds) > 0:
        session_ind = np.where(all_eog_inds[:, 0] == utils.namesbase_with_ext(ica_fname))[0][0]
        eog_inds = [int(all_eog_inds[session_ind, 1])]
    else:
        eog_inds = []
    if only_examine_ica:
        meg.fit_ica(ica_fname=ica_fname, do_plot=True, examine_ica=True, n_jobs=args.n_jobs)
        return
    if not op.isfile(new_raw_fname) or overwrite_raw or not op.isfile(ica_fname):
        if not op.isfile(args.raw_fname):
            raw = mne.io.read_raw_ctf(op.join(MEG_DIR, subject, 'raw', ctf_raw_data), preload=True)
            raw.save(args.raw_fname)
        if not op.isfile(new_raw_no_filter_fname):
            raw = mne.io.read_raw_fif(args.raw_fname, preload=True)
            raw = meg.remove_artifacts(
                raw, remove_from_raw=True, overwrite_ica=args.overwrite_ica, save_raw=True,
                raw_fname=new_raw_fname, new_raw_fname=new_raw_no_filter_fname, ica_fname=ica_fname,
                do_plot=args.do_plot_ica, eog_inds=eog_inds, eog_channel=eog_channel, n_jobs=args.n_jobs)
        else:
            raw = mne.io.read_raw_fif(new_raw_no_filter_fname, preload=True)
        meg.calc_noise_cov(None, args.noise_t_min, args.noise_t_max, args.noise_cov_fname, args, raw)
        if filter_raw_data:
            raw.filter(raw_data_filter_freqs[0], raw_data_filter_freqs[1],  h_trans_bandwidth='auto',
                       filter_length='auto', phase='zero')
        print('Saving new raw file in {}'.format(new_raw_fname))
        if overwrite_raw or not op.isfile(new_raw_fname):
            raw.save(new_raw_fname, overwrite=True)
    else:
        raw = mne.io.read_raw_fif(new_raw_fname, preload=True)
    evoked, epochs = None, None
    if not op.isfile(args.epo_fname) or not op.isfile(args.evo_fname):
        _, evoked, epochs = meg.calc_evokes_wrapper(subject, condition, args, raw=raw)
    if evoked is not None and plot_evoked:
        fig = evoked[0].plot_joint(times=[-0.5, 0.05, 0.150, 0.250, 0.6])
        plt.show()
    if calc_stc_per_session:
        meg.calc_fwd_inv_wrapper(subject, condition, args)
        # stcs_conds = None
        if not utils.both_hemi_files_exist(stc_hemi_template.format(cond=cond, method=inverse_method, hemi='{hemi}')):
            _, stcs_conds, stcs_num = meg.calc_stc_per_condition_wrapper(subject, condition, inverse_method, args)
        # meg.calc_labels_avg_per_condition_wrapper(subject, condition, args.atlas, inverse_method, stcs_conds, args)


def check_if_all_done(new_raw_fname, cond, inverse_method, calc_stc_per_session, stc_template_hemi,
                      labels_data_template, args):
    all_done = all([op.isfile(f) for f in [new_raw_fname, args.epo_fname, args.evo_fname]])
    if all_done:
        all_done = \
            all([op.isfile(f) for f in [args.inv_fname, args.fwd_fname]]) and \
            utils.both_hemi_files_exist(stc_template_hemi.format(cond=cond, method=inverse_method, hemi='{hemi}')) # and \
            # all([utils.both_hemi_files_exist(labels_data_template.format(args.atlas, em, '{hemi}')) and \
            #      op.isfile(meg.get_labels_minmax_template(labels_data_template).format(args.atlas, em))
            #      for em in args.extract_mode])
        return all_done if calc_stc_per_session else True
    return all_done


def combine_evokes(subject, cond, sessions, filter_raw_data, raw_data_filter_freqs):
    freqs_str = '-{}-{}'.format(raw_data_filter_freqs[0], raw_data_filter_freqs[1]) if filter_raw_data else ''
    combined_evoked_fname = op.join(MEG_DIR, subject, '{}{}-ave.fif'.format(cond, freqs_str))
    if not op.isfile(combined_evoked_fname):
        all_evokes = []
        for session in sessions:
            evo_fname = op.join(MEG_DIR, subject, '{}-session{}{}-ave.fif'.format(cond, session, freqs_str))
            evoked = mne.read_evokeds(evo_fname)[0]
            evoked.apply_baseline()
            all_evokes.append(evoked)
        combined_evoked = mne.combine_evoked(all_evokes, 'nave')
        combined_evoked.comment = cond
        mne.write_evokeds(combined_evoked_fname, combined_evoked)

#
# def combine_noise_covs(subject, conditions, sessions, noise_t_min, noise_t_max, args):
#     noise_cov_fname = op.join(MEG_DIR, subject, 'noise-cov.fif')
#     args.epochs_fname = op.join(MEG_DIR, subject, 'epo.fif')
#     if op.isfile(noise_cov_fname):
#         return
#     all_epochs = []
#     for session in sessions:
#         for cond in conditions.keys():
#             all_epochs.append(mne.read_epochs(op.join(MEG_DIR, subject, '{}-session{}-epo.fif'.format(cond, session))))
#     all_epochs = mne.concatenate_epochs(all_epochs)
#     all_epochs.save(args.epochs_fname)
#     noise_cov = meg.calc_noise_cov(None, noise_t_min, noise_t_max, args)
#     noise_cov.save(noise_cov_fname)


def plot_motor_response(subject, atlas, rois, sfreq, sessions, args):
    for em in args.extract_mode:
        for roi in rois:
            plot_roi(subject, atlas, em, roi, sfreq, sessions, args)


def plot_roi(subject, atlas, extract_method, roi, sfreq, sessions, args):
    from collections import defaultdict
    data_freqs = '-{}-{}'.format(int(args.l_freq), int(args.h_freq))

    labels_data = defaultdict(list)
    for session in sessions:
        labels_data_template = op.join(MEG_DIR, subject, 'labels_data_session{}' + data_freqs + '_{}_{}_{}.npz')
        for hemi in utils.HEMIS:
            d = utils.Bag(np.load(labels_data_template.format(session, atlas, extract_method, hemi)))
            for cond_ind, cond in enumerate(d.conditions):
                ind = np.where(d.names=='{}-{}'.format(roi, hemi))[0]
                if len(ind) == 0:
                    return
                data = d.data[ind, :, cond_ind]
                labels_data[(hemi, cond)].append(data)
    fig, axs = plt.subplots(2,2, True, True, figsize=(8, 8))
    time = np.arange(-2, 2 + 1/sfreq, 1/sfreq)
    for cond_ind, cond in enumerate(d.conditions):
        for hemi_ind, hemi in enumerate(utils.HEMIS):
            ax = axs[hemi_ind, cond_ind]
            data = np.array(labels_data[(hemi, cond)]).squeeze().T
            ax.plot(time, data)
            ax.set_title('{} hemi, {} index ({} Hz)'.format('right' if hemi=='rh' else 'left', cond, data_freqs[1:]))
            ax.set_xlim([-1.5, 0.5])
            ax.set_ylim([-1.5, 2])
    plt.suptitle(roi)

    # plt.show()
    utils.make_dir(op.join(MEG_DIR, subject, 'figures'))
    fig_fname = op.join(MEG_DIR, subject, 'figures', '{}.png'.format(roi))
    print('Saving {}'.format(fig_fname))
    plt.savefig(fig_fname)


if __name__ == '__main__':
    from itertools import product
    args = pu.init_args(meg.read_cmd_args(dict(
        subject='DC',
        atlas='laus250',
        inverse_method='MNE',
        t_min=-2, t_max=2,
        noise_t_min=-2.5, noise_t_max=-1.5,
        l_freq=1, h_freq=15,
        extract_mode='mean_flip',
        bad_channels=[],
        stim_channels='STIM',
        pick_ori='normal',
        reject=False,
        overwrite_epochs=False,
        overwrite_inv=False,
        overwrite_noise_cov=False,
        overwrite_ica=False,
        do_plot_ica=False,
        fwd_usingEEG=False)))

    subject = args.subject[0]
    meg.init(subject, args)
    conditions = dict(left=4, right=8)
    raw_files_template = op.join(MEG_DIR, args.subject[0], 'raw', 'DC_{cond}Index_day?.ds')
    sessions = sorted([f[-4] for f in glob.glob(raw_files_template.format(cond=list(conditions.keys())[0]))])
    motor_rois = ['precentral_{}'.format(ind) for ind in range(1, 20)]  # ['precentral_11', 'precentral_5']
    sfreq = 625
    eog_channel = 'MZF01-1410' # Doesn't give good results, so we'll use manuualy pick ICA componenets
    for inverse_method in args.inverse_method:
        analyze(subject, raw_files_template, inverse_method, conditions, sessions, args)
        # dipole_fit(conditions, True, (int(args.l_freq), int(args.h_freq)), extract_mode=args.extract_mode)
        # plot_motor_response(subject, args.atlas, motor_rois,  sfreq, sessions, args)
    print('Finish!')