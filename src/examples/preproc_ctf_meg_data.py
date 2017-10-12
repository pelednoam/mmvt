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


def analyze(subject):
    flags = {}
    args = pu.init_args(meg.read_cmd_args(dict(
        subject=subject,
        task='tapping',
        conditions='left',
        atlas='laus250',
        inverse_method='MNE',
        t_min=-2, t_max=2,
        noise_t_min=-2.5, noise_t_max=-1.5,
        bad_channels=[],
        stim_channels='STIM',
        pick_ori='normal',
        reject=False,
        overwrite_epochs=False,
        overwrite_inv=False,
        overwrite_noise_cov=False,
        overwrite_ica=True,
        do_plot_ica=False,
        fwd_usingEEG=False)))
    meg.init(subject, args)
    overwrite_raw = False
    save_raw = False
    only_examine_ica = False
    plot_evoked = False
    calc_stc_per_session = True
    conditions = dict(left=4, right=8)
    eog_channel = 'MZF01-1410' # Doesn't give good results, so we'll use manuualy pick ICA componenets
    eog_inds_fname = op.join(MEG_DIR, subject, 'ica_eog_comps.txt')
    args.stc_template = op.join(MEG_DIR, subject, '{cond}-{method}.stc')
    args.stc_hemi_template = op.join(MEG_DIR, subject, '{cond}-{method}-{hemi}.stc')
    args.noise_cov_fname = op.join(MEG_DIR, subject, 'noise-cov.fif')
    if op.isfile(eog_inds_fname):
        all_eog_inds = np.genfromtxt(eog_inds_fname, dtype=np.str, delimiter=',', autostrip=True)
    else:
        all_eog_inds = []
    for cond in conditions.keys():
        sessions = [f[-4] for f in glob.glob(op.join(MEG_DIR, subject, 'raw', 'DC_{}Index_day?.ds'.format(cond)))]
        raw_files = glob.glob(op.join(MEG_DIR, subject, 'raw', 'DC_{}Index_day?.ds'.format(cond)))
        args.conditions = condition = {cond:4 if cond == 'left' else 8}
        for ctf_raw_data in raw_files:
            calc_per_session(
                subject, condition, ctf_raw_data, args, sessions, all_eog_inds, eog_channel, calc_stc_per_session,
                only_examine_ica, overwrite_raw, save_raw, plot_evoked)
        combine_evokes(subject, cond, sessions)

        # combine_noise_covs(subject, conditions, sessions, args.noise_t_min, args.noise_t_max, args)
        args.inv_fname = op.join(MEG_DIR, subject, '{}-inv.fif'.format(cond))
        args.fwd_fname = op.join(MEG_DIR, subject, '{}-fwd.fif'.format(cond))
        args.evo_fname = op.join(MEG_DIR, subject, '{}-ave.fif'.format(cond))
        meg.calc_fwd_inv_wrapper(subject, condition, args)
    args.evo_fname = op.join(MEG_DIR, subject, '{cond}-ave.fif')
    args.inv_fname = op.join(MEG_DIR, subject, '{cond}-inv.fif')
    _, stcs_conds, _ = meg.calc_stc_per_condition_wrapper(subject, conditions, args.inverse_method, args)
    meg.calc_labels_avg_per_condition_wrapper(subject, conditions, args.atlas, args.inverse_method, stcs_conds, args, flags, {})

        # dipoles_times = [(0.25, 0.35)]
        # dipoles_names =['peak_left_motor']
        # meg.dipoles_fit(dipoles_times, dipoles_names, evokes=None, min_dist=5., use_meg=True, use_eeg=False, n_jobs=6)


def calc_per_session(subject, condition, ctf_raw_data, args, sessions, all_eog_inds, eog_channel, calc_stc_per_session,
                     only_examine_ica, overwrite_raw, save_raw, plot_evoked):
    session = ctf_raw_data[-4]
    cond = list(condition.keys())[0]
    new_raw_fname = op.join(MEG_DIR, subject, '{}-session{}-ica-raw.fif'.format(cond, session))
    args.epo_fname = op.join(MEG_DIR, subject, '{}-session{}-epo.fif'.format(cond, session))
    args.evo_fname = op.join(MEG_DIR, subject, '{}-session{}-ave.fif'.format(cond, session))
    args.inv_fname = op.join(MEG_DIR, subject, '{}-session{}-inv.fif'.format(cond, session))
    args.fwd_fname = op.join(MEG_DIR, subject, '{}-session{}-fwd.fif'.format(cond, session))
    args.noise_cov_fname = op.join(MEG_DIR, subject, '{}-session{}-noise-cov.fif'.format(cond, session))
    args.stc_template = op.join(MEG_DIR, subject, '{cond}-session' + session + '-{method}.stc')
    stc_hemi_template = '{}{}'.format(args.stc_template[:-4], '-{hemi}.stc')
    args.labels_data_template = op.join(MEG_DIR, subject, 'labels_data_session' + session + '_{}_{}_{}.npz')
    if check_if_all_done(new_raw_fname, cond, calc_stc_per_session, stc_hemi_template, args.labels_data_template, args):
        return
    ica_fname = op.join(MEG_DIR, subject, '{}-session{}-ica.fif'.format(cond, session))
    if len(all_eog_inds) > 0:
        session_ind = np.where(all_eog_inds[:, 0] == utils.namesbase_with_ext(ica_fname))[0][0]
        eog_inds = [int(all_eog_inds[session_ind, 1])]
    if only_examine_ica:
        meg.fit_ica(ica_fname=ica_fname, do_plot=True, examine_ica=True)
        return
    if not op.isfile(new_raw_fname) or overwrite_raw or not op.isfile(ica_fname):
        raw = mne.io.read_raw_ctf(op.join(MEG_DIR, subject, 'raw', ctf_raw_data), preload=True)
        raw = meg.remove_artifacts(
            raw, remove_from_raw=True, overwrite_ica=args.overwrite_ica, save_raw=save_raw,
            raw_fname=new_raw_fname, ica_fname=ica_fname, do_plot=args.do_plot_ica, eog_inds=eog_inds,
            eog_channel=eog_channel)
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
        # args.stc_hemi_template = stc_hemi_template.format('{cond}', session, '{method}', '{hemi}')
        stcs_conds = None
        if not op.isfile(args.stc_template.format(cond=cond, method=args.inverse_method)):
            _, stcs_conds, stcs_num = meg.calc_stc_per_condition_wrapper(subject, condition, args.inverse_method, args)
        meg.calc_labels_avg_per_condition_wrapper(subject, condition, args.atlas, args.inverse_method[0], stcs_conds, args)


def check_if_all_done(new_raw_fname, cond, calc_stc_per_session, stc_template_hemi, labels_data_template, args):
    all_done = False
    if all([op.isfile(f) for f in [new_raw_fname, args.epo_fname, args.evo_fname]]):
        if calc_stc_per_session:
            all_done = all([op.isfile(f) for f in [args.inv_fname, args.fwd_fname]]) and \
                       utils.both_hemi_files_exist(stc_template_hemi.format(cond=cond, method=args.inverse_method[0], hemi='{hemi}')) and \
                       utils.both_hemi_files_exist(labels_data_template.format(cond, args.inverse_method[0], '{hemi}'))
        else:
            all_done = True
    return all_done


def combine_evokes(subject, cond, sessions):
    combined_evoked_fname = op.join(MEG_DIR, subject, '{}-ave.fif'.format(cond))
    if not op.isfile(combined_evoked_fname):
        all_evokes = []
        for session in sessions:
            evoked = mne.read_evokeds(op.join(MEG_DIR, subject, '{}-session{}-ave.fif'.format(cond, session)))[0]
            evoked.apply_baseline()
            all_evokes.append(evoked)
        combined_evoked = mne.combine_evoked(all_evokes, 'nave')
        combined_evoked.comment = cond
        mne.write_evokeds(combined_evoked_fname, combined_evoked)


def combine_noise_covs(subject, conditions, sessions, noise_t_min, noise_t_max, args):
    noise_cov_fname = op.join(MEG_DIR, subject, 'noise-cov.fif')
    args.epochs_fname = op.join(MEG_DIR, subject, 'epo.fif')
    if op.isfile(noise_cov_fname):
        return
    all_epochs = []
    for session in sessions:
        for cond in conditions.keys():
            all_epochs.append(mne.read_epochs(op.join(MEG_DIR, subject, '{}-session{}-epo.fif'.format(cond, session))))
    all_epochs = mne.concatenate_epochs(all_epochs)
    all_epochs.save(args.epochs_fname)
    noise_cov = meg.calc_noise_cov(None, noise_t_min, noise_t_max, args)
    noise_cov.save(noise_cov_fname)


def plot_motor_response():
    pass

if __name__ == '__main__':
    analyze('DC')
    print('Finish!')