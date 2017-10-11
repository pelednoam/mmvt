import matplotlib.pyplot as plt
import os.path as op
import glob
import numpy as np
import mne

from src.preproc import meg
from src.utils import utils

LINKS_DIR = utils.get_links_dir()
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')


def analyze(subject):
    flags = {}
    args = meg.read_cmd_args(dict(
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
        overwrite_epochs=True,
        overwrite_inv=True,
        overwrite_noise_cov=False,
        overwrite_ica=True,
        do_plot_ica=False))
    fname_format, fname_format_cond, conditions = meg.init(subject, args)
    overwrite_raw = False
    save_raw = False
    only_examine_ica = False
    eog_channel = 'MZF01-1410' # Doesn't give good results, so we'll use manuualy pick ICA componenets
    eog_inds = []
    eog_inds_fname = op.join(MEG_DIR, subject, 'ica_eog_comps.txt')
    if op.isfile(eog_inds_fname):
        all_eog_inds = np.genfromtxt(eog_inds_fname, dtype=np.str, delimiter=',', autostrip=True)
    else:
        all_eog_inds = []
    for cond in ['right', 'left']:
        raw_files = glob.glob(op.join(MEG_DIR, subject, 'raw', 'DC_{}Index_day?.ds'.format(cond)))
        args.conditions = conditions = {cond:4 if cond == 'left' else 8}
        for ctf_raw_data in raw_files:
            session = ctf_raw_data[-4]
            new_raw_fname = op.join(MEG_DIR, subject, '{}-session{}-ica-raw.fif'.format(cond, session))
            ica_fname = op.join(MEG_DIR, subject, '{}-session{}-ica.fif'.format(cond, session))
            if len(all_eog_inds) > 0:
                session_ind = np.where(all_eog_inds[:, 0] == utils.namesbase_with_ext(ica_fname))[0][0]
                eog_inds = [int(all_eog_inds[session_ind, 1])]
            if only_examine_ica:
                meg.fit_ica(ica_fname=ica_fname, do_plot=True, examine_ica=True)
                continue
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
            args.epochs_name = op.join(MEG_DIR, subject, '{}-session{}-epo.fif'.format(cond, session))
            args.evoked_name = op.join(MEG_DIR, subject, '{}-session{}-ave.fif'.format(cond, session))
            flags, evoked, epochs = meg.calc_evokes_wrapper(subject, conditions, args, flags, raw=raw)

            # if evoked is not None:
            #     fig = evoked[0].plot_joint(times=[-0.5, 0.05, 0.150, 0.250, 0.6])
            #     plt.show()
            # flags = meg.calc_fwd_inv_wrapper(subject, conditions, args, flags)
            # flags, stcs_conds, _ = meg.calc_stc_per_condition_wrapper(subject, conditions, args.inverse_method, args, flags)
            # flags = meg.calc_labels_avg_per_condition_wrapper(subject, conditions, args.atlas, args.inverse_method, stcs_conds, args, flags)
            dipoles_times = [(0.25, 0.35)]
            dipoles_names =['peak_left_motor']
            # meg.dipoles_fit(dipoles_times, dipoles_names, evokes=None, min_dist=5., use_meg=True, use_eeg=False, n_jobs=6)


if __name__ == '__main__':
    analyze('DC')
    print('Finish!')