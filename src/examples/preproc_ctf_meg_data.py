import matplotlib.pyplot as plt
import os.path as op
import glob
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
        # atlas='laus250',
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
        overwrite_ica=True))
    fname_format, fname_format_cond, conditions = meg.init(subject, args)
    for cond in ['left', 'right']:
        raw_files = glob.glob(op.join(MEG_DIR, subject, 'raw', 'DC_{}Index_day?.ds'.format(cond)))
        conditions[cond] = 4
        args.conditions = conditions
        for ctf_raw_data in raw_files:
            session = ctf_raw_data[-4]
            new_raw_fname = op.join(MEG_DIR, subject, '{}-session{}-ica-raw.fif'.format(cond, session))
            if not op.isfile(new_raw_fname):
                raw = mne.io.read_raw_ctf(op.join(MEG_DIR, subject, 'raw', ctf_raw_data), preload=True)
                raw = meg.remove_artifacts(
                    raw, remove_from_raw=True, overwrite_ica=args.overwrite_ica,
                    raw_fname='{}-session{}-raw.fif'.format(cond, session), do_plot=False)
                print('Saving new raw file in {}'.format(new_raw_fname))
                raw.save(new_raw_fname)
            else:
                raw = mne.io.read_raw_fif(meg.RAW_ICA, preload=True)
            args.epochs_name = '{}-session{}'.format(cond, session)
            args.evoked_name = '{}-session{}'.format(cond, session)
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