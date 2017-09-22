import matplotlib.pyplot as plt
import os.path as op
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
        bad_channels=[],
        stim_channels='STIM',
        reject=False,
        overwrite_epochs=False))
    fname_format, fname_format_cond, conditions = meg.init(subject, args)
    conditions['left'] = 4
    raw = mne.io.read_raw_ctf(op.join(MEG_DIR, subject, 'raw', 'DC_leftIndex_day1.ds'), preload=True)
    if not op.isfile(meg.RAW):
        raw.save(meg.RAW)
    flags, evoked, epochs = meg.calc_evokes_wrapper(subject, conditions, args, flags, raw=raw)
    if evoked is not None:
        fig = evoked[0].plot_joint(times=[-0.5, 0.05, 0.150, 0.250, 0.6])
        plt.show()
    flags = meg.calc_fwd_inv_wrapper(subject, conditions, args, flags)
    flags, stcs_conds, _ = meg.calc_stc_per_condition_wrapper(subject, conditions, args.inverse_method, args, flags)
    flags = meg.calc_labels_avg_per_condition_wrapper(subject, conditions, args.atlas, args.inverse_method, stcs_conds, args, flags)


if __name__ == '__main__':
    analyze('DC')