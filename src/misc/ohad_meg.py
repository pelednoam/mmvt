import os.path as op
import glob

from src.preproc import meg
from src.utils import utils

LINKS_DIR = utils.get_links_dir()
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')


def meg_prepoc(args):
    # -s nmr00479 -f make_forward_solution,calc_inverse_operator,calc_stc --use_empty_room_for_noise_cov 1 --apply_on_raw 1 --recreate_src_surface 1 --overwrite_fwd 1
    inv_method = 'MNE'
    for subject in args.subject:
        raw_files = glob.glob(op.join(MEG_DIR, subject, '*_??_raw.fif'))
        args.subject = subject
        for raw_fname in raw_files:
            meg_args = meg.read_cmd_args(dict(
                subject=args.subject, mri_subject=args.subject,
                inverse_method=inv_method,
                raw_fname=raw_fname,
                function='make_forward_solution,calc_inverse_operator,calc_stc',
                use_empty_room_for_noise_cov=True,
                apply_on_raw=True,
                pick_ori='normal'
            ))
            ret = meg.call_main(meg_args)
