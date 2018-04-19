import mne
import numpy as np
import os.path as op
from src.utils import utils

LINKS_DIR = utils.get_links_dir()
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')


def get_digitization_points(subject, raw_fname):
    raw = mne.io.read_raw_fif(raw_fname)
    info = raw.info
    pos = np.array([p['r'] for p in info['dig']])
    kind = np.array([p['kind'] for p in info['dig']])
    ident = np.array([p['ident'] for p in info['dig']])
    coord_frame = np.array([p['coord_frame'] for p in info['dig']])
    utils.make_dir(op.join(MMVT_DIR, subject, 'meg'))
    np.savez(op.join(MMVT_DIR, subject, 'meg', 'digitization_points.npz'),
             pos=pos, kind=kind, ident=ident, coord_frame=coord_frame)


if __name__ == '__main__':
    # subject = 'sample'
    # raw_fname = op.join(MEG_DIR, subject, 'sample_audvis_raw.fif')
    # get_digitization_points(subject, raw_fname)
    get_digitization_points('bects020', '/cluster/neuromind/cat/bects/subjects/pBECTS020/sleep_source/pBECTS020_rest03_filtered_raw.fif')