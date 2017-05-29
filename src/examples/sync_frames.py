import numpy as np
import os.path as op
import glob

from src.utils import preproc_utils as pu
from src.utils import utils

SUBJECTS_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()


def load_times(subject, bipolar, times_field='time'):
    meta_fname = op.join(MMVT_DIR, subject, 'electrodes', 'electrodes{}_meta_data.npz'.format(
        '_bipolar' if bipolar else ''))
    d = np.load(meta_fname)
    times = d[times_field]
    return times


def load_frames(subject, frames_fol_name):
    frames_fol = op.join(MMVT_DIR, subject, 'figures', frames_fol_name)
    frames = glob.glob(op.join(frames_fol, '*.png'))
    return frames


def cut_frames_into_blocks(frames, times):
    diffs = np.diff(times)
    diffs_sort_inds = np.argsort(diffs)
    print(diffs[diffs_sort_inds[-5:]])
    pass


def duplicate_frames(frames, times, times_per_frame):
    new_frames = []
    time = 0
    frame_ind = 0
    times_ind = 1
    while frame_ind < len(frames) and times_ind < len(times):
        new_frames.append(frames[frame_ind])
        time += times_per_frame
        if time >= times[times_ind]:
            frame_ind += 1
            times_ind += 1
    return new_frames


def create_dup_frames_links(subject, dup_frames, fol):
    fol  = op.join(MMVT_DIR, subject, 'figures', fol)
    utils.delete_folder_files(fol)
    utils.make_dir(fol)
    for ind, frame in enumerate(dup_frames):
        utils.make_link(frame, op.join(fol, 'dup_{}.{}'.format(ind, utils.file_type(frame))))


if __name__ == '__main__':
    subject = 'mg106'
    times = load_times(subject, bipolar=True)
    frames = load_frames(subject, frames_fol_name='MG106_HGP_sess1')
    # cut_frames_into_blocks(frames, times)
    new_frames = duplicate_frames(frames, times, times_per_frame=0.1)
    create_dup_frames_links(subject, new_frames, 'MG106_HGP_sess1_dup')