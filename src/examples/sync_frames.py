import numpy as np
import os.path as op
import glob
import shutil

from src.utils import preproc_utils as pu
from src.utils import matlab_utils as mu
from src.utils import utils
from src.utils import movies_utils as movu


SUBJECTS_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()
ELECTRODES_DIR = utils.get_link_dir(utils.get_links_dir(), 'electrodes')


def load_times(subject, fname, bipolar=True, times_field='time'):
    # meta_fname = op.join(MMVT_DIR, subject, 'electrodes', 'electrodes{}_meta_data.npz'.format(
    #     '_bipolar' if bipolar else ''))
    # d = np.load(meta_fname)
    # times = d[times_field]
    mat_fname = op.join(ELECTRODES_DIR, subject, fname)
    d = mu.load_mat_to_bag(mat_fname)
    times = d[times_field].squeeze()
    return times


def load_frames(subject, frames_fol_name):
    frames_fol = op.join(MMVT_DIR, subject, 'figures', frames_fol_name)
    frames = glob.glob(op.join(frames_fol, '*.png'))
    frames = sorted(frames, key=utils.natural_keys)
    return frames


def get_breakind(times):
    break_ind = np.argsort(np.diff(times))[-1] + 1
    print('break: {}'.format(times[break_ind]))
    return break_ind


def duplicate_frames(frames, times, times_per_frame, last_frame_len, max_frames_num=np.inf, max_frame_time=np.inf):
    new_frames = []
    time, frame_time, frame_ind = 0, 0, 0
    times_ind = 1
    while frame_ind < len(frames) and times_ind < len(times) and frame_ind < max_frames_num:
        new_frames.append(frames[frame_ind])
        time += times_per_frame
        frame_time += times_per_frame
        if time >= times[times_ind] or frame_time > max_frame_time:
            frame_ind += 1
            times_ind += 1
            frame_time = 0
    if len(np.unique(new_frames)) != max_frames_num:
        new_frames.extend([frames[-1]] * last_frame_len * int(1 / times_per_frame))
    return new_frames


def create_dup_frames_links(subject, dup_frames, fol):
    fol = op.join(MMVT_DIR, subject, 'figures', fol)
    utils.delete_folder_files(fol)
    utils.make_dir(fol)
    for ind, frame in enumerate(dup_frames):
        utils.make_link(frame, op.join(fol, 'dup_{}.{}'.format(ind, utils.file_type(frame))))
    return fol


if __name__ == '__main__':
    subject = 'mg106'
    file_name = 'MG106_HGP_sess1'
    times_per_frame = 0.1
    cut_in_break = False
    last_frame_len = 5 * 60

    times = load_times(subject, '{}.mat'.format(file_name))
    frames = load_frames(subject, frames_fol_name=file_name)
    break_ind = get_breakind(times) if cut_in_break else len(times)
    last_frame_len = last_frame_len
    print('max time: {} minutes'.format(times[break_ind if cut_in_break else len(times) - 1] / 60))
    new_frames = duplicate_frames(frames, times, times_per_frame, last_frame_len,  max_frames_num=break_ind)
    dup_fol = create_dup_frames_links(subject, new_frames, '{}_dup'.format(file_name))
    movie_fname = movu.combine_images(dup_fol, file_name, frame_rate=int(1 / times_per_frame))
    new_movie_fname = op.join(MMVT_DIR, subject, 'figures', utils.namesbase_with_ext(movie_fname))
    utils.remove_file(new_movie_fname)
    shutil.move(movie_fname, new_movie_fname)