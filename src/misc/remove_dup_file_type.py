import os.path as op
import os
import glob
import shutil

from src.utils import utils
from src.utils import movies_utils as mu

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
MMVT_DIR = op.join(LINKS_DIR, 'mmvt')


def remove_dup_file_type(root, ft):
    files = glob.glob(op.join(root, '*.{0}.{0}'.format(ft)))
    for fname in files:
        new_fname = fname[:-len(ft)-1]
        os.rename(fname, new_fname)


def remove_odd_files(root, ft):
    files = glob.glob(op.join(root, '*.{}'.format(ft)))
    first = True
    for fname in files:
        file_num = int(utils.namebase(fname).split('_')[1])
        if file_num % 2 == 1:
            if first:
                backup_fol = utils.make_dir(op.join(root, 'backup'))
                first = False
            shutil.move(fname, op.join(backup_fol, utils.namebase_with_ext(fname)))


def make_movie(root, movie_name='', fr=20, overwrite=False):
    if movie_name == '':
        movie_name = utils.namebase(root)
    tmp_output_fname = op.join(root, 'new_images', '{}.mp4'.format(movie_name))
    output_fname = op.join(utils.get_parent_fol(root), '{}.mp4'.format(movie_name))
    if not op.isfile(output_fname) or overwrite:
        mu.combine_images(root, movie_name, frame_rate=fr,  copy_files=True)
    if op.isfile(tmp_output_fname):
        shutil.move(tmp_output_fname, output_fname)
    if op.isdir(op.join(root, 'new_images')):
        shutil.rmtree(op.join(root, 'new_images'))


def combine_movies(fol, final_movie_name, fps=60, movie_type='mp4'):
    from moviepy.editor import VideoFileClip, concatenate_videoclips

    parts = [VideoFileClip(op.join(fol, '{}_{}.{}'.format(movie_name, fps, movie_type))) for movie_name in [
        'meg_helmet_with_brain', 'eeg_with_brain', 'meg', 'connections', 'electrodes']]
    final_movie = concatenate_videoclips(parts, method='chain')
    final_movie.write_videofile(op.join(fol, '{}.{}'.format(final_movie_name, movie_type)), fps=fps, threads=1)


if __name__ == '__main__':
    ft = 'jpeg'
    root = op.join(MMVT_DIR, 'matt_hibert', 'figures')
    # for fol in [d for d in glob.glob(op.join(root, '*')) if op.isdir(d)]:
    for fol in [op.join(root, d) for d in ['slicing_movie']]:
        remove_dup_file_type(fol, ft)
        # remove_odd_files(fol, ft)
        make_movie(fol, '{}_20'.format(utils.namebase(fol)), 20, True)
    # combine_movies(root, 'modalities_movie', 60)
    print('finish!')