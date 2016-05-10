import os.path as op
import glob
from src.utils import utils


def cut_movie(movie_fol, movie_name, out_movie_name, subclips_times):
    from moviepy import editor
    # subclips_times [(3, 4), (6, 17), (38, 42)]
    video = editor.VideoFileClip(op.join(movie_fol, movie_name))
    subclips = []
    for from_t, to_t in subclips_times:
        clip = video.subclip(from_t, to_t)
        subclips.append(clip)
    final_clip = editor.concatenate_videoclips(subclips)
    final_clip.write_videofile(op.join(movie_fol, out_movie_name))


def crop_movie(movie_fol, movie_name, out_movie_name, crop_ys=(60, 1170)):
    from moviepy import editor
    video = editor.VideoFileClip(op.join(movie_fol, movie_name))
    crop_video = video.crop(y1=crop_ys[0], y2=crop_ys[1])
    crop_video.write_videofile(op.join(movie_fol, out_movie_name))


def add_text_to_movie(movie_fol, movie_name, out_movie_name, subs):
    from moviepy import editor

    def annotate(clip, txt, txt_color='red', fontsize=50, font='Xolonium-Bold'):
        """ Writes a text at the bottom of the clip. """
        txtclip = editor.TextClip(txt, fontsize=fontsize, font=font, color=txt_color)
        # txtclip = txtclip.on_color((clip.w, txtclip.h + 6), color=(0, 0, 255), pos=(6, 'center'))
        cvc = editor.CompositeVideoClip([clip, txtclip.set_pos(('center', 'bottom'))])
        return cvc.set_duration(clip.duration)

    video = editor.VideoFileClip(op.join(movie_fol, movie_name))
    annotated_clips = [annotate(video.subclip(from_t, to_t), txt) for (from_t, to_t), txt in subs]
    final_clip = editor.concatenate_videoclips(annotated_clips)
    final_clip.write_videofile(op.join(movie_fol, out_movie_name))


def create_animated_gif(movie_fol, movie_name, out_movie_name):
    from moviepy import editor
    video = editor.VideoFileClip(op.join(movie_fol, movie_name))
    video.write_gif(op.join(movie_fol, out_movie_name), fps=12)


def edit_movie_example():
    movie_fol = '/cluster/neuromind/npeled/videos/recordmydesktop'
    movie_fol = '/cluster/neuromind/npeled/Documents/brain-map'
    # cut_movie(movie_fol, 'out-7.ogv', 'freeview-mmvt.mp4')
    # crop_movie(movie_fol, 'freeview-mmvt.mp4', 'freeview-mmvt_crop.mp4')
    subs = [((0, 4), 'Clicking on the OFC activation in Freeview'),
            ((4, 9), 'The cursor moved to the same coordinates in the MMVT'),
            ((9, 12), 'Finding the closest activation in the coordinates'),
            ((12, 16), 'The activation is displayed with its statistics')]
    # add_text_to_movie(movie_fol, 'freeview-mmvt_crop.mp4', 'freeview-mmvt_crop_text.mp4', subs)
    create_animated_gif(movie_fol, 'mg78_elecs_coh_meg_diff.mp4', 'mg78_elecs_coh_meg_diff.gif')


def combine_movies(fol, movie_name, movie_type='mp4'):
    # First convert the part to avi, because mp4 cannot be concat
    cmd = 'ffmpeg -i concat:"'
    parts = sorted(glob.glob(op.join(fol, '{}_*.{}'.format(movie_name, movie_type))))
    for part_fname in parts:
        part_name, _ = op.splitext(part_fname)
        cmd = '{}{}.avi|'.format(cmd, op.join(fol, part_name))
        utils.remove_file('{}.avi'.format(part_name))
        utils.run_script('ffmpeg -i {} -codec copy {}.avi'.format(part_fname, op.join(fol, part_name)))
    # cmd = '{}" -c copy -bsf:a aac_adtstoasc {}'.format(cmd[:-1], op.join(fol, '{}.{}'.format(movie_name, movie_type)))
    cmd = '{}" -c copy {}'.format(cmd[:-1], op.join(fol, '{}.{}'.format(movie_name, movie_type)))
    print(cmd)
    utils.remove_file('{}.{}'.format(op.join(fol, movie_name), movie_type))
    utils.run_script(cmd)
    # clean up
    # todo: doesn't clean the part filess
    utils.remove_file('{}.avi'.format(op.join(fol, movie_name)))
    for part_fname in parts:
        part_name, _ = op.splitext(part_fname)
        utils.remove_file('{}.avi'.format(part_name))
