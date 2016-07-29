import os.path as op
import glob
from src.utils import utils

# from moviepy.config import change_settings
# change_settings({"IMAGEMAGICK_BINARY": r"/usr/bin/convert"})

# https://www.vultr.com/docs/install-imagemagick-on-centos-6
# https://github.com/BVLC/caffe/issues/3884


def check_movipy():
    import moviepy.config as conf
    if conf.try_cmd([conf.FFMPEG_BINARY])[0]:
        print("MoviePy : ffmpeg successfully found.")
    else:
        print("MoviePy : can't find or access ffmpeg.")

    if conf.try_cmd([conf.IMAGEMAGICK_BINARY])[0]:
        print("MoviePy : ImageMagick successfully found.")
    else:
        print("MoviePy : can't find or access ImageMagick.")


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


def add_text_to_movie(movie_fol, movie_name, out_movie_name, subs, fontsize=50, txt_color='red', font='Xolonium-Bold'):
    # Should install ImageMagick
    # For centos6: https://www.vultr.com/docs/install-imagemagick-on-centos-6
    from moviepy import editor

    def annotate(clip, txt, txt_color=txt_color, fontsize=fontsize, font=font):
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


def edit_movie_example2():
    movie_fol = '/home/noam/Videos/mmvt'
    if not op.isdir(movie_fol):
        movie_fol = '/autofs/cluster/neuromind/npeled/videos'
    subclips_times = [(2, 57)]
    # cut_movie(movie_fol, 'out-13.ogv', 'freeview-mmvt-electrodes.mp4', subclips_times)
    # crop_movie(movie_fol, 'freeview-mmvt-electrodes.mp4', 'freeview-mmvt-electrodes_crop.mp4')
    subs = [((0, 4), 'The brain is a 3d object imported from FreeSurfer'),
            ((4, 18), 'To see the depth electrodes, the user can hide the hemispheres'),
            ((18, 30), 'The user can choose an electrode and see it location both in MMVT and FreeView'),
            ((30, 35), "Also, the user can choose to see only the current electrodes' lead"),
            ((35, 40), "The program estimates the each electrode's sources and plot the probabilities as colors"),
             # "Yellow: low probability, red: high probability"),
            ((40, 55), "Chagning to a different lead")]
    add_text_to_movie(movie_fol, 'freeview-mmvt-electrodes.mp4', 'freeview-mmvt-electrodes_sub.mp4', subs, fontsize=80)
    # create_animated_gif(movie_fol, 'mg78_elecs_coh_meg_diff.mp4', 'mg78_elecs_coh_meg_diff.gif')


def edit_movie_example3():
    movie_fol = '/home/noam/Pictures/mmvt/mg99/lvf4-3_4_1'
    movie_name = 'mg99_LVF4-3_stim_srouces_long.mp4'
    out_movie_name = 'mg99_LVF4-3_stim_srouces.mp4'
    subclips_times = [(0, 29)]
    cut_movie(movie_fol, movie_name, out_movie_name, subclips_times)

if __name__ == '__main__':
    check_movipy()
    edit_movie_example2()