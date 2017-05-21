import os.path as op
import glob
from src.utils import utils


FFMPEG_DIR = utils.get_link_dir(utils.get_links_dir(), 'ffmpeg')
FFMPEG_DIR = op.join(FFMPEG_DIR, 'bin') if utils.is_windows() else FFMPEG_DIR
FFMPEG_CMD = op.join(FFMPEG_DIR, 'ffmpeg') if op.isdir(FFMPEG_DIR) else 'ffmpeg'

# from moviepy.config import change_settings
# change_settings({"IMAGEMAGICK_BINARY": r"/usr/bin/convert"})

# https://www.vultr.com/docs/install-imagemagick-on-centos-6
# https://github.com/BVLC/caffe/issues/3884


def check_movipy():
    try:
        import moviepy.config as conf
        if conf.try_cmd([conf.FFMPEG_BINARY])[0]:
            print("MoviePy : ffmpeg successfully found.")
        else:
            print("MoviePy : can't find or access ffmpeg.")

        if conf.try_cmd([conf.IMAGEMAGICK_BINARY])[0]:
            print("MoviePy : ImageMagick successfully found.")
        else:
            print("MoviePy : can't find or access ImageMagick.")
    except:
        print("Can't import moviepy")


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


def crop_movie(fol, movie_name, out_movie_name, crop_ys=(60, 1170), **kwargs):
    from moviepy import editor
    video = editor.VideoFileClip(op.join(fol, '{}.mp4'.format(movie_name)))
    crop_video = video.crop(y1=crop_ys[0], y2=crop_ys[1])
    crop_video.write_videofile(op.join(fol, '{}.mp4'.format(out_movie_name)))


def add_text_to_movie(movie_fol, movie_name, out_movie_name, subs, fontsize=50, txt_color='red', font='Xolonium-Bold'):
    # Should install ImageMagick
    # For centos6: https://www.vultr.com/docs/install-imagemagick-on-centos-6
    # For centos7: http://helostore.com/blog/install-imagemagick-on-centos-7
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


def create_animated_gif(movie_fol, movie_name, out_movie_name, fps=10):
    from moviepy import editor
    video = editor.VideoFileClip(op.join(movie_fol, movie_name))
    video.write_gif(op.join(movie_fol, out_movie_name), fps=fps)


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


def combine_images(fol, movie_name, frame_rate=10, start_number=-1, images_prefix='', images_format='',
                   images_type='', ffmpeg_cmd='ffmpeg', movie_name_full_path=False, debug_mode = True, **kwargs):
    images_type, images_prefix, images_format, images_format_len, start_number = find_images_props(
        fol, start_number, images_prefix, images_format, images_type)
    if utils.is_windows():
        fol = change_frames_names(fol, images_prefix, images_type, images_format_len)
    images_prefix = op.join(fol, images_prefix)
    if not movie_name_full_path:
        movie_name = op.join(fol, movie_name)
    combine_images_cmd = '{ffmpeg_cmd} -framerate {frame_rate} '
    if start_number != 1:
        # You might want to use a static ffmpeg if your ffmepg version doesn't support the start_number flag, like:
        # ffmpeg_cmd = '~/space1/Downloads/ffmpeg-git-static/ffmpeg'
        combine_images_cmd += '-start_number {start_number} '
    # Not working in windows:
    # combine_images_cmd += '-pattern_type glob -i "*.{images_type}" '
    combine_images_cmd += '-i {images_prefix}{images_format}.{images_type} '
    # http://stackoverflow.com/questions/20847674/ffmpeg-libx264-height-not-divisible-by-2
    combine_images_cmd += '-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
    combine_images_cmd += '-c:v libx264 -r 30 -pix_fmt yuv420p {movie_name}.mp4'
    if debug_mode:
        combine_images_cmd += ' -loglevel debug'
    rs = utils.partial_run_script(locals())
    rs(combine_images_cmd)


def find_images_props(fol, start_number=-1, images_prefix='', images_format='', images_type=''):
    if images_type == '':
        images_types = set([utils.file_type(image) for image in glob.glob(op.join(fol, '{}*.*'.format(images_prefix)))])
        for opt_type in ['png', 'jpg', 'bmp', 'gif']:
            if opt_type in images_types:
                images_type = opt_type
                print('Images type is {}'.format(images_type))
                break
        if images_type == '':
            raise Exception("Can't find the images type!")
    images = glob.glob(op.join(fol, '{}*.{}'.format(images_prefix, images_type)))
    image_nb = utils.namebase(images[0])
    number = utils.read_numbers_rx(image_nb)[0]
    if images_prefix == '':
        images_prefix = image_nb[:-len(number)]
    if images_format == '':
        images_format = '%0{}d'.format(len(number))
    if start_number == -1:
        start_number = min([int(utils.namebase(image)[len(images_prefix):]) for image in images])
    return images_type, images_prefix, images_format, len(number), start_number


def change_frames_names(fol, images_prefix, images_type, images_format_len, new_fol_name='new_images'):
    import shutil
    images = glob.glob(op.join(fol, '{}*.{}'.format(images_prefix, images_type)))
    images.sort(key=lambda x: int(utils.namebase(x)[len(images_prefix):]))
    images_formats = {1: '{0:0>1}', 2: '{0:0>2}', 3: '{0:0>3}', 4: '{0:0>4}', 5: '{0:0>5}'}
    root = op.join(op.sep.join(images[0].split(op.sep)[:-1]))
    new_fol = op.join(root, new_fol_name)
    utils.delete_folder_files(new_fol)
    utils.make_dir(new_fol)
    for num, image_fname in enumerate(images):
        num_str = images_formats[images_format_len].format(num + 1)
        new_image_fname = op.join(new_fol, '{}{}.{}'.format(images_prefix, num_str, images_type))
        print('{} -> {}'.format(image_fname, new_image_fname))
        shutil.copy(image_fname, new_image_fname)
    return new_fol

if __name__ == '__main__':
    import argparse
    from src.utils import args_utils as au

    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('--fol', required=False)
    parser.add_argument('--movie_name', required=False, default='output')
    parser.add_argument('--out_movie_name', required=False, default='output2')
    parser.add_argument('--ffmpeg_cmd', required=False, default=FFMPEG_CMD)
    parser.add_argument('--frame_rate', required=False, default=10)
    parser.add_argument('-f', '--function', help='function name', required=False, default='combine_images')
    args = utils.Bag(au.parse_parser(parser))
    locals()[args.function](**args)

    # combine_images_to_movie('/autofs/space/thibault_001/users/npeled/mmvt/mg78/figures/inflated_labels_selection', 'inflated_labels_selection',
    #                         ffmpeg_cmd='~/space1/Downloads/ffmpeg-git-static/ffmpeg')

