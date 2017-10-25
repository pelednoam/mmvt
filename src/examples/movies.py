import os.path as op
import glob
from src.utils import movies_utils as mu
from src.utils import utils


def edit_movie_example():
    movie_fol = '/cluster/neuromind/npeled/Documents/brain-map'
    mu.cut_movie(movie_fol, 'out-7.ogv', 'freeview-mmvt.mp4')
    mu.crop_movie(movie_fol, 'freeview-mmvt.mp4', 'freeview-mmvt_crop.mp4')
    subs = [((0, 4), 'Clicking on the OFC activation in Freeview'),
            ((4, 9), 'The cursor moved to the same coordinates in the MMVT'),
            ((9, 12), 'Finding the closest activation in the coordinates'),
            ((12, 16), 'The activation is displayed with its statistics')]
    mu.add_text_to_movie(movie_fol, 'freeview-mmvt_crop.mp4', 'freeview-mmvt_crop_text.mp4', subs)
    mu.create_animated_gif(movie_fol, 'mg78_elecs_coh_meg_diff.mp4', 'mg78_elecs_coh_meg_diff.gif')


def edit_movie_example2():
    movie_fol = '/home/noam/Videos/mmvt'
    if not op.isdir(movie_fol):
        movie_fol = '/cluster/neuromind/npeled/videos/mmvt'
    subclips_times = [(2, 34)]
    mu.cut_movie(movie_fol, 'out-6.ogv', 'freeview-mmvt-electrodes.mp4', subclips_times)
    mu.crop_movie(movie_fol, 'freeview-mmvt-electrodes.mp4', 'freeview-mmvt-electrodes_crop.mp4')
    subs = [((0, 3), 'Choosing the LAT lead'),
            ((3, 7), 'Choosing the first electrode in the lead (LAT1)'),
            ((7, 11), "The current electrode's averaged evoked response"),
            ((11, 14), "The program estimates the electrode's sources"),
            ((14, 18), "The sources' probabilities are colored from yellow to red"),
            ((18, 20), "The electrode (green dot) in FreeView"),
            ((20, 24), "Going over the different electrodes in the lead"),
            ((24, 26), "By combing MMVT and Freeview"),
            ((26, 32), "The user can benefit from both 3D and 2D views")]
    mu.add_text_to_movie(movie_fol, 'freeview-mmvt-electrodes.mp4', 'freeview-mmvt-electrodes_sub.mp4', subs, fontsize=60)
    mu.create_animated_gif(movie_fol, 'mg78_elecs_coh_meg_diff.mp4', 'mg78_elecs_coh_meg_diff.gif')


def edit_movie_example3():
    movie_fol = '/home/noam/Desktop'
    if not op.isdir(movie_fol):
        movie_fol = '/cluster/neuromind/npeled/videos/mmvt/mmvt-meg-fmri-electrodes3'
    subclips_times = [(2, 46)]
    # mu.cut_movie(movie_fol, 'f4580000-2784.avi', 'mmvt-meg-fmri-electrodes.mp4', subclips_times)
    # mu.crop_movie(movie_fol, 'mmvt-meg-fmri-electrodes.mp4', 'mmvt-meg-fmri-electrodes_crop.mp4')
    subs = [((0, 6), 'The brain is a 3D object'),
            ((6, 13), 'Adding the invasive electrodes'),
            ((13, 17), ' '),
            ((17, 20), "Selecting a time point from all the cortical labels' evoked responses"),
            ((20, 26), "Plotting the MEG for the selected time point"),
            ((26, 31), "Choosing and plotting the fMRI constrast"),
            ((31, 36), " "),
            ((36, 39), "Selecting a time point from an electrode's evoked response"),
            ((39, 44), "Plotting the electrodes' activity")]

    # mu.add_text_to_movie(movie_fol, 'mmvt-meg-fmri-electrodes.mp4', 'mmvt-meg-fmri-electrodes_sub.mp4', subs, fontsize=60)
    mu.create_animated_gif(movie_fol, 'mmvt-meg-fmri-electrodes_sub.mp4', 'mmvt-meg-fmri-electrodes_sub.gif')


def edit_movie_example4():
    movie_fol = '/home/noam/Pictures/mmvt/mg99/lvf4-3_4_1'
    movie_name = 'mg99_LVF4-3_stim_srouces_long.mp4'
    out_movie_name = 'mg99_LVF4-3_stim_srouces.mp4'
    subclips_times = [(0, 29)]
    mu.cut_movie(movie_fol, movie_name, out_movie_name, subclips_times)


def edit_skull_movie():
    movie_fol = '/homes/5/npeled/space1/mmvt/DC/movies'
    subs = [
        ((0, 5),'Skull thickness is plotted on top of the skull according to the colormap (in mm)'),
        ((5, 10), 'Picking the thickness in specific coordinates'),
        ((10, 17), 'Using the MRI to pick a point on the skull'),
        ((17, 20), 'Adding the implantable device'),
        ((20, 26), 'Aligning and rotating the device. The min, max and avg thickness under the device is calculated'),
        ((26, 32), 'Aligning the device to the cursor position')
    ]
    mu.add_text_to_movie(movie_fol, 'skull.mp4', 'skull_sbus.mp4', subs, fontsize=60)


def ski_movie():
    do_merge_images = False
    do_cut = False
    do_crop = True
    do_merge = True

    images_fol = '/home/npeled/mmvt/mg78/figures/meg_white'
    frames = glob.glob(op.join(images_fol, '*.png'))
    frames = sorted(frames, key=utils.natural_keys)
    fps = 1
    frames_output_fname = op.join(images_fol, 'meg_trans.mp4')
    if do_merge_images:
        mu.images_to_video(frames, fps, frames_output_fname)

    movie_fol1 = '/home/npeled/Documents/darpa_year3_meeting'
    movie_name1 = 'Skiing.mp4'
    out_movie_name1 = 'skiing_cut.mp4'
    subclips_times = [(35, 50)]
    if do_cut:
        mu.cut_movie(movie_fol1, movie_name1, out_movie_name1, subclips_times)

    movie_fol2 = '/home/npeled/mmvt/mg78/figures/meg_white'
    movie_name2 = 'meg_white.mp4'
    out_movie_name2 = 'meg_white_crop.mp4'
    if do_crop:
        mu.crop_movie(movie_fol2, movie_name2, out_movie_name2, crop_xs=(100, 670))

    movie1_fname = op.join(movie_fol1, out_movie_name1)
    movie2_fname = op.join(movie_fol2, 'meg_trans.mp4')
    output_fname = op.join(movie_fol1, 'ski_brain.mp4')
    if do_merge:
        mu.movie_in_movie(movie1_fname, movie2_fname, output_fname, pos=('right', 'top'), margin=1)
        # mu.movie_in_movie(movie1_fname, movie2_fname, output_fname, pos=('right', 'top'), margin=0)


def image_in_image():
    from src.utils import figures_utils as fu
    from PIL import Image
    ski_frames = sorted(glob.glob('/home/npeled/Documents/darpa_year3_meeting/ski_frames/*.png'), key=utils.natural_keys)
    meg_frames = sorted(glob.glob('/home/npeled/mmvt/mg78/figures/meg_white/*.png'), key=utils.natural_keys)
    output_fol = '/home/npeled/Documents/darpa_year3_meeting/ski_meg'
    meg_ind = 0
    for ski_ind, ski_frame in enumerate(ski_frames):
        output_frame_fname = op.join(output_fol, 'ski_meg_{}.png'.format(str(ski_ind).zfill(3)))
        print('Writing {}/{}'.format(ski_ind, len(ski_frames)))
        fu.merge_with_alpha(ski_frame, meg_frames[meg_ind], output_frame_fname)
        if ski_ind % 10 == 0:
            meg_ind += 1


if __name__ == '__main__':
    # image_in_image()
    # ski_movie()
    # edit_movie_example3()
    edit_skull_movie()