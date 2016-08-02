import os.path as op
from src.utils import movies_utils as mu


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
    mu.add_text_to_movie(movie_fol, 'mmvt-meg-fmri-electrodes.mp4', 'mmvt-meg-fmri-electrodes_sub.mp4', subs, fontsize=60)
    # mu.create_animated_gif(movie_fol, 'mg78_elecs_coh_meg_diff.mp4', 'mg78_elecs_coh_meg_diff.gif')


def edit_movie_example4():
    movie_fol = '/home/noam/Pictures/mmvt/mg99/lvf4-3_4_1'
    movie_name = 'mg99_LVF4-3_stim_srouces_long.mp4'
    out_movie_name = 'mg99_LVF4-3_stim_srouces.mp4'
    subclips_times = [(0, 29)]
    mu.cut_movie(movie_fol, movie_name, out_movie_name, subclips_times)


if __name__ == '__main__':
    edit_movie_example3()