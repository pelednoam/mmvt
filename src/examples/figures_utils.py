import os.path as op
import glob
from src.utils import figures_utils as fu
from src.utils import utils

LINKS_DIR = utils.get_links_dir()
MMVT_DIR = op.join(LINKS_DIR, 'mmvt')


def example1():
    figures_fol = '/cluster/neuromind/npeled/mmvt/fsaverage5c/figures/final2'
    colors_map = 'YlOrRd'
    data_max, data_min = 0.2, 0.3

    for fig_name in glob.glob(op.join(figures_fol, '*.png')):
        fu.combine_brain_with_color_bar(
            data_max, data_min, fig_name, colors_map, dpi=100,
            x_left_crop=350, x_right_crop=200)


def example2():
    figures_fol = '/cluster/neuromind/npeled/mmvt/fsaverage5c/figures/connections'
    colors_map = 'YlOrRd'
    data_max, data_min = 0.15, 0.6

    for fig_name in glob.glob(op.join(figures_fol, '*.png')):
        fu.combine_brain_with_color_bar(
            data_max, data_min, fig_name, colors_map, dpi=100,
            x_left_crop=350, x_right_crop=200)


def example3():
    cm_big = 'YlOrRd'
    cm_small = 'PuBu'
    data_max, data_min = -5, 5
    fu.plot_color_bar_from_two_color_maps(data_max, data_min, cm_small, cm_big, ax=None, fol='/homes/5/npeled/space1/Pictures')


def example4(subject='colin27', map_name='s32_spmT', figure_name='splitted_lateral_medial_pial_white.png'):
    data_min, data_max = utils.load(
        op.join(MMVT_DIR, subject, 'fmri', 'fmri_activity_map_minmax_{}.pkl'.format(map_name)))
    data_min = utils.ceil_floor(data_min)
    data_max = utils.ceil_floor(data_max)
    figure_fname = op.join(MMVT_DIR, subject, 'figures', figure_name)
    colors_map = 'BuPu_YlOrRd'
    background = 'white' if 'white' in figure_name else 'black'
    fu.combine_brain_with_color_bar(
        data_max, data_min, figure_fname, colors_map,
        x_left_crop=300, x_right_crop=300, y_top_crop=0, y_buttom_crop=0,
        w_fac=1.5, h_fac=1, facecolor=background)


def example5():
    figures_fol = '/home/npeled/mmvt/nmr00698/figures/'
    colors_map = 'BuPu_YlOrRd'
    data_max, data_min = 1, -1

    for fig_name in glob.glob(op.join(figures_fol, '*.png')):
        fu.combine_brain_with_color_bar(
            data_max, data_min, fig_name, colors_map, dpi=100)


def example6():
    figures_fol = '/home/npeled/mmvt/nmr01216/figures'
    colors_map = 'RdOrYl'
    data_max, data_min = 2, 6
    background = '#393939'

    files = glob.glob(op.join(figures_fol, '*.png'))
    images_hemi_inv_list = set([utils.namebase(fname)[3:] for fname in files if utils.namebase(fname)[:2] in ['rh', 'lh']])
    files = [[fname for fname in files if utils.namebase(fname)[3:] == img_hemi_inv] for img_hemi_inv in images_hemi_inv_list]
    for files_coup in files:
        hemi = 'rh' if utils.namebase(files_coup[0]).startswith('rh') else 'lh'
        coup_template = files_coup[0].replace(hemi, '{hemi}')
        coup = {}
        for hemi in utils.HEMIS:
            coup[hemi] = coup_template.format(hemi=hemi)
        new_image_fname = op.join(utils.get_fname_folder(files_coup[0]), utils.namebase_with_ext(files_coup[0])[3:])

        fu.crop_image(coup['lh'], coup['lh'], dx=150, dy=0, dw=150, dh=0)
        fu.crop_image(coup['rh'], coup['rh'], dx=150, dy=0, dw=0, dh=0)
        fu.combine_two_images(coup['lh'], coup['rh'], new_image_fname, facecolor=background)
        fu.combine_brain_with_color_bar(
            data_max, data_min, new_image_fname, colors_map, dpi=200, overwrite=True,
            w_fac=1.2, h_fac=1.2, ddh=0.7, dy=0.13, ddw=0.4, dx=-0.08)
        for hemi in utils.HEMIS:
            utils.remove_file(coup[hemi])
        # fu.combine_brain_with_color_bar(
        #     data_max, data_min, new_image_fname, colors_map, dpi=100, overwrite=True, w_fac=1.2, h_fac=1.2, ddh=0.7, dy=0.13)


def example7():
    images_names = glob.glob('/home/npeled/mmvt/nmr01216/figures/back/*.png')
    for fig_name in images_names:
        fu.combine_brain_with_color_bar(
            data_max, data_min, fig_name, args.cb_cm, dpi=100, overwrite=True, ticks=ticks,
            w_fac=1.2, h_fac=1.2, ddh=0.7, dy=0.13)


if __name__ == '__main__':
    example6()

