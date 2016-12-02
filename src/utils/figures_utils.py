from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os.path as op
import os
import glob
import numpy as np

PICS_COMB_HORZ, PICS_COMB_VERT = range(2)


def plot_color_bar(data_max, data_min, color_map, ax=None, fol='', do_save=True):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.subplot(199)
    norm = mpl.colors.Normalize(vmin=data_min, vmax=data_max)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=color_map, norm=norm, orientation='vertical')#, ticks=color_map_bounds)
    color_map_name = color_map if isinstance(color_map, str) else color_map.name
    if do_save:
        plt.savefig(op.join(fol, '{}_colorbar.jpg'.format(color_map_name)))
    else:
        plt.show()
    return cb


def plot_color_bar_from_rwo_color_maps(data_max, data_min, fol=''):
    import matplotlib.colors as mcolors

    colors1 = plt.cm.PuBu(np.linspace(1, 0, 128))
    colors2 = plt.cm.YlOrRd(np.linspace(0, 1, 128))
    colors = np.vstack((colors1, colors2))
    mymap = mcolors.LinearSegmentedColormap.from_list('BuPu_YlOrRd', colors)
    plot_color_bar(data_max, data_min, mymap, fol=fol)


def crop_image(image_fname, new_fname, dx=0, dy=0, dw=0, dh=0):
    image = Image.open(image_fname)
    w, h = image.size
    image.crop((dx, dy, w - dw, h -dh)).save(new_fname)


def combine_two_images(figure1_fname, figure2_fname, new_image_fname, comb_dim=PICS_COMB_HORZ, dpi=100,
                       facecolor='black'):
    image1 = Image.open(figure1_fname)
    image2 = Image.open(figure2_fname)
    if comb_dim==PICS_COMB_HORZ:
        new_img_width = image1.size[0] + image2.size[0]
        new_img_height = max(image1.size[1], image2.size[1])
    else:
        new_img_width = max(image1.size[0], image2.size[0])
        new_img_height = image1.size[1] + image2.size[1]
    w, h = new_img_width / dpi, new_img_height / dpi
    fig = plt.figure(figsize=(w, h), dpi=dpi, facecolor=facecolor)
    fig.canvas.draw()
    if comb_dim == PICS_COMB_HORZ:
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
    else:
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)
    ax1.imshow(image1)
    ax2.imshow(image2)
    plt.savefig(new_image_fname, facecolor=fig.get_facecolor(), transparent=True)


def combine_four_brain_perspectives(fol, inflated=False, dpi=100, facecolor='black', crop=True, **kargs):
    figs = []
    patterns = ['*{}{}*'.format(patt, '_inf' if inflated else '') for patt in
                ['lateral_rh', 'lateral_lh', 'medial_rh', 'medial_lh']]
    for patt in patterns:
        files = [f for f in glob.glob(op.join(fol, patt)) if 'crop' not in f]
        if len(files) == 1:
            figs.append(files[0])
        elif len(files) == 0:
            print("Couldn't find {} in {} !!!".format(patt, fol))
    if len(figs) == 4:
        if crop:
            crop_figs = []
            for fig in figs:
                new_fig_fname = '{}_crop{}'.format(op.splitext(fig)[0], op.splitext(fig)[1])
                crop_figs.append(new_fig_fname)
                dx = dw = 20 if inflated else 50
                crop_image(fig, new_fig_fname, dx=dx, dw=dw)
        new_image_fname = combine_four_images(
            crop_figs if crop else figs, op.join(fol, 'splitted_lateral_medial.png'), dpi, facecolor)
        if crop:
            dx = 50 if inflated else 30
            dh = 20 if inflated else 20
            crop_image(new_image_fname, new_image_fname, dx=dx, dh=dh)
    # for fname in glob.glob(op.join(fol, '*crop*')):
    #     os.remove(fname)


def combine_four_images(figs, new_image_fname, dpi=100,
                       facecolor='black'):
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    images = [Image.open(fig) for fig in figs]
    new_img_width = images[0].size[0] + images[1].size[0]
    new_img_height = images[0].size[1] + images[2].size[1]
    w, h = new_img_width / dpi, new_img_height / dpi
    # fig, axes = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(w, h), dpi=dpi, facecolor=facecolor)
    # fig.canvas.draw()
    # axs = list(itertools.chain(*axes))
    fig = plt.figure(figsize=(w, h), dpi=dpi, facecolor=facecolor)
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=0, hspace=0)  # set the spacing between axes.

    for g, image in zip(gs, images):
        ax = plt.subplot(g)
        ax.imshow(image)
        ax.axis('off')
    plt.savefig(new_image_fname, facecolor=fig.get_facecolor(), transparent=True, bbox_inches='tight')
    plt.close()
    return new_image_fname


def combine_nine_images(figs, new_image_fname, dpi=100, facecolor='black'):
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    images = [Image.open(fig) for fig in figs]
    new_img_width =  images[0].size[0] + images[1].size[0] + images[2].size[0]
    new_img_height = images[0].size[1] + images[3].size[1] + images[6].size[1]
    w, h = new_img_width / dpi, new_img_height / dpi
    # fig, axes = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(w, h), dpi=dpi, facecolor=facecolor)
    # fig.canvas.draw()
    # axs = list(itertools.chain(*axes))
    fig = plt.figure(figsize=(w, h), dpi=dpi, facecolor=facecolor)
    gs = gridspec.GridSpec(3, 3)
    gs.update(wspace=0, hspace=0)  # set the spacing between axes.

    for g, image in zip(gs, images):
        ax = plt.subplot(g)
        ax.imshow(image)
        ax.axis('off')
    plt.savefig(new_image_fname, facecolor=fig.get_facecolor(), transparent=True, bbox_inches='tight')
    plt.close()
    return new_image_fname


def combine_brain_with_color_bar(data_max, data_min, figure_fname, colors_map, root, overwrite=False, dpi=100,
                                 x_left_crop=0, x_right_crop=0, y_top_crop=0, y_buttom_crop=0):

    image = Image.open(figure_fname)
    img_width, img_height = image.size
    img_width_fac = 2
    w, h = img_width/dpi * img_width_fac, img_height/dpi * 3/2
    fig = plt.figure(figsize=(w, h), dpi=dpi, facecolor='black')
    fig.canvas.draw()
    gs = gridspec.GridSpec(18, 18)
    brain_ax = plt.subplot(gs[:, :-2])
    plt.tight_layout()
    plt.axis('off')
    im = brain_ax.imshow(image, animated=True)
    ax_cb = plt.subplot(gs[:, -2:-1])
    ax_cb.tick_params(axis='y', colors='white')
    resize_and_move_ax(ax_cb, dy=0.03, ddh=0.92, ddw=0.8, dx=-0.1)
    plot_color_bar(data_max, data_min, colors_map, ax_cb)
    # plt.show()

    image_fname = figure_fname if overwrite else op.join(root, '{}_cb.{}'.format(figure_fname[:-4], figure_fname[-3:]))
    plt.savefig(image_fname, facecolor=fig.get_facecolor(), transparent=True)
    image = Image.open(image_fname)
    w, h = image.size
    image.crop((x_left_crop, y_top_crop, w-x_right_crop, h-y_buttom_crop)).save(image_fname)


def resize_and_move_ax(ax, dx=0, dy=0, dw=0, dh=0, ddx=1, ddy=1, ddw=1, ddh=1):
    ax_pos = ax.get_position() # get the original position
    ax_pos_new = [ax_pos.x0 * ddx + dx, ax_pos.y0  * ddy + dy,  ax_pos.width * ddw + dw, ax_pos.height * ddh + dh]
    ax.set_position(ax_pos_new) # set a new position


def example1():
    figures_fol = '/cluster/neuromind/npeled/mmvt/fsaverage5c/figures/final2'
    colors_map = 'YlOrRd'
    data_max, data_min = 0.2, 0.3

    for fig_name in glob.glob(op.join(figures_fol, '*.png')):
        combine_brain_with_color_bar(
            data_max, data_min, fig_name, colors_map, figures_fol, dpi=100,
            x_left_crop=350, x_right_crop=200)


def example2():
    figures_fol = '/cluster/neuromind/npeled/mmvt/fsaverage5c/figures/connections'
    colors_map = 'YlOrRd'
    data_max, data_min = 0.15, 0.6

    for fig_name in glob.glob(op.join(figures_fol, '*.png')):
        combine_brain_with_color_bar(
            data_max, data_min, fig_name, colors_map, figures_fol, dpi=100,
            x_left_crop=350, x_right_crop=200)


def example3():
    cm_big = 'YlOrRd'
    cm_small = 'PuBu'
    data_max, data_min = -5, 5
    plot_color_bar_from_rwo_color_maps(data_max, data_min, cm_small, cm_big, ax=None, fol='/homes/5/npeled/space1/Pictures')


if __name__ is '__main__':
    import argparse
    from src.utils.utils import Bag
    from src.utils import args_utils as au
    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('--fol', help='folder', required=True)
    parser.add_argument('--inflated', help='inflated', required=False, default=0, type=au.is_true)
    parser.add_argument('--dpi', required=False, default=100, type=int)
    parser.add_argument('--crop', required=False, default=1, type=au.is_true)
    parser.add_argument('--facecolor', required=False, default='black')
    parser.add_argument('-f', '--function', help='function name', required=False, default='combine_four_brain_perspectives')
    args = Bag(au.parse_parser(parser))
    locals()[args.function](**args)

    # example2()
    # combine_two_images('/cluster/neuromind/npeled/Documents/ELA/figs/amygdala_electrode.png',\
    #                    '/cluster/neuromind/npeled/Documents/ELA/figs/grid_most_prob_rois.png',
    #                    '/cluster/neuromind/npeled/Documents/ELA/figs/ela_example2.jpg',comb_dim=PICS_COMB_VERT,
    #                    dpi=100, facecolor='black')
    # example3()
    # plot_color_bar_from_rwo_color_maps(10, -10, fol='C:\\Users\\2014\\mmvt\\ESZC25\\figures')
    # combine_four_brain_perspectives('/homes/5/npeled/space1/mmvt/colin27/figures/ver3', facecolor='black', crop=True)
    print('finish!')