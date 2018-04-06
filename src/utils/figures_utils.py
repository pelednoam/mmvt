from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os.path as op
import os
import glob
import numpy as np

from src.utils import utils
from src.utils import color_maps_utils as cmu

PICS_COMB_HORZ, PICS_COMB_VERT = range(2)


# @utils.tryit()
def plot_color_bar(data_max, data_min, colors_map, ax=None, fol='', do_save=True, cb_ticks=None,
                   background_color='black', cb_ticks_font_size=10, cb_title='', colorbar_name='', dpi=100, **kargs):
    import matplotlib as mpl

    if ',' in background_color:
        background_color = [float(x) for x in background_color.split(',')]
    color_map_name = colors_map if isinstance(colors_map, str) else colors_map.name
    color_map = find_color_map(colors_map)
    if ax is None:
        fig = plt.figure(dpi=dpi, facecolor=background_color) #, figsize=(w, h))
        fig.canvas.draw()
        ax = plt.gca() #plt.subplot(199)
        ax.tick_params(axis='y', colors='white' if background_color in ['black', [0, 0, 0]] else 'black')
    norm = mpl.colors.Normalize(vmin=data_min, vmax=data_max)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=color_map, norm=norm, orientation='vertical')#, ticks=color_map_bounds)
    if cb_ticks is not None:
        cb.set_ticks(cb_ticks)
        cb.ax.tick_params(labelsize=cb_ticks_font_size)
    if cb_title != '':
        cb.ax.set_ylabel(cb_title.strip(), color='white' if background_color in ['black', [0, 0, 0]] else 'black')
    resize_and_move_ax(ax, ddw=0.07, ddh=0.8)
    if colorbar_name == '':
        colorbar_name = '{}_colorbar.jpg'.format(color_map_name)
    if do_save:
        fname = op.join(fol, colorbar_name)
        plt.savefig(fname, facecolor=fig.get_facecolor(), transparent=True, bbox_inches='tight')
    else:
        plt.show()
    return cb


def find_color_map(color_map):
    color_map_name = color_map if isinstance(color_map, str) else color_map.name
    if color_map_name not in plt.cm.cmap_d:
        color_map_name = color_map_name.replace('-', '_')
        if color_map_name in cmu.cms:
            color_map = cmu.get_cm_obj(color_map_name)
        else:
            raise Exception("Can't find colormap {}!".format(color_map_name))
    return color_map


def plot_color_bar_from_two_color_maps(data_max, data_min, fol='', **kargs):
    import matplotlib.colors as mcolors

    colors1 = plt.cm.PuBu(np.linspace(1, 0, 128))
    colors2 = plt.cm.YlOrRd(np.linspace(0, 1, 128))
    colors = np.vstack((colors1, colors2))
    mymap = mcolors.LinearSegmentedColormap.from_list('BuPu_YlOrRd', colors)
    plot_color_bar(data_max, data_min, mymap, fol=fol)


def crop_image(image_fname, new_fname, dx=0, dy=0, dw=0, dh=0, **kargs):
    image = Image.open(image_fname)
    w, h = image.size
    image.crop((dx, dy, w - dw, h -dh)).save(new_fname)


def combine_two_images(figure1_fname, figure2_fname, new_image_fname, comb_dim=PICS_COMB_HORZ, dpi=100,
                       facecolor='black', w_fac=1, h_fac=1, **kargs):
    image1 = Image.open(figure1_fname)
    image2 = Image.open(figure2_fname)
    if comb_dim==PICS_COMB_HORZ:
        new_img_width = image1.size[0] + image2.size[0]
        new_img_height = max(image1.size[1], image2.size[1])
    else:
        new_img_width = max(image1.size[0], image2.size[0])
        new_img_height = image1.size[1] + image2.size[1]
    w, h = new_img_width / dpi * w_fac, new_img_height / dpi * h_fac
    fig = plt.figure(figsize=(w, h), dpi=dpi, facecolor=facecolor)
    fig.canvas.draw()

    gs = gridspec.GridSpec(1, 2)
    gs.update(wspace=0, hspace=0)  # set the spacing between axes.
    for g, image in zip(gs, (image1, image2)):
        ax = plt.subplot(g)
        ax.imshow(image)
        ax.axis('off')
    plt.savefig(new_image_fname, facecolor=fig.get_facecolor(), transparent=True, bbox_inches='tight')

    # if comb_dim == PICS_COMB_HORZ:
    #     ax1 = plt.subplot(121)
    #     ax2 = plt.subplot(122)
    # else:
    #     ax1 = plt.subplot(211)
    #     ax2 = plt.subplot(212)
    # ax1.imshow(image1)
    # ax2.imshow(image2)
    # plt.savefig(new_image_fname, facecolor=fig.get_facecolor(), transparent=True)


def combine_two_figures_with_cb(fname1, fname2, data_max, data_min, cb_cm, cb_ticks=[], crop_figures=True,
                                background='black', cb_ticks_font_size=10):
    if crop_figures:
        crop_image(fname1, fname1, dx=150, dy=0, dw=50, dh=70)
        crop_image(fname2, fname2, dx=150 + 50, dy=0, dw=0, dh=70)
    new_image_fname = op.join(utils.get_parent_fol(fname1), '{}_{}.{}'.format(
        utils.namebase(fname1), utils.namebase(fname2), utils.file_type(fname1)))
    combine_two_images(fname1, fname2, new_image_fname, facecolor=background, dpi=200, w_fac=1, h_fac=1)
    if len(cb_ticks) == 0:
        cb_ticks = [data_min, data_max]
    fol = utils.get_parent_fol(fname1)
    plot_color_bar(data_max, data_min, cb_cm, do_save=True, cb_ticks=cb_ticks, fol=fol, background_color=background,
                   cb_ticks_font_size=cb_ticks_font_size)
    cb_fname = op.join(fol, '{}_colorbar.jpg'.format(cb_cm))
    cb_img = Image.open(cb_fname)
    return combine_brain_with_color_bar(new_image_fname, cb_img, overwrite=True)


def combine_four_brain_perspectives_output_fname(fol, inflated=False, facecolor='black', clusters_name=''):
    return op.join(fol, '{}splitted_lateral_medial_{}_{}.png'.format(
        '{}_'.format(clusters_name if clusters_name != '' else ''),
        'inflated' if inflated else 'pial', facecolor))


def get_brain_perspectives_figures(fol, inflated=False, facecolor='black', clusters_name='', inflated_ratio=1):
    figs = []
    patterns = ['{}{}_{}_{}*'.format('{}_'.format(clusters_name if clusters_name != '' else ''), perpective,
        'inflated_{}'.format(inflated_ratio) if inflated else 'pial', facecolor) for perpective in
        ['lateral_rh', 'lateral_lh', 'medial_rh', 'medial_lh']]
    for patt in patterns:
        files = [f for f in glob.glob(op.join(fol, patt)) if 'crop' not in f]
        if len(files) == 1:
            figs.append(files[0])
        elif len(files) == 0:
            print("Couldn't find {} in {} !!!".format(patt, fol))
    return figs


def combine_four_brain_perspectives(fol, inflated=False, dpi=100, facecolor='black', clusters_name='', inflated_ratio=1,
                                    crop=True, overwrite=True, **kargs):
    figs = get_brain_perspectives_figures(fol, inflated, facecolor, clusters_name, inflated_ratio)
    if len(figs) == 4:
        fig_name = combine_four_brain_perspectives_output_fname(fol, inflated, facecolor, clusters_name)
        if overwrite or not op.isfile(fig_name):
            if crop:
                crop_figs = []
                for fig in figs:
                    new_fig_fname = '{}_crop{}'.format(op.splitext(fig)[0], op.splitext(fig)[1])
                    crop_figs.append(new_fig_fname)
                    dx = dw = 20 if inflated else 50
                    crop_image(fig, new_fig_fname, dx=dx, dw=dw)
            combine_four_images(
                crop_figs if crop else figs, fig_name, dpi, facecolor)
            if crop:
                dx = 50 if inflated else 30
                dh = 20 if inflated else 20
                crop_image(fig_name, fig_name, dx=dx, dh=dh)
    else:
        raise Exception('Wrong number of perspectives! {}'.format(figs))
    for fname in glob.glob(op.join(fol, '*crop*')):
        os.remove(fname)
    return fig_name


def combine_four_images(figs, new_image_fname, dpi=100, facecolor='black', **kargs):
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


def combine_nine_images(figs, new_image_fname, dpi=100, facecolor='black', **kargs):
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


def add_colorbar_to_image(figure_fname, data_max, data_min, colors_map, background_color='black',
                          cb_ticks=[], cb_ticks_font_size=10, cb_title='', **kargs):
    fol = utils.get_fname_folder(figure_fname)
    if ',' in background_color:
        background_color = [float(x) for x in background_color.split(',')]
    if len(cb_ticks) == 0:
        cb_ticks = [data_min, data_max]
    cb_fname = op.join(fol, '{}_colorbar.jpg'.format(colors_map))
    plot_color_bar(data_max, data_min, colors_map, do_save=True, cb_ticks=cb_ticks, fol=fol,
                   background_color=background_color, cb_ticks_font_size=cb_ticks_font_size, title=cb_title)
    cb_img = Image.open(cb_fname)
    # crop_image(figure_fname, figure_fname, dx=150, dy=0, dw=150, dh=0)
    combine_brain_with_color_bar(figure_fname, cb_img, overwrite=True, cb_ticks=cb_ticks)


def combine_two_images_and_add_colorbar(lh_figure_fname, rh_figure_fname, new_image_fname, data_max, data_min,
        colors_map, background_color='black', cb_ticks=[], cb_ticks_font_size=10, add_cb=True, crop_figures=True,
        remove_original_figures=False, cb_title='', **kargs):
    fol = utils.get_fname_folder(lh_figure_fname)
    if ',' in background_color:
        background_color = [float(x) for x in background_color.split(',')]
    cb_fname = op.join(fol, '{}_colorbar.jpg'.format(colors_map))
    plot_color_bar(data_max, data_min, colors_map, do_save=True, cb_ticks=cb_ticks, fol=fol,
                   facecolor=background_color, cb_ticks_font_size=cb_ticks_font_size, title=cb_title)
    cb_img = Image.open(cb_fname)
    if add_cb:
        if crop_figures:
            crop_image(lh_figure_fname, lh_figure_fname, dx=150, dy=0, dw=50, dh=70)
            crop_image(rh_figure_fname, rh_figure_fname, dx=150 + 50, dy=0, dw=0, dh=70)
        combine_two_images(lh_figure_fname, rh_figure_fname, new_image_fname, facecolor=background_color,
                           dpi=200, w_fac=1, h_fac=1)
        combine_brain_with_color_bar(new_image_fname, cb_img, overwrite=True, cb_ticks=cb_ticks)
    else:
        if crop_figures:
            crop_image(lh_figure_fname, lh_figure_fname, dx=150, dy=0, dw=150, dh=0)
            crop_image(rh_figure_fname, dx=150, dy=0, dw=150, dh=0)
        combine_two_images(lh_figure_fname, rh_figure_fname, new_image_fname, facecolor=background_color)
    if remove_original_figures:
        if lh_figure_fname != new_image_fname:
            utils.remove_file(lh_figure_fname)
        if rh_figure_fname != new_image_fname:
            utils.remove_file(rh_figure_fname)
        # utils.remove_file(cb_fname)


def combine_brain_with_color_bar(image_fname, cb_img=None, w_offset=10, overwrite=False, cb_max=None, cb_min=None,
                                 cb_cm=None, background='black', cb_ticks=[], cb_ticks_font_size=10):
    if cb_img is None:
        if len(cb_ticks) == 0:
            cb_ticks = [cb_min, cb_max]
        fol = utils.get_parent_fol(image_fname)
        cb_fname = op.join(fol, '{}_colorbar.jpg'.format(cb_cm))
        plot_color_bar(cb_max, cb_min, cb_cm, do_save=True, cb_ticks=cb_ticks, fol=fol, background_color=background,
                       cb_ticks_font_size=cb_ticks_font_size)
        cb_img = Image.open(cb_fname)

    background = Image.open(image_fname)
    bg_w, bg_h = background.size
    cb_w, cb_h = cb_img.size
    offset = (int((bg_w - cb_w)) - w_offset, int((bg_h - cb_h) / 2))
    background.paste(cb_img, offset)
    if not overwrite:
        image_fol = utils.get_fname_folder(image_fname)
        image_fname = op.join(image_fol, '{}_cb.{}'.format(image_fname[:-4], image_fname[-3:]))
    background.save(image_fname)
    return image_fname


def combine_brain_with_color_bar_old(data_max, data_min, figure_fname, colors_map, overwrite=False, dpi=100,
                                 x_left_crop=0, x_right_crop=0, y_top_crop=0, y_buttom_crop=0,
                                 w_fac=2, h_fac=3/2, facecolor='black', ticks=None,
                                 dy=0.03, ddh=0.92, ddw=0.8, dx=-0.1, **kargs):
    image_fol = utils.get_fname_folder(figure_fname)
    if not overwrite:
        image_fname = op.join(image_fol, '{}_cb.{}'.format(figure_fname[:-4], figure_fname[-3:]))
    else:
        image_fname = figure_fname
    # if op.isfile(image_fname) and not overwrite:
    #     return
    image = Image.open(figure_fname)
    img_width, img_height = image.size
    w, h = (img_width/dpi) * w_fac, (img_height/dpi) * h_fac
    fig = plt.figure(figsize=(w, h), dpi=dpi, facecolor=facecolor)
    fig.canvas.draw()
    gs = gridspec.GridSpec(18, 18)
    gs.update(wspace=0, hspace=0)  # set the spacing between axes.
    brain_ax = plt.subplot(gs[:, :-2])
    plt.tight_layout()
    plt.axis('off')
    im = brain_ax.imshow(image, animated=True)
    ax_cb = plt.subplot(gs[:, -2:-1])
    ax_cb.tick_params(axis='y', colors='white' if facecolor=='black' else 'black')
    resize_and_move_ax(ax_cb, dy=dy, dx=dx, ddh=ddh, ddw=ddw)
    plot_color_bar(data_max, data_min, colors_map, ax_cb, cb_ticks=ticks)
    plt.savefig(image_fname, facecolor=fig.get_facecolor(), transparent=True)
    plt.close()
    image = Image.open(image_fname)
    w, h = image.size
    image.crop((x_left_crop, y_top_crop, w-x_right_crop, h-y_buttom_crop)).save(image_fname)


def resize_and_move_ax(ax, dx=0, dy=0, dw=0, dh=0, ddx=1, ddy=1, ddw=1, ddh=1, **kargs):
    ax_pos = ax.get_position() # get the original position
    ax_pos_new = [ax_pos.x0 * ddx + dx, ax_pos.y0  * ddy + dy,  ax_pos.width * ddw + dw, ax_pos.height * ddh + dh]
    ax.set_position(ax_pos_new) # set a new position


def merge_with_alpha(background, foreground, output_fname, pos=(0,0), fg_ratio=(1/2, 1/2),
                     margin=0, delta=(0,0)):
    from PIL import Image

    if isinstance(background, str):
        background = Image.open(background)
    elif not isinstance(background, object):
        raise Exception('background: Can be file name or Image object')
    if isinstance(background, str):
        foreground = Image.open(foreground)
    elif not isinstance(foreground, object):
        raise Exception('background: Can be file name or Image object')
    # w, h = foreground.size
    # foreground = foreground.crop((100, 0, 670, h))
    fw, fh = foreground.size
    bw, bh = background.size
    foreground = foreground.resize((int(fw * fg_ratio[0]), int(fh * fg_ratio[1])))
    pos = list(pos)
    if pos[0] == 'left':
        pos[0] = margin + delta[0]
    elif pos[0] == 'right':
        pos[0] = bw - fw - margin + delta[0]
    elif pos[0] < 0:
        pos[0] = bw + pos[0]
    if pos[1] == 'top':
        pos[1] = margin + delta[1]
    elif pos[1] == 'bottom':
        pos[1] = bh - fh - margin + delta[1]
    elif pos[1] < 0:
        pos[1] = bh + pos[1] + delta[1]
    background.paste(foreground, pos, foreground)
    background.save(output_fname)


def get_image_w_h(image_fname):
    image = Image.open(image_fname)
    w, h = image.size
    return w, h


if __name__ is '__main__':
    import argparse
    from src.utils.utils import Bag
    from src.utils import args_utils as au
    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('--fol', help='folder', required=False)
    parser.add_argument('--inflated', help='inflated', required=False, default=0, type=au.is_true)
    parser.add_argument('--dpi', required=False, default=100, type=int)
    parser.add_argument('--crop', required=False, default=1, type=au.is_true)
    parser.add_argument('--facecolor', required=False, default='black')

    parser.add_argument('--data_max', required=False, default=0, type=float)
    parser.add_argument('--data_min', required=False, default=0, type=float)
    parser.add_argument('--figure_fname', required=False, default='')
    parser.add_argument('--lh_figure_fname', required=False, default='')
    parser.add_argument('--rh_figure_fname', required=False, default='')
    parser.add_argument('--new_image_fname', required=False, default='')

    parser.add_argument('--colors_map', required=False, default='')
    parser.add_argument('--x_left_crop', required=False, default=0, type=float)
    parser.add_argument('--x_right_crop', required=False, default=0, type=float)
    parser.add_argument('--y_top_crop', required=False, default=0, type=float)
    parser.add_argument('--y_buttom_crop', required=False, default=0, type=float)
    parser.add_argument('--w_fac', required=False, default=2, type=float)
    parser.add_argument('--h_fac', required=False, default=3/2, type=float)
    parser.add_argument('--background_color', required=False, default='black')
    parser.add_argument('--colorbar_name', required=False, default='')
    parser.add_argument('--cb_title', required=False, default='')
    parser.add_argument('--cb_ticks', required=False, default='', type=au.float_arr_type)
    parser.add_argument('--cb_ticks_font_size', required=False, default=10, type=int)
    parser.add_argument('--add_cb', required=False, default=1, type=au.is_true)
    parser.add_argument('--crop_figures', required=False, default=1, type=au.is_true)
    parser.add_argument('--remove_original_figures', required=False, default=0, type=au.is_true)

    parser.add_argument('-f', '--function', help='function name', required=True,
                        default='combine_four_brain_perspectives', type=au.str_arr_type)
    args = Bag(au.parse_parser(parser))
    for func in args.function:
        locals()[func](**args)

    # combine_two_figures_with_cb('/home/npeled/mmvt/colin27/figures/image_21.png',
    #                             '/home/npeled/mmvt/colin27/figures/image_22.png', 4.051952362060547, 0.0, 'YlOrRd')
    #

    # example2()
    # combine_two_images('/cluster/neuromind/npeled/Documents/ELA/figs/amygdala_electrode.png',\
    #                    '/cluster/neuromind/npeled/Documents/ELA/figs/grid_most_prob_rois.png',
    #                    '/cluster/neuromind/npeled/Documents/ELA/figs/ela_example2.jpg',comb_dim=PICS_COMB_VERT,
    #                    dpi=100, facecolor='black')
    # example3()
    # plot_color_bar_from_two_color_maps(10, -10, fol='C:\\Users\\2014\\mmvt\\ESZC25\\figures')
    # combine_four_brain_perspectives('/homes/5/npeled/space1/mmvt/colin27/figures/ver3', facecolor='black', crop=True)
    print('finish!')