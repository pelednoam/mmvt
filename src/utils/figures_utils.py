import os.path as op
import glob


def plot_color_bar(data_max, data_min, color_map, ax=None, fol='', do_save=False):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.subplot(199)
    norm = mpl.colors.Normalize(vmin=data_min, vmax=data_max)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=color_map, norm=norm, orientation='vertical')#, ticks=color_map_bounds)
    plt.savefig(op.join(fol, '{}_colorbar.jpg'.format(color_map)))
    return cb


def combine_brain_with_color_bar(data_max, data_min, figure_fname, colors_map, root, overwrite=False, dpi=100,
                                 x_left_crop=0, x_right_crop=0, y_top_crop=0, y_buttom_crop=0):
    from PIL import Image
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

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


if __name__ is '__main__':
    example2()