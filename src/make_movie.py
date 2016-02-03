import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os.path as op
import glob
from PIL import Image
from src import utils
import time
import numbers
import os
import numpy as np

LINKS_DIR = utils.get_links_dir()
BLENDER_ROOT_FOLDER = os.path.join(LINKS_DIR, 'mmvt')


def ani_frame(subject, time_range, ms_before_stimuli, labels_time_dt, fol, dpi, fps, video_fname, cb_data_type='meg',
              data_to_show_in_graph = ('electrodes', 'coherence'), cb_title='', bitrate=5000, fol2='', ylabels=()):
    def get_t(image_index):
        return int(utils.namebase(images[image_index])[1:])

    def plot_graph(graph1_ax, data_to_show_in_graph, graph2_ax=None, ylabels=()):
        graph_data, graph_colors = utils.load(op.join(fol, 'data.pkl'))
        axes = [graph1_ax]
        if graph2_ax:
            axes = [graph1_ax, graph2_ax]

        ind = 0
        colors = ['r', 'b', 'g']
        for data_type, data_values in graph_data.items():
            if isinstance(data_values, numbers.Number):
                continue
            if data_type not in data_to_show_in_graph:
                continue
            ax = axes[ind]
            ylabel = data_type if len(ylabels) <= ind else ylabels[ind]
            ax.set_ylabel(ylabel, color=colors[ind] if graph2_ax else 'k')
            if graph2_ax:
                for tl in ax.get_yticklabels():
                    tl.set_color(colors[ind])
            for k, values in data_values.items():
                if np.allclose(values, 0):
                    continue
                color = colors[ind] if graph2_ax else tuple(graph_colors[data_type][k])
                ax.plot(time_range, values, label=k, color=color, alpha=0.2)# color=tuple(graph_colors[data_type][k]))
            ind += 1

        graph1_ax.set_xlabel('Time (ms)')
        # labels = list(range(-ms_before_stimuli, len(time_range)-ms_before_stimuli, labels_time_dt))
        # labels[labels.index(0)] = 'stimuli'
        # graph1_ax.set_xticklabels(labels)

        ymin, ymax = graph1_ax.get_ylim()
        t0 = get_t(0)
        t_line, = graph1_ax.plot([t0, t0], [ymin, ymax], 'g-')
        return graph_data, graph_colors, t_line, ymin, ymax

    def two_brains_two_graphs():
        brain_ax = plt.subplot(gs[:-g2, :g3])
        brain_ax.set_aspect('equal')
        brain_ax.get_xaxis().set_visible(False)
        brain_ax.get_yaxis().set_visible(False)

        image = mpimg.imread(images[0])
        im = brain_ax.imshow(image, animated=True)#, cmap='gray',interpolation='nearest')

        brain_ax2 = plt.subplot(gs[:-g2, g3:-1])
        brain_ax2.set_aspect('equal')
        brain_ax2.get_xaxis().set_visible(False)
        brain_ax2.get_yaxis().set_visible(False)

        image2 = mpimg.imread(images2[0])
        im2 = brain_ax2.imshow(image2, animated=True)#, cmap='gray',interpolation='nearest')

        graph1_ax = plt.subplot(gs[-g2:, :])
        graph2_ax = graph1_ax.twinx()
        ax_cb = plt.subplot(gs[:-g2, -1])
        plt.tight_layout()
        resize_and_move_ax(brain_ax, dx=0.04)
        resize_and_move_ax(brain_ax2, dx=-0.00)
        resize_and_move_ax(ax_cb, ddw=0.5, ddh=0.8, dx=-0.01, dy=0.06)
        for graph_ax in [graph1_ax, graph2_ax]:
            resize_and_move_ax(graph_ax, dx=0.04, dy=0.05, ddw=0.89)
        return ax_cb, im, im2, graph1_ax, graph2_ax

    def one_brain_one_graph(gs, g2):
        brain_ax = plt.subplot(gs[:-g2, :-1])
        brain_ax.set_aspect('equal')
        brain_ax.get_xaxis().set_visible(False)
        brain_ax.get_yaxis().set_visible(False)

        image = mpimg.imread(images[0])
        im = brain_ax.imshow(image, animated=True)#, cmap='gray',interpolation='nearest')

        graph_ax = plt.subplot(gs[-g2:, :])
        ax_cb = plt.subplot(gs[:-g2, -1])
        plt.tight_layout()
        # resize_and_move_ax(brain_ax, dx=0.03)
        resize_and_move_ax(ax_cb, ddw=1, dx=-0.06)
        resize_and_move_ax(graph_ax, dx=0.04, dy=0.03, ddw=0.91)
        return ax_cb, im, graph_ax

    images = sorted(glob.glob(op.join(fol, 'f*.png')), key=lambda x:int(utils.namebase(x)[1:]))#[:20]
    first_image = Image.open(images[0])
    img_width, img_height = first_image.size
    print('video: width {} height {} dpi {}'.format(img_width, img_height, dpi))
    img_width_fac = 2 if fol2 != '' else 1.1
    w, h = img_width/dpi * img_width_fac, img_height/dpi * 3/2
    fig = plt.figure(figsize=(w, h), dpi=dpi)
    fig.canvas.draw()
    g = 15
    g2 = int(g / 3)
    g3 = int ((g-1) / 2)
    gs = gridspec.GridSpec(g, g)#, height_ratios=[3, 1])

    if fol2 != '':
        images2 = sorted(glob.glob(op.join(fol2, 'f*.png')), key=lambda x:int(utils.namebase(x)[1:]))#[:20]
        ax_cb, im, im2, graph1_ax, graph2_ax = two_brains_two_graphs()
    else:
        ax_cb, im, graph1_ax = one_brain_one_graph(gs, g2)
        graph2_ax, im2 = None, None

    # gs.update(left=0.05, right=0.48, wspace=0.05)
    graph_data, graph_colors, t_line, ymin, ymax = plot_graph(graph1_ax, data_to_show_in_graph, graph2_ax, ylabels)
    plot_color_bar(ax_cb, graph_data, cb_title, cb_data_type)

    now = time.time()
    plt.show()

    def init_func():
        return update_img(0)

    def update_img(image_index):
        # print(image_fname)
        utils.time_to_go(now, image_index, len(images))
        image = mpimg.imread(images[image_index])
        im.set_data(image)
        if im2:
            image2 = mpimg.imread(images2[image_index])
            im2.set_data(image2)

        current_t = get_t(image_index)
        t_line.set_data([current_t, current_t], [ymin, ymax])
        return [im]

    ani = animation.FuncAnimation(fig, update_img, len(images), init_func=init_func, interval=30, blit=True)
    writer = animation.writers['ffmpeg'](fps=fps, bitrate=bitrate)
    ani.save(op.join(fol, video_fname),writer=writer,dpi=dpi)
    return ani


def plot_color_bar(ax, graph_data, cb_title, data_type):
    import matplotlib as mpl
    data_max = max([max(v) for v in graph_data[data_type].values()])
    data_min = min([min(v) for v in graph_data[data_type].values()])
    data_max_min = utils.get_max_abs(data_max, data_min)
    vmin, vmax = -data_max_min, data_max_min
    cmap = mpl.cm.jet
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical')#, ticks=color_map_bounds)
    cb.set_label(cb_title)


def resize_and_move_ax(ax, dx=0, dy=0, dw=0, dh=0, ddx=1, ddy=1, ddw=1, ddh=1):
    ax_pos = ax.get_position() # get the original position
    ax_pos_new = [ax_pos.x0 * ddx + dx, ax_pos.y0  * ddy + dy,  ax_pos.width * ddw + dw, ax_pos.height * ddh + dh]
    ax.set_position(ax_pos_new) # set a new position


if __name__ == '__main__':
    subject = 'mg78'
    fol = '/home/noam/Pictures/mmvt/movie1'
    fol2 = '/home/noam/Pictures/mmvt/movie2'
    data_to_show_in_graph = ('electrodes', 'coherence')
    video_fname = 'mg78_elecs_coh_meg.mp4'
    cb_title = 'MEG dSPM difference'
    ms_before_stimuli, labels_time_dt = 500, 500
    time_range = range(2500)
    ylabels = []

    fol = '/home/noam/Pictures/mmvt/fsaverage'
    fol2 = ''
    data_to_show_in_graph = ('meg')
    video_fname = 'fsaverage_meg_ttest.mp4'
    cb_title = 'MEG t values'
    ms_before_stimuli, labels_time_dt = 0, 100
    time_range = range(1000)
    ylabels = ['MEG t-values']

    cb_data_type = 'meg'
    dpi = 100
    fps = 10
    bitrate = 5000

    ani_frame(subject, time_range, ms_before_stimuli, labels_time_dt, fol, dpi, fps, video_fname,
              cb_data_type, data_to_show_in_graph, cb_title, bitrate, fol2=fol2, ylabels=ylabels)

