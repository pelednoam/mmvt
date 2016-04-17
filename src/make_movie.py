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
import re

LINKS_DIR = utils.get_links_dir()
BLENDER_ROOT_FOLDER = os.path.join(LINKS_DIR, 'mmvt')


def ani_frame(subject, time_range, ms_before_stimuli, labels_time_dt, images, dpi, fps, video_fname, cb_data_type,
        data_to_show_in_graph, fol, fol2, cb_title='', bitrate=5000, images2=(), ylabels=(), xlabels=(),
        xlabel='Time (ms)', show_first_pic=False):
    def get_t(image_index):
        return int(re.findall('\d+', utils.namebase(images[image_index]))[0])

    def plot_graph(graph1_ax, data_to_show_in_graph, fol, fol2='', graph2_ax=None, ylabels=()):
        graph_data, graph_colors = utils.load(op.join(fol, 'data.pkl'))
        if fol2 != '':
            graph_data2, graph_colors2 = utils.load(op.join(fol2, 'data.pkl'))
            if len(graph_data.keys()) == 1 and len(graph_data2.keys()) == 1:
                graph2_data_item = list(graph_data2.keys())[0]
                graph_data['{}2'.format(graph2_data_item)] = graph_data2[graph2_data_item]
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
                # todo: tuple doesn't have ndim, not sure what to do here
                # if graph_colors[data_type][k].ndim > 1:
                #     color = graph_colors[data_type][k][0]
                ax.plot(time_range, values, label=k, color=color, alpha=0.2)# color=tuple(graph_colors[data_type][k]))
            ind += 1

        graph1_ax.set_xlabel(xlabel)
        # labels = list(range(-ms_before_stimuli, len(time_range)-ms_before_stimuli, labels_time_dt))
        # labels[labels.index(0)] = 'stimuli'
        if len(xlabels) > 0:
            graph1_ax.set_xticks([0, 1, 2])
            graph1_ax.set_xticklabels(xlabels)

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
        if cb_data_type != '':
            ax_cb = plt.subplot(gs[:-g2, -1])
        else:
            ax_cb = None
        plt.tight_layout()
        resize_and_move_ax(brain_ax, dx=0.04)
        resize_and_move_ax(brain_ax2, dx=-0.00)
        if cb_data_type != '':
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
        resize_and_move_ax(graph_ax, dx=0.05, dy=0.03, ddw=0.89)
        return ax_cb, im, graph_ax

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
        ax_cb, im, im2, graph1_ax, graph2_ax = two_brains_two_graphs()
    else:
        ax_cb, im, graph1_ax = one_brain_one_graph(gs, g2)
        graph2_ax, im2 = None, None

    # gs.update(left=0.05, right=0.48, wspace=0.05)
    graph_data, graph_colors, t_line, ymin, ymax = plot_graph(
        graph1_ax, data_to_show_in_graph, fol, fol2, graph2_ax, ylabels)
    plot_color_bar(ax_cb, graph_data, cb_title, cb_data_type)

    now = time.time()
    if show_first_pic:
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


def plot_color_bar(ax, graph_data, cb_title, data_type=''):
    if data_type == '':
        return
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


def create_movie(subject, time_range, ms_before_stimuli, labels_time_dt, fol, dpi, fps, video_fname, cb_data_type,
    data_to_show_in_graph, cb_title='', bitrate=5000, fol2='', ylabels=(), xlabels=(), xlabel='Time (ms)',
    pics_type='png', show_first_pic=False, n_jobs=1):

    images1 = get_pics(fol, pics_type)
    images1_chunks = utils.chunks(images1, len(images1) / n_jobs)
    if fol2 != '':
        images2 = get_pics(fol2, pics_type)
        if len(images2) != len(images1):
            raise Exception('fol and fol2 have different number of pictures!')
        images2_chunks = utils.chunks(images2, int(len(images2) / n_jobs))
    else:
        images2_chunks = [''] * int(len(images1) / n_jobs)
    params = [(images1_chunk, images2_chunk, subject, time_range, ms_before_stimuli, labels_time_dt, dpi, fps,
               video_fname, cb_data_type, data_to_show_in_graph, cb_title, bitrate, ylabels, xlabels, xlabel, show_first_pic,
               fol, fol2, run) for run, (images1_chunk, images2_chunk) in enumerate(zip(images1_chunks, images2_chunks))]
    utils.run_parallel(_create_movie_parallel, params, n_jobs)
    video_name, video_type = os.path.splitext(video_fname)
    combine_movies(fol, video_name, video_type[1:])


def combine_movies(fol, movie_name, movie_type='mp4'):
    # First convert the part to avi, because mp4 cannot be concat
    cmd = 'ffmpeg -i concat:"'
    parts = sorted(glob.glob(op.join(fol, '{}_*.{}'.format(movie_name, movie_type))))
    for part_fname in parts:
        part_name, _ = os.path.splitext(part_fname)
        cmd = '{}{}.avi|'.format(cmd, op.join(fol, part_name))
        utils.remove_file('{}.avi'.format(part_name))
        utils.run_script('ffmpeg -i {} -codec copy {}.avi'.format(part_fname, op.join(fol, part_name)))
    # cmd = '{}" -c copy -bsf:a aac_adtstoasc {}'.format(cmd[:-1], op.join(fol, '{}.{}'.format(movie_name, movie_type)))
    cmd = '{}" -c copy {}'.format(cmd[:-1], op.join(fol, '{}.{}'.format(movie_name, movie_type)))
    print(cmd)
    utils.remove_file('{}.{}'.format(op.join(fol, movie_name), movie_type))
    utils.run_script(cmd)
    # clean up
    utils.remove_file('{}.avi'.format(op.join(fol, movie_name)))
    for part_fname in parts:
        part_name, _ = os.path.splitext(part_fname)
        utils.remove_file('{}.avi'.format(part_name))


def _create_movie_parallel(params):
    (images1, images2, subject, time_range, ms_before_stimuli, labels_time_dt, dpi, fps,
        video_fname, cb_data_type, data_to_show_in_graph, cb_title, bitrate, ylabels, xlabels, xlabel,
        show_first_pic, fol, fol2, run) = params
    video_name, video_type = os.path.splitext(video_fname)
    video_fname = '{}_{}{}'.format(video_name, run, video_type)
    ani_frame(subject, time_range, ms_before_stimuli, labels_time_dt, images1, dpi, fps, video_fname, cb_data_type,
        data_to_show_in_graph, fol, fol2, cb_title, bitrate, images2, ylabels, xlabels, xlabel, show_first_pic)


def get_pics(fol, pics_type='png'):
    # return sorted(glob.glob(op.join(fol, '*.{}'.format(pics_type))), key=lambda x:int(utils.namebase(x)[1:]))
    return sorted(glob.glob(op.join(fol, '*.{}'.format(pics_type))), key=lambda x:re.findall('\d+', utils.namebase(x)))


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
    xlabels = []
    cb_data_type = 'meg'
    fps = 10

    fol = '/home/noam/Pictures/mmvt/fsaverage'
    fol2 = ''
    data_to_show_in_graph = ('meg')
    video_fname = 'fsaverage_meg_ttest.mp4'
    cb_title = 'MEG t values'
    ms_before_stimuli, labels_time_dt = 0, 100
    time_range = range(1000)
    ylabels = ['MEG t-values']
    xlabels = []
    cb_data_type = 'meg'
    fps = 10

    fol = '/home/noam/Pictures/mmvt/movie1'
    fol2 = ''
    data_to_show_in_graph = ('meg_labels')
    video_fname = 'mg78_labels_demo.mp4'
    cb_title = 'MEG activity'
    ms_before_stimuli, labels_time_dt = 500, 500
    time_range = range(2500)
    ylabels = ['MEG activity']
    xlabels = []
    cb_data_type = 'meg_labels'
    fps = 10

    subject = ['fsaverage', 'pp009']
    fol = '/home/noam/Videos/mmvt/meg_con/healthy'
    fol2 = '/home/noam/Videos/mmvt/meg_con/pp009'
    data_to_show_in_graph = ('coherence', 'coherence2')
    video_fname = 'pp009_healthy_meg_coh.mp4'
    cb_title = ''
    ms_before_stimuli, labels_time_dt = 0, 1
    time_range = range(3)
    ylabels = ['Healthy', 'pp009']
    xlabels = ['Risk onset', 'Reward onset', 'Shock?']
    xlabel = ''
    cb_data_type = ''
    fps = 100

    dpi = 100
    bitrate = 5000
    pics_type = 'png'
    show_first_pic = True
    n_jobs = 1

    # images = get_pics(fol, pics_type)
    # images2 = get_pics(fol2, pics_type) if fol2 != '' else []
    # ani_frame(subject, time_range, ms_before_stimuli, labels_time_dt, fol, dpi, fps, video_fname,
    #           cb_data_type, data_to_show_in_graph, cb_title, bitrate, fol2=fol2, ylabels=ylabels, pics_type=pics_type)

    create_movie(subject, time_range, ms_before_stimuli, labels_time_dt, fol, dpi, fps, video_fname, cb_data_type,
        data_to_show_in_graph, cb_title, bitrate, fol2, ylabels, xlabels, xlabel, pics_type, show_first_pic, n_jobs)
