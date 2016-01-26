import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os.path as op
import glob
from PIL import Image
from src import utils
import time



def ani_frame(time_range, ms_before_stimuli, time_dt, fol, dpi, fps, video_fname):
    def get_t(image_index):
        return int(utils.namebase(images[image_index])[1:])

    def plot_graph(ax2):
        graph_data, graph_colors = utils.load(op.join(fol, 'data.pkl'))
        # for k, values in graph_data.items():
        #     ax2.plot(time_range, values, label=k, color=tuple(graph_colors[k]))

        # fig, ax1 = plt.subplots()
        axes = [ax2]
        if len(graph_data.keys()) > 1:
            ax3 = ax2.twinx()
            axes = [ax2, ax3]
        for (data_type, data_values), ax in zip(graph_data.items(), axes):
            ax.set_ylabel(data_type)
            for k, values in data_values.items():
                ax.plot(time_range, values, label=k, color=tuple(graph_colors[data_type][k]))

        ax2.set_xlabel('Time (ms)')
        labels = list(range(-ms_before_stimuli, len(time_range)-ms_before_stimuli, time_dt))
        labels[1] = 'stimuli'
        ax2.set_xticklabels(labels)

        ymin, ymax = ax2.get_ylim()
        t0 = get_t(0)
        t_line, = ax2.plot([t0, t0], [ymin, ymax], 'g-')
        return t_line, ymin, ymax


    images = sorted(glob.glob(op.join(fol, 'f*.png')), key=lambda x:int(utils.namebase(x)[1:]))#[:20]
    im = Image.open(images[0])
    img_width, img_height = im.size

    print('video: width {} height {} dpi {}'.format(img_width, img_height, dpi))
    w, h = img_width/dpi, img_height/dpi * 1.5
    fig = plt.figure(figsize=(w, h), dpi=dpi)
    fig.canvas.draw()
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    ax = fig.add_subplot(gs[0])
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    image = mpimg.imread(images[0])
    im = ax.imshow(image, animated=True)#, cmap='gray',interpolation='nearest')
    # im.set_clim([0,1])

    ax2 = fig.add_subplot(gs[1])
    # ax2 = utils.load(op.join(fol, 'plt.pkl'))
    # fig.axes.append(ax2)
    t_line, ymin, ymax = plot_graph(ax2)

    # plt.tight_layout()
    now = time.time()

    def init_func():
        return update_img(0)

    def update_img(image_index):
        # print(image_fname)
        utils.time_to_go(now, image_index, len(images))
        image = mpimg.imread(images[image_index])
        im.set_data(image)
        current_t = get_t(image_index)
        t_line.set_data([current_t, current_t], [ymin, ymax])
        return [im]

    ani = animation.FuncAnimation(fig, update_img, len(images), init_func=init_func, interval=30, blit=True)
    writer = animation.writers['ffmpeg'](fps=fps)

    ani.save(op.join(fol, video_fname),writer=writer,dpi=dpi)
    return ani


if __name__ == '__main__':
    fol = '/home/noam/mmvt/mg78/images/4c1c9'
    dpi = 100
    fps = 10
    video_fname = 'mg78_elecs_coh_meg.mp4'
    time_range = range(2500)
    ani_frame(time_range, 500, 500, fol, dpi, fps, video_fname)