import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os.path as op
import glob
from PIL import Image
from src import utils
import time


def ani_frame(fol, dpi, fps):
    images = sorted(glob.glob(op.join(fol, '*.png')), key=lambda x:int(utils.namebase(x)[1:]))
    im = Image.open(images[0])
    img_width, img_height = im.size

    print('video: width {} height {} dpi {}'.format(img_width, img_height, dpi))
    fig = plt.figure(figsize=(img_width/dpi, img_height/dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    image = mpimg.imread(images[0])
    im = ax.imshow(image, animated=True)#, cmap='gray',interpolation='nearest')
    im.set_clim([0,1])
    plt.tight_layout()
    now = time.time()

    def init_func():
        return update_img(0)

    def update_img(image_index):
        # print(image_fname)
        utils.time_to_go(now, image_index, len(images))
        image = mpimg.imread(images[image_index])
        im.set_data(image)
        return [im]

    ani = animation.FuncAnimation(fig, update_img, len(images), init_func=init_func, interval=30, blit=True)
    writer = animation.writers['ffmpeg'](fps=fps)

    ani.save(op.join(fol, 'demo.mp4'),writer=writer,dpi=dpi)
    # ani.save(op.join(fol, 'demo.mp4'))

    return ani


if __name__ == '__main__':
    fol = '/home/noam/mmvt/movies/mg78_elecs_coh_meg'
    dpi = 100
    fps = 10
    ani_frame(fol, dpi, fps)