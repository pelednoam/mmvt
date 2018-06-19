import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import os.path as op

from src.utils import utils
from src.utils import figures_utils as figu

LINKS_DIR = utils.get_links_dir()
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')


def create_BuPu_YlOrRd_cm(n=128):
    colors1 = plt.cm.PuBu(np.linspace(1, 0, n))
    colors2 = plt.cm.YlOrRd(np.linspace(0, 1, n))
    colors = np.vstack((colors1, colors2))
    colors_map = mcolors.LinearSegmentedColormap.from_list('BuPu_YlOrRd', colors)
    return colors_map


def create_PuBu_RdOrYl_cm(n=128):
    colors1 = plt.cm.PuBu(np.linspace(0.2, 1, n))
    colors2 = plt.cm.YlOrRd(np.linspace(1, 0.2, n))
    colors = np.vstack((colors1, colors2))
    colors_map = mcolors.LinearSegmentedColormap.from_list('PuBu_RdOrYl', colors)
    return colors_map

#
# def create_jet_cm(n=256):
#     colors = plt.cm.jet(np.linspace(0, 1, n))
#     colors_map = mcolors.LinearSegmentedColormap.from_list('jet', colors)
#     return colors_map
#
#
# def create_gray_cm(n=256):
#     colors = plt.cm.gray(np.linspace(0, 1, n))
#     colors_map = mcolors.LinearSegmentedColormap.from_list('gray', colors)
#     return colors_map
#
#
# def create_YlOrRd_cm(n=256):
#     colors = plt.cm.YlOrRd(np.linspace(0, 1, n))
#     colors_map = mcolors.LinearSegmentedColormap.from_list('YlOrRd', colors)
#     return colors_map
#
#
# def create_RdOrYl_cm(n=256):
#     colors = plt.cm.YlOrRd(np.linspace(1, 0, n))
#     colors_map = mcolors.LinearSegmentedColormap.from_list('YlOrRd', colors)
#     return colors_map
#
#
# def create_hot_cm(n=256):
#     colors = plt.cm.hot(np.linspace(0, 1, n))
#     colors_map = mcolors.LinearSegmentedColormap.from_list('hot', colors)
#     return colors_map


def create_linear_segmented_colormap(cm_name, n=256):
    colors = plt.cm.__dict__[cm_name](np.linspace(0, 1, n))
    colors_map = mcolors.LinearSegmentedColormap.from_list(cm_name, colors)
    return colors_map


def color_map_to_np_mat(colors_map):
    N = colors_map.N
    cm_mat = np.zeros((N, 3))
    for ind, col_name in enumerate(['red', 'green', 'blue']):
        cm_mat[:, ind] = np.array(colors_map._segmentdata[col_name])[:, 1]
    return cm_mat


def save_colors_map(cm_mat, cm_name):
    color_maps_fol = op.join(MMVT_DIR, 'color_maps')
    utils.make_dir(color_maps_fol)
    np.save(op.join(color_maps_fol, '{}.npy'.format(cm_name)), cm_mat)


def check_cm_mat(cm_mat):
    plt.figure()
    for i, color in enumerate(cm_mat):
        plt.plot([0, 1], [i, i], color=color)
    plt.ylim([0, 256])
    plt.show()


def get_cm_obj(cm_name):
    if cm_name == 'BuPu_YlOrRd':
        return create_BuPu_YlOrRd_cm()
    elif cm_name == 'PuBu_RdOrYl':
        return create_PuBu_RdOrYl_cm()
    else:
        return create_linear_segmented_colormap(cm_name)
#     cm_func = cms.get(cm_name, None)
#     if cm_func is None:
#         print('{} is not in the cms dic!'.format(cm_name))
#         return None
#     else:
#         return cm_func()


def create_cm(cm_name):
    cm = get_cm_obj(cm_name)
    cm_mat = color_map_to_np_mat(cm)
    # check_cm_mat(cm_mat)
    save_colors_map(cm_mat, cm_name)
    figu.plot_color_bar(1, -1, cm, do_save=True, fol=op.join(MMVT_DIR, 'color_maps'))

#
# cms = {'BuPu_YlOrRd':create_BuPu_YlOrRd_cm, 'PuBu_RdOrYl':create_PuBu_RdOrYl_cm,
#        'YlOrRd':create_YlOrRd_cm, 'RdOrYl': create_RdOrYl_cm, 'gray':create_gray_cm,
#        'jet':create_jet_cm, 'hot':create_hot_cm}

if __name__ == '__main__':
    # create_cm('YlOrRd')
    # create_cm('RdOrYl')
    # create_cm('gray')
    # create_cm('jet')
    # create_cm('hot')
    create_cm('tab10')