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


def combine_two_colormaps(cm1_name, cm2_name, new_cm_name='', invert_cm1=False, invert_cm2=False, cm1_minmax=(0, 1),
                          cm2_minmax=(0, 1), n=128):
    if new_cm_name == '':
        new_cm_name = '{}-{}'.format(cm1_name, cm2_name)
    cm1_linespace = np.linspace(cm1_minmax[0], cm1_minmax[1], n) if not invert_cm1 else np.linspace(
        cm1_minmax[1], cm1_minmax[0], n)
    cm2_linespace = np.linspace(cm2_minmax[0], cm2_minmax[1], n) if not invert_cm2 else np.linspace(
        cm2_minmax[1], cm2_minmax[0], n)
    colors1 = plt.cm.__dict__[cm1_name](cm1_linespace)
    colors2 = plt.cm.__dict__[cm2_name](cm2_linespace)
    colors = np.vstack((colors1, colors2))
    colors_map = mcolors.LinearSegmentedColormap.from_list(new_cm_name, colors)
    return colors_map


def create_linear_segmented_colormap(cm_name, n=256):
    if cm_name in plt.cm.__dict__:
        colors = plt.cm.__dict__[cm_name](np.linspace(0, 1, n))
    else: # try the inverse cm_name (pubu - bupu)
        if len(cm_name) == 4:
            inverse_cm_name = '{}{}'.format(cm_name[2:4], cm_name[0:2])
        elif len(cm_name) == 6:
            inverse_cm_name = '{}{}{}'.format(cm_name[4:6], cm_name[2:4], cm_name[0:2])
        if inverse_cm_name in plt.cm.__dict__:
            colors = plt.cm.__dict__[inverse_cm_name](np.linspace(1, 0, n))
        else:
            raise Exception('Can\'t find the colormap {}!'.format(cm_name))
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


def get_cm_obj(cm_name, new_cm_name='', invert_cm1=False, invert_cm2=False,  cm1_minmax=(0, 1), cm2_minmax=(0, 1)):
    # special cases:
    if cm_name == 'BuPu-YlOrRd':
        return create_BuPu_YlOrRd_cm()
    elif cm_name == 'PuBu-RdOrYl':
        return create_PuBu_RdOrYl_cm()

    if '-' in cm_name:
        cm1_name, cm2_name = cm_name.split('-')
        if new_cm_name == '':
            new_cm_name = '{}-{}'.format(cm1_name, cm2_name)
        return combine_two_colormaps(cm1_name, cm2_name, new_cm_name, invert_cm1, invert_cm2, cm1_minmax, cm2_minmax)
    else:
        return create_linear_segmented_colormap(cm_name)

        # if cm_name == 'BuPu_YlOrRd':
    #     return create_BuPu_YlOrRd_cm()
    # elif cm_name == 'PuBu_RdOrYl':
    #     return create_PuBu_RdOrYl_cm()
    # else:
    #     if cm2_name == '':
    #         return create_linear_segmented_colormap(cm_name)
    #     else:
    #         if new_cm_name == '':
    #             new_cm_name = '{}_{}'.format(cm_name, cm2_name)
    #         return combine_two_colormaps(cm_name, cm2_name, new_cm_name, cm1_min, cm2_min)
#     cm_func = cms.get(cm_name, None)
#     if cm_func is None:
#         print('{} is not in the cms dic!'.format(cm_name))
#         return None
#     else:
#         return cm_func()


def create_cm(cm_name, new_cm_name='', invert_cm1=False, invert_cm2=False, cm1_minmax=(0, 1), cm2_minmax=(0, 1)):
    if '-' in cm_name:
        cm1_name, cm2_name = cm_name.split('-')
        if new_cm_name == '':
            new_cm_name = '{}-{}'.format(cm1_name, cm2_name)
    else:
        new_cm_name = cm_name
    cm = get_cm_obj(cm_name, new_cm_name, invert_cm1, invert_cm2, cm1_minmax, cm2_minmax)
    cm_mat = color_map_to_np_mat(cm)
    # check_cm_mat(cm_mat)
    save_colors_map(cm_mat, new_cm_name)
    figu.plot_color_bar(1, -1, cm, do_save=True, fol=op.join(MMVT_DIR, 'color_maps'))

#
# cms = {'BuPu_YlOrRd':create_BuPu_YlOrRd_cm, 'PuBu_RdOrYl':create_PuBu_RdOrYl_cm,
#        'YlOrRd':create_YlOrRd_cm, 'RdOrYl': create_RdOrYl_cm, 'gray':create_gray_cm,
#        'jet':create_jet_cm, 'hot':create_hot_cm}

if __name__ == '__main__':
    # create_cm('YlOrRd')
    # create_cm('RdOrYl')
    create_cm('PuBu')
    create_cm('BuPu')
    # create_cm('gray')
    # create_cm('jet')
    # create_cm('hot')
    # create_cm('tab10')
    # create_cm('gist_earth-YlOrRd', cm1_minmax=(0.1, 0.8))# , invert_cm1=True)
    # create_cm('viridis-YlOrRd')