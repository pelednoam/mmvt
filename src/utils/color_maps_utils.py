import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def create_BuPu_YlOrRd_cm():
    colors1 = plt.cm.PuBu(np.linspace(1, 0, 128))
    colors2 = plt.cm.YlOrRd(np.linspace(0, 1, 128))
    colors = np.vstack((colors1, colors2))
    colors_map = mcolors.LinearSegmentedColormap.from_list('BuPu_YlOrRd', colors)
    return colors_map


def color_map_to_np_mat(colors_map):
    N = colors_map.N
    colors_mat = np.zeros((N, 3))
    for ind, col_name in enumerate(['red', 'green', 'blue']):
        colors_mat[:, ind] = np.array(colors_map._segmentdata[col_name])[:, 1]
    return colors_mat


if __name__ == '__main__':
    create_BuPu_YlOrRd_cm()