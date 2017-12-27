import os.path as op
import glob
import numpy as np
import matplotlib.pyplot as plt

from src.utils import utils

links_dir = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(links_dir, 'subjects', 'SUBJECTS_DIR')
MMVT_DIR = utils.get_link_dir(links_dir, 'mmvt')


def plot_ct_vals(subject):
    files = glob.glob(op.join(MMVT_DIR, subject, 'ct', 'voxel_neighbors_ct_values_*.npy'))
    if len(files) == 0:
        return
    fname = sorted([(op.getmtime(f), f) for f in files])[-1][1]
    print('Loading {}'.format(fname))
    ct_vals = np.load(fname)
    plt.hist(ct_vals, bins=len(ct_vals))
    plt.title(utils.namebase(fname))
    plt.show()


if __name__ == '__main__':
    plot_ct_vals('mg105')