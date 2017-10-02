import scipy.io as sio
import os.path as op
import numpy as np

from src.utils import utils
from src.utils import labels_utils as lu

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
MMVT_DIR = op.join(LINKS_DIR, 'mmvt')


def create_coloring(x, subject, atlas, conditions, colors_map='YlOrRd', exclude=['unknown', 'corpuscallosum'],
                    colors_min_val=None, colors_max_val=None):
    labels = lu.read_labels(subject, SUBJECTS_DIR, atlas, exclude=tuple(exclude), sorted_according_to_annot_file=True,
                            only_names=True)
    for cond_id, cond_name in enumerate(conditions):
        values = x[:, cond_id]
        if colors_min_val is None:
            colors_min_val = np.min(x)
        if colors_max_val is None:
            colors_max_val = np.max(x)
        colors = utils.arr_to_colors(values, colors_min_val, colors_max_val, colors_map=colors_map)
        coloring_fname = op.join(MMVT_DIR, subject, 'coloring', 'labels_{}_coloring.csv'.format(cond_name))
        write_coloring_file(coloring_fname, labels, colors)
    values_diff = np.squeeze(np.diff(x))
    abs_max = max(map(abs, [np.max(values_diff), np.min(values_diff)]))
    colors = utils.mat_to_colors(values_diff, -abs_max, abs_max, 'RdBu', flip_cm=True)
    coloring_fname = op.join(MMVT_DIR, subject, 'coloring', 'labels_{}_{}_diff_coloring.csv'.format(*conditions))
    write_coloring_file(coloring_fname, labels, colors)


def write_coloring_file(coloring_fname, labels, colors):
    with open(coloring_fname, 'w') as output_file:
        for label_name, color_rgb in zip(labels, colors):
            output_file.write('{},{},{},{}\n'.format(label_name, *color_rgb))


if __name__ == '__main__':
    subject = 'fsaverage5c'
    fsaverage = 'fsaverage5c'
    atlas = 'laus125'
    colors_map = 'YlOrRd'

    root = '/cluster/neuromind/npeled/linda'
    values_mat = sio.loadmat(op.join(root, 'figure_file2.mat'))['figure_file2']
    conditions = ['task', 'rest']
    exclude = ['unknown', 'corpuscallosum']
    create_coloring(values_mat, subject, atlas, conditions, colors_map, exclude, 0.25, 0.3)
    print('finish!')