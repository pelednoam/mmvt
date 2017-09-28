import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import nibabel as nib
from nibabel.viewers import OrthoSlicer3D
from nibabel.orientations import axcodes2ornt, aff2axcodes
from nibabel.affines import voxel_sizes
import matplotlib.pyplot as plt
import pickle
import numpy as np
# import dill
import os
import os.path as op

from src.utils import utils
from src.utils.preproc_utils import MMVT_DIR, SUBJECTS_DIR


def open_slicer(mri_fname, x, y, z):
    mri_file = nib.load(mri_fname)
    mri_data = mri_file.get_data()
    viewer = OrthoSlicer3D(mri_data, mri_file.affine)
    viewer._set_position(x, y, z)
    viewer.show()


def save_slices(subject, fname, x, y, z, modality='mri'):
    """ Function to display row of image slices """
    header = nib.load(fname)
    affine = np.array(header.affine, float)
    data = header.get_data()
    images_fol = op.join(MMVT_DIR, subject, 'figures', 'slices')
    utils.make_dir(images_fol)

    clim = np.percentile(data, (1., 99.))
    codes = axcodes2ornt(aff2axcodes(affine))
    order = np.argsort([c[0] for c in codes])
    flips = np.array([c[1] < 0 for c in codes])[order]
    sizes = [data.shape[order] for order in order]
    scalers = voxel_sizes(affine)
    coordinates = np.array([x, y, z])[order].astype(int)

    r = [scalers[order[2]] / scalers[order[1]],
         scalers[order[2]] / scalers[order[0]],
         scalers[order[1]] / scalers[order[0]]]
    for ii, xax, yax, ratio, prespective in zip([0, 1, 2], [1, 0, 0], [2, 2, 1], r, ['Sagital', 'Coronal', 'Axial']):
        fig = plt.figure()
        fig.set_size_inches(1. * sizes[xax] / sizes[yax], 1, forward=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        d = get_image_data(data, order, flips, ii, coordinates)
        ax.imshow(
            d, vmin=clim[0], vmax=clim[1], aspect=1,
            cmap='gray', interpolation='nearest', origin='lower')
        lims = [0, sizes[xax], 0, sizes[yax]]
        ax.axis(lims)
        ax.set_aspect(ratio)
        ax.patch.set_visible(False)
        ax.set_frame_on(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.axes.get_xaxis().set_visible(False)

        x, y, z = coordinates
        image_fname = op.join(images_fol, '{}_{}_{}_{}_{}.png'.format(modality, prespective, x, y, z))
        print('Saving {}'.format(image_fname))
        plt.savefig(image_fname, dpi=sizes[xax])


def get_image_data(image_data, order, flips, ii, pos):
    data = np.rollaxis(image_data, axis=order[ii])[pos[ii]]  # [data_idx] # [pos[ii]]
    xax = [1, 0, 0][ii]
    yax = [2, 2, 1][ii]
    if order[xax] < order[yax]:
        data = data.T
    if flips[xax]:
        data = data[:, ::-1]
    if flips[yax]:
        data = data[::-1]
    return data


# def set_data(image_data, images, order, sizes, idxs, flips):
#     # """Set the plot data using a physical position"""
#     # deal with slicing appropriately
#     data_idx = list()
#     for size, idx in zip(sizes, idxs):
#         data_idx.append(max(min(int(round(idx)), size - 1), 0))
#     for ii in range(3):
#         # sagittal: get to S/A
#         # coronal: get to S/L
#         # axial: get to A/L
#         data = np.rollaxis(image_data, axis=order[ii])[data_idx[ii]]
#         xax = [1, 0, 0][ii]
#         yax = [2, 2, 1][ii]
#         if order[xax] < order[yax]:
#             data = data.T
#         if flips[xax]:
#             data = data[:, ::-1]
#         if flips[yax]:
#             data = data[::-1]
#         images[ii].set_data(data)



def update_slicer(x, y, z):
    viewer = load_viewer()
    viewer._set_position(x, y, z)
    viewer.show()


def save_viewer(viewer):
    current_path = os.path.dirname(os.path.realpath(__file__))
    save(viewer, os.path.join(current_path, 'slicer.pkl'))


def load_viewer():
    current_path = os.path.dirname(os.path.realpath(__file__))
    viewer = load(os.path.join(current_path, 'slicer.pkl'))
    return viewer


def load(fname):
    with open(fname, 'rb') as fp:
        obj = pickle.load(fp)
        # obj = dill.load(fp)
    return obj


def save(obj, fname):
    # print('dump protocol 4!')
    with open(fname, 'wb') as fp:
        pickle.dump(obj, fp, protocol=4)
        # dill.dump(obj, fp, protocol=4)


if __name__ == '__main__':
    import sys
    # print(sys.argv)
    if len(sys.argv) == 6:
        mri_fname = sys.argv[1]
        x, y, z = map(float, sys.argv[2:5])
        print(x, y, z)
        if sys.argv[5] == 'load':
            open_slicer(mri_fname, x, y, z)
        elif sys.argv[5] == 'update':
            update_slicer(x, y, z)
    else:
        # open_slicer('/home/noam/mmvt/mg78/freeview/orig.mgz', 0.9862712 , -45.62893867,  12.2588706)
        subject = 'nmr00979'
        subject = '100307'
        trans_fname = op.join(MMVT_DIR, subject, 'orig_trans.npz')
        d = utils.Bag(np.load(trans_fname))
        ras_tkr2vox = np.linalg.inv(d.vox2ras_tkr)
        mri_fname = op.join(SUBJECTS_DIR, subject, 'mri', 'T1.mgz')
        # ct_fname = op.join(MMVT_DIR, subject, 'freeview', 'ct_nas.nii.gz')
        # x, y, z = 0.9862712 , -45.62893867,  12.2588706
        # x, y, z = 125, 151, 68

        point = np.array([-2.3, 1.1, 2.2]) * 10
        x, y, z = utils.apply_trans(ras_tkr2vox, point).astype(np.int)
        # x, y, z = 151, 106, 139
        # x, y, z = 166, 118, 113
        p = [133, 122, 84]
        # open_slicer(op.join(SUBJECTS_DIR, 'sample', 'mri', 'T1.mgz'), 0, 0, 0)

        # ct_header = nib.load(ct_fname).get_header()
        # ct_vox2ras_tkr = ct_header.get_vox2ras_tkr()
        # ct_ras_tkr2vox = np.linalg.inv(ct_vox2ras_tkr)
        # ct_x, ct_y, ct_z = utils.apply_trans(ct_ras_tkr2vox, point).astype(np.int)

        save_slices(subject, mri_fname, x, y, z, 'mri')
        # save_slices(subject, ct_fname, 128, 139, 110, 'ct')

        # print('Error! Should send the mri fname, x, y and z')