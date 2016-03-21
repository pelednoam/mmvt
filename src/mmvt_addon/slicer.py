import nibabel as nib
from nibabel.viewers import OrthoSlicer3D
import matplotlib.pyplot as plt
import pickle
import dill
import os


def open_slicer(mri_fname, x, y, z):
    mri_file = nib.load(mri_fname)
    mri_data = mri_file.get_data()
    viewer = OrthoSlicer3D(mri_data, mri_file.affine)
    viewer._set_position(x, y, z)
    viewer.show()


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
        # obj = pickle.load(fp)
        obj = dill.load(fp)
    return obj


def save(obj, fname):
    # print('dump protocol 4!')
    with open(fname, 'wb') as fp:
        # pickle.dump(obj, fp, protocol=4)
        dill.dump(obj, fp, protocol=4)

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
        print('Error! Should send the mri fname, x, y and z')