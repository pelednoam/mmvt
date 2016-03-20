from nibabel.viewers import OrthoSlicer3D
import matplotlib.pyplot as plt


def open_slicer(mri_fname, x, y, z):
    import nibabel as nib
    mri_file = nib.load(mri_fname)
    mri_data = mri_file.get_data()
    viewer = OrthoSlicer3D(mri_data, mri_file.affine)
    print('open slice viewer')
    viewer._set_position(x, y, z)
    viewer.show()
    print('after show')


if __name__ == '__main__':
    import sys
    print(sys.argv)
    plt.close("all")
    if len(sys.argv) > 2:
        mri_fname = sys.argv[1]
        x, y, z = map(float, sys.argv[2:5])
        print(x, y, z)
        open_slicer(mri_fname, x, y, z)
    else:
        print('Error! Should send the mri fname')
