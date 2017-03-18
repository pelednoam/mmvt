import os.path as op
from surfer import Brain
from surfer.io import read_stc
from mayavi import mlab
import nibabel as nib
import numpy as np

time_point = 1225
time = [time_point]
hemi = 'lh'
colormap = 'hot'

subject_id, surface = 'mg78', 'pial'
brain = Brain(subject_id, hemi, surface, size=(800, 400))

plot_stc = False
plot_labels = True

if (plot_stc):
    root = '/home/npeled/code/links/meg/MSIT/ep001'
    stc1_fname = op.join(root, 'ep001_msit_nTSSS_interference_neutral_1-15-dSPM-lh.stc')
    stc2_fname = op.join(root, 'ep001_msit_nTSSS_interference_interference_1-15-dSPM-lh.stc')
    stc1 = read_stc(stc1_fname)
    stc2 = read_stc(stc2_fname)

    data1 = stc1['data'][:, time_point]
    data2 = stc2['data'][:, time_point]
    data = data1 - data2
    print(data.shape)
    vertices = stc1['vertices']

    brain.add_data(data, colormap=colormap, vertices=vertices,
                   smoothing_steps=10, time=time,
                   hemi=hemi)
    mlab.show()

if (plot_labels):
    aparc_file = '/home/npeled/subjects/mg78/label/lh.laus250.annot'
    labels, ctab, names = nib.freesurfer.read_annot(aparc_file)
    d = np.load('/home/npeled/mmvt/mg78/labels_data_laus250_lh.npz')
    data = np.diff(d['data'][:, time]).squeeze()
    vtx_data = data[labels]
    brain.add_data(vtx_data, colormap=colormap, alpha=.8)
    mlab.show()

