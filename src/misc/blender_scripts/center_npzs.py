import glob
import os.path as op
import os
import numpy as np
from src.utils import utils

fol = '/home/npeled/Angelique/Niles'
npz_files = glob.glob(op.join(fol, 'objects', '*.npz'))
utils.make_dir(op.join(fol, 'npzs_center'))
all_vertices= []
for npz_fname in npz_files:
    d = np.load(npz_fname)
    vertices, faces = d['vertices'], d['faces']
    all_vertices = vertices if all_vertices == [] else np.vstack((all_vertices, vertices))
avg = np.mean(all_vertices, 0)
for npz_fname in npz_files:
    new_npz_fname = op.join(fol, 'npzs_center', '{}.npz'.format(utils.namebase(npz_fname)))
    if op.isfile(new_npz_fname):
        os.remove(new_npz_fname)
    d = np.load(npz_fname)
    vertices, faces = d['vertices'], d['faces']
    vertices -= avg
    np.savez(new_npz_fname, vertices=vertices, faces=faces)

