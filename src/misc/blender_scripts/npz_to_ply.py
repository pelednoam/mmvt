import numpy as np
import os.path as op
import os
import glob
import re

from src.utils import utils

fol = '/home/npeled/Angelique/Niles'
tar_fol = ' '
npz_files = glob.glob(op.join(fol, 'npzs_center', '*.npz'))
utils.make_dir(op.join(fol, 'plys'))
electrodes_coords, electrodes_names = [], []
for npz_fname in npz_files:
    d = np.load(npz_fname)
    vertices, faces = d['vertices'], d['faces']
    ply_fname = op.join(fol, 'plys', '{}.ply'.format(utils.namebase(npz_fname)))
    if op.isfile(ply_fname):
        os.remove(ply_fname)
    if faces.ndim == 1:
        print('{} is an electrode'.format(ply_fname))
        name = utils.namebase(npz_fname).replace('Electrode', '')
        num = re.sub('\D', ',', name).split(',')[-1]
        if not '.' in name and num == '':
            continue
        electrodes_coords.append(np.mean(vertices, 0))
        electrodes_names.append(name)
    else:
        print('Writing {}'.format(ply_fname))
        utils.write_ply_file(vertices, faces, ply_fname, False)
np.savez(op.join(fol, 'electrodes.npz'), pos=electrodes_coords, names=electrodes_names)