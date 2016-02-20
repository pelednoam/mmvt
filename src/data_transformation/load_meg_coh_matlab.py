import scipy.io as sio
import os.path as op
import numpy as np
import matplotlib.pyplot as plt

from src import utils
# from src import preproc_for_blender

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
BLENDER_ROOT_DIR = op.join(LINKS_DIR, 'mmvt')

TATIANA_ROOT = '/autofs/space/franklin_003/users/npeled/glassy_brain'

def load_tatiana_data():
    d = sio.loadmat(op.join(TATIANA_ROOT, 'coh.m'))
    # labels is a cell array, so convert in to a list
    d['labels'] = [str(l[0][0]) for l in d['labels']]
    d['hemis'] = [str(l[0][0]) for l in d['hemis']]
    d['conditions'] = [str(l[0][0]) for l in d['conditions']]
    np.savez(op.join(TATIANA_ROOT, 'coh'), **d)


def load_meg_coh_python_data():
    d = utils.Bag(np.load(op.join(TATIANA_ROOT, 'coh.npz')))
    # 'coh_colors', 'locations', 'labels', 'conditions', 'data', 'hemis'
    print(d)


if __name__ == '__main__':
    subject = 'mg78'
    # load_tatiana_data()
    # load_meg_coh_python_data()
    print('finish!')