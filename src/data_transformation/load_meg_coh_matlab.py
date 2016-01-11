import scipy.io as sio
import os.path as op
import numpy as np

ROOT = '/autofs/space/franklin_003/users/npeled/glassy_brain'

def load_data():
    d = sio.loadmat(op.join(ROOT, 'coh.m'))
    d['labels'] = [str(l[0][0]) for l in d['labels']]
    np.savez(op.join(ROOT, 'coh'), **d)

def load_python_data():
    d = np.load(op.join(ROOT, 'coh.npz'))
    print(d)

if __name__ == '__main__':
    load_data()
    load_python_data()