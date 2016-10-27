# Only vanilla python here (being used by intall_addon which is called from setup)

import os
import os.path as op
import subprocess
from sys import platform as _platform

IS_WINDOWS = _platform == "win32"


def make_dir(fol):
    if not os.path.isdir(fol):
        os.makedirs(fol)
    return fol


def namebase(file_name):
    return op.splitext(op.basename(file_name))[0]


def get_current_fol():
    return op.dirname(op.realpath(__file__))


def get_parent_fol(curr_dir='', levels=1):
    if curr_dir == '':
        curr_dir = get_current_fol()
    parent_fol = op.split(curr_dir)[0]
    for _ in range(levels - 1):
        parent_fol = get_parent_fol(parent_fol)
    return parent_fol


def run_script(cmd, verbose=False):
    if verbose:
        print('running: {}'.format(cmd))
    if IS_WINDOWS:
        output = subprocess.call(cmd)
    else:
        output = subprocess.check_output('{} | tee /dev/stderr'.format(cmd), shell=True)

    print(output)
    return output
