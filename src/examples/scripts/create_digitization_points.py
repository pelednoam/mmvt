import numpy as np
import os.path as op
import glob
from mne.transforms import _get_trans, apply_trans


def run(mmvt):
    dig_fname = op.join(mmvt.utils.get_user_fol(), 'meg', 'digitization_points.npz')
    if not op.isfile(dig_fname):
        print("Can't find the digitization points file! ({})".format(dig_fname))
        print("Run src.preproc.meg -s {} -f get_digitization_points".format(mmvt.utils.get_user()))
        return

    trans_format = op.join(mmvt.utils.get_user_fol(), 'trans', '*-trans.fif')
    head_mri_trans_files = glob.glob(trans_format)
    if len(head_mri_trans_files) == 0:
        print("Can't find any trans file! ({})".format(trans_format))
        return

    trans = None
    for trans_fname in head_mri_trans_files:
        try:
            trans = _get_trans(trans_fname, fro='head', to='mri')[0]
        except:
            pass
    if trans is None:
        print("None of the trans files was a head-mri transformation!")
        return

    dig = np.load(dig_fname)
    parnet_name = 'Deep_electrodes'
    mmvt.data.create_empty_if_doesnt_exists(parnet_name, parent_obj_name=parnet_name)
    layers_array = [False] * 20
    layers_array[mmvt.ELECTRODES_LAYER] = True

    for point, kind, ident in zip(dig['pos'], dig['kind'], dig['ident']):
        x, y, z = apply_trans(trans['trans'], point) * 1e2
        mmvt.data.create_electrode(x, y, z, '{}_{}'.format(ident, kind), 0.15, layers_array, parnet_name)
