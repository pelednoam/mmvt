from multiprocessing.connection import Listener
import traceback
import os.path as op
import shutil
import nibabel as nib
import functools
import numpy as np

from src.preproc import anatomy as anat
from src.utils.preproc_utils import MMVT_DIR, SUBJECTS_DIR
from src.utils import utils


@functools.lru_cache(maxsize=None)
def get_data_and_header(subject, modality):
    # print('Loading header and data for {}, {}'.format(subject, modality))
    if modality == 'mri':
        fname = op.join(MMVT_DIR, subject, 'freeview', 'T1.mgz')
        if not op.isfile(fname):
            subjects_fname = op.join(SUBJECTS_DIR, subject, 'mri', 'T1.mgz')
            if op.isfile(subjects_fname):
                shutil.copy(subjects_fname, fname)
            else:
                print("Can't find subject's T1.mgz!")
                return False
    elif modality == 'ct':
        fname = op.join(MMVT_DIR, subject, 'freeview', 'ct.mgz')
        if not op.isfile(fname):
            subjects_fname = op.join(SUBJECTS_DIR, subject, 'mri', 'ct.mgz')
            if op.isfile(subjects_fname):
                shutil.copy(subjects_fname, fname)
            else:
                print("Can't find subject's CT! ({})".format(fname))
                return False
    else:
        print('create_slices: The modality {} is not supported!')
        return False
    header = nib.load(fname)
    data = header.get_data()
    percentiles_fname = op.join(MMVT_DIR, subject, 'freeview', '{}_1_99_precentiles.npy'.format(modality))
    if not op.isfile(percentiles_fname):
        precentiles = np.percentile(data, (1, 99))
        np.save(percentiles_fname, precentiles)
    return header, data


@functools.lru_cache(maxsize=None)
def get_modality_trans_file(subject, modality):
    if modality == 'mri':
        trans_fname = op.join(MMVT_DIR, subject, 'orig_trans.npz')
        if not op.isfile(trans_fname):
            anat.save_subject_orig_trans(subject)
    elif modality == 'ct':
        trans_fname = op.join(MMVT_DIR, subject, 'ct_trans.npz')
        if not op.isfile(trans_fname):
            anat.save_subject_ct_trans(subject)
    else:
        print('The modality {} is not supported!'.format(modality))
        return None
    trans = np.load(trans_fname)
    return utils.Bag(trans)


def convert_coordinates(subject, xyz, modality, coordinates_system):
    point = np.array(list(map(float, xyz.split(','))))
    trans = get_modality_trans_file(subject, modality)
    if trans is None:
        return None
    if coordinates_system == 'tk_ras':
        vox = utils.apply_trans(trans.ras_tkr2vox, point).astype(np.int)
        print('T1: {}'.format(vox))
    elif coordinates_system == 'vox':
        vox = point
    return vox


class AddonListener(object):

    listener = None
    conn = None

    def __init__(self, port, authkey):
        try:
            # check_if_open()
            address = ('localhost', port)
            print('addon_listener: trying to listen to localhost, {}'.format(port))
            self.listener = Listener(address, authkey=authkey)
            self.conn = self.listener.accept()
            print('connection accepted from', self.listener.last_accepted)
        except:
            print('Error in init_listener')
            print(traceback.format_exc())

    def listen(self):
        while True:
            try:
                if self.conn is None:
                    raise Exception('self.conn is None! bye bye!')
                msg = self.conn.recv()
                if msg == 'close\n':
                    self.conn.close()
                    break
                else:
                    if isinstance(msg, dict):
                        msg = utils.Bag(msg['data'])
                        # print(msg)
                        for modality in msg.modalities.split(','):
                            xyz = convert_coordinates(msg.subject, msg.xyz, modality, msg.coordinates_system)
                            if xyz is None:
                                continue
                            header, data = get_data_and_header(msg.subject, modality)
                            anat.create_slices(msg.subject, xyz, modality, header, data)
                        output_fol = op.join(MMVT_DIR, msg.subject, 'figures', 'slices')
                        modalities = '_'.join(msg.modalities.split(','))
                        with open(op.join(output_fol, '{}_slices.txt'.format(modalities)), 'w') as f:
                            f.write('Slices created for {}'.format(xyz))

            except EOFError:
                print('Bye bye!')
                self.listener.close()
                break
            except:
                # pass
                print(traceback.format_exc())
        print('Stop listening!')
        self.listener.close()


def main():
    listener = AddonListener(6000, b'mmvt')
    listener.listen()


if __name__ == '__main__':
    import sys
    sys.stdout.write('In addon_listener!\n')
    sys.stdout.flush()
    main()