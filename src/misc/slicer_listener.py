from multiprocessing.connection import Listener
import traceback
import os.path as op
import shutil
import nibabel as nib
import functools

from src.preproc import anatomy as anat
from src.utils.preproc_utils import MMVT_DIR, SUBJECTS_DIR
from src.utils import utils


@functools.lru_cache(maxsize=None)
def get_data_and_header(subject, modality):
    print('Loading header and data for {}, {}'.format(subject, modality))
    if modality == 'mri':
        fname = op.join(MMVT_DIR, subject, 'freeview', 'T1.mgz')
        if not op.isfile(fname):
            subjects_fname = op.join(SUBJECTS_DIR, subject, 'mri', 'T1.mgz')
            if op.isfile(subjects_fname):
                shutil.copy(subjects_fname, fname)
            else:
                print("Can't find subject's T1.mgz!")
                return False
    else:
        print('create_slices: The modality {} is not supported!')
        return False
    header = nib.load(fname)
    data = header.get_data()
    return header, data


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
                msg = self.conn.recv()
                print(msg)
                # do something with msg
                if msg == 'close\n':
                    self.conn.close()
                    break
                else:
                    if isinstance(msg, dict):
                        msg = utils.Bag(msg['data'])
                        header, data = get_data_and_header(msg.subject, msg.modality)
                        anat.create_slices(msg.subject, msg.xyz, msg.modality, header, data)
            except:
                pass
                # print(traceback.format_exc())
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