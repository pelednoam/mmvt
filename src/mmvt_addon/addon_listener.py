# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing.connection import Listener
from multiprocessing import Process
from subprocess import Popen, PIPE, STDOUT
from threading import Thread
import subprocess
import os.path as op
import os

import traceback
import numpy as np

def check_if_open():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = s.connect_ex(('localhost', 6000))

    if result == 0:
        print('socket is open')
    s.close()


# def open_slicer(mri):
#     import subprocess
#     import os.path as op
#     import os
#     print('open_slicer!')
#     current_path = os.path.dirname(os.path.realpath(__file__))
#     x, y, z = mri['position']
#     cmd = 'python {}'.format(op.join(current_path, 'slicer.py {} {} {} {}'.format(
#         mri['mri_fname'], x, y, z)))
#     subprocess.call(cmd, shell=True)
#     # self.p = Popen(cmd, shell=True, stdout=PIPE, stdin=PIPE, stderr=PIPE)


# http://stackoverflow.com/a/6921402/1060738
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
                        if msg['cmd'] == 'plot':
                            self.plot_data(msg['data'])
                        if msg['cmd'] == 'open_slice_viewer':
                            self.open_slice_viewer(msg['data'])
                        if msg['cmd'] == 'slice_viewer_change_pos':
                            self.slice_viewer_change_pos(msg['data'])
                    # print(msg)
            except:
                pass
                # print(traceback.format_exc())
        print('Stop listening!')
        self.listener.close()

    def plot_data(self, data):
        x, y = data['x'], data['y']
        plt.plot(x, y)
        plt.show()

    def open_slice_viewer(self, mri):
        self.mri = mri
        try:
            from nibabel.viewers import OrthoSlicer3D
            viewer_exist = True
        except:
            viewer_exist = False
            print('You need to install the latest nibabel dev version')

        if not viewer_exist:
            return

        self._open_slicer('load')

    def _open_slicer(self, func):
        print('open_slicer!')
        current_path = os.path.dirname(os.path.realpath(__file__))
        x, y, z = self.mri['position']
        cmd = 'python {}'.format(op.join(current_path, 'slicer.py {} {} {} {} {}'.format(
            self.mri['mri_fname'], x, y, z, func)))
        p = Popen(cmd, stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
        try:
            self.p.kill()
        except:
            pass
        self.p = p

    def slice_viewer_change_pos(self, data):
        print('Goto position {}'.format(data['position']))
        # self.thread.stop()
        self.mri['position'] = data['position']
        # print(self.mri)
        self._open_slicer('update')



def main():
    listener = AddonListener(6000, b'mmvt')
    listener.listen()


if __name__ == '__main__':
    import sys
    sys.stdout.write('In addon_listener!')
    sys.stdout.flush()
    # print()
    main()