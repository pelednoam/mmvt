import socket
import time
import os.path as op
import numpy as np
from itertools import cycle

from src.utils import utils

LINKS_DIR = utils.get_links_dir()
MMVT_DIR = op.join(LINKS_DIR, 'mmvt')


def load_electrodes_data(bipolar=False):
    source_fname = op.join(MMVT_DIR, 'mg78', 'electrodes', 'electrodes{}_data_diff.npz'.format(
        '_bipolar' if bipolar else ''))
    f = np.load(source_fname)
    ind = np.where(f['names']=='LMF6')[0][0]
    cond_ind = np.where(f['conditions']=='interference')[0]
    data = f['data'][ind, :, cond_ind].squeeze()
    return cycle(data)


def bind_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = ('localhost', 10000)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(server_address)
    return sock, server_address


def send_data(sock, server_address, data, interval=0.01): #0.01):
    while True:
        data_to_send = next(data)
        data_to_send = str(data_to_send).encode('utf-8')
        sock.sendto(data_to_send, server_address)
        # print('{}: sending data {}'.format(utils.get_time(), data_to_send))
        time.sleep(interval)


if __name__ == '__main__':
    data = load_electrodes_data()
    sock, server_address = bind_socket()
    send_data(sock, server_address, data)