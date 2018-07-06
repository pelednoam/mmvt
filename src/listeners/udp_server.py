import socket
import time
import os.path as op
import numpy as np
from itertools import cycle

from src.utils import utils

LINKS_DIR = utils.get_links_dir()
MMVT_DIR = op.join(LINKS_DIR, 'mmvt')


def load_electrodes_data(bipolar=False, electrode_name='LMF6', condition='interference'):
    source_fname = op.join(MMVT_DIR, 'mg78', 'electrodes', 'electrodes{}_data_diff.npz'.format(
        '_bipolar' if bipolar else ''))
    f = np.load(source_fname)
    if electrode_name == 'all':
        ind = np.arange(0, len(f['names']))
    else:
        ind = np.where(f['names']==electrode_name)[0][0]
    cond_ind = np.where(f['conditions']==condition)[0]
    data = f['data'][ind, :, cond_ind].squeeze().T
    return cycle(data)


def bind_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = ('localhost', 10000)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(server_address)
    return sock, server_address


def bind_to_multicast():
    import struct
    # Create the datagram socket
    # sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    # Set a timeout so the socket does not block indefinitely when trying
    # to receive data.
    # sock.settimeout(0.2)
    # Set the time-to-live for messages to 1 so they do not go past the
    # local network segment.
    # ttl = struct.pack('b', 1)
    # sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
    return sock


def broadcast_data(sock, multicast_group='239.255.43.21', port=45454, interval=0.0006):
    import time
    ind = 0
    times = []
    now = time.time()
    while True:
        ind += 1
        if ind % 1000 == 0:
            print(np.mean(times))
            times = []
        data_to_send = next(data)
        # data_to_send = str(data_to_send).encode('utf-8')
        data_to_send = ','.join([str(d) for d in data_to_send]).encode('utf-8')
        sent = sock.sendto(data_to_send, (multicast_group, port))
        # print('{}: sending data {}'.format(utils.get_time(), data_to_send))
        times.append(time.time() - now)
        now = time.time()

        time.sleep(interval)


def send_data(sock, server_address, data, interval=0.0006):
    import time
    ind = 0
    times = []
    now = time.time()
    while True:
        ind += 1
        if ind % 1000 == 0:
            print(np.mean(times))
            times = []
        data_to_send = next(data)
        # data_to_send = str(data_to_send).encode('utf-8')
        data_to_send = ','.join([str(d) for d in data_to_send]).encode('utf-8')
        sock.sendto(data_to_send, server_address)
        # print('{}: sending data {}'.format(utils.get_time(), data_to_send))
        times.append(time.time() - now)
        now = time.time()

        time.sleep(interval)


if __name__ == '__main__':
    data = load_electrodes_data(False, 'all')
    # sock, server_address = bind_socket()
    # send_data(sock, server_address, data)

    port = 45454
    multicast_group = '239.255.43.21'
    sock = bind_to_multicast()
    # broadcast_data(sock, multicast_group, port)
    send_data(sock, port, data)