import numpy as np
import os.path as op
from src.utils import utils


def read_packet(packet_fname):
    packet = utils.load(packet_fname)
    print("asdf")


if __name__ == '__main__':
    root = '/homes/5/npeled/space1/Rina'
    packet_fname = op.join(root, 'rina_packet.pkl')
    read_packet(packet_fname)