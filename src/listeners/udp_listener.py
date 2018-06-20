import socket
import sys
import numpy as np
import traceback
from datetime import datetime
import matplotlib.pyplot as plt

from src.utils import utils
from src.utils import args_utils as au

SERVER = '127.0.0.1'  # '172.17.146.219' # socket.gethostbyname(socket.gethostname())
PORT = 59154
MULTICAST_GROUP = '127.0.0.1' # '239.255.43.21'

def stdout_print(str):
    sys.stdout.write(str)
    sys.stdout.write('\n')
    sys.stdout.flush()


def start_udp_listener(buffer_size=10, multicast=True):
    stdout_print('UDP listener: Start listening')
    udp_listening = True

    try:
        buffer = []
        if multicast:
            sock = bind_to_multicast(PORT, MULTICAST_GROUP)
        else:
            sock = bind_to_server(SERVER, PORT)
        while udp_listening:
            # time.sleep(0.0001)
            if multicast:
                next_val, address = sock.recvfrom(2048)
            else:
                next_val = sock.recv(2048)
            next_val = next_val.decode(sys.getfilesystemencoding(), 'ignore')
            buffer.append(next_val)
            if len(buffer) >= buffer_size:
                # stdout_print(str(datetime.now() - time))
                # time = datetime.now()
                stdout_print(','.join(buffer))
                buffer = []
    except:
        print(traceback.format_exc())


def bind_to_server(server, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = (server, port)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(server_address)
    return sock


def bind_to_multicast(port=45454, multicast_group='239.255.43.21'):
    import struct
    # Create the socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Bind to the server address
    # sock.bind((multicast_group, port))
    sock.bind(('', port))
    # Tell the operating system to add the socket to the multicast group
    # on all interfaces.
    # group = socket.inet_aton(multicast_group)
    # mreq = struct.pack('4sL', group, socket.INADDR_ANY)
    mreq = struct.pack('4sl', socket.inet_aton(multicast_group), socket.INADDR_ANY)

    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    return sock


def start_udp_listener_timeout(buffer_size=10, multicast=True):
    stdout_print('UDP listener: Start listening')
    udp_listening = True

    try:
        if multicast:
            sock = bind_to_multicast(PORT, MULTICAST_GROUP)
        else:
            sock = bind_to_server(SERVER, PORT)
        buffer = []
        errs_num = 0
        errs_total, extra_times, messages_time = [], [], []
        time = datetime.now()

        while udp_listening:
            try:
                # sock.settimeout(0.0012)
                sock.settimeout(0.1)
                if multicast:
                    next_val, address = sock.recvfrom(2048)
                else:
                    next_val = sock.recv(2048)
            except socket.timeout as e:
                errs_num += 1
                err = e.args[0]
                if err == 'timed out':
                    print('timed out')
                    next_val = np.zeros((101, 1))
                else:
                    raise Exception(e)
            else:
                messages_time.append((datetime.now() - time).microseconds / 1000)
                time = datetime.now()
                next_val = next_val.decode(sys.getfilesystemencoding(), 'ignore')
                next_val = np.array([to_float(f) for f in next_val.split(',')])
                next_val = next_val[..., np.newaxis]
            buffer = next_val if buffer == [] else np.hstack((buffer, next_val))
            if buffer.shape[1] >= buffer_size:
                # print('{}/{} packets with errors'.format(errs_num, buffer_size))
                errs_total.append(errs_num)
                errs_num = 0
                # diff = (datetime.now() - time)
                # extra_times.append((diff.microseconds - 10000) / 1000)
                # stdout_print(str(datetime.now() - time))

                # stdout_print(','.join(buffer))
                buffer = []
    except:
        print(traceback.format_exc())

    # import matplotlib.pyplot as plt
    # plt.hist(errs_total)


def to_float(x_str):
    try:
        return float(x_str)
    except ValueError:
        return 0.0


def listen_raw():
    # http://stackoverflow.com/questions/1117958/how-do-i-use-raw-socket-in-python
    HOST = socket.gethostbyname(socket.gethostname())
    s = socket.socket(socket.AF_INET, socket.SOCK_RAW)
    # s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # s = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_IP)
    s.bind((HOST, PORT))
    # s.ioctl(socket.SIO_RCVALL, socket.RCVALL_ON)
    while True:
        x = s.recvfrom(2048)
        print(x)
    s.ioctl(socket.SIO_RCVALL, socket.RCVALL_OFF)


# def main(args):
#     cmd = '{} -m src.udp.udp_listener -f start_udp_listener -b {}'.format(args.python_cmd, args.buffer_size)
#     out_queue, in_queue = mu.run_command_in_new_thread(
#         cmd, read_stderr=False, read_stdin=False, stdout_func=reading_from_rendering_stdout_func)
#
#
#     while True:
#         stdout_print('listening to stdin!')
#         line = sys.stdin.read()
#         if line != '':
#             stdout_print('UDP listener: received "{}"'.format(line))
#             if line == 'stop':
#                 stdout_print('Stop listening')
#                 udp_listening = False
#
#     # except:
#     #     print(traceback.format_exc())
#
#
def read_cmd_args(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='UDP listener')
    parser.add_argument('-b', '--buffer_size', required=False, default=10, type=int)
#     parser.add_argument('-p', '--python_cmd', required=False, default='python')
#     parser.add_argument('-f', '--function', required=False, default='')
    return utils.Bag(au.parse_parser(parser, argv))
#

if __name__ == '__main__':
    args = read_cmd_args()
    # if args.function == 'start_udp_listener':
    #     start_udp_listener(args.buffer_size)
    # else:
    #     main(args)
    # import time
    # sys.stdin.write('stop')
    # time.sleep(3)
    start_udp_listener_timeout(args.buffer_size, multicast=False)
    # listen_raw()
