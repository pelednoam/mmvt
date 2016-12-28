import socket
import sys
import traceback
from datetime import datetime

from src.utils import utils
from src.utils import args_utils as au

def stdout_print(str):
    sys.stdout.write(str)
    sys.stdout.write('\n')
    sys.stdout.flush()


def start_udp_listener(buffer_size=10):
    stdout_print('UDP listener: Start listening')
    udp_listening = True

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_address = ('localhost', 10000)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(server_address)
        buffer = []

        while udp_listening:
            # time.sleep(0.0001)
            next_val = sock.recv(1024)
            next_val = next_val.decode(sys.getfilesystemencoding(), 'ignore')
            buffer.append(next_val)
            if len(buffer) >= buffer_size:
                # stdout_print(str(datetime.now() - time))
                # time = datetime.now()
                stdout_print(','.join(buffer))
                buffer = []
    except:
        print(traceback.format_exc())


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
    start_udp_listener(args.buffer_size)
