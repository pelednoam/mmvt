import socket
import select
import sys
import traceback


def stdout_print(str):
    sys.stdout.write(str)
    sys.stdout.write('\n')
    sys.stdout.flush()


def start_listener():
    stdout_print('Start listening')
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_address = ('localhost', 10000)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(server_address)

        while True:
            next_val = sock.recv(1024)
            next_val = next_val.decode(sys.getfilesystemencoding(), 'ignore')
            stdout_print(next_val)
    except:
        print(traceback.format_exc())


if __name__ == '__main__':
    start_listener()