from multiprocessing.connection import Listener
import matplotlib.pyplot as plt
import traceback


def check_if_open():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = s.connect_ex(('localhost', 6000))

    if result == 0:
        print('socket is open')
    s.close()


# http://stackoverflow.com/a/6921402/1060738
def init_listener():
    try:
        check_if_open()
        address = ('localhost', 6000)
        listener = Listener(address, authkey=b'mmvt')
        conn = listener.accept()
        print('connection accepted from', listener.last_accepted)
        return listener, conn
    except:
        # print(traceback.format_exc())
        print('Error in init_listener')
        return None, None


def listen(listener, conn):
    while True:
        try:
            msg = conn.recv()
            # do something with msg
            if msg == 'close\n':
                conn.close()
                break
            else:
                if isinstance(msg, dict):
                    if msg['cmd'] == 'plot':
                        data = msg['data']
                        x, y = data['x'], data['y']
                        plot_data(x, y)
                print(msg)
        except:
            pass
            # print(traceback.format_exc())
    print('Stop listening!')
    listener.close()


def plot_data(x, y):
    plt.plot(x, y)
    plt.show()


def main():
    listener, conn = init_listener()
    listen(listener, conn)


if __name__ == '__main__':
    main()