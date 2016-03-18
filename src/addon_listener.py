from multiprocessing.connection import Listener
import matplotlib.pyplot as plt

# http://stackoverflow.com/a/6921402/1060738
def init_listener():
    address = ('localhost', 6000)
    listener = Listener(address, authkey=b'mmvt')
    conn = listener.accept()
    print('connection accepted from', listener.last_accepted)
    return listener, conn


def listen(listener, conn):
    while True:
        msg = conn.recv()
        # do something with msg
        if msg == 'close':
            conn.close()
            break
        else:
            if isinstance(msg, dict):
                if msg['cmd'] == 'plot':
                    plot_data(msg['data'])
            print(msg)
    listener.close()


def plot_data(data):
    x, y = data
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    listener, conn = init_listener()
    listen(listener, conn)