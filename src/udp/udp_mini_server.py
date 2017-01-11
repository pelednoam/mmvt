import socket
import time

MCAST_GRP = '224.1.1.1'
MCAST_PORT = 5007

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 32)
# sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)

while True:
    sock.sendto(b'robot', (MCAST_GRP, MCAST_PORT))
    print('robot')
    time.sleep(1)