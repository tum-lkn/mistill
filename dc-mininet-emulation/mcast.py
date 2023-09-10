import socket
import ipaddress
import struct

import netifaces
import sys
import time
import argparse


def tor_ip(pod: int, tor: int, prefix=10, *args, **kwargs) -> str:
    return f'{prefix}.{pod}.{tor}.1'


def agg_ip(pod: int, agg: int, k: int, prefix=10) -> str:
    return f'{prefix}.{pod}.{agg + int(k / 2)}.1'


def core_ip(core_idx: int, k: int, prefix=10) -> str:
    connected_pod_sw_num, core_num_in_grp = divmod(core_idx, int(k / 2))
    return f'{prefix}.{k}.{connected_pod_sw_num + 1}.{core_num_in_grp + 1}'


def all_mcast_ips(k=4):
    ips = []
    for pod in range(k):
        for switch in range(int(k / 2)):
            ips.append(tor_ip(pod, switch, prefix=239))
            ips.append(agg_ip(pod, switch, k, prefix=239))
    for idx in range(int(k**2 / 4)):
        ips.append(core_ip(idx, k, prefix=239))
    return ips


def receiver(MCAST_GRP, MCAST_PORT, host_ip):
    # MCAST_GRP = '224.1.1.1'
    # MCAST_PORT = 5007
    IS_ALL_GROUPS = MCAST_GRP == 'all'
    # IP = '10.0.0.2'
    IP = host_ip
    print(f"Start listening for mcast group {MCAST_GRP} on Port {MCAST_PORT} - Own IP is {IP}.")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if IS_ALL_GROUPS:
        # on this port, receives ALL multicast groups
        sock.bind(('', MCAST_PORT))
        MCAST_GRP = all_mcast_ips()
    else:
        # on this port, listen ONLY to MCAST_GRP
        MCAST_GRP = [MCAST_GRP]
        sock.bind((MCAST_GRP, MCAST_PORT))

    for grp in MCAST_GRP:
        mreq = socket.inet_aton(grp) + socket.inet_aton(IP)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    while True:
        # For Python 3, change next line to "print(sock.recv(10240))"
        bt = sock.recv(10240)
        val = struct.unpack('<Q', bt)[0]
        for i in range(1):
            for j in range(64):
                print((val >> j) & 1, end='')
                print(1 - ((val >> j) & 1), end=" ")
        print()
        for j in range(64):
            print((val >> j) & 1, end=' ')
        print()
        print(val)
        #print("received: ", struct.unpack('>Q', bt))


def sender(MCAST_GRP, MCAST_PORT, host_ip):
    # MCAST_GRP = '224.1.1.1'
    # MCAST_PORT = 5007
    IP = host_ip
    print(f"Start sending to group {MCAST_GRP} to Port {MCAST_PORT} - own IP {IP}")
    # regarding socket.IP_MULTICAST_TTL
    # ---------------------------------
    # for all packets sent, after two hops on the network the packet will not
    # be re-sent/broadcast (see https://www.tldp.org/HOWTO/Multicast-HOWTO-6.html)
    MULTICAST_TTL = 10

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    # sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, MULTICAST_TTL)

    mreq = socket.inet_aton(MCAST_GRP) + socket.inet_aton(IP)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    # For Python 3, change next line to 'sock.sendto(b"robot", ...' to avoid the
    # "bytes-like object i encode(encoding: Text=..., errors: Text=...) -> bytes
    while True:
        # sock.sendto(f"Msg from {host_ip}".encode('ascii'), (MCAST_GRP, MCAST_PORT))
        sock.sendto(f"12345678".encode('ascii'), (MCAST_GRP, MCAST_PORT))
        time.sleep(0.5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        help="Mode program is started in, must be in {sender, receiver}.",
        type=str
    )
    parser.add_argument(
        'groupip',
        help="IP Address of the multicast group to which the program should send or receive from.",
        type=str
    )
    parser.add_argument(
        'port',
        help="Port on which the receiver listens and to which the sender sends.",
        type=int
    )
    parser.add_argument(
        "hostip",
        help="IP address of the interface the host should send or receiver multicast messages from/to."
    )
    # route add -host 224.1.1.1 dev h2-eth0
    parsed_args, _ = parser.parse_known_args()
    print(parsed_args)
    {
        'sender': sender,
        'receiver': receiver
    }[parsed_args.mode](parsed_args.groupip, parsed_args.port, parsed_args.hostip)
