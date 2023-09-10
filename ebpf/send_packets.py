import argparse
import socket
import random
import time
#######################################################
# run function with command line input as follows:
# python3 send_packets.py "h-2 h-5 h-27" 10 --I enp6s0f1
# to send 10 pps to the hosts h2, h5 and h27 over interface enp6s0f1
# see definitions and additional arguments at the bottom of this file
#######################################################

IP = '10.0.0.12'

def create_hosts_tors_pods(k: int, max_num_pods: int):
    dests=[]
    tors=[]
    pods=[]
    iter1=[]
    iter2=[]
    it = min(k, max_num_pods)
    for a in range(it):
        for b in range(int(k/2)):
            for c in range(2,int(k/2+2)):
                ip = "10." + str(a) + "." + str(b) + "." + str(c)
                dests.append(ip)
                iter1.append(ip)
                iter2.append(ip)
            tors.append(' '.join(iter1))
            iter1.clear()
        pods.append(' '.join(iter2))
        iter2.clear()
    return dests, tors, pods

def send_single_packet(dest_ip: str, dest_port: int, interface: str):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, str(interface+'\0').encode('utf-8'))
    dst = (dest_ip, dest_port)
    ba = bytearray()
    for i in range(100): 
        ba.append(random.randint(0, 255))
    sock.sendto(ba, dst) 
""
def send_pps(comms: str, pps: int, sp: str, I: str, k: int, max_num_pods: int):
    dests = comms.split()
    hosts, tors, pods = create_hosts_tors_pods(k,max_num_pods)
    if(sp=="poisson"):
        print("Sending on average " +str(pps)+ " packets per second to " +comms+ " over interface " +I)
    else: # deterministic
        print("Sending " +str(pps)+ " packets per second to " +comms+ " over interface " +I)
    while(True):
        for y in dests:
            x=y.split("-")
            type = x[0]
            number = int(x[1])
            if(type=="h"):#
                h = hosts[number]
                send_single_packet(h,5007,I)
            elif(type=="tor"):
                hosts_tor = tors[number].split()
                for h in hosts_tor:
                    send_single_packet(h,5007,I)
            elif(type=="pod"):
                hosts_pod = pods[number].split()
                for h in hosts_pod:
                    send_single_packet(h,5007,I)
        if(sp=="poisson"):
            time.sleep(random.expovariate(pps)) # poisson arrival time
        else: # deterministic
            time.sleep(1/pps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        dest="comms",
        type=str,
        help="Hosts, that should be communicated with. Input as follows: \"h-1 h-3 h-5 h-7\", \"tor-0 tor-3\" or \"pod-0\""
    )
    parser.add_argument(
        dest="pps",
        type=int,
        help="Number of packets to be send per second."
    )
    parser.add_argument(
        dest="send_process",
        type=str,
        help="Poisson or deterministic send process."
    )
    parser.add_argument(
        '--I',
        type=str,
        default='enp1s0f1',
        nargs="?",
        help="Interface, which is connected to the mininet server."
    )
    parser.add_argument(
        '--k',
        type=int,
        default=8,
        help="Degree of the Fat-Tree, must be >= 4. Default is 8."
    )
    parser.add_argument(
        '--max_num_pods',
        type=int,
        nargs="?",
        default=1000,
        help="Maximum number of pods. If set to a value smaller than k, only max_num_pods pods are added to the Fat-Tree."
    )
    parsed_args, _ = parser.parse_known_args()
    try:
        send_pps(
            comms=parsed_args.comms,
            pps=parsed_args.pps,
            sp=parsed_args.send_process,
            I=parsed_args.I,
            k=parsed_args.k,
            max_num_pods=parsed_args.max_num_pods
        )
    except KeyboardInterrupt:
        print("") # to start the new line in the terminal after "^C"
        exit()