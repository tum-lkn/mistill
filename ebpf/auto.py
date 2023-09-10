import sys
from subprocess import Popen
import os
import argparse
import signal
from time import sleep
#######################################################
# run function with command line input as follows:
# sudo python3 auto.py 10 1
# to send 10 pps and call the test run "test_run_1"
#######################################################

def run(cmd):
    return Popen(cmd,shell=True,preexec_fn=os.setsid)

def kill(pro):
    os.killpg(os.getpgid(pro.pid), signal.SIGTERM)

def execute_test(pps: str, name_int: str, tc: str, interface: str):
    send_s = "python3 send_packets.py \""
    send_e = "\" " + pps + " --I " + interface + " poisson"
    utr_s = "./read/utr.o "
    ktr_s = "./read/ktr.o "
    test_run = "test_run_" + name_int + "/"

    pre_read = run("mkdir test_data/" + test_run) # one time at the beginning instead of in every iteration
    if tc == "regular":
        print("REGULAR TEST CASE")
        try:
            for i in range(0,128): # 128 # 100 is only a test
                mpls = run("make all")
                sleep(1) # in order to execute the programs with a slight time difference
                user = run("cd build && make user.o && ./user.o")
                sleep(1) # in order to execute the programs with a slight time difference
                send = run(send_s + " ".join(["h-"+str(j) for j in range(0, i+1)]) + send_e) # original
                # send = run(send_s + " ".join(["h-"+str(j) for j in range(0, 128)]) + send_e) # only a test!!
                sleep(180) # 180s = 3 min
                pre_read = run("make track_time")
                sleep(1)
                read = run(utr_s + test_run + " part_" + str(i))
                read2 = run(ktr_s + test_run + " part_" + str(i))
                kill(send)
                kill(user)
                mpls = run("make all_del time_del")
                mpls.wait()
                print("test --- " + str(i) + " --- test")
        except KeyboardInterrupt:
            kill(send)
            kill(user)
            mpls = run("make all_del time_del")
            print("exit \n")
            return
    else: # link up down test case
        print("LINK FAILURE TEST CASE")
        send_e = "\" " + pps + " --I " + interface + " deterministic"
        try:
            for i in range(0,5): # 5*5 minutes = 25 min -- in 5 files to prevent eBPF Maps from overflowing
                mpls = run("make all")
                sleep(1) # in order to execute the programs with a slight time difference
                user = run("cd build && make user.o && ./user.o")
                sleep(1) # in order to execute the programs with a slight time difference
                send = run(send_s + "h-16" + send_e) # link up down test case: only send packets to host h16
                sleep(300) # 300s = 5 min
                pre_read = run("make track_time")
                sleep(1)
                read = run(utr_s + test_run + " part_" + str(i))
                read2 = run(ktr_s + test_run + " part_" + str(i))
                kill(send)
                kill(user)
                mpls = run("make all_del time_del")
                mpls.wait()
                print("test --- " + str(i) + " --- test")
        except KeyboardInterrupt:
            kill(send)
            kill(user)
            mpls = run("make all_del time_del")
            print("exit \n")
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        dest="pps",
        type=str, # since it is only used to call other functions via cmd
        help="Number of packets to be send per second."
    )
    parser.add_argument(
        dest="test_name",
        type=str,
        help="Name under which the current test should be saved."
    )
    parser.add_argument(
        "--tcase",
        type=str,
        default='regular',
        nargs="?",
        help="Regular or link up down test case. The regular test has a poisson send process. Otherwise the send process is deterministic."
    )
    parser.add_argument(
        '--I',
        type=str,
        default='enp6s0f1',
        nargs="?",
        help="Interface, which is connected to the mininet server."
    )
    parsed_args, _ = parser.parse_known_args()
    execute_test(
        pps=parsed_args.pps,
        name_int=parsed_args.test_name,
        tc=parsed_args.tcase,
        interface=parsed_args.I,
    )