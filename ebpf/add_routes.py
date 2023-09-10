import sys
from subprocess import Popen
import os
import signal
from time import sleep

def run(cmd):
    return Popen(cmd,shell=True,preexec_fn=os.setsid)

def kill(pro):
    os.killpg(os.getpgid(pro.pid), signal.SIGTERM)

mode = sys.argv[1] # add or del
interface = sys.argv[2] # enp6s0f1 on beast
# on beast: if all pods are built on hazard, then every route except for the four hosts at tor0 has to be added
# so additionally a=1 and for a=0 b is in range(1,4)
for a in range(2,8): # first two pods are clear
    for b in range(0,4):
        for c in range(2,6):
            add_r = run("ip route " + mode + " 10." + str(a) + "." + str(b) + "." + str(c) + " via 10.0.0.1 dev " + interface)
            if(mode == "add"):
                print("Added: 10." + str(a) + "." + str(b) + "." + str(c))
            if(mode == "del"):
                print("Deleted: 10." + str(a) + "." + str(b) + "." + str(c))
            add_r.wait()