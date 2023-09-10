from subprocess import Popen
import os
import signal
import time
import sys
import random
import argparse

def run(cmd):
    return Popen(cmd,shell=True,preexec_fn=os.setsid)

def kill(pro):
    os.killpg(os.getpgid(pro.pid), signal.SIGTERM)

def execute_test(test_run: str):
    try:
        read = run("python3 read_stats.py " + test_run)
        for i in range(5):
            for j in range (10): # 10 for a five minute run
                time.sleep(5)
                data = {}
                data["down"] = time.time()
                down = run("sudo ip link set tor0-eth5 down") # test: default eth5
                print("Link down")
                # time.sleep(15)
                time.sleep(random.uniform(14,16))
                data["up"] = time.time()
                up = run("sudo ip link set tor0-eth5 up")# test: default eth5
                print("Link up")
                time.sleep(10)
            time.sleep(3)
            print("iteration " + str(i) + " done")
        kill(read)
        return
    except KeyboardInterrupt:
        kill(read)
        return

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        dest="test_name",
        type=str,
        help="Name under which the current test should be saved."
    )
    parsed_args, _ = parser.parse_known_args()
    execute_test(
        test_run=parsed_args.test_name,
    )