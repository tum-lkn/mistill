import time
import json
import os
import sys

# Wait for 1ms. Decrease waiting time a bit to account for missing processing and the inaccuricies in the time.sleep() call. Do not use busy wait to reduce the load on the CPU.
FACTOR = 0.9999
DELTA = 0.001 * FACTOR

def read_file(filename):
    with open(filename,"r") as fh:
        content = fh.readline().strip()
    return content

interface_names = [f"tor0-eth{i:d}" for i in range(5,9)]
with open("/home/zeidler/dc-mininet-emulation/stat_info/test_" + sys.argv[1] + ".json","w") as fh: # argv[1] is the name of the test run
    # line = {"Data":[
    fh.write('{"Data":[' + os.linesep)
    while True:
        try:
            # ts_start = time.time()
            line = {name: {
                "ts": time.time(),
                "operstate": read_file(f"/sys/class/net/{name}/operstate"),
                "tx_bytes": int(read_file(f"/sys/class/net/{name}/statistics/tx_bytes")),
                "rx_bytes": int(read_file(f"/sys/class/net/{name}/statistics/rx_bytes"))
                } for name in interface_names}
            fh.write(json.dumps(line) + "," + os.linesep)
            # delta = time.time() - ts_start
            # if delta >= DELTA:
            #     # raise ValueError(f"Collecting data takes too long, takes {delta}ms instead of {DELTA}ms")
            #     pass
            #     # print("Collecting data takes too long, takes "+ str(delta) +"s instead of "+ str(DELTA) +"ms")
            # else:
            #     time.sleep(0.001-delta) # collect data every 1 ms
            time.sleep(0.0006)
        except KeyboardInterrupt: # does not happen, since process is killed and not terminated by ^C
            line = {name: {
                "ts": time.time(),
                "operstate": read_file(f"/sys/class/net/{name}/operstate"),
                "tx_bytes": int(read_file(f"/sys/class/net/{name}/statistics/tx_bytes")),
                "rx_bytes": int(read_file(f"/sys/class/net/{name}/statistics/rx_bytes"))
                } for name in interface_names}
            fh.write(json.dumps(line) + os.linesep)
            fh.write(json.dumps(line) + ']}' + os.linesep)
            print("exit")
            break
