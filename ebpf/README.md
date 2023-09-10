run ip route show and check whether the subspace of the other host is visible. If not, run add_routes.py.
Use "sudo tshark -i enp1s0f1 -Y "!(icmpv6)"" to monitor packet flows. Change device to enp6s0f1 for the beast.

Run mininet with following command in folder topos: "sudo HOME=~ python3 nwx.py /home/sim/dc-mininet-emulation 8 --max_num_pods 2 --eth enp3s0"
And clean up with: "sudo mn --clean"
For the test on the hazard, run: "sudo HOME=~ python3 nwx.py /home/sim/dc-mininet-emulation 8 --eth enp4s0f0"

change rights for folders with jsons, that are created with root privileges: "sudo chmod 777 -c -R <folder>"

## Requirements

* A recent kernel. Tested with 5.13 and 5.15
* iproute2. Version 5.9 on Server 1 and 5.15 on the beast.
  Installation is not needed, the `tc` binary is enough.
* Clang eBPF backend required. Clang version 13 on Server 1 and 14 on the beast.
* libbpf - tested with 0.4.0 on Server 1 and 0.8.0 on the beast
* LLVM - needed?
* Cmake - version 3.18.4 on Server 1 and 3.24.0 on the beast
* Torch - CPU version or GPU version for CUDA 11.3 (11.5 is on the beast)
* Python 3.9
* Pandas

# OLD INFO:

Run sysctl -w kernel.bpf_stats_enabled=1 after a system reboot for bpftool to work.
Check after reboot if interfaces are defined as they should be and execute netplan apply in the console.
Sending pings for testing from server 1 via: ping -I enp1s0f1 10.2.1.2

# Combination of eBPF, MPLS and NNs in user and kernel space for a routing scenario

The code is executed every time the system sends an IP-packet. MPLS labels are added from a table which is kept up to date with the help of a Neural Network and update messages by the network.

Code was tested on Pop!_OS 21.04 with self-compiled iproute2 v5.9.0, libbpf 0.3.0 and clang 12.0.0.
Attention: libbpf needs at least Ubuntu (or Pop!) 21.04!
