# Fat-tree topo
This project implements a Fat-Tree in Mininet. The program configures the
Mininet switches to forward ARP and IP traffic using ECMP, i.e., all hosts
in the network can reach each other. Further, the program configures MPLS rules
for segment routing. If a MPLS label is present, the packet is forwarded to the
corresponding port and the label is popped.

The available labels are `{100, 101, ..., 100 + k - 1}`.
The label `100` identifies the zeroth port on the switch, the label `101` the
first port, etc.

Further, each switch in the topology as an additional host connected to it on
port `k + 1`. This host emulates the switche's local control plane. The naming
scheme is `<switch-name>Ctrl`. The task
of the host is to use the NN from the `traced-hlsa-module.pbt` torch script
to create HNSAs and send them out via multi cast. The program also configures the
Mininet switches with the necessary multi-cast forwarding rules.

To create the HNSAs, each control host retrieves the current state of the
switch. The host then executes the NN to produce the HNSA. The HNSA is
serialized into a long, and the long is send over the network in a multi-cast
message.

The host retrieves the switch state via a SystemV message queue. The queue is
filled via an external process. This process retrieves the state from the
filesystem in `/sys/class/net/<interface-name>/`. This process thus controls how
often and how fast updates can be send by the control plane switches. Using
this external process is necessary, since the Mininet hosts cannot access
these files.

# Running the program
Run the program with:

```
sudo HOME=~ python3 nwx.py <workdir> <k>
```
to spin up a k-Fat-Tree. The argument <workdir> points to an **existing** directory.
Actually, its the path to this repository, including the `dc-mininet-emulation` part.
The program will write log-files and other output files into this directory.
Before starting make sure the file `cpp/traced-hlsa-module.pbt` exists. Also,
build the `makeHnsa` executable by running
```
cd cpp && cmake build . && make
```
This builds the C++ executable that executes the NN that reads the switch
stats, computes the HNSA messages, and sends the Multicast update messages.

With the command:
```
sudo HOME=~ python3 nwx.py <workdir> <k> --max_num_pods 2 --eth eth0
```
the program builds a k-Fat-Tree with a limited number of pods, in this example
with only two pods. The topology will have all core switches, but only two pods
with hosts, tors and aggs. Further, the program will connect the physical interface
`eth0` as the `k+2`th port on the first tor in the first pod (i.e., `tor0`).
This allows the communication of an external host with the mininet topology.
The external host will receive all multi-cast messages that other hosts connected
to that ToR would receives as well. Further, the host can communicate with the
other hosts in the network. The external host must have the IP `10.0.0.k+4`
configured on the interface it is connected with.

# General workflow
```
sudo HOME=~ python3 nwx.py /home/mininet 8 --max_num_pods 2 --eth eth0
```
starts the Mininet topology and opens the mininet CLI. The `HOME=~` thing
allows X11 forwarding for xterms (for this, connect with `ssh -X` to the machine).
External host is connected with `10.0.0.12`

Once the exepriment of whatever is finished, enter `exit` in the Mininet CMD.
The program will quit and clean up behind itself. Finally, run

```
sudo mn --clean
```
to clean up mininet. If somthing goes wrong, or just to check that everything
is well, use the commands:
```
ipcs -q
sudo ipcrm msg <msqid>
ps -e | grep makeHnsa
rm <workdir>/producer-quit
```
The first command shows open SystemV message queues. It should be empty. If a
queue exists, remove it with the second command. The third command checks
for the external process. Ideally, the command should come up empty. If it returns
something, kill the process.

The last command removes a signalling file. If this file is present, the producer will
exist. The producer should remove the file itself, but sometimes he does not.
