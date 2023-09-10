# DEVICE=enp1s0f1 # server 1
DEVICE=enp6s0f1 # BEAST

help:
	@echo "Combination of eBPF, MPLS and NNs in user and kernel space for a routing scenario"
	@echo
	@echo "later"

all:
	clang -O2 -target bpf -I/usr/include/x86_64-linux-gnu -c -g mpls.c -o mpls.o
	sudo tc qdisc add dev $(DEVICE) clsact
	sudo tc filter add dev $(DEVICE) egress bpf direct-action obj mpls.o sec egress	
	sudo tc filter show dev $(DEVICE) egress
	sudo tc filter add dev $(DEVICE) ingress bpf direct-action obj mpls.o sec ingress
	sudo tc filter show dev $(DEVICE) ingress
	sudo tc exec bpf dbg

all_del:
	sudo tc filter delete dev $(DEVICE) egress
	sudo tc filter delete dev $(DEVICE) ingress
	rm mpls.o
	sudo tc qdisc delete dev $(DEVICE) clsact
	sudo rm /sys/fs/bpf/tc/globals/network_state
	sudo rm /sys/fs/bpf/tc/globals/flow_table
	sudo rm /sys/fs/bpf/tc/globals/nn_labels
	sudo rm /sys/fs/bpf/tc/globals/dest_IPs

del:
	sudo tc filter delete dev $(DEVICE) egress
	sudo tc filter delete dev $(DEVICE) ingress
	rm mpls.o
	sudo tc qdisc delete dev $(DEVICE) clsact

debug:
	gcc -o read/rnl.o read/read_nn_labels.cpp 
	gcc -o read/rft.o read/read_flw_tbl.cpp 
	gcc -o read/rdi.o read/read_dest_ips.cpp 
	gcc -o read/rns.o read/read_net_stt.cpp 

track_time:
	gcc -o read/ktr.o read/kernel_time_read.cpp
	gcc -o read/utr.o read/user_time_read.cpp

track_time_old:
	gcc -o read/rti.o read/read_t_ingress.cpp
	gcc -o read/rte.o read/read_t_egress.cpp
	gcc -o read/rtit.o read/read_t_iteration.cpp
	gcc -o read/rtnn.o read/read_t_nn_exe.cpp
	gcc -o read/rtnh.o read/read_t_nn_handling.cpp

time_del:
	sudo rm /sys/fs/bpf/tc/globals/t_kernel
	sudo rm /sys/fs/bpf/tc/globals/t_user

time_del_old:
	sudo rm /sys/fs/bpf/tc/globals/t_ingress
	sudo rm /sys/fs/bpf/tc/globals/t_egress
	sudo rm /sys/fs/bpf/tc/globals/t_iteration
	sudo rm /sys/fs/bpf/tc/globals/t_nn_exe
	sudo rm /sys/fs/bpf/tc/globals/t_nn_handling

net:
	ip route add dev enp1s0f1 10.2.1.2
	ip route add 10.0.0.2 via 10.2.1.2 dev enp1s0f1
	ip route add 10.0.0.4 via 10.2.1.2 dev enp1s0f1
	ip route add 10.0.0.5 via 10.2.1.2 dev enp1s0f1
	ip route add 10.0.1.0/24 via 10.2.1.2 dev enp1s0f1
	ip route add 10.0.2.0/24 via 10.2.1.2 dev enp1s0f1
	ip route add 10.0.3.0/24 via 10.2.1.2 dev enp1s0f1
	ip route add 10.1.0.0/16 via 10.2.1.2 dev enp1s0f1
	ip route add 10.2.0.0/16 via 10.2.1.2 dev enp1s0f1
	ip route add 10.3.0.0/16 via 10.2.1.2 dev enp1s0f1
	ip route add 10.4.0.0/16 via 10.2.1.2 dev enp1s0f1
	ip route add 10.5.0.0/16 via 10.2.1.2 dev enp1s0f1
	ip route add 10.6.0.0/16 via 10.2.1.2 dev enp1s0f1
	ip route add 10.7.0.0/16 via 10.2.1.2 dev enp1s0f1
	ip route add dev enp1s0f1 228.0.0.0/8