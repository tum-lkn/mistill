import time
from typing import List
import os
import sys
from mininet.topo import Topo
from mininet.net import Mininet, Intf
from mininet.util import dumpNodeConnections
from mininet.log import setLogLevel
from mininet.cli import CLI
from functools import partial
from mininet.node import OVSSwitch
import subprocess
import argparse

BW = 100  # Link bandwidth in Mbit/s
START_TOR_ONLY = False


def get_logdir(work_dir) -> str:
    # pwd = os.environ['HOME']
    # logdir = os.path.join(pwd, 'dc-emulation-logs')
    logdir = os.path.join(work_dir, 'dc-emulation-logs')
    if not os.path.exists(logdir):
        print("Make logdir ", logdir)
        os.mkdir(logdir)
    return logdir


def hostname(pod: int, tor: int, host: int, k: int) -> str:
    #return f'h-{pod:d}-{tor:d}-{host:d}'
    return f'h{int(k**2 / 4 * pod + int(k / 2) * tor + host)}'


def torname(pod: int, tor: int, k: int) -> str:
    # return f'tor-{pod:d}-{tor:d}'
    return f'tor{pod * int(k / 2) + tor}'


def aggname(pod: int, agg: int, k: int) -> str:
    # return f'agg-{pod:d}-{agg:d}'
    return f'agg{pod * int(k / 2) + agg}'


def corename(num: int) -> str:
    # return f'core-{num:d}'
    return f'core{num:d}'


def ctrl_name(switch_name: str) -> str:
    return f'{switch_name}Ctrl'


def hostip(pod: int, tor: int, host: int) -> str:
    return f'10.{pod:d}.{tor:d}.{host + 2:d}'


def tor_ip(pod: int, tor: int, prefix=10, *args, **kwargs) -> str:
    return f'{prefix}.{pod}.{tor}.1'


def agg_ip(pod: int, agg: int, k: int, prefix=10) -> str:
    return f'{prefix}.{pod}.{agg + int(k / 2)}.1'


def core_ip(core_idx: int, k: int, prefix=10) -> str:
    connected_pod_sw_num, core_num_in_grp = divmod(core_idx, int(k / 2))
    return f'{prefix}.{k}.{connected_pod_sw_num + 1}.{core_num_in_grp + 1}'


def print_ovs_cmd(name: str, cmd: str, args: str):
    print(f'sudo ovs-ofctl {cmd} {name} {args}')


class FatTree(Topo):

    ecmp_upstream_gid = 3
    mc_downstream_gid = 4
    mc_up_down_gid = 5
    mc_prefix = 239
    mpls_start = 101

    def add_hosts(self, k: int):
        for pod in range(min(k, self.max_num_pods)):
            for tor in range(int(k / 2)):
                for host in range(int(k / 2)):
                    self.addHost(hostname(pod, tor, host, k), ip=hostip(pod, tor, host))

    def add_host_ips(self, k: int):
        for pod in range(min(k, self.max_num_pods)):
            for tor in range(int(k / 2)):
                for host in range(int(k / 2)):
                    name = f'h-{pod}-{tor}-{host}'
                    ip = f'10.{pod}.{tor}.{host + 1}'
                    self.g.node[name].setIP(ip=ip, prefixLen=24, intf=0)

    def add_pod_switch(self, k: int, name_fct: callable, ip_fct: callable):
        for pod in range(min(k, self.max_num_pods)):
            for switch in range(int(k / 2)):
                self.addSwitch(name_fct(pod, switch, k))
                self.add_ctrl_host(name_fct(pod, switch, k), ip_fct(pod, switch, k=k, prefix=10))

    def add_tors(self, k: int):
        self.add_pod_switch(k, torname, tor_ip)

    def add_aggs(self, k: int):
        self.add_pod_switch(k, aggname, agg_ip)

    def add_cores(self, k: int):
        for core in range(int(k**2 / 4)):
            self.addSwitch(corename(core))
            self.add_ctrl_host(corename(core), core_ip(core, k))

    def add_ctrl_host(self, switch_name: str, switch_ip: str):
        self.addHost(ctrl_name(switch_name), ip=switch_ip)

    def add_ctrl_host_link(self, switch_name: str):
        self.addLink(ctrl_name(switch_name), switch_name, bw=BW)

    def add_links_host_tor(self, k: int):
        for pod in range(min(k, self.max_num_pods)):
            for tor in range(int(k / 2)):
                tor_name = torname(pod, tor, k)
                for host in range(int(k / 2)):
                    host_name = hostname(pod, tor, host, k)
                    self.addLink(host_name, tor_name, bw=BW)

    def add_links_tor_agg(self, k: int):
        for pod in range(min(k, self.max_num_pods)):
            for tor in range(int(k / 2)):
                tor_name = torname(pod, tor, k)
                for agg in range(int(k / 2)):
                    agg_name = aggname(pod, agg, k)
                    self.addLink(tor_name, agg_name, bw=BW)

    def add_links_core_aggregation(self, k: int):
        for pod in range(min(k, self.max_num_pods)):
            for agg in range(int(k / 2)):
                agg_name = aggname(pod, agg, k)
                for idx in range(int(k / 2)):
                    core_idx = agg * int(k / 2) + idx
                    core_name = corename(core_idx)
                    print(f'self.addLink({agg_name}, {core_name}, port1={int(k / 2) + idx}, port2={pod}')
                    self.addLink(agg_name, core_name, bw=BW)

    def add_control_plane_links(self, k):
        for pod in range(min(k, self.max_num_pods)):
            for switch in range(int(k / 2)):
                self.add_ctrl_host_link(torname(pod, switch, k))
                self.add_ctrl_host_link(aggname(pod, switch, k))
        for core in range(int(k**2 / 4)):
            self.add_ctrl_host_link(corename(core))

    def add_host_rules(self, k: int):
        for pod in range(min(k, self.max_num_pods)):
            for tor in range(int(k / 2)):
                for host in range(int(k / 2)):
                    name = hostname(pod, tor, host, k)
                    ip = f'10.{pod}.{tor}.{host + 1}'
                    self.g.node[name].cmd(f'ip route add 10.0.0.0/8 via {ip} dev {name}-eth0')

    def add_ecmp_upstream_group(self, name: str, k: int):
        cmd = (
            name,
            'add-group',
            f'group_id={self.ecmp_upstream_gid},type=select,selection_method=hash,'
            f'selection_method_param=0,{",".join([f"bucket=output:{port + 1}" for port in range(int(k / 2), k)])}'
        )
        print_ovs_cmd(*cmd)
        self.ovs_ofctl_cmds.append(cmd)

    def _add_mc_downstream_group(self, name: str, ports: List[int]):
        """
        Add a group to the switch that replicates multicast messages downstream.
        """
        cmd = (
            name,
            'add-group',
            f'group_id={self.mc_downstream_gid},type=all,{",".join([f"bucket=output:{port}" for port in ports])}'
        )
        print_ovs_cmd(*cmd)
        self.ovs_ofctl_cmds.append(cmd)

    def _add_mc_up_down_group(self, name: str, k: int):
        """
        Add a meta-group with two buckets. One bucket is responsible to forward
        a multicast packet with ECMP upstream. The other bucket is responsible
        for forwarding and replicating a packet downstream.
        """
        ports = [p for p in range(1, int(k / 2 + 1))]
        if name == 'tor0':
            ports.append(k + 2)
        cmd = (
            name,
            'add-group',
            f'group_id={self.mc_up_down_gid},type=all,' +
            ','.join([f'bucket=output:{p}' for p in ports]) +
            f',bucket=group:{self.ecmp_upstream_gid}' # Has to be last one, processing stops after that.
        )
        print_ovs_cmd(*cmd)
        self.ovs_ofctl_cmds.append(cmd)

    def add_mc_downstream_group_pod_switches(self, name: str, k: int):
        if name == 'tor0':
            ports = list(range(1, int(k / 2 + 1)))
            ports.append(k + 2)
            self._add_mc_downstream_group(name, ports)
        else:
            self._add_mc_downstream_group(name, list(range(1, int(k / 2 + 1))))

    def add_mc_downstream_group_core_switches(self, name: str, k: int):
        num_pods = min(k, self.max_num_pods)
        ports = list(range(1, num_pods + 1))
        self._add_mc_downstream_group(name, ports)

    def tor_mc_traffic(self, pod: int, tor: int, k: int):
        # The ToR sends its own multicast messages upstream to an Aggregation
        # switch. The aggregation switch duplicates the messages across all ToR
        # switches, including the ToR it received the message from. Then,
        # the ToR will also duplicate the messages to the hosts. Use this
        # approach to avoid sending the same message two times to the same host
        # and keeping the rule base small.
        name = torname(pod, tor, k)
        # Send every IP packet with a multicast destination address per ECMP
        # upstream. All other traffic from this interface is handled via the
        # other IP rules.
        cmd = (
            name,
            'add-flow',
            f'dl_type=0x0800,in_port={k + 1},ip_dst={self.mc_prefix}.{pod}.{tor}.0/24,'
            f'priority=40002,actions=group:{self.mc_up_down_gid}'
        )
        print_ovs_cmd(*cmd)
        self.ovs_ofctl_cmds.append(cmd)

        # Replicate all multicast messages to downstream nodes. Rule will apply
        # to multicast messages the ToR receives from the aggregation switches.
        # All other IP traffic is forwarded with the normal IP rules.
        cmd = (
            name,
            'add-flow',
            f'dl_type=0x0800,ip_dst={self.mc_prefix}.0.0.0/8,priority=40001,'
            f'actions=group:{self.mc_downstream_gid}'
        )
        print_ovs_cmd(*cmd)
        self.ovs_ofctl_cmds.append(cmd)

    def agg_mc_traffic(self, pod: int, agg: int, k: int):
        name = aggname(pod, agg, k)
        # Match all Multicast IP traffic and send it to the group that will
        # duplicate messages downstream and forward via ECMP upstream.
        cmd = (
            name,
            'add-flow',
            f'dl_type=0x0800,ip_dst={self.mc_prefix}.0.0.0/8,priority=40002,'
            f'actions=group:{self.mc_up_down_gid}'
        )
        print_ovs_cmd(*cmd)
        self.ovs_ofctl_cmds.append(cmd)

    def core_mc_traffic(self, core_idx: int):
        name = corename(core_idx)
        # Match all Multicast IP traffic and send it to the group that will
        # duplicate messages downstream and forward via ECMP upstream.
        cmd = (
            name,
            'add-flow',
            f'dl_type=0x0800,ip_dst={self.mc_prefix}.0.0.0/8,priority=40002,'
            f'actions=group:{self.mc_downstream_gid}'
        )
        print_ovs_cmd(*cmd)
        self.ovs_ofctl_cmds.append(cmd)

    def _tor_mininet_ip_traffic(self, tor: int, pod: int, k: int):
        name = torname(pod, tor, k)
        # Add a forwarding rule for each host connected to the ToR switch.
        # Forward based on the IP address.
        for port in range(int(k / 2)):
            cmd = (
                name,
                'add-flow',
                f'dl_type=0x0800,ip_dst={hostip(pod, tor, port)},priority=40000,actions=output:{port + 1}'
            )
            print_ovs_cmd(*cmd)
            self.ovs_ofctl_cmds.append(cmd)
            # Send IP packets for the aggregation switches to the correct one.
            cmd = (
                name,
                'add-flow',
                f'dl_type=0x0800,ip_dst={agg_ip(pod, port, k)},priority=40000,actions=output:{int(k / 2) + port + 1}'
            )
            print_ovs_cmd(*cmd)
            self.ovs_ofctl_cmds.append(cmd)
            # Add rule for core switches.
            cmd = (
                name,
                'add-flow',
                f'dl_type=0x0800,ip_dst=10.{k}.{port + 1}.0/24,priority=40000,actions=output:{int(k / 2) + port + 1}'
            )
            print_ovs_cmd(*cmd)
            self.ovs_ofctl_cmds.append(cmd)
        if tor == 0 and pod == 0:
            # Add rules for the external host, i.e., the first ToR has k/2 + 1 hosts.
            cmd = (
                name,
                'add-flow',
                f'dl_type=0x0800,ip_dst={hostip(pod, tor, k + 2)},priority=40000,actions=output:{int(k + 2)}'
            )
            print_ovs_cmd(*cmd)
            self.ovs_ofctl_cmds.append(cmd)
        # Add a rule that catches all IP traffic and sends it to group 3, i.e.,
        # balances the packets over the aggregation switches.
        cmd = (
            name,
            'add-flow',
            f'dl_type=0x0800,priority=35000,actions=group:3'
        )
        print_ovs_cmd(*cmd)
        self.ovs_ofctl_cmds.append(cmd)
        # Add a rule for the control plane host.
        cmd = (
            name,
            'add-flow',
            f'dl_type=0x0800,ip_dst={tor_ip(pod, tor)},priority=40000,actions=output:{k + 1}'
        )
        print_ovs_cmd(*cmd)
        self.ovs_ofctl_cmds.append(cmd)

    def _tor_mininet_arp_traffic(self, tor: int, pod: int, k: int):
        name = torname(pod, tor, k)
        # Add a forwarding rule for each host connected to the ToR switch.
        # Forward based on the IP address.
        for port in range(int(k / 2)):
            cmd = (
                name,
                'add-flow',
                f'dl_type=0x0806,arp_tpa={hostip(pod, tor, port)},priority=40000,actions=output:{port + 1}'
            )
            print_ovs_cmd(*cmd)
            self.ovs_ofctl_cmds.append(cmd)
            # Send ARP replies for the aggregation switches to the correct one.
            cmd = (
                name,
                'add-flow',
                f'dl_type=0x0806,arp_tpa={agg_ip(pod, port, k)},priority=40000,actions=output:{int(k / 2) + port + 1}'
            )
            print_ovs_cmd(*cmd)
            self.ovs_ofctl_cmds.append(cmd)
            # Add rule for core switches.
            cmd = (
                name,
                'add-flow',
                f'dl_type=0x0806,arp_tpa=10.{k}.{port + 1}.0/24,priority=40000,actions=output:{int(k / 2) + port + 1}'
            )
            print_ovs_cmd(*cmd)
            self.ovs_ofctl_cmds.append(cmd)
        if tor == 0 and pod == 0:
            # Add rules for the external host, i.e., the first ToR has k/2 + 1 hosts.
            cmd = (
                name,
                'add-flow',
                f'dl_type=0x0806,arp_tpa={hostip(pod, tor, k + 2)},priority=40000,actions=output:{int(k + 2)}'
            )
            print_ovs_cmd(*cmd)
            self.ovs_ofctl_cmds.append(cmd)
        # Add a rule that catches all IP traffic and sends it to group 3, i.e.,
        # balances the packets over the aggregation switches.
        cmd = (
            name,
            'add-flow',
            f'dl_type=0x0806,priority=39999,actions=group:3'
        )
        print_ovs_cmd(*cmd)
        self.ovs_ofctl_cmds.append(cmd)
        # Add a rule for the control plane host.
        cmd = (
            name,
            'add-flow',
            f'dl_type=0x0806,arp_tpa={tor_ip(pod, tor)},priority=40000,actions=output:{k + 1}'
        )
        print_ovs_cmd(*cmd)
        self.ovs_ofctl_cmds.append(cmd)

    def _agg_mininet_ip_traffic(self, agg: int, pod: int, k: int):
        name = aggname(pod, agg, k)
        # Add a forwarding rule for each ToR connected to the Agg switch.
        # Forward based on the IP address.
        for port in range(int(k / 2)):
            cmd = (
                name,
                'add-flow',
                f'dl_type=0x0800,ip_dst=10.{pod}.{port}.0/24,priority=40000,actions=output:{port + 1}'
            )
            print_ovs_cmd(*cmd)
            self.ovs_ofctl_cmds.append(cmd)
            # Add rule for connected core switches.
            cmd = (
                name,
                'add-flow',
                f'dl_type=0x0800,ip_dst=10.{k}.{agg + 1}.{port + 1},priority=40000,actions=output:{int(k / 2) + port + 1}'
            )
            print_ovs_cmd(*cmd)
            self.ovs_ofctl_cmds.append(cmd)
        # Rule catches IP traffic for this pod that the previous rules did
        # not match. Rule checks if IP traffic is for this pod. If so, the packet is
        # for the aggregation control plane. Else it would have been captured
        # by the previous rule.
        cmd = (
            name,
            'add-flow',
            f'dl_type=0x0800,ip_dst=10.{pod}.0.0/16,priority=39999,actions=output:1'
        )
        print_ovs_cmd(*cmd)
        self.ovs_ofctl_cmds.append(cmd)
        # Add a rule that catches all IP traffic and sends it to group 3, i.e.,
        # balances the packets over the core switches.
        cmd = (
            name,
            'add-flow',
            f'dl_type=0x0800,priority=39998,actions=group:3'
        )
        print_ovs_cmd(*cmd)
        self.ovs_ofctl_cmds.append(cmd)
        # Add a forwarding rule for the control plane host.
        cmd = (
            name,
            'add-flow',
            f'dl_type=0x0800,ip_dst={agg_ip(pod, agg, k)},priority=40001,actions=output:{k + 1}'
        )
        print_ovs_cmd(*cmd)
        self.ovs_ofctl_cmds.append(cmd)
        # Default rule for all other core switches.
        cmd = (
            name,
            'add-flow',
            f'dl_type=0x0800,ip_dst=10.{k}.0.0/16,priority=39999,actions=output:1'
        )
        print_ovs_cmd(*cmd)
        self.ovs_ofctl_cmds.append(cmd)

    def _agg_mininet_arp_traffic(self, agg: int, pod: int, k: int):
        name = aggname(pod, agg, k)
        # Add a forwarding rule for each ToR connected to the Agg switch.
        # Forward based on the IP address.
        for port in range(int(k / 2)):
            cmd = (
                name,
                'add-flow',
                f'dl_type=0x0806,arp_tpa=10.{pod}.{port}.0/24,priority=40000,actions=output:{port + 1}'
            )
            print_ovs_cmd(*cmd)
            self.ovs_ofctl_cmds.append(cmd)
            # Add rule for core switches.
            cmd = (
                name,
                'add-flow',
                f'dl_type=0x0806,arp_tpa=10.{k}.{agg + 1}.{port + 1},priority=40000,actions=output:{int(k / 2) + port + 1}'
            )
            print_ovs_cmd(*cmd)
            self.ovs_ofctl_cmds.append(cmd)
        # Rule catches ARP traffic for this pod that the previous rules did
        # not match. Rule checks if ARP traffic is for this pod. If so, the packet is
        # for the aggregation control plane. Else it would have been captured
        # by the previous rule.
        cmd = (
            name,
            'add-flow',
            f'dl_type=0x0806,arp_tpa=10.{pod}.0.0/16,priority=39999,actions=output:1'
        )
        print_ovs_cmd(*cmd)
        self.ovs_ofctl_cmds.append(cmd)
        # Add a rule that catches all ARP traffic and sends it to group 3, i.e.,
        # balances the packets over the core switches.
        cmd = (
            name,
            'add-flow',
            f'dl_type=0x0806,priority=39998,actions=group:3'
        )
        print_ovs_cmd(*cmd)
        self.ovs_ofctl_cmds.append(cmd)
        # Add a rule for the control plane host.
        cmd = (
            name,
            'add-flow',
            f'dl_type=0x0806,arp_tpa={agg_ip(pod, agg, k)},priority=40001,actions=output:{k + 1}'
        )
        print_ovs_cmd(*cmd)
        self.ovs_ofctl_cmds.append(cmd)
        # Default rule for all other core switches.
        cmd = (
            name,
            'add-flow',
            f'dl_type=0x0806,arp_tpa=10.{k}.0.0/16,priority=39999,actions=output:1'
        )
        print_ovs_cmd(*cmd)
        self.ovs_ofctl_cmds.append(cmd)

    def _core_mininet_ip_traffic(self, core_idx: int, k: int):
        name = corename(core_idx)
        num_pods = min(k, self.max_num_pods)
        for pod in range(num_pods):
            cmd = (
                name,
                'add-flow',
                f'dl_type=0x0800,ip_dst=10.{pod}.0.0/16,actions=output:{pod + 1}'
            )
            print_ovs_cmd(*cmd)
            self.ovs_ofctl_cmds.append(cmd)
        # Add a forwarding rule for the control plane host.
        cmd = (
            name,
            'add-flow',
            f'dl_type=0x0800,ip_dst={core_ip(core_idx, k)},priority=40001,actions=output:{num_pods + 1}'
        )
        print_ovs_cmd(*cmd)
        self.ovs_ofctl_cmds.append(cmd)

    def _core_mininet_arp_traffic(self, core_idx: int, k: int):
        name = corename(core_idx)
        num_pods = min(k, self.max_num_pods)
        for pod in range(num_pods):
            cmd = (
                name,
                'add-flow',
                f'dl_type=0x0806,arp_tpa=10.{pod}.0.0/16,actions=output:{pod + 1}'
            )
            print_ovs_cmd(*cmd)
            self.ovs_ofctl_cmds.append(cmd)
        # Add a rule for the control plane host.
        cmd = (
            name,
            'add-flow',
            f'dl_type=0x0806,arp_tpa={core_ip(core_idx, k)},priority=40001,actions=output:{num_pods + 1}'
        )
        print_ovs_cmd(*cmd)
        self.ovs_ofctl_cmds.append(cmd)

    def add_mininet_traffic(self, k: int):
        for pod in range(min(k, self.max_num_pods)):
            for sw in range(int(k / 2)):
                name = torname(pod, sw, k)
                print(f'\n# Add IP and ARP rules for switch {name}')
                print('################################################################################')
                self.add_ecmp_upstream_group(name, k)
                self.add_mc_downstream_group_pod_switches(name, k)
                # Add the up/down group to the ToR switch as well. OVS will not
                # send a packet out to the port it received it in. Thus, the ToR
                # has to forward multicast packets to the connected hosts itself.
                self._add_mc_up_down_group(name, k)
                self._tor_mininet_arp_traffic(sw, pod, k)
                self._tor_mininet_ip_traffic(sw, pod, k)
                self.tor_mc_traffic(pod, sw, k)

                name = aggname(pod, sw, k)
                print(f'\n# Add IP and ARP rules for switch {name}')
                print('################################################################################')
                self.add_ecmp_upstream_group(name, k)
                self.add_mc_downstream_group_pod_switches(name, k)
                self._add_mc_up_down_group(name, k)
                # The up/downgroup will send the message to all ToRs except the
                # one that send the message, and forward the message to any
                # connected core switch.
                self._agg_mininet_arp_traffic(sw, pod, k)
                self._agg_mininet_ip_traffic(sw, pod, k)
                self.agg_mc_traffic(pod, sw, k)
        for cidx in range(int(k**2 / 4)):
            print(f'\n# Add IP and ARP rules for switch {corename(cidx)}')
            print('################################################################################')
            self.add_mc_downstream_group_core_switches(corename(cidx), k)
            self._core_mininet_ip_traffic(cidx, k)
            self._core_mininet_arp_traffic(cidx, k)
            self.core_mc_traffic(cidx)

    def add_mpls_rules(self, k: int):
        def make_cmd(n, p, dl_t='0x8847'):
            return (
                n,
                'add-flow',
                f'dl_type=0x8847,mpls_label={self.mpls_start + p},actions=pop_mpls:{dl_t},output:{p + 1}'
            )

        # TODO: Handle ARP packets, too. To do so, use a label range starting at 200 instead of 100
        #       The eBPF program should use a label starting at 200 for the last hop, i.e.,
        #       the forwarding decision of the ToR switches. The other labels for forwarding
        #       decisions on aggregation and core switches can remain as they are.
        #       This process can be repeated for arbitrary protocols, i.e., signal
        #       the protocol type for specific eth types to the ToR switches with
        #       appropriate labels. The ToR switch can then correctly re-establish
        #       the corresponding eth type.
        for pod in range(min(k, self.max_num_pods)):
            for sw in range(int(k / 2)):
                for i, name_fct in enumerate([torname, aggname]):
                    name = name_fct(pod, sw, k)
                    print(f'\n# Add MPLS rules for switch {name}')
                    print('################################################################################')
                    for port in range(k):
                        if i == 0 and port < 4:
                            # Downstream on ToR switches restore IP ETH_TYPE
                            # On all other switches and directions keep the
                            # MPLS unicast.
                            cmd = make_cmd(name, port, dl_t='0x0800')
                        else:
                            cmd = make_cmd(name, port, dl_t='0x8847')
                        self.ovs_ofctl_cmds.append(cmd)
                        print_ovs_cmd(*cmd)
        for cidx in range(int(k**2 / 4)):
            name = corename(cidx)
            print(f'\n# Add MPLS rules for switch {name}')
            print('################################################################################')
            for port in range(k):
                cmd = make_cmd(name, port, '0x0800')
                self.ovs_ofctl_cmds.append(cmd)
                print_ovs_cmd(*cmd)

    def add_mc_rules(self, k: int):
        for pod in range(min(k, self.max_num_pods)):
            for sw in range(int(k / 2)):
                self.tor_mc_traffic(pod, sw, k)
                self.agg_mc_traffic(pod, sw, k)
        for cidx in range(int(k**2 / 4)):
            self.core_mc_traffic(cidx)

    def add_default_routes(self, net: Mininet, k: int):
        def defaultroute(name: str, ip: str):
            cmd = f'\troute add -host {ip} dev {name}-eth0'
            print(cmd)
            net.getNodeByName(name).cmd(cmd)

        for pod in range(min(k, self.max_num_pods)):
            for switch in range(int(k / 2)):
                defaultroute(ctrl_name(torname(pod, switch, k)), tor_ip(pod, switch, prefix=self.mc_prefix))
                defaultroute(ctrl_name(aggname(pod, switch, k)), agg_ip(pod, switch, k, prefix=self.mc_prefix))
        for coreidx in range(int(k**2 / 4)):
            defaultroute(ctrl_name(corename(coreidx)), core_ip(coreidx, k, prefix=self.mc_prefix))

    def start_mcast_sender(self, net: Mininet, k: int, use_avx: bool):
        def start_sender(name: str, sw_name: str, k: int, node_idx: int, ip: str):
            if use_avx:
                progname = "avxconsumer"
            else:
                progname = "consumer"
            cmd = f"\t{exe_path} {progname} " + \
                  f"{k:d} {self.work_dir} {sw_name} {node_idx:d} {ip} > {os.path.join(self.log_dir, f'{name}.log')} &"
            print(cmd)
            net.getNodeByName(name).cmd(cmd)

        exe_path = os.path.join(self.work_dir, 'cpp', 'makeHnsa')
        the_node_idx = 2
        for pod in range(min(k, self.max_num_pods)):
            for tor in range(int(k / 2)):
                if START_TOR_ONLY and pod * k / 2 + tor > 0:
                    break
                sw_name = f'tor{tor + 4*pod}'
                node_name = ctrl_name(torname(pod, tor, k))
                start_sender(node_name, sw_name, k, the_node_idx, tor_ip(pod, tor, self.mc_prefix))
                the_node_idx += 1
            for agg in range(int(k / 2)):
                if START_TOR_ONLY and agg >= 0:
                    break
                sw_name = f'agg{agg + 4*pod}'
                node_name = ctrl_name(aggname(pod, agg, k))
                start_sender(node_name, sw_name, k, the_node_idx, agg_ip(pod, agg, k, self.mc_prefix))
                the_node_idx += 1
        for core in range(int(k**2 / 4)):
            if START_TOR_ONLY and core >= 0:
                break
            sw_name = f"core{core}"
            node_name = ctrl_name(corename(core))
            start_sender(node_name, sw_name, k, the_node_idx, core_ip(core, k, self.mc_prefix))
            the_node_idx += 1
            
    def start_sender(self, net: Mininet, k: int, work_dir: str):
        def start_app(name: str, own_ip: str, group_ip: str):
            net.getNodeByName(name).cmd(f'python3 {os.path.join(work_dir, "mcast.py")} sender '
                                        f'{self.work_dir} {group_ip} 5007 {own_ip} > {os.path.join(self.log_dir, f"{name}.log")} &')
            
        for pod in range(min(k, self.max_num_pods)):
            for switch in range(int(k / 2)):
                start_app(
                    name=ctrl_name(torname(pod, switch, k)),
                    own_ip=tor_ip(pod, switch),
                    group_ip=tor_ip(pod, switch, prefix=self.mc_prefix)
                )
                start_app(
                    name=ctrl_name(aggname(pod, switch, k)),
                    own_ip=agg_ip(pod, switch, k),
                    group_ip=agg_ip(pod, switch, k, prefix=self.mc_prefix)
                )
        for coreidx in range(int(k**2 / 4)):
            start_app(
                name=ctrl_name(corename(coreidx)),
                own_ip=core_ip(coreidx, k),
                group_ip=core_ip(coreidx, k, prefix=self.mc_prefix)
            )

    def start_receiver(self, net: Mininet, k: int, work_dir: str):
        def start_app(name: str, own_ip: str, group_ip: str):
            net.getNodeByName(name).cmd(f'python3 {os.path.join(work_dir, "mcast.py")} receiver '
                                        f'{self.work_dir} {group_ip} 5007 {own_ip} > {os.path.join(self.log_dir, f"{name}.log")}.log &')

        for pod in range(min(k, self.max_num_pods)):
            for tor in range(int(k / 2)):
                for host in range(int(k / 2)):
                    start_app(
                        name=hostname(pod, tor, host, k),
                        own_ip=hostip(pod, tor, host),
                        group_ip='all'
                    )

    def build(self, k: int, log_dir: str, work_dir: str, max_num_pods:int = 100) -> None:
        self.max_num_pods = max_num_pods
        self.ovs_ofctl_cmds = []
        self.log_dir = log_dir
        self.work_dir = work_dir
        self.add_hosts(k)
        self.add_tors(k)
        self.add_aggs(k)
        self.add_cores(k)
        self.add_links_host_tor(k)
        self.add_links_tor_agg(k)
        self.add_links_core_aggregation(k)
        self.add_control_plane_links(k)
        self.add_mininet_traffic(k)
        self.add_mpls_rules(k)
        self.add_mc_rules(k)


def simple_test(work_dir: str, max_num_pods: int, k: int, ext_eth: str, use_avx: bool):
    if work_dir[-1] == '/':
        work_dir = work_dir[:-1]
    log_dir = get_logdir(work_dir)

    quit_p = os.path.join(work_dir, 'producer-quit')
    if os.path.exists(quit_p):
        os.remove(quit_p)

    exe_path = os.path.join(work_dir, 'cpp', 'makeHnsa')
    assert os.path.exists(exe_path), f"The executable {exe_path} does not exist. Did you build it?"
    assert os.path.exists(os.path.join(work_dir, 'cpp', 'traced-hlsa-module.pt')), \
        f"Could not find the torchscript file {os.path.join(work_dir, 'cpp', 'traced-hlsa-module.pt')}"

    time.sleep(4)
    topo = FatTree(k=k, log_dir=log_dir, work_dir=work_dir, max_num_pods=max_num_pods)
    net = Mininet(topo)
    net.start()
    net.waitConnected()
    print("Dumping host connections")
    dumpNodeConnections(net.hosts)

    if ext_eth != '':
        print(f"Add {ext_eth} to node tor0 - use the IP 10.0.0.{k + 2} for a host connected on this port.")
        tor0 = net.getNodeByName('tor0')
        tor0.cmd(f"ovs-vsctl add-port tor0 {ext_eth}")
    # intf = Intf('eth0', node=tor0)

    for name, cmd, args in topo.ovs_ofctl_cmds:
        sw = net.getNodeByName(name)
        # sw.vsctl('set', 'bridge', 'protocols=OpenFlow10,OpenFlow11,OpenFlow12,OpenFlow13,OpenFlow14,OpenFlow15')
        sw.dpctl(cmd, args)
    print("Add default routes")
    topo.add_default_routes(net, k)
    print("Start data gathering process")
    subprocess.Popen(f"{exe_path} producer "
                     f"{k} {work_dir} {topo.max_num_pods} {36000}> {os.path.join(log_dir, 'producer.log')}", shell=True)
    time.sleep(0.5)
    print("Start HNSA sender apps")
    topo.start_mcast_sender(net, k, use_avx)

    # net.getNodeByName('agg0Ctrl').cmd('route add -host 239.0.2.1 dev agg0Ctrl-eth0')
    # net.getNodeByName('h0').cmd('python3 /home/mininet/dc-emulation/mcast.py receiver all 5007')
    # net.getNodeByName('agg0Ctrl').cmd('python3 /home/mininet/dc-emulation/mcast.py sender 239.4.1.2')
    # net.pingAll(timeout='5')
    CLI(net)
    print("Signal producer to quit at ", quit_p)
    with open(quit_p, 'w') as fh:
        fh.write("quit")
    count = 0
    while os.path.exists(quit_p) and count < 10:
        print("Wait for producer to quit.")
        time.sleep(1)
        count += 1
    net.stop()


topos = {'FatTree': (lambda: FatTree(4))}


if __name__ == '__main__':
    # For example, run with: 
    #   /home/mininet/dc-emulation/topos/nwx.py /home/mininet 4 --max_num_pods 2 --eth eth0
    # To create a k=4 fat tree with only two pods and eth0 connected to first tor
    # in first pod.
    #
    # If you run instead:
    #   /home/mininet/dc-emulation/topos/nwx.py /home/mininet 4
    # A full k = 4 Fat tree is build with no external interface added.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        dest='workdir',
        type=str,
        help="Path to an existing directory. The program will write log files and"
             " other output into this directory."
    )
    parser.add_argument(
        dest="k",
        type=int,
        help="Degree of the Fat-Tree, must be >= 4."
    )
    parser.add_argument(
        '--max_num_pods',
        type=int,
        nargs="?",
        default=1000,
        help="Maximum number of pods. If set to a value smaller than k, only max_num_pods pods are added to the Fat-Tree."
    )
    parser.add_argument(
        '--eth',
        type=str,
        default='',
        nargs='?',
        help="Name of a physical interface on the machine Mininet is running. "
             "If set, the program adds the interface to the first Tor switch "
             "in the first pod as port k + 2. If a host is connected, make sure the"
             "IP is set to 10.0.0.<k + 2>"
    )
    parser.add_argument(
        "--avx",
        action='store_true',
        help="Calculate HNSAs with the AVX-based NN implementation"
    )
    parsed_args, _ = parser.parse_known_args()
    setLogLevel('info')
    print(f"simple_test({parsed_args.workdir}, {parsed_args.max_num_pods:d}, {parsed_args.k:d}, {parsed_args.eth})")
    simple_test(
        work_dir=parsed_args.workdir,
        k=parsed_args.k,
        max_num_pods=parsed_args.max_num_pods,
        ext_eth=parsed_args.eth,
        use_avx=parsed_args.avx
    )
    # simple_test('/home/mininet', 8, "eth0")
