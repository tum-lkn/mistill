"""
Module that implements a fat-tree
"""
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple


def _make_host_nodes(k: int, ratio=1) -> List[str]:
    """
    Create an array of host node identifiers.

    Args:
        k: Degree of fat tree.
        ratio: Oversubscription ratio.

    Returns:
        The hosts.
    """
    num_hosts = int(k**2 / 2) * int(k / 2 * ratio)
    return ['h-{:04d}'.format(i) for i in range(num_hosts)]


def _make_agg_nodes(k: int) -> List[str]:
    """
    Create an array for aggregation nodes.
    Args:
        k:

    Returns:

    """
    num_nodes = int(k * k / 2)
    return ['agg-{:04d}'.format(i) for i in range(num_nodes)]


def _make_tor_nodes(k: int) -> List[str]:
    """
    Create an array for aggregation nodes.
    Args:
        k:

    Returns:

    """
    num_nodes = int(k * k / 2)
    return ['tor-{:04d}'.format(i) for i in range(num_nodes)]


def _make_core_nodes(k: int) -> List[str]:
    """
    Create identifiers for Core nodes.
    Args:
        k:

    Returns:

    """
    num_nodes = int((k / 2)**2)
    return ['core-{:04d}'.format(i) for i in range(num_nodes)]


def _make_host_tor_links(hosts: List[str], tors: List[str], k: int, oversubscription=1) -> List[Tuple[str, str]]:
    """
    Create a list that contains the end points of the hosts-to-tor links.
    Args:
        hosts:
        tors:

    Returns:

    """
    links = []
    num_hosts_per_tor = int(k / 2 * oversubscription)
    for i, host in enumerate(hosts):
        tor_num = int(i / num_hosts_per_tor)
        links.append((host, tors[tor_num]))
    return links


def _make_tor_to_agg_links(tors: List[str], aggs: List[str], k: int) -> List[Tuple[str, str]]:
    """
    Create a list that contains the tuple for tor-aggregation layer
    Args:
        tors:
        aggs:
        k:

    Returns:

    """
    links = []
    nodes_per_pod = int(k / 2)
    for i, tor in enumerate(tors):
        agg_0 = int(i / nodes_per_pod) * nodes_per_pod
        for j in range(nodes_per_pod):
            agg_num = agg_0 + j
            links.append((tor, aggs[agg_num]))
    return links


def _make_agg_to_core_links(aggs: List[str], cores: List[str], k: int) -> List[Tuple[str, str]]:
    """
    Make the links beteween core and aggregation layer.
    Args:
        aggs:
        cores:
        k:

    Returns:

    """
    links = []
    num_aggs = int(k / 2)
    for i, core in enumerate(cores):
        agg_idx = int(i / int(k / 2))
        for pod in range(k):
            links.append((aggs[pod * num_aggs + agg_idx], core))
    return links


def _latitude(name: str, k: int) -> float:
    num = name.split('-')[1].lstrip('0')
    num = 0 if len(num) == 0 else int(num)
    if name.startswith('h'):
        return num
    elif name.startswith('tor') or name.startswith('agg'):
        pod = int(num / (k / 2.))
        num = num - pod * (k / 2.)
        # Move latitude to first host of pod.
        latitude = pod * k**2 / 4.
        # Move latitude to first host under the tor/agg.
        latitude += num * k / 2
        # Move latitude to the center of the hosts.
        latitude += k / 4 - 0.5
        return latitude
    else:
        num_hosts = k**3 / 4
        num_cores = (k / 2)**2
        spacing = num_hosts / num_cores
        return num * spacing


def _longitude(name: str) -> float:
    if name.startswith('h'):
        return 0
    elif name.startswith('tor'):
        return 3
    elif name.startswith('agg'):
        return 6
    elif name.startswith('core'):
        return 9


def make_topo(k: int, oversubpscription=1) -> nx.DiGraph:
    """
    Make a directed graph.

    Args:
        k:
        oversubpscription:

    Returns:

    """
    hosts = _make_host_nodes(k, oversubpscription)
    tors = _make_tor_nodes(k)
    aggs = _make_agg_nodes(k)
    cores = _make_core_nodes(k)

    g = nx.DiGraph()
    g.graph['type'] = 'FatTree'
    for u, v in _make_host_tor_links(hosts, tors, k, oversubpscription):
        g.add_edge(u, v)
        g.add_edge(v, u)
    for u, v in _make_tor_to_agg_links(tors, aggs, k):
        g.add_edge(u, v)
        g.add_edge(v, u)
    for u, v in _make_agg_to_core_links(aggs, cores, k):
        g.add_edge(u, v)
        g.add_edge(v, u)

    for u in g.nodes():
        g.nodes[u]['Latitude'] = _latitude(u, k)
        g.nodes[u]['Longitude'] = _longitude(u)

    return g


def make_pod(k: int) -> nx.DiGraph:
    """
    Create a graph that represents one pod.
    Args:
        k:

    Returns:

    """
    g = nx.DiGraph()

    num_hosts = int(k**2 / 4.)
    num_tors = int(k / 2)
    for i in range(num_hosts):
        g.add_node('h-{:d}'.format(i))
    for i in range(num_tors):
        g.add_node('tor-{:d}'.format(i))
        g.add_node('agg-{:d}'.format(i))

    for tor in range(num_tors):
        for host in range(num_tors):
            g.add_edge('h-{:d}'.format(tor * num_tors + host), 'tor-{:d}'.format(tor))
            g.add_edge('tor-{:d}'.format(tor), 'h-{:d}'.format(tor * num_tors + host))

    for tor in range(num_tors):
        for agg in range(num_tors):
            g.add_edge('tor-{:d}'.format(tor), 'agg-{:d}'.format(agg))
            g.add_edge('agg-{:d}'.format(agg), 'tor-{:d}'.format(tor))
    return g
