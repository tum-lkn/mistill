"""
Implements embeddings based on default IP address assignment.
"""
import numpy as np
import networkx as nx
from typing import Dict, Any
from dataprep.sp_prep import IDX


def _to_binary(decimal: int) -> np.array:
    return np.unpackbits(np.array([decimal], dtype=np.uint8)).astype(np.float32)


def _get_num_from_str(name: str) -> int:
    s = name.split('-')[1].lstrip(('0'))
    if len(s) == 0:
        return 0
    else:
        return int(s)


class HostIp(object):
    def __init__(self, k: int):
        self.k = k
        self.hosts_per_tor = int(k / 2)
        self.hosts_per_pod = int(k / 2 * self.hosts_per_tor)
        self.tors_per_pod = int(k / 2)

    def __call__(self, node: str) -> np.array:
        num = _get_num_from_str(node)
        total_tor = int(num / self.hosts_per_tor)
        pod, pod_tor = divmod(total_tor, self.tors_per_pod)
        host = num - total_tor * self.hosts_per_tor
        return np.concatenate([
            _to_binary(pod),
            _to_binary(pod_tor),
            _to_binary(host + 2)  # Offset of two because of ToR addressing, i.e., .1 is ToR.
        ])


class TorIp(object):
    def __init__(self, k: int):
        self.k = k
        self.tors_per_pod = int(k / 2)

    def __call__(self, node: str) -> np.array:
        num = _get_num_from_str(node)
        pod = int(num / self.tors_per_pod)
        tor = num - pod * self.tors_per_pod
        return np.concatenate([
            _to_binary(pod),
            _to_binary(tor),
            np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        ])


class AggIp(object):
    def __init__(self, k: int):
        self.k = k
        self.aggs_per_pod = int(k / 2)

    def __call__(self, node: str) -> np.array:
        num = _get_num_from_str(node)
        pod = int(num / self.aggs_per_pod)
        agg = num - pod * self.aggs_per_pod
        return np.concatenate([
            _to_binary(pod),
            _to_binary(agg + self.aggs_per_pod),  # Offset because first k/2 values are used by ToR.
            np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        ])


class CoreIp(object):
    def __init__(self, k: int):
        self.k = k
        self.grid_dim = int(k / 2)

    def __call__(self, node: str) -> np.array:
        num = _get_num_from_str(node)
        i_lvl = int(num / self.grid_dim)
        j_lvl = num - i_lvl * self.grid_dim
        return np.concatenate([
            _to_binary(self.k),
            _to_binary(i_lvl + 1),
            _to_binary(j_lvl + 1)
        ])


def fat_tree_ip_scheme(graph: nx.Graph, k: int) -> Dict[int, np.array]:
    map = {
        'h': HostIp(k),
        'tor': TorIp(k),
        'agg': AggIp(k),
        'core': CoreIp(k)
    }
    return {graph.nodes[node]['idx']: map[node.split('-')[0]](node) for node in graph.nodes()}


def adj_embeddings(graph: nx.Graph) -> Dict[int, np.array]:
    """
    The embedding used for each node is the distance to every other node in
    the network, i.e., one row of the distance matrix.

    Args:
        graph:

    Returns:

    """
    adj_matrix = np.zeros([graph.number_of_nodes()] * 2, dtype=np.float32)
    distance_dict = nx.all_pairs_shortest_path_length(graph)
    for source, d in distance_dict:
        for target, distance in d.items():
            src_idx = graph.nodes[source][IDX]
            dst_idx = graph.nodes[target][IDX]
            adj_matrix[src_idx, dst_idx] = distance
    return {i: adj_matrix[i, :] for i in range(graph.number_of_nodes())}

