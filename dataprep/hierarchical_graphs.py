"""
This module implements the simple version of the hierarchical community detection
algorithm presented in https://arxiv.org/pdf/1810.01509.pdf
"""
import networkx as nx
import numpy as np
import scipy
import sys

sys.path.insert(0, '/opt/project')
import dataprep.sp_prep as dp
import present
import topos.fattree


class Level(object):
    def __init__(self, graph):
        self.graph = graph
        self.negative_group = None
        self.positive_group = None


def make_adj(graph, node_to_idx):
    adj = np.zeros([graph.number_of_nodes()] * 2, dtype=np.float32)
    for u, v in graph.edges():
        adj[node_to_idx[u], node_to_idx[v]] = 1.
        adj[node_to_idx[v], node_to_idx[u]] = 1.
    return adj


def cluster(level: Level, remaining_descents: int):
    if remaining_descents == 0:
        return
    node_to_idx = {}
    idx_to_node = {}
    for u, d in level.graph.nodes(data=True):
        node_to_idx[u] = d['idx']
        idx_to_node[d['idx']] = u
    adj = make_adj(level.graph, node_to_idx)

    # Get the largest two eigenvalues and according eigenvectors of the
    # adjacency matrix. The eigenvalues are returned in ascending order,
    # i.e., the first entry is the second largest eigenvalue.
    n = level.graph.number_of_nodes()
    eigvals, eigvecs = scipy.linalg.eigh(adj, subset_by_index=[n-2, n-1])

    c1 = []
    c2 = []
    for i, x in enumerate(eigvecs[:, 0]):
        if x < 0:
            c1.append(idx_to_node[i])
        else:
            c2.append(idx_to_node[i])

    if len(c1) > 1:
        l1 = Level(dp.add_index_to_nodes(nx.subgraph(level.graph, c1)))
        level.negative_group = l1
        cluster(l1, remaining_descents - 1)
        print("Connected: ", nx.is_connected(l1.graph))
    if len(c2) > 1:
        l2 = Level(dp.add_index_to_nodes(nx.subgraph(level.graph, c2)))
        print("Connected: ", nx.is_connected(l2.graph))
        level.positive_group = l2
        cluster(l2, remaining_descents - 1)


if __name__ == '__main__':
    graph = dp.add_index_to_nodes(nx.read_graphml("/opt/project/data/dfn.graphml"))
    graph = dp.add_index_to_nodes(nx.random_internet_as_graph(208))
    level = Level(graph)
    cluster(level, 2)
    present.plot_hierarchy(level)
