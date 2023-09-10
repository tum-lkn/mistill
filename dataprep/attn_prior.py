"""
Module that implements the creation of a prior for the attention weights.
"""
import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Any
from dataprep.link_failures import _get_pod, _get_num_no_leaves
from dataprep.sp_prep import IDX, NS_IDX, H_IDX
import pandas as pd


def empty_prior(k):
    return np.zeros(int(5 / 4 * k ** 2), dtype=np.float32)


def _core_group(name: str, k: int) -> List[str]:
    num = name.split('-')[1].lstrip('0')
    num = 0 if len(num) == 0 else int(num)

    start = int(num / (k / 2)) * int(k / 2)
    end = int(start + (k / 2))
    return ['core-{:04d}'.format(i) for i in range(start, end)]


def _get_tor_num(name: str, k: int) -> int:
    num = name.split('-')[1].lstrip('0')
    num = 0 if len(num) == 0 else int(num)
    return int(num / (k / 2))


def _get_pod_num(name: str, k: int) -> int:
    num = name.split('-')[1].lstrip('0')
    num = 0 if len(num) == 0 else int(num)
    if name.startswith('h-'):
        pod = int(num / (k**2 / 4))
    elif name.startswith('agg-') or name.startswith('tor-'):
        pod = int(num / (k / 2))
    else:
        raise KeyError
    return pod


def _get_tor_to_host(name: str, k: int) -> str:
    num = name.split('-')[1].lstrip('0')
    num = 0 if len(num) == 0 else int(num)
    return "tor-{:04d}".format(int(num / (k / 2)))


def _get_agg_tor_names_one_pod(pod_num: int, k: int, template: str) -> List[str]:
    start = int(pod_num * k / 2)
    end = int((pod_num + 1) * k / 2)
    return [template.format(i) for i in range(start, end)]


def _get_tor_names_in_pod(pod_num: int, k: int) -> List[str]:
    return _get_agg_tor_names_one_pod(pod_num, k, 'tor-{:04d}')


def _get_agg_names_in_pod(pod_num: int, k: int) -> List[str]:
    return _get_agg_tor_names_one_pod(pod_num, k, 'agg-{:04d}')


def _prior_dst(host: str, graph: nx.Graph, k: int) -> np.array:
    prior = empty_prior(k)
    tor = _get_tor_to_host(host, k)
    prior[graph.nodes[tor][NS_IDX]] = 1.
    for agg in _get_agg_names_in_pod(_get_pod_num(host, k), k):
        prior[graph.nodes[agg][NS_IDX]] = 1.
    return prior


def _prior_cur_loc_tor(tor: str, graph: nx.Graph, k: int) -> np.array:
    prior = empty_prior(k)
    prior[graph.nodes[tor][NS_IDX]] = 1.
    pod_num = _get_pod_num(tor, k)
    for agg in _get_agg_names_in_pod(pod_num, k):
        prior[graph.nodes[agg][NS_IDX]] = 1.
    return prior


def _prior_cur_loc_agg(agg: str, graph: nx.Graph, k: int) -> np.array:
    prior = empty_prior(k)
    pod_num = _get_pod_num(agg, k)
    for agg in _get_agg_names_in_pod(pod_num, k):
        prior[graph.nodes[agg][NS_IDX]] = 1.
    return prior


def _prior_cur_loc_core(core: str, graph: nx.Graph, k: int) -> np.array:
    prior = empty_prior(k)
    for core in _core_group(core, k):
        prior[graph.nodes[core][NS_IDX]] = 1
    return prior


def priors_fat_tree(graph: nx.Graph, k: int, num_heads: int) -> Tuple[np.array, np.array]:
    """
    Creates a three dimensional array. The first dimension corresponds to
    nodes. The second dimension and third dimension to the attention
    weights for this specific node as destination.

    Args:
        graph:

    Returns:

    """
    nnl = _get_num_no_leaves(graph)
    for cur_loc in graph.nodes():
        if not NS_IDX in graph.nodes[cur_loc]:
            continue
        for dst in graph.nodes():
            if not dst.startswith('h-'):
                continue
            prior = np.zeros(nnl, dtype=np.float32)
            pod_dst = _get_pod(dst, k)
            if cur_loc.startswith('core-'):
                core_grp = _core_group(cur_loc, k)
                for i in range(int(core_grp * k / 2), int(core_grp * k / 2 + k / 2)):
                    ni = 'core-{:04d}'.format(i)
                    ni_idx = graph.nodes[ni][NS_IDX]
                    prior[ni_idx] = 1
            elif cur_loc.startswith('tor-'):
                pod = _get_pod(cur_loc, k)
                for i in range(int(pod * k / 2), int(pod * k / 2 + k / 2)):
                    ni = 'agg-{:04d}'.format(i)
                    ni_idx = graph.nodes[ni][NS_IDX]
                    prior[ni_idx] = 1.
            else:
                pass
            ni = 'tor-{:04d}'.format(_get_tor(dst, k))
            prior[graph.nodes[ni][NS_IDX]] = 1

            for i in range(int(pod * k / 2), int(pod * k / 2 + k / 2)):
                ni = 'agg-{:04d}'.format(i)
                ni_idx = graph.nodes[ni][NS_IDX]
                prior[ni_idx] = 1.


def _activations_dst_tor_head(host: str, graph: nx.Graph, k: int) -> np.array:
    prior = empty_prior(k)
    tor = _get_tor_to_host(host, k)
    prior[graph.nodes[tor][NS_IDX]] = 1.
    return prior


def _activations_other_tors_dst_pod(host: str, graph: nx.Graph, k: int) -> np.array:
    ntor = _get_tor_to_host(host, k)
    tors = _get_tor_names_in_pod(_get_pod_num(host, k), k)
    priors = []
    for tor in tors:
        if tor == ntor:
            continue
        else:
            prior = empty_prior(k)
            prior[graph.nodes[tor][NS_IDX]] = 1
            priors.append(prior)
    return priors


def _activation_dst_agg(host: str, graph: nx.Graph, k: int) -> List[np.array]:
    pod_num = _get_pod_num(host, k)
    aggs = _get_agg_names_in_pod(pod_num, k)
    activations = []
    for agg in aggs:
        prior = empty_prior(k)
        prior[graph.nodes[agg][NS_IDX]] = 1
        activations.append(prior)
    return activations


def _self_activations(node: str, graph: nx.Graph, k: int) -> np.array:
    prior = empty_prior(k)
    prior[graph.nodes[node][NS_IDX]] = 1
    return prior


def _activations_src_agg(tor_or_agg: str, graph: nx.Graph, k: int) -> List[np.array]:
    pod_num = _get_pod_num(tor_or_agg, k)
    batch = _get_agg_names_in_pod(pod_num, k)
    priors = []
    for node in batch:
        prior = empty_prior(k)
        prior[graph.nodes[node][NS_IDX]] = 1
        priors.append(prior)
    return priors


def _activations_core(core: str, graph: nx.Graph, k: int) -> List[np.array]:
    priors = []
    for core in _core_group(core, k):
        prior = empty_prior(k)
        prior[graph.nodes[core][NS_IDX]] = 1
        priors.append(prior)
    return priors


def activations_fat_tree(graph: nx.Graph, k: int) -> np.array:
    nws = [n for n, d in graph.nodes(data=True) if NS_IDX in d]
    nns = [n for n, d in graph.nodes(data=True) if NS_IDX not in d]

    node_to_state_idx = {n: graph.nodes[n][NS_IDX] for n in nws}
    prior = np.zeros((len(nws), len(nws)), dtype=np.float32)
    for host in ['h-0000']:
        for state_node in nws:
            idx = graph.nodes[state_node][NS_IDX]
            prior[idx, :] = prior[idx, :] + _activations_dst_tor_head(host, graph, k)
            for x in _activation_dst_agg(host, graph, k):
                prior[idx, :] = prior[idx, :] + x

            prior[idx, :] = prior[idx, :] + _self_activations(state_node, graph, k)
            priors = _activations_core(state_node, graph, k) if state_node.startswith('core-') \
                else _activations_src_agg(state_node, graph, k)
            for x in priors:
                prior[idx, :] = prior[idx, :] + x
    names = [0] * len(nws)
    for n in nws:
        names[node_to_state_idx[n]] = n
    return pd.DataFrame(prior, index=names, columns=names)


def _attn_head_activations_dsts(graph: nx.Graph, k: int) -> np.array:
    """
    Calculate optimal activations of self-attention layer. Has shape
    (num_hosts, ceil(k / 2) + 1, num_switches).
    Args:
        graph:
        k:

    Returns:

    """
    nns = [n for n, d in graph.nodes(data=True) if NS_IDX not in d]
    activations = np.zeros(
        (len(nns), int(k / 2 + 1), graph.number_of_nodes() - len(nns)),
        dtype=np.float32
    )
    for host in nns:
        idx = graph.nodes[host][H_IDX]
        for i, x in enumerate(_activation_dst_agg(host, graph, k)):
            activations[idx, i, :] = x
        activations[idx, -1, :] = _activations_dst_tor_head(host, graph, k)
    return activations


def _attn_head_activations_cur_loc(graph: nx.Graph, k: int) -> np.array:
    """
    Calculate optimal activations of self-attention layer. Has shape
    (num_switches, ceil(k / 2) + 1, num_switches).
    Args:
        graph:
        k:

    Returns:

    """
    nws = [n for n, d in graph.nodes(data=True) if NS_IDX in d]
    activations = np.zeros(
        (len(nws), int(k / 2 + 1), len(nws))
    )
    for state_node in nws:
        idx = graph.nodes[state_node][NS_IDX]
        priors = _activations_core(state_node, graph, k) if state_node.startswith('core-') \
            else _activations_src_agg(state_node, graph, k)
        for i, x in enumerate(priors):
            activations[idx, i, :] = x
        activations[idx, -1, :] = _self_activations(state_node, graph, k)
    return activations


def _attn_head_activations_all(graph: nx.Graph, k: int) -> np.array:
    activations = np.zeros(
        (graph.number_of_nodes(), int(k / 2 + 1), int(5 / 4 * k**2)),
        dtype=np.float32
    )
    activations_host = _attn_head_activations_dsts(graph, k)
    activations_sw = _attn_head_activations_cur_loc(graph, k)
    for node in graph.nodes():
        idx = graph.nodes[node][IDX]
        if node.startswith('h-'):
            idx_h = graph.nodes[node][H_IDX]
            acti = activations_host[idx_h]
        else:
            idx_sw = graph.nodes[node][NS_IDX]
            acti = activations_sw[idx_sw]
        activations[idx] = acti
    return activations


def _attn_head_only_src_tor_agg(graph: nx.Graph, k: int) -> Tuple[np.array, np.array]:
    activations = np.zeros(
        (graph.number_of_nodes(), int(k / 2 - 1), int(5 / 4 * k**2)),
        dtype=np.float32
    )
    is_contained = np.zeros(graph.number_of_nodes(), dtype=np.float32)
    for node in graph.nodes():
        idx = graph.nodes[node]['idx']
        if node.startswith('h-'):
            for i, prior in enumerate(_activations_other_tors_dst_pod(node, graph, k)):
                activations[idx, i, :] = prior
        elif node.startswith('core-'):
            is_contained[idx] = 2
            sidx = graph.nodes[node][NS_IDX]
            for i in range(int(k / 2 - 1)):
                prior = empty_prior(k)
                prior[sidx] = 1
                activations[idx, i, :] = prior
        else:
            is_contained[idx] = 1
    return is_contained, activations


if __name__ == '__main__':
    import h5py
    import os
    from topos.fattree import make_topo
    from dataprep.sp_prep import add_index_to_nodes
    k = 16
    graph = add_index_to_nodes(make_topo(k))

    dpath = '/opt/project/data/fat-tree-k16'
    # file = h5py.File(os.path.join(dpath, 'attn_activations_dst.h5'), 'w')
    # file.create_dataset("activations", data=_attn_head_activations_dsts(graph, k))
    # file.close()

    # file = h5py.File(os.path.join(dpath, 'attn_activations_cur_loc.h5'), 'w')
    # file.create_dataset("activations", data=_attn_head_activations_cur_loc(graph, k))
    # file.close()

    file = h5py.File(os.path.join(dpath, 'attn_activations_all.h5'), 'w')
    file.create_dataset("activations", data=np.expand_dims(_attn_head_activations_all(graph, k), axis=-2))
    file.close()

    # file = h5py.File(os.path.join(dpath, 'attn_dst_tor.h5'), 'w')
    # is_contained, activations = _attn_head_only_src_tor_agg(graph, k)
    # file.create_dataset("activations", data=np.expand_dims(activations, axis=-2))
    # file.create_dataset("is_contained", data=is_contained)
    # file.close()
