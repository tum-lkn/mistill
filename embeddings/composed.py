"""
Embeddings that are composed out of multiple learned embeddings.
"""
from typing import Dict, List, Any, Tuple
import numpy as np
import networkx as nx
from topos.fattree import make_pod, make_topo
from embeddings.learned import learned_embeddings_gumbel, TradeOffEnergyFunction
from dataprep.sp_prep import add_index_to_nodes


def _get_num_from_str(name: str) -> int:
    s = name.split('-')[1].lstrip(('0'))
    if len(s) == 0:
        return 0
    else:
        return int(s)


def _combine_star_and_pod(fat_tree: nx.DiGraph, pod: nx.DiGraph, k: int,
                                  star_embedding: Dict[int, np.array],
                                  pod_embedding: Dict[int, np.array]) -> Dict[int, np.array]:
    d_star = star_embedding[0].size
    d_pod = pod_embedding[0].size
    d_combined = d_star + d_pod

    combined = {d['idx']: np.zeros(d_combined, dtype=np.float32) for _, d in fat_tree.nodes(data=True)}

    for node in fat_tree.nodes():
        tmp = node.split('-')
        node_t = tmp[0]
        node_num = _get_num_from_str(node)

        idx = fat_tree.nodes[node]['idx']
        if node_t == 'h':
            pod_num, node_num_in_pod = divmod(node_num, int(k**2 / 4))
        elif node_t == 'tor' or node_t == 'agg':
            pod_num, node_num_in_pod = divmod(node_num, int(k / 2))
        else:
            # use only the star embedding for the core switches. Node with
            # index 0 is always the central node.
            embd = star_embedding[0]
            combined[idx][:d_star] = embd
            continue

        pod_name = '{:s}-{:d}'.format(node_t, node_num_in_pod)
        # Since node with index 0 is always the central one in a star, just
        # increment pod_num by one to get an embedding from one of the peripheral
        # nodes.
        star_embd = star_embedding[pod_num + 1]
        pod_embd = pod_embedding[pod.nodes[pod_name]['idx']]
        combined[idx] = np.concatenate([star_embd, pod_embd])
    return combined


def star_pod_decomposition(k: int, d_star: int, d_pod: int) -> Dict[int, np.array]:
    """
    Use a decomposotion of the fat tree in star and pod. The hub of the star
    corresponds to the core. The satelites of the star to pods. For each
    structure, embeddings are learned and then combined. Each core node
    gets the embedding of the hub. Each node in a pod gets a combination of one
    satelite embedding and the corresponding pod embedding.

    Args:
        k:
        d_star:
        d_pod:

    Returns:

    """
    star = add_index_to_nodes(nx.star_graph(k + 1))
    pod = add_index_to_nodes(make_pod(k))
    fat_tree = add_index_to_nodes(make_topo(k))

    alphas = np.repeat(np.linspace(5, 245, 25), 30)
    embds_star, nd, loss = learned_embeddings_gumbel(
        graph=star,
        ndims=d_star,
        e_fct=TradeOffEnergyFunction(d_star, gamma=2 / 58, alpha=alphas),
        n_samples=alphas.size
    )
    embds_pod, nd, loss = learned_embeddings_gumbel(
        graph=pod,
        ndims=d_pod,
        e_fct=TradeOffEnergyFunction(d_pod, gamma=2 / 58, alpha=alphas),
        n_samples=alphas.size
    )

    embedding = _combine_star_and_pod(fat_tree, pod, k, embds_star, embds_pod)
    return embedding
