"""
Module that generates random embeddings.
"""
import numpy as np
import networkx as nx
from typing import Any, Dict


def _has_duplicates(embd, embeddings) -> bool:
    equals = np.max(np.sum(np.logical_not(np.logical_xor(embd, embeddings)), axis=1))
    equals = equals == embeddings.shape[1]
    return equals


def _check_duplicates(embeddings: np.array, random: np.random.RandomState, p: float) -> np.array:
    """
    Go over embeddings and check for duplicates. If duplicates exist, sample
    a new embedding.
    The `embeddings` argument is mutated..

    Args:
        embeddings:
        random:
        p:

    Returns:

    """
    for i in range(embeddings.shape[0] - 1):
        count = 0
        embd = np.expand_dims(embeddings[i, :], 0)
        while _has_duplicates(embd, embeddings[i + 1:, :]) and count < 100:
            embeddings[i, :] = random.binomial(n=1, p=p, size=embeddings.shape[1])
            embd = np.expand_dims(embeddings[i, :], 0)
            print(embeddings[i, :])
            count += 1
        if count == 100:
            raise ValueError("Not possible to generate non-duplicates")
    return embeddings


def _independent_bernoulli_based(graph: nx.Graph, dim: int, p=0.5, seed=1) -> Dict[int, np.array]:
    """
    Create embeddings based on independent binomials. Each embedding is a binomial
    distribution parameterized with argument `p`.
    Creates non-duplicate embeddings.

    Args:
        num_nodes:
        dim:
        p:
        seed:

    Returns:
        Array of shape (num_nodes, dim)
    """
    random = np.random.RandomState(seed=seed)
    embedding = random.binomial(n=1, p=p, size=(graph.number_of_nodes(), dim))
    embedding = _check_duplicates(embedding, random, p)
    mapping = {}
    for i, n in enumerate(graph.nodes()):
        mapping[graph.nodes[n]['idx']] = embedding[i, :]
    return mapping
