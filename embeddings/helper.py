"""
Implements common functions used in all embedding files.
"""
import torch
import numpy as np
from typing import Union


def _power_adj(adj: torch.Tensor, adj_p: torch.Tensor, bfs: torch.Tensor,
               power: int) -> torch.Tensor:
    """
    Recursively calculate the breadth first search neighborhoods. In each step
    of the recursion raise the adjacency matrix to an additional power.

    Args:
        adj: Original adjacency matrix.
        adj_p: pth power of the adjacency matrix.
        power: Exponent.

    Returns:
        bfs: Adjacency matrix with edges set between all pairs of nodes that can
            reach themselves with a walk of length `power`.
    """
    if power == 1:
        return bfs
    else:
        adj_p = torch.matmul(adj_p, adj)
        return _power_adj(adj, adj_p, torch.clamp(adj_p + bfs, 0, 1), power - 1)


def make_bfs_neighborhoods(adj: torch.Tensor, walk_length: int) -> torch.Tensor:
    """
    Return an adjacency matrix that adds edges to all nodes that are reachable
    from every other node within a walk length of `walk_length`. For example,
    if a graph has the edges (a, b) and (b, c), then for `walk_length = 2` the
    additional edge (a, c) is added to the adjacency matrix.

    Args:
        adj: Adjacency matrix of the graph.
        walk_length: The length of the breadth-first search in which
            neighbors should be added.

    Returns:
        adj_p: Adjacency with additional edges based on the neighborhoods that
            are generated.
    """
    bfs = _power_adj(adj, adj, adj, walk_length) * (1. - torch.eye(adj.shape[0]))
    return bfs
