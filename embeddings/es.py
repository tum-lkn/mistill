"""
Uses NES to mutate embeddings.
"""
import torch
import embeddings.learned as embdl
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Any
import present
import embeddings.helper as helper
from embeddings.node2vec_walk import make_neighborhoods


def make_noise(num_samples: int, num_nodes: int, num_dims: int):
    normal = torch.randn(num_samples, num_nodes, num_dims)
    uniform = torch.rand(num_samples, num_nodes, num_dims) > 0.1
    uniform = uniform.type(torch.float32)
    return normal * uniform


def apply_gradient(params: torch.Tensor, grads: torch.Tensor, lr: float) -> torch.Tensor:
    """
    Apply the gradient to the parameters.

    Args:
        params: Tensor with parameters of shape (|V|, d).
        grads: Tensor with gradients of shape (|V|, d).

    Returns:
        params_prime: Tensor of shape (|V|, d).
    """
    params_prime = params + lr * grads
    # print("Gradient", torch.mean(grads))
    return params_prime


def calculate_gradient(scaled_noise: torch.Tensor) -> torch.Tensor:
    """
    Calcualte the gradient from the scaled noise.

    Args:
        scaled_noise: Tensor of shape (N, |V|, d).

    Returns:
        grad: Tensor of shape (|V|, d).
    """
    grad = torch.sum(scaled_noise, 0)
    return grad


def scale_noise(noise: torch.Tensor, fitness: torch.Tensor, sum_log_probs: torch.Tensor,
                d_H=None) -> torch.Tensor:
    """
    Scale noise with the fitness values. Fitness values are assigned to noise
    vectors based on the sum_log_probs vector. The better the sum_log_probs vector,
    the higher the fitness and thus the more the gradient should point into
    that direction.

    Args:
        noise: Noise used to perturb the parameters (N, |V|, d).
        fitness: Vector of scalars, first element `fitness[0]` is the fitness corresponding
            to the best sample, the last element `fitness[-1] is the fitness
            for the worst sample. Has shape (d).
        sum_log_probs: The sum of the edge log probabilities. Has shape (d).
        d_H: The hamming distances between nodes across all samples.

    Returns:
        scaled_noise: Noise scaled with fitness values. Has shape (N, |V|, d).
    """
    sorted_fitness = fitness[torch.argsort(sum_log_probs)]
    # reshape sorted fitness to shape (N, 1, 1).
    scaled_noise = sorted_fitness.unsqueeze(1).unsqueeze(2) * noise
    return scaled_noise


def mutate_params(x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """
    Add white noise to x.

    Args:
        x: Logits for Bernoullise of shape (|V|, d).
        noise: white noise of shape (N, |V|, d)

    Returns:
        y: Mutated logits of shape (N, |V|, d).
    """
    return x.unsqueeze(0) + noise


def make_embeddings(x: torch.Tensor, threshold=0.5) -> torch.Tensor:
    """
    Create binary embeddings from logits. Transform the logits to the
    interval (0, 1) using the sigmoid activation function and then obtain
    a binary vector by thresholding.

    Args:
        x: Logits for Bernoullis of shape (N, |V|, d).

    Returns:
        y: Binary embedding of shape (N, |V|, d).
    """
    bools = torch.sigmoid(x) > threshold
    y = bools.type_as(x)
    return y


def make_utilities(population_size):
    utility_values = np.log(population_size / 2. + 1) - \
                     np.log(np.arange(1, population_size + 1))
    utility_values[utility_values < 0] = 0.
    utility_values /= np.sum(utility_values)
    utility_values -= 1. / population_size
    utility_values = utility_values.astype(np.float32)
    return utility_values


def learned_embeddings(graph: nx.Graph, ndims: int, e_fct: callable,
                              n_samples=10, lr=1.) -> Tuple[Dict[int, np.array], int, float]:
    # n_graph = add_index_to_nodes(make_neighborhoods(graph, 1., 0.5, 5, 5))
    mapping, adj = embdl._make_adj(graph)
    # adj = helper.make_bfs_neighborhoods(torch.tensor(adj, device=embdl.DEV), 2)
    adj = torch.tensor(adj, device=embdl.DEV).unsqueeze(0)
    # adj = adj.unsqueeze(0)
    params = torch.randn(graph.number_of_nodes(), ndims, requires_grad=True, device=embdl.DEV)
    embed = lambda x: make_embeddings(x)
    fitness = torch.tensor(make_utilities(n_samples))

    all_losses = []
    best_loss = 1e9
    best_params = None
    num_iter = 100000
    patience = 50000
    iter = 0
    while iter < num_iter:
        iter += 1
        noise = make_noise(n_samples, graph.number_of_nodes(), ndims)
        mutations = mutate_params(params, noise)
        embeddings = embed(mutations)
        # print("embeddings", embeddings.shape)
        losses, d_H = embdl.sum_log_prob(embeddings, adj, e_fct)
        losses = -1. * losses
        all_losses.append(losses.cpu().detach().numpy())

        bl = torch.min(losses)
        if bl < best_loss:
            print(iter, bl.cpu().detach().numpy())
            best_loss = bl
            best_params = mutations[torch.argmax(losses), :, :]
            if iter + patience > num_iter:
                num_iter = iter + patience

        scaled_noise = scale_noise(noise, fitness, losses)
        grads = calculate_gradient(scaled_noise)
        params = apply_gradient(params, grads, lr)
    embeddings = embed(best_params)

    d_H = embdl.hamming_distance(torch.unsqueeze(embeddings, dim=0))
    tmp = d_H == 0
    num_duplicates = (torch.sum(torch.sum(tmp.type_as(params), axis=-1), axis=-1) - graph.number_of_nodes()) / 2.
    print("Number of duplicates", num_duplicates.cpu().detach().numpy())

    present.plot_graph(graph, mapping, d_H[0, :, :].cpu().detach().numpy())
    present.print_embedding(embeddings)
    # present.plot_embedding(embeddings[best_idx, :, :].cpu().detach().numpy())

    return {idx: embeddings[idx, :] for idx in mapping.values()}, \
           num_duplicates.cpu().detach().numpy(), \
           best_loss.cpu().detach().numpy()


if __name__ == '__main__':
    from dataprep.sp_prep import add_index_to_nodes
    graph = add_index_to_nodes(nx.read_graphml("/opt/project/data/dfn.graphml"))
    embds, nd, loss = learned_embeddings(
        graph=graph,
        ndims=10,
        e_fct=embdl.TradeOffEnergyFunction(5000, gamma=2 / 58, alpha=5, cancel_self_edges=True),
        n_samples=10
    )
    print("Embedding has {} duplicates".format(nd))
