"""
Learn embeddings for a graph using evolutionary strategies. Conventions in this
document:
    - N denotes the population size of ES.
    - V is the set of nodes, |V| thus the number of nodes.
    - E is the set of edges
    - d is the dimensionality of the embedding.
"""
import networkx as nx
import numpy as np
import torch
from typing import Dict, Any, Tuple
import present
import logging
logging.basicConfig(level=logging.INFO)


if torch.cuda.is_available():  
  DEV = torch.device("cuda:0")
else:  
  DEV = torch.device("cpu")


class EnergyFunction(object):
    """
    Base class for energy functions.
    """
    def optimize(self):
        pass

    def __call__(self, d_H: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SimpleEnergyFunction(EnergyFunction):
    """
    Class that implements an energy function corresponding of the negative
    distance.
    """
    def __init__(self, lr=None):
        super(SimpleEnergyFunction, self).__init__()

    def __call__(self, d_H: torch.Tensor) -> torch.Tensor:
        return -1. * d_H


class LearnableScaledEnergyFunction(EnergyFunction):
    """
    Energy function that implements an energy function using the negative distance
    between nodes and adds a scalar, learnable parameter that scales the distance
    matrix.
    """
    def __init__(self, lr=1e-3):
        super(LearnableScaledEnergyFunction, self).__init__()
        self.log_alpha = torch.tensor(0., dtype=torch.float32, requires_grad=True)
        self.b = torch.tensor(0, dtype=torch.float32, requires_grad=True)
        self.optimizer = torch.optim.Adam([self.log_alpha, self.b], lr=lr)

    def optimize(self):
        self.log_alpha.backward()
        self.b.backward()
        self.optimizer.step()

    def __call__(self, d_H: torch.Tensor) -> torch.Tensor:
        return -1. * torch.exp(self.log_alpha) * d_H


class TradeOffEnergyFunction(LearnableScaledEnergyFunction):
    """
    Implements an energy function that trades of similarity with dissimlarity,
    i.e., the lowest energy is not obtained with perfect equality, but with a
    certain level of difference. Implements the energy function:

    ```
    -a * (\frac{1}{d}d_H - y * (1 - \frac{1}{d}d_H))
    ```
    Parameter a is learnable. Parameter y is fixed and trades off the importance
    of being disimilar with being similar.
    """
    def __init__(self, num_dims: int, lr=1e-3, gamma=None, alpha=None,
                 cancel_self_edges=True):
        """

        Args:
            num_dims: Number of dimensions of parameters.
            lr: Learning rate for optimizer.
            gamma: Parameter trading off benefit of being disimilar.
            cancel_self_edges: If true set energy on main diagonal to zero,
                i.e., self-edges do have a probability of zero.
        """
        super(TradeOffEnergyFunction, self).__init__(lr)
        self.gamma = gamma
        self.cancel_self_edges = cancel_self_edges
        if alpha is None:
            self.alpha = None
        elif type(alpha) in [float, int]:
            self.alpha = torch.tensor(alpha, dtype=torch.float32, device=DEV)
        elif type(alpha) == list:
            self.alpha = torch.tensor(np.array(alpha).reshape(-1, 1, 1), dtype=torch.float32, device=DEV)
        else:
            # assuming numpy array
            self.alpha = torch.tensor(alpha.reshape(-1, 1, 1), dtype=torch.float32, device=DEV)
        self.is_build = False
        self.d = num_dims
        self.cancel_out = None

    def optimize(self):
        pass

    def _build(self, d_H: torch.Tensor):
        """
        Fill in missing parameters based on passed values.

        Args:
            d_H: Distance Matrix.

        Returns:
            None
        """
        self.is_build = True
        # Assign super low energy values to the main diagonal, i.e., to the self
        # edges. They are thus essentially zeroed out. Use a positive large number
        # since I subtract in __call__ --> large number becomes negative.
        self.cancel_out = torch.eye(d_H.shape[-1], dtype=torch.float32, device=DEV).unsqueeze(0) * 1e6
        if self.gamma is None:
            self.gamma = 1. / d_H.shape[-1]
        if self.alpha is None:
            self.alpha = torch.tensor(1, dtype=torch.float32, device=DEV)

    def __call__(self, d_H: torch.Tensor) -> torch.Tensor:
        if not self.is_build:
            self._build(d_H)
        avg_d_H = d_H / d_H.shape[-1]
        # E = -1. * (torch.exp(self.log_alpha) + 0.1) * (avg_d_H - self.gamma * (1. - avg_d_H))
        # E = -1. * self.alpha * (avg_d_H - self.gamma * (1. - avg_d_H))
        # E = -1. * self.alpha * torch.square(avg_d_H - self.gamma)
        E = -1. * self.alpha * avg_d_H
        if self.cancel_self_edges:
            return E - self.cancel_out
        else:
            return E


class Embedding(object):
    """
    Abstract base class for functors creating embeddings.
    """
    def optimize(self):
        pass

    def __call__(self, params: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SigmoidEmbedding(Embedding):
    """
    Creates continuous embeddings through element wise application of the
    sigmoid function
    """
    def __init__(self):
        super(SigmoidEmbedding, self).__init__()

    def __call__(self, params: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(params)


class HardEmbedding(Embedding):
    """
    Create embedding through applying the sigmoid function and then thresholding.
    """
    def __init__(self, threshold):
        """
        Initializes object.

        Args:
            threshold: All values above threshold are cast to one, all others
                to zero.
        """
        super(HardEmbedding, self).__init__()
        self.threshold = threshold

    def __call__(self, params, threshold=None):
        if threshold is None:
            threshold = self.threshold
        bools = torch.sigmoid(params) > threshold
        y = bools.type_as(params)
        return y


class GumbelSoftmaxEmbedding(Embedding):
    """
    Uses the GumbelSoftamx function to create the embeddings of the nodes.
    Has a learnable temperature parameter that is used to implement a schedule.
    """
    def __init__(self, temperature=10, learnable=True, lr=1e-3):
        """
        Initializes object.

        Args:
            temperature: The temperature that is used for the gumbel softmax
                function.
            learnable: Set to true to automatically adapt the temperature.
        """
        super(GumbelSoftmaxEmbedding, self).__init__()
        self.learnable = True
        self.log_temp = torch.tensor(np.log(temperature), dtype=torch.float32, requires_grad=learnable)
        self.temp = temperature
        self.lr = lr
        self.log_temp = torch.log(torch.tensor(self.temp, dtype=torch.float32))
        if learnable:
            self.optimizer = torch.optim.Adam([self.log_temp], lr=lr)
        else:
            self.optimizer = None
        self.is_build = False
        self.mask = None

    def _build(self, params):
        self.mask = torch.ones_like(params)
        self.mask[:, :, :, 1] = 0

    def optimize(self):
        if self.learnable:
            # self.log_temp.backward()
            # self.optimizer.step()
            self.temp = self.temp * (1 - self.lr)
            self.log_temp = np.log(self.temp)

    def __call__(self, params: torch.Tensor, hard=False) -> torch.Tensor:
        assert params.ndim == 4
        if not self.is_build:
            self._build(params)
        castings = torch.nn.functional.gumbel_softmax(params, self.temp + 0.1, hard=hard)
        castings = castings * self.mask
        castings = torch.sum(castings, -1)
        return castings


def _make_adj(graph: nx.Graph) -> Tuple[Dict[Any, int], np.array]:
    """
    Create an adjacency matrix and a mapping of node identifier to indices
    in the adjacency matrix.
    The mapping can be used to retrieve rows/cols of a node in the graph
    from the adjacency matrix.

    Args:
        graph: Graph for which adjacency matrix should be created.

    Returns:
        mapping: Mapping of index to node identifier.
        adj: Adjacency matrix.
    """
    adj = np.zeros((graph.number_of_nodes(), graph.number_of_nodes()), dtype=np.float32)
    mapping = {d: graph.nodes[d]['idx'] for d in graph.nodes()}
    for u in graph.nodes():
        for v in graph.neighbors(u):
            adj[mapping[u], mapping[v]] = 1.
            if not graph.is_directed():
                adj[mapping[v], mapping[u]] = 1.
    return mapping, adj


def _make_spf_distance_mat(graph: nx.Graph, mapping: Dict[Any, int]) -> np.array:
    mat = np.zeros([graph.number_of_nodes()] * 2, dtype=np.float32)
    for u, d in dict(nx.all_pairs_shortest_path_length(graph)).items():
        for v, l in d.items():
            mat[mapping[u], mapping[v]] = l
    mat += np.eye(graph.number_of_nodes(), dtype=np.float32)
    return mat


def _permute_last_two_dims(x: torch.Tensor) -> torch.Tensor:
    """
    Takes as input a tensor and permutes the last two dimensions. Currently
    supported only for three dimensions.

    Args:
        x: Tensor of shape (A, B, C).

    Returns:
        y: Tensor of shape (A, C, B).
    """
    # print(x.shape)
    y = x.permute(0, 2, 1)
    return y


def _find_best_threshold(params: torch.Tensor) -> float:
    """
    Perform a line search over possible split and take the split that results
    in the smallest possible number of duplicates.

    Args:
        params: Logits for the Bernoulli distributions.

    Returns:
        threshold: Bes threshold to create embeddings.
    """
    num_duplicates = []
    stops = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for i in stops:
        num_duplicates.append(float(torch.sum(hamming_distance(HardEmbedding(i)(params)) == 0).numpy()))
    logging.info("Min num duplicates {}".format(np.min(num_duplicates)))
    return stops[np.argmin(num_duplicates)]


def hamming_distance(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the hamming distances between all pairs of nodes as:
        x * (1 - y)^T + (1 - x) * y^T

    Args:
        x: Tensor that contains the binary embeddings. Has shape (N, |V|, d)

    Returns:
        distances: Tensor of shape (N, |V|, |V|)
    """
    y = _permute_last_two_dims(x)
    distances = torch.matmul(x, (1. - y)) + torch.matmul((1. - x), y)
    return distances


def sum_log_prob(x: torch.Tensor, adj: torch.Tensor, energy_fct: callable) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the log-loss of the current binary embeddings.

    Args:
        x: Binary embeddings of shape (N, |V|, d).
        adj: Binary adjacency matrix of shape (|V|, |V|).
        energy_fct: Energy function that returns energy values given the
            distance matrix.

    Returns:
        l_prob: Tensor of shape (N).
        d_H: The hamming distances between the nodes.
    """
    d_H = hamming_distance(x)
    # print("hamming dist", d_H.shape)
    E = energy_fct(d_H)
    logsum = torch.logsumexp(E, 1, True)
    logits = E - logsum
    edge_probs = logits * adj
    l_prob = torch.sum(torch.sum(edge_probs, 1), 1)
    return l_prob, d_H


def learned_embeddings_gumbel(graph: nx.Graph, ndims: int, e_fct: callable,
                              n_samples=10, random_init=False) -> Tuple[Dict[int, np.array], int, float]:
    mapping, adj = _make_adj(graph)
    adj = torch.tensor(adj, device=DEV).unsqueeze(0)
    if random_init:
        params = torch.randn(n_samples, graph.number_of_nodes(), ndims, 2, requires_grad=True, device=DEV)
    else:
        params = torch.zeros(n_samples, graph.number_of_nodes(), ndims, 2, requires_grad=True, device=DEV)
    optimizer = torch.optim.Adam([params], lr=1e-4)
    embed = GumbelSoftmaxEmbedding(lr=2e-5)

    losses = None
    all_losses = []
    best_loss = 1e9
    num_iter = 100000
    patience = 50000
    iter = 0
    while iter < num_iter:
        iter += 1
        embeddings = embed(params)
        losses, _ = sum_log_prob(embeddings, adj, e_fct)
        losses = -1. * losses
        all_losses.append(losses.cpu().detach().numpy())
        loss = torch.sum(losses)
        if loss.cpu().detach().numpy() < best_loss:
            print(iter, loss.cpu().detach().numpy(), embed.temp)
            best_loss = loss.cpu().detach().numpy()
            if iter + patience > num_iter:
                num_iter = iter + patience
        loss.backward()
        optimizer.step()
        e_fct.optimize()
        embed.optimize()
    embeddings = embed(params, hard=True)

    d_H = hamming_distance(embeddings)
    tmp = d_H == 0
    num_duplicates = (torch.sum(torch.sum(tmp.type_as(params), axis=-1), axis=-1) - graph.number_of_nodes()) / 2.
    print("Number of duplicates", num_duplicates.cpu().detach().numpy())
    best_idx = torch.argmin(num_duplicates)

    present.plot_losses(np.row_stack(all_losses))
    present.plot_graph(graph, mapping, d_H[best_idx, :, :].cpu().detach().numpy())
    present.print_embedding(embeddings[best_idx, :, :])
    # present.plot_embedding(embeddings[best_idx, :, :].cpu().detach().numpy())

    embeddings = embeddings.cpu().detach().numpy()
    return {idx: embeddings[best_idx, idx, :] for idx in mapping.values()},\
           num_duplicates[best_idx].cpu().detach().numpy(),\
           losses[best_idx].cpu().detach().numpy()


def learned_embeddings(graph: nx.Graph, ndims: int, e_fct: callable, n_samples=10) -> Dict[Any, np.array]:
    mapping, adj = _make_adj(graph)
    adj = torch.tensor(adj).unsqueeze(0)
    params = torch.randn(n_samples, graph.number_of_nodes(), ndims, requires_grad=True, device=DEV)
    optimizer = torch.optim.Adam([params], lr=1e-4)
    embed = SigmoidEmbedding()

    losses = None
    all_losses = []
    best_loss = 1e9
    num_iter = 150000
    patience = 50000
    iter = 0
    while iter < num_iter:
        iter += 1
        embeddings = embed(params)
        losses, _ = sum_log_prob(embeddings, adj, e_fct)
        losses = -1. * losses
        all_losses.append(losses.cpu().detach().numpy())
        loss = torch.sum(losses)
        if loss.cpu().detach().numpy() < best_loss:
            print(iter, loss.cpu().detach().numpy())
            if iter + patience < num_iter:
                num_iter = iter + patience
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # e_fct.optimize()
        # embed.optimize()
    present.plot_losses(np.row_stack(all_losses))
    best_idx = torch.argmin(losses)
    print('best loss is ', torch.min(losses))

    thres = _find_best_threshold(params[best_idx, :, :].unsqueeze(0))
    logging.info("Best threshold is {}".format(thres))

    embeddings = HardEmbedding(thres)(params)
    d_H = hamming_distance(embeddings)

    present.plot_graph(graph, mapping, d_H[best_idx, :, :].cpu().detach().numpy())
    present.print_embedding(embeddings[best_idx, :, :].cpu().detach().numpy())
    # present.plot_embedding(embeddings[best_idx, :, :].cpu().detach().numpy())
    return {n: embeddings[best_idx, idx, :] for n, idx in mapping.items()}


if __name__ == '__main__':
    from dataprep.sp_prep import add_index_to_nodes
    from topos.fattree import make_pod
    # learned_embeddings(nx.karate_club_graph(), 11, 20)
    # graph = add_index_to_nodes(nx.read_graphml("/opt/project/data/dfn.graphml"))
    graph = add_index_to_nodes(make_pod(8))
    embds, nd, loss = learned_embeddings_gumbel(
        graph=graph,
        ndims=12,
        e_fct=TradeOffEnergyFunction(10, gamma=2 / 58, alpha=5), #alpha=np.linspace(1, 100, 10)),
        n_samples=10
    )
    print("Embedding has {} duplicates with loss {}".format(nd, loss))
    # es_embeddings()

    # tmp = hamming_distance(ps)
    # for i in range(tmp.shape[1]):
    #     for j in range(tmp.shape[2]):
    #         print("{:d}|{:d}".format(int(adj[i, j]), int(tmp[0, i, j])), end=' ')
    #     print()
    # for i in range(ps.shape[1]):
    #     for j in range(ps.shape[2]):
    #         print("{:d}".format(int(ps[0, i, j])), end=' ')
    #     print()
    # plt.imshow(tmp[0, :, :], cmap='Greens')
    # plt.savefig("distmat.pdf", format="pdf")

