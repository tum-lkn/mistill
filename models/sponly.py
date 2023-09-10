"""
Implements neural network models for shortest paths only.
"""
import torch
from typing import List, Tuple, Dict, Any
import layers.attn as attn
import models.utils as mutils


class SpfConfig(object):
    @classmethod
    def from_dict(cls, dc: Dict[str, Any]) -> 'SpfConfig':
        return cls(
            max_degree=dc['max_degree'],
            dim_embedding=dc['dim_embedding'],
            num_heads=dc['num_heads'],
            dim_attn_hidden=dc['dim_attn_hidden'],
            dim_attn_out=dc['dim_attn_out'],
            dim_out_fcn=dc['dim_out_fcn'],
            mode=dc.get('mode', "attn"),
            num_nodes=dc.get("num_nodes", None)
        )

    def __init__(self, max_degree: int, dim_embedding: int, num_heads: int,
                 dim_attn_hidden: int, dim_attn_out: int, dim_out_fcn: List[int],
                 mode: str, num_nodes=None):
        """

        Args:
            max_degree: Maximum node degree in network.
            dim_embedding: Dimensionality of node embedding.
            num_heads: Number of attention heads.
            dim_attn_hidden: Dimensionality of space keys and queries of the
                SelfAttentionLayer are transformed to.
            dim_attn_out: Output dimensionality of SelfAttentionLayer, i.e.,
                dimension of space values are transformed to.
            dim_fcn: Hidden layer sizes of fully connected layers producing
                the final output.
        """
        self.max_degree = max_degree
        self.dim_embedding = dim_embedding
        self.num_heads = num_heads
        self.dim_attn_hidden = dim_attn_hidden
        self.dim_attn_out = dim_attn_out
        self.dim_fcn = dim_out_fcn
        self.num_nodes = num_nodes
        self.mode = mode

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


class SpfModel(torch.nn.Module):
    """
    Implements a neural network with two inputs:
        1) Embeddings of neighbors.
        2) Embedding of destination
    The neighborhood encoding is inserted into a multi-head attention mechanism.
    The embedding of the destination serves as query. The neighborhood
    embedding as keys and values.
    The output is passed through a sequence of fully connected layer. The
    output corresponds to the maximum degree in the network and represents
    the port, i.e., neighbor, to which the packet should be forwarded.
    The neighbor sequence for encodings and output is the same, i.e., the ith
    output and the ith encoding correspond to the same neighbor.
    """
    def __init__(self, config: SpfConfig):
        """
        Initializes object.
        Args:
            config:
        """
        super(SpfModel, self).__init__()
        self.config = config

        if self.config.mode == 'attn':
            self.multi_head_attention = attn.MultiHeadAttentionLayer(
                num_heads=config.num_heads,
                attention_class='SelfAttentionLayer',
                dim_in=2 * config.dim_embedding,
                dim_out=config.dim_attn_out,
                dim_hidden=config.dim_attn_hidden
            )
            self.fcn_out = mutils.make_sequential(
                layer_sizes=config.dim_fcn,
                dim_in=config.num_heads * config.dim_attn_out,
                activation='relu'
            )
            self.logits = torch.nn.Linear(config.dim_fcn[-1], config.max_degree)
        else:
            self.fcn_out = mutils.make_sequential(
                layer_sizes=config.dim_fcn,
                dim_in=2 * config.dim_embedding,
                activation='relu'
            )
        self.logits = torch.nn.Linear(config.dim_fcn[-1], config.max_degree)

    def forward(self, queries: torch.Tensor, others: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Perform forward pass.
        Args:
            queries: Queries of shape (BS, 1, E).
            others: Values and Keys of shape (BS, T_max, E).
            mask: Attention mask of shape (BS, T_max, 1).

        Returns:
            output: Tensor of shape (BS, T_max), i.e., predicted output port.
            scores: List of tensors, attention scores for each head.
        """
        if self.config.mode == 'attn':
            output, scores = self.multi_head_attention(
                keys=others,
                values=others,
                queries=queries,
                attention_mask=mask
            )
            # Since queries have second dimension of one its possible to just remove
            # this dimension and get a (BS, num_out) output.
            fc_in = torch.squeeze(output, dim=1)
        else:
            scores = torch.zeros(1)
            fc_in = queries
        output = self.logits(self.fcn_out(fc_in))
        return output, scores


class EmbeddingSpfModel(SpfModel):
    """
    Implements a neural network with two inputs:
        1) indices of neighbors.
        2) index of destination
    The indices are used to retrieve corresponding parameters for a gumbel
    softmax layer that is used to learn binary representations for the nodes
    jointly with the classification objective.
    The overall architecture is then the same as for the SpfModel, except of
    the additional paramters for the Gumbel-Softmax function.
    """
    def __init__(self, config: SpfConfig):
        super(EmbeddingSpfModel, self).__init__(config)
        assert config.num_nodes is not None
        self.params_embeddings = torch.nn.Parameter(
            data=torch.ones(config.num_nodes, config.dim_embedding, 2),
            requires_grad=True
        )
        self.tau = 0.6
        # self.tau = torch.tensor(10., requires_grad=True)
        # self.tau_optim = torch.optim.Adam([self.tau], lr=1e-2)

    def _step_tau(self):
        # self.tau.backward()
        # self.tau_optim.step()
        # self.tau_optim.zero_grad()
        pass

    def forward(self, queries: torch.Tensor, others: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Perform forward pass.
        Args:
            queries: Queries of shape (BS, 1, 1).
            others: Values and Keys of shape (BS, T_max, 1).
            mask: Attention mask of shape (BS, T_max, 1).

        Returns:
            output: Tensor of shape (BS, T_max), i.e., predicted output port.
            scores: List of tensors, attention scores for each head.
        """
        if self.config.mode == 'attn':
            params_dsts = torch.reshape(
                self.params_embeddings[torch.flatten(queries).long(), :, :],
                [-1, self.config.dim_embedding, 2]
            )
            embeddings_dsts = torch.nn.functional.gumbel_softmax(
                params_dsts,
                tau=self.tau,  # torch.clamp(self.tau, 0.1, 11.),
                hard=not self.training,
                dim=-1
            )# [:, :, 0]
            embeddings_dsts = torch.reshape(embeddings_dsts, [-1, 2 * self.config.dim_embedding])
            params_neighbors = torch.reshape(
                self.params_embeddings[torch.flatten(others).long(), :, :],
                [-1, self.config.max_degree, self.config.dim_embedding, 2]
            )
            embeddings_neighbors = torch.nn.functional.gumbel_softmax(
                params_neighbors,
                tau=self.tau, #torch.clamp(self.tau, 0.1, 11.),
                hard=not self.training,
                dim=-1
            )# [:, :, :, 0]
            embeddings_neighbors = torch.reshape(
                embeddings_neighbors,
                [-1, self.config.max_degree, 2 * self.config.dim_embedding]
            )
            output, scores = self.multi_head_attention(
                keys=embeddings_neighbors,
                values=embeddings_neighbors,
                queries=torch.unsqueeze(embeddings_dsts, 1),
                attention_mask=mask
            )
        else:
            params_dsts = torch.reshape(
                self.params_embeddings[torch.flatten(queries).long(), :, :],
                [-1, 2 * self.config.dim_embedding, 2]
            )
            embeddings_dsts = torch.nn.functional.gumbel_softmax(
                params_dsts,
                tau=self.tau,  # torch.clamp(self.tau, 0.1, 11.),
                hard=not self.training,
                dim=-1
            )[:, :, 0]
            output = embeddings_dsts
            scores = torch.tensor([1.])
        # Since queries have second dimension of one its possible to just remove
        # this dimension and get a (BS, num_out) output.
        # fc_in = torch.squeeze(output, dim=1)
        output = self.logits(self.fcn_out(output))
        return output, scores


if __name__ == '__main__':
    import numpy as np
    import networkx as nx
    from torch.utils.data import DataLoader
    import dataprep.sp_prep as dprep
    from dataprep.datasets import DistributionalSpfDataSet
    from models.utils import full_cross_entropy
    from embeddings.learned import hamming_distance
    import present

    DEV = 'cpu'

    graph = dprep.add_index_to_nodes(nx.read_graphml("/opt/project/data/dfn.graphml"))
    data = dprep.distributional_spf_dataset(graph)

    config = SpfConfig(
        max_degree=np.max(list(dict(nx.degree(graph)).values())),
        dim_embedding=10,
        num_heads=3,
        dim_attn_hidden=10,
        dim_attn_out=10,
        dim_out_fcn=[10, 10],
        num_nodes=graph.number_of_nodes()
    )

    targets = []
    masks = []
    neighbors = []
    destinations = []
    for sample in data:
        targets.append(sample[dprep.H5_TARGET].reshape(1, -1))
        masks.append(sample[dprep.H5_MASK].reshape(1, -1, 1))
        destinations.append(np.array([sample[dprep.H5_DESTINATION]]).reshape(1, 1, 1))
        nb = np.zeros(config.max_degree, dtype=np.int32)
        nb[:len(sample[dprep.H5_NEIGHBORS])] = sample[dprep.H5_NEIGHBORS]
        neighbors.append(nb.reshape(1, -1, 1))

    dataset = DistributionalSpfDataSet(
        neighbors=np.concatenate(neighbors),
        destinations=np.concatenate(destinations),
        attention_masks=np.concatenate(masks),
        targets=np.concatenate(targets)
    )
    loader_train = DataLoader(dataset, batch_size=512, shuffle=True)
    model = EmbeddingSpfModel(config)
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=1e-3
    )

    for epoch in range(10000):
        model.train()
        count = 0.
        epoch_loss = 0.
        for batch, sample in enumerate(loader_train):
            count += 1
            pred, scores = model(
                queries=sample['destination'].to(DEV),
                others=sample['neighbors'].to(DEV),
                mask=sample['attention_mask'].to(DEV)
            )
            loss = full_cross_entropy(pred, sample['target'].to(DEV))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            model._step_tau()

            epoch_loss += loss
        print(epoch, epoch_loss / count)

    embedding = torch.nn.functional.gumbel_softmax(model.params_embeddings, hard=True, dim=-1)[:, :, 0]

    d_H = hamming_distance(torch.unsqueeze(embedding, 0))[0, :, :]
    present.plot_graph(graph, {u: d['idx'] for u, d in graph.nodes(data=True)}, d_H.cpu().detach().numpy())
