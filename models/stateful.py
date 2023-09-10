"""
Implements Neural Network models that process the full state of the network.
"""
import torch
from typing import List, Tuple, Dict, Any, Union
import layers.attn as attn
import models.utils as mutils


class GumbelSoftmaxConfig(object):
    """
    Configures the gumbel softmax function.

    Attributes:
        temperature: The temperature to use when sampling values.
        arity: The dimension of the random variable. Is in {2, 3, ...}. Only
            one of the entries in the tuple is set to one.
        num_blocks: How many n-ary variables constitute the corresponding layers.
            The layer then has num_blocks ones and arity * num_blocks - num_blocks
            zeros.
    """

    @classmethod
    def from_dict(cls, d) -> 'GumbelSoftmaxConfig':
        return GumbelSoftmaxConfig(**d)

    def __init__(self, temperature: float, arity: int, num_blocks: int):
        self.temperature = temperature
        self.arity = arity
        self.num_blocks = num_blocks

    def to_dict(self) -> Dict[str, Union[float, int]]:
        return {
            "temperature": self.temperature,
            "arity": self.arity,
            "num_blocks": self.num_blocks
        }


class StatefulConfig(object):
    """
    Class representing the configuration for a neural network that processes
    the network wide state.
    """
    @classmethod
    def from_dict(cls, dc: Dict[str, Any]) -> 'StatefulConfig':
        hlsa_gs = dc.get('hlsa_gs', None)
        if hlsa_gs is not None:
            hlsa_gs = GumbelSoftmaxConfig.from_dict(hlsa_gs)
        return cls(
            link_attns=[attn.MultiHeadAttentionModuleConfig(**x) for x in dc['link_attns']],
            hlsas_attn=attn.MultiHeadAttentionModuleConfig(**dc['hlsa_attns']),
            neighbor_attns=attn.MultiHeadAttentionModuleConfig(**dc['neighbor_attns']),
            final_fcns=dc['final_fcns'],
            max_degree=dc['max_degree'],
            dim_embedding=dc['dim_embedding'],
            pool_links=dc['pool_links'],
            packets_droppeable=dc["packets_droppeable"],
            num_nodes=dc['num_nodes'],
            num_nodes_with_state=dc['num_nodes_with_state'],
            hlsa_attn_key=dc.get('hlsa_attn_key', 'current_loc'),
            multiclass=dc.get('multiclass', False),
            cur_loc_and_dst_q_hlsa=dc.get('cur_loc_and_dst_q_hlsa', False),
            # hlsa_model=dc.get('hlsa_model', None),
            hlsa_model=dc['hlsa_model'],
            hlsa_gs=hlsa_gs,
            alpha_l1_hlsa_attn_weights=dc.get('alpha_l1_hlsa_attn_weights', 0.),
            alpha_l1_hlsas=dc.get('alpha_l1_hlsas', 0.),
            neighbor_model=dc.get("neighbor_model", "attn"),
            policy=dc.get("policy", None)
        )

    def from_json(self, path: str) -> 'StatefulConfig':
        pass

    def __init__(self, link_attns: List[attn.MultiHeadAttentionModuleConfig],
                 hlsas_attn: attn.MultiHeadAttentionModuleConfig,
                 neighbor_attns: attn.MultiHeadAttentionModuleConfig,
                 final_fcns: List[int], dim_embedding: int, max_degree: int,
                 pool_links: str, hlsa_model: str, policy: str,
                 packets_droppeable: bool, alpha_l1_hlsa_attn_weights: float,
                 alpha_l1_hlsas: float, neighbor_model: str, num_nodes=None,
                 num_nodes_with_state=None, hlsa_attn_key='current_loc', multiclass=False,
                 cur_loc_and_dst_q_hlsa=False, hlsa_gs: GumbelSoftmaxConfig=None):
        """

        Args:
            link_attns: Configurations for attention modules that are used in the
                process of producing the HLSAs.
            hlsas_attn: Configuration for one attention module that attends over
                the HLSAs of each node in the network.
            neighbor_attns: Attention mechanism combining the embeddings of the
                direct neighbors of each node.
            final_fcns: Layer sizes for a sequence of dense layers that takes
                a concatenation of the output of the neighbor fcns and the
                hlsas_attn and produce the final output.
            dim_embedding: Dimensionality of node embeddings.
            max_degree: Maximum degree in the network.
            pool_links: Pooling function applied to the output of the attention
                module attending to the node incident links. Must be in
                {average, max, sum}.
            hlsa_model: Identifies model for the computation of HLSAs. Possible
                values are {None, fcn, attn, gs}.
            policy: The forwarding policy that is learned. Its metadata and not
                used during the construction of the model.
            packets_droppeable: Switch can decide to drop a packet. If set to
                true add an additional output neuron indicating dropping.
            alpha_l1_hlsa_attn_weights: Add a L1 regularizer on the attention
                scores to the loss function (intention: enforce sparse patterns).
            alpha_l1_hlsas: Add L1 regularizer on the HLSAs to the loss function.
                Intention is to enforce sparse HLSA messages.
            neighbor_model: String that identifies how the neighbor information
                is aggregated. Can be in {fcn, attn}.
            num_nodes: Number of nodes in the graph.
            num_nodes_with_state: Number of nodes that are represented in the
                state array of the network.
            hlsa_attn_key: The data to comput the forward pass is passed as dict.
                This argument is the name of the key with which the query for the
                HLSA attention is retrieved.
            multiclass: Boolean that indicates distributional or multiclass setting.
                If True, then the classification task is treated as multi-class
                classification, i.e., multiple outputs can be one. If set to False,
                then a distributional output is assumed and a softmax activation
                is applied to the output.
            cur_loc_and_dst_q_hlsa: Boolean indicating whether embedding of current
                location and destination should be concatenated and used as
                queries for the HLSA attention module. When False (default)
                current location (or whats passed as it) is used instead.
            hlsa_gs: Configuration for the gumbel softmax layer that is applied
                to the output of the HLSA module to produce the HLSA messages.
        """
        self.link_attns = link_attns
        self.hlsas_attn = hlsas_attn
        self.neighbor_attns = neighbor_attns
        self.final_fcns = final_fcns
        self.dim_embedding = dim_embedding
        self.max_degree = max_degree
        self.pool_links = pool_links
        self.pool_neighbors = 'squeeze'
        self.pool_hlsas = 'squeeze'
        self.packets_droppeable = packets_droppeable
        self.num_nodes = num_nodes
        self.num_nodes_with_state = num_nodes_with_state
        self.hlsa_attn_key = hlsa_attn_key
        self.multiclass = multiclass
        self.cur_loc_and_dst_q_hlsa = cur_loc_and_dst_q_hlsa
        self.hlsa_model = hlsa_model
        self.hlsa_gs = hlsa_gs
        self.alpha_l1_hlsa_attn_weights = alpha_l1_hlsa_attn_weights
        self.alpha_l1_hlsas = alpha_l1_hlsas
        self.neighbor_model = neighbor_model
        self.policy = policy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "link_attns": [c.to_dict() for c in self.link_attns],
            "hlsas_attn": self.hlsas_attn.to_dict(),
            "neighbor_attns": self.neighbor_attns.to_dict(),
            "final_fcns": self.final_fcns,
            'dim_embedding': self.dim_embedding,
            'max_degree': self.max_degree,
            "pool_links": self.pool_links,
            "pool_neighbors": self.pool_neighbors,
            "pool_hlsas": self.pool_hlsas,
            "packets_droppeable": self.packets_droppeable,
            "num_nodes": self.num_nodes,
            "num_nodes_with_state": self.num_nodes_with_state,
            "hlsa_attn_key": self.hlsa_attn_key,
            "multiclass": self.multiclass,
            "cur_loc_and_dst_q_hlsa": self.cur_loc_and_dst_q_hlsa,
            "hlsa_model": self.hlsa_model,
            "hlsa_gs": self.hlsa_gs.to_dict(),
            "alpha_l1_hlsa_attn_weights": self.alpha_l1_hlsa_attn_weights,
            "alpha_l1_hlsas": self.alpha_l1_hlsas,
            "neighbor_model": self.neighbor_model,
            "policy": self.policy
        }


class FailureWeightSensorModelConfig(object):
    """
    Class representing the configuration for a neural network that processes
    the network wide state.
    """
    @classmethod
    def from_dict(cls, dc: Dict[str, Any]) -> 'FailureWeightSensorModelConfig':
        hlsa_gs = dc.get('hlsa_gs', None)
        if hlsa_gs is not None:
            hlsa_gs = GumbelSoftmaxConfig.from_dict(hlsa_gs)
        return cls(
            link_attns=[attn.MultiHeadAttentionModuleConfig(**x) for x in dc['link_attns']],
            hlsas_attn=attn.MultiHeadAttentionModuleConfig(**dc['hlsas_attn']),
            neighbor_attns=attn.MultiHeadAttentionModuleConfig(**dc['neighbor_attns']),
            link_attns_weight=[attn.MultiHeadAttentionModuleConfig(**x) for x in dc['link_attns_weight']],
            final_fcns=dc['final_fcns'],
            max_degree=dc['max_degree'],
            dim_embedding=dc['dim_embedding'],
            pool_links=dc['pool_links'],
            packets_droppeable=dc["packets_droppeable"],
            num_nodes=dc['num_nodes'],
            num_nodes_with_state=dc['num_nodes_with_state'],
            hlsa_attn_key=dc.get('hlsa_attn_key', 'current_loc'),
            multiclass=dc.get('multiclass', False),
            cur_loc_and_dst_q_hlsa=dc.get('cur_loc_and_dst_q_hlsa', False),
            hlsa_model=dc['hlsa_model'],
            hlsa_gs=hlsa_gs,
            alpha_l1_hlsa_attn_weights=dc.get('alpha_l1_hlsa_attn_weights', 0.),
            alpha_l1_hlsas=dc.get('alpha_l1_hlsas', 0.),
            neighbor_model=dc.get("neighbor_model", "attn"),
            hlsas_weight_attn=attn.MultiHeadAttentionModuleConfig(**dc["hlsas_weight_attn"]),
            hlsa_weight_gs=GumbelSoftmaxConfig.from_dict(dc['hlsa_weight_gs']),
            hlsa_weight_model=dc.get("hlsa_weight_model", "attn"),
            pool_links_weight=dc['pool_links_weight']
        )

    def from_json(self, path: str) -> 'FailureWeightSensorModelConfig':
        pass

    def __init__(self, link_attns: List[attn.MultiHeadAttentionModuleConfig],
                 hlsas_attn: attn.MultiHeadAttentionModuleConfig,
                 hlsas_weight_attn: attn.MultiHeadAttentionModuleConfig,
                 link_attns_weight: List[attn.MultiHeadAttentionModuleConfig],
                 neighbor_attns: attn.MultiHeadAttentionModuleConfig,
                 final_fcns: List[int], dim_embedding: int, max_degree: int,
                 pool_links: str, pool_links_weight: str, packets_droppeable: bool,
                 hlsa_model: str, hlsa_weight_model: str, alpha_l1_hlsa_attn_weights: float,
                 alpha_l1_hlsas: float, neighbor_model: str, num_nodes=None,
                 num_nodes_with_state=None, hlsa_attn_key='current_loc',
                 multiclass=False, cur_loc_and_dst_q_hlsa=False,
                 hlsa_gs: GumbelSoftmaxConfig=None,
                 hlsa_weight_gs: GumbelSoftmaxConfig=None):
        """

        Args:
            link_attns: Configurations for attention modules that are used in the
                process of producing the HLSAs.
            graph_attn: One Single-head attention module that combines the
                HLSAs of each router.
            neighbor_attns: Attention mechanism combining the embeddings of the
                direct neighbors of each node.
            final_fcns: Layer sizes for a sequence of dense layers that takes
                a concatenation of the output of the neighbor fcns and the
                graph_fcns and produce the final output.
            dim_embedding: Dimensionality of node embeddings.
            max_degree: Maximum degree in the network.
            pool_links: Pooling function applied to the output of the attention
                module attending to the node incident links. Must be in
                {average, max, sum}.
            packets_droppeable: Switch can decide to drop a packet. If set to
                true add an additional output neuron indicating dropping.
            hlsa_model: Identifies model for the computation of HLSAs. Possible
                values are {None, fcn, attn}.
            num_nodes_with_state: Number of nodes that are represented in the
                state array of the network.
            num_nodes: Number of nodes in the graph.
            cur_loc_and_dst_q_hlsa: Boolean indicating whether embedding of current
                location and destination should be concatenated and used as
                queries for the HLSA attention module. When False (default)
                current location (or whats passed as it) is used instead.
        """
        self.link_attns = link_attns
        self.hlsas_attn = hlsas_attn
        self.neighbor_attns = neighbor_attns
        self.final_fcns = final_fcns
        self.dim_embedding = dim_embedding
        self.max_degree = max_degree
        self.pool_links = pool_links
        self.pool_neighbors = 'squeeze'
        self.pool_hlsas = 'squeeze'
        self.packets_droppeable = packets_droppeable
        self.num_nodes = num_nodes
        self.num_nodes_with_state = num_nodes_with_state
        self.hlsa_attn_key = hlsa_attn_key
        self.multiclass = multiclass
        self.cur_loc_and_dst_q_hlsa = cur_loc_and_dst_q_hlsa
        self.hlsa_model = hlsa_model
        self.hlsa_gs = hlsa_gs
        self.alpha_l1_hlsa_attn_weights = alpha_l1_hlsa_attn_weights
        self.alpha_l1_hlsas = alpha_l1_hlsas
        self.neighbor_model = neighbor_model
        self.hlsas_weight_attn = hlsas_weight_attn
        self.pool_hlsas_weights = 'squeeze'
        self.hlsa_weight_gs = hlsa_weight_gs
        self.hlsa_weight_model = hlsa_weight_model
        self.link_attns_weight = link_attns_weight
        self.pool_links_weight = pool_links_weight

    def to_dict(self) -> Dict[str, Any]:
        return {
            "link_attns": [c.to_dict() for c in self.link_attns],
            "hlsas_attn": self.hlsas_attn.to_dict(),
            "neighbor_attns": self.neighbor_attns.to_dict(),
            "final_fcns": self.final_fcns,
            'dim_embedding': self.dim_embedding,
            'max_degree': self.max_degree,
            "pool_links": self.pool_links,
            "pool_neighbors": self.pool_neighbors,
            "pool_hlsas": self.pool_hlsas,
            "packets_droppeable": self.packets_droppeable,
            "num_nodes": self.num_nodes,
            "num_nodes_with_state": self.num_nodes_with_state,
            "hlsa_attn_key": self.hlsa_attn_key,
            "multiclass": self.multiclass,
            "cur_loc_and_dst_q_hlsa": self.cur_loc_and_dst_q_hlsa,
            "hlsa_model": self.hlsa_model,
            "hlsa_gs": self.hlsa_gs.to_dict(),
            "alpha_l1_hlsa_attn_weights": self.alpha_l1_hlsa_attn_weights,
            "alpha_l1_hlsas": self.alpha_l1_hlsas,
            "neighbor_model": self.neighbor_model,
            "pool_hlsas_weights": self.pool_hlsas_weights,
            "hlsa_weight_gs": self.hlsa_weight_gs.to_dict(),
            "hlsa_weight_model": self.hlsa_weight_model,
            "hlsas_weight_attn": self.hlsas_weight_attn.to_dict(),
            "link_attns_weight": [c.to_dict() for c in self.link_attns_weight],
            "pool_links_weight": self.pool_links_weight
        }


class StatefulModel(torch.nn.Module):
    """
    Implements a stateful neural network that uses the full state of the
    communication network to make forwarding decisions.

    NetworkState            Current Location          Neighbors    Destination
    N x V x a x b           N x 1 x d                 N x a x d    N x 1 x d
         |                    |                           |          |
    Multi-head attention      |                       Multi-head attention
    N x V x c                 |                       N x e
         |      |--------------                           |
    Single-Head attention                                 |
    N x f                                                 |
         |                                                |
         --------------------------------------------------
         |
    Output
    N x a
    """
    def __init__(self, config: StatefulConfig):
        """
        Initializes object.

        Args:
            config: Configuration for the module.
        """
        super(StatefulModel, self).__init__()
        self.config = config

        # The pooling functions remove the second last dimension, which is the
        # dimension of the objects being attended over.
        pool_factory = {
            'sum': lambda x: torch.sum(x, dim=-2),
            'max': lambda x: torch.max(x, dim=-2)[0],
            'average': lambda x: torch.mean(x, dim=-2),
            'squeeze': lambda x: torch.squeeze(x, dim=-2)
        }

        #     keys=network_state,
        #     queries=queries_links,
        #     values=network_state,
        #     attention_mask=network_state_mask
        if self.config.hlsa_model is None:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> NO HLSA")
            # Take the raw state as HLSAs. Reshape the last dimension.
            self.attend_over_links = lambda keys, queries, values, attention_mask: torch.reshape(
                values,
                [
                    values.shape[0],
                    values.shape[1],
                    values.shape[2] * values.shape[3]
                ]
            )
            self.attend_over_links_rs = self.attend_over_links
            self.pool_attended_links = lambda x: x
            self.pool_attended_links_rs = self.pool_attended_links
            # Ugly but necessary to load old models. Activate these two lines
            # for the driver to work on old models.
            # self.attend_over_links = mutils.make_sequential_attn_modules(config.link_attns)
            # self.pool_attended_links = pool_factory[config.pool_links]
        elif self.config.hlsa_model == 'attn':
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ATTN HLSA")
            self.attend_over_links = mutils.make_sequential_attn_modules(config.link_attns)
            self.pool_attended_links = pool_factory[config.pool_links]
        elif self.config.hlsa_model == 'fcn':
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> FCN HLSA")
            self._attend_over_links = mutils.make_sequential_from_attn_configs(self.config.link_attns)
            self.attend_over_links = lambda keys, queries, values, attention_mask: \
                self._attend_over_links(torch.reshape(
                    values, [
                        values.shape[0],
                        values.shape[1],
                        values.shape[2] * values.shape[3]
                    ]
                ))
            self.pool_attended_links = lambda x: x
        else:
            raise KeyError(
                "Unknown value {} for hlsa_model, must be in {{None, fcn, attn}}".format(
                    self.hlsa_model
                )
            )

        self.attend_over_hlsas = attn.AttentionModule(config.hlsas_attn)
        self.pool_attended_hlsas = pool_factory[config.pool_hlsas]

        if self.config.neighbor_model == 'attn':
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ATTN NEIGHBORS")
            self.attend_over_neighbors = attn.AttentionModule(config.neighbor_attns)
            self.pool_attended_neighbors = pool_factory[config.pool_neighbors]
        elif self.config.neighbor_model == 'fcn':
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> FCN NEIGHBORS")
            self.attend_over_neighbors = mutils.make_sequential(
                layer_sizes=[config.neighbor_attns.dim_fcn],
                dim_in=config.neighbor_attns.dim_in,
                activation='relu'
            )
        else:
            raise KeyError("Unknown model name {} for neighbors.".format(config.neighbor_model))

        self.process_embedding = mutils.make_sequential(
            layer_sizes=config.final_fcns,
            activation='relu',
            dim_in=config.hlsas_attn.dim_fcn + config.neighbor_attns.dim_fcn
        )
        self.make_logits = torch.nn.Linear(
            config.final_fcns[-1],
            out_features=config.max_degree + 1 if self.config.packets_droppeable else 0,
            bias=True
        )
        self.make_logits_ecmp = torch.nn.Linear(
            config.final_fcns[-1],
            out_features=config.max_degree + 1 if self.config.packets_droppeable else 0,
            bias=True
        )

    def l1_loss(self):
        # loss = self.config.alpha_l1_hlsas * torch.mean(torch.sum(torch.sum(self.hlsas_binary, dim=-1), dim=-1))
        converged = None
        for act in self.attend_over_hlsas.last_scores:
            converged = act if converged is None else converged + act
            # loss += self.config.alpha_l1_hlsa_attn_weights * torch.mean(torch.sum(torch.sum(act, dim=-1), dim=-1))
        act = act - 0.00001
        act = torch.relu(torch.tanh(act * 10000))
        loss = torch.mean(self.config.alpha_l1_hlsa_attn_weights * act)
        return loss

    def sample_hlsa_activations(self, hlsas):
        hlsas_binary = torch.nn.functional.gumbel_softmax(
            logits=torch.reshape(
                hlsas,
                [
                    -1,
                    self.config.num_nodes_with_state,
                    self.config.hlsa_gs.num_blocks,
                    self.config.hlsa_gs.arity
                ]
            ),
            tau=self.config.hlsa_gs.temperature,
            hard=not self.training
        )
        hlsas_binary = torch.reshape(
            hlsas_binary,
            [
                -1,
                self.config.num_nodes_with_state,
                self.config.hlsa_gs.num_blocks * self.config.hlsa_gs.arity
            ]
        )
        return hlsas_binary

    def reduce_tau(self, factor=1e-3):
        self.attend_over_hlsas.reduce_tau(factor)

    def compute_hlsas(self, network_state: torch.Tensor, network_state_mask: torch.Tensor,
                      embd_nodes_state: torch.Tensor=None) -> torch.Tensor:
        # If the embeddings of nodes with states are None use the network
        # state as queries instead.
        if embd_nodes_state is None:
            queries_links = network_state
        else:
            queries_links = embd_nodes_state

        if self.config.hlsa_model is None:
            # Ugly but necessary to load old models.
            hlsas = self.attend_over_links_rs(
                keys=network_state,
                queries=queries_links,
                values=network_state,
                attention_mask=network_state_mask
            )
            hlsas = self.pool_attended_links_rs(hlsas)
        else:
            hlsas = self.attend_over_links(
                keys=network_state,
                queries=queries_links,
                values=network_state,
                attention_mask=network_state_mask
            )
            hlsas = self.pool_attended_links(hlsas)

        if self.config.hlsa_gs is None:
            return torch.sigmoid(hlsas)
        else:
            return self.sample_hlsa_activations(hlsas)

    def forward(self, network_state: torch.Tensor, network_state_mask: torch.Tensor,
                embeddings_neighbors: torch.Tensor, mask_embeddings: torch.Tensor,
                embd_current_location: torch.Tensor, embd_destination: torch.Tensor,
                embeddings: torch.Tensor, embd_nodes_state: torch.Tensor=None,
                hlsa_attn_head_activations: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the logits.

        - NNS: Number of nodes with state.

        Args:
            network_state: Representation of all links in the network grouped
                by node. Has shape (BS, NNS, max_degree, d).
            network_state_mask: Mask for attention layer attending over the
                network state. Indicates actual neighbors. Has
                shape (BS, NNS, max_degree, 1).
            embeddings_neighbors: The embeddings of the direct neighbors of a
                node. Has shape (BS, max_degree, D).
            mask_embeddings: Mask for the attention module attending over the
                neighbors of a node. Zeroes out non-existing neighbors for nodes
                having less than the maximum degree. Has Shape (BS, max_degree, 1).
            embd_current_location: Embedding of the current switch. The embedding
                is used as query for the attention layer attending over the
                HLSAs. Has shape (BS, D).
            embd_destination: Embedding of the destination node. The embedding
                is used as query for the attention over the embeddings of the
                neighbors. Has shape (BS, D).
            embeddings: Embeddings of all nodes in the graph. Used as keys for
                the attention mechanism attending over the hidden link state
                advertisements. The order of the embeddings must be the same
                as the one of the HLSAs. Has shape (V, D).
            embd_nodes_state: Embeddings for the nodes for which a state is
                calculated. Used as queries for the attention over links if
                set. Has shape (BS, NNS, 1, D).

        Returns:
            Returns the logits for the output layer.
        """
        self.hlsas_binary = self.compute_hlsas(
            network_state=network_state,
            network_state_mask=network_state_mask,
            embd_nodes_state=embd_nodes_state
        )
        if self.config.cur_loc_and_dst_q_hlsa:
            q_attend_hlsas = torch.cat(
                [
                    torch.unsqueeze(embd_current_location, dim=1),
                    torch.unsqueeze(embd_destination, dim=1)
                ],
                dim=-1
            )
        else:
            q_attend_hlsas = torch.unsqueeze(embd_current_location, dim=1)

        rep1 = self.attend_over_hlsas(
            keys=embeddings,
            queries=q_attend_hlsas,
            values=self.hlsas_binary,
            weights=hlsa_attn_head_activations
        )
        # print("rep1 is ", rep1.shape)
        rep1 = self.pool_attended_hlsas(rep1)
        # print("rep1 after pooling is ", rep1.shape)
        if self.config.neighbor_model == "attn":
            rep2 = self.attend_over_neighbors(
                keys=embeddings_neighbors,
                queries=torch.unsqueeze(embd_destination, dim=1),
                values=embeddings_neighbors,
                attention_mask=mask_embeddings
            )
            # print("rep2 is ", rep2.shape)
            rep2 = self.pool_attended_neighbors(rep2)
        else:
            rep2 = self.attend_over_neighbors(torch.cat(
                [embd_destination, embd_current_location],
                dim=1
            ))
        # print("rep2 after pooling is ", rep2.shape)

        # print("shapes are: ", rep1.shape, rep2.shape)
        rep = torch.cat([rep1, rep2], dim=1)
        rep = self.process_embedding(rep)
        out = self.make_logits(rep)
        out_ecmp = self.make_logits_ecmp(rep)
        return out, out_ecmp, self.l1_loss()


class HlsaModel(StatefulModel):
    """
    Module used to export the convertion of local link state into a HNSA
    to a torchscript.
    """
    def __init__(self, config: StatefulConfig):
        super(HlsaModel, self).__init__(config)

    def forward(self, network_state: torch.Tensor, network_state_mask: torch.Tensor,
                embd_nodes_state: torch.Tensor=None) -> torch.Tensor:
        if embd_nodes_state is None:
            queries_links = network_state
        else:
            queries_links = embd_nodes_state
        # print("{:10s} {}".format("network_state", network_state.shape))
        # print("{:10s} {}".format("network_state_mask", network_state_mask.shape))
        # print("{:10s} {}".format("queries_links", queries_links.shape))
        # print("{:10s} {}".format("embeddings_neighbors", embeddings_neighbors.shape))
        # print("{:10s} {}".format("mask_embeddings", mask_embeddings.shape))
        # print("{:10s} {}".format("embd_current_location", embd_current_location.shape))
        # print("{:10s} {}".format("embd_destination", embd_destination.shape))
        # print("{:10s} {}".format("embeddings", embeddings.shape))
        if self.config.hlsa_model is None:
            # Ugly but necessary to load old models.
            hlsas = self.attend_over_links_rs(
                keys=network_state,
                queries=queries_links,
                values=network_state,
                attention_mask=network_state_mask
            )
            # print("hlsas is ", hlsas.shape)
            hlsas = self.pool_attended_links_rs(hlsas)
            # print("hlsas after pooling with {} is ".format(self.config.pool_links), hlsas.shape)
        else:
            hlsas = self.attend_over_links(
                keys=network_state,
                queries=queries_links,
                values=network_state,
                attention_mask=network_state_mask
            )
            # print("hlsas is ", hlsas.shape)
            hlsas = self.pool_attended_links(hlsas)
            # print("hlsas after pooling with {} is ".format(self.config.pool_links), hlsas.shape)

        if self.config.hlsa_gs is None:
            # self.hlsas_binary = hlsas
            self.hlsas_binary = torch.sigmoid(hlsas)
        else:
            self.hlsas_binary = self.sample_hlsa_activations(hlsas)
        return self.hlsas_binary


class ForwardingModel(StatefulModel):
    """
    Model that makes forwarding decisions based on existing HLSAs and the
    other info. Used to export model into torchscript.
    """
    def __init__(self, config: StatefulConfig):
        super(ForwardingModel, self).__init__(config)

    def forward(self, hlsas_binary: torch.Tensor, embeddings_neighbors: torch.Tensor,
                mask_embeddings: torch.Tensor, embd_current_location: torch.Tensor,
                embd_destination: torch.Tensor, embeddings: torch.Tensor,
                hlsa_attn_head_activations: torch.Tensor=None) -> torch.Tensor:
        """
        Computes the logits.

        - NNS: Number of nodes with state.

        Args:
            network_state: Representation of all links in the network grouped
                by node. Has shape (BS, NNS, max_degree, d).
            network_state_mask: Mask for attention layer attending over the
                network state. Indicates actual neighbors. Has
                shape (BS, NNS, max_degree, 1).
            embeddings_neighbors: The embeddings of the direct neighbors of a
                node. Has shape (BS, max_degree, D).
            mask_embeddings: Mask for the attention module attending over the
                neighbors of a node. Zeroes out non-existing neighbors for nodes
                having less than the maximum degree. Has Shape (BS, max_degree, 1).
            embd_current_location: Embedding of the current switch. The embedding
                is used as query for the attention layer attending over the
                HLSAs. Has shape (BS, D).
            embd_destination: Embedding of the destination node. The embedding
                is used as query for the attention over the embeddings of the
                neighbors. Has shape (BS, D).
            embeddings: Embeddings of all nodes in the graph. Used as keys for
                the attention mechanism attending over the hidden link state
                advertisements. The order of the embeddings must be the same
                as the one of the HLSAs. Has shape (V, D).
            embd_nodes_state: Embeddings for the nodes for which a state is
                calculated. Used as queries for the attention over links if
                set. Has shape (BS, NNS, 1, D).

        Returns:
            Returns the logits for the output layer.
        """
        # print("{:10s} {}".format("network_state", network_state.shape))
        # print("{:10s} {}".format("network_state_mask", network_state_mask.shape))
        # print("{:10s} {}".format("embeddings_neighbors", embeddings_neighbors.shape))
        # print("{:10s} {}".format("mask_embeddings", mask_embeddings.shape))
        # print("{:10s} {}".format("embd_current_location", embd_current_location.shape))
        # print("{:10s} {}".format("embd_destination", embd_destination.shape))
        # print("{:10s} {}".format("embeddings", embeddings.shape))
        if self.config.cur_loc_and_dst_q_hlsa:
            q_attend_hlsas = torch.cat(
                [
                    torch.unsqueeze(embd_current_location, dim=1),
                    torch.unsqueeze(embd_destination, dim=1)
                ],
                dim=-1
            )
        else:
            q_attend_hlsas = torch.unsqueeze(embd_current_location, dim=1)

        rep1 = self.attend_over_hlsas(
            keys=embeddings,
            queries=q_attend_hlsas,
            values=hlsas_binary,
            weights=hlsa_attn_head_activations
        )
        # print("rep1 is ", rep1.shape)
        rep1 = self.pool_attended_hlsas(rep1)
        # print("rep1 after pooling is ", rep1.shape)
        if self.config.neighbor_model == "attn":
            rep2 = self.attend_over_neighbors(
                keys=embeddings_neighbors,
                queries=torch.unsqueeze(embd_destination, dim=1),
                values=embeddings_neighbors,
                attention_mask=mask_embeddings
            )
            # print("rep2 is ", rep2.shape)
            rep2 = self.pool_attended_neighbors(rep2)
        else:
            rep2 = self.attend_over_neighbors(torch.cat(
                [embd_destination, embd_current_location],
                dim=1
            ))
        # print("rep2 after pooling is ", rep2.shape)

        # print("shapes are: ", rep1.shape, rep2.shape)
        rep = torch.cat([rep1, rep2], dim=1)
        rep = self.process_embedding(rep)
        out = self.make_logits(rep)
        out_ecmp = self.make_logits_ecmp(rep)
        return torch.cat([out, out_ecmp], dim=-1)


class FailureWeightSensorModel(torch.nn.Module):
    """
    Similar to the stateful model, however, a separate attention mechanism is
    used for the weight of the graph and the availability information.
    """
    def __init__(self, config: FailureWeightSensorModelConfig):
        """
        Initializes object.

        Args:
            config: Configuration for the module.
        """
        super(FailureWeightSensorModel, self).__init__()
        self.config = config

        # The pooling functions remove the second last dimension, which is the
        # dimension of the objects being attended over.
        pool_factory = {
            'sum': lambda x: torch.sum(x, dim=-2),
            'max': lambda x: torch.max(x, dim=-2)[0],
            'average': lambda x: torch.mean(x, dim=-2),
            'squeeze': lambda x: torch.squeeze(x, dim=-2)
        }

        #     keys=network_state,
        #     queries=queries_links,
        #     values=network_state,
        #     attention_mask=network_state_mask
        if self.config.hlsa_model is None:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> NO HLSA LF")
            # Take the raw state as HLSAs. Reshape the last dimension.
            self.attend_over_links = lambda keys, queries, values, attention_mask: torch.reshape(
                values,
                [
                    values.shape[0],
                    values.shape[1],
                    values.shape[2] * values.shape[3]
                ]
            )
            self.attend_over_links_rs = self.attend_over_links
            self.pool_attended_links = lambda x: x
            self.pool_attended_links_rs = self.pool_attended_links
            # Ugly but necessary to load old models. Activate these two lines
            # for the driver to work on old models.
            # self.attend_over_links = mutils.make_sequential_attn_modules(config.link_attns)
            # self.pool_attended_links = pool_factory[config.pool_links]
        elif self.config.hlsa_model == 'attn':
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ATTN HLSA")
            self.attend_over_links = mutils.make_sequential_attn_modules(config.link_attns)
            self.pool_attended_links = pool_factory[config.pool_links]
        elif self.config.hlsa_model == 'fcn':
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> FCN HLSA")
            self._attend_over_links = mutils.make_sequential_from_attn_configs(self.config.link_attns)
            self.attend_over_links = lambda keys, queries, values, attention_mask: \
                self._attend_over_links(torch.reshape(
                    values, [
                        values.shape[0],
                        values.shape[1],
                        values.shape[2] * values.shape[3]
                    ]
                ))
            self.pool_attended_links = lambda x: x
        else:
            raise KeyError(
                "Unknown value {} for hlsa_model, must be in {{None, fcn, attn}}".format(
                    self.hlsa_model
                )
            )

        if self.config.hlsa_weight_model is None:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> NO HLSA Weight")
            # Take the raw state as HLSAs. Reshape the last dimension.
            self.attend_over_links_w = lambda keys, queries, values, attention_mask: torch.reshape(
                values,
                [
                    values.shape[0],
                    values.shape[1],
                    values.shape[2] * values.shape[3]
                ]
            )
            self.attend_over_links_w_rs = self.attend_over_links
            self.pool_attended_links_w = lambda x: x
            self.pool_attended_links_w_rs = self.pool_attended_links_w
        elif self.config.hlsa_weight_model == 'attn':
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ATTN HLSA Weight")
            self.attend_over_links_w = mutils.make_sequential_attn_modules(config.link_attns_weight)
            self.pool_attended_links_w = pool_factory[config.pool_links_weight]
        elif self.config.hlsa_weight_model == 'fcn':
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> FCN HLSA")
            self._attend_over_links_w = mutils.make_sequential_from_attn_configs(self.config.link_attns_weight)
            self.attend_over_links_w = lambda keys, queries, values, attention_mask: \
                self._attend_over_links(torch.reshape(
                    values, [
                        values.shape[0],
                        values.shape[1],
                        values.shape[2] * values.shape[3]
                    ]
                ))
            self.pool_attended_links_w = lambda x: x
        else:
            raise KeyError(
                "Unknown value {} for hlsa_model, must be in {{None, fcn, attn}}".format(
                    self.config.hlsa_weight_model
                )
            )

        self.attend_over_hlsas = attn.AttentionModule(config.hlsas_attn)
        self.pool_attended_hlsas = pool_factory[config.pool_hlsas]

        self.attend_over_w_hlsas = attn.AttentionModule(config.hlsas_weight_attn)
        self.pool_attended_hlsas = pool_factory[config.pool_hlsas_weights]

        if self.config.neighbor_model == 'attn':
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ATTN NEIGHBORS")
            self.attend_over_neighbors = attn.AttentionModule(config.neighbor_attns)
            self.pool_attended_neighbors = pool_factory[config.pool_neighbors]
        elif self.config.neighbor_model == 'fcn':
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> FCN NEIGHBORS")
            self.attend_over_neighbors = mutils.make_sequential(
                layer_sizes=[config.neighbor_attns.dim_fcn],
                dim_in=config.neighbor_attns.dim_in,
                activation='relu'
            )
        else:
            raise KeyError("Unknown model name {} for neighbors.".format(config.neighbor_model))

        self.process_embedding = mutils.make_sequential(
            layer_sizes=config.final_fcns,
            activation='relu',
            dim_in=config.hlsas_attn.dim_fcn + config.neighbor_attns.dim_fcn + config.hlsas_weight_attn.dim_fcn
        )
        self.make_logits = torch.nn.Linear(
            config.final_fcns[-1],
            out_features=config.max_degree + 1 if self.config.packets_droppeable else 0,
            bias=True
        )

    def l1_loss(self):
        loss = self.config.alpha_l1_hlsas * torch.mean(torch.sum(torch.sum(self.hlsas_binary, dim=-1), dim=-1))
        for act in self.attend_over_hlsas.last_scores:
            loss += self.config.alpha_l1_hlsa_attn_weights * torch.mean(torch.sum(torch.sum(act, dim=-1), dim=-1))
        return loss

    def sample_hlsa_activations(self, hlsas: torch.Tensor, gs_conf: GumbelSoftmaxConfig):
        hlsas_binary = torch.nn.functional.gumbel_softmax(
            logits=torch.reshape(
                hlsas,
                [
                    -1,
                    self.config.num_nodes_with_state,
                    gs_conf.num_blocks,
                    gs_conf.arity
                ]
            ),
            tau=gs_conf.temperature,
            hard=not self.training
        )
        hlsas_binary = torch.reshape(
            hlsas_binary,
            [
                -1,
                self.config.num_nodes_with_state,
                gs_conf.num_blocks * gs_conf.arity
            ]
        )
        return hlsas_binary

    def forward(self, network_state: torch.Tensor, network_state_w: torch.Tensor,
                network_state_mask: torch.Tensor, embeddings_neighbors: torch.Tensor,
                mask_embeddings: torch.Tensor, embd_current_location: torch.Tensor,
                embd_destination: torch.Tensor, embeddings: torch.Tensor,
                embd_nodes_state: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the logits.

        - NNS: Number of nodes with state.

        Args:
            network_state: Representation of all links in the network grouped
                by node. Has shape (BS, NNS, max_degree, d).
            network_state_mask: Mask for attention layer attending over the
                network state. Indicates actual neighbors. Has
                shape (BS, NNS, max_degree, 1).
            embeddings_neighbors: The embeddings of the direct neighbors of a
                node. Has shape (BS, max_degree, D).
            mask_embeddings: Mask for the attention module attending over the
                neighbors of a node. Zeroes out non-existing neighbors for nodes
                having less than the maximum degree. Has Shape (BS, max_degree, 1).
            embd_current_location: Embedding of the current switch. The embedding
                is used as query for the attention layer attending over the
                HLSAs. Has shape (BS, D).
            embd_destination: Embedding of the destination node. The embedding
                is used as query for the attention over the embeddings of the
                neighbors. Has shape (BS, D).
            embeddings: Embeddings of all nodes in the graph. Used as keys for
                the attention mechanism attending over the hidden link state
                advertisements. The order of the embeddings must be the same
                as the one of the HLSAs. Has shape (V, D).
            embd_nodes_state: Embeddings for the nodes for which a state is
                calculated. Used as queries for the attention over links if
                set. Has shape (BS, NNS, 1, D).

        Returns:
            Returns the logits for the output layer.
        """
        # print("{:10s} {}".format("network_state", network_state.shape))
        # print("{:10s} {}".format("network_state_mask", network_state_mask.shape))
        # print("{:10s} {}".format("embeddings_neighbors", embeddings_neighbors.shape))
        # print("{:10s} {}".format("mask_embeddings", mask_embeddings.shape))
        # print("{:10s} {}".format("embd_current_location", embd_current_location.shape))
        # print("{:10s} {}".format("embd_destination", embd_destination.shape))
        # print("{:10s} {}".format("embeddings", embeddings.shape))
        queries_links = network_state if embd_nodes_state is None else embd_nodes_state
        # print("{:10s} {}".format("network_state", network_state.shape))
        # print("{:10s} {}".format("network_state_mask", network_state_mask.shape))
        # print("{:10s} {}".format("queries_links", queries_links.shape))
        # print("{:10s} {}".format("embeddings_neighbors", embeddings_neighbors.shape))
        # print("{:10s} {}".format("mask_embeddings", mask_embeddings.shape))
        # print("{:10s} {}".format("embd_current_location", embd_current_location.shape))
        # print("{:10s} {}".format("embd_destination", embd_destination.shape))
        # print("{:10s} {}".format("embeddings", embeddings.shape))
        if self.config.hlsa_model is None:
            # Ugly but necessary to load old models.
            hlsas = self.attend_over_links_rs(
                keys=network_state,
                queries=queries_links,
                values=network_state,
                attention_mask=network_state_mask
            )
            # print("hlsas is ", hlsas.shape)
            hlsas = self.pool_attended_links_rs(hlsas)
            # print("hlsas after pooling with {} is ".format(self.config.pool_links), hlsas.shape)
        else:
            hlsas = self.attend_over_links(
                keys=network_state,
                queries=queries_links,
                values=network_state,
                attention_mask=network_state_mask
            )
            # print("hlsas is ", hlsas.shape)
            hlsas = self.pool_attended_links(hlsas)
            # print("hlsas after pooling with {} is ".format(self.config.pool_links), hlsas.shape)
        hlsas_w = self.attend_over_links_w(
            keys=network_state_w,
            queries=network_state_w,
            values=network_state_w,
            attention_mask=network_state_mask
        )
        hlsas_w = self.pool_attended_links_w(hlsas_w)

        self.hlsas_binary = torch.sigmoid(hlsas) if self.config.hlsa_gs is None \
            else self.sample_hlsa_activations(hlsas, self.config.hlsa_gs)

        self.hlsas_w_binary = torch.sigmoid(hlsas_w) if self.config.hlsa_weight_gs is None \
            else self.sample_hlsa_activations(hlsas_w, self.config.hlsa_weight_gs)

        if self.config.cur_loc_and_dst_q_hlsa:
            q_attend_hlsas = torch.cat(
                [
                    torch.unsqueeze(embd_current_location, dim=1),
                    torch.unsqueeze(embd_destination, dim=1)
                ],
                dim=-1
            )
        else:
            q_attend_hlsas = torch.unsqueeze(embd_current_location, dim=1)

        rep1 = self.attend_over_hlsas(
            keys=embeddings,
            queries=q_attend_hlsas,
            values=self.hlsas_binary
        )
        # print("rep1 is ", rep1.shape)
        rep1 = self.pool_attended_hlsas(rep1)
        # print("rep1 after pooling is ", rep1.shape)

        rep3 = self.attend_over_w_hlsas(
            keys=embeddings,
            queries=q_attend_hlsas,
            values=self.hlsas_w_binary
        )
        rep3 = self.pool_attended_hlsas(rep3)

        if self.config.neighbor_model == "attn":
            rep2 = self.attend_over_neighbors(
                keys=embeddings_neighbors,
                queries=torch.unsqueeze(embd_destination, dim=1),
                values=embeddings_neighbors,
                attention_mask=mask_embeddings
            )
            # print("rep2 is ", rep2.shape)
            rep2 = self.pool_attended_neighbors(rep2)
        else:
            rep2 = self.attend_over_neighbors(torch.cat(
                [embd_destination, embd_current_location],
                dim=1
            ))
        # print("rep2 after pooling is ", rep2.shape)

        # print("shapes are: ", rep1.shape, rep2.shape)
        rep = torch.cat([rep1, rep2, rep3], dim=1)
        rep = self.process_embedding(rep)
        out = self.make_logits(rep)
        return out, self.l1_loss()


class EmbeddingStatefulModel(StatefulModel):
    """
    The overall architecture is the same as for the StatefulModel. Instead of
    pre-defined embeddings, this architecture learns the embeddings, though.
    """
    def __init__(self, config: StatefulConfig):
        """
        Initializes object.

        Args:
            config: Configuration for the module.
        """
        super(EmbeddingStatefulModel, self).__init__(config)
        assert config.num_nodes is not None
        self.params_embeddings = torch.nn.Parameter(
            data=torch.ones(config.num_nodes, config.dim_embedding, 2),
            requires_grad=True
        )
        self.tau = torch.tensor(10., requires_grad=True)
        self.tau_optim = torch.optim.Adam([self.tau], lr=1e-2)

    def _step_tau(self):
        self.tau.backward()
        self.tau_optim.step()
        self.tau_optim.zero_grad()

    def forward(self, network_state: torch.Tensor, network_state_mask: torch.Tensor,
                embeddings_neighbors: torch.Tensor, mask_embeddings: torch.Tensor,
                embd_current_location: torch.Tensor, embd_destination: torch.Tensor) -> torch.Tensor:
        """
        Computes the logits.

        Args:
            network_state: Representation of all links in the network grouped
                by node. Has shape (BS, V, max_degree, d).
            network_state_mask: Mask for attention layer attending over the
                network state. Indicates actual neighbors. Has
                shape (BS, V, max_degree, 1).
            embeddings_neighbors: The embeddings of the direct neighbors of a
                node. Has shape (BS, max_degree, D).
            mask_embeddings: Mask for the attention module attending over the
                neighbors of a node. Zeroes out non-existing neighbors for nodes
                having less than the maximum degree. Has Shape (BS, max_degree, 1).
            embd_current_location: Embedding of the current switch. The embedding
                is used as query for the attention layer attending over the
                HLSAs. Has shape (BS, D).
            embd_destination: Embedding of the destination node. The embedding
                is used as query for the attention over the embeddings of the
                neighbors. Has shape (BS, D).

        Returns:
            Returns the logits for the output layer.
        """
        embeddings_neighbors_ = torch.reshape(
            self.params_embeddings[torch.flatten(embeddings_neighbors).long(), :, :],
            [-1, self.config.max_degree, self.config.dim_embedding, 2]
        )
        embd_current_location_ = torch.reshape(
            self.params_embeddings[torch.flatten(embd_current_location).long(), :, :],
            [-1, self.config.dim_embedding, 2]
        )
        embd_destination_ = torch.reshape(
            self.params_embeddings[torch.flatten(embd_destination).long(), :, :],
            [-1, self.config.dim_embedding, 2]
        )

        embeddings_neighbors_ = torch.nn.functional.gumbel_softmax(
            embeddings_neighbors_,
            tau=torch.clamp(self.tau, 0.1, 11.),
            hard=not self.training,
            dim=-1
        )[:, :, :, 0]
        embd_current_location_ = torch.nn.functional.gumbel_softmax(
            embd_current_location_,
            tau=torch.clamp(self.tau, 0.1, 11.),
            hard=not self.training,
            dim=-1
        )[:, :, 0]
        embd_destination_ = torch.nn.functional.gumbel_softmax(
            embd_destination_,
            tau=torch.clamp(self.tau, 0.1, 11.),
            hard=not self.training,
            dim=-1
        )[:, :, 0]
        # print("network_state: ", type(network_state),
        #       "network_state_mask: ", type(network_state_mask),
        #       "embeddings_neighbors: ", type(embeddings_neighbors_),
        #       "mask_embeddings: ", type(mask_embeddings),
        #       "embd_current_location: ", type(embd_current_location_),
        #       "embd_destination: ", type(embd_destination_)
        # )
        return super(EmbeddingStatefulModel, self).forward(
            network_state=network_state,
            network_state_mask=network_state_mask,
            embeddings_neighbors=embeddings_neighbors_,
            mask_embeddings=mask_embeddings,
            embd_current_location=embd_current_location_,
            embd_destination=embd_destination_
        )
