"""
This module implements custom datasets.
"""
import h5py
import torch
import numpy as np
import os
import dataprep.input_output as inout
from torch.utils.data import Dataset
from dataprep.sp_prep import spf_dataset, expand_distributional_spf_dataset2, H5_CUR_LOC, H5_NEIGHBORS, H5_MASK, H5_DESTINATION, H5_TARGET
from typing import Dict, List, Tuple
import networkx as nx


class SpfDataSet(Dataset):
    """
    Dataset for shortest path data. One sample consists of three tensors:
        - Neighbor Encodings: (D, E),
        - Destination Encoding: (E)
        - Attention Mask: (D, 1),
        - Output: (D).
    Where D is the maximum degree in the graph and E the dimensionality of the
    embedding of nodes. All tensors have a first dimension corresponding to the
    batch size.
    """
    @classmethod
    def from_hdf5(cls, path_to_file: str) -> 'SpfDataSet':
        """
        Reads data stored in a HDF5 file and makes a data-set out of it. This
        method assumes that data is generated with the `_make_spf_dataset`
        function in the file `data_gen/spf_only_data.py`.

        Args:
            path_to_file:

        Returns:

        """
        file = h5py.File(path_to_file, 'r')
        dataset = cls(
            neighbors=file['values'][()],
            destinations=file['queries'][()],
            attention_masks=file['masks_v'][()],
            targets=file['zs'][()]
        )
        file.close()
        return dataset

    @classmethod
    def from_graph(cls, graph, include=None) -> 'SpfDataSet':
        if include is None:
            include = lambda: True
        hosts = [u for u in graph.nodes() if include(u)]
        queries, values, masks_v, zs, cur_locs = spf_dataset(graph, hosts)
        return cls(values, queries, masks_v, zs)

    def __init__(self, neighbors: np.array, destinations: np.array,
                 attention_masks: np.array, targets: np.array):
        super(SpfDataSet, self).__init__()
        assert neighbors.shape[0] == destinations.shape[0] == targets.shape[0] == attention_masks.shape[0],\
            "All inputs must have the same first dimensions."
        assert neighbors.shape[-1] == destinations.shape[-1], \
            "Destinations and neighbors must have the same last dimension."
        assert neighbors.ndim == 3, "Neighbors must have three dimensions."
        assert destinations.ndim == 3, "Destinations must have three dimensions."
        assert destinations.shape[1] == 1, "Destinations must have shape 1 as second dim."
        assert targets.ndim in [1, 2], "Targets must have two or one dimensions."
        assert attention_masks.shape[1] == neighbors.shape[1], \
            "Attention Masks and neighbors must have the same second dimension"
        self.neighbors = neighbors
        self.destinations = destinations
        self.attention_masks = attention_masks
        if targets.ndim == 2:
            self.targets = np.argmax(targets, axis=1)
        else:
            self.targets = targets

    def __len__(self) -> int:
        return self.targets.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return {
            'neighbors': self.neighbors[idx],
            'destination': self.destinations[idx],
            'attention_mask': self.attention_masks[idx],
            'target': self.targets[idx]
        }

    @property
    def max_degree(self) -> int:
        return self.neighbors.shape[1]

    @property
    def embedding_dim(self) -> int:
        return self.neighbors.shape[2]


class DistributionalSpfDataSet(Dataset):
    """
    Dataset for shortest path data. One sample consists of three tensors:
        - Neighbor Encodings: (D, E),
        - Destination Encoding: (E)
        - Attention Mask: (D, 1),
        - Output: (D).
    Where D is the maximum degree in the graph and E the dimensionality of the
    embedding of nodes. All tensors have a first dimension corresponding to the
    batch size.
    """
    @classmethod
    def from_hdf5(cls, path_to_embeddings: str, path_to_template: str, max_num=-1) -> 'DistributionalSpfDataSet':
        """
        Reads data stored in a HDF5 file and makes a data-set out of it. This
        method assumes that data is generated with the `_make_spf_dataset`
        function in the file `data_gen/spf_only_data.py`.

        Args:
            path_to_file:

        Returns:

        """
        data = expand_distributional_spf_dataset2(
            full_path_to_template=path_to_template,
            full_path_to_embeddings=path_to_embeddings,
        )
        dataset = cls(
            neighbors=data[H5_NEIGHBORS],
            destinations=data[H5_DESTINATION],
            attention_masks=data[H5_MASK],
            targets=data[H5_TARGET],
            current_locations=data[H5_CUR_LOC]
        )
        return dataset

    def __init__(self, neighbors: np.array, destinations: np.array,
                 attention_masks: np.array, targets: np.array,
                 current_locations: np.array):
        super(DistributionalSpfDataSet, self).__init__()
        assert neighbors.shape[0] == destinations.shape[0] == targets.shape[0] == attention_masks.shape[0], \
            "All inputs must have the same first dimensions."
        assert neighbors.shape[-1] == destinations.shape[-1], \
            "Destinations and neighbors must have the same last dimension."
        assert neighbors.ndim == 3, "Neighbors must have three dimensions."
        assert destinations.ndim == 3, "Destinations must have three dimensions."
        assert destinations.shape[1] == 1, "Destinations must have shape 1 as second dim."
        assert targets.ndim == 2, "Targets must have two dimensions."
        assert attention_masks.shape[1] == neighbors.shape[1], \
            "Attention Masks and neighbors must have the same second dimension"

        self.neighbors = neighbors
        self.destinations = destinations
        self.attention_masks = attention_masks
        self.targets = targets
        self.current_locations = current_locations

    def __len__(self) -> int:
        return self.targets.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return {
            'destination': self.destinations[idx, 0],
            'target': self.targets[idx],
            "attention_mask": np.expand_dims(self.attention_masks[idx], -1),
            "neighbors": self.neighbors[idx],
            # 'attention_mask': np.array([0], dtype=np.float32),
            # 'neighbors': np.array([0], dtype=np.float32),
            "cur_loc": self.current_locations[idx],
        }
        # return {
        #     'neighbors': self.neighbors[idx],
        #     'destination': self.destinations[idx],
        #     'attention_mask': self.attention_masks[idx],
        #     'target': self.targets[idx]
        # }

    @property
    def max_degree(self) -> int:
        return self.neighbors.shape[1]

    @property
    def embedding_dim(self) -> int:
        return self.neighbors.shape[2]


class DistributionalDataSetWithoutEmbedding(DistributionalSpfDataSet):
    """
    Dataset for shortest path data. One sample consists of three tensors:
        - Neighbor Encodings: (D, 1),
        - Destination Encoding: (1)
        - Attention Mask: (D, 1),
        - Output: (D).
    Where D is the maximum degree in the graph. Neighbors and destination
    encodings are represented with ints.
    """
    @classmethod
    def from_hdf5(cls, path_to_embeddings: str, path_to_template: str) -> 'DistributionalDataSetWithoutEmbedding':
        """
        Reads data stored in a HDF5 file and makes a data-set out of it. This
        method assumes that data is generated with the `_make_spf_dataset`
        function in the file `data_gen/spf_only_data.py`.

        Args:
            path_to_file:

        Returns:

        """
        f = h5py.File(path_to_template, 'r')
        dataset = DistributionalDataSetWithoutEmbedding(
            neighbors=np.expand_dims(f[H5_NEIGHBORS][()].astype(np.float32), -1),
            destinations=np.expand_dims(f[H5_DESTINATION][()].astype(np.float32), -1),
            attention_masks=f[H5_MASK][()].astype(np.float32),
            targets=f[H5_TARGET][()].astype(np.float32),
            current_locations=f[H5_CUR_LOC][()].astype(np.float32)
        )
        return dataset

    def __init__(self, neighbors: np.array, destinations: np.array,
                 attention_masks: np.array, targets: np.array, current_locations: np.array):
        super(DistributionalDataSetWithoutEmbedding, self).__init__(
            neighbors, destinations, attention_masks, targets, current_locations
        )


class StatefulDataset(Dataset):
    """
        Dataset for stateful forwarding data. One sample consists of six tensors:
            - Network State: (V, D, F).
            - Network State Mask: (V, D, 1).
            - Current Location encoding: (1, E).
            - Neighbors: (max_degree, E).
            - Destination encoding: (1, E).
            - Output (D (+ 1)?)
        Where D is the maximum degree in the graph, E the dimensionality of the
        embedding of nodes, V the number of nodes in the graph and F the number of features
        per edge. All tensors have an additional first dimension corresponding to the
        batch size.

        If instead of pre-computed embeddings learned embeddings should be used,
        then E = 1 and the returned values correspond to indexes that index into
        a tensor of embeddings that must be provided by the model.

        The size of the output can be D, i.e., the maximum degree of D + 1. In the
        latter case the network supports dropping of packets if no viable route is
        available.
    """
    @classmethod
    def load_dset(cls, full_path_templates: str, policy: str):
        data = inout.read_link_failure_data(full_path_templates)
        print(type(data['targets']))
        data["targets_ecmp"]     = data["targets"]["ecmp"].astype(np.float32)
        data["targets"]          = data["targets"][policy].astype(np.float32)
        data["network_states"]   = data["network_states"].astype(np.float32)
        data["cur_locs"]         = data["cur_locs"].astype(np.int32)
        data["destinations"]     = data["destinations"].astype(np.int32)
        data["all_masks"]        = data["all_masks"].astype(np.float32)
        data["all_neighbors"]    = data["all_neighbors"].astype(np.int32)
        data["nodes_with_state"] = data["nodes_with_state"].astype(np.int32)
        # data["targets"]          = data["targets"][indices].astype(np.float32)
        # data["network_states"]   = data["network_states"][indices].astype(np.float32)
        # data["cur_locs"]         = data["cur_locs"][indices].astype(np.int32)
        # data["destinations"]     = data["destinations"][indices].astype(np.int32)
        # data["all_masks"]        = data["all_masks"].astype(np.float32)
        # data["all_neighbors"]    = data["all_neighbors"].astype(np.int32)
        # data["nodes_with_state"] = data["nodes_with_state"].astype(np.int32)
        return data

    @classmethod
    def from_hdf5(cls, full_path_templates: str, policy: str, full_path_embeddings: str=None) -> 'StatefulDataset':
        data = cls.load_dset(full_path_templates, policy)
        if full_path_embeddings is None:
            data['embeddings'] = None
        else:
            data['embeddings'] = inout.read_embeddings(full_path_embeddings)
        return cls(**data)

    @classmethod
    def from_hdf5_files(cls, full_path_templates_dir: str, policy: str,
            full_path_embeddings: str=None, ignore_masks=True) -> 'StatefulDataset':
        """
            Create dataset from multiple hdf5 files. Read in the files one after
            the other and extend the existing dataset.
        """
        attn_keys = ['attn_activations_all', 'attn_activations_dst_tor', '']
        data = None
        break_at = 30 if len(os.listdir(full_path_templates_dir)) > 5 else 0
        for i, f in enumerate(os.listdir(full_path_templates_dir)):
            print(i, f)
            if not f.endswith('.h5'):
                continue
            else:
                tmp = cls.load_dset(os.path.join(full_path_templates_dir, f), policy)
                if data is None:
                    data = tmp
                else:
                    for k, v in data.items():
                        if k == 'nodes_with_state':
                            continue
                        elif k == 'all_neighbors':
                            continue
                        elif k == 'all_masks':
                            continue
                        elif ignore_masks and k in attn_keys:
                            continue
                        else:
                            data[k] = np.concatenate([v, tmp[k]])
            if i >= break_at:
                break
        if full_path_embeddings is None:
            data['embeddings'] = None
        else:
            data['embeddings'] = inout.read_embeddings(full_path_embeddings).astype(np.float32)
        return cls(**data)

    def __init__(self, network_states: np.array, all_masks: np.array,
                 cur_locs: np.array, destinations: np.array, all_neighbors: np.array,
                 targets: np.array, nodes_with_state: np.array, embeddings: np.array=None,
                 targets_ecmp=None, use_embeddings_as_queries_for_attention_over_links=False,
                 generate_weights=False, binarize_targets=False, attn_activations_all=None,
                 attn_activations_dst_tor=None, use_att_activations_dst_tor=None):
        """
        Initializes object. Note that the names of the arguments should match
        up with the keys in the dataset returned by the function in
        dataprep.link_failures.get_link_failure_datasets to pipe results through.

        NNL is the number of non-leaf-nodes in the graph.

        Args:
            network_states: All recorded network states. One network state is
                associated with each target. Has shape (N, NNL, D, F).
            all_masks: A static binary tensor indicating the neighborhood
                relationships between nodes. Has shape (V, D, F).
            embeddings: Binary matrix of node embeddings. Has shape (V, E).
            cur_locs: Indices of nodes. Used to index into the embeddings
                matrix. Has shape (N, 1).
            destinations: Indices of nodes. Used to index into the embeddings
                matrix. Represent the current destination node the packet should
                be forwarded to. Has shape (N, 1).
            all_neighbors: Indices that index into embeddings and represent the
                neighborhood relations between the nodes. Has shape (V, D, F).
            targets: The outputs that should be learned. Has shape (N, D).
            nodes_with_state: Array containing the indices of the nodes that
                are not leafes. Has shape (NNL,).
        """
        self.network_states = network_states
        self.network_masks = all_masks
        self.embeddings = embeddings
        self.current_locations = cur_locs
        self.destinations = destinations
        self.neighbohoods = all_neighbors
        self.targets = targets
        self.nodes_with_state = nodes_with_state
        self.generate_weights = generate_weights
        self.binarize_targets = binarize_targets
        self.use_embeddings_as_queries_for_attention_over_links = use_embeddings_as_queries_for_attention_over_links
        self.expand_state = False
        self.separate_weight_lf = False
        self.targets_ecmp = targets_ecmp
        self.attn_activations_all = attn_activations_all
        self.attn_activations_dst_tor = attn_activations_dst_tor
        self.use_att_activations_dst_tor = use_att_activations_dst_tor
        if self.embeddings is not None:
            print("{:20s} {}".format("embeddings",       str(self.embeddings.shape)))
        print("{:20s} {}".format("network_states",    str(self.network_states.shape)))
        print("{:20s} {}".format("network_masks",    str(self.network_masks.shape)))
        print("{:20s} {}".format("current_locations",str(self.current_locations.shape)))
        print("{:20s} {}".format("destinations",     str(self.destinations.shape)))
        print("{:20s} {}".format("neighbohoods",     str(self.neighbohoods.shape)))
        print("{:20s} {}".format("targets",          str(self.targets.shape)))
        print("{:20s} {}".format("nodes_with_state", str(self.nodes_with_state.shape)))
        print("{:20s} {}".format("binarize_targets", str(self.binarize_targets)))
        print("{:20s} {}".format(
            "use_embeddings_as_queries_for_attention_over_links",
            str(self.use_embeddings_as_queries_for_attention_over_links))
        )
        print("WARNING: Activation weights are statically set to NONE!")

    def load_weights(self, path):
        file = h5py.File(os.path.join(path, 'attn_activations_all.h5'), "r")
        self.attn_activations_all = file['activations'][()]
        file.close()

        # file = h5py.File(os.path.join(path, 'attn_dst_tor.h5'), 'r')
        # self.attn_activations_dst_tor = file['activations'][()]
        # self.use_att_activations_dst_tor = file['is_contained'][()]
        # file.close()

    def __len__(self) -> int:
        return self.targets.shape[0]

    @property
    def max_degree(self) -> int:
        return self.network_masks.shape[-2]

    @property
    def num_nodes_with_state(self) -> int:
        return self.nodes_with_state.size

    @property
    def num_nodes(self) -> int:
        return self.network_masks.shape[0]

    @property
    def embedding_dim(self) -> int:
        assert self.embeddings is not None, 'Embeddings is not set, dimensionality unknown.'
        return self.embeddings.shape[-1]

    @property
    def num_features_state(self) -> int:
        return self.network_states.shape[-1]

    def _get_weight(self, target) -> float:
        if target[0] > 0:
            return 6.561464518880614
        elif np.sum(target[1:] > 0.99) > 0:
            return 6.599353263380189
        elif np.sum(np.multiply(target > 0.124, target < 0.126)) > 0:
            return 9.79431929480901 
        elif np.sum(np.multiply(target > 0.24, target < 0.26)) > 0:
            return 7.706535141800248
        elif np.sum(np.multiply(target > 0.32, target < 0.34)) > 0:
            return 3.840172039707379
        elif np.sum(np.multiply(target > 0.49, target < 0.51)) > 0:
            return 97.8952520802741
        elif np.sum(np.multiply(target > 0.14, target < 0.15)) > 0:
            return 5.700766753128296
        elif np.sum(np.multiply(target > 0.16, target < 0.17)) > 0:
            return 58.13953488372093
        elif np.sum(np.multiply(target > 0.19, target < 0.21)) > 0:
            return 1030.9278350515463
        else:
            return 1.

    @classmethod
    def expand_edge_weights(cls, state) -> np.array:
        eye = np.expand_dims(np.eye(state.shape[1], state.shape[1], dtype=np.float32), axis=0)
        tmp = state[:, :, -1].flatten()
        mask = tmp == 0
        tmp[mask] = -1.
        # tmp[np.logical_not(mask)] += 10
        tmp = tmp.reshape([state.shape[0], state.shape[1], 1])
        append = np.multiply(eye, tmp)
        # state[:, :, -1] += 4
        # append = np.zeros((state.shape[0], state.shape[1], 10), dtype=np.float32)
        # edges = np.array([[[6.261467, 8.739021, 9.163736, 9.485467, 9.758730,
        #                     10.022729, 10.274696, 10.551240, 10.863578,
        #                     11.305934, 13.687495]]], dtype=np.float32)
        # indices = np.argmin(np.abs(np.subtract(edges, np.expand_dims(state[:, :, -1], axis=-1))), axis=-1)

        # append.reshape(-1, 10)[np.arange(state.shape[0] * state.shape[1]), indices.flatten()] = 1.
        state = np.concatenate((state[:, :, 0:4], append), axis=-1)
        return state

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            raise RuntimeError
            # idx = idx.tolist()
        # The the current locations for the batch.
        cur_locs_idx = self.current_locations[idx, 0]
        # Get the destinations for the batch.
        dsts_idx = self.destinations[idx, 0]
        # Get the mask. Use the indices as well since mask is static over time.
        neighbor_mask = self.network_masks[cur_locs_idx, :, :]
        # Get the corresponding neighborhoods to the current locations. Currently
        # are indices into the embeddings matrix.
        neighborhood = self.neighbohoods[cur_locs_idx, :, :]

        if self.embeddings is None:
            cur_locs = cur_locs_idx
            dsts = dsts_idx
        else:
            # Flatten the neighborhood index tensor and retrieve the embeddings
            # stored at the corresponding locations.
            tmp = self.embeddings[neighborhood.flatten(), :]
            # Reconstruct the correct shape and neighborhoods.
            neighborhood = tmp.reshape(self.max_degree, self.embedding_dim)
            # Non-existing neighbors have a value of -1 or 0. This retrieves a
            # valid embedding. Zero this embedding out using the mask.
            neighborhood = np.multiply(neighborhood, neighbor_mask)
            # Retrieve the embeddings for current locations and destinations.
            cur_locs = self.embeddings[cur_locs_idx, :]
            dsts = self.embeddings[dsts_idx, :]

        target = self.targets[idx, :].copy()
        target_ecmp = self.targets_ecmp[idx, :].copy()
        if self.binarize_targets:
            target[target > 1e-6] = 1.
        weight = self._get_weight(target) if self.generate_weights else 1.
        weight = np.array([weight], dtype=np.float32)

        state = self.network_states[idx, :, :, :].copy()
        state_last_dim = state.shape[-1]
        if self.expand_state:
            addendum = np.repeat(
                np.expand_dims(np.eye(state.shape[-2], state.shape[-2], dtype=np.float32), axis=0),
                repeats=state.shape[-3],
                axis=0
            )
            state = np.concatenate((state[..., :-1], addendum), axis=-1)
        if self.separate_weight_lf:
            state_w = state[..., state_last_dim - 1:]
            state_lf = state[..., :state_last_dim - 1]
        else:
            state_w = np.zeros([1], dtype=np.float32)
            state_lf = state
        # if state.shape[-1] == 5:
        #     state = self.expand_edge_weights(state)

        if self.use_embeddings_as_queries_for_attention_over_links:
            q_links = np.expand_dims(self.embeddings[self.nodes_with_state], axis=1).copy()
        else:
            q_links = state

        # if self.use_att_activations_dst_tor[cur_locs_idx] == 1:
        #     tmp = self.attn_activations_dst_tor[dsts_idx]
        # else:
        #     tmp = self.attn_activations_dst_tor[cur_locs_idx]
        if self.attn_activations_all is None:
            actis = None
        else:
            actis = np.concatenate(
                [self.attn_activations_all[dsts_idx], self.attn_activations_all[cur_locs_idx]],
                axis=0
            )
        # print("WARNING: Activation weights are statically set to NONE!")
        # Took the weights completly out of the returns statements because torch crashes else.
        # Also commented out the corresponding line in the trainable. Also in
        # eval/lf_eval line 1232, 672, 993
        actis = None

        return {
            'network_state': state_lf,
            'network_state_w': state_w,
            # Add an emtpy first dimention to the network masks. The masks are
            # the same for each state, thus re-use them. The masks get correctly
            # broadcasted along the first dimension in the attention modules.
            'network_masks': self.network_masks[self.nodes_with_state].copy(),
            'embd_nodes_state': q_links,
            'neighbors': neighborhood.copy(),
            'destination': dsts.copy(),
            'destination_idx': dsts_idx,
            'current_loc': cur_locs.copy(),
            'current_loc_idx': cur_locs_idx,
            'neighbor_mask': neighbor_mask.copy(),
            'target': target,
            'target_ecmp': target_ecmp,
            'embeddings': self.embeddings[self.nodes_with_state].copy(),
            'weights': weight
            # "hlsa_attn_activations": actis
        }

    def to_dict(self) -> Dict[str, np.array]:
        return {
            "network_states": self.network_states,
            "all_masks": self.network_masks,
            "embeddings": self.embeddings,
            "cur_locs": self.current_locations,
            "destinations": self.destinations,
            "all_neighbors": self.neighbohoods,
            "targets": self.targets,
            "targets_ecmp": self.targets_ecmp,
            "nodes_with_state": self.nodes_with_state,
            "use_embeddings_as_queries_for_attention_over_links": self.use_embeddings_as_queries_for_attention_over_links,
            "binarize_targets": self.binarize_targets,
            "attn_activations_all": self.attn_activations_all,
            "attn_activations_dst_tor": self.attn_activations_dst_tor,
            "use_att_activations_dst_tor": self.use_att_activations_dst_tor
        }


class StatefulGPUDataset(Dataset):
    """
        Dataset for stateful forwarding data. One sample consists of six tensors:
            - Network State: (V, D, F).
            - Network State Mask: (V, D, 1).
            - Current Location encoding: (1, E).
            - Neighbors: (max_degree, E).
            - Destination encoding: (1, E).
            - Output (D (+ 1)?)
        Where D is the maximum degree in the graph, E the dimensionality of the
        embedding of nodes, V the number of nodes in the graph and F the number of features
        per edge. All tensors have an additional first dimension corresponding to the
        batch size.

        If instead of pre-computed embeddings learned embeddings should be used,
        then E = 1 and the returned values correspond to indexes that index into
        a tensor of embeddings that must be provided by the model.

        The size of the output can be D, i.e., the maximum degree of D + 1. In the
        latter case the network supports dropping of packets if no viable route is
        available.

        This dataset directly moves the tensors to the GPU memory.
    """
    @classmethod
    def from_hdf5(cls, full_path_templates: str, full_path_embeddings: str=None) -> 'StatefulGPUDataset':
        data = inout.read_link_failure_data(full_path_templates)
        if full_path_embeddings is None:
            data['embeddings'] = None
        else:
            data['embeddings'] = inout.read_embeddings(full_path_embeddings)

        data["network_states"]   = data["network_states"].astype(np.float32)
        data["all_masks"]        = data["all_masks"].astype(np.float32)
        data["targets"]          = data["targets"].astype(np.float32)

        data["cur_locs"]         = data["cur_locs"].astype(np.int32)
        data["destinations"]     = data["destinations"].astype(np.int32)
        data["all_neighbors"]    = data["all_neighbors"].astype(np.int32)
        data["nodes_with_state"] = data["nodes_with_state"].astype(np.int32)
        return cls(**data)

    @classmethod
    def from_hdf5_files(cls, full_path_templates_dir: str,
                        full_path_embeddings: str=None) -> 'StatefulGPUDataset':
        """
            Create dataset from multiple hdf5 files. Read in the files one after
            the other and extend the existing dataset.
        """
        data = None
        for f in os.listdir(full_path_templates_dir):
            if not f.endswith('.h5'):
                continue
            else:
                tmp = inout.read_link_failure_data(os.path.join(full_path_templates_dir, f))
                if data is None:
                    data = tmp
                else:
                    for k, v in data.items():
                        if k == 'nodes_with_state':
                            continue
                        elif k == 'all_neighbors':
                            continue
                        elif k == 'all_masks':
                            continue
                        else:
                            data[k] = np.concatenate([v, tmp[k]])
        if full_path_embeddings is None:
            data['embeddings'] = None
        else:
            data['embeddings'] = inout.read_embeddings(full_path_embeddings).astype(np.float32)

        data["network_states"]   = data["network_states"].astype(np.float32)
        data["all_masks"]        = data["all_masks"].astype(np.float32)
        data["targets"]          = data["targets"].astype(np.float32)

        data["cur_locs"]         = data["cur_locs"].astype(np.int32)
        data["destinations"]     = data["destinations"].astype(np.int32)
        data["all_neighbors"]    = data["all_neighbors"].astype(np.int32)
        data["nodes_with_state"] = data["nodes_with_state"].astype(np.int32)
        return cls(**data)

    def __init__(self, network_states: torch.Tensor, all_masks: np.array,
                 cur_locs: np.array, destinations: np.array, all_neighbors: np.array,
                 targets: np.array, nodes_with_state: np.array, embeddings: np.array=None,
                 device='cuda:0', use_embeddings_as_queries_for_attention_over_links=False):
        """
        Initializes object. Note that the names of the arguments should match
        up with the keys in the dataset returned by the function in
        dataprep.link_failures.get_link_failure_datasets to pipe results through.

        NNL is the number of non-leaf-nodes in the graph.

        Args:
            network_states: All recorded network states. One network state is
                associated with each target. Has shape (N, NNL, D, F).
            all_masks: A static binary tensor indicating the neighborhood
                relationships between nodes. Has shape (V, D, F).
            embeddings: Binary matrix of node embeddings. Has shape (V, E).
            cur_locs: Indices of nodes. Used to index into the embeddings
                matrix. Has shape (N, 1).
            destinations: Indices of nodes. Used to index into the embeddings
                matrix. Represent the current destination node the packet should
                be forwarded to. Has shape (N, 1).
            all_neighbors: Indices that index into embeddings and represent the
                neighborhood relations between the nodes. Has shape (V, D, F).
            targets: The outputs that should be learned. Has shape (N, D).
            nodes_with_state: Array containing the indices of the nodes that
                are not leafes. Has shape (NNL,).
        """
        self.network_states = torch.tensor(network_states, device=device)
        self.network_masks = torch.tensor(all_masks, device=device)
        self.embeddings = torch.tensor(embeddings, device=device)
        self.current_locations = torch.tensor(cur_locs, device=device)
        self.destinations = torch.tensor(destinations, device=device)
        self.neighbohoods = torch.tensor(all_neighbors, device=device)
        self.targets = torch.tensor(targets, device=device)
        self.nodes_with_state = torch.tensor(nodes_with_state, device=device)
        self.use_embeddings_as_queries_for_attention_over_links = use_embeddings_as_queries_for_attention_over_links
        print("{:20s} {}".format("network_states",    str(self.network_states.shape)))
        print("{:20s} {}".format("network_masks",    str(self.network_masks.shape)))
        print("{:20s} {}".format("embeddings",       str(self.embeddings.shape)))
        print("{:20s} {}".format("current_locations",str(self.current_locations.shape)))
        print("{:20s} {}".format("destinations",     str(self.destinations.shape)))
        print("{:20s} {}".format("neighbohoods",     str(self.neighbohoods.shape)))
        print("{:20s} {}".format("targets",          str(self.targets.shape)))
        print("{:20s} {}".format("nodes_with_state", str(self.nodes_with_state.shape)))
        print("{:20s} {}".format(
            "use_embeddings_as_queries_for_attention_over_links",
            str(self.use_embeddings_as_queries_for_attention_over_links))
        )

    def __len__(self) -> int:
        return self.targets.shape[0]

    @property
    def max_degree(self) -> int:
        return self.network_masks.shape[-2]

    @property
    def num_nodes_with_state(self) -> int:
        return self.nodes_with_state.size

    @property
    def num_nodes(self) -> int:
        return self.network_masks.shape[0]

    @property
    def embedding_dim(self) -> int:
        assert self.embeddings is not None, 'Embeddings is not set, dimensionality unknown.'
        return self.embeddings.shape[-1]

    @property
    def num_features_state(self) -> int:
        return self.network_states.shape[-1]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            raise RuntimeError
            # idx = idx.tolist()
        # The the current locations for the batch.
        cur_locs = self.current_locations[idx, 0]
        # Get the destinations for the batch.
        dsts = self.destinations[idx, 0]
        # Get the mask. Use the indices as well since mask is static over time.
        neighbor_mask = self.network_masks[cur_locs, :, :]
        # Get the corresponding neighborhoods to the current locations. Currently
        # are indices into the embeddings matrix.
        neighborhood = self.neighbohoods[cur_locs, :, :]

        if self.embeddings is not None:
            # Flatten the neighborhood index tensor and retrieve the embeddings
            # stored at the corresponding locations.
            tmp = self.embeddings.index_select(0, neighborhood.flatten())
            # Reconstruct the correct shape and neighborhoods.
            neighborhood = torch.reshape(tmp, (self.max_degree, self.embedding_dim))
            # Non-existing neighbors have a value of -1 or 0. This retrieves a
            # valid embedding. Zero this embedding out using the mask.
            neighborhood = torch.mul(neighborhood, neighbor_mask)
            # Retrieve the embeddings for current locations and destinations.
            cur_locs = self.embeddings[cur_locs, :]
            dsts = self.embeddings[dsts, :]

        if self.use_embeddings_as_queries_for_attention_over_links:
            q_links = np.expand_dims(self.embeddings[self.nodes_with_state], axis=1)
        else:
            q_links = None

        return {
            'network_state': self.network_states[idx, :, :, :],
            # Add an emtpy first dimention to the network masks. The masks are
            # the same for each state, thus re-use them. The masks get correctly
            # broadcasted along the first dimension in the attention modules.
            'network_masks': self.network_masks.index_select(self.nodes_with_state, 0),
            'embd_nodes_state': q_links,
            'neighbors': neighborhood,
            'destination': dsts,
            'current_loc': cur_locs,
            'neighbor_mask': neighbor_mask,
            'target': self.targets[idx, :],
            'embeddings': self.embeddings.index_select(self.nodes_with_state, 0)
        }

    def to_dict(self) -> Dict[str, np.array]:
        return {
            "network_states": self.network_states,
            "all_masks": self.network_masks,
            "embeddings": self.embeddings,
            "cur_locs": self.current_locations,
            "destinations": self.destinations,
            "all_neighbors": self.neighbohoods,
            "targets": self.targets,
            "nodes_with_state": self.nodes_with_state,
            "use_embeddings_as_queries_for_attention_over_links": self.use_embeddings_as_queries_for_attention_over_links
        }


class StatefulFileDataset(Dataset):
    def __init__(self, folder_path, is_train, embeddings):
        self.embeddings = embeddings
        self.fps = []
        self.n_samples = []
        if is_train:
            s = 1
            e = 19
        else:
            s = 19
            e = 21
        for i in range(s, e):
            self.fps.append(h5py.File(os.path.join(
                folder_path,
                "link-failure-data-{:d}.h5".format(i)
            ), "r"))
            self.n_samples.append(self.fps[-1]['targets'].shape[0])

    def __len__(self):
        return np.sum(self.n_samples)

    @property
    def max_degree(self) -> int:
        return self.fps[0]['all_masks'].shape[-2]

    @property
    def embedding_dim(self) -> int:
        assert self.embeddings is not None, 'Embeddings is not set, dimensionality unknown.'
        return self.embeddings.shape[-1]

    @property
    def num_features_state(self) -> int:
        return self.fps[0]['network_states'].shape[-1]

    @property
    def num_nodes(self) -> int:
        return self.fps[0]['network_states'].shape[1]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            raise RuntimeError
            # idx = idx.tolist()
        fp = None
        start = 0
        for i, step in enumerate(self.n_samples):
            if start <= idx < start + step:
                idx = idx - start
                fp = self.fps[i]
                break
            else:
                start += step
        assert fp is not None, "Did not find file for index {}".format(idx)
        # The the current locations for the batch.
        cur_locs = int(fp['cur_locs'][idx, 0])
        # Get the destinations for the batch.
        dsts = int(fp['destinations'][idx, 0])
        # Get the mask. Use the indices as well since mask is static over time.
        neighbor_mask = fp['all_masks'][cur_locs, :, :]
        # Get the corresponding neighborhoods to the current locations. Currently
        # are indices into the embeddings matrix.
        neighborhood = fp['all_neighbors'][cur_locs, :, :]
        nodes_with_state = fp['nodes_with_state'][()].astype(np.int32)

        if self.embeddings is None:
            embeddings = np.arange(self.num_nodes).astype(np.float32)
        else:
            # Flatten the neighborhood index tensor and retrieve the embeddings
            # stored at the corresponding locations.
            embeddings = self.embeddings
            tmp = self.embeddings[neighborhood.flatten().astype(np.int64), :]
            # Reconstruct the correct shape and neighborhoods.
            neighborhood = tmp.reshape(self.max_degree, self.embedding_dim)
            # Non-existing neighbors have a value of -1 or 0. This retrieves a
            # valid embedding. Zero this embedding out using the mask.
            neighborhood = np.multiply(neighborhood, neighbor_mask)
            # Retrieve the embeddings for current locations and destinations.
            cur_locs = self.embeddings[cur_locs, :].copy()
            dsts = self.embeddings[dsts, :].copy()
        return {
            'network_state': fp['network_states'][idx, :, :, :].copy(),
            # Add an emtpy first dimention to the network masks. The masks are
            # the same for each state, thus re-use them. The masks get correctly
            # broadcasted along the first dimension in the attention modules.
            'network_masks': fp['all_masks'][nodes_with_state, :, :].copy(),
            'neighbors': neighborhood.copy(),
            'destination': dsts,
            'current_loc': cur_locs,
            'neighbor_mask': neighbor_mask.copy(),
            'target': fp['targets'][idx, :].copy(),
            'embeddings': embeddings[nodes_with_state].copy()
        }

    def __del__(self):
        for fp in self.fps:
            fp.close()


def filter_dataset(ds: StatefulDataset, graph: nx.Graph, prefix: str) -> StatefulDataset:
    index_to_name = {d['idx']: n for n, d in graph.nodes(data=True)}
    indices = []
    for i in range(ds.current_locations.shape[0]):
        cur_loc_idx = ds.current_locations[i, 0]
        if index_to_name[cur_loc_idx].startswith(prefix):
            indices.append(i)
        else:
            continue
    ds_new = StatefulDataset(
        network_states=ds.network_states[indices],
        all_masks=ds.network_masks,
        cur_locs=ds.current_locations[indices],
        destinations=ds.destinations[indices],
        all_neighbors=ds.neighbohoods,
        targets=ds.targets[indices],
        nodes_with_state=ds.nodes_with_state,
        embeddings=ds.embeddings,
        use_embeddings_as_queries_for_attention_over_links=ds.use_embeddings_as_queries_for_attention_over_links,
        generate_weights=ds.generate_weights
    )
    return ds_new

