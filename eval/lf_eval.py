import torch
import numpy as np
from typing import List, Dict, Tuple, Any, Union
import pandas as pd
import logging
import networkx as nx
import json
from torch.utils.data import DataLoader
import itertools as itt
import os
import h5py
import subprocess
import shutil
import time

import dataprep.sp_prep as sp_prep
import dataprep.link_failures as lf_prep
from topos.fattree import make_topo
from dataprep.input_output import read_graph, read_embeddings, read_link_failure_data
from dataprep.datasets import StatefulDataset, filter_dataset
from models.stateful import StatefulConfig, StatefulModel
from models.utils import full_cross_entropy, multi_class_loss, _multi_class_loss
import present
import eval.utils as evutils
from training.utils import expand_stateful_config

EDGE_WEIGHT = lf_prep.EDGE_WEIGHT

if torch.cuda.is_available():
    DEV = torch.device("cuda:0")
else:
    DEV = torch.device("cpu")

MAX_NUM_BLOCKS = 32
MAX_NUM_BLOCKS = 64


# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('lf_eval')
logger.setLevel(logging.DEBUG)


def print_trial_summary(logdir: str, dsetpath: str, filter: str):
    logger.warn("Works only for the LLT Both stuff")
    anas = evutils.load_multiple_tune_results(
        'data/training-results',
        'FatTreeK8LinkFailuresIpEmbeddingBothHlsasCombinedTors2HostsLltMC'
    )
    dfs = [ana.dataframe('cross_entropy-val', 'min') for ana in anas]
    df = pd.concat(dfs, axis=0)
    df.sort_values(by='cross_entropy-val', inplace=True)
    sep = '\t'
    print(" 1) Cross Entropy")
    print(" 2) Cross Entropy Validation")
    print(" 3) L1 weight Attention")
    print(" 4) L1 Weight HLSAs")
    print(" 5) Final FCNs")
    print(" 6) FCN Link Ebeddings")
    print(" 7) FCNs Neighbor Attention")
    print(" 8) Arity")
    print(" 9) Num Blocks")
    print("10) Num Heads")
    print("11) Dim FCN")
    print("12) Dim Hidden")
    print("13) Dim Out")
    print("14) Learning Rate")
    print("15) Batch Size")

    print("{:6d}\t{:6d}\t{:8d}\t{:8d}\t{:11d}\t{:3d}\t{:11d}\t{:1d}\t{:2d}\t{:1d}\t{:3d}\t{:2d}\t{:2d}\t{:8d}\t{:3d}".format(
        1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    ))
    for i in range(10):
        print("{:.4f}".format(df.iloc[i]['cross_entropy']), end=sep)
        print("{:.4f}".format(df.iloc[i]['cross_entropy-val']), end=sep)
        print("{:.6f}".format(df.iloc[i]['config/model_config']["alpha_l1_hlsa_attn_weights"]), end=sep)
        print("{:.6f}".format(df.iloc[i]['config/model_config']["alpha_l1_hlsas"]), end=sep)
        print("{:11s}".format(','.join([str(x) for x in df.iloc[i]["config/model_config"]['final_fcns']])), end=sep)
        print("{:11s}".format(','.join([str(x['dim_fcn']) for x in df.iloc[i]["config/model_config"]['link_attns']])), end=sep)
        print("{:3d}".format(df.iloc[i]["config/model_config"]['neighbor_attns']['dim_fcn']), end=sep)
        print("{:1d}".format(df.iloc[i]["config/model_config"]['hlsa_gs']['arity']), end=sep)
        print("{:2d}".format(df.iloc[i]["config/model_config"]['hlsa_gs']['num_blocks']), end=sep)
        print("{:1d}".format(df.iloc[i]["config/model_config"]['hlsa_attns']['num_heads']), end=sep)
        print("{:3d}".format(df.iloc[i]["config/model_config"]['hlsa_attns']['dim_fcn']), end=sep)
        print("{:2d}".format(df.iloc[i]["config/model_config"]['hlsa_attns']['dim_hidden']), end=sep)
        print("{:2d}".format(df.iloc[i]["config/model_config"]['hlsa_attns']['dim_out']), end=sep)
        print("{:.6f}".format(df.iloc[i]['config/optimizer']['lr']), end=sep)
        print("{:3d}".format(df.iloc[i]['config/batch_size']), end=sep)
        print()


def compute_hlsas_attn(model: StatefulModel, dsetpath) -> pd.DataFrame:
    assert model.config.hlsa_model == 'attn', "only ATTN supported"
    ds = StatefulDataset.from_hdf5(
        full_path_templates=os.path.join(dsetpath, 'val/link-failure-data-26.h5'),
        full_path_embeddings='/opt/project/data/fat-tree-k8/fat-tree-k8-ip-embedding.h5'
    )
    loader = DataLoader(ds, batch_size=1, shuffle=True)
    hlsas = []
    indices = []
    for i, sample in enumerate(loader):
        tmp = model.attend_over_links(
            keys=sample['network_state'],
            values=sample['network_state'],
            queries=sample['network_state'],
            attention_mask=sample['network_masks']
        )
        tmp = model.sample_hlsa_activations(model.pool_attended_links(tmp)).detach().numpy()
        idx = np.arange(80)
        hlsas.append(tmp[0, :, :])
        indices.append(idx)
        if i >= 0:
            break
    return pd.DataFrame(np.concatenate(hlsas), index=np.concatenate(indices))


def compute_hlsas(model: StatefulModel) -> pd.DataFrame:
    def e_to_s(e):
        if e == 1:
            return np.array([[[1., 0., 1., 0]]], dtype=np.float32)
        else:
            return np.array([[[0., 1., 0., 1]]], dtype=np.float32)
    def e_to_s2(e):
        if e == 1:
            return np.array([[[1., 0., 1., 0, random.uniform()]]], dtype=np.float32)
        else:
            return np.array([[[0., 1., 0., 1, random.uniform()]]], dtype=np.float32)

    assert model.config.hlsa_model == 'fcn', "Only FCN supported at the moment"
    random = np.random.RandomState(seed=1)
    iterables = [[0, 1] for _ in range(8)]
    input = np.concatenate([np.concatenate([e_to_s2(s) for s in states], axis=1)
                          for states in itt.product(*iterables)])
    hlsas = torch.squeeze(torch.sigmoid(model.attend_over_links(
        keys=None,
        values=torch.tensor(np.expand_dims(input, axis=1)),
        queries=None,
        attention_mask=None
    )), dim=-2).detach().numpy()
    index = ["-".join(["{:d}".format(s) for s in states]) for states in itt.product(*iterables)]
    return pd.DataFrame(hlsas, index=index)


def get_attn_scores_nghbs_df(model: StatefulModel, graph: nx.Graph, cur_loc: str,
                             dst: str, neighbor_idx: Dict[Any, Dict[Any, int]],
                             embeddings: np.array) -> pd.DataFrame:
    max_degree = 8
    dim_embedding = 24
    cur_loc_idx = graph.nodes[dst][sp_prep.IDX]
    neighbors = np.zeros([1, max_degree, dim_embedding], dtype=np.float32)
    mask = np.ones([1, max_degree, 1], dtype=np.float32)
    embd_dst = np.expand_dims(np.expand_dims(embeddings[cur_loc_idx, :], axis=0), axis=0)

    cols = []
    for v in graph.neighbors(cur_loc):
        v_idx = graph.nodes[dst][sp_prep.IDX]
        neighbors[0, neighbor_idx[cur_loc][v], :] = embeddings[v_idx, :]
        cols.append(v)
    model.attend_over_neighbors(
        keys=torch.tensor(neighbors),
        queries=torch.tensor(embd_dst),
        values=torch.tensor(neighbors),
        attention_mask=torch.tensor(mask)
    )
    tuples = []
    all_scores = model.attend_over_neighbors.last_scores
    values = []
    for i, scores in enumerate(all_scores):
        tuples.append((dst, 'head_{:d}'.format(i)))
        values.append(torch.squeeze(scores, dim=-2).detach().numpy())
    frame = pd.DataFrame(
        np.concatenate(values),
        columns=cols,
        index=pd.MultiIndex.from_tuples(tuples)
    )
    return frame


def get_attn_scores_nghbs(model: StatefulModel, graph: nx.Graph,
                      embeddings_path: str) -> List[pd.DataFrame]:
    """
    obtain the attention scores of the location module.

    Returns:

    """
    logger.debug("Get embeddings...")
    embeddings = read_embeddings(embeddings_path)
    neighbor_idx = sp_prep._neighbor_to_index(graph)

    cur_locs = [
        'tor-0000',
        'tor-0001',
        'tor-0002',
        'tor-0003',
        'tor-0008',
        'tor-0013',
        'tor-0018',
        'tor-0023',
        'agg-0000',
        'agg-0001',
        'agg-0002',
        'agg-0003',
        'agg-0008',
        'agg-0013',
        'agg-0018',
        'agg-0023',
        'core-0000',
        'core-0001',
        'core-0004',
        'core-0005',
        'core-0008',
        'core-0009',
        'core-0012',
        'core-0013'
    ]
    dsts = [
        'h-0000',
        'h-0001',
        'h-0004',
        'h-0005',
        'h-0008',
        'h-0009',
        'h-0012',
        'h-0013',
        'h-0032',
        'h-0033',
        'h-0048',
        'h-0049',
        'h-0068',
        'h-0069',
        'h-0088',
        'h-0089'
    ]
    frames = {}
    for cur_loc in cur_locs:
        frames[cur_loc] = pd.concat([get_attn_scores_nghbs_df(
            model=model,
            graph=graph,
            cur_loc=cur_loc,
            dst=dst,
            embeddings=embeddings,
            neighbor_idx=neighbor_idx
        ) for dst in dsts])
    return frames


def get_attention_scores(model: StatefulModel, graph: nx.Graph, embeddings_path: str,
        dsetpath: str, destination=None) -> List[pd.DataFrame]:
    """
        Given a model, graph, and embeddings calcualte the attention scores over
        the nodes in the graph, indicating the HLSAs that are exchanged between
        nodes.

        Args:
            model: A neural network model that should be evaluated.
            graph: Networkx Graph of the network for which weights should be
                evaluated.
            embeddings_path: Full path to the hdf5 file storing the node embeddings.
            dataset_path: Path to the directory containing the individual files
                making up the overall dataset.

        Returns:
            scores: Pandas data frame with the node names as indices and the
                attention scores for each node as values.
    """
    logger.debug("Get embeddings...")
    embeddings = read_embeddings(embeddings_path)
    # sfd = StatefulFileDataset(dataset_path, False, embeddings)
    logger.debug("Load dataset...")
    sfd = StatefulDataset.from_hdf5(
        full_path_templates=os.path.join(dsetpath, 'val/link-failure-data-26.h5'),
        full_path_embeddings=os.path.join(dsetpath, 'fat-tree-k{:d}-ip-embedding.h5'.format(k)),
        policy=model.config.policy
    )
    sfd.use_embeddings_as_queries_for_attention_over_links = True
    logger.debug("Get first sample")
    sample = sfd[0]
    logger.debug("Compute HSLAs with\n\tnetwork_state: {}\n\tnetwork_masks: {}".format(
        str(torch.unsqueeze(torch.tensor(sample['network_state']), dim=0).shape),
        str(torch.unsqueeze(torch.tensor(sample['network_masks']), dim=0).shape)
    ))
    hslas = model.attend_over_links(
        keys=torch.unsqueeze(torch.tensor(sample['network_state']), dim=0),
        # queries=torch.unsqueeze(torch.tensor(sample['network_state']), dim=0),
        queries=torch.unsqueeze(torch.unsqueeze(torch.tensor(sample['current_loc']), dim=0), dim=0),
        values=torch.unsqueeze(torch.tensor(sample['network_state']), dim=0),
        attention_mask=torch.unsqueeze(torch.tensor(sample['network_masks']), dim=0)
    )
    logger.debug("HLSAS have shape {}".format(str(hslas.shape)))
    hslas = model.pool_attended_links(hslas)
    logger.debug("Pooled HLSAS have shape {}".format(str(hslas.shape)))
    if model.config.hlsa_gs is None:
        hslas = torch.sigmoid(hslas)
    else:
        hslas = model.sample_hlsa_activations(hslas)

    idx_to_node = {d['idx']: n for n, d in graph.nodes(data=True)}
    scores = None
    if destination is None:
        destination = input("Name of destination: ")
    assert graph.has_node(destination), "graph does not have a node {}".format(destination)
    embd_dst = embeddings[graph.nodes[destination][sp_prep.IDX]]
    nodes = [''] * lf_prep._get_num_no_leaves(graph)
    for n, x in graph.nodes(data=True):
        if sp_prep.NS_IDX in x:
            nodes[x[sp_prep.NS_IDX]] = n
    for i in range(graph.number_of_nodes()):
        node = idx_to_node[i]
        if node.startswith('h-'):
            continue
        ####
        q = torch.cat([torch.tensor(embeddings[graph.nodes[node][sp_prep.IDX], :]), torch.tensor(embd_dst)])
        logger.debug("Attend over HLSAs for node {}:\n\tkeys: {}\n\tqueries: {}\n\tvalues : {}\n\ttarget: {}".format(
            node,
            str(torch.unsqueeze(torch.tensor(sample['embeddings']), dim=0).shape),
            str(torch.unsqueeze(torch.unsqueeze(q, dim=0), dim=0).shape),
            str(hslas.shape),
            str(embd_dst.flatten().tolist())
        ))
        _ = model.attend_over_hlsas(
            keys=torch.unsqueeze(torch.tensor(sample['embeddings']), dim=0),
            queries=torch.unsqueeze(torch.unsqueeze(q, dim=0), dim=0),
            values=hslas
        )
        logger.debug("Attended over neighbors, retrieve scores")
        tmp = model.attend_over_hlsas.last_scores
        # Check if multi head attention is used and initialize scores accordingly.
        # In case no MHA is used, wrap the scores in a list to make the for
        # loop work and reduce code.
        if type(tmp) == list:
            if scores is None:
                scores = [{} for _ in range(len(tmp))]
        else:
            if scores is None:
                scores = [{}]
            tmp = [tmp]
        for j, x in enumerate(tmp):
            if graph.has_edge(node, destination):
                scores[j][node] = np.zeros(x.detach().numpy().size)
            else:
                scores[j][node] = x.detach().numpy().flatten()
    logger.debug("construct dataframes:\n\tscores: {}\n\tdicts sizes: {}\n\tsample_size: {}\n\tnodes have: {}".format(
        len(scores),
        str([len(x) for x in scores]),
        str(scores[0][node].shape),
        len(nodes)
    ))
    dfs = [pd.DataFrame.from_dict(x, orient='index', columns=nodes) for x in scores]
    return dfs


def _matrix_to_str(probs, np_targets):
    s = 'Probs|Target comparison: \n'
    for i in range(probs.shape[0]):
        s += "{:3d})\t".format(i)
        for j in range(probs.shape[1]):
            s += "{:.2f}|{:.2f}  ".format(probs[i, j], np_targets[i, j])
        s += "\n"
    s += "\n----------"
    return s


def _matrix_to_str_adv(probs, np_targets, np_cur_locs, np_dsts, idx_name):
    s = 'Probs|Target comparison: \n'
    for i in range(probs.shape[0]):
        s += "{:3d}) {:9s}->{:9s}:\t".format(i, idx_name[np_cur_locs[i]], idx_name[np_dsts[i]])
        for j in range(probs.shape[1]):
            s += "{:.2f}|{:.2f}  ".format(probs[i, j], np_targets[i, j])
        s += "\n"
    s += "\n----------"
    return s


def _filter_samples(wanted_dsts: List[Any], wanted_locs: List[Any], graph: nx.Graph,
                    states: np.array, targets: np.array, destinations: np.array,
                    cur_locs: np.array) -> tuple:
    """
        Reduce the samples to only those start nodes and destinations that are
        wanted by the user.

        Args:
            wanted_dsts: List of destination nodes for which evaluation should take
                place.
            wanted_locs: List of start nodes for which evaluation should take place.
            graph: Original graph, required to get node indices.
            states: Network states.
            targets: Target vectors.
            destinations: Destination indices from data prep.
            cur_locs: Current locations from data prep.

        Returns:
            Filtered states, targets, destinations and cur_locs.
    """
    dsts_idx = [graph.nodes[n]['idx'] for n in wanted_dsts]
    start_idx = [graph.nodes[n]['idx'] for n in wanted_locs]
    indices = []
    for i in range(destinations.size):
        if int(destinations[i]) in dsts_idx:
            indices.append(i)
        if int(cur_locs[i]) in start_idx:
            indices.append(i)
    return states[indices], targets[indices], destinations[indices], cur_locs[indices]


def _create_initial_input(graph: nx.Graph, num_outputs: int, num_non_leaves: int,
                          pairs: List[Tuple[Any, Any]], output_mode: str,
                          value_index: Dict[Any, Dict[Any, int]]) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
    """
        Create the initial input based on the link failures that are applied
        and the hosts that should communicate.
    """
    states, targets, destinations, cur_locs = zip(*[lf_prep._link_failure_to_samples(
        graph=graph,
        links=[],
        num_outputs=num_outputs,
        value_index=value_index,
        adj={},
        output_types=[output_mode, lf_prep.OUTPUT_ECMP],
        num_non_leaves=num_non_leaves,
        cur_loc=u,
        destination=v
    ) for u, v in pairs])
    return (
        np.concatenate(states),
        np.concatenate([t[output_mode] for t in targets]),
        np.concatenate([t[lf_prep.OUTPUT_ECMP] for t in targets]),
        np.concatenate(destinations).flatten(),
        np.concatenate(cur_locs).flatten()
    )


def _expand_states(states) -> np.array:
    expanded_states = []
    for i in range(states.shape[0]):
        expanded_states.append(np.expand_dims(
            StatefulDataset.expand_edge_weights(states[i, :, :, :]),
            axis=0
        ))
    return np.concatenate(expanded_states)


def _new_targets(graph: nx.Graph,
                 pairs: List[Tuple[Any, Any]], num_non_leaves: int,
                 num_outputs: int, output_mode: str,
                 value_index: Dict[Any, Dict[Any, int]]) -> np.array:
    """
        Create the initial input based on the link failures that are applied
        and the hosts that should communicate.
    """
    return _create_initial_input(
        graph=graph,
        pairs=pairs,
        output_mode=output_mode,
        num_non_leaves=num_non_leaves,
        num_outputs=num_outputs,
        value_index=value_index)[1]


def _driver_random_lengths(graph: nx.Graph, to_check: np.array, seed: int,
                           pairs: List[Tuple[str, str]], output_mode: str) -> np.array:
    def kl_div(cur_loc, dst):
        output_generator = lf_prep.OutputCalculator(graph, cur_loc, dst)
        target = output_generator._get_output_wcmp(8, value_index)
        tmp = target.copy()
        tmp[tmp > 0] = 1.
        tmp = tmp / np.expand_dims(np.sum(tmp, axis=1), axis=1)
        kldiv = np.sum(target * np.log(np.clip(target, 1e-3, 1) / np.clip(tmp, 1e-3, 1)))
        return kldiv

    value_index = sp_prep._neighbor_to_index(graph)
    random = np.random.RandomState(seed=seed)
    distances = []
    for i, (src, dst) in enumerate(pairs):
        if to_check[i]:
            paths = list(nx.all_shortest_paths(graph, src, dst))
            path = paths[random.randint(0, len(paths))]
            distance = 0
            count = 0
            for u, v in zip(path[:-1], path[1:]):
                count += 1
                if output_mode == lf_prep.OUTPUT_LCP:
                    distance += graph.edges[u, v][lf_prep.EDGE_WEIGHT]
                elif output_mode == lf_prep.OUTPUT_HULA:
                    distance = np.max([distance, graph.edges[u, v][lf_prep.EDGE_WEIGHT]])
                elif output_mode == lf_prep.OUTPUT_WCMP:
                    distance += kl_div(u, dst)
                else:
                    raise KeyError(f"Unknown output mode {output_mode}")
            if output_mode == lf_prep.OUTPUT_WCMP:
                distances.append(distance / count)
            else:
                distances.append(distance)
    return np.array(distances)


def _driver_opt_lengths(graph: nx.Graph, to_check: np.array,
                        pairs: List[Tuple[str, str]], output_mode: str) -> np.array:
    distances = []
    for i, (src, dst) in enumerate(pairs):
        if to_check[i]:
            tmp = []
            for path in nx.all_shortest_paths(graph, src, dst, ):
                d = 0
                for u, v in zip(path[:-1], path[1:]):
                    if output_mode == lf_prep.OUTPUT_LCP:
                        d += graph.edges[u, v][lf_prep.EDGE_WEIGHT]
                    elif output_mode == lf_prep.OUTPUT_HULA:
                        d = np.max([d, graph.edges[u, v][lf_prep.EDGE_WEIGHT]])
                    elif output_mode == lf_prep.OUTPUT_WCMP:
                        d = 0.
                    else:
                        raise KeyError(f"Unknown output mode {output_mode}")
                tmp.append(d)
            distances.append(np.min(tmp))
    return np.array(distances)


class GetActivationsForDriver(object):
    def __init__(self, graph, path=None):
        self.graph = graph
        self.path = path
        if self.path is None:
            self.path = '/opt/project/data/fat-tree-k8/tors-to-hosts-llt-ecmp-core/'
        f = h5py.File(os.path.join(self.path, 'attn_activations_all.h5'), 'r')
        self.attn_weights = f['activations'][()]
        f.close()
        # file = h5py.File(os.path.join(self.path, 'attn_dst_tor.h5'), 'r')
        # self.attn_activations_dst_tor = file['activations'][()]
        # self.use_att_activations_dst_tor = file['is_contained'][()]
        # file.close()

    def __call__(self, dst, cur_loc):
        cur_loc_idx = self.graph.nodes[cur_loc]['idx']
        dst_idx = self.graph.nodes[dst]['idx']
        #if self.use_att_activations_dst_tor[cur_loc_idx] == 1:
        #    tmp = self.attn_activations_dst_tor[dst_idx]
        #else:
        #    tmp = self.attn_activations_dst_tor[cur_loc_idx]
        return np.expand_dims(
            np.concatenate([
                self.attn_weights[dst_idx],
                self.attn_weights[cur_loc_idx]
                # tmp
            ]),
            axis=0
        )


def driver(graph: nx.Graph, model: StatefulModel, embeddings: np.array,
           max_num_failures: int, pairs: List[Tuple[str, str]], output_mode: str,
           seed=1, mode='sample') -> tuple:
    random = np.random.RandomState(seed=seed)
    num_outputs = sp_prep._calc_num_outputs(graph)
    np_embeddings = embeddings
    num_non_leaves = lf_prep._get_num_no_leaves(graph)
    is_graph_weighted = lf_prep.EDGE_WEIGHT in graph.edges[('tor-0000', 'agg-0000')]

    value_index = sp_prep._neighbor_to_index(graph)
    index_neighbor = evutils._index_to_neighbor(graph)
    index_value = index_neighbor
    index_node = {d['idx']: n for n, d in graph.nodes(data=True)}
    # get_activations = GetActivationsForDriver(graph, './data/fat-tree-k8')

    aggregate = {
        'hula': lambda a, b: float(np.max([a, b])),
        'lcp': lambda a, b: a + b,
        'wcmp': lambda a, b: a + b}[output_mode]

    logger.debug("obtain adjacency dict...")
    # adj = sp_prep._make_distance_dict(graph, weight=None)
    # with open('/opt/project/data/fat-tree-k8/distance-mat-dict.json', 'w') as fh:
    #     json.dump(adj, fh)
    with open('/opt/project/data/fat-tree-k8/distance-mat-dict.json', 'r') as fh:
        adj = json.load(fh)
    logger.debug("\t--> Done")

    nodes_with_state = np.zeros(num_non_leaves, dtype=np.int32)
    for n, d in graph.nodes(data=True):
        if sp_prep.NS_IDX in d:
            nodes_with_state[d[sp_prep.NS_IDX]] = d[sp_prep.IDX]

    np_all_masks = np.expand_dims(
        lf_prep._get_all_masks(
            graph=graph,
            max_degree=num_outputs,
            failed_links=[],
            neighbor_to_index=value_index
        ),
        axis=0
    )
    np_masks_nodes_with_state = np_all_masks[:, nodes_with_state, :, :]
    np_all_neighbors = lf_prep._get_all_neighbors(
        graph=graph,
        max_degree=num_outputs,
        neighbor_to_index=value_index
    )

    logger.debug("Fail links...")
    failed_links = lf_prep._sample_failure_fat_tree(random, max_num_failures, 8, graph)
    for u, v in lf_prep._sample_node_failure_fat_tree(random, 3, 8, graph):
        if (u, v) not in failed_links:
            failed_links.append((u, v))
    logger.debug("Links failed are: {}\n".format(
        "\n\t".join(["({} <-> {})".format(u,v) for u, v in failed_links]))
    )
    graph.remove_edges_from(failed_links)

    new_pairs = []
    for src, dst in pairs:
        if graph.degree()[src] == 0 or graph.has_edge(src, dst):
            continue
        else:
            new_pairs.append((src, dst))
    logger.debug("{} of the pairs started on failed node".format(len(pairs) - len(new_pairs)))
    pairs = new_pairs
    initial_pairs = new_pairs

    logger.debug("Create samples")
    np_states, np_targets, _, np_destinations_idx, np_cur_locs_idx = _create_initial_input(
        graph=graph,
        pairs=pairs,
        num_non_leaves=num_non_leaves,
        value_index=value_index,
        num_outputs=num_outputs,
        output_mode=output_mode
    )
    if is_graph_weighted and model.config.hlsa_model is None:
        logger.debug("{:25s}: {:s}".format("np_states", str(np_states.shape)))
        np_states = _expand_states(np_states)

    np_embeddings_nodes_with_states = np.expand_dims(embeddings[nodes_with_state], 0)
    np_is_finished  = np.zeros(np_destinations_idx.size)
    np_reached_dst  = np.zeros(np_destinations_idx.size)
    np_lengths      = np.zeros(np_destinations_idx.size)
    np_distances    = np.zeros(np_destinations_idx.size)
    np_wrong_drops  = np.zeros(np_destinations_idx.size)
    np_correct_drops= np.zeros(np_destinations_idx.size)
    np_to_drop      = np.zeros(np_destinations_idx.size)
    np_destinations = np_embeddings[np_destinations_idx, :]

    logger.debug("{:25s}: {:s}".format("np_all_neighbors", str(np_all_neighbors.shape)))
    logger.debug("{:25s}: {:s}".format("np_states", str(np_states.shape)))
    logger.debug("{:25s}: {:s}".format("np_all_masks", str(np_all_masks.shape)))
    logger.debug("{:25s}: {:s}".format("np_masks_nodes_with_state", str(np_masks_nodes_with_state.shape)))
    logger.debug("{:25s}: {:s}".format("np_targets", str(np_targets.shape)))
    logger.debug("{:25s}: {:s}".format("np_destinations_idx", str(np_destinations_idx.shape)))
    logger.debug("{:25s}: {:s}".format("np_cur_locs_idx", str(np_cur_locs_idx.shape)))
    logger.debug("{:25s}: {:s}".format("np_embeddings", str(np_embeddings.shape)))
    logger.debug("{:25s}: {:s}".format("np_embeddings_nodes_with_states", str(np_embeddings_nodes_with_states.shape)))
    logger.debug("{:s}: {:s}".format("np_destinations", str(np_destinations.shape)))

    pairs = [(u, v) for u, v in initial_pairs]
    max_num_steps = 6
    for step in range(max_num_steps):
        logger.info("Step {:d}\n===============".format(step))
        np_neighbor_masks = np_all_masks[0, np_cur_locs_idx, :, :]
        np_embd_neighbors = np_all_neighbors[np_cur_locs_idx, :, :].astype(np.int32)
        np_embd_neighbors = np_embeddings[np_embd_neighbors.flatten(), :].reshape(
            -1, np_all_neighbors.shape[1], embeddings.shape[-1]
        )
        np_embd_cur_locs  = np_embeddings[np_cur_locs_idx, :]
        # np_attn_actis = np.concatenate([get_activations(dst, cloc) for cloc, dst in pairs])

        # logger.debug("{:s}: {:s}".format("np_neighbor_masks", str(np_neighbor_masks.shape)))
        # logger.debug("{:s}: {:s}".format("np_embd_neighbors", str(np_embd_neighbors.shape)))
        # logger.debug("{:s}: {:s}".format("np_embd_cur_locs", str(np_embd_cur_locs.shape)))

        t_neighbor_masks   = torch.tensor(np_neighbor_masks)
        t_embd_nodes_w_state = torch.tensor(np.expand_dims(np.expand_dims(embeddings[nodes_with_state], axis=1), axis=0))
        t_embd_neighbors   = torch.tensor(np_embd_neighbors)
        t_states           = torch.tensor(np_states)
        t_all_masks        = torch.tensor(np_masks_nodes_with_state)
        t_embd_cur_loc     = torch.tensor(np_embd_cur_locs)
        t_embd_destination = torch.tensor(np_destinations)
        t_embd_n_w_s       = torch.tensor(np_embeddings_nodes_with_states)
        # t_attn_actis = torch.tensor(np_attn_actis)

        # Use no_grad context to not calculate any gradient information greatly
        # reducing computational cost and consumed memory. Without this, I quickly
        # would run into OOM.
        with torch.no_grad():
            # logger.debug("Tensor {:25s}:\t{:s}".format("network_states",        str(t_states.shape)))
            # logger.debug("Tensor {:25s}:\t{:s}".format("network_state_mask",    str(t_all_masks.shape)))
            # logger.debug("Tensor {:25s}:\t{:s}".format("embd_nodes_w_state",    str(t_embd_nodes_w_state.shape)))
            # logger.debug("Tensor {:25s}:\t{:s}".format("embeddings_neighbors",  str(t_embd_neighbors.shape)))
            # logger.debug("Tensor {:25s}:\t{:s}".format("mask_embeddings",       str(t_neighbor_masks.shape)))
            # logger.debug("Tensor {:25s}:\t{:s}".format("embd_current_location", str(t_embd_cur_loc.shape)))
            # logger.debug("Tensor {:25s}:\t{:s}".format("embd_destination",      str(t_embd_destination.shape)))
            # logger.debug("Tensor {:25s}:\t{:s}".format("embd_n_w_s",            str(t_embd_n_w_s.shape)))
            preds, preds_ecmp, _ = model.forward(
                network_state=t_states.to(device=DEV),
                network_state_mask=t_all_masks.to(device=DEV),
                embeddings_neighbors=t_embd_neighbors.to(device=DEV),
                mask_embeddings=t_neighbor_masks.to(device=DEV),
                embd_current_location=t_embd_cur_loc.to(device=DEV),
                embd_destination=t_embd_destination.to(device=DEV),
                embeddings=t_embd_n_w_s.to(device=DEV),
                embd_nodes_state=t_states.to(device=DEV)  # t_embd_nodes_w_state.to(device=DEV)
                # hlsa_attn_head_activations=t_attn_actis.to(device=DEV)
            )
            # Immediately detach resulst. Else, the computational graph gets
            # stored as well. Also contributing towards potential OOM.
            predictions = preds.detach()
            predictions_ecmp = preds_ecmp.detach()
            logger.debug("Loss are {} vs {} (optimal vs. prediction)".format(
                _multi_class_loss(
                    probs=torch.tensor(np.clip(np_targets, 0.05, 0.95)),
                    target=torch.tensor(np_targets),
                    weights=torch.tensor([[1.]])
                ),
                _multi_class_loss(
                    probs=preds,
                    target=torch.tensor(np_targets).to(device=DEV),
                    weights=torch.tensor([[1.]]).to(device=DEV)
                )
            ))
        if model.config.multiclass:
            probs = torch.sigmoid(predictions)
            probs_ecmp = torch.sigmoid(predictions_ecmp)
            probs_ecmp = probs_ecmp > 0.5
            probs_ecmp = probs_ecmp.type_as(probs)
            probs = torch.mul(probs, probs_ecmp)
            probs = probs / torch.unsqueeze(torch.sum(probs, dim=-1), dim=-1)
        else:
            probs = torch.softmax(predictions, axis=-1)

        kl_divergence = np.sum(np_targets * np.log(np.divide(np.clip(np_targets, 1e-3, 1), probs.detach().numpy())), axis=1)
        tmp = np_targets.copy()
        tmp[tmp > 0] = 1.
        tmp = np.clip(tmp / np.expand_dims(np.sum(tmp, axis=1), axis=-1), 1e-3, 1)
        kl_divergence_rnd = np.sum(np_targets * np.log(np.divide(np.clip(np_targets, 1e-3, 1), tmp)), axis=1)


        mask = np.logical_not(np_is_finished)
        logger.debug("{:s}".format(_matrix_to_str(probs.to("cpu").numpy()[mask], np_targets[mask])))
        # Creates an on-dimensional arrays of index predictions.
        if mode == 'sample':
            samples = torch.squeeze(torch.multinomial(probs, 1), axis=1).to('cpu')
        else:
            samples = torch.argmax(probs, dim=1).to('cpu')

        np_predictions = samples.numpy()
        # s_idx:  Sample index.
        # cl_idx: Current location index.
        # n_idx:  Neighbor index.
        for s_idx, (cl_idx, n_idx) in enumerate(zip(np_cur_locs_idx, samples)):
            if np_targets[s_idx, 0] > 0:
                np_to_drop[s_idx] = 1
            if np_is_finished[s_idx] == 1:
                # Destination already reached or dropped. Do nothing.
                continue
            # If packet has destination not yet reached or got dropped
            # increase path length by one.
            np_lengths[s_idx] += 1
            current_node = index_node[int(cl_idx)]
            next_node = current_node
            dst_node = index_node[np_destinations_idx[s_idx]]

            if np_predictions[s_idx] >= len(index_value[current_node]) + 1:
                # If output is larger than the nodes has neighbors stay on the
                # current node.
                next_node_idx = cl_idx
                logger.debug("{:4d})\t{}: Chose non-existing neighbor with p={} | dst is {}.".format(
                    s_idx,
                    index_node[cl_idx],
                    probs[s_idx, n_idx],
                    index_node[np_destinations_idx[s_idx]]
                ))
            elif n_idx == 0 and np_targets[s_idx, 0] == 0:
                # Packet gets dropped although it should not. Stick with current
                # location and set finished to one.
                logger.debug("{:4d})\t{}: Dropped although should not with p={} | dst is {}.".format(
                    s_idx,
                    index_node[cl_idx],
                    probs[s_idx, n_idx],
                    index_node[np_destinations_idx[s_idx]]
                ))
                np_wrong_drops[s_idx] += 1
                np_is_finished[s_idx] = 1
                next_node_idx = cl_idx
            elif n_idx == 0 and np_targets[s_idx, 0] > 0:
                # Packet gets correctly dropped.
                logger.debug("{:4d})\t{}: Correctly dropped with p={} | dst is {}.".format(
                    s_idx,
                    index_node[cl_idx],
                    probs[s_idx, n_idx],
                    index_node[np_destinations_idx[s_idx]]
                ))
                np_is_finished[s_idx] = 1
                np_correct_drops[s_idx] = 1
                next_node_idx = cl_idx
            else:
                # Correct the n_idx by shift of one due to introduction of
                # drop node.
                next_node = index_value[current_node][int(n_idx) - 1]
                if (current_node, next_node) in failed_links:# or (next_node, current_node) in failed_links:
                    logger.debug("{:4d})\t{}: Tried to use failed edge to {} with p={} | dst is {}".format(
                        s_idx,
                        index_node[cl_idx],
                        next_node,
                        probs[s_idx, n_idx],
                        index_node[np_destinations_idx[s_idx]]
                    ))
                    next_node = current_node
                elif graph.nodes[next_node]['idx'] == int(np_destinations_idx[s_idx]):
                    # check if the next node is the destination.
                    if is_graph_weighted:
                        if output_mode == 'wcmp':
                            np_distances[s_idx] = aggregate(np_distances[s_idx], kl_divergence[s_idx])
                        else:
                            np_distances[s_idx] = aggregate(np_distances[s_idx], graph.edges[current_node, next_node][lf_prep.EDGE_WEIGHT])
                    np_is_finished[s_idx] = 1
                    np_reached_dst[s_idx] = 1
                    logger.debug("{:4d})\t{}: Reached dst with p={} | dst is {}.".format(
                        s_idx,
                        index_node[cl_idx],
                        probs[s_idx, n_idx],
                        index_node[np_destinations_idx[s_idx]]
                    ))
                elif dst_node.startswith('h-') and graph.has_edge(next_node, dst_node) \
                        and np_lengths[s_idx] < max_num_steps - 1:
                    np_is_finished[s_idx] = 1
                    np_reached_dst[s_idx] = 1
                    np_lengths[s_idx] += 1
                    if is_graph_weighted:
                        if output_mode == 'wcmp':
                            np_distances[s_idx] = aggregate(np_distances[s_idx], kl_divergence[s_idx])
                        else:
                            np_distances[s_idx] = aggregate(np_distances[s_idx], graph.edges[current_node, next_node][lf_prep.EDGE_WEIGHT])
                            np_distances[s_idx] = aggregate(np_distances[s_idx], graph.edges[next_node, dst_node][lf_prep.EDGE_WEIGHT])
                    logger.debug("{:4d})\t{}: Next hop is {} with p={}, weight={} + {}, connected to dst | dst is {}.".format(
                        s_idx,
                        index_node[cl_idx],
                        next_node,
                        probs[s_idx, n_idx],
                        graph.edges[current_node, next_node][lf_prep.EDGE_WEIGHT] if is_graph_weighted else -1,
                        graph.edges[next_node, dst_node][lf_prep.EDGE_WEIGHT] if is_graph_weighted else -1,
                        index_node[np_destinations_idx[s_idx]]
                    ))
                elif next_node.startswith('h-'):
                    logger.debug("{:4d})\t{}: Next would be {} with p={}, stay on current node. | dst is {}.".format(
                        s_idx,
                        index_node[cl_idx],
                        next_node,
                        probs[s_idx, n_idx],
                        index_node[np_destinations_idx[s_idx]]
                    ))
                    next_node = current_node
                else:
                    if is_graph_weighted:
                        if output_mode == 'wcmp':
                            np_distances[s_idx] = aggregate(np_distances[s_idx], kl_divergence[s_idx])
                        else:
                            np_distances[s_idx] = aggregate(np_distances[s_idx], graph.edges[current_node, next_node][lf_prep.EDGE_WEIGHT])
                    logger.debug("{:4d})\t{}: Moved to {} with p={:.4f} weight {:.4f} | dst is {}.".format(
                        s_idx,
                        index_node[cl_idx],
                        next_node,
                        float(probs[s_idx, n_idx]),
                        float(graph.edges[current_node, next_node][lf_prep.EDGE_WEIGHT]) if is_graph_weighted else -1,
                        index_node[np_destinations_idx[s_idx]]
                    ))
                next_node_idx = graph.nodes[next_node]['idx']
            np_cur_locs_idx[s_idx] = next_node_idx
            pairs[s_idx] = (next_node, pairs[s_idx][1])
        if np.sum(np_is_finished) == np_is_finished.size:
            # Pairs only get added if destinations have not been reached. Thus,
            # if pairs are empty, we are done.
            break
        np_targets = _new_targets(
            graph=graph,
            pairs=pairs,
            num_non_leaves=num_non_leaves,
            value_index=value_index,
            num_outputs=num_outputs,
            output_mode=output_mode
        )
    mask = np.logical_and(
        np_is_finished == 1,
        np.logical_not(np.logical_or(np_to_drop, np_wrong_drops))
    )
    opt_lengths = np.array([adj[u][v] for u, v in initial_pairs])[mask]
    np_lengths = np_lengths[mask]

    opt_distances = _driver_opt_lengths(graph, mask, initial_pairs, output_mode)
    rnd_distances = _driver_random_lengths(graph, mask, seed, initial_pairs, output_mode)
    np_distances = np_distances[mask]

    if output_mode == 'wcmp':
        np_distances = np_distances / np_lengths

    logger.info("Average length: {}, optimal: {}".format(
        np.mean(np_lengths),
        np.mean(opt_lengths)
    ))
    if is_graph_weighted:
        print(", ".join(["{:.4f}".format(x) for x in opt_distances]))
        print(", ".join(["{:.4f}".format(x) for x in rnd_distances]))
        print(", ".join(["{:.4f}".format(x) for x in np_distances]))
        logger.info("Average distance: {}, optimal: {}, random: {}".format(
            np.mean(np_distances),
            np.mean(opt_distances),
            np.mean(rnd_distances)
        ))
    logger.info("Finished {:d} of {:d}".format(
        int(np.sum(np_reached_dst)),
        np_is_finished.size
    ))
    logger.info("Wrongly dropped {:d} of {:d}".format(
        int(np.sum(np_wrong_drops)),
        np_wrong_drops.size
    ))
    logger.info("Dropped {} of {} correctly.".format(
        np.sum(np_correct_drops),
        np.sum(np_to_drop)
    ))
    graph.add_edges_from(failed_links)
    return opt_lengths, np_lengths, np_is_finished, np_wrong_drops, \
           np_correct_drops, np_to_drop, np_distances, opt_distances, rnd_distances


def _expected_path_lengths(graph: nx.Graph, sources: List[str], destination: str) -> np.array:
    expected_path_lengths = []
    t = destination
    max_weight = np.max([d[EDGE_WEIGHT] for u, v, d in graph.edges(data=True) if EDGE_WEIGHT in d])
    for s in sources:
        if graph.has_edge(s, t): continue
        try:
            expectation = 0
            for path in nx.all_shortest_paths(graph, s, t):
                weight = 0.
                prob = 1.
                for u, v in zip(path[:-2], path[1:-1]):
                    weight = np.max([weight, graph.edges[u, v][EDGE_WEIGHT] / max_weight])
                    prob *= graph.edges[u, v]['probability']
                expectation += prob * weight
            expected_path_lengths.append(expectation)
        except nx.NetworkXNoPath:
            pass
    return np.array(expected_path_lengths)


def _kl_divs(graph: nx.Graph, graph_target: nx.Graph, graph_pred: nx.Graph, destination: str) -> np.array:
    expected_path_lengths = []
    t = destination
    for u in graph.nodes():
        if graph.has_edge(u, t): continue
        if graph.degree[u] == 0: continue
        div = 0
        for v in graph.neighbors(u):
            p = graph_pred.edges[u, v]['probability'] + 1e-6
            q = graph_target.edges[u, v]['probability'] + 1e-6
            div += p * np.log(p / q)
        expected_path_lengths.append(div)
    return np.array(expected_path_lengths)


def _add_probabilities(graph: nx.Graph, pairs: List[Tuple[str, str]], probs: np.array,
                       value_index: Dict[str, Dict[str, int]]) -> nx.Graph:
    for i, (u, t) in enumerate(pairs):
        for v, idx in value_index[u].items():
            if graph.has_edge(u, v):
                graph.edges[u, v]['probability'] = probs[i, idx + 1]
    return graph


def _splitting_ratios_nn(model, graph: nx.Graph, sources: List[str],
                         destinations: List[str], k: int,
                         embeddings: np.array) -> Tuple[np.array, np.array, np.array]:
    nns = [n for n, d in graph.nodes(data=True) if sp_prep.NS_IDX in d]
    num_non_leaves = len(nns)
    num_outputs = k + 1
    value_index = sp_prep._neighbor_to_index(graph)
    get_activations = GetActivationsForDriver(graph, './data/fat-tree-k8')
    e_nn = np.array([])
    e_t = np.array([])
    e_ecmp = np.array([])

    nodes_with_state = np.zeros(num_non_leaves, dtype=np.int32)
    for n, d in graph.nodes(data=True):
        if sp_prep.NS_IDX in d:
            nodes_with_state[d[sp_prep.NS_IDX]] = d[sp_prep.IDX]

    all_masks = np.expand_dims(
        lf_prep._get_all_masks(
            graph=graph,
            max_degree=num_outputs,
            failed_links=[],
            neighbor_to_index=value_index
        ),
        axis=0
    )
    masks_nodes_with_state = all_masks[:, nodes_with_state, :, :]
    all_neighbors = lf_prep._get_all_neighbors(
        graph=graph,
        max_degree=num_outputs,
        neighbor_to_index=value_index
    )
    embeddings_nodes_with_states = np.expand_dims(embeddings[nodes_with_state], 0)

    for t in destinations:
        print("Obtain weights for ", t)
        pairs = [(u, t) for u in nns if not graph.has_edge(u, t)]
        states, targets, targets_ecmp, dsts, cur_locs_idx = _create_initial_input(
            graph=graph,
            output_mode='wcmp',
            num_outputs=k,
            num_non_leaves=len(nns),
            value_index=value_index,
            pairs=pairs
        )
        neighbor_masks = all_masks[0, cur_locs_idx, :, :]
        embd_neighbors = all_neighbors[cur_locs_idx, :, :].astype(np.int32)
        embd_neighbors = embeddings[embd_neighbors.flatten(), :].reshape(
            -1, all_neighbors.shape[1], embeddings.shape[-1]
        )
        embd_cur_locs = embeddings[cur_locs_idx, :]
        embd_dsts = embeddings[dsts, :]
        attn_actis = np.concatenate([get_activations(dst, cloc) for cloc, dst in pairs])

        t_neighbor_masks   = torch.tensor(neighbor_masks)
        t_embd_neighbors   = torch.tensor(embd_neighbors)
        t_states           = torch.tensor(states)
        t_all_masks        = torch.tensor(masks_nodes_with_state)
        t_embd_cur_loc     = torch.tensor(embd_cur_locs)
        t_embd_destination = torch.tensor(embd_dsts)
        t_embd_n_w_s       = torch.tensor(embeddings_nodes_with_states)
        t_attn_actis = torch.tensor(attn_actis)

        with torch.no_grad():
            preds, preds_ecmp, _ = model.forward(
                network_state=t_states.to(device=DEV),
                network_state_mask=t_all_masks.to(device=DEV),
                embeddings_neighbors=t_embd_neighbors.to(device=DEV),
                mask_embeddings=t_neighbor_masks.to(device=DEV),
                embd_current_location=t_embd_cur_loc.to(device=DEV),
                embd_destination=t_embd_destination.to(device=DEV),
                embeddings=t_embd_n_w_s.to(device=DEV),
                embd_nodes_state=t_states.to(device=DEV)  # t_embd_nodes_w_state.to(device=DEV)
                # hlsa_attn_head_activations=t_attn_actis.to(device=DEV)
            )
            # Immediately detach resulst. Else, the computational graph gets
            # stored as well. Also contributing towards potential OOM.
            predictions = preds.detach()
            predictions_ecmp = preds_ecmp.detach()

            # probs_ecmp = torch.sigmoid(predictions_ecmp)
            # probs_ecmp = torch.softmax(predictions_ecmp, dim=-1)
            # probs_ecmp = probs_ecmp > 0.1
            # probs_ecmp = probs_ecmp.type_as(predictions)
            # probs = torch.add(predictions, (probs_ecmp - 1) * 1e6)
            probs = predictions
            probs = torch.softmax(probs, axis=-1).numpy()
            targets_ecmp = targets_ecmp / np.expand_dims(np.sum(targets_ecmp, axis=-1), axis=-1)
            # e_nn = np.concatenate([
            #     e_nn,
            #     _expected_path_lengths(_add_probabilities(graph, pairs, probs, value_index), sources, t)
            # ])
            e_t = np.concatenate([
                e_t,
                _expected_path_lengths(_add_probabilities(graph, pairs, targets, value_index), sources, t)
            ])
            # e_ecmp = np.concatenate([
            #     e_ecmp,
            #     _expected_path_lengths(_add_probabilities(graph, pairs, targets_ecmp, value_index), sources, t)
            # ])
            e_nn = np.concatenate([e_nn, np.sum(probs * np.log(probs / (targets + 1e-6) + 1e-6), axis=1)])

            targets_e = targets.copy()
            targets_e[targets_e > 0] = 1
            targets_e /= np.expand_dims(np.sum(targets_e, axis=1), axis=1)
            e_ecmp = np.concatenate([e_ecmp, np.sum(targets_e * np.log(targets_e / (targets + 1e-6) + 1e-6), axis=1)])
    return e_nn, e_t, e_ecmp


def run_splitting_ratios(logdir, dsetpath: str, filter: str, result_path='/opt/project/data'):
    model = evutils.load_model(StatefulConfig, StatefulModel, logdir, max_num_blocks=MAX_NUM_BLOCKS)
    model.to(DEV)
    model.eval()
    random = np.random.RandomState(seed=3)
    max_num_failures = 2

    def get_graph() -> nx.Graph:
        graph = lf_prep._add_gaussian_edge_weights(sp_prep.add_index_to_nodes(make_topo(8)), random)
        if random.uniform() > 10.5:
            failed_links = lf_prep._sample_failure_fat_tree(random, max_num_failures, 8, graph)
            for u, v in lf_prep._sample_node_failure_fat_tree(random, 1, 8, graph):
                if (u, v) not in failed_links:
                    failed_links.append((u, v))
            graph.remove_edges_from(failed_links)
        return graph

    def add_results(f, name, data):
        g = f.create_group(name)
        for i, run in enumerate(data):
            g.create_dataset('run_{:d}'.format(i), data=run)

    embeddings = read_embeddings(os.path.join(dsetpath, 'fat-tree-k8-ip-embedding.h5'))
    sources = ['tor-{:04d}'.format(i) for i in range(32)]
    destinations = ['h-{:04d}'.format(i) for i in range(128)]
    a, b, c = zip(*[_splitting_ratios_nn(
        model,
        get_graph(),
        sources,
        destinations,
        8,
        embeddings
    ) for _ in range(10)])

    f = h5py.File(os.path.join(result_path, 'wcmp-results-uni.h5'), 'w')
    add_results(f, 'rexm', a)
    add_results(f, 'opt', b)
    add_results(f, 'ecmp', c)
    f.close()

    fig, ax = present.get_fig(1)
    present.compare_cdfs(
        cdfs=[
            [present._make_cdf(x) for x in c],
            # [present._make_cdf(x) for x in b],
            [present._make_cdf(x) for x in a]
        ],
        xlabel="Expected Bandwidth",
        ylabel="P(X < x)",
        labels=['ECMP', 'MISTILL'],
        ax=ax,
        alpha=0.7
    )
    # ax.set_xscale('log')
    present.save_fig(
        folder=result_path,
        name='wcmp_cdfs_expected_weight',
        format='pdf',
        fig=fig
    )
    return a, b, c


def random_driver(graph: nx.Graph, pairs, output_mode, seed):
    def kl_div(cur_loc, dst):
        output_generator = lf_prep.OutputCalculator(graph, cur_loc, dst)
        target = output_generator._get_output_wcmp(8, value_index)
        tmp = target.copy()
        tmp[tmp > 0] = 1.
        tmp = tmp / np.expand_dims(np.sum(tmp, axis=1), axis=1)
        kldiv = np.sum(target * np.log(np.clip(target, 1e-3, 1) / np.clip(tmp, 1e-3, 1)))
        return kldiv

    value_index = sp_prep._neighbor_to_index(graph)
    random = np.random.RandomState(seed=seed)
    finished = []
    lengths = []
    dists = []
    aggregate = {
        'hula': lambda a, b: float(np.max([a, b])),
        'lcp': lambda a, b: a + b,
        'wcmp': lambda a, b: a + b
    }[output_mode]
    for src, dst in pairs:
        cur_loc = src
        num_steps = 0
        dist = 0
        while cur_loc != dst and num_steps < 12:
            num_steps += 1
            next_loc = random.choice(list(graph.neighbors(cur_loc)))
            if next_loc.startswith('h-'):
                # Do not increase distance traveled and stay on current node
                # but keep num steps increased by one.
                dist += 0
                next_loc = cur_loc
            elif graph.has_edge(next_loc, dst) and num_steps < 12:
                # If the next location is directly connected to the destination
                # fast forward packet to destination. But only if budget still
                # contains steps. Increase dist by length from current location
                # to next location, and from next location to destination.
                # also increase number of steps to account for additional hop.
                if output_mode == 'wcmp':
                    dist = aggregate(dist, kl_div(cur_loc, dst))
                else:
                    dist = aggregate(dist, graph.edges[cur_loc, next_loc][lf_prep.EDGE_WEIGHT])
                    dist = aggregate(dist, graph.edges[next_loc, dst][lf_prep.EDGE_WEIGHT])
                num_steps += 1
                cur_loc = dst
            else:
                # A transition somewhere in the graph. Increase distance by
                # edge traveled.
                if output_mode == 'wcmp':
                    dist = aggregate(dist, kl_div(cur_loc, dst))
                else:
                    dist = aggregate(dist, graph.edges[cur_loc, next_loc][lf_prep.EDGE_WEIGHT])
        x = 0
        if cur_loc == dst:
            x = 1
            lengths.append(num_steps)
            dists.append(dist)
        finished.append(x)
    if output_mode == 'wcmp':
        dists = np.array(dists) / np.array(lengths)
    else:
        dists = np.array(dists)
    return np.array(finished), np.array(lengths), dists


def get_pattern(target) -> Union[float, str]:
    if target[0] > 0:
        return 'drop'
    elif np.sum(target[1:] > 0.99) > 0:
        return 'down'
    elif np.sum(np.multiply(target > 0.124, target < 0.126)) > 0:
        return 0.125
    elif np.sum(np.multiply(target > 0.24, target < 0.26)) > 0:
        return 0.25
    elif np.sum(np.multiply(target > 0.32, target < 0.34)) > 0:
        return 0.33
    elif np.sum(np.multiply(target > 0.49, target < 0.51)) > 0:
        return 0.5
    elif np.sum(np.multiply(target > 0.14, target < 0.15)) > 0:
        return 0.146
    elif np.sum(np.multiply(target > 0.16, target < 0.17)) > 0:
        return 0.166
    elif np.sum(np.multiply(target > 0.19, target < 0.21)) > 0:
        return 0.2
    else:
        return 0


def eval_patterns(a, dsetpath, b) -> None:
    total = 0
    patterns = {}
    ds = read_link_failure_data(os.path.join(dsetpath, 'val', 'link-failure-data-26.h5'))
    for i in range(ds['targets'].shape[0]):
        pattern = get_pattern(ds['targets'][i, :])
        if pattern == 0:
            print(ds['targets'][i, :])
        if pattern not in patterns:
            patterns[pattern] = 0.
        patterns[pattern] += 1
        total += 1

    print("Fractions: ")
    for k, v in patterns.items():
        print("{}: {}".format(k, v / total))
    print("Weights: ")
    for k, v in patterns.items():
        print("{}: {}".format(k, 1. / (v / total)))


def baseline_eval(model: StatefulModel, dsetpath: str, filter: str) -> None:
    """
        Make and print predictions from the validation set to get an idea how
        well the model learned.
    """
    x = input("Evaluate ECMP (Y/N)?: ")
    if x.lower() == 'y':
        eval_ecmp = True
    else:
        eval_ecmp = False
    k = 8
    if 'fat-tree-k16' in dsetpath:
        k = 16
    ds = StatefulDataset.from_hdf5(
        full_path_templates=os.path.join(dsetpath, 'val/link-failure-data-26.h5'),
        full_path_embeddings=os.path.join(dsetpath, 'fat-tree-k{:d}-ip-embedding.h5'.format(k)),
        policy=model.config.policy
    )
    graph = read_graph(os.path.join(dsetpath, 'fat-tree-k{:d}.json'.format(k)))
    if filter is not None:
        ds = filter_dataset(
            ds=ds,
            graph=graph,
            prefix=filter
        )
    index_to_name = {d[sp_prep.IDX]: n for n, d in graph.nodes(data=True)}
    print("Number of dforps in ds", ds.targets[:, 0].sum(), " of ", ds.targets.shape[0])
    print("Number of triplets: ", np.sum(np.logical_and(ds.targets > 0.32, ds.targets < 0.34)) / 3)
    print("Number of halfs: ", np.sum(np.logical_and(ds.targets > 0.49, ds.targets < 0.51)) / 2)
    print("Number of quarteres: ", np.sum(np.logical_and(ds.targets > 0.24, ds.targets < 0.251)) / 4)
    print("Number of downstreams: ", np.sum(ds.targets[1:, :] > 0.99))
    print("Number of super broad", np.sum(np.logical_and(ds.targets > 0.12, ds.targets < 0.13)) / 8)
    ds.use_embeddings_as_queries_for_attention_over_links = True
    ds.binarize_targets = model.config.multiclass
    print("Set binarization to: {}".format(model.config.multiclass))
    ds.load_weights(dsetpath)
    loader = DataLoader(ds, batch_size=10)
    losses = []
    optimal_losses = []
    loss_fct = multi_class_loss if model.config.multiclass else full_cross_entropy
    transform = lambda x: torch.log(x / (1. - x + 1e-3)) if model.config.multiclass \
        else lambda x: torch.log(x + 1e-6)
    time.sleep(5)
    with torch.no_grad():
        for batch, sample in enumerate(loader):
            # print(sample['network_state'].shape)
            # print(sample['embd_nodes_state'].shape)
            preds, preds_ecmp, _ = model.forward(
                network_state=sample["network_state"].to(DEV),
                network_state_mask=sample["network_masks"].to(DEV),
                embeddings_neighbors=sample["neighbors"].to(DEV),
                mask_embeddings=sample["neighbor_mask"].to(DEV),
                embd_current_location=sample[model.config.hlsa_attn_key].to(DEV),
                embd_destination=sample["destination"].to(DEV),
                embeddings=sample['embeddings'].to(DEV),
                embd_nodes_state=sample['embd_nodes_state'].to(DEV)
                # hlsa_attn_head_activations=sample['hlsa_attn_activations']
            )
            preds_ecmp.detach()
            preds.detach()
            if eval_ecmp:
                preds = preds_ecmp
                target = sample['target_ecmp']
            else:
                target = sample['target']
            losses.append(loss_fct(
                logits=preds,
                target=target.to(DEV),
                weights=torch.tensor([[1.]]).to(DEV)
            ).to('cpu').numpy() + 1e-6)
            if model.config.multiclass:
                optimal_losses.append(_multi_class_loss(
                    probs=torch.clamp(target, 0.05, 0.95),
                    target=target,
                    weights=torch.tensor([[1.]])
                ).numpy() + 1e-6)
            else:
                optimal_losses.append(loss_fct(
                    logits=torch.log(target + 1e-6),
                    target=target,
                    weights=torch.tensor([[1.]])
                    ).to('cpu').numpy()
                )
            # if np.sum(sample['target'].numpy() == 0) in [6]:
            # if sample['target'].numpy()[0, 0] > 0:
            if losses[-1] > 0:
                logger.debug("Has a loss of {}, optimal: {}".format(losses[-1], optimal_losses[-1]))
                if model.config.multiclass:
                    probs = torch.sigmoid(preds).to('cpu').numpy()
                else:
                    probs = torch.softmax(preds, axis=-1).to('cpu').numpy()
                logger.debug("{:s}".format(_matrix_to_str_adv(
                    probs,
                    target,
                    sample['current_loc_idx'].numpy(),
                    sample['destination_idx'].numpy(),
                    index_to_name
                )))
            if batch > 500:
                break
        logger.debug('Average loss of model over full file is  : {}'.format(np.mean(losses)))
        logger.debug('Average loss of optimal over full file is: {}'.format(np.mean(optimal_losses)))


def run_baseline(logdir, dsetpath: str, filter: str):
    model = evutils.load_model(StatefulConfig, StatefulModel, logdir, max_num_blocks=MAX_NUM_BLOCKS)
    model.to(DEV)
    model.eval()
    baseline_eval(model, dsetpath, filter)


def run_driver(logdir, dsetpath: str, filter: str, result_path='/opt/project/data'):
    def make_pairs():
        pairs = []
        for tor_id in range(0, 32):
        # for tor_id in random.randint(0, 32, size=5):
            for i in range(0, 128):
                if not (tor_id * 4 <= i < (tor_id + 1) * 4):
                    pairs.append(("tor-{:04d}".format(tor_id), 'h-{:04d}'.format(i)))
        return pairs

    def create_group(f, arr, name):
        g = f.create_group(name)
        for i, a in enumerate(arr):
            g.create_dataset("run_{:d}".format(i + 1), data=a)

    print(f"Load model from {logdir}")
    model = evutils.load_model(StatefulConfig, StatefulModel, logdir, max_num_blocks=MAX_NUM_BLOCKS)
    print("hlsa_model is: ", str(model.config.hlsa_model))

    output_mode = ''
    while output_mode not in ['hula', 'lcp', 'wcmp']:
        output_mode = input("Choose aggregation (hula/lcp/wcmp): ")

    if output_mode == 'wcmp':
        run_splitting_ratios(logdir, dsetpath, filter, result_path)
    else:
        model.to(DEV)
        model.eval()
        random = np.random.RandomState(seed=1)
        embeddings = read_embeddings(os.path.join(dsetpath, 'fat-tree-k8-ip-embedding.h5'))
        pairs = [
            ('tor-0000', 'tor-0012'),
            ('tor-0000', 'agg-0012'),
            ('tor-0000', 'h-0064'),
            ('tor-0000', 'core-0008'),
            ('agg-0000', 'h-0064'),
            ('agg-0000', 'tor-0012'),
            ('agg-0000', 'agg-0001'),
            ('agg-0000', 'agg-0012'),
            ('agg-0000', 'core-0005'),
            ('core-0000', 'h-0064'),
            ('core-0000', 'tor-0012'),
            ('core-0000', 'agg-0012'),
            ('core-0000', 'core-0001')
        ]
        pairs = [('tor-0008', 'h-{:04d}'.format(i)) for i in range(4, 128)]
        pairs = [('tor-0008', 'h-{:04d}'.format(i)) for i in range(4, 32)]
        # baseline_eval(model)

        num_runs = 10
        opt_lengths, lengths, finished, wrong_drops, correct_drops, to_drop, dist, opt_dist, ecmp_dist = zip(*[driver(
            graph=lf_prep._add_gaussian_edge_weights(sp_prep.add_index_to_nodes(make_topo(8)), random),
            model=model,
            embeddings=embeddings,
            max_num_failures=10,
            seed=i + 1,
            output_mode=output_mode,
            pairs=make_pairs(),
            mode='argmax'
        ) for i in range(num_runs)])

        random = np.random.RandomState(seed=1)
        rnd_finished, rnd_lengths, rnd_dists = zip(*[random_driver(
            graph=lf_prep._add_gaussian_edge_weights(sp_prep.add_index_to_nodes(make_topo(8)), random),
            pairs=make_pairs(),
            output_mode=output_mode,
            seed=i + 1
        ) for i in range(num_runs)])
        print("Lengths:     ", ["{:.4f}".format(np.mean(l)) for i, l in enumerate(lengths)])
        print("opt_lengths: ", ["{:.4f}".format(np.mean(l)) for i, l in enumerate(opt_lengths)])
        print()

        print("dist:     ", ["{:.4f}".format(np.mean(l)) for i, l in enumerate(dist)])
        print("opt_dist: ", ["{:.4f}".format(np.mean(l)) for i, l in enumerate(opt_dist)])
        print("rnd_dist: ", ["{:.4f}".format(np.mean(l)) for i, l in enumerate(ecmp_dist)])
        print()

        print("fraction_finished: ", ["{:.4f}".format(np.mean(l)) for l in finished])
        tmp = np.array([np.mean(np.add(f, t) > 0.1) for f, t in zip(finished, to_drop)])
        print("finished accounting for dropped: {}".format(tmp))
        print("Correct vs. required drops vs. wrong drops: {} - {} - {}".format(
            np.sum([np.sum(cd) for cd in correct_drops]),
            np.sum([np.sum(td) for td in to_drop]),
            np.sum([np.sum(td) for td in wrong_drops])
        ))

        f = h5py.File(os.path.join(result_path, 'driver-results.h5'), 'w')
        create_group(f, opt_lengths, "optimal_lengths")
        create_group(f, lengths, "llsrp_lengths")
        create_group(f, finished, "is_finished")
        create_group(f, wrong_drops, "wrong_drops")
        create_group(f, correct_drops, "correct_drops")
        create_group(f, to_drop, "to_drop")
        create_group(f, dist, "llsrp_path_weights")
        create_group(f, opt_dist, "opt_path_weights")
        create_group(f, ecmp_dist, "ecmp_path_weights")
        create_group(f, rnd_finished, "rnd_finished")
        create_group(f, rnd_lengths, "rnd_lengths")
        create_group(f, rnd_dists, "rnd_dists")
        f.close()

        fig, ax = present.get_fig(1)
        present.compare_cdfs(
            cdfs=[
                [present._make_cdf(x) for x in opt_lengths],
                [present._make_cdf(x) for x in lengths],
            ],
            xlabel="Length [Hops]",
            ylabel="P(X < x)",
            labels=['OPT', 'RExM'],
            ax=ax,
            alpha=0.7
        )
        present.save_fig(
            folder=result_path,
            name='cdfs_lengths',
            format='pdf',
            fig=fig
        )
        fig, ax = present.get_fig(1)
        present.compare_cdfs(
            cdfs=[
                [present._make_cdf(x) for x in ecmp_dist],
                [present._make_cdf(x) for x in opt_dist],
                [present._make_cdf(x) for x in dist]
            ],
            xlabel="Metric",
            ylabel="P(X < x)",
            labels=['ECMP', 'OPT', 'RExM'],
            ax=ax,
            alpha=0.7
        )
        present.save_fig(
            folder=result_path,
            name='cdfs_weights',
            format='pdf',
            fig=fig
        )


def run_compute_hlsas(logdir, dsetpath: str, filter: str):
    model = evutils.load_model(StatefulConfig, StatefulModel, logdir, max_num_blocks=MAX_NUM_BLOCKS)
    if model.config.hlsa_model == 'fcn':
        frame = compute_hlsas(model)
    else:
        frame =compute_hlsas_attn(model, dsetpath)
    frame.to_hdf('/opt/project/data/hlsas.h5', key='hlsas')


def run_scores(logdir, dsetpath: str, filter: str):
    model = evutils.load_model(StatefulConfig, StatefulModel, logdir, max_num_blocks=MAX_NUM_BLOCKS)
    graph = read_graph(os.path.join(dsetpath, 'fat-tree-k8.json'))
    hosts = ['h-{:04d}'.format(i) for i in range(128)]
    if os.path.exists('/opt/project/data/scores'):
        pass
    else:
        os.mkdir('/opt/project/data/scores')
    for host in hosts:
        scores = get_attention_scores(
            model=model,
            graph=graph,
            embeddings_path=os.path.join(dsetpath, 'fat-tree-k8-ip-embedding.h5'),
            dataset_path=os.path.join(dsetpath, 'val/link-failure-data-26.h5'),
            destination=host
        )
        print(scores[0].head())
        print(scores[0].shape)
        print(len(scores))
        for i, df in enumerate(scores):
            df.to_hdf('/opt/project/data/scores/scores-mha-{}.h5'.format(host), key='head{:d}'.format(i))
    subprocess.run("tar -C /opt/project/data -czf scores-mha-x.tar.gz scores/", shell=True)
    shutil.rmtree('/opt/project/data/scores')
    shutil.move('/opt/project/scores-mha-x.tar.gz', '/opt/project/data/scores-mha-x.tar.gz')


def _get_converged_scores(graph, logdir, host, dsetpath):
    from present import _converge_dfs
    model = evutils.load_model(StatefulConfig, StatefulModel, logdir, max_num_blocks=MAX_NUM_BLOCKS)
    scores = get_attention_scores(
        model=model,
        graph=graph,
        embeddings_path=os.path.join(dsetpath, 'fat-tree-k8-ip-embedding.h5'),
        dataset_path=os.path.join(dsetpath, 'val/link-failure-data-26.h5'),
        destination=host
    )
    return _converge_dfs(scores)


def converge_scores(logdir: str, dsetpath: str, filter: str):
    from present import _converge_dfs
    exp_dir, pref = os.path.split(logdir)
    print(exp_dir, pref)
    pref = pref[:-2]

    logdirs = evutils.get_logdirs_best_models(exp_dir, 4., pref)
    hosts = ['h-0000']
    graph = read_graph(os.path.join(dsetpath, 'fat-tree-k8.json'))
    for host in hosts:
        converged = []
        for logdir in logdirs:
            model = evutils.load_model(StatefulConfig, StatefulModel, logdir, max_num_blocks=MAX_NUM_BLOCKS)
            scores = get_attention_scores(
                model=model,
                graph=graph,
                embeddings_path=os.path.join(dsetpath, 'fat-tree-k8-ip-embedding.h5'),
                dataset_path=os.path.join(dsetpath, 'val/link-failure-data-26.h5'),
                destination=host
            )
            converged.append(_converge_dfs(scores))
        final = _converge_dfs(converged) / len(converged)
        final.to_hdf('/opt/project/data/{}-average-converged-scores.h5', key='scores')


def run_scores_neighbors(logdir, dsetpath: str, filter: str):
    model = evutils.load_model(StatefulConfig, StatefulModel, logdir)
    graph = read_graph(os.path.join(dsetpath, 'fat-tree-k8.json'))
    frames = get_attn_scores_nghbs(
        model=model,
        graph=graph,
        embeddings_path=os.path.join(dsetpath, 'fat-tree-k8-ip-embedding.h5'),
    )
    for k, df in frames.items():
        df.to_hdf('/opt/project/data/scores-neighbors.h5', key=k)


def get_val_loss_progress(logdir: str, dsetpath: str, filter: str) -> None:
    df = pd.read_csv(os.path.join(logdir, 'progress.csv'))
    print("Iter\ttrain      \tVal        \ttrain-single\ttrain-ecmp  \tval-single  \tval-ecmp    ")
    for idx, row in df.iterrows():
        print("{:3d}\t{:12.6f}\t{:12.6f}".format(idx, row['cross_entropy'], row['cross_entropy-val']), end='')
        if 'cross_entropy_single' in df.columns:
            print("{:12.6f}\t{:12.6f}\t{:12.6f}\t{:12.6f}".format(
                row['cross_entropy_single'], row['cross_entropy_ecmp'],
                row['cross_entropy_single-val'], row['cross_entropy_ecmp-val']
            ))
        else:
            print()


def print_params(logdir: str, dsetpath: str, filter: str) -> None:
    with open(os.path.join(logdir, 'params.json'), 'r') as fh:
        print(json.dumps(expand_stateful_config(json.load(fh)), indent=1))


def print_num_model_params(logdir: str, dsetpath: str, filter: str) -> None:
    model = evutils.load_model(StatefulConfig, StatefulModel, logdir)
    print("Number of parameters: ", evutils.get_num_params(model))


def user_input_based():
    while True:
        experiment_path = '/opt/project/data/training-results'
        experiments = []
        for f in os.listdir(experiment_path):
            # part = f[:-2]
            part = f
            if part in experiments or part[-3] != '0':
                continue
            else:
                experiments.append(part)
        experiments.sort()
        options = ''
        for i, e in enumerate(experiments):
            options = "{}\n\t{:2d}) {:s}".format(options, i + 1, e)
        exp_num = int(input("Choose experiment: {}\nYour choice: ".format(options)))
        prefix = experiments[exp_num - 1]
        print("You choose {:d}) {}".format(exp_num, prefix))
        ldir = evutils.get_logdir_best_model(experiment_path, prefix, 'cross_entropy-val')

        dset_path = '/opt/project/data/fat-tree-k8'
        # This is the correct path at the moment with the most up to date data.
        dset_path = os.path.join(dset_path, "tors-to-hosts")
        # if prefix.find("Host2Host") > 0:
        #     dset_path = os.path.join(dset_path, 'link-failures-all-hosts')
        # if prefix.find("Gs") > 0:
        #     dset_path = os.path.join(dset_path, "tors-to-hosts")
        # elif prefix[-1] == '0':
        #     dset_path = os.path.join(dset_path, 'tors-to-hosts')
        # elif prefix.startswith("FatTreeK16IpEmbeddingPrepWeightsTors2HostsLcp"):
        #     dset_path = os.path.join(os.path.split(dset_path)[0], 'fat-tree-k16', 'tors-to-hosts-lcp')
        # elif prefix.startswith("FatTreeK16IpEmbeddingPrepWeightsTors2HostsHula") > -1:
        #     dset_path = os.path.join(os.path.split(dset_path)[0], 'fat-tree-k16', 'tors-to-hosts-hula')
        # elif prefix.startswith("FatTreeK16IpEmbeddingPrepWeightsTors2HostsWcmp") > -1:
        #     dset_path = os.path.join(os.path.split(dset_path)[0], 'fat-tree-k16', 'tors-to-hosts-wcmp')
        # elif prefix.startswith("FatTreeK8IpEmbeddingPrepWeightsTors2HostsHula") > 0:
        #     dset_path = os.path.join(dset_path, "tors-to-hosts-hula")
        # elif prefix.find("Wcmp") > 0:
        #     dset_path = os.path.join(dset_path, 'tors-to-hosts-llt-wcmp')
        # elif prefix.find("PrepWeights") > 0:
        #     dset_path = os.path.join(dset_path, 'tors-to-hosts-llt-ecmp-core')
        # elif prefix.find("WebPattern") > 0:
        #     dset_path = os.path.join(dset_path, "link-failures-web-pattern")
        # elif prefix.find("LltCECMP") > 0 or prefix.find("AndEcmptrain") > 0:
        #     dset_path = os.path.join(dset_path, 'tors-to-hosts-llt-ecmp-core')
        # elif prefix.find("Llt") > 0:
        #     dset_path = os.path.join(dset_path, 'tors-to-hosts-llt')
        # elif prefix.find("Weighted") > 0:
        #     dset_path = os.path.join(dset_path, 'tors-to-hosts-weighted')
        # elif prefix.find("Tors2Host") > 0:
        #     dset_path = os.path.join(dset_path, 'tors-to-hosts')
        # else:
        #     dset_path = os.path.join(dset_path, "link-failures-all-pairs")
        if prefix[-5:].find("Agg") > 0:
            filter = 'agg'
        elif prefix[-5:].find("Core") > 0:
            filter = 'core'
        elif prefix[-5:].find("Tor") > 0:
            filter = 'tor'
        else:
            filter = None
        print("Resulting data set is: {}.\nFilter is: {}".format(dset_path, filter))

        programs = {
            'scores HLSAs': run_scores,
            'scores neighbors': run_scores_neighbors,
            'compute hlsas': run_compute_hlsas,
            'driver': run_driver,
            'baseline': run_baseline,
            'progress': get_val_loss_progress,
            'params': print_params,
            'pattern': eval_patterns,
            "model params": print_num_model_params,
            "summary": print_trial_summary
        }
        keys = list(programs.keys())
        options = ''
        for i, e in enumerate(keys):
            options = "{}\n\t{:2d}) {:s}".format(options, i + 1, e)
        p_num = int(input("Choose program: {}\nYour choice: ".format(options)))
        print("You choose {:d}) {}".format(p_num, keys[p_num - 1]))
        programs[keys[p_num - 1]](ldir, dset_path, filter)


if __name__ == '__main__':
    # ldir = evutils.get_logdir_best_model('/opt/project/trained_models', 'FatTreeK8', 'cross_entropy-val')
    # run_scores(ldir)
    # run_driver(ldir)
    # run_baseline(ldir)
    # eval_patterns()
    user_input_based()




