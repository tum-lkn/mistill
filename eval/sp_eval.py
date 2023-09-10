"""
This module drives a simulation that evaluates for a trained model the possible
source and destination pairs. That is, how well the model can find the shortest
paths it was trained for.
"""
import torch
import numpy as np
import multiprocessing as mp
import ctypes
from typing import List, Dict, Tuple, Any
import networkx as nx
import itertools
import h5py
import os
import logging
logger = logging.getLogger('sp_eval')
logger.setLevel(logging.DEBUG)

import dataprep.sp_prep as sp_prep
from dataprep.input_output import read_graph, read_embeddings
from models.sponly import SpfConfig, SpfModel
from models.utils import full_cross_entropy
import eval.utils as evutils
import present

from dataprep.datasets import DistributionalSpfDataSet, DistributionalDataSetWithoutEmbedding
from torch.utils.data import DataLoader


DEV = torch.device("cpu")


VARS = {}
EMBEDDINGS = 'embeddings'
VALUE_INDEX = 'value_index'
INDEX_VALUE = 'index_value'
ADJACENCY = 'adj'
NODES = 'nodes'
EDGES = 'edges'
LENGTHS = 'lengths'
PREDICTIONS = 'predictions'
IS_FINISHED = 'is_finished'
SHAPE_TARGET = "shape_target"
SHAPE_NEIGHBORS = "shape_neighbors"
SHAPE_MASK = "shape_mask"
SHAPE_DESTINATION = "shape_destination"
SHAPE_EMBEDDINGS = "shape_embeddings"


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



def init_make_input(nodes, edges, adj, shape_targets, shape_neighbors, shape_masks,
                   shape_destinations, shape_embeddings, targets, neighbors, masks, destinations,
                   value_index, embeddings):
    """
        Initialize the global variable cache with shapes and array references.
    """
    VARS[ADJACENCY] = adj
    VARS[NODES] = nodes
    VARS[EDGES] = edges
    VARS[SHAPE_TARGET] = shape_targets
    VARS[SHAPE_NEIGHBORS] = shape_neighbors
    VARS[SHAPE_MASK] = shape_masks
    VARS[SHAPE_DESTINATION] = shape_destinations
    VARS[sp_prep.H5_TARGET] = targets
    VARS[sp_prep.H5_NEIGHBORS] = neighbors
    VARS[sp_prep.H5_MASK] = masks
    VARS[sp_prep.H5_DESTINATION] = destinations
    VARS[VALUE_INDEX] = value_index
    VARS[EMBEDDINGS] = embeddings
    VARS[SHAPE_EMBEDDINGS] = shape_embeddings


def init_process_predictions(predictions, is_finished, lengths, index_value):
    """
        Initialize global Dict with pointers and shared state for the processing
        of the prediction of the neural network
    """
    VARS[PREDICTIONS] = predictions
    VARS[IS_FINISHED] = is_finished
    VARS[LENGTHS] = lengths
    VARS[INDEX_VALUE] = index_value


def fill(args: List[Tuple[int, Any, Any, int]]) -> None:
    """
        Fill the placeholder arrays with input that should be passed to the
        neural network.

        Args:
            args: A list containing a tuple with the index, source, destination
                and number of outputs. The index indexes into the input array.
                Source and destination are the labels of the current graph and
                the destination node.
        """
    graph = nx.Graph()
    graph.add_nodes_from(VARS[NODES])
    graph.add_edges_from(VARS[EDGES])

    adj = VARS[ADJACENCY]
    value_index = VARS[VALUE_INDEX]


    np_targets = np.frombuffer(VARS[sp_prep.H5_TARGET]).reshape(VARS[SHAPE_TARGET])
    np_neighors = np.frombuffer(VARS[sp_prep.H5_NEIGHBORS]).reshape(VARS[SHAPE_NEIGHBORS])
    np_masks = np.frombuffer(VARS[sp_prep.H5_MASK]).reshape(VARS[SHAPE_MASK])
    np_destinations = np.frombuffer(VARS[sp_prep.H5_DESTINATION]).reshape(VARS[SHAPE_DESTINATION])
    np_embeddings = np.frombuffer(VARS[EMBEDDINGS]).reshape(VARS[SHAPE_EMBEDDINGS])

    # Zero pad the neighbors. Only the existing ones are returned which might
    # return in failure if the number of incident nodes does not equal the
    # maximum degree.
    for idx, u, dst, num_outputs in args:
        sample = sp_prep._get_sample(graph, u, dst, num_outputs, adj, value_index)
        nids = sample[sp_prep.H5_NEIGHBORS]
        if nids.size == VARS[SHAPE_NEIGHBORS][1]:
            tmp = np_embeddings[nids, :]
        else:
            tmp = np.zeros((VARS[SHAPE_NEIGHBORS][1] - nids.size, VARS[SHAPE_NEIGHBORS][-1]))
            tmp = np.concatenate((tmp, np_embeddings[nids, :]))

        np_targets[idx, :] = sample[sp_prep.H5_TARGET].astype(np.float64)
        np_neighors[idx, :, :] = tmp.astype(np.float64)
        np_masks[idx, :, :] = sample[sp_prep.H5_MASK].reshape(-1, 1).astype(np.float64)
        np_destinations[idx, :] = np_embeddings[sample[sp_prep.H5_DESTINATION], :].astype(np.float64)


def process_predictions(args: List[Tuple[int, Any, Any, int]]) -> List[Tuple[int, Any, Any, int]]:
    """
        Process the predictions and create new pairs based on that. That is,
        based on the predictions progress the source node one step.
    """
    np_predictions = np.frombuffer(VARS[PREDICTIONS])
    np_is_finished = np.frombuffer(VARS[IS_FINISHED])
    np_lengths = np.frombuffer(VARS[LENGTHS])
    index_value = VARS[INDEX_VALUE]

    new_pairs = []
    for idx, u, dst, num_outputs in args:
        if np_predictions[idx] >= len(index_value[u]):
            # If output is larger than the nodes has neighbors stay on the
            # current node.
            next_node = u
        else:
            next_node = index_value[u][int(np_predictions[idx])]
            if np_is_finished[idx] == 0:
                # If this pair is not yet finished, increase the length to one
                # and check if the next node is the destination.
                np_lengths[idx] = np_lengths[idx] + 1
                if next_node == dst:
                    np_is_finished[idx] = 1
        print("{:3d}) Take at {:10s} port {} to node {:10s} to get to {:10s}".format(
            idx, u, int(np_predictions[idx]), next_node, dst
        ))
        new_pairs.append((idx, next_node, dst, num_outputs))
    return new_pairs


def _rec_prod(l: List[float]) -> int:
    if len(l) == 0:
        return int(1)
    else:
        return int(l[0] * _rec_prod(l[1:]))


def make_prediction(model: SpfModel, shared_destinations, shared_masks, shared_neighbors,
                   shape_destinations, shape_masks, shape_neighbors,
                   shared_targets, shape_targets) -> np.array:
    np_destinations = np.frombuffer(shared_destinations).reshape(
        shape_destinations[0], 1, shape_destinations[1]
    ).astype(np.float32)
    np_masks = np.frombuffer(shared_masks).reshape(shape_masks).astype(np.float32)
    np_neighors = np.frombuffer(shared_neighbors).reshape(shape_neighbors).astype(np.float32)

    t_destinations = torch.tensor(np_destinations, device=DEV)
    t_masks = torch.tensor(np_masks, device=DEV)
    t_neighbors = torch.tensor(np_neighors, device=DEV)

    np_targets = np.frombuffer(shared_targets).reshape(shape_targets)

    predictions, scores = model.forward(
        queries=t_destinations,
        mask=t_masks,
        others=t_neighbors
    )
    probs = torch.softmax(predictions, axis=-1)
    for i in range(probs.shape[0]):
        print("{:3d}) ".format(i), end='')
        for j in range(probs.shape[1]):
            print("{:.2f}|{:.2f}  ".format(probs[i, j], np_targets[i, j]), end='')
        print()
    samples = torch.squeeze(torch.multinomial(probs, 1), axis=1).to('cpu')
    return samples.numpy().astype(np.float64)


def driver2(graph: nx.Graph, model: SpfModel, embeddings: np.array,
           hosts=None) -> Tuple[np.array, np.array]:
    if hosts is None:
        hosts = list(graph.nodes())
    pairs = []
    num_outputs = sp_prep._calc_num_outputs(graph)
    dim_embedding = embeddings[0].size
    np_embeddings = embeddings

    value_index = sp_prep._neighbor_to_index(graph)
    index_neighbor = evutils._index_to_neighbor(graph)
    index_value = index_neighbor
    idx_node = {d['idx']: n for n, d in graph.nodes(data=True)}
    adj = sp_prep._make_distance_dict(graph)
    the_pairs = []

    num_pairs = 0
    for u, v in itertools.product(hosts, hosts):
        if u == v:
            continue
        else:
            the_pairs.append([u, v])
            pairs.append((num_pairs, u, v, num_outputs))
            num_pairs += 1
    print(pairs)
    np_is_finished = np.zeros(len(pairs))
    np_lengths = np.zeros(len(pairs))
    shape_neighbors = (num_pairs, num_outputs, dim_embedding)
    shape_masks = (num_pairs, num_outputs, 1)
    shape_destinations = (num_pairs, dim_embedding)
    shape_targets = (num_pairs, num_outputs)
    shape_embeddings = (graph.number_of_nodes(), dim_embedding)

    for i in range(10):
        np_targets = []
        np_neighbors = []
        np_masks = []
        np_destinations = []
        np_cur_locs = []
        cur_locs_idx = []
        print(i)
        for idx, u, dst, num_outputs in pairs:
            sample = sp_prep._get_sample(graph, u, dst, num_outputs, adj, value_index)

            nids = sample[sp_prep.H5_NEIGHBORS]
            tmp = np.zeros(shape_neighbors[1:], dtype=np.float32)
            for i, neighbor in enumerate(nids):
                if neighbor == -1:
                    continue
                else:
                    tmp[i, :] =  np_embeddings[neighbor, :]

            np_targets.append(np.expand_dims(sample[sp_prep.H5_TARGET], 0))
            np_neighbors.append(np.expand_dims(tmp, 0))
            np_masks.append(np.expand_dims(sample[sp_prep.H5_MASK].reshape(-1, 1), 0))
            np_destinations.append(np.expand_dims(
                np_embeddings[sample[sp_prep.H5_DESTINATION]].reshape(1, -1),
                0
            ))
            np_cur_locs.append(np.expand_dims(
                np_embeddings[graph.nodes[u]['idx']].reshape(1, -1),
                axis=1
            ))
            cur_locs_idx.append(graph.nodes[u]['idx'])

        np_cur_locs = np.concatenate(np_cur_locs)
        np_targets = np.concatenate(np_targets).astype(np.float32)
        np_neighors = np.concatenate(np_neighbors).astype(np.float32)
        np_masks = np.concatenate(np_masks).astype(np.float32)
        np_destinations = np.concatenate(np_destinations).astype(np.float32)

        # print('tartest', np_targets.shape)
        # print('neighbors', np_neighors.shape)
        # print('masks', np_masks.shape)
        # print('destinations', np_destinations.shape)
        np_destinations = np.squeeze(np.concatenate([np_destinations, np_cur_locs], axis=-1), axis=1)
        t_destinations = torch.tensor(np_destinations, device=DEV)
        t_masks = torch.tensor(np_masks, device=DEV)
        t_neighbors = torch.tensor(np_neighors, device=DEV)

        predictions, scores = model.forward(
            queries=t_destinations,
            mask=t_masks,
            others=t_neighbors
        )

        probs = torch.sigmoid(predictions).detach().numpy()
        # tmp = np.sum(probs, axis=1).reshape(-1, 1)
        # probs = probs / tmp
        sample = np.argmax(probs, axis=-1)
        # probs = torch.softmax(predictions, axis=-1)
        # samples = torch.squeeze(torch.multinomial(probs, 1), axis=1).to('cpu')
        print(the_pairs, cur_locs_idx)
        for i in range(probs.shape[0]):
            print("{:3d}) {:9s} to {:9s} on {:9s}: ".format(i, the_pairs[i][0], the_pairs[i][1], idx_node[cur_locs_idx[i]]), end='')
            for j in range(probs.shape[1]):
                print("{:.2f}|{:.2f}  ".format(probs[i, j], np_targets[i, j]), end='')
            print()

        np_predictions = sample
        new_pairs = []
        for idx, u, dst, num_outputs in pairs:
            if np_predictions[idx] >= len(index_value[u]):
                # If output is larger than the nodes has neighbors stay on the
                # current node.
                next_node = u
            else:
                next_node = index_value[u][int(np_predictions[idx])]
                if np_is_finished[idx] == 0:
                    # If this pair is not yet finished, increase the length to one
                    # and check if the next node is the destination.
                    np_lengths[idx] = np_lengths[idx] + 1
                    if next_node == dst:
                        np_is_finished[idx] = 1
            # print("{:3d}) Take at {:10s} port {} to node {:10s} to get to {:10s}".format(
            #     idx, u, int(np_predictions[idx]), next_node, dst
            # ))
            new_pairs.append((idx, next_node, dst, num_outputs))
        pairs = new_pairs
    print("Average length", np.mean(np_lengths))
    print("finished ", np.sum(np_is_finished), "of ", np_is_finished.size)
    return np_lengths, np_is_finished


def driver(graph: nx.Graph, model: SpfModel, embeddings: np.array) -> Tuple[np.array, np.array]:
    """
        Drives the overall process of evaluating stuff
    """
    dim_embedding = embeddings.shape[-1]
    num_outputs = sp_prep._calc_num_outputs(graph)
    hosts = list(graph.nodes())
    hosts = ['h-0001', 'h-0012']

    manager = mp.Manager()
    edges = manager.list([(u, v, d) for u, v, d in graph.edges(data=True)])
    nodes = manager.list([(u, d) for u, d in graph.nodes(data=True)])
    value_index = manager.dict(sp_prep._neighbor_to_index(graph))
    index_neighbor = manager.dict(evutils._index_to_neighbor(graph))
    adj = manager.dict(sp_prep._make_distance_dict(graph))

    pairs = [[] for _ in range(16)]
    num_pairs = 0
    for u, v in itertools.product(hosts, hosts):
        if u == v:
            continue
        else:
            pairs[num_pairs % len(pairs)].append((num_pairs, u, v, num_outputs))
            num_pairs += 1
        if num_pairs > 100:
            break

    shared_is_finished = mp.RawArray('d', num_pairs)
    tmp = np.frombuffer(shared_is_finished)
    np.copyto(tmp, np.zeros(num_pairs))
    shared_lengths = mp.RawArray('d', num_pairs)
    shared_predictions = mp.RawArray('d', num_pairs)
    np_predictions = np.frombuffer(shared_predictions)

    shape_neighbors = (num_pairs, num_outputs, dim_embedding)
    shape_masks = (num_pairs, num_outputs, 1)
    shape_destinations = (num_pairs, dim_embedding)
    shape_targets = (num_pairs, num_outputs)
    shape_embeddings = (graph.number_of_nodes(), dim_embedding)

    shared_neighbors = mp.RawArray('d', _rec_prod(shape_neighbors))
    shared_masks = mp.RawArray('d', _rec_prod(shape_masks))
    shared_destinations = mp.RawArray('d', _rec_prod(shape_destinations))
    shared_targets = mp.RawArray('d', _rec_prod(shape_targets))
    shared_embeddings = mp.RawArray('d', _rec_prod(shape_embeddings))

    tmp = np.frombuffer(shared_embeddings).reshape(shape_embeddings)
    np.copyto(tmp, embeddings)

    args = (nodes, edges, adj, shape_targets, shape_neighbors, shape_masks,
            shape_destinations, shape_embeddings, shared_targets, shared_neighbors,
            shared_masks, shared_destinations, value_index, shared_embeddings)
    args2 = (shared_predictions, shared_is_finished, shared_lengths, index_neighbor)

    for i in range(10):
        print(i)
        with mp.Pool(initializer=init_make_input, initargs=args) as pool:
            pool.map(fill, pairs)

        predictions = make_prediction(
            model=model,
            shared_destinations=shared_destinations,
            shared_masks=shared_masks,
            shared_neighbors=shared_neighbors,
            shape_destinations=shape_destinations,
            shape_masks=shape_masks,
            shape_neighbors=shape_neighbors,
            shared_targets=shared_targets,
            shape_targets=shape_targets
        )
        np.copyto(np_predictions, predictions)

        with mp.Pool(initializer=init_process_predictions, initargs=args2) as pool:
            pairs = pool.map(process_predictions, pairs)
    lengths = np.frombuffer(shared_lengths).copy()
    finished = np.frombuffer(shared_is_finished).copy()
    print("Average length", np.mean(lengths))
    print("finished ", np.sum(finished), "of ", finished.size)
    return lengths, finished


def baseline_eval(model: SpfModel, dsetpath: str, filter: str) -> None:
    """
        Make and print predictions from the validation set to get an idea how
        well the model learned.
    """
    ds = DistributionalSpfDataSet.from_hdf5(
        os.path.join(dsetpath, 'k-regular-dist-embedding.h5'),
        os.path.join(dsetpath, 'spf-distributional.h5')
    )
    graph = read_graph('/opt/project/data/regular/k-regular.json')
    loader = DataLoader(ds, batch_size=10, shuffle=True)
    losses = []
    optimal_losses = []
    loss_fct = full_cross_entropy
    with torch.no_grad():
        for batch, sample in enumerate(loader):
            preds, scores = model(
                queries=sample['destination'],
                others=sample['neighbors'],
                mask=sample['attention_mask']
            )
            preds.detach()
            for s in scores:
                s.detach()
                tmpa = s.numpy()
                print("\n".join(
                    ["\t".join(["{:.4f}".format(tmpa[g, 0, f]) for f in range(tmpa.shape[2])]) for g in range(tmpa.shape[0])]))
            losses.append(loss_fct(
                logits=preds,
                target=sample['target'],
                weights=torch.tensor([[1.]])
            ).to('cpu').numpy() + 1e-6)
            optimal_losses.append(loss_fct(
                logits=torch.log(sample['target'] + 1e-6),
                target=sample['target'],
                weights=torch.tensor([[1.]])
            ).numpy())
            # if np.sum(sample['target'].numpy() == 0) in [6]:
            # if sample['target'].numpy()[0, 0] > 0:
            logger.debug("Has a loss of {}, optimal: {}".format(losses[-1], optimal_losses[-1]))
            probs = torch.softmax(preds, axis=-1).to('cpu').numpy()
            logger.debug("{:s}".format(_matrix_to_str(
                probs,
                sample['target']
            )))
            if batch > 250:
                break
        logger.debug('Average loss of model over full file is  : {}'.format(np.mean(losses)))
        logger.debug('Average loss of optimal over full file is: {}'.format(np.mean(optimal_losses)))


def run_baseline(logdir, dsetpath: str, filter: str):
    # logdir = '/opt/project/trained_models/StateTrainable_97248_00018'
    model = evutils.load_model(SpfConfig, SpfModel, logdir)#, checkpoint_dir='checkpoint_1000')
    model.to(DEV)
    baseline_eval(model, dsetpath, filter)


if __name__ == '__main__':
    ldir = evutils.get_logdir_best_model(
        './data/training-results/regular-sfp-distr-dist-embedding',
        'RegularSpfDistributionalDist',
        'cross_entropy'
    )
    run_baseline(ldir, './data/regular', '')
    # logdir = './data/training-results/fat-tree-k8-sfp-dist-ip-embedding/' +\
    #         'FatTreeSpfDistributionalIp002/SpTrainable_42_batch_size=256,' +\
    #         'dim_attn_hidden=29,dim_attn_out=30,dim_out_fcn=[26, 22],' +\
    #         'num_heads=5,lr=0.001_2020-08-19_19-00-13epua8k1q'
    # model = evutils.load_model(SpfConfig, SpfModel, logdir, checkpoint_dir='checkpoint_1000')
    # model.to(DEV)
    # graph = read_graph('./data/fat-tree-k8/fat-tree-k8.json')
    # # embeddings = read_embeddings('./data/fat-tree-k8/fat-tree-k8-ip-embedding.h5').astype(np.float64)
    # embeddings = {}
    # f = h5py.File('./data/fat-tree-k8/fat-tree-k8-ip-embedding.h5', "r")
    # for k in f.keys():
    #     embeddings[k] = f[k]['embedding'][()]
    # f.close()

    # hosts = ['h-{:04d}'.format(x) for x in np.arange(128)]
    # hosts = None
    # fig, ax = present.get_fig(1)
    # for i in range(10):
    #     lengths, is_finished = driver2(graph, model, embeddings, hosts=hosts)
    #     present.plot_cdf(lengths, ax=ax, alpha=0.7)
    # ax.set_xlabel("Path length [hops]")
    # ax.set_ylabel("P(X) > x")
    # present.save_fig('./img/cdf-path-lengths-all-nodes.pdf', format='pdf', fig=fig)

    # ds = DistributionalSpfDataSet.from_hdf5('./data/fat-tree-k8/fat-tree-k8-ip-embedding.h5',
    #                                    './data/fat-tree-k8/spf-distributional.h5', 10)
    # loader_train = DataLoader(ds, batch_size=10, shuffle=True)
    # for batch, sample in enumerate(loader_train):
    #     print("dst", sample['destination'].shape)
    #     print("neighbors", sample['neighbors'].shape)
    #     print('attention_mask', sample['attention_mask'].shape)
    #     pred, scores = model(
    #         queries=sample['destination'].to(DEV),
    #         others=sample['neighbors'].to(DEV),
    #         mask=sample['attention_mask'].to(DEV)
    #     )
    #     loss = full_cross_entropy(pred, sample['target'].to(DEV))
    #     # print(loss, end=' ')

    #     probs = torch.softmax(pred, axis=-1)
    #     for i in range(probs.shape[0]):
    #         print("{:3d}) ".format(i), end='')
    #         for j in range(probs.shape[1]):
    #             print("{:.2f}|{:.2f}  ".format(probs[i, j], sample['target'][i, j]), end='')
    #         print()
    #     break



