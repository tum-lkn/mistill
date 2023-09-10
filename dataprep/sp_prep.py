"""
Implements functionality to prepare or otherwise mutate data.
"""
import networkx as nx
import numpy as np
import itertools
import h5py
import multiprocessing as mp
from typing import Any, Tuple, Dict, List
import dataprep.utils as dutils


H5_MASK = 'mask'
H5_TARGET = 'target'
H5_NEIGHBORS = 'neighbors'
H5_DESTINATION = 'destination'
H5_CUR_LOC = 'H5_CUR_LOC'

# Numerical index from 0 to num_nodes - 1. Each node has this index.
IDX = 'idx'
# Numerical index from 0 to num_nodes_with_degree_larger_one - 1. Only nodes
# that have a degree larger one have this index.
NS_IDX = 'network_state_index'
H_IDX = 'host_index'
EMBD_KEY = 'embedding'


def _calc_lon_lat(graph: nx.Graph, node: Any) -> Tuple[float, float]:
    """
        --> Data prep
        Iterate over the neighbors and calculate the average latitude and longitude
        over them, so they have this attribute.

        Args:
            graph: Graphe.
            node: Node for which coordinates should be created based on neighbors.

        Returns:

    """
    lat = 0
    lon = 0
    count = 0.
    for u in graph.neighbors(node):
        if "Latitude" in graph.nodes[u] and "Longitude" in graph.nodes[u]:
            lat = graph.nodes[u]["Latitude"]
            lon = graph.nodes[u]["Longitude"]
            count += 1
    return lon / count, lat / count


def _fill_missing_coordinates(graph):
    """
    -> Data prep
    Try to fill missing coordinates in a graph by calculating the average
    over the coordinates of all neighbors.

    Args:
        graph: Networkx Graph.

    Returns:
        None.
    """
    for n, d in graph.nodes(data=True):
        if "Latitude" not in d or "Longitude" not in d:
            lon, lat = _calc_lon_lat(graph, n)
            d["Latitude"] = lat
            d["Longitude"] = lon
    return graph


def add_embedding_to_graph(embeddings: np.array, mapping: Dict[Any, int], graph: nx.Graph):
    """
    Add the embeddings to the graph in the node attribute `ebemdding`. The
    embeddings are added as python list to make the graph serializable.

    Args:
        embeddings: Array of shape (|V|, d).
        mapping: Maps node identifier in graph to index in embeddings.
        graph: Graph that should be augmented.

    Returns:
        None
    """
    for u in graph.nodes():
        idx = mapping[u]
        graph.nodes[u]['embedding'] = embeddings[idx].tolist()


def _neighbor_to_index(graph: nx.Graph) -> Dict[Any, Dict[Any, int]]:
    """
    Create mapping of neighbors of nodes to ints.

    Args:
        graph: A graph.

    Returns:
        Dict that maps node identifiers to a dict that maps the node identifiers
            of the neighbors to indices.
    """
    maps = {}
    for u in graph.nodes():
        maps[u] = {}
        # maps[u][u] = 0
        for i, v in enumerate(graph.neighbors(u)):
            maps[u][v] = i # + 1
    return maps


def _make_flat_index(graph: nx.Graph) -> Tuple[Dict[Any, Dict[Any, int]], Dict[Any, Tuple[int, int]]]:
    """
    Creates a mapping that maps a node and its neighbor to an index in a flat
    array.

    Args:
        graph: Some graph.

    Returns:
        Dict that performs the mapping.
    """
    idx = 0
    map = {}
    start = {}
    for u in graph.nodes():
        map[u] = {}
        s = idx
        for v in graph.neighbors(u):
            map[u][v] = idx
            idx += 1
        start[u] = (s, idx)
    return map, start


def _extract_embedding(graph: nx.Graph, node: Any) -> np.array:
    """
    Extracts the embedding from the graph and converts it to an array.

    Args:
        graph: Graph that we operate on.
        node: Some node.

    Returns:
        embedding.
    """
    return np.array(graph.nodes[node]['embedding'], dtype=np.float32)


def _make_values(value_index: Dict[Any, Dict[Any, int]], graph: nx.Graph,
                 tail: Any, max_degree: int, ndims: int) -> np.array:
    """
    Create one value entry for the given graph and current location.
    Args:
        value_index: Mapping to index.
        graph: Graph being traversed.
        tail: Current node on the path.
        max_degree: Maximum degree of graph.
        ndims: Number of dimensions of embedding.

    Returns:
        values: np.array
    """
    # values = np.zeros([1, max_degree + 1, ndims], dtype=np.float32)
    # values[0, 0, :] = _extract_embedding(graph, tail)
    values = np.zeros([1, max_degree, ndims], dtype=np.float32)
    for v in graph.neighbors(tail):
        idx = value_index[tail][v]
        values[0, idx, :] = _extract_embedding(graph, v)
    return values


def _make_value_mask(graph: nx.Graph, tail: Any, max_degree: int) -> np.array:
    """
    Create a binary mask that indicates non-zero values.

    Args:
        graph: Graph.
        tail: current node.
        max_degree: Maximum degree of graph.

    Returns:
        mask: numpy array indicating non-zero values.
    """
    mask = np.zeros([1, max_degree, 1], dtype=np.float32)
    non_zero = nx.degree(graph)[tail] + 1
    mask[0, :non_zero, 0] = 1
    return mask


def _make_target(v_index: Dict[Any, Dict[Any, int]], head: Any, tail: Any,
                 num_outputs: int) -> np.array:
    """
    Create the target vector. Set the output corresponding to the edge to the
    next hop to one.

    Args:
        v_index: Mapping from edge to output.
        head: Next hop in path.
        tail: Current location.
        num_outputs: Total number of outputs.

    Returns:
        z: target vector.
    """
    z = np.zeros((1, num_outputs), np.float32)
    # Subtract one since this index contains a self-loop at index 0. See
    # function `_neighbor_to_index`.
    z[0, v_index[tail][head] - 1] = 1
    return z


def _make_target_mask(start_index: Dict[Any, Tuple[int, int]], tail: Any, num_outputs: int) -> np.array:
    """
    Create mask for the target vector.

    Args:
        start_index: Mapping that tells me the start and end for each node in
            the output vector.
        tail: The current location.
        num_outputs: The total number of outputs.

    Returns:
        mask: np.array
    """
    mask = np.zeros((1, num_outputs), dtype=np.float32)
    beginning, end = start_index[tail]
    mask[0, beginning:end] = 1
    return mask


def _make_query(graph: nx.Graph, dst: Any) -> np.array:
    """
    Create one query vector.

    Args:
        graph:  Graph that we are traversing.
        dst: Destination node of path.

    Returns:
        embd: Query, i.e., embedding of destination node.
    """
    embd = np.expand_dims(np.expand_dims(_extract_embedding(graph, dst), 0), 0)
    return embd


def _make_weight(cur_loc: str, dst: str, k: int) -> float:
    """
    Calculate sample weights based on the frequency of corresponding forwarding
    decisions. Assumes a fat-tree topology.

    Args:
        cur_loc:
        dst:

    Returns:

    """
    pod_cur_loc = dutils.get_pod(cur_loc, k)
    pod_dst = dutils.get_pod(dst, k)


def _calc_num_outputs(graph: nx.Graph) -> int:
    degrees = list(dict(nx.degree(graph)).values())
    max_degree = np.max(degrees)
    if graph.is_directed():
        max_degree = max_degree / 2
    return int(max_degree)


def spf_dataset(graph: nx.Graph, nodes=None) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
    """
    Create a dataset for the shortest paths.

    Note: max_degree is the maximum degree over all nodes.

    Args:
        graph: Graph for which dataset should be created. Must contain the
            `embedding` key as node attribute.

    Returns:
        queries: Embedding of destination node (N, 1, d).
        values: Embeddings of adjacent nodes, shape (N, max_deg, d)
        masks_v: Array of zeros and ones indicating which of the rows in values
            are set (1, max_deg, 1).
        zs: one-hot encoded vector (N, max_degree).
        cur_locs: Embedding of current location of shape (N, d).
    """
    ndims = _extract_embedding(graph, list(graph.nodes())[0]).size
    if nodes is None:
        nodes = list(graph.nodes())
    num_outputs = _calc_num_outputs(graph)
    value_index = _neighbor_to_index(graph)

    queries = []
    values = []
    masks_v = []
    zs = []
    cur_locs = []
    weights = []

    for src, dst in itertools.product(nodes, nodes):
        if src == dst:
            continue
        for path in nx.all_shortest_paths(graph, src, dst):
            for tail, head in zip(path[1:-1], path[2:]):
                values.append(_make_values(value_index, graph, tail, num_outputs, ndims))
                queries.append(_make_query(graph, dst))
                zs.append(_make_target(value_index, head, tail, num_outputs))
                cur_locs.append(np.expand_dims(_extract_embedding(graph, tail), axis=0))
                masks_v.append(_make_value_mask(graph, tail, num_outputs))
                weights.append(_make_weight(tail, dst))
    queries = np.concatenate(queries)
    values = np.concatenate(values)
    masks_v = np.concatenate(masks_v)
    zs = np.concatenate(zs)
    cur_locs = np.concatenate(cur_locs)
    return queries, values, masks_v, zs, cur_locs


def _calc_lengths(graph, source, weight) -> Dict[str, Dict[str, float]]:
    lengths = {}
    for target in graph.nodes():
        if target == source:
            lengths[target] = 0
        else:
            lengths[target] = nx.shortest_path_length(graph, source=source, target=target, weight=weight)
    return {source: lengths}


def _calc_lengths_mp(args) -> Dict[str, Dict[str, float]]:
    graph = args['graph']
    sources = args['sources']
    weight = args['weight']
    ret = {}
    for source in sources:
        lengths = {}
        for target in graph.nodes():
            if target == source:
                lengths[target] = 0
            else:
                lengths[target] = nx.shortest_path_length(graph, source=source, target=target, weight=weight)
        ret[source] = lengths
    return ret


def _make_distance_dict(graph: nx.Graph, weight=None, use_mp=False) -> Dict[Any, Dict[Any, int]]:
    """
    Create the distance matrix as nested dict of path lengths.

    Args:
        graph:
        sources:

    Returns:

    """
    ret = {}
    if use_mp:
        srcs = [[] for _ in range(mp.cpu_count())]
        for i, src in enumerate(graph.nodes):
            srcs[i % len(srcs)].append(src)
        pool = mp.Pool()
        tmp = pool.map(
            _calc_lengths_mp,
            [{
                'sources': src,
                'graph': graph,
                'weight': weight
            } for src in srcs]
        )
        pool.close()
        for t in tmp:
            ret.update(t)
    else:
        for source in graph.nodes():
            ret.update(_calc_lengths(graph, source, weight))
    return ret


def _get_distributional_output(graph: nx.Graph, u: Any, dst: Any, num_outputs: int,
                               adj: Dict[Any, Dict[Any, int]],
                               neighbor_to_idx: Dict[Any, Dict[Any, int]],
                               num_paths: int) -> np.array:
    """
    Get the distribution over output ports.

    Args:
        graph: Graph outputs should be computed for.
        u: Node for which the distribution over neighbors should be computed.
        dst: Destination node based on which the output distribution is computed.
        num_outputs: The size of the number of outputs.
        adj: Maps pairs of nodes to the distance between them.
        neighbor_to_idx: Maps the neighbors of every node to a number.
        num_paths: If set to a number larger 0 returns the k-shortest paths,
            where num_paths is k. For example, for num_paths=4 function returns
            the output distribution for the four shortest paths.

    Returns:
        output: numpy array that defines a categorical distribution over
            neighbors. An entry in `output` is non-zero, if the corresponding
            neighbor of `u` lies on a shortest path to `dst`. That is, each
            neighbor `v` of `u` that lies on a path from `u` to `dst` with
            minimal length has an equal probability of being chosen.
    """
    output = np.zeros((1, num_outputs), dtype=np.float32)

    # For each neighbor `v` of `u` get the distance to `dst` and sort them
    # ascending.
    distances = []
    for v in nx.neighbors(graph, u):
        distances.append((v, adj[v][dst]))
    distances.sort(key=lambda x: x[1])

    # Iterate over neighbors until the distance is larger than the minimum
    # distance, i.e., the distance of the first node in the list.
    path_count = 0
    for v, d in distances:
        if d > distances[0][1] and path_count >= num_paths:
            break
        else:
            output[0, neighbor_to_idx[u][v]] = 1
            path_count += 1
    # normalize the distribution.
    # output /= output.sum()
    return output


def _get_neighbors(graph: nx.Graph, u: Any, neighbor_to_idx: Dict[Any, Dict[Any, int]]) -> np.array:
    """
    Get a list of neighbors sorted according to the previously defined range.

    Args:
        graph:
        u:
        neighbor_to_idx:

    Returns:

    """
    max_deg = int(np.max(list(dict(graph.degree()).values())) / 2)
    # neighbors = np.repeat(-1, repeats=len(neighbor_to_idx[u])).astype(np.int32)
    neighbors = (-1 * np.ones((1, max_deg))).astype(np.int32)
    for v, idx in neighbor_to_idx[u].items():
        neighbors[0, idx] = graph.nodes[v]['idx']
    return neighbors


def _get_mask(u: Any, neighbor_to_idx: Dict[Any, Dict[Any, int]], num_outputs: int) -> np.array:
    """
    Create a maks indicating which outputs correspond to existing neighbors. Will
    be used as attention mask.

    Args:
        u:
        neighbor_to_idx:
        num_outputs:

    Returns:

    """
    mask = np.zeros((1, num_outputs), dtype=np.float32)
    for idx in neighbor_to_idx[u].values():
        mask[0, idx] = 1
    return mask


def _get_sample(graph, u, dst, num_outputs, adj, value_index, num_paths):
    sample = {
        H5_TARGET: _get_distributional_output(graph, u, dst, num_outputs, adj, value_index, num_paths),
        H5_NEIGHBORS: _get_neighbors(graph, u, value_index),
        H5_MASK: _get_mask(u, value_index, num_outputs),
        H5_DESTINATION: np.array([graph.nodes[dst]['idx']]),
        H5_CUR_LOC: np.array([graph.nodes[u]['idx']])
    }
    return sample


def _get_sample_mp(args: Dict[str, Any]) -> Dict[str, np.array]:
    graph = args['graph']
    pairs = args['pairs']
    num_outputs = args['num_outputs']
    adj = args['adj']
    value_index = args['value_index']
    num_paths = args['num_paths']
    # print(type(pairs), type(pairs[0]))
    samples = {
        H5_TARGET: [],
        H5_NEIGHBORS: [],
        H5_MASK: [],
        H5_DESTINATION: [],
        H5_CUR_LOC: []
    }
    for u, dst in pairs:
        sample = _get_sample(graph, u, dst, num_outputs, adj, value_index, num_paths)
        for k, v in sample.items():
            samples[k].append(v)
    # samples = {k: np.concatenate(v) for k, v in samples.items()}
    ret = {}
    for k, v in samples.items():
        try:
            ret[k] = np.concatenate(v)
        except Exception as e:
            print(e)
            print(v)
            print(k)
            raise e
    return samples


def _make_pairs_mp(args):
    nodes = args['nodes']
    destinations = args['destinations']
    # print("make_pairs", len(nodes), len(destinations), type(nodes[0]), type(destinations[0]))

    pairs = []
    for dst in destinations:
        for u in nodes:
            if u == dst:
                continue
            else:
                pairs.append((u, dst))
    return pairs


def distributional_spf_dataset(graph: nx.Graph, num_paths: int, nodes=None, use_mp=False) -> Dict[str, np.array]:
    """
        Compute a distribution over output ports. If two of the neighbors of a specific
        node lie on the shortest paths to a destination, then the corresponding
        ports have a probability of 0.5 and all other ports a probability of 0.

        Args:
            graph: The graph for which dataset should be calculated.
            num_paths: The number of k-shortest paths that should be returned.
                For a value of 0 all shortest paths are returned. A value of 4
                returns four paths from each node to every other node no matter
                how long the paths are. However, if there are more than
                num_paths shortest paths, then those are used in the output
                calculation. For example, a k=8 fat-tree has 16 shortest path
                between any pair of nodes in different pods. The output is then
                based on those 16 paths. In contrast, in case of a k-regular graph,
                there might be only one shortest path, in this case, up to three
                non-shortest paths are considered for the calculation as well.
            nodes: The nodes that should be considered. If None calculated between
                all pairs of nodes.
            use_mp: Whether to use multi-processing.

        Returns:
    """
    if nodes is None:
        nodes = list(graph.nodes())
    num_outputs = _calc_num_outputs(graph)
    value_index = _neighbor_to_index(graph)
    print("Calculate adjacency...")
    adj = _make_distance_dict(graph, use_mp=use_mp)

    samples = []
    if use_mp:
        destinations = [[] for _ in range(mp.cpu_count())]
        for i, dst in enumerate(nodes):
            destinations[i % len(destinations)].append(dst)
        pool = mp.Pool()
        print("Create Pairs...")
        ps = pool.map(
            _make_pairs_mp,
            [{'destinations': x, 'nodes': [n for n in graph.nodes()]} for x in destinations]
        )
        pool.close()
        pool = mp.Pool()
        print("Create Dataset...", len(ps), len(ps[0]), type(ps), type(ps[0]))
        ret = pool.map(
            _get_sample_mp,
            [{
                'graph': graph,
                'pairs': p,
                'num_outputs': num_outputs,
                'adj': adj,
                'value_index': value_index,
                'num_paths': num_paths
            } for p in ps]
        )
        pool.close()
        print("Reduce Dataset...")
        for r in ret:
            samples.append(r)
    else:
        for dst in nodes:
            for u in graph.nodes():
                if u == dst:
                    continue
                samples.append(_get_sample(graph, u, dst, num_outputs, adj, value_index, num_paths))
    ret = {k: np.concatenate([sample[k] for sample in samples]) for k in samples[0].keys()}
    return ret


def add_node_embedding(graph: nx.Graph, embeddings: Dict[Any, np.array]) -> nx.Graph:
    """
    Add the node embedding to the nodes in the graph.
    Convert embeddings to a python list in order to store graph in json file.

    Note:
        Mutates the `graph` argument.

    Args:
        graph:
        embeddings:

    Returns:

    """
    for n in graph.nodes():
        embd = embeddings[graph.nodes[n]['idx']]
        assert embd.ndim == 1, "Embeddings are expected to be one-dimensional"
        graph.nodes[n]['embedding'] = embd.tolist()
    return graph


def hamming_distance(graph: nx.DiGraph) -> Tuple[Dict[Any, int], np.array]:
    """
    Calculate the hamming distance between the binary embeddings of the nodes
    in the passed graph.

    Args:
        graph:

    Returns:
        map: Dinctionary that maps node identifiers to indices to the distance
            matrix.
        dist: Distance matrix.
    """
    x = []
    map = {}
    for i, u in enumerate(graph.nodes()):
        map[u] = i
        x.append(graph.nodes[u]['embedding'])
    x = np.row_stack(x)
    dist_mat = np.dot(x, (1 - x).T) + np.dot(1-x, x.T)
    return map, dist_mat


def add_index_to_nodes(graph: nx.DiGraph, undirected_edge_index=False) -> nx.DiGraph:
    """
    Functions require nodes in the graph to have an `idx` attribute. This
    attribute contains a postive, increasing number ranging from {0, ..., num_nodes - 1}.
    This index is used for array and dataset stuff.

    Args:
        graph:
        undirected_edge_index:

    Returns:

    """
    degree_thresh = 2 if graph.is_directed() else 1
    state_idx = 0
    host_idx = 0
    degrees = nx.degree(graph)

    for idx, node in enumerate(graph.nodes()):
        graph.nodes[node][IDX] = idx
        if degrees[node] > degree_thresh:
            graph.nodes[node][NS_IDX] = state_idx
            state_idx += 1
        else:
            graph.nodes[node][H_IDX] = host_idx
            host_idx += 1
    graph.graph['edge_indices'] = {}
    if undirected_edge_index:
        counter: Dict[Tuple[Any, Any], int] = {}
        for u, v in graph.edges():
            if (v, u) in counter:
                idx = counter[(v, u)]
                offset = 1
            else:
                idx = len(counter)
                counter[(u, v)] = idx
                offset = 0
            graph.edges[u, v][IDX] = (idx, offset)
            graph.graph['edge_indices'][u, v] = (idx, offset)
    else:
        for i, (u, v) in enumerate(graph.edges()):
            graph.edges[u, v][IDX] = (i, 0)
            graph.graph['edge_indices'][u, v] = (i, 0)
    return graph


def expand_distributional_spf_dataset(full_path_to_template: str,
                                      full_path_to_embeddings: str,
                                      max_num=-1) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    Reads the template for the dataset out of the corresponding hdf5 file and
    expands the node place holder with the corresponding embeeddings in the
    embeddings dataset.

    Args:
        full_path_to_template:
        full_path_to_embeddings:

    Returns:
        destinations: Array of shape (N, D).
        neighbors: Array of shape (N, num_outputs, D).
        masks: Array of shape (N, num_outputs, 1).
        targets: Array of shape (N, num_outputs).

    """
    templates = h5py.File(full_path_to_template, 'r')
    embeddings = h5py.File(full_path_to_embeddings, 'r')

    embd_dim = embeddings['0']['embedding'][()].size

    masks = []
    targets = []
    neighbors = []
    destinations = []
    for i, sample in enumerate(templates.keys()):
        if i >= max_num and max_num > 0:
            break
        masks.append(templates[sample][H5_MASK][()].reshape(1, -1, 1).astype(np.float32))
        targets.append(templates[sample][H5_TARGET][()].reshape(1, -1).astype(np.float32))

        dst = templates[sample][H5_DESTINATION][()]
        destinations.append(embeddings['{:d}'.format(dst)]['embedding'][()].reshape(1, 1, -1).astype(np.float32))

        nbs = np.zeros((1, targets[0].shape[-1], embd_dim), dtype=np.float32)
        for i, neighbor in enumerate(templates[sample][H5_NEIGHBORS][()]):
            if neighbor == -1:
                continue
            else:
                embd = embeddings['{:d}'.format(neighbor)]['embedding'][()].astype(np.float32)
                nbs[0, i, :] = embd
        neighbors.append(nbs)
    return np.concatenate(destinations), np.concatenate(neighbors), np.concatenate(masks), np.concatenate(targets)


def expand_distributional_spf_dataset2(full_path_to_template: str,
                                      full_path_to_embeddings: str) -> Dict[str, np.array]:
    """
    Reads the template for the dataset out of the corresponding hdf5 file and
    expands the node place holder with the corresponding embeeddings in the
    embeddings dataset.

    Args:
        full_path_to_template:
        full_path_to_embeddings:

    Returns:
        destinations: Array of shape (N, D).
        neighbors: Array of shape (N, num_outputs, D).
        masks: Array of shape (N, num_outputs, 1).
        targets: Array of shape (N, num_outputs).

    """
    from dataprep.input_output import read_embeddings,  read_link_failure_data
    embeddings = read_embeddings(full_path_to_embeddings)
    templates = read_link_failure_data(full_path_to_template)

    templates[H5_DESTINATION] = embeddings[templates[H5_DESTINATION], :]
    templates[H5_CUR_LOC] = embeddings[templates[H5_CUR_LOC], :]
    shape = templates[H5_NEIGHBORS].shape
    tmp = templates[H5_NEIGHBORS].flatten()
    tmp = embeddings[tmp, :]
    templates[H5_NEIGHBORS] = tmp.reshape(shape[0], shape[1], embeddings.shape[-1])

    return templates
