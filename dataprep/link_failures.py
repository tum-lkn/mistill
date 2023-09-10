"""
Create link failure datasets.
"""
import multiprocessing as mp
import numpy as np
import time
import itertools
import networkx as nx
import json
import logging
from typing import Any, Dict, List, Tuple
import dataprep.sp_prep as sprep
from dataprep.input_output import read_edges_to_pairs
import dataprep.utils as dutils
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('link-failures.py')


# MANAGER = mp.Manager()
# EDGE_TO_PAIRS = mp.Manager().dict()
EDGE_WEIGHT = "gaussian"
OUTPUT_HULA = 'hula'
OUTPUT_WCMP = 'wcmp'
OUTPUT_LCP = 'lcp'
OUTPUT_ECMP = 'ecmp'


def _get_num(name: str) -> int:
    return dutils.get_num(name)


def _get_pod(name: str, k: int) -> int:
    return dutils.get_pod(name, k)


def _get_incident(core: str, agg: str, k: int) -> bool:
    num_core = core.split('-')[1].lstrip('0')
    num_agg = agg.split('-')[1].lstrip('0')
    num_core = 0 if len(num_core) == 0 else int(num_core)
    num_agg = 0 if len(num_agg) == 0 else int(num_agg)

    pod_agg = _get_pod(agg, k)
    agg_pos = int(num_agg - pod_agg * k / 2)
    return agg_pos * k / 2 <= num_core < (agg_pos + 1) * k / 2


class EdgeToPair(object):
    """
    Wrapper class that stores the communication pairs for a specific edge.

    For each type, stores the current location and the destination pairs. Current
    locations can be ToRs, Aggs and Core swiches. Destinations include also hosts.

    Hosts are excluded from current locations, since they are not part of the
    forwarding process, that is, the neural network will not make predictions
    on the servers but only on the switches in the network. Thus, hosts cannot
    be current locations.

    Important: Even if the graph itself is undirected, edge to pairs will have
        directed edegs! That is, if u - v, then pairs will bie u -> v and v <- u.
    """
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'EdgeToPair':
        x = EdgeToPair(d['tail'], d['head'], d['k'])
        x.pairs = d['pairs']
        return x

    def __init__(self, tail: Any, head: Any, k: int):
        """
        Initializes object.
        Args:
            tail: The tail of the edge.
            head: The head of the edge.
        """
        self.tail = tail
        self.head = head
        # self.node_types = ['h', 'agg', 'tor', 'core']
        self.k = k

        # The first one is the current location, the second one the destination.
        self.pairs = {
            'tor': {
                'agg-same-pod': [],  # Send to an aggregation switch in the same pod, dirac.
                'agg-other-pod': [],
                'h-same-pod': [],  # Send to an host under a different ToR in the same pod, broad.
                'h-other-pod': [],
                'tor-other-pod': [],
                'tor-same-pod': [],
                'core': [],  # Send to a specific core node, dirac.
            },
            # broad distribution for agg-same-pod and sending-out differs. In the
            # first case, braod distribution over all ToR switches, in the second
            # broad distribution over all adjacent core switches.
            'agg': {
                'adjacent-core': [],  # Send to one specific core node directly attached to agg, dirac.
                'distant-core': [],  # Send to one specific core node not directly attached, broad.
                'tor-same-pod': [],
                'tor-other-pod': [],
                'agg-same-pod': [],  # Send to a specific aggregation node in the same pod, braod.
                'agg-other-pod': [],
                'h-same-pod': [],
                'h-other-pod': []
            },
            'core': {
                'tor': [],
                'agg': [],
                'h': [],
                'core': []  # Send to any other core node, broad.
            }
        }

    def get_cur_loc_types(self) -> List[str]:
        return list(self.pairs.keys())

    def get_destination_types(self, cur_loc_type: str) -> List[str]:
        return list(self.pairs[cur_loc_type].keys())

    def _second_type_for_tor(self, cur_loc: str, destination: str) -> str:
        if destination.startswith('core-'):
            return 'core'
        else:
            pod_cur_loc = _get_pod(cur_loc, self.k)
            pod_dest = _get_pod(destination, self.k)
            if destination.startswith('agg-'):
                if pod_cur_loc == pod_dest:
                    return 'agg-same-pod'
                else:
                    return 'agg-other-pod'
            elif destination.startswith('h-'):
                if pod_cur_loc == pod_dest:
                    return 'h-same-pod'
                else:
                    return 'h-other-pod'
            elif destination.startswith('tor-'):
                if pod_cur_loc == pod_dest:
                    return 'tor-same-pod'
                else:
                    return 'tor-other-pod'
            else:
                raise KeyError('Unknown destination name ', destination)

    def _second_type_for_agg(self, cur_loc: str, destination: str) -> str:
        if destination.startswith('core-'):
            if _get_incident(destination, cur_loc, self.k):
                return 'adjacent-core'
            else:
                return 'distant-core'
        else:
            pod_cur_loc = _get_pod(cur_loc, self.k)
            pod_dst = _get_pod(destination, self.k)
            if destination.startswith('tor') and pod_cur_loc == pod_dst:
                return 'tor-same-pod'
            elif destination.startswith('tor') and pod_cur_loc != pod_dst:
                return 'tor-other-pod'
            elif destination.startswith('h') and pod_cur_loc == pod_dst:
                return 'h-same-pod'
            elif destination.startswith('h') and pod_cur_loc != pod_dst:
                return 'h-other-pod'
            elif destination.startswith('agg') and pod_cur_loc == pod_dst:
                return 'agg-same-pod'
            elif destination.startswith('agg') and pod_cur_loc != pod_dst:
                return 'agg-other-pod'
            else:
                raise KeyError("Unkown destination key ", destination)

    def _second_type_for_core(self, cur_loc: str, destination: str) -> str:
        return destination.split('-')[0]

    def add_pair(self, cur_loc: str, destination: str) -> None:
        """
        Add a communication pair to the pairs of this edge. Insert them based
        on the node types.

        Args:
            cur_loc: Identifier of the current location.
            destination: Destination node identifier.

        Returns:

        """
        t1 = cur_loc.split('-')[0]
        t2 = {
            'tor': self._second_type_for_tor,
            'agg': self._second_type_for_agg,
            'core': self._second_type_for_core
        }[t1](cur_loc, destination)
        container = self.pairs[t1][t2]
        if (cur_loc, destination) not in container:
            container.append((cur_loc, destination))

    def sample_pair(self, random: np.random.RandomState) -> Tuple[str, str]:
        t1 = random.choice(list(self.pairs.keys()))
        t2 = random.choice(list(self.pairs[t1].keys()))
        pairs = self.pairs[t1][t2]
        return pairs[random.randint(0, len(pairs))]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'head': self.head,
            'tail': self.tail,
            'pairs': self.pairs,
            'k': self.k
        }

    def get_num_pairs(self) -> int:
        num_pairs = 0
        for t1, second_lv in self.pairs.items():
            for t2, third_lv in second_lv.items():
                num_pairs += len(third_lv)
        return num_pairs

    def finish(self):
        topop = []
        for t1, values in self.pairs.items():
            for t2, lists in values.items():
                if len(lists) == 0:
                    topop.append((t1, t2))
        for k1, k2 in topop:
            self.pairs[k1].pop(k2)
        topop = []
        for k, v in self.pairs.items():
            if len(v) == 0:
                topop.append(k)
        for k in topop:
            self.pairs.pop(k)


def _calc_spfs(args: Dict[str, Any]) -> List[List[Any]]:
    """
    Calculates the shortest paths between src and dst on graph and updates the
    mappings accordingly.
    As a result, the same source and destination pair can occur in multiple paths
    should multiple shortest paths exist.

    Args:
        args:

    Returns:
        p: Path from source to destination, includes src and dst
    """
    srcs = args['srcs']
    dsts = args['dsts']
    graph = args['graph']
    paths = []
    for src, dst in zip(srcs, dsts):
        try:
            for p in nx.all_shortest_paths(graph, src, dst):
                paths.append(p)
        except Exception as e:
            logger.error("Error during spf calculation of {} to {}.".format(src, dst))
            logger.exception(e)
    return paths


def _split_path(paths: List[List[Any]]) -> List[Tuple[Tuple[Any, Any], Tuple[Any, Any]]]:
    """
    Splits path into edges.

    Args:
        paths: All shortest paths between a pair of nodes.

    Returns:
        List of tuples of edge, (source, dst) pairs, i.e., (tail, head), (cur_loc, dst).
    """
    l = []
    if paths is None:
        return l
    else:
        for path in paths:
            for u, v in zip(path[:-1], path[1:]):
                l.append(((u, v), (path[0], path[-1])))
        return l


def _reduce_pairs_flat(pairs: List[List[Tuple[Tuple[Any, Any], Tuple[Any, Any]]]]) -> Dict[Tuple[Any, Any], List[Tuple[Any, Any]]]:
    """
    @Note Deprecated by _reduce_pairs_by_type. Might be useful for non-fat-tree
    topologies, though. THats why I keep it.

    Reduce the pairs obtained from the split_path function to a dictionary that
    maps edges to the source and destination pairs whose shortest paths run
    over that edge.

    Args:
        pairs:

    Returns:

    """
    ret = {}
    for part in pairs:
        for e, pair in part:
            if e not in ret:
                ret[e] = []
            if pair in ret[e]:
                continue
            else:
                ret[e].append(pair)
    return ret


def _reduce_pairs_by_type(pairs: List[List[Tuple[Tuple[Any, Any], Tuple[Any, Any]]]],
                          k: int) -> Dict[Tuple[Any, Any], EdgeToPair]:
    """
    Reduce the pairs obtained from the split_path function to a dictionary that
    maps edges to the source and destination pairs whose shortest paths run
    over that edge.

    Args:
        pairs:

    Returns:

    """
    ret = {}
    for part in pairs:
        for e, pair in part:
            if e not in ret:
                ret[e] = EdgeToPair(tail=e[0], head=e[1], k=k)
            ret[e].add_pair(*pair)
    for v in ret.values():
        v.finish()
    return ret


def _make_edge_to_pairs_from_pairs(graph: nx.DiGraph, k: int, pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], EdgeToPair]:
    slots = [{'srcs': [], 'dsts': [], 'graph': graph} for _ in range(mp.cpu_count())]
    i = 0
    for source, sink in pairs:
        for path in nx.all_shortest_paths(graph, source, sink):
            for cur_loc in path:
                if cur_loc.startswith('h-'):
                    continue
                elif cur_loc.startswith('tor') and graph.has_edge(cur_loc, sink):
                    continue
                else:
                    slots[i % len(slots)]['srcs'].append(cur_loc)
                    slots[i % len(slots)]['dsts'].append(sink)
                    i += 1
    print(slots)
    pool = mp.Pool()
    paths = pool.map(_calc_spfs, slots)
    pairs = pool.map(_split_path, paths)
    pool.close()
    return _reduce_pairs_by_type(pairs, k)


def _make_edge_to_pairs_dict(graph: nx.DiGraph, k: int, hosts=None) -> Dict[Tuple[Any, Any], EdgeToPair]:
    """
    Make a dictionary that maps each edge in the graph to a list of source and
    destination pairs whose shortest path contains the edge. If an edge fails
    the shortest paths of the corresponding pairs have to be recalculated.

    Exclude all pairs that start with a host-node. Those are not valid start
    points for predictions with the neural network.
    Also excludes all ToR-host pairs where the corresponding ToR and host is
    directly connected.

    Note:
        Even if the graph is undirected, the returned dictionary will have the
        directed edges as keys. That is, the dictionary will have twice as many
        entries as an undirected graph has edges (given every edge actually
        carries a shortest path).

    Args:
        graph:
        hosts: Shortest paths will be calculated over the cartesian product
            of those nodes.

    Returns:

    """
    if hosts is None:
        hosts = graph.nodes()
    slots = [{'srcs': [], 'dsts': [], 'graph': graph} for _ in range(mp.cpu_count())]
    for i, (s, t) in enumerate(itertools.product(hosts, hosts)):
        if s == t:
            continue
        # Skip all current-location - destination pairs that start with a host.
        if s.startswith('h-'):
            continue
        # Identify all ToR-host pairs.
        if s.startswith('tor-') and t.startswith('h-'):
            # Check if this host is directly connected to this ToR. If so, skip
            # this pair. If the host is not directly connected to this ToR
            # include.
            if graph.has_edge(s, t):
                continue
        slots[i % len(slots)]['srcs'].append(s)
        slots[i % len(slots)]['dsts'].append(t)
    pool = mp.Pool()
    paths = pool.map(_calc_spfs, slots)
    pairs = pool.map(_split_path, paths)
    pool.close()
    return _reduce_pairs_by_type(pairs, k)


def _make_edge_to_paths(graph: nx.DiGraph, pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], List[str]]:
    """
    Create a mapping that returns for every edge in graph the paths running along
    that edge.

    Args:
        graph: Graph for which mapping should be calculated.
        pairs: Soure and destination pairs.

    Returns:
        Mapping of edges to paths running over this edge.
    """
    ret = {}
    for i, (source, sink) in enumerate(pairs):
        if i % 100 == 0:
            print("Processed {:4d} of {:4d}.".format(i, len(pairs)))
        for path in nx.all_shortest_paths(graph, source, sink):
            for u, v in zip(path[:-1], path[1:]):
                if (u, v) not in ret:
                    ret[(u, v)] = []
                ret[(u, v)].append(path)
    return ret


def _get_output(graph: nx.DiGraph, u: Any, dst: Any, num_outputs: int,
                adj: Dict[Any, Dict[Any, int]],
                neighbor_to_idx: Dict[Any, Dict[Any, int]]) -> np.array:
    """
        Returns the target for the network to train. The first entry indicates if
        a packet should get dropped or not. It is set to one if the destination is
        not reachable from the current node. Else, the distribution over output ports
        is calculated and appended.

        Args:
            graph:
            u:
            dst:
            num_outputs:
            adj:
            neighbor_to_idx:

        Returns:
            output: numpy array with distribution over neighors.
    """
    output = np.zeros((1, num_outputs + 1), dtype=np.float32)
    if adj[u][dst] is None:
        output[0, 0] = 1.
    else:
        # unreachability is not problem. If u can reach dst, then all its
        # neighbors can also reach dst.
        output[0, 1:] = sprep._get_distributional_output(
            graph=graph,
            u=u,
            dst=dst,
            num_outputs=num_outputs,
            adj=adj,
            neighbor_to_idx=neighbor_to_idx
        )
    return output


class OutputCalculator(object):
    def __init__(self, graph: nx.DiGraph, cur_loc: str, dst: str):
        self.graph = graph
        self.cur_loc = cur_loc
        self.dst = dst
        try:
            self.tunnels = list(nx.all_shortest_paths(graph, cur_loc, dst))
        except nx.NetworkXNoPath:
            self.tunnels = None

    def _get_output_lcp(self, num_outputs: int, neighbor_to_idx: Dict[str, Dict[str, int]]) -> np.array:
        """
        Get the target for balancing traffic over the least utilized tunnel.

        1) Calculate all shortest paths between u and the destination based on the
            hop distance, called tunnel going forwards.
        3) Note all neighbors that initiate a tunnel and set the corresponding
            outputs to one in the output_ecmp array.
        2) Calculate the weight of each tunnel by summing up the weight of the
            consituting edges.
        3) Select the neighbor initializing the tunnel with the least weight as target
            in the output array.
        4) In case no tunnels were found in step 1, i.e., u is not connected to
            dst by any path, then the first entry is set to one.

        In case of core nodes, the output_ecmp is also the output array. This is
        because core switches would require a full view over the network, i.e.,
        updates from all nodes in the network to make a shortest tunnel forwarding
        decision. Thus, core switches are trained to balance traffic over their
        direct neighbors. This contigency occurs if the aggregation switch the
        core switch is connected to fails.

        Args:
            num_outputs:
            neighbor_to_idx:

        Returns:
            output: Target vector based on the weights of the tunnels, i.e., most
                likely one output is set only.
        """
        output = np.zeros((1, num_outputs + 1), dtype=np.float32)
        if self.tunnels is None:
            output[0, 0] = 1
        else:
            neighbors = []
            weights = []
            for path in self.tunnels:
                neighbor = path[1]
                neighbors.append(neighbor)
                weight = 0
                for a, b in zip(path[:-1], path[1:]):
                    weight += self.graph.edges[a, b][EDGE_WEIGHT]
                weights.append(weight)

            if self.cur_loc.startswith('core-'):
                min_weight = 1e9
            else:
                min_weight = np.min(weights)
            for n, w in zip(neighbors, weights):
                if w <= min_weight:
                    output[0, neighbor_to_idx[self.cur_loc][n] + 1] = 1.
        return output

    def _get_output_wcmp(self, num_outputs: int, neighbor_to_idx: Dict[str, Dict[str, int]]) -> np.array:
        """
        Get the target for balancing traffic over the least utilized tunnel.

        1) Calculate all shortest paths between u and the destination based on the
            hop distance, called tunnel going forwards.
        3) Note all neighbors that initiate a tunnel and set the corresponding
            outputs to one in the output_ecmp array.
        2) Calculate the weight of each tunnel by summing up the weight of the
            consituting edges.
        3) Select the neighbor initializing the tunnel with the least weight as target
            in the output array.
        4) In case no tunnels were found in step 1, i.e., u is not connected to
            dst by any path, then the first entry is set to one.

        In case of core nodes, the output_ecmp is also the output array. This is
        because core switches would require a full view over the network, i.e.,
        updates from all nodes in the network to make a shortest tunnel forwarding
        decision. Thus, core switches are trained to balance traffic over their
        direct neighbors. This contigency occurs if the aggregation switch the
        core switch is connected to fails.

        Args:
            num_outputs:
            neighbor_to_idx:

        Returns:
            output: Target vector based on the weights of the tunnels, i.e., most
                likely one output is set only.
        """
        output = np.zeros((1, num_outputs + 1), dtype=np.float32)
        if self.tunnels is None:
            output[0, 0] = 1
        else:
            neighbors = []
            weights = []
            for path in self.tunnels:
                neighbor = path[1]
                neighbors.append(neighbor)
                weight = 0
                # Edge weight is free capacity --> thus sum up the free capacity
                # along each path and route along the neighbors with maximum.
                for a, b in zip(path[:-1], path[1:]):
                    weight = self.graph.edges[a, b][EDGE_WEIGHT] + weight
                weights.append(weight)

            tmp = {n: 0 for n in neighbors}
            for n, w in zip(neighbors, weights):
                tmp[n] = np.max([tmp[n], w])
            nom = 0
            for w in tmp.values():
                nom += w
            for n, w in tmp.items():
                output[0, neighbor_to_idx[self.cur_loc][n] + 1] = w / nom
        return output

    def _get_output_hula(self, num_outputs: int, neighbor_to_idx: Dict[str, Dict[str, int]]) -> np.array:
        """
        Get the target for balancing traffic over the least utilized tunnel.

        1) Calculate all shortest paths between u and the destination based on the
            hop distance, called tunnel going forwards.
        3) Note all neighbors that initiate a tunnel and set the corresponding
            outputs to one in the output_ecmp array.
        2) Calculate the weight of each tunnel by summing up the weight of the
            consituting edges.
        3) Select the neighbor initializing the tunnel with the least weight as target
            in the output array.
        4) In case no tunnels were found in step 1, i.e., u is not connected to
            dst by any path, then the first entry is set to one.

        In case of core nodes, the output_ecmp is also the output array. This is
        because core switches would require a full view over the network, i.e.,
        updates from all nodes in the network to make a shortest tunnel forwarding
        decision. Thus, core switches are trained to balance traffic over their
        direct neighbors. This contigency occurs if the aggregation switch the
        core switch is connected to fails.

        Args:
            num_outputs:
            neighbor_to_idx:

        Returns:
            output: Target vector based on the weights of the tunnels, i.e., most
                likely one output is set only.

        """
        output = np.zeros((1, num_outputs + 1), dtype=np.float32)
        if self.tunnels is None:
            output[0, 0] = 1
        else:
            neighbors = []
            weights = []
            for path in self.tunnels:
                neighbor = path[1]
                neighbors.append(neighbor)
                weight = 0
                for a, b in zip(path[:-1], path[1:]):
                    weight = np.max([self.graph.edges[a, b][EDGE_WEIGHT], weight])
                weights.append(weight)

            min_weight = np.min(weights)
            for n, w in zip(neighbors, weights):
                if w <= min_weight:
                    output[0, neighbor_to_idx[self.cur_loc][n] + 1] = 1.
        return output

    def _get_output_ecmp(self, num_outputs: int, neighbor_to_idx: Dict[str, Dict[str, int]]) -> np.array:
        output = np.zeros((1, num_outputs + 1), dtype=np.float32)
        if self.tunnels is None:
            output[0, 0] = 1
        else:
            for path in self.tunnels:
                neighbor = path[1]
                output[0, neighbor_to_idx[self.cur_loc][neighbor] + 1] = 1.
        return output

    def __call__(self, output_type: str, num_outputs: int, neighbor_to_idx: Dict[str, Dict[str, int]]) -> np.array:
        if output_type == OUTPUT_ECMP:
            return self._get_output_ecmp(num_outputs, neighbor_to_idx)
        elif output_type == OUTPUT_LCP:
            return self._get_output_lcp(num_outputs, neighbor_to_idx)
        elif output_type == OUTPUT_HULA:
            return self._get_output_hula(num_outputs, neighbor_to_idx)
        elif output_type == OUTPUT_WCMP:
            return self._get_output_wcmp(num_outputs, neighbor_to_idx)
        raise ValueError("Unknown output type {}".format(output_type))


def _get_output_distances(graph: nx.DiGraph, u: str, dst: str, num_outputs: int,
                     neighbor_to_idx: Dict[str, Dict[str, int]]) -> np.array:
    output = np.zeros((1, num_outputs + 1), dtype=np.float32)
    try:
        for v in nx.neighbors(graph, u):
            dist = nx.shortest_path_length(graph, v, dst, weight=EDGE_WEIGHT)
            output[0, neighbor_to_idx[u][v] + 1] = dist
    except nx.NetworkXNoPath:
        output[0, 0] = 1
    return output


def _get_reduced_network_state2(graph: nx.DiGraph, num_non_leaves: int, max_degree: int,
                                neighbor_to_idx: Dict[Any, Dict[Any, int]]) -> np.array:
    """
        Include only those nodes in the state that have an degree larger one.

        Info:
            The returned state has three dimensions:
                - dim0: Is one, used for concatenation.
                - dim1: Number of nodes with a degree larger one. State includes
                    every node with a degree larger than one.
                - dim2: Max_degree: The state info contains a view over all edges
                    in the network. Each node calculates a HLSA from its incident
                    edges.
                - dim3: The state of each link. Currently has four attributes:
                    - Attribute 1: Is one if the link from the current node to its
                        neighboring node is *up* else zero.
                    - Attribute 2: Is one if the link from the current node to its
                        neighboring node is *down*, else zero.
                    - Attribute 3: Is one if the link from the neighbor to the
                        current node is *up*, else zero.
                    - Attribute 4: Is one if the link from the neighbor to the
                        current node is *down*, else zero.
                    If edge weights are set, then the weight in each direction is
                    added:
                    - Attribute 5: Edge weight from the current node to the neighbor.
                    - Attribute 6: Edge weight from the neighbor to the current node.
                    If the link is down, a weight of -1 is used instead.
            In addition, node failure are handled assuming they died suddenly.
            In this case, normal operation is mimicked, i.e., all edges of that
            node are assumed to be up and in use. This would correspond to a
            old HLSA entry on the other nodes. The other nodes have to figure out
            the node failure from the states of the neighbors.
        Args:
            graph: Graph with failed edges.
            failed_links: Links that failed.
            max_degree: Maximum degree in the network.
            num_non_leaves: Number of nodes in the original graph that have a
                degree of larger one.
            neighbor_to_idx: Dictionary mapping the neighbors of every node to
                an index.

        Returns:
        state: Currently a binary array of shape (1, num_non_leaves, max_degree, 4).
    """
    assert graph.is_directed(), "Graph must be directed, else not sure if its working."
    graph_is_weighted = EDGE_WEIGHT is not None

    # Create a default state. Fill it with data as if all edges in the graph
    # were down.
    state = np.zeros((1, num_non_leaves, max_degree, 4), dtype=np.float32)
    state[:, :, :, 1] = 1
    state[:, :, :, 3] = 1
    if graph_is_weighted:
        state = np.concatenate((
            state,
            -1 * np.ones((1, num_non_leaves, max_degree, 2), dtype=np.float32)
        ), axis=-1)

    for u in graph.nodes():
        if not sprep.NS_IDX in graph.nodes[u]:
            continue
        u_idx = graph.nodes[u][sprep.NS_IDX]
        neighbors = list(graph.neighbors(u))
        if len(neighbors) == 0:
            # If the node has no neighbors, then it failed. The assumption is,
            # that it failed after normal operation, i.e., all edges on this node
            # were up and running and traffic was traversing this node. Would
            # then represent the HLSA last received from this node.
            # for i in range(max_degree):
            #     state[:, u_idx, i, 0] = 1
            #     state[:, u_idx, i, 1] = 0
            #     state[:, u_idx, i, 2] = 1
            #     state[:, u_idx, i, 3] = 0
            #     if graph_is_weighted:
            #         state[:, u_idx, i, 4] = _sample_uniform_noise()
            #         state[:, u_idx, i, 5] = _sample_uniform_noise()
            pass
        else:
            # The node has some active edges. Set the corresponding entries for
            # the node u itself and for its neighbor.
            for v in neighbors:
                v_neighbor_idx = neighbor_to_idx[u][v]
                u_neighbor_idx = neighbor_to_idx[v][u]
                state[:, u_idx, v_neighbor_idx, 0] = 1
                state[:, u_idx, v_neighbor_idx, 1] = 0
                if graph_is_weighted:
                    state[:, u_idx, v_neighbor_idx, 4] = graph.edges[u, v][EDGE_WEIGHT]

                if not sprep.NS_IDX in graph.nodes[v]:
                    if graph.has_edge(v, u):
                        state[0, u_idx, v_neighbor_idx, 2] = 1
                        state[0, u_idx, v_neighbor_idx, 3] = 0
                    if graph_is_weighted:
                        state[0, u_idx, v_neighbor_idx, 5] = graph.edges[v, u][EDGE_WEIGHT]
                else:
                    v_idx = graph.nodes[v][sprep.NS_IDX]
                    state[:, v_idx, u_neighbor_idx, 2] = 1
                    state[:, v_idx, u_neighbor_idx, 3] = 0
                    if graph_is_weighted:
                        state[:, v_idx, u_neighbor_idx, 5] = graph.edges[u, v][EDGE_WEIGHT]
    return state


def _get_reduced_network_state(graph: nx.DiGraph, failed_links: List[Tuple[Any, Any]],
                       num_non_leaves: int, max_degree: int,
                       neighbor_to_idx: Dict[Any, Dict[Any, int]]) -> np.array:
    """
        Include only those nodes in the state that have an degree larger one.

        Info:
            The returned state has three dimensions:
                - dim0: Is one, used for concatenation.
                - dim1: Number of nodes with a degree larger one. State includes
                    every node with a degree larger than one.
                - dim2: Max_degree: The state info contains a view over all edges
                    in the network. Each node calculates a HLSA from its incident
                    edges.
                - dim3: The state of each link. Currently has four attributes:
                    - Attribute 1: Is one if the link from the current node to its
                        neighboring node is *up* else zero.
                    - Attribute 2: Is one if the link from the current node to its
                        neighboring node is *down*, else zero.
                    - Attribute 3: Is one if the link from the neighbor to the
                        current node is *up*, else zero.
                    - Attribute 4: Is one if the link from the neighbor to the
                        current node is *down*, else zero.
        Args:
            graph: Graph with failed edges.
            failed_links: Links that failed.
            max_degree: Maximum degree in the network.
            num_non_leaves: Number of nodes in the original graph that have a
                degree of larger one.
            neighbor_to_idx: Dictionary mapping the neighbors of every node to
                an index.

        Returns:
        state: Currently a binary array of shape (1, num_non_leaves, max_degree, 4).
    """
    if EDGE_WEIGHT is None:
        state = np.zeros((1, num_non_leaves, max_degree, 4), dtype=np.float32)
    else:
        state = np.zeros((1, num_non_leaves, max_degree, 5), dtype=np.float32)

    def process_edge(u, v, offset, weight=None):
        if sprep.NS_IDX in graph.nodes[u]:
            u_idx = graph.nodes[u][sprep.NS_IDX]
            # On node u, the outgoing edge from to v is active.
            state[0, u_idx, neighbor_to_idx[u][v], 0 + offset] = 1
            if weight is not None:
                state[0, u_idx, neighbor_to_idx[u][v], -1] = graph.edges[(u, v)][weight]
        if sprep.NS_IDX in graph.nodes[v]:
            v_idx = graph.nodes[v][sprep.NS_IDX]
            # On node v, the incoming edge from u is active.
            state[0, v_idx, neighbor_to_idx[v][u], 2 + offset] = 1

        if not graph.is_directed():
            # If graph is undirected, edge (v, u) is not included, thus manually
            # add here. If graph is directed, then (v, u) will be iterated
            # over if it exists.
            if sprep.NS_IDX in graph.nodes[u]:
                u_idx = graph.nodes[u][sprep.NS_IDX]
                # On node u, the incoming edge from v is active.
                state[0, u_idx, neighbor_to_idx[u][v], 2 + offset] = 1
            if sprep.NS_IDX in graph.nodes[v]:
                v_idx = graph.nodes[v][sprep.NS_IDX]
                # On node v, the outgoing edge to node u is active.
                state[0, v_idx, neighbor_to_idx[v][u], 0 + offset] = 1
                if weight is not None:
                    # Put the weight for the reverse direction here and not
                    # above since for directed graphs, the reverse direction can
                    # fail and then lead to an error. If the graph is undirected,
                    # and the reverse direction exists, then it will be set in the
                    # above part, since the edge will be iterated over.
                    state[0, v_idx, neighbor_to_idx[v][u], -1] = graph.edges[(v, u)][weight]

    for u, v in graph.edges():
        process_edge(u, v, 0, EDGE_WEIGHT)

    for u, v in failed_links:
        process_edge(u, v, 1, None)

    return state


def _get_network_state(graph: nx.DiGraph, failed_links: List[Tuple[Any, Any]],
                       max_degree: int, neighbor_to_idx: Dict[Any, Dict[Any, int]]) -> np.array:
    """
        Return the state of the network.

        Info:
            The returned state has three dimensions:
                - dim0: Is one, used for concatenation.
                - dim1: Number of nodes. State includes every node.
                - dim2: Max_degree: The state info contains a view over all edges
                    in the network. Each node calculates a HLSA from its incident
                    edges.
                - dim3: The state of each link. Currently has four attributes:
                    - Attribute 1: Is one if the link from the current node to its
                        neighboring node is *up* else zero.
                    - Attribute 2: Is one if the link from the current node to its
                        neighboring node is *down*, else zero.
                    - Attribute 3: Is one if the link from the neighbor to the
                        current node is *up*, else zero.
                    - Attribute 4: Is one if the link from the neighbor to the
                        current node is *down*, else zero.
        Args:
            graph:
            failed_links:
            max_degree:
            neighbor_to_idx:

        Returns:
        state: Currently a binary array of shape (1, num_nodes, max_degree, 4).
    """
    state = np.zeros((1, graph.number_of_nodes(), max_degree, 4), dtype=np.float32)

    def process_edge(u, v, offset):
        u_idx = graph.nodes[u][sprep.IDX]
        v_idx = graph.nodes[v][sprep.IDX]
        # On node u, the outgoing edge from to v is active.
        state[0, u_idx, neighbor_to_idx[u][v], 0 + offset] = 1
        # On node v, the incoming edge from u is active.
        state[0, v_idx, neighbor_to_idx[v][u], 2 + offset] = 1
        if not graph.is_directed():
            # If graph is undirected, edge (v, u) is not included, thus manually
            # add here. If graph is directed, then (v, u) will be iterated
            # over if it exists.

            # On node u, the incoming edge from v is active.
            state[0, u_idx, neighbor_to_idx[u][v], 2 + offset] = 1
            # On node v, the outgoing edge to node u is active.
            state[0, v_idx, neighbor_to_idx[v][u], 0 + offset] = 1

    for u, v in graph.edges():
        process_edge(u, v, 0)

    for u, v in failed_links:
        process_edge(u, v, 1)

    return state


def _get_all_neighbors(graph: nx.DiGraph, max_degree: int,
                   neighbor_to_index: Dict[Any, Dict[Any, int]]) -> np.array:
    """
        Get attention masks for all nodes.

        Args:
            graph:
            max_degree:
            neighbor_to_index:

        Returns:
            mask: Integer array of shape (num_nodes, max_degree, 1).
    """
    neighbors = np.zeros((graph.number_of_nodes(), max_degree, 1), dtype=np.float32)
    for u, v in graph.edges():
        u_idx = graph.nodes[u][sprep.IDX]
        neighbors[u_idx, neighbor_to_index[u][v], 0] = graph.nodes[v][sprep.IDX]
        if not graph.is_directed():
            v_idx = graph.nodes[u][sprep.IDX]
            neighbors[v_idx, neighbor_to_index[v][u], 0] = graph.nodes[u][sprep.IDX]
    return neighbors


def _get_all_masks(graph: nx.DiGraph, max_degree: int, failed_links: List[Tuple[Any, Any]],
                   neighbor_to_index: Dict[Any, Dict[Any, int]]) -> np.array:
    """
        Get attention masks for all nodes.

        Args:
            graph:
            max_degree:
            neighbor_to_index:

        Returns:
            mask: Binary array of shape (num_nodes, max_degree, 1).
    """

    def set_mask(u, v):
        u_idx = graph.nodes[u][sprep.IDX]
        mask[u_idx, neighbor_to_index[u][v], 0] = 1.
        if not graph.is_directed():
            v_idx = graph.nodes[u][sprep.IDX]
            mask[v_idx, neighbor_to_index[v][u], 0] = 1

    mask = np.zeros((graph.number_of_nodes(), max_degree, 1), dtype=np.float32)
    for u, v in graph.edges():
        set_mask(u, v)
    # Do also include the neighbors that are no longer reachable.
    for u, v in failed_links:
        set_mask(u, v)
    return mask


def _apply_link_failures(graph: nx.DiGraph, edges_to_pairs: Dict[Tuple[Any, Any], Tuple[Any, Any]],
                         failed_links: List[Tuple[Any, Any]]) -> Tuple[nx.DiGraph, List[Tuple[Any, Any]]]:
    """
        Remove the edges from the graph and retrieve the node pairs for which the
        shoretst paths must be re-calculated.

        Args:
            graph:
            failed_links:

        Returns:
            graph: Graph with removed edges, note that the passed graph is mutated as well.
            to_recalculate: Union of all source destination pairs that had a shortest
                path running over one of the failed links.
    """
    to_recalculate = []
    for u, v in failed_links:
        graph.remove_edge(u, v)
        for s, t in edges_to_pairs[(u, v)]:
            # if (s, t) not in to_recalculate:
            to_recalculate.append((s, t))
    return graph, to_recalculate


def _update_adjacencies(graph: nx.DiGraph, to_recalculate: List[Tuple[Any, Any]],
                        adj: Dict[Any, Dict[Any, int]]) -> Dict[Any, Dict[Any, int]]:
    """
        Recalculate the shortest paths for all pairs whose path got invalid due to
        a link failure. If two nodes are no longer reachable indicate this with None.
        Mutates the argument adj.

        Args:
            graph:
            to_recalculate:
            adj:

        Returns:
            adj.
    """
    for s, t in to_recalculate:
        try:
            adj[s][t] = nx.shortest_path_length(graph, s, t)
        except nx.NetworkXNoPath as e:
            # logger.info("No path between {} and {} anymore.".format(s, t))
            adj[s][t] = None
        except Exception as e:
            logger.exception(e)
            raise e
    return adj


def _update_adjacencies_ligth(graph: nx.DiGraph, current_loc: Any, dst: Any,
                              adj: Dict[Any, Dict[Any, int]]) -> Dict[Any, Dict[Any, int]]:
    """
        Update the adjacency matrix for only the one sample that is changed.
    """
    changes = {current_loc: {dst: adj[current_loc][dst]}}
    for u in nx.neighbors(graph, current_loc):
        changes[u] = {}
    try:
        adj[current_loc][dst] = nx.shortest_path_length(graph, current_loc, dst, weight=EDGE_WEIGHT)
        for v in nx.neighbors(graph, current_loc):
            changes[v][dst] = adj[v][dst]
            adj[v][dst] = nx.shortest_path_length(graph, v, dst, weight=EDGE_WEIGHT)
    except nx.NetworkXNoPath as e:
        adj[current_loc][dst] = None
    except Exception as e:
        logger.exception(e)
        raise e
    return changes


def _restore_changes(adj: Dict[Any, Dict[Any, int]], changes: Dict[Any, Dict[Any, int]]):
    """
        Restore the adjacencey matrix with the previously stored original calues
        thate have changed.
    """
    for u, tmp in changes.items():
        for v, d in tmp.items():
            adj[u][v] = d


def _sample_pair(edge_to_pairs: Dict[Tuple[Any, Any], EdgeToPair],
                 failed_links: List[Tuple[Any, Any]], random: np.random.RandomState) -> List[Tuple[Any, Any]]:
    """
    Sample a communication pair from one of those running over a failed link.

    1) Sample failed edge at random.
    2) Sample pair at random.
        2.1) Sample node type of source.
        2.2) Sample node type of dst.
        2.3) Sample random pair matching those.

    Args:
        edge_to_pairs:
        failed_links:

    Returns:

    """
    source = None
    sink = None
    source2 = None
    sink2 = None
    # Return a pair from any edge if no link failures are sampled.
    if len(failed_links) == 0:
        failed_links = list(edge_to_pairs.keys())
    for i in range(20):
        # Important: Even if the graph itself is undirected, edge_to_pairs will
        # operate on directed edges!
        e = failed_links[random.randint(0, len(failed_links))]
        t1 = random.choice(edge_to_pairs[e].get_cur_loc_types())
        t2 = random.choice(edge_to_pairs[e].get_destination_types(t1))
        n_pairs = len(edge_to_pairs[e].pairs[t1][t2])
        if n_pairs > 0:
            source, sink = edge_to_pairs[e].pairs[t1][t2][random.randint(0, n_pairs)]
            break
    # if len(failed_links) > 0:
    #     # In case of link failures, get any pair that is not affected by
    #     # this particular failure.
    #     all_links = list(edge_to_pairs.keys())
    #     for i in range(20):
    #         e = all_links[random.randint(0, len(all_links))]
    #         t1 = random.choice(edge_to_pairs[e].get_cur_loc_types())
    #         t2 = random.choice(edge_to_pairs[e].get_destination_types(t1))
    #         n_pairs = len(edge_to_pairs[e].pairs[t1][t2])
    #         if n_pairs > 0:
    #             source2, sink2 = edge_to_pairs[e].pairs[t1][t2][random.randint(0, n_pairs)]
    #             break
    # return [(source, sink), (source2, sink2)]
    return [(source, sink)]


def _sample_pairs_from_paths(edge_to_paths: Dict[Tuple[str, str], List[str]],
                             failed_links: List[Tuple[str, str]], graph: nx.DiGraph,
                             random: np.random.RandomState) -> List[Tuple[str, str]]:
    def fill_pairs(path, pairs):
        dst_tor = path[-2]

        # Get the hosts connected to the ToR switch.
        hs = [n for n in graph.neighbors(dst_tor) if n.startswith('h-')]
        # If the ToR failed (or all hosts), then get the hosts connected to the
        # ToR.
        if len(hs) == 0:
            num = _get_num(dst_tor)
            hs = ['h-{:04d}'.format(i) for i in range(num * 4, (num + 1) * 4)]
        for cur_loc in path[:-2]:
            if nx.degree(graph)[cur_loc] == 0:
                # Do not include nodes without any neighbors.
                continue
            dst = hs[random.randint(0, len(hs))]
            pairs.append((cur_loc, dst))

    pairs = []
    if len(failed_links) == 0:
        failed_links = list(edge_to_paths.keys())
    i = 0
    while len(pairs) == 0 and i < 20:
        i += 1
        if i == 10:
            failed_links = list(edge_to_paths.keys())
        e = failed_links[random.randint(0, len(failed_links))]
        if e[0].startswith('h-'):
            # Swap the pairs here. If the host-to-tor direction failed, then
            # consider the tor to host direction. If the upstream direction fails
            # then this has no influence on the network. Thus those edges are
            # not in edge_to_paths. Since both directions fail, the other one
            # is relevant instead.
            e = (e[1], e[0])
        if len(edge_to_paths[e]) == 0:
            continue
        path = edge_to_paths[e][random.randint(0, len(edge_to_paths[e]))]
        fill_pairs(path, pairs)
        # Get an existing shortest path.
        try:
            paths = list(nx.all_shortest_paths(graph, path[0], path[-1]))
            path = paths[random.randint(0, len(paths))]
            fill_pairs(path, pairs)
        except nx.NetworkXNoPath:
            pass
        # weights = []
        # paths = []
        # try:
        #     for path in nx.all_shortest_paths(graph, path[0], path[-1]):
        #         paths.append(path)
        #         weight = 0
        #         for u, v in zip(path[:-1], path[1:]):
        #             weight += graph.edges[u, v][EDGE_WEIGHT]
        #         weights.append(weight)
        #     path = paths[np.argmin(weights)]
        #     for cur_loc in path[:-2]:
        #         if nx.degree(graph)[cur_loc] == 0:
        #             # Do not include nodes without any neighbors.
        #             continue
        #         pairs.append((cur_loc, path[-1]))
        # except nx.NetworkXNoPath:
        #     pass
        # break
    assert len(pairs) > 0
    return pairs


def _sample_pair_tor0() -> List[Tuple[Any, Any]]:
    source = 'tor-0000'
    return [(source, 'h-{:04d}'.format(i)) for i in range(4, 128)]


def _sample_pairs_single_source(node: str) -> List[Tuple[str, str]]:
    if node.startswith('tor'):
        return _sample_pair_tor0()
    assert node.startswith('agg') or node.startswith('core')
    return [(node, 'h-{:04d}'.format(i)) for i in range(0, 128)]


def _link_failure_to_samples(graph: nx.DiGraph, links: List[Tuple[Any, Any]],
        num_outputs: int, value_index: Dict[Any, Dict[Any, int]], output_types: List[str],
        adj: Dict[Any, Dict[Any, int]], cur_loc: Any, destination: Any,
        num_non_leaves: int) -> Tuple[np.array, Dict[str, np.array], np.array, np.array]:
    """
        For one source destination pair sampled from those running over one of 
        the failed links recalculate the shortest paths and create the
        corresponding samples.

        Args:
            graph: Graph in which link failures are already applied.
            links: The links that have failed, i.e., those links are missing
                from the argument `graph`.
            num_outputs: The number of outputs corresponding to the maximum
                degree in the network.
            value_index: Mapping that maps the neighbors of every node to a
                number between 0 and num_outputs.
            adj: Adjcaceny matrix containing the shortest path distance between
                pair of node on the graph with applied link failures.
            degrees: Dictionary mapping node names to the node degree in the
                original graph (i.e., graph without link failures).
            num_non_leave: Number of nodes that are no leaves, i.e., have
                a degree larger one (or two in case of directed) in the original
                graph.
            to_recalculate: List containing source destination pairs that should
                be recalculated since they have a shortest path running over one
                of the failed links.
            seed: Seed for random number generator.

        Returns:
            network_state: Array of shape (BS, num_nodes, max_degree, num_features)
                containing the state of the network, i.e., the link state.
            targets: Array of shape (BS, num_outputs + 1), distribution over
                neighbors.
            destinations: Array of shape (BS, 1), index of destination node.
            cur_locs: Array of shape (BS, 1), index of current locations, i.e.,
                index of node the targets belong to.
    """
    # backup = _update_adjacencies_ligth(graph, cur_loc, destination, adj)
    output_gen = OutputCalculator(graph, cur_loc, destination)
    state = _get_reduced_network_state2(
        graph=graph,
        max_degree=num_outputs,
        neighbor_to_idx=value_index,
        num_non_leaves=num_non_leaves
    )
    network_states = []
    destinations = []
    cur_locs = []

    network_states.append(state)
    targets = {t: output_gen(t, num_outputs, value_index) for t in output_types}

    destinations.append([graph.nodes[destination][sprep.IDX]])
    cur_locs.append([graph.nodes[cur_loc][sprep.IDX]])

    # _restore_changes(adj, backup)

    return np.concatenate(network_states), targets, np.array(destinations), np.array(cur_locs)


def _process_failed_links(args) -> Tuple[np.array, Dict[str, np.array], np.array, np.array]:
    """
        Make state and targets for a set of link failures.

        Args:
            args: Array with the required attributes.

        Note: The attribute `args` has the following items:
            - links: List of tuples of nodes representing edges that are failing.
            - graph: The graph in which the edges are failing.
            - edges_to_pairs: Dictionary mapping edges to the source destination
                pairs that have a shortest path running along it.
            - num_outputs: The number of outputs, i.e., the maximum degree in the
                network.
            - value_index: A dictionary mapping each neigbhor of a node to an index.
                This dictionary is used to have the output and the location of a
                neighbor in attention stuff be equal.
            - adj: A dictionary mapping pairs of nodes to their distance.

        Note: The target has max_degree + 1 dimensions. The first dimension is set
            to one if a destination node is no longer reachable from a source node.

        Returns:
            network_states: Array of network states of shape (M, num_nodes, max_degree, 4).
            targets: Array of shape (M, max_degree + 1).
            destinations: Array of shape (M, 1). Contains the indices of the destination
                nodes.
            cur_locs: Array of shape (M, 1). Contains the indices of the nodes that
                are associated with the target.
    """
    links = args['links']
    graph = args['graph']
    edges_to_pairs = args['edges_to_pairs']
    num_outputs = args['num_outputs']
    value_index = args['value_index']
    num_non_leaves = args['num_no_leaves']
    seed = args['seed']
    output_types = args['output_types']
    random = np.random.RandomState(seed=seed)

    for u, v in links:
        graph.remove_edge(u, v)

    # graph, to_recalculate = _apply_link_failures(graph, edges_to_pairs, links)
    # adj = _update_adjacencies(graph, to_recalculate, adj)
    # cur_loc, destination = _sample_pair(edges_to_pairs, links, random)
    # pairs = _sample_pair(edges_to_pairs, links, random)
    pairs = _sample_pairs_from_paths(edges_to_pairs, links, graph, random)
    # pairs = _sample_pairs_single_source('core-0000')
    # print(links)
    states, targets, dests, cur_locs = zip(*[
        _link_failure_to_samples(
            graph=graph,
            links=[],
            num_outputs=num_outputs,
            value_index=value_index,
            adj={},
            output_types=output_types,
            cur_loc=cur_loc,
            destination=destination,
            num_non_leaves=num_non_leaves
        ) for cur_loc, destination in pairs])
    for u, v in links:
        graph.add_edge(u, v)
    targets_ret = {k: None for k in targets[0].keys()}
    for k in targets_ret.keys():
        targets_ret[k] = np.concatenate([t[k] for t in targets])
    return np.concatenate(states), targets_ret, np.concatenate(dests), np.concatenate(cur_locs)


def _sample_num_failures(random: np.random.RandomState, max_num_failures: int) -> int:
    return random.randint(0, max_num_failures + 1)


def _sample_failures(random: np.random.RandomState, edges: List[Tuple[Any, Any]],
        max_num_failures: int) -> List[Tuple[Any, Any]]:
    """
        Sample a list of edges that fail.

        Args:
            random: Random number generator.
            edges: List of all edges in the graph that are fallible.
            max_num_failures: Maximum number of concurrent failures.

        Returns:
            failed_edges: A list containing the edges that failed.
    """
    a = np.arange(len(edges))
    num_failures = _sample_num_failures(random, max_num_failures)
    if num_failures == 0:
        failed_edges = random.choice(a, replace=False, size=num_failures)
        failed_edges = [edges[i] for i in failed_edges]
    else:
        failed_edges = []
    return failed_edges


def _sample_failure_fat_tree(random: np.random.RandomState, max_num_failures: int,
                             k: int, graph: nx.DiGraph) -> List[Tuple[Any, Any]]:
    """
    Sample every type of edge equally often. Depending on the edge that fails
    and the corresponding pairs that are sampled the patterns the neural network
    has to learn differ strongly. To enable better learning, sample the different
    edge types with equal probability.

    Args:
        random:
        max_num_failures:
        k:

    Returns:

    """
    num_failures = _sample_num_failures(random, max_num_failures)
    num_cores = k**2 / 4
    num_tors = k**2 / 2
    num_aggs = k**2 / 2
    randint = lambda x: random.randint(0, x)
    tail_samples = [
        lambda: 'tor-{:04d}'.format(randint(num_tors)),
        lambda: 'core-{:04d}'.format(randint(num_cores)),
        lambda: 'agg-{:04d}'.format(randint(num_aggs))
    ]
    failures = []
    while len(failures) < num_failures:
        tail = tail_samples[random.randint(0, len(tail_samples))]()
        head = random.choice(list(graph.neighbors(tail)))
        if (tail, head) in failures:
            continue
        # elif not graph.is_directed() and (head, tail) in failures:
        #     continue
        else:
            failures.append((tail, head))
            if graph.is_directed():
                failures.append((head, tail))
    return failures


def _sample_node_failure_fat_tree(random: np.random.RandomState, max_num_failures: int,
                                  k: int, graph: nx.DiGraph) -> List[Tuple[Any, Any]]:
    """
    Randomly sample a node failure and return all edges that lead from or to this
    node. Those are then removed from the graph in addition to normal edge
    failures.

    Args:
        random:
        max_num_failures:
        k:
        graph:

    Returns:

    """
    num_failures = _sample_num_failures(random, max_num_failures)
    num_cores = k**2 / 4
    num_tors = k**2 / 2
    num_aggs = k**2 / 2
    randint = lambda x: random.randint(0, x)
    tail_samples = [
        lambda: 'tor-{:04d}'.format(randint(num_tors)),
        lambda: 'core-{:04d}'.format(randint(num_cores)),
        lambda: 'agg-{:04d}'.format(randint(num_aggs))
    ]
    failures = []
    while len(failures) < num_failures:
        loc = tail_samples[random.randint(0, len(tail_samples))]()
        if loc in failures:
            continue
        else:
            failures.append(loc)
    edges = []
    for u in failures:
        for v in graph.neighbors(u):
            edges.append((u, v))
            if graph.is_directed() and graph.has_edge(v, u):
                edges.append((v, u))
    return edges


def _are_hosts_under_same_pod(host_one: int, host_two: int, k: int) -> bool:
    """
    Check if the two hosts are connected to the same tor.

    Args:
        host_one:
        host_two:

    Returns:

    """
    tor_one = int(host_one / (k / 2))
    tor_two = int(host_two / (k / 2))
    return tor_one == tor_two


def _create_all_pairs_of_hosts_not_same_tor(k: int) -> List[Tuple[str, str]]:
    """
    Create all pairs of hosts that are not connected to the same ToR node.
    Args:
        k:

    Returns:

    """
    pairs = []
    for i, j in itertools.product(list(range(int(k ** 3 / 4))), list(range(int(k ** 3 / 4)))):
        if i == j:
            continue
        if _are_hosts_under_same_pod(i, j, k):
            continue
        pairs.append(('h-{:04d}'.format(i), 'h-{:04d}'.format(j)))
    return pairs


def _create_pod_to_pod_pattern(k: int) -> List[Tuple[str, str]]:
    """
    Assume a three layered pattern in the fat-tree:
    pod1 <-> pod2 <-> pod3
    Repeat this pattern for all pods, truncate if not enough.

    Args:
        k:

    Returns:

    """
    assert k == 8, "not implemented for k unequal 8"
    pairs = []
    n_hosts = int(k ** 2 / 4)
    pods = [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7)]
    for src_pod, dst_pod in pods:
        it = zip(
            list(range(n_hosts * src_pod, n_hosts * (src_pod + 1))),
            list(range(n_hosts * dst_pod, n_hosts * (dst_pod + 1))),
        )
        for src_idx, dst_idx in it:
            src = 'h-{:04d}'.format(src_idx)
            dst = 'h-{:04d}'.format(dst_idx)
            pairs.append((src, dst))
            pairs.append((dst, src))
    return pairs


def _sample_pair_of_hosts_not_same_tor(k: int, random: np.random.RandomState) -> Tuple[str, str]:
    num_hosts = int(k**3 / 4)
    pair = None
    while pair is None:
        host_one = random.randint(0, num_hosts)
        host_two = random.randint(0, num_hosts)
        if _are_hosts_under_same_pod(host_one, host_two, k):
            continue
        else:
            pair = ('h-{:04d}'.format(host_one), 'h-{:04d}'.format(host_two))
    return pair


def _sample_host_to_host_failures(random: np.random.RandomState, graph: nx.DiGraph,
                                  failed_links: List[Tuple[str, str]],
                                  edge_to_pairs: Dict[Tuple[str, str], EdgeToPair]) -> List[Tuple[str, str]]:
    """

    Args:
        random:
        max_num_failures:
        k:
        graph:
        failed_edges:

    Returns:

    """
    source = None
    sink = None
    # Return a pair from any edge if no link failures are sampled.
    if len(failed_links) == 0:
        failed_links = list(edge_to_pairs.keys())
    for i in range(20):
        e = failed_links[random.randint(0, len(failed_links))]
        num_pairs = len(edge_to_pairs[e].pairs['tor']['h-other-pod'])
        if num_pairs == 0:
            continue
        else:
            source, sink = edge_to_pairs[e].pairs['tor']['h-other-pod'][random.randint(0, num_pairs)]

    paths = nx.all_shortest_paths(graph, source, sink)
    path = paths[random.randint([0, len(paths)])]
    pairs = [(u, v) for u, v in zip(path[0:-1], paths[1:])]
    return pairs


def _get_num_no_leaves(graph: nx.DiGraph) -> int:
    """
        Get the number of nodes that have a degree of larger one in case of a
        undirected, and a degree larger two in case of a directed graph.
    """
    thresh = 2 if graph.is_directed() else 1
    return int(np.sum(np.array(list(dict(nx.degree(graph)).values())) > thresh))


def _sample_uniform_noise(random=None):
    # Interpretation: Free capacity.
    if random is None:
        random = np.random.RandomState(seed=int(time.time()))
    return float(random.uniform())


def _sample_lognormal_noise(random=None):
    # Interpretation: Free capacity. Probability of free capacity of 100%, i.e.,
    # no utilization is small.
    if random is None:
        random = np.random.RandomState(seed=int(time.time()))
    return float(np.clip(random.lognormal(0, 3), 0, 100)) / 100.


def _add_gaussian_edge_weights(graph: nx.DiGraph, random: np.random.RandomState) -> nx.DiGraph:
    """
    Add gaussian noise to the edges. The gaussian is centered at 6 and has
    a standard deviation of 1. This makes preparation for the neural network
    easier.

    Args:
        graph:
        random:

    Returns:

    """
    assert type(EDGE_WEIGHT) == str, "EDGE WEIGHT must be set to a string"
    lognorm = random.uniform() > 5.5
    for u, v in graph.edges():
        # graph.edges[(u, v)][EDGE_WEIGHT] = float(np.abs(random.normal(10., 1)))
        if lognorm:
            graph.edges[(u, v)][EDGE_WEIGHT] = _sample_lognormal_noise(random)
        else:
            graph.edges[(u, v)][EDGE_WEIGHT] = _sample_uniform_noise(random)
        # graph.edges[(u, v)][EDGE_WEIGHT] = float(np.round(random.uniform(0.1, 1), decimals=1))
    return graph


def get_link_failure_dataset(graph: nx.DiGraph, max_num_failures: int, seed: int,
                             num_samples: int, output_types: List[str], k: int=None,
                             cached_edges_to_pairs: str=None) -> Dict[str, np.array]:
    """
        Simulate a number of link failures and return the resulting data set.
        For each link failure all affected shortest paths are re-calculated. The
        output is a distribution over neighbors where the probabilities are proportional
        to how many neighbor lie on a shortest path.

        Args:
            graph: The graph on which link failures should be simulated.
            max_num_failures: The maximum number of simultaneous link failures.
            seed: Seed for the random number generators.
            num_samples: Number of different settings that are evaluated, i.e., different
                sets of simultaneous link failures.
            k: K parameter for fat tree, optional, required for fat tree type graphs.
            cached_edges_to_pairs: Path to a json file storing the edges-to-pairs
                structure and can be re-used instead of recomputing them.

        Returns:
            ret: Dictionary containing the resulting data set:
                - network_states: The state of the network represented as a tensor
                    of shape (N, |V|, D, F).
                - targets: The learning target, i.e., distribution over neighbors
                    and a special drop port of shape (N, D + 1).
                - destinations: Indices for the nodes that are the destination
                    for the current forwarding decision. Has shape (N, 1),
                - cur_locs: Indices for the current locations that must make a
                    forwarding decision. Has shape (N, 1).
                - all_masks: Masks using for attention mechanisms. Indicates which
                    of the entries in the third dimension of the states correspond
                    to existing neighbors. This is indicated with a one. Has
                    the shape (N, |V|, D, 1).
                - all_neighbors: Tensor containing for each node the indices of its
                    neighbors. If a node has less neighbors than the maximum degree,
                    the remaining ones are filled with -1. Has shape (|V|, D, 1).
                N corresponds to the number of training samples. One link failure
                prodcues multiple samples. V is the set of nodes, |V| is thus the
                number of nodes in the graph. D is the maximum degree in the network.
                The degree refers to the undirected graph. F is the number of
                features per neighbor.
    """
    if 'type' in graph.graph and graph.graph['type'] == 'FatTree':
        assert k is not None, "Graph is of type fat-tree but k is not set"
        assert graph.number_of_nodes() == k**3 / 4 + 5 / 4 * k**2, \
            "Number of nodes of graph and the ones calculated with k do not match {} vs {}".format(
                graph.number_of_nodes(),k**3 / 4 + 5 / 4 * k**2
            )
        graph_type = 'FatTree'
    else:
        graph_type = 'default'

    num_outputs = sprep._calc_num_outputs(graph)
    value_index = sprep._neighbor_to_index(graph)
    # adj = sprep._make_distance_dict(graph)
    num_no_leaves = _get_num_no_leaves(graph)
    logger.debug("Make edge_to_pairs...")
    # if cached_edges_to_pairs is None:
    #     edges_to_pairs = _make_edge_to_pairs_dict(graph, k)
    # else:
    #     edges_to_pairs = read_edges_to_pairs(cached_edges_to_pairs)

    # switches = [n for n in graph.nodes() if not n.startswith('h-')]
    # all = [n for n in graph.nodes()]
    # pairs = []
    # for u, v in itertools.product(switches, all):
    #     if u.startswith('tor-') and v.startswith('h-') and graph.has_edge(u, v):
    #         continue
    #     else:
    #         pairs.append(u, v)
    tors = [n for n in graph.nodes() if n.startswith('tor-')]
    hosts = [n for n in graph.nodes() if n.startswith('h-')]
    pairs = [(tor, host) for tor, host in itertools.product(tors, hosts) if not graph.has_edge(tor, host)]
    edges_to_paths = _make_edge_to_paths(graph, pairs)

    logger.debug("Get all masks...")
    all_masks = _get_all_masks(
        graph=graph,
        max_degree=num_outputs,
        failed_links=[],
        neighbor_to_index=value_index
    )
    logger.debug("Get all neighbors...")
    all_neighbors = _get_all_neighbors(
        graph=graph,
        max_degree=num_outputs,
        neighbor_to_index=value_index
    )
    nodes_with_state = np.zeros(num_no_leaves, dtype=np.int32)
    for n, d in graph.nodes(data=True):
        if sprep.NS_IDX in d:
            nodes_with_state[d[sprep.NS_IDX]] = d[sprep.IDX]

    logger.debug("Start getting state...")
    random = np.random.RandomState(seed=seed)
    edges = list(graph.edges())
    failure_selector = {
        'FatTree': lambda: _sample_failure_fat_tree(random, max_num_failures, k, graph),
        'default': lambda: _sample_failures(random, edges, max_num_failures)
    }
    node_failure_selector = {
        'FatTree': lambda: _sample_node_failure_fat_tree(random, 3, k, graph),
        'defaul': lambda: []
    }
    data = []
    t1 = time.time()
    for abc in range(num_samples):
        if abc % 100 == 0:
            logging.info("Did {:5d} of {:5d} in {:6.4f}s".format(abc, num_samples, time.time() - t1))
        try:
            _add_gaussian_edge_weights(graph, random)
            failed_edges = failure_selector[graph_type]()
            # Check if duplicate edges are sampled. Edges can be remove donly once
            # else nx raises an error.
            for u, v in node_failure_selector[graph_type]():
                if (u, v) not in failed_edges:
                    failed_edges.append((u, v))
            data.append(_process_failed_links({
                'links': failed_edges,
                'graph': graph, #.copy(),
                'edges_to_pairs': edges_to_paths,
                'num_outputs': num_outputs,
                'value_index': value_index,
                "output_types": output_types,
                # "deep" copy of dictionary. Must be in this way, since references
                # are not copied when calling copy on the outer dictionary.
                # 'adj': adj, # {d: k.copy() for d, k in adj.items()},
                'num_no_leaves': num_no_leaves,
                'seed': int(random.randint(1, int(2**32-1)))
            }))
        except Exception as e:
            logger.exception(e)
            print(e)

    network_states = []
    destinations = []
    cur_locs = []
    targets_ret = {k: [] for k in data[0][1].keys()}
    for n, t, d, c in data:
        network_states.append(n)
        for k in t.keys():
            targets_ret[k].append(t[k])
        destinations.append(d)
        cur_locs.append(c)
    network_states = np.concatenate(network_states)
    targets = {k: np.concatenate(targets_ret[k]) for k in targets_ret.keys()}
    destinations = np.concatenate(destinations)
    cur_locs = np.concatenate(cur_locs)

    logger.debug("Shape of {:20s} {}".format("network_states", str(network_states.shape)))
    logger.debug("Shape of {:20s} {}".format("targets", str(targets[OUTPUT_ECMP].shape)))
    logger.debug("Shape of {:20s} {}".format("destinations", str(destinations.shape)))
    logger.debug("Shape of {:20s} {}".format("cur_locs", str(cur_locs.shape)))
    logger.debug("Shape of {:20s} {}".format("all_masks", str(all_masks.shape)))
    logger.debug("Shape of {:20s} {}".format("all_neighbors", str(all_neighbors.shape)))
    logger.debug("Shape of {:20s} {}".format("nodes_with_state", str(nodes_with_state.shape)))

    return {
        'network_states': network_states,
        'targets': targets,
        'destinations': destinations,
        'cur_locs': cur_locs,
        'all_masks': all_masks,
        'all_neighbors': all_neighbors,
        'nodes_with_state': nodes_with_state
    }


if __name__ == '__main__':
    from topos.fattree import make_topo
    from dataprep.input_output import write_edges_to_pairs, write_edges_to_paths
    # edge_to_pairs = _make_edge_to_pairs_from_pairs(
    #     graph=make_topo(k=8),
    #     k=8,
    #     pairs=_create_pod_to_pod_pattern(8)
    # )
    # write_edges_to_pairs('/opt/project/data/edges-to-pairs-pod-to-pod.json', edge_to_pairs)
    tg = make_topo(k=8)
    tors = [n for n in tg.nodes() if n.startswith('tor-')]
    hosts = [n for n in tg.nodes() if n.startswith('h-')]
    pairs = [(tor, host) for tor, host in itertools.product(tors, hosts) if not tg.has_edge(tor, host)]
    edges_to_paths = _make_edge_to_paths(tg, pairs)
    write_edges_to_paths('/opt/project/data/fat-tree-k8/edges_to_paths.json', edges_to_paths)


