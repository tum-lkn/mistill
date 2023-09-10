import unittest
import numpy as np
import networkx as nx
import multiprocessing as mp
import itertools
import dataprep.link_failures as lf
import dataprep.sp_prep as sprep
from topos.fattree import make_topo
import json


class LinkFailureDataTest(unittest.TestCase):
    def setUp(self) -> None:
        """
        Graph in this class looks like this:

        1 -- 2 -- 3
             |    |
             5 -- 4
        Returns:

        """
        self.graph = nx.Graph()
        self.graph.add_edge(1, 2)
        self.graph.add_edge(2, 3)
        self.graph.add_edge(3, 4)
        self.graph.add_edge(2, 5)
        self.graph.add_edge(5, 4)
        sprep.add_index_to_nodes(self.graph)

        self.spfs = [
            [[1, 2]],
            [[1, 2, 3]],
            [[1, 2, 3, 4], [1, 2, 5, 4]],
            [[1, 2, 5]],
            [[2, 1]],
            [[2, 3]],
            [[2, 3, 4], [2, 5, 4]],
            [[2, 5]],
            [[3, 2, 1]],
            [[3, 2]],
            [[3, 4]],
            [[3, 2, 5], [3, 4, 5]],
            [[4, 3, 2, 1], [4, 5, 2, 1]],
            [[4, 3, 2], [4, 5, 2]],
            [[4, 3]],
            [[4, 5]],
            [[5, 2, 1]],
            [[5, 2]],
            [[5, 2, 3], [5, 4, 3]],
            [[5, 4]]
        ]
        self.edge_to_pairs = {
            (1, 2): [(1, 2), (1, 3), (1, 4), (1, 5)],
            (2, 1): [(2, 1), (3, 1), (4, 1), (5, 1)],
            (2, 3): [(1, 3), (1, 4), (2, 3), (2, 4), (5, 3)],
            (3, 2): [(3, 1), (4, 1), (3, 2), (4, 2), (3, 5)],
            (2, 5): [(1, 5), (2, 5), (3, 5), (1, 4), (2, 4)],
            (5, 2): [(5, 1), (5, 2), (5, 3), (4, 1), (4, 2)],
            (3, 4): [(1, 4), (2, 4), (3, 4), (3, 5)],
            (4, 3): [(4, 1), (4, 2), (4, 3), (5, 3)],
            (4, 5): [(4, 1), (4, 2), (4, 5), (3, 5)],
            (5, 4): [(1, 4), (2, 4), (5, 4), (5, 3)]
        }

    def test_spf_calculation(self):
        hosts = list(self.graph.nodes())
        pool = mp.Pool()
        paths_r = pool.map(lf._calc_spf, [{'src': s, 'dst': t, 'graph': self.graph}
                                     for s, t in itertools.product(hosts, hosts) if s != t])
        pool.close()
        self.assertEqual(len(paths_r), len(self.spfs))
        contained = True
        for paths in self.spfs:
            for path in paths:
                for paths_p in paths_r:
                    for path_p in paths_p:
                        contained = True
                        for h in path:
                            contained = contained and (h in path_p)
                        if contained:
                            break
                    if contained:
                        break
                self.assertTrue(contained, "Path {} not found in {}".format(list(path), str(paths_r)))

    def test_dict_creation(self):
        pool = mp.Pool()
        pairs = pool.map(lf._split_path, self.spfs)
        pool.close()
        ret = lf._reduce_pairs(pairs)
        self.assertEqual(len(self.edge_to_pairs), len(ret))
        for e in self.edge_to_pairs:
            self.assertEqual(
                len(self.edge_to_pairs[e]),
                len(ret[e]),
                'pairs for edge {} is {}, expected {}'.format(e, self.edge_to_pairs[e], ret[e])
            )
            for p in self.edge_to_pairs[e]:
                self.assertIn(p, ret[e], "{} not found in {} for edge {}".format(p, ret[e], e))

    def test_process_failed_edge(self):
        for n, d in self.graph.nodes(data=True):
            print(n, d)
        num_outputs = sprep._calc_num_outputs(self.graph)
        value_index = sprep._neighbor_to_index(self.graph)
        adj = sprep._make_distance_dict(self.graph)
        edges_to_pairs = lf._make_edge_to_pairs_dict(self.graph)

        states, targets, destinations, locs = lf._process_failed_links({
            "links": [(5, 4)],
            "graph": self.graph,
            'edges_to_pairs': edges_to_pairs,
            'num_outputs': num_outputs,
            'value_index': value_index,
            'adj': adj,
            'num_no_leaves': 4,
            'seed': 1
        })
        self.assertTrue(self.graph.has_edge(1, 2), 'edge 1->2 is missing')
        self.assertTrue(self.graph.has_edge(2, 1), 'edge 2->1 is missing')
        self.assertTrue(self.graph.has_edge(2, 3), 'edge 2->3 is missing')
        self.assertTrue(self.graph.has_edge(3, 2), 'edge 3->2 is missing')
        self.assertTrue(self.graph.has_edge(3, 4), 'edge 3->4 is missing')
        self.assertTrue(self.graph.has_edge(4, 3), 'edge 4->3 is missing')
        self.assertTrue(self.graph.has_edge(2, 5), 'edge 2->5 is missing')
        self.assertTrue(self.graph.has_edge(5, 2), 'edge 5->2 is missing')
        self.assertTrue(self.graph.has_edge(5, 4), 'edge 5->4 is missing')
        self.assertTrue(self.graph.has_edge(4, 5), 'edge 4->5 is missing')

        self.assertEqual(states.ndim, 4, "Expected 4 got {} dimensions".format(states.ndim))
        self.assertEqual(1, states.shape[0], "Expected dim0 of 1 got {}.".format(states.shape[0]))
        self.assertEqual(4, states.shape[1], "Expected dim1 of 4 got {}.".format(states.shape[1]))
        self.assertEqual(3, states.shape[2], "Expected dim2 of 3 got {}.".format(states.shape[2]))
        self.assertEqual(4, states.shape[3], "Expected dim3 of 4 got {}.".format(states.shape[3]))

        self.assertEqual(targets.ndim, 2, "Expected 2 got {} dimensions".format(targets.ndim))
        self.assertEqual(targets.shape[0], 1, "Expected dim0 of 1 got {}.".format(targets.shape[0]))
        self.assertEqual(targets.shape[1], 4, "Expected dim0 of 4 got {}.".format(targets.shape[1]))

        self.assertEqual(destinations.ndim, 2, "Expected 2 got {} dimensions".format(destinations.ndim))
        self.assertEqual(destinations.shape[0], 1, "Expected dim0 of 1 got {}.".format(destinations.shape[0]))
        self.assertEqual(destinations.shape[1], 1, "Expected dim0 of 4 got {}.".format(destinations.shape[1]))

        self.assertEqual(locs.ndim, 2, "Expected 2 got {} dimensions".format(locs.ndim))
        self.assertEqual(locs.shape[0], 1, "Expected dim0 of 1 got {}.".format(locs.shape[0]))
        self.assertEqual(locs.shape[1], 1, "Expected dim0 of 4 got {}.".format(locs.shape[0]))

        print(states[0])
        # The state for the first node is no longer contained, since it has
        # a degree of one and is thus excluded from the state.
        # self.assertEqual(states[0, 0, 0, 0], 1, str(states[0, 0]))
        # self.assertEqual(states[0, 0, 0, 1], 0, str(states[0, 0]))
        # self.assertEqual(states[0, 0, 0, 2], 1, str(states[0, 0]))
        # self.assertEqual(states[0, 0, 0, 3], 0, str(states[0, 0]))
        # self.assertEqual(states[0, 0, 1, 0], 0, str(states[0, 0]))
        # self.assertEqual(states[0, 0, 1, 1], 0, str(states[0, 0]))
        # self.assertEqual(states[0, 0, 1, 2], 0, str(states[0, 0]))
        # self.assertEqual(states[0, 0, 1, 3], 0, str(states[0, 0]))
        # self.assertEqual(states[0, 0, 2, 0], 0, str(states[0, 0]))
        # self.assertEqual(states[0, 0, 2, 1], 0, str(states[0, 0]))
        # self.assertEqual(states[0, 0, 2, 2], 0, str(states[0, 0]))
        # self.assertEqual(states[0, 0, 2, 3], 0, str(states[0, 0]))

        self.assertEqual(states[0, 0, 0, 0], 1, str(states[0, 0]))
        self.assertEqual(states[0, 0, 0, 1], 0, str(states[0, 0]))
        self.assertEqual(states[0, 0, 0, 2], 1, str(states[0, 0]))
        self.assertEqual(states[0, 0, 0, 3], 0, str(states[0, 0]))
        self.assertEqual(states[0, 0, 1, 0], 1, str(states[0, 0]))
        self.assertEqual(states[0, 0, 1, 1], 0, str(states[0, 0]))
        self.assertEqual(states[0, 0, 1, 2], 1, str(states[0, 0]))
        self.assertEqual(states[0, 0, 1, 3], 0, str(states[0, 0]))
        self.assertEqual(states[0, 0, 2, 0], 1, str(states[0, 0]))
        self.assertEqual(states[0, 0, 2, 1], 0, str(states[0, 0]))
        self.assertEqual(states[0, 0, 2, 2], 1, str(states[0, 0]))
        self.assertEqual(states[0, 0, 2, 3], 0, str(states[0, 0]))

        self.assertEqual(states[0, 1, 0, 0], 1, str(states[0, 1]))
        self.assertEqual(states[0, 1, 0, 1], 0, str(states[0, 1]))
        self.assertEqual(states[0, 1, 0, 2], 1, str(states[0, 1]))
        self.assertEqual(states[0, 1, 0, 3], 0, str(states[0, 1]))
        self.assertEqual(states[0, 1, 1, 0], 1, str(states[0, 1]))
        self.assertEqual(states[0, 1, 1, 1], 0, str(states[0, 1]))
        self.assertEqual(states[0, 1, 1, 2], 1, str(states[0, 1]))
        self.assertEqual(states[0, 1, 1, 3], 0, str(states[0, 1]))
        self.assertEqual(states[0, 1, 2, 0], 0, str(states[0, 1]))
        self.assertEqual(states[0, 1, 2, 1], 0, str(states[0, 1]))
        self.assertEqual(states[0, 1, 2, 2], 0, str(states[0, 1]))
        self.assertEqual(states[0, 1, 2, 3], 0, str(states[0, 1]))

        self.assertEqual(states[0, 2, 0, 0], 1, str(states[0, 2]))
        self.assertEqual(states[0, 2, 0, 1], 0, str(states[0, 2]))
        self.assertEqual(states[0, 2, 0, 2], 1, str(states[0, 2]))
        self.assertEqual(states[0, 2, 0, 3], 0, str(states[0, 2]))
        self.assertEqual(states[0, 2, 1, 0], 0, str(states[0, 2]))
        self.assertEqual(states[0, 2, 1, 1], 1, str(states[0, 2]))
        self.assertEqual(states[0, 2, 1, 2], 0, str(states[0, 2]))
        self.assertEqual(states[0, 2, 1, 3], 1, str(states[0, 2]))
        self.assertEqual(states[0, 2, 2, 0], 0, str(states[0, 2]))
        self.assertEqual(states[0, 2, 2, 1], 0, str(states[0, 2]))
        self.assertEqual(states[0, 2, 2, 2], 0, str(states[0, 2]))
        self.assertEqual(states[0, 2, 2, 3], 0, str(states[0, 2]))

        self.assertEqual(states[0, 3, 0, 0], 1, str(states[0, 3]))
        self.assertEqual(states[0, 3, 0, 1], 0, str(states[0, 3]))
        self.assertEqual(states[0, 3, 0, 2], 1, str(states[0, 3]))
        self.assertEqual(states[0, 3, 0, 3], 0, str(states[0, 3]))
        self.assertEqual(states[0, 3, 1, 0], 0, str(states[0, 3]))
        self.assertEqual(states[0, 3, 1, 1], 1, str(states[0, 3]))
        self.assertEqual(states[0, 3, 1, 2], 0, str(states[0, 3]))
        self.assertEqual(states[0, 3, 1, 3], 1, str(states[0, 3]))
        self.assertEqual(states[0, 3, 2, 0], 0, str(states[0, 3]))
        self.assertEqual(states[0, 3, 2, 1], 0, str(states[0, 3]))
        self.assertEqual(states[0, 3, 2, 2], 0, str(states[0, 3]))
        self.assertEqual(states[0, 3, 2, 3], 0, str(states[0, 3]))

    def test_create_dataset(self):
        dset = lf.get_link_failure_dataset(self.graph, 3, 1, 100)


class TestFatTree(unittest.TestCase):
    """
          Core                Core                Core                Core

      Agg      Agg        Agg      Agg        Agg      Agg        Agg      Agg
       |        |          |        |          |        |          |        |
      ToR      ToR        ToR      ToR        ToR      ToR        ToR      ToR
      / \      /  \       /  \     /  \       /  \     / \        / \      / \
    H00 H01  H02 H03    H04 H05  H06 H07    H08 H09  H10 H11    H12 H13  H14 H15
    """
    def setUp(self) -> None:
        self.fat_tree = make_topo(4)
        sprep.add_index_to_nodes(self.fat_tree)
        self.possible_cur_locs = [
            'tor-0000',
            'tor-0001',
            'tor-0002',
            'tor-0003',
            'tor-0004',
            'tor-0005',
            'tor-0006',
            'tor-0007',
            'agg-0000',
            'agg-0001',
            'agg-0002',
            'agg-0003',
            'agg-0004',
            'agg-0005',
            'agg-0006',
            'agg-0007',
            'core-0000',
            'core-0001',
            'core-0002',
            'core-0003'
        ]
        self.possible_dsts = [
            'tor-0000',
            'tor-0001',
            'tor-0002',
            'tor-0003',
            'tor-0004',
            'tor-0005',
            'tor-0006',
            'tor-0007',
            'agg-0000',
            'agg-0001',
            'agg-0002',
            'agg-0003',
            'agg-0004',
            'agg-0005',
            'agg-0006',
            'agg-0007',
            'core-0000',
            'core-0001',
            'core-0002',
            'core-0003',
            'h-0000',
            'h-0001',
            'h-0002',
            'h-0003',
            'h-0004',
            'h-0005',
            'h-0006',
            'h-0007',
            'h-0008',
            'h-0009',
            'h-0010',
            'h-0011',
            'h-0012',
            'h-0013',
            'h-0014',
            'h-0015'
        ]

    def assertArrayAlmostEqual(self, first: np.array, second: np.array, places=6, msg=None):
        self.assertEqual(first.size, second.size, 'Arrays do not have the same amount of elements')
        self.assertEqual(first.ndim, second.ndim, 'Arrays to not have the same amount of dimensions')
        self.assertEqual(1, first.ndim, 'Arrays have more than one dimension')
        if msg is None:
            msg = 'Expected array {}, actual is {}'.format(
                json.dumps(first.tolist()),
                json.dumps(second.tolist())
            )
        for i in range(first.size):
            self.assertAlmostEqual(
                first=first[i],
                second=second[i],
                places=places,
                msg=msg
            )

    def test_get_pod(self):
        p1 = lf._get_pod('h-0000', 4)
        p2 = lf._get_pod('h-0004', 4)
        self.assertNotEqual(p1, p2)
        self.assertEqual(p1, 0)
        self.assertEqual(p2, 1)

        p1 = lf._get_pod('tor-0000', 4)
        p2 = lf._get_pod('tor-0002', 4)
        self.assertNotEqual(p1, p2)
        self.assertEqual(p1, 0)
        self.assertEqual(p2, 1)

    def test_calc_spfs(self):
        paths = lf._calc_spfs({
            'srcs': ['tor-0000'],
            'dsts': ['h-0010'],
            'graph': self.fat_tree
        })
        self.assertEqual(first=4, second=len(paths))

    def test_split_paths(self):
        paths = [
            ['tor-0000', 'agg-0000', 'core-0000'],
            ['tor-0001', 'agg-0001', 'core-0003', 'agg-0001', 'tor-0002', 'h-0003']
        ]
        pairs = lf._split_path(paths)
        self.assertEqual(
            first=7,
            second=len(pairs),
            msg='Expected 7 pairs, got {} instead.'.format(len(pairs))
        )
        self.assertIn(
            member=(('tor-0000', 'agg-0000'), ('tor-0000', 'core-0000')),
            container=pairs
        )
        self.assertIn(
            member=(('agg-0000', 'core-0000'), ('tor-0000', 'core-0000')),
            container=pairs
        )
        self.assertIn(
            member=(('tor-0001', 'agg-0001'), ('tor-0001', 'h-0003')),
            container=pairs
        )
        self.assertIn(
            member=(('agg-0001', 'core-0003'), ('tor-0001', 'h-0003')),
            container=pairs
        )
        self.assertIn(
            member=(('core-0003', 'agg-0001'), ('tor-0001', 'h-0003')),
            container=pairs
        )
        self.assertIn(
            member=(('agg-0001', 'tor-0002'), ('tor-0001', 'h-0003')),
            container=pairs
        )
        self.assertIn(
            member=(('tor-0002', 'h-0003'), ('tor-0001', 'h-0003')),
            container=pairs
        )

    def test_get_incident(self):
        cores = ['core-0000', 'core-0001', 'core-0002', 'core-0003']
        aggs = ['agg-0000', 'agg-0001', 'agg-0002', 'agg-0003', 'agg-0004', 'agg-0005', 'agg-0006', 'agg-0007']
        for c, a in itertools.product(cores, aggs):
            self.assertEqual(
                first=lf._get_incident(c, a, 4),
                second=self.fat_tree.has_edge(c, a),
                msg='incidence wrong, {} - {} is predicted to be {} but is {}'.format(
                    c,
                    a,
                    lf._get_incident(c, a, 4),
                    self.fat_tree.has_edge(c, a)
                )
            )

    def test_make_edges_to_pairs(self):
        pairs = lf._make_edge_to_pairs_dict(self.fat_tree, 4)
        num_pairs = self.fat_tree.number_of_edges() - 16
        self.assertTrue(self.fat_tree.is_directed(), 'expected grapht obe directed.')
        self.assertEqual(
            first=num_pairs,
            second=len(pairs),
            msg='Expected {} edge_to_pairs entries, got {} instead'.format(
                len(self.possible_cur_locs) * len(self.possible_dsts),
                len(pairs)
            )
        )
        num_pairs = 1 + 2 + 3 + 7 + 14 # pairs --> tor dict
        num_pairs += 2 + 1 + 6  # pairs --> agg dict
        num_pairs += 4 + 2  # pairs --> core dict
        self.assertEqual(
            first=num_pairs,
            second=pairs[('tor-0000', 'agg-0000')].get_num_pairs(),
            msg='expected edge to carry paths of {} pairs, got {} instead.'.format(
                num_pairs,
                pairs[('tor-0000', 'agg-0000')].get_num_pairs(),
            )
        )

    def test_sampling(self):
        pairs = lf._make_edge_to_pairs_dict(self.fat_tree, 4)
        random = np.random.RandomState(seed=1)
        pair = pairs[('tor-0000', 'agg-0000')].sample_pair(random)
        print(pair)

    def test_to_dict(self):
        pairs = lf._make_edge_to_pairs_dict(self.fat_tree, 4)
        d = pairs[('tor-0000', 'agg-0000')].to_dict()
        pair = lf.EdgeToPair.from_dict(d)
        a = ''

    def test_neighbor_to_index(self):
        """
        Tests the correct order of neighbors for the fat-tree. The first half
        of the entries should send down, the second half should send up for tor
        and aggregation.

        Returns:

        """
        neighbor_to_idx = sprep._neighbor_to_index(self.fat_tree)
        self.assertEqual(
            first=0,
            second=neighbor_to_idx['h-0000']['tor-0000'],
            msg='Expected tor-0000 to be at idx 0, got {} instead'.format(
                neighbor_to_idx['h-0000']['tor-0000']
            )
        )

        self.assertEqual(
            first=2,
            second=neighbor_to_idx['tor-0000']['agg-0000'],
            msg='Expected agg-0000 to be at idx 2, got {} instead'.format(
                neighbor_to_idx['tor-0000']['agg-0000']
            )
        )
        self.assertEqual(
            first=3,
            second=neighbor_to_idx['tor-0000']['agg-0001'],
            msg='Expected agg-0001 to be at idx 3, got {} instead'.format(
                neighbor_to_idx['tor-0000']['agg-0001']
            )
        )
        self.assertEqual(
            first=0,
            second=neighbor_to_idx['tor-0000']['h-0000'],
            msg='Expected h-0000 to be at idx 0, got {} instead'.format(
                neighbor_to_idx['tor-0000']['h-0000']
            )
        )
        self.assertEqual(
            first=1,
            second=neighbor_to_idx['tor-0000']['h-0001'],
            msg='Expected h-0001 to be at idx 1, got {} instead'.format(
                neighbor_to_idx['tor-0000']['h-0001']
            )
        )

        self.assertEqual(
            first=2,
            second=neighbor_to_idx['agg-0000']['core-0000'],
            msg='Expected core-0000 to be at idx 2, got {} instead'.format(
                neighbor_to_idx['agg-0000']['core-0000']
            )
        )
        self.assertEqual(
            first=3,
            second=neighbor_to_idx['agg-0000']['core-0001'],
            msg='Expected core-0001 to be at idx 3, got {} instead'.format(
                neighbor_to_idx['agg-0000']['core-0001']
            )
        )
        self.assertEqual(
            first=0,
            second=neighbor_to_idx['agg-0000']['tor-0000'],
            msg='Expected tor-0000 to be at idx 0, got {} instead'.format(
                neighbor_to_idx['agg-0000']['tor-0000']
            )
        )
        self.assertEqual(
            first=1,
            second=neighbor_to_idx['agg-0000']['tor-0001'],
            msg='Expected tor-0001 to be at idx 1, got {} instead'.format(
                neighbor_to_idx['agg-0000']['tor-0001']
            )
        )

        self.assertEqual(
            first=0,
            second=neighbor_to_idx['core-0000']['agg-0000'],
            msg='Expected agg-0000 to be at idx 0, got {} instead'.format(
                neighbor_to_idx['core-0000']['agg-0000']
            )
        )
        self.assertEqual(
            first=1,
            second=neighbor_to_idx['core-0000']['agg-0002'],
            msg='Expected agg-0002 to be at idx 1, got {} instead'.format(
                neighbor_to_idx['core-0000']['agg-0002']
            )
        )
        self.assertEqual(
            first=2,
            second=neighbor_to_idx['core-0000']['agg-0004'],
            msg='Expected agg-0004 to be at idx 2, got {} instead'.format(
                neighbor_to_idx['core-0000']['agg-0004']
            )
        )
        self.assertEqual(
            first=3,
            second=neighbor_to_idx['core-0000']['agg-0006'],
            msg='Expected agg-0006 to be at idx 3, got {} instead'.format(
                neighbor_to_idx['core-0000']['agg-0006']
            )
        )

    def test_reduced_network_state_creation(self):
        graph = self.fat_tree.to_undirected()
        neighbor_to_idx = sprep._neighbor_to_index(graph)
        failed_links = [('tor-0000', 'agg-0000'), ('core-0000', 'agg-0002')]
        num_non_leaves = int(4**2 / 4 + 4 ** 2)
        graph.remove_edges_from(failed_links)
        state = lf._get_reduced_network_state(
            graph=graph,
            failed_links=failed_links,
            num_non_leaves=num_non_leaves,
            max_degree=4,
            neighbor_to_idx=neighbor_to_idx
        )
        self.assertListEqual(
            list1=list(state.shape),
            list2=[1, num_non_leaves, 4, 4]
        )
        tmp = np.squeeze(np.sum(np.sum(state, axis=2), axis=1), axis=0)
        self.assertEqual(
            first=(num_non_leaves - 4) * 4 + 3 * 4,
            second=tmp[0],
            msg="Expected {} outgoing edges indicated to be up, got {} instead".format(
                (num_non_leaves - 4) * 4 + 3 * 4,
                tmp[0]
            )
        )
        self.assertEqual(
            first=(num_non_leaves - 4) * 4 + 3 * 4,
            second=tmp[2],
            msg="Expected {} incoming edges indicated to be up, got {} instead".format(
                (num_non_leaves - 4) * 4 + 3 * 4,
                tmp[2]
            )
        )
        self.assertEqual(
            first=4,
            second=tmp[1],
            msg="Expected {} outgoing edges indicated to be down, got {} instead".format(
                4,
                tmp[1]
            )
        )
        self.assertEqual(
            first=4,
            second=tmp[3],
            msg="Expected {} incoming edges indicated to be down, got {} instead".format(
                4,
                tmp[3]
            )
        )

    def test_get_output(self):
        graph = self.fat_tree.to_undirected()
        neighbor_to_idx = sprep._neighbor_to_index(graph)
        # print(json.dumps(neighbor_to_idx['tor-0006'], indent=1))
        # print(json.dumps(neighbor_to_idx['agg-0007'], indent=1))
        adj = sprep._make_distance_dict(graph)

        failed_links = [('tor-0000', 'agg-0000'), ('tor-0004', 'h-0009'), ('core-0002', 'agg-0003'),
                        ('core-0003', 'agg-0003')]
        graph.remove_edges_from(failed_links)

        backup = lf._update_adjacencies_ligth(graph, 'tor-0000', 'h-0008', adj)
        output = lf._get_output(graph, 'tor-0000', 'h-0008', 4, adj, neighbor_to_idx)
        self.assertArrayAlmostEqual(
            first=np.array([0., 0., 0., 0., 1.]),
            second=output.flatten()
        )
        lf._restore_changes(adj, backup)

        backup = lf._update_adjacencies_ligth(graph, 'agg-0000', 'h-0008', adj)
        output = lf._get_output(graph, 'agg-0000', 'h-0008', 4, adj, neighbor_to_idx)
        self.assertArrayAlmostEqual(
            first=np.array([0., 0., 0., 0.5, 0.5]),
            second=output.flatten()
        )
        lf._restore_changes(adj, backup)

        backup = lf._update_adjacencies_ligth(graph, 'agg-0004', 'h-0008', adj)
        output = lf._get_output(graph, 'agg-0004', 'h-0008', 4, adj, neighbor_to_idx)
        self.assertArrayAlmostEqual(
            first=np.array([0., 1., 0., 0., 0.]),
            second=output.flatten()
        )
        lf._restore_changes(adj, backup)

        backup = lf._update_adjacencies_ligth(graph, 'agg-0000', 'h-0009', adj)
        output = lf._get_output(graph, 'agg-0000', 'h-0009', 4, adj, neighbor_to_idx)
        self.assertArrayAlmostEqual(
            first=np.array([1., 0., 0., 0., 0.]),
            second=output.flatten()
        )
        lf._restore_changes(adj, backup)

        backup = lf._update_adjacencies_ligth(graph, 'tor-0006', 'h-0006', adj)
        output = lf._get_output(graph, 'tor-0006', 'h-0006', 4, adj, neighbor_to_idx)
        self.assertArrayAlmostEqual(
            first=np.array([0., 0., 0., 1., 0.]),
            second=output.flatten()
        )
        lf._restore_changes(adj, backup)

    def test_link_failure_to_sample(self):
        graph = self.fat_tree.to_undirected()
        adj = sprep._make_distance_dict(graph)
        neighbor_to_idx = sprep._neighbor_to_index(graph)
        num_non_leaves = int(4**2 / 4 + 4 ** 2)

        failed_links = [('tor-0000', 'agg-0000'), ('core-0000', 'agg-0002')]
        graph.remove_edges_from(failed_links)

        state, target, destination, cur_loc = lf._link_failure_to_samples(
            graph=graph,
            links=failed_links,
            num_outputs=4,
            value_index=neighbor_to_idx,
            adj=adj,
            cur_loc='tor-0000',
            destination='h-0009',
            num_non_leaves=num_non_leaves
        )
        tmp = np.squeeze(np.sum(np.sum(state, axis=2), axis=1), axis=0)
        self.assertListEqual(
            list1=[(num_non_leaves - 4) * 4 + 3 * 4, 4, (num_non_leaves - 4) * 4 + 3 * 4, 4],
            list2=tmp.tolist()
        )
        self.assertArrayAlmostEqual(
            first=np.array([0., 0., 0., 0., 1.]),
            second=target.flatten()
        )

    def test_link_failure_to_sample_extensive(self):
        graph = make_topo(8)
        graph = sprep.add_index_to_nodes(graph)
        neighbor_to_idx = sprep._neighbor_to_index(graph)
        adj = sprep._make_distance_dict(graph)
        num_non_leaves = lf._get_num_no_leaves(graph)
        hosts = ['h-{:04d}'.format(i) for i in range(0, 128)]

        print(neighbor_to_idx['agg-0002'])
        print(neighbor_to_idx['core-0000'])

        failed_links = [('tor-0000', 'agg-0000'), ('core-0000', 'agg-0004')]
        graph.remove_edges_from(failed_links)

        for h in hosts:
            state, target, destination, cur_loc = lf._link_failure_to_samples(
                graph=graph,
                links=failed_links,
                num_outputs=8,
                value_index=neighbor_to_idx,
                adj=adj,
                cur_loc='agg-0000',
                destination=h,
                num_non_leaves=num_non_leaves
            )
            target[target > 0] = 1
            print(target)

    def test_sample_link_failure_fat_tree(self):
        random = np.random.RandomState(seed=1)
        lengths = {i: 0 for i in range(11)}
        edges = {(u, v): 0 for u, v in self.fat_tree.edges()}
        for i in range(1000):
            failures = lf._sample_failure_fat_tree(random, 10, 4, self.fat_tree)
            lengths[len(failures)] += 1
            self.assertLessEqual(0, len(failures))
            self.assertGreaterEqual(10, len(failures))
            for u, v in failures:
                self.assertFalse(u.startswith('h'), 'Got unexpected pari {}'.format(json.dumps((u, v))))
                self.assertTrue(self.fat_tree.has_node(u))
                self.assertTrue(self.fat_tree.has_node(v))
                self.assertTrue(self.fat_tree.has_edge(u, v))
                edges[(u, v)] += 1
        print(json.dumps(lengths, indent=1))
        print(json.dumps({str(e): v for e, v in edges.items()}, indent=1))

    def test_pair_sampling(self):
        random = np.random.RandomState(seed=1)
        edge_to_pairs = lf._make_edge_to_pairs_dict(self.fat_tree, 4)
        destinations = {n: 0 for n in self.fat_tree.nodes()}
        cur_locs = {n: 0 for n in self.fat_tree.nodes()}
        for i in range(10000):
            failures = lf._sample_failure_fat_tree(random, 10, 4, self.fat_tree)
            ret = lf._sample_pair(edge_to_pairs, failures, random)
            for cur_loc, dst in ret:
                destinations[dst] += 1
                cur_locs[cur_loc] += 1
        for k, v in cur_locs.items():
            if k.startswith('h'):
                self.assertEqual(
                    first=0,
                    second=v,
                    msg='Host {} should never be a current location, but is {} times'.format(k, v)
                )
        print(json.dumps(destinations, indent=1))
        print(json.dumps(cur_locs, indent=1))

    def test_process_failed_links_consistency(self):
        random = np.random.RandomState(seed=1)
        graph = make_topo(8).to_undirected()
        sprep.add_index_to_nodes(graph)
        num_outputs = sprep._calc_num_outputs(graph)
        value_index = sprep._neighbor_to_index(graph)
        print(value_index['tor-0022'])
        print(value_index['agg-0022'])
        adj = sprep._make_distance_dict(graph)
        adj_backup = sprep._make_distance_dict(graph)
        num_no_leaves = lf._get_num_no_leaves(graph)
        edges_to_pairs = lf._make_edge_to_pairs_dict(graph, 4)

        for i in range(100):
            lf._process_failed_links({
                'links': lf._sample_failure_fat_tree(random, 10, 4, graph),
                'graph': graph,
                'edges_to_pairs': edges_to_pairs,
                'num_outputs': num_outputs,
                'value_index': value_index,
                # "deep" copy of dictionary. Must be in this way, since references
                    # are not copied when calling copy on the outer dictionary.
                'adj': adj, # {d: k.copy() for d, k in adj.items()},
                'num_no_leaves': num_no_leaves,
                'seed': int(random.randint(1, int(2**32-1)))
            })
            for u, v in self.fat_tree.to_undirected().edges():
                self.assertTrue(
                    graph.has_edge(u, v),
                    msg="Graph is missing edge ({}, {}) in iter {}".format(u, v, i)
                )

            for k, row in adj.items():
                for kk in row.keys():
                    self.assertAlmostEqual(
                        adj[k][kk],
                        adj_backup[k][kk],
                        msg="Adjacency matrix missmatch for {} {} in iter {}".format(
                            k,
                            kk,
                            i
                        ))

    def test_running_full_stack(self):
        num_non_leaves = int(4**2 / 4 + 4 ** 2)
        dataset = lf.get_link_failure_dataset(
            graph=self.fat_tree.to_undirected(),
            max_num_failures=10,
            seed=1,
            num_samples=100,
            k=4
        )
        for k, v in dataset.items():
            print(k, v.shape)
        self.assertListEqual(
            list1=list(dataset['network_states'].shape),
            list2=[100, num_non_leaves, 4, 4]
        )
        self.assertListEqual(
            list1=list(dataset['targets'].shape),
            list2=[100, 5]
        )
        self.assertListEqual(
            list1=list(dataset['destinations'].shape),
            list2=[100, 1]
        )
        self.assertListEqual(
            list1=list(dataset['cur_locs'].shape),
            list2=[100, 1]
        )
        self.assertListEqual(
            list1=list(dataset['all_masks'].shape),
            list2=[self.fat_tree.number_of_nodes(), 4, 1]
        )
        self.assertListEqual(
            list1=list(dataset['all_neighbors'].shape),
            list2=[self.fat_tree.number_of_nodes(), 4, 1]
        )
        self.assertListEqual(
            list1=list(dataset['nodes_with_state'].shape),
            list2=[num_non_leaves]
        )

    def test_hosts_under_same_pod(self):
        self.assertTrue(lf._are_hosts_under_same_pod(0, 1, 8))
        self.assertTrue(lf._are_hosts_under_same_pod(0, 2, 8))
        self.assertTrue(lf._are_hosts_under_same_pod(0, 3, 8))
        self.assertFalse(lf._are_hosts_under_same_pod(0, 5, 8))
        self.assertFalse(lf._are_hosts_under_same_pod(0, 8, 8))
        self.assertFalse(lf._are_hosts_under_same_pod(0, 12, 8))
        self.assertFalse(lf._are_hosts_under_same_pod(0, 127, 8))
        self.assertFalse(lf._are_hosts_under_same_pod(0, 58, 8))

    def test_sample_pairs_from_paths(self):
        tors = [n for n in self.fat_tree.nodes() if n.startswith('tor-')]
        hosts = [n for n in self.fat_tree.nodes() if n.startswith('h-')]
        pairs = [(tor, host) for tor, host in itertools.product(tors, hosts) \
                     if not self.fat_tree.has_edge(tor, host)]
        edge_to_paths = lf._make_edge_to_paths(self.fat_tree, pairs)
        pairs = lf._sample_pairs_from_paths(edge_to_paths, [('tor-0000', 'agg-0000')], np.random.RandomState(seed=1))
        print(pairs)

    def test_crete_all_pairs_of_hosts_not_same_tor(self):
        pairs = lf._create_all_pairs_of_hosts_not_same_tor(4)
        pairs_test = [
            ('h-0000', 'h-0002'),
            ('h-0000', 'h-0003'),
            ('h-0000', 'h-0004'),
            ('h-0000', 'h-0005'),
            ('h-0000', 'h-0006'),
            ('h-0000', 'h-0007'),
            ('h-0000', 'h-0008'),
            ('h-0000', 'h-0009'),
            ('h-0000', 'h-0010'),
            ('h-0000', 'h-0011'),
            ('h-0000', 'h-0012'),
            ('h-0000', 'h-0013'),
            ('h-0000', 'h-0014'),
            ('h-0000', 'h-0015'),

            ('h-0001', 'h-0002'),
            ('h-0001', 'h-0003'),
            ('h-0001', 'h-0004'),
            ('h-0001', 'h-0005'),
            ('h-0001', 'h-0006'),
            ('h-0001', 'h-0007'),
            ('h-0001', 'h-0008'),
            ('h-0001', 'h-0009'),
            ('h-0001', 'h-0010'),
            ('h-0001', 'h-0011'),
            ('h-0001', 'h-0012'),
            ('h-0001', 'h-0013'),
            ('h-0001', 'h-0014'),
            ('h-0001', 'h-0015'),

            ('h-0002', 'h-0000'),
            ('h-0002', 'h-0001'),
            ('h-0002', 'h-0004'),
            ('h-0002', 'h-0005'),
            ('h-0002', 'h-0006'),
            ('h-0002', 'h-0007'),
            ('h-0002', 'h-0008'),
            ('h-0002', 'h-0009'),
            ('h-0002', 'h-0010'),
            ('h-0002', 'h-0011'),
            ('h-0002', 'h-0012'),
            ('h-0002', 'h-0013'),
            ('h-0002', 'h-0014'),
            ('h-0002', 'h-0015'),

            ('h-0003', 'h-0000'),
            ('h-0003', 'h-0001'),
            ('h-0003', 'h-0004'),
            ('h-0003', 'h-0005'),
            ('h-0003', 'h-0006'),
            ('h-0003', 'h-0007'),
            ('h-0003', 'h-0008'),
            ('h-0003', 'h-0009'),
            ('h-0003', 'h-0010'),
            ('h-0003', 'h-0011'),
            ('h-0003', 'h-0012'),
            ('h-0003', 'h-0013'),
            ('h-0003', 'h-0014'),
            ('h-0003', 'h-0015'),

            ('h-0004', 'h-0000'),
            ('h-0004', 'h-0001'),
            ('h-0004', 'h-0002'),
            ('h-0004', 'h-0003'),
            ('h-0004', 'h-0006'),
            ('h-0004', 'h-0007'),
            ('h-0004', 'h-0008'),
            ('h-0004', 'h-0009'),
            ('h-0004', 'h-0010'),
            ('h-0004', 'h-0011'),
            ('h-0004', 'h-0012'),
            ('h-0004', 'h-0013'),
            ('h-0004', 'h-0014'),
            ('h-0004', 'h-0015'),

            ('h-0005', 'h-0000'),
            ('h-0005', 'h-0001'),
            ('h-0005', 'h-0002'),
            ('h-0005', 'h-0003'),
            ('h-0005', 'h-0006'),
            ('h-0005', 'h-0007'),
            ('h-0005', 'h-0008'),
            ('h-0005', 'h-0009'),
            ('h-0005', 'h-0010'),
            ('h-0005', 'h-0011'),
            ('h-0005', 'h-0012'),
            ('h-0005', 'h-0013'),
            ('h-0005', 'h-0014'),
            ('h-0005', 'h-0015'),

            ('h-0006', 'h-0000'),
            ('h-0006', 'h-0001'),
            ('h-0006', 'h-0002'),
            ('h-0006', 'h-0003'),
            ('h-0006', 'h-0004'),
            ('h-0006', 'h-0005'),
            ('h-0006', 'h-0008'),
            ('h-0006', 'h-0009'),
            ('h-0006', 'h-0010'),
            ('h-0006', 'h-0011'),
            ('h-0006', 'h-0012'),
            ('h-0006', 'h-0013'),
            ('h-0006', 'h-0014'),
            ('h-0006', 'h-0015'),

            ('h-0007', 'h-0000'),
            ('h-0007', 'h-0001'),
            ('h-0007', 'h-0002'),
            ('h-0007', 'h-0003'),
            ('h-0007', 'h-0004'),
            ('h-0007', 'h-0005'),
            ('h-0007', 'h-0008'),
            ('h-0007', 'h-0009'),
            ('h-0007', 'h-0010'),
            ('h-0007', 'h-0011'),
            ('h-0007', 'h-0012'),
            ('h-0007', 'h-0013'),
            ('h-0007', 'h-0014'),
            ('h-0007', 'h-0015'),

            ('h-0008', 'h-0000'),
            ('h-0008', 'h-0001'),
            ('h-0008', 'h-0002'),
            ('h-0008', 'h-0003'),
            ('h-0008', 'h-0004'),
            ('h-0008', 'h-0005'),
            ('h-0008', 'h-0006'),
            ('h-0008', 'h-0007'),
            ('h-0008', 'h-0010'),
            ('h-0008', 'h-0011'),
            ('h-0008', 'h-0012'),
            ('h-0008', 'h-0013'),
            ('h-0008', 'h-0014'),
            ('h-0008', 'h-0015'),

            ('h-0009', 'h-0000'),
            ('h-0009', 'h-0001'),
            ('h-0009', 'h-0002'),
            ('h-0009', 'h-0003'),
            ('h-0009', 'h-0004'),
            ('h-0009', 'h-0005'),
            ('h-0009', 'h-0006'),
            ('h-0009', 'h-0007'),
            ('h-0009', 'h-0010'),
            ('h-0009', 'h-0011'),
            ('h-0009', 'h-0012'),
            ('h-0009', 'h-0013'),
            ('h-0009', 'h-0014'),
            ('h-0009', 'h-0015'),

            ('h-0010', 'h-0000'),
            ('h-0010', 'h-0001'),
            ('h-0010', 'h-0002'),
            ('h-0010', 'h-0003'),
            ('h-0010', 'h-0004'),
            ('h-0010', 'h-0005'),
            ('h-0010', 'h-0006'),
            ('h-0010', 'h-0007'),
            ('h-0010', 'h-0008'),
            ('h-0010', 'h-0009'),
            ('h-0010', 'h-0012'),
            ('h-0010', 'h-0013'),
            ('h-0010', 'h-0014'),
            ('h-0010', 'h-0015'),

            ('h-0011', 'h-0000'),
            ('h-0011', 'h-0001'),
            ('h-0011', 'h-0002'),
            ('h-0011', 'h-0003'),
            ('h-0011', 'h-0004'),
            ('h-0011', 'h-0005'),
            ('h-0011', 'h-0006'),
            ('h-0011', 'h-0007'),
            ('h-0011', 'h-0008'),
            ('h-0011', 'h-0009'),
            ('h-0011', 'h-0012'),
            ('h-0011', 'h-0013'),
            ('h-0011', 'h-0014'),
            ('h-0011', 'h-0015'),

            ('h-0012', 'h-0000'),
            ('h-0012', 'h-0001'),
            ('h-0012', 'h-0002'),
            ('h-0012', 'h-0003'),
            ('h-0012', 'h-0004'),
            ('h-0012', 'h-0005'),
            ('h-0012', 'h-0006'),
            ('h-0012', 'h-0007'),
            ('h-0012', 'h-0008'),
            ('h-0012', 'h-0009'),
            ('h-0012', 'h-0010'),
            ('h-0012', 'h-0011'),
            ('h-0012', 'h-0014'),
            ('h-0012', 'h-0015'),

            ('h-0013', 'h-0000'),
            ('h-0013', 'h-0001'),
            ('h-0013', 'h-0002'),
            ('h-0013', 'h-0003'),
            ('h-0013', 'h-0004'),
            ('h-0013', 'h-0005'),
            ('h-0013', 'h-0006'),
            ('h-0013', 'h-0007'),
            ('h-0013', 'h-0008'),
            ('h-0013', 'h-0009'),
            ('h-0013', 'h-0010'),
            ('h-0013', 'h-0011'),
            ('h-0013', 'h-0014'),
            ('h-0013', 'h-0015'),

            ('h-0014', 'h-0000'),
            ('h-0014', 'h-0001'),
            ('h-0014', 'h-0002'),
            ('h-0014', 'h-0003'),
            ('h-0014', 'h-0004'),
            ('h-0014', 'h-0005'),
            ('h-0014', 'h-0006'),
            ('h-0014', 'h-0007'),
            ('h-0014', 'h-0008'),
            ('h-0014', 'h-0009'),
            ('h-0014', 'h-0010'),
            ('h-0014', 'h-0011'),
            ('h-0014', 'h-0012'),
            ('h-0014', 'h-0013'),

            ('h-0015', 'h-0000'),
            ('h-0015', 'h-0001'),
            ('h-0015', 'h-0002'),
            ('h-0015', 'h-0003'),
            ('h-0015', 'h-0004'),
            ('h-0015', 'h-0005'),
            ('h-0015', 'h-0006'),
            ('h-0015', 'h-0007'),
            ('h-0015', 'h-0008'),
            ('h-0015', 'h-0009'),
            ('h-0015', 'h-0010'),
            ('h-0015', 'h-0011'),
            ('h-0015', 'h-0012'),
            ('h-0015', 'h-0013')
        ]

        self.assertEqual(
            first=len(pairs),
            second=len(pairs_test),
            msg="Expected {} pairs, got {} instead".format(len(pairs), len(pairs_test))
        )
        for pair in pairs:
            self.assertIn(
                pair,
                pairs_test,
                msg="Pair ({}, {}) not in test set.".format(pair[0], pair[1])
            )

    def test_sample_pair_of_hosts_not_same_tor(self):
        pairs_test = [
            ('h-0000', 'h-0002'),
            ('h-0000', 'h-0003'),
            ('h-0000', 'h-0004'),
            ('h-0000', 'h-0005'),
            ('h-0000', 'h-0006'),
            ('h-0000', 'h-0007'),
            ('h-0000', 'h-0008'),
            ('h-0000', 'h-0009'),
            ('h-0000', 'h-0010'),
            ('h-0000', 'h-0011'),
            ('h-0000', 'h-0012'),
            ('h-0000', 'h-0013'),
            ('h-0000', 'h-0014'),
            ('h-0000', 'h-0015'),

            ('h-0001', 'h-0002'),
            ('h-0001', 'h-0003'),
            ('h-0001', 'h-0004'),
            ('h-0001', 'h-0005'),
            ('h-0001', 'h-0006'),
            ('h-0001', 'h-0007'),
            ('h-0001', 'h-0008'),
            ('h-0001', 'h-0009'),
            ('h-0001', 'h-0010'),
            ('h-0001', 'h-0011'),
            ('h-0001', 'h-0012'),
            ('h-0001', 'h-0013'),
            ('h-0001', 'h-0014'),
            ('h-0001', 'h-0015'),

            ('h-0002', 'h-0000'),
            ('h-0002', 'h-0001'),
            ('h-0002', 'h-0004'),
            ('h-0002', 'h-0005'),
            ('h-0002', 'h-0006'),
            ('h-0002', 'h-0007'),
            ('h-0002', 'h-0008'),
            ('h-0002', 'h-0009'),
            ('h-0002', 'h-0010'),
            ('h-0002', 'h-0011'),
            ('h-0002', 'h-0012'),
            ('h-0002', 'h-0013'),
            ('h-0002', 'h-0014'),
            ('h-0002', 'h-0015'),

            ('h-0003', 'h-0000'),
            ('h-0003', 'h-0001'),
            ('h-0003', 'h-0004'),
            ('h-0003', 'h-0005'),
            ('h-0003', 'h-0006'),
            ('h-0003', 'h-0007'),
            ('h-0003', 'h-0008'),
            ('h-0003', 'h-0009'),
            ('h-0003', 'h-0010'),
            ('h-0003', 'h-0011'),
            ('h-0003', 'h-0012'),
            ('h-0003', 'h-0013'),
            ('h-0003', 'h-0014'),
            ('h-0003', 'h-0015'),

            ('h-0004', 'h-0000'),
            ('h-0004', 'h-0001'),
            ('h-0004', 'h-0002'),
            ('h-0004', 'h-0003'),
            ('h-0004', 'h-0006'),
            ('h-0004', 'h-0007'),
            ('h-0004', 'h-0008'),
            ('h-0004', 'h-0009'),
            ('h-0004', 'h-0010'),
            ('h-0004', 'h-0011'),
            ('h-0004', 'h-0012'),
            ('h-0004', 'h-0013'),
            ('h-0004', 'h-0014'),
            ('h-0004', 'h-0015'),

            ('h-0005', 'h-0000'),
            ('h-0005', 'h-0001'),
            ('h-0005', 'h-0002'),
            ('h-0005', 'h-0003'),
            ('h-0005', 'h-0006'),
            ('h-0005', 'h-0007'),
            ('h-0005', 'h-0008'),
            ('h-0005', 'h-0009'),
            ('h-0005', 'h-0010'),
            ('h-0005', 'h-0011'),
            ('h-0005', 'h-0012'),
            ('h-0005', 'h-0013'),
            ('h-0005', 'h-0014'),
            ('h-0005', 'h-0015'),

            ('h-0006', 'h-0000'),
            ('h-0006', 'h-0001'),
            ('h-0006', 'h-0002'),
            ('h-0006', 'h-0003'),
            ('h-0006', 'h-0004'),
            ('h-0006', 'h-0005'),
            ('h-0006', 'h-0008'),
            ('h-0006', 'h-0009'),
            ('h-0006', 'h-0010'),
            ('h-0006', 'h-0011'),
            ('h-0006', 'h-0012'),
            ('h-0006', 'h-0013'),
            ('h-0006', 'h-0014'),
            ('h-0006', 'h-0015'),

            ('h-0007', 'h-0000'),
            ('h-0007', 'h-0001'),
            ('h-0007', 'h-0002'),
            ('h-0007', 'h-0003'),
            ('h-0007', 'h-0004'),
            ('h-0007', 'h-0005'),
            ('h-0007', 'h-0008'),
            ('h-0007', 'h-0009'),
            ('h-0007', 'h-0010'),
            ('h-0007', 'h-0011'),
            ('h-0007', 'h-0012'),
            ('h-0007', 'h-0013'),
            ('h-0007', 'h-0014'),
            ('h-0007', 'h-0015'),

            ('h-0008', 'h-0000'),
            ('h-0008', 'h-0001'),
            ('h-0008', 'h-0002'),
            ('h-0008', 'h-0003'),
            ('h-0008', 'h-0004'),
            ('h-0008', 'h-0005'),
            ('h-0008', 'h-0006'),
            ('h-0008', 'h-0007'),
            ('h-0008', 'h-0010'),
            ('h-0008', 'h-0011'),
            ('h-0008', 'h-0012'),
            ('h-0008', 'h-0013'),
            ('h-0008', 'h-0014'),
            ('h-0008', 'h-0015'),

            ('h-0009', 'h-0000'),
            ('h-0009', 'h-0001'),
            ('h-0009', 'h-0002'),
            ('h-0009', 'h-0003'),
            ('h-0009', 'h-0004'),
            ('h-0009', 'h-0005'),
            ('h-0009', 'h-0006'),
            ('h-0009', 'h-0007'),
            ('h-0009', 'h-0010'),
            ('h-0009', 'h-0011'),
            ('h-0009', 'h-0012'),
            ('h-0009', 'h-0013'),
            ('h-0009', 'h-0014'),
            ('h-0009', 'h-0015'),

            ('h-0010', 'h-0000'),
            ('h-0010', 'h-0001'),
            ('h-0010', 'h-0002'),
            ('h-0010', 'h-0003'),
            ('h-0010', 'h-0004'),
            ('h-0010', 'h-0005'),
            ('h-0010', 'h-0006'),
            ('h-0010', 'h-0007'),
            ('h-0010', 'h-0008'),
            ('h-0010', 'h-0009'),
            ('h-0010', 'h-0012'),
            ('h-0010', 'h-0013'),
            ('h-0010', 'h-0014'),
            ('h-0010', 'h-0015'),

            ('h-0011', 'h-0000'),
            ('h-0011', 'h-0001'),
            ('h-0011', 'h-0002'),
            ('h-0011', 'h-0003'),
            ('h-0011', 'h-0004'),
            ('h-0011', 'h-0005'),
            ('h-0011', 'h-0006'),
            ('h-0011', 'h-0007'),
            ('h-0011', 'h-0008'),
            ('h-0011', 'h-0009'),
            ('h-0011', 'h-0012'),
            ('h-0011', 'h-0013'),
            ('h-0011', 'h-0014'),
            ('h-0011', 'h-0015'),

            ('h-0012', 'h-0000'),
            ('h-0012', 'h-0001'),
            ('h-0012', 'h-0002'),
            ('h-0012', 'h-0003'),
            ('h-0012', 'h-0004'),
            ('h-0012', 'h-0005'),
            ('h-0012', 'h-0006'),
            ('h-0012', 'h-0007'),
            ('h-0012', 'h-0008'),
            ('h-0012', 'h-0009'),
            ('h-0012', 'h-0010'),
            ('h-0012', 'h-0011'),
            ('h-0012', 'h-0014'),
            ('h-0012', 'h-0015'),

            ('h-0013', 'h-0000'),
            ('h-0013', 'h-0001'),
            ('h-0013', 'h-0002'),
            ('h-0013', 'h-0003'),
            ('h-0013', 'h-0004'),
            ('h-0013', 'h-0005'),
            ('h-0013', 'h-0006'),
            ('h-0013', 'h-0007'),
            ('h-0013', 'h-0008'),
            ('h-0013', 'h-0009'),
            ('h-0013', 'h-0010'),
            ('h-0013', 'h-0011'),
            ('h-0013', 'h-0014'),
            ('h-0013', 'h-0015'),

            ('h-0014', 'h-0000'),
            ('h-0014', 'h-0001'),
            ('h-0014', 'h-0002'),
            ('h-0014', 'h-0003'),
            ('h-0014', 'h-0004'),
            ('h-0014', 'h-0005'),
            ('h-0014', 'h-0006'),
            ('h-0014', 'h-0007'),
            ('h-0014', 'h-0008'),
            ('h-0014', 'h-0009'),
            ('h-0014', 'h-0010'),
            ('h-0014', 'h-0011'),
            ('h-0014', 'h-0012'),
            ('h-0014', 'h-0013'),

            ('h-0015', 'h-0000'),
            ('h-0015', 'h-0001'),
            ('h-0015', 'h-0002'),
            ('h-0015', 'h-0003'),
            ('h-0015', 'h-0004'),
            ('h-0015', 'h-0005'),
            ('h-0015', 'h-0006'),
            ('h-0015', 'h-0007'),
            ('h-0015', 'h-0008'),
            ('h-0015', 'h-0009'),
            ('h-0015', 'h-0010'),
            ('h-0015', 'h-0011'),
            ('h-0015', 'h-0012'),
            ('h-0015', 'h-0013')
        ]
        for i in range(1000):
            pair = lf._sample_pair_of_hosts_not_same_tor(4, np.random.RandomState(seed=1))
            self.assertIn(pair, pairs_test)

    def test_tmp(self):
        import torch
        from models.utils import full_cross_entropy
        from dataprep.input_output import read_link_failure_data, write_link_failure_data
        tree = make_topo(8)
        tree = sprep.add_index_to_nodes(tree)
        dataset = lf.get_link_failure_dataset(
            graph=tree,
            max_num_failures=10,
            seed=25,
            num_samples=100,
            k=8,
            cached_edges_to_pairs='/opt/project/data/fat-tree-k8/edges-to-pairs-host-to-host.json',
            output_types=['wcmp', 'hula', 'lcp', 'ecmp']
        )
        write_link_failure_data(dataset, 'test.h5')
        dataset = read_link_failure_data('test.h5')
        targets = dataset['targets']
        print("Number of drops: ", targets[:, 0].sum())
        print("Number of triplets: ", np.sum(np.logical_and(targets > 0.32, targets < 0.34)) / 3)
        print("Number of halfs: ", np.sum(np.logical_and(targets > 0.49, targets < 0.51)) / 2)
        print("Number of quarteres: ", np.sum(np.logical_and(targets > 0.24, targets < 0.251)) / 4)
        print("Number of downstreams: ", np.sum(targets[1:, :] > 0.99))
        print(full_cross_entropy(torch.tensor(np.log(targets + 1e-6)), torch.tensor(targets), torch.tensor([[1.]])))
        tor_indices = [d['idx'] for n, d in tree.nodes(data=True) if n.startswith('tor')]
        for i in range(dataset['cur_locs'].shape[0]):
            cl = dataset['cur_locs'][i, 0]
            if cl not in tor_indices:
                continue
            ts = dataset['targets'][i, :]
            ts[ts > 1e-6] = 1
            print(ts)

    def test_dataset_filter(self):
        from dataprep.datasets import StatefulDataset, filter_dataset
        tree = make_topo(8)
        tree = sprep.add_index_to_nodes(tree)
        dataset = lf.get_link_failure_dataset(
            graph=tree.to_undirected(),
            max_num_failures=10,
            seed=25,
            num_samples=100,
            k=8
        )
        ds = StatefulDataset(
            network_states=dataset['network_states'],
            all_masks=dataset['all_masks'],
            cur_locs=dataset['cur_locs'],
            destinations=dataset['destinations'],
            all_neighbors=dataset['all_neighbors'],
            targets=dataset['targets'],
            nodes_with_state=dataset['nodes_with_state'],
            embeddings=None,
            use_embeddings_as_queries_for_attention_over_links=False,
            generate_weights=False
        )
        ds2 = filter_dataset(ds, tree, "core")
        self.assertLessEqual(ds.targets.shape[0], ds2.targets.shape[0])

