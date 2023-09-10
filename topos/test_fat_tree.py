import unittest
import networkx as nx
import topos.fattree as ft


class TestFatTree(unittest.TestCase):
    def setUp(self) -> None:
        self.hosts = [
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
        self.tors = [
            'tor-0001',
            'tor-0002',
            'tor-0003',
            'tor-0004',
            'tor-0005',
            'tor-0006',
            'tor-0007',
            'tor-0008'
        ]
        self.aggs = [
            'agg-0001',
            'agg-0002',
            'agg-0003',
            'agg-0004',
            'agg-0005',
            'agg-0006',
            'agg-0007',
            'agg-0008'
        ]
        self.cores = [
            'core-0001',
            'core-0002',
            'core-0003',
            'core-0004'
        ]
        self.edges = [
            ('h-0000', 'tor-0000'),
            ('h-0001', 'tor-0000'),
            ('h-0002', 'tor-0001'),
            ('h-0003', 'tor-0001'),
            ('h-0004', 'tor-0002'),
            ('h-0005', 'tor-0002'),
            ('h-0006', 'tor-0003'),
            ('h-0007', 'tor-0003'),
            ('h-0008', 'tor-0004'),
            ('h-0009', 'tor-0004'),
            ('h-0010', 'tor-0005'),
            ('h-0011', 'tor-0005'),
            ('h-0012', 'tor-0006'),
            ('h-0013', 'tor-0006'),
            ('h-0014', 'tor-0007'),
            ('h-0015', 'tor-0007'),
            ('tor-0000', 'agg-0000'),
            ('tor-0000', 'agg-0001'),
            ('tor-0001', 'agg-0000'),
            ('tor-0001', 'agg-0001'),
            ('tor-0002', 'agg-0002'),
            ('tor-0002', 'agg-0003'),
            ('tor-0003', 'agg-0002'),
            ('tor-0003', 'agg-0003'),
            ('tor-0004', 'agg-0004'),
            ('tor-0004', 'agg-0005'),
            ('tor-0005', 'agg-0004'),
            ('tor-0005', 'agg-0005'),
            ('tor-0006', 'agg-0006'),
            ('tor-0006', 'agg-0007'),
            ('tor-0007', 'agg-0006'),
            ('tor-0007', 'agg-0007'),
            ('agg-0000', 'core-0000'),
            ('agg-0000', 'core-0001'),
            ('agg-0001', 'core-0002'),
            ('agg-0001', 'core-0003'),
            ('agg-0002', 'core-0000'),
            ('agg-0002', 'core-0001'),
            ('agg-0003', 'core-0002'),
            ('agg-0003', 'core-0003'),
            ('agg-0004', 'core-0000'),
            ('agg-0004', 'core-0001'),
            ('agg-0005', 'core-0002'),
            ('agg-0005', 'core-0003'),
            ('agg-0006', 'core-0000'),
            ('agg-0006', 'core-0001'),
            ('agg-0007', 'core-0002'),
            ('agg-0007', 'core-0003'),
        ]

    def graph_creation(self):
        g = ft.make_topo(4)
        for h in self.hosts:
            self.assertTrue(g.has_node(h))
        for h in self.tors:
            self.assertTrue(g.has_node(h))
        for h in self.aggs:
            self.assertTrue(g.has_node(h))
        for h in self.cores:
            self.assertTrue(g.has_node(h))

        for u, v in self.edges:
            self.assertTrue(g.has_edge(u, v))
            self.assertTrue(g.has_edge(v, u))


class TestPod(unittest.TestCase):

    def setUp(self) -> None:
        self.hosts = [
            'h-0',
            'h-1',
            'h-2',
            'h-3',
            'h-4',
            'h-5',
            'h-6',
            'h-7',
            'h-8',
            'h-9',
            'h-10',
            'h-11',
            'h-12',
            'h-13',
            'h-14',
            'h-15'
        ]
        self.tors = [
            'tor-0',
            'tor-1',
            'tor-2',
            'tor-3'
        ]
        self.aggs = [
            'agg-0',
            'agg-1',
            'agg-2',
            'agg-3'
        ]
        self.edges = [
            ('h-0', 'tor-0'),
            ('h-1', 'tor-0'),
            ('h-2', 'tor-0'),
            ('h-3', 'tor-0'),

            ('h-4', 'tor-1'),
            ('h-5', 'tor-1'),
            ('h-6', 'tor-1'),
            ('h-7', 'tor-1'),

            ('h-8', 'tor-2'),
            ('h-9', 'tor-2'),
            ('h-10', 'tor-2'),
            ('h-11', 'tor-2'),

            ('h-12', 'tor-3'),
            ('h-13', 'tor-3'),
            ('h-14', 'tor-3'),
            ('h-15', 'tor-3'),

            ('tor-0', 'agg-0'),
            ('tor-0', 'agg-1'),
            ('tor-0', 'agg-2'),
            ('tor-0', 'agg-3'),

            ('tor-1', 'agg-0'),
            ('tor-1', 'agg-1'),
            ('tor-1', 'agg-2'),
            ('tor-1', 'agg-3'),

            ('tor-2', 'agg-0'),
            ('tor-2', 'agg-1'),
            ('tor-2', 'agg-2'),
            ('tor-2', 'agg-3'),

            ('tor-3', 'agg-0'),
            ('tor-3', 'agg-1'),
            ('tor-3', 'agg-2'),
            ('tor-3', 'agg-3'),
        ]

    def test_graph_creation(self) -> None:
        graph = ft.make_pod(8)
        self.assertEqual(
            len(self.hosts) + len(self.aggs) + len(self.tors),
            graph.number_of_nodes(),
            "Number of nodes does not match"
        )
        self.assertEqual(
            len(self.edges) * 2,
            graph.number_of_edges(),
            "expected {} edges got {}".format(len(self.edges) * 2, graph.number_of_edges())
        )
        for host in self.hosts:
            self.assertTrue(graph.has_node(host), 'node {} not in graph'.format(host))
        for tor in self.tors:
            self.assertTrue(graph.has_node(tor), 'node {} not in graph'.format(tor))
        for agg in self.aggs:
            self.assertTrue(graph.has_node(agg), 'node {} not in graph'.format(agg))

        for u, v in self.edges:
            self.assertTrue(graph.has_edge(u, v), 'edge {}-{} missing'.format(u, v))
            self.assertTrue(graph.has_edge(v, u), 'edge {}-{} missing'.format(v, u))
