import unittest
import os
import sys
sys.path.insert(0, '/opt/project')
import dataprep.sp_prep as dataprep
from topos.fattree import make_topo
import dataprep.input_output as dio
from embeddings.arbitrary import _independent_bernoulli_based


class TestDistributionalDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = dataprep.add_index_to_nodes(make_topo(4))
        self.embds = _independent_bernoulli_based(self.graph, 16)
        dio.write_embedding(self.embds, 'test-embeddings.h5')

    def test_distributional_spf(self):
        ret = dataprep.distributional_spf_dataset(self.graph)
        for sample in ret:
            self.assertAlmostEqual(1, sample[dataprep.H5_TARGET])

    def test_assembly(self):
        ret = dataprep.distributional_spf_dataset(self.graph)

        dio.write_distributional_spf_dataset(ret, 'test-set.h5')
        destinations, values, masks, targets = dataprep.expand_distributional_spf_dataset('test-set.h5', 'test-embeddings.h5')
        self.assertEqual(masks.ndim, 3)
        self.assertEqual(values.ndim, 3)
        self.assertEqual(targets.ndim, 2)
        self.assertEqual(destinations.ndim, 2)
        self.assertEqual(destinations.shape[1], 16)
        self.assertEqual(masks.shape[1], 4)
        self.assertEqual(masks.shape[2], 1)
        self.assertEqual(targets.shape[1], 4)
        self.assertEqual(values.shape[1], 4)
        self.assertEqual(values.shape[2], 16)
        os.remove('test-set.h5')

    def tearDown(self) -> None:
        os.remove('test-embeddings.h5')
        if os.path.exists('test-set.h5'):
            os.remove('test-set.h5')

    def test_make_neighbor_to_index(self):
        graph = make_topo(8)
        graph = dataprep.add_index_to_nodes(graph)
        nti = dataprep._neighbor_to_index(graph)
        self.assertEqual(0, nti['tor-0000']['h-0000'])
        self.assertEqual(1, nti['tor-0000']['h-0001'])
        self.assertEqual(2, nti['tor-0000']['h-0002'])
        self.assertEqual(3, nti['tor-0000']['h-0003'])
        self.assertEqual(4, nti['tor-0000']['agg-0000'])
        self.assertEqual(5, nti['tor-0000']['agg-0001'])
        self.assertEqual(6, nti['tor-0000']['agg-0002'])
        self.assertEqual(7, nti['tor-0000']['agg-0003'])

        self.assertEqual(0, nti['agg-0000']['tor-0000'])
        self.assertEqual(1, nti['agg-0000']['tor-0001'])
        self.assertEqual(2, nti['agg-0000']['tor-0002'])
        self.assertEqual(3, nti['agg-0000']['tor-0003'])
        self.assertEqual(4, nti['agg-0000']['core-0000'])
        self.assertEqual(5, nti['agg-0000']['core-0001'])
        self.assertEqual(6, nti['agg-0000']['core-0002'])
        self.assertEqual(7, nti['agg-0000']['core-0003'])


