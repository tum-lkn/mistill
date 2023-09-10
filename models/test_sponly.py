import unittest
import torch
from torch.utils.data import DataLoader
import numpy as np
import models.sponly as sponly
from models.utils import full_cross_entropy
from topos.fattree import make_topo as make_fat_tree
from embeddings.defaults import fat_tree_ip_scheme
from dataprep.sp_prep import add_node_embedding, add_index_to_nodes
from dataprep.datasets import SpfDataSet


class TestSpfModel(unittest.TestCase):

    def setUp(self) -> None:
        self.config = sponly.SpfConfig(
            max_degree=4,
            dim_embedding=24,
            num_heads=4,
            dim_attn_hidden=7,
            dim_attn_out=4,
            dim_out_fcn=[16, 4]
        )

    def test_forward(self):
        model = sponly.SpfModel(self.config)
        random = np.random.RandomState(seed=1)
        queries = torch.tensor(random.binomial(1, 0.5, 35).reshape(5, 1, 7), dtype=torch.float32)
        others = torch.tensor(random.binomial(1, 0.5, 210).reshape(5, 6, 7), dtype=torch.float32)
        mask = torch.tensor(random.binomial(1, 0.5, 30).reshape(5, 6, 1), dtype=torch.float32)

        output, scores = model.forward(queries, others, mask)
        self.assertEqual(2, output.ndim)
        self.assertEqual(5, output.shape[0])
        self.assertEqual(6, output.shape[1])
        self.assertEqual(5, np.sum(output.detach().numpy()))

        self.assertEqual(3, len(scores))
        self.assertEqual(3, scores[0].ndim)
        self.assertEqual(3, scores[1].ndim)
        self.assertEqual(3, scores[2].ndim)

        self.assertEqual(5, scores[0].shape[0])
        self.assertEqual(1, scores[0].shape[1])
        self.assertEqual(6, scores[0].shape[2])

    def test_training(self):
        tree = add_index_to_nodes(make_fat_tree(4))
        tree = add_node_embedding(tree, fat_tree_ip_scheme(tree, 4))
        dataset = SpfDataSet.from_graph(tree, include=lambda x: x.startswith('h-'))
        loader = DataLoader(dataset, batch_size=20, shuffle=True)

        losses = []
        model = sponly.SpfModel(config=self.config)
        opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
        loss_fct = torch.nn.CrossEntropyLoss()
        for i in range(10):
            losses.append(0)
            for j, sample in enumerate(loader):
                pred, scores = model(
                    queries=sample['destination'],
                    others=sample['neighbors'],
                    mask=sample['attention_mask']
                )
                loss = loss_fct(pred, sample['target'])
                loss.backward()
                opt.step()
                opt.zero_grad()
                print(loss)
                losses[-1] += loss
            print()
        print(losses)

    def test_training_full_cep(self):
        tree = add_index_to_nodes(make_fat_tree(4))
        tree = add_node_embedding(tree, fat_tree_ip_scheme(tree, 4))
        dataset = SpfDataSet.from_graph(tree, include=lambda x: x.startswith('h-'))
        loader = DataLoader(dataset, batch_size=20, shuffle=True)

        losses = []
        model = sponly.SpfModel(config=self.config)
        opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
        loss_fct = full_cross_entropy
        for i in range(10):
            losses.append(0)
            for j, sample in enumerate(loader):
                pred, scores = model(
                    queries=sample['destination'],
                    others=sample['neighbors'],
                    mask=sample['attention_mask']
                )
                loss = loss_fct(pred, torch.rand(20, 4))
                loss.backward()
                opt.step()
                opt.zero_grad()
                print(loss)
                losses[-1] += loss
            print()
        print(losses)


