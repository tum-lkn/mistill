import unittest
import torch
from torch.utils.data import DataLoader
import numpy as np
import models.stateful as stateful
import layers.attn as attn
from models.utils import full_cross_entropy
from topos.fattree import make_topo as make_fat_tree
from embeddings.defaults import fat_tree_ip_scheme
from dataprep.sp_prep import add_node_embedding, add_index_to_nodes
from dataprep.datasets import SpfDataSet


class TestStatefulModel(unittest.TestCase):

    def setUp(self) -> None:
        self.d = 7
        self.max_degree = 6
        self.bs = 20
        self.num_nodes = 25
        self.num_ft = 12
        self.config = stateful.StatefulConfig(
            link_attns=[
                attn.MultiHeadAttentionModuleConfig(
                    num_heads=2,
                    dim_fcn=10,
                    dim_hidden=5,
                    dim_in=self.num_ft * self.max_degree,
                    dim_out=5
                )
            ],
            hlsas_attn=attn.MultiHeadAttentionModuleConfig(
                num_heads=2,
                dim_fcn=10,
                dim_hidden=10,
                dim_in=10,
                dim_out=10,
                dim_q=self.d * 2,
                dim_v=10,
                dim_k=self.d,
                attn_activation='gs'
            ),
            neighbor_attns=attn.MultiHeadAttentionModuleConfig(
                num_heads=3,
                dim_fcn=10,
                dim_hidden=10,
                dim_in=self.d * 2,
                dim_out=10
            ),
            final_fcns=[16, 16],
            dim_embedding=self.d,
            max_degree=self.max_degree,
            pool_links='average',
            hlsa_model='fcn',
            policy='test',
            packets_droppeable=True,
            alpha_l1_hlsa_attn_weights=0.,
            alpha_l1_hlsas=0.,
            neighbor_model='fcn',
            num_nodes=self.num_nodes,
            num_nodes_with_state=25,
            hlsa_attn_key='current_loc',
            multiclass=False,
            cur_loc_and_dst_q_hlsa=True,
            hlsa_gs=stateful.GumbelSoftmaxConfig(
                temperature=0.6,
                arity=2,
                num_blocks=5
            )
        )

    def test_build(self):
        model = stateful.StatefulModel(self.config)
        num_parameters = (8 + 8) + 5 + (3 * 3 + 2) + 4 + 2
        params = [p for p in model.parameters()]
        self.assertEqual(num_parameters, len(params))

    def test_forward(self):
        model = stateful.StatefulModel(self.config)
        network_state = torch.randn(self.bs, self.num_nodes, self.max_degree, self.num_ft)
        network_mask = torch.tensor(np.random.binomial(
            n=1,
            p=0.5,
            size=(self.bs, self.num_nodes, self.max_degree, 1)
        ).astype(np.float32))
        neighbor_embeddings = torch.tensor(np.random.binomial(
            n=1,
            p=0.5,
            size=(self.bs, self.max_degree, self.d)
        ).astype(np.float32))
        neighbor_masks = torch.tensor(np.random.binomial(
            n=1,
            p=0.5,
            size=(self.bs, self.max_degree, 1)
        ).astype(np.float32))
        cur_loc_embds = torch.tensor(np.random.binomial(
            n=1,
            p=0.5,
            size=(self.bs, self.d)
        ).astype(np.float32))
        embeddings = torch.tensor(np.random.binomial(
            n=1,
            p=0.5,
            size=(self.num_nodes, self.d)
        ).astype(np.float32))
        embd_nodes_state = torch.unsqueeze(embeddings, dim=1)

        ret = model(
            network_state=network_state,
            network_state_mask=network_mask,
            embeddings_neighbors=neighbor_embeddings,
            mask_embeddings=neighbor_masks,
            embd_current_location=cur_loc_embds,
            embd_destination=cur_loc_embds,
            embeddings=embeddings,
            embd_nodes_state=embd_nodes_state
        )
        model.reduce_tau()
        ret = ret.detach().numpy()
        self.assertEqual(2, ret.ndim)
        self.assertEqual(self.bs, ret.shape[0])
        self.assertEqual(self.max_degree + 1, ret.shape[1])
