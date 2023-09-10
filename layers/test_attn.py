import unittest
import numpy as np
import torch
import layers.attn as attn


class TestSelfAttentionlayer(unittest.TestCase):

    def test_unmasked_attention(self):
        arr = torch.tensor(np.array([
            [
                [1., 1., 1.],
                [2., 2., 2.]
            ],
            [
                [1., 2., 3.],
                [3., 2., 1.]
            ]
        ]), dtype=torch.float32)
        layer = attn.SelfAttentionLayer(3, 7, 5)
        result, scores = layer(arr, arr, arr)
        self.assertEqual(scores.ndim, 3)
        self.assertEqual(scores.shape[0], 2)
        self.assertEqual(scores.shape[1], 2)
        self.assertEqual(scores.shape[2], 2)
        self.assertEqual(result.ndim, 3)
        self.assertEqual(result.shape[0], 2)
        self.assertEqual(result.shape[1], 2)
        self.assertEqual(result.shape[2], 5)

    def test_unmasked_attention_4d(self):
        arr = torch.randn(2, 3, 5, 7)
        layer = attn.SelfAttentionLayer(7, 9, 11)
        result, scores = layer(arr, arr, arr)
        self.assertEqual(4, scores.ndim)
        self.assertEqual(2, scores.shape[0])
        self.assertEqual(3, scores.shape[1])
        self.assertEqual(5, scores.shape[2])
        self.assertEqual(5, scores.shape[3])
        self.assertEqual(4, result.ndim)
        self.assertEqual(2, result.shape[0])
        self.assertEqual(3, result.shape[1])
        self.assertEqual(5, result.shape[2])
        self.assertEqual(11, result.shape[3])

    def test_unmasked_attention_4d_extra_query(self):
        arr = torch.randn(2, 3, 5, 7)
        q = torch.randn(2, 3, 1, 7)
        layer = attn.SelfAttentionLayer(7, 9, 11)
        result, scores = layer(arr, q, arr)
        self.assertEqual(4, scores.ndim)
        self.assertEqual(2, scores.shape[0])
        self.assertEqual(3, scores.shape[1])
        self.assertEqual(1, scores.shape[2])
        self.assertEqual(5, scores.shape[3])
        self.assertEqual(4, result.ndim)
        self.assertEqual(2, result.shape[0])
        self.assertEqual(3, result.shape[1])
        self.assertEqual(1, result.shape[2])
        self.assertEqual(11, result.shape[3])

    def test_make_outer_mask(self):
        mask = torch.tensor(np.array([
            [[1], [1], [1], [0]],
            [[0], [1], [0], [1]]
        ]), dtype=torch.float32)
        ret = attn.SelfAttentionLayer._make_outer_mask(mask)
        self.assertEqual(ret.ndim, 3)
        self.assertEqual(ret.shape[0], 2)
        self.assertEqual(ret.shape[1], 1)
        self.assertEqual(ret.shape[2], 4)

        self.assertAlmostEqual(-1e9, ret[0, 0, 3].numpy())
        self.assertAlmostEqual(-1e9, ret[1, 0, 0].numpy())
        self.assertAlmostEqual(-1e9, ret[1, 0, 2].numpy())

        self.assertAlmostEqual(0, ret[0, 0, 0].numpy())
        self.assertAlmostEqual(0, ret[0, 0, 1].numpy())
        self.assertAlmostEqual(0, ret[0, 0, 2].numpy())
        self.assertAlmostEqual(0, ret[1, 0, 1].numpy())
        self.assertAlmostEqual(0, ret[1, 0, 3].numpy())

    def test_make_outer_mask_4d(self):
        # mask has shape (2, 3, 4, 1)
        mask = torch.tensor(np.array([
            [
                [[1], [1], [1], [0]],
                [[0], [1], [0], [1]],
                [[0], [0], [1], [1]]
            ],
            [
                [[0], [1], [1], [0]],
                [[1], [0], [0], [1]],
                [[1], [1], [0], [0]]
            ]
        ]), dtype=torch.float32)
        ret = attn.SelfAttentionLayer._make_outer_mask(mask)
        self.assertEqual(ret.ndim, 4)
        self.assertEqual(ret.shape[0], 2)
        self.assertEqual(ret.shape[1], 3)
        self.assertEqual(ret.shape[2], 1)
        self.assertEqual(ret.shape[3], 4)

        # Check for -1e9 wherever input is zero.
        self.assertAlmostEqual(-1e9, ret[0, 0, 0, 3].numpy())
        self.assertAlmostEqual(-1e9, ret[0, 1, 0, 0].numpy())
        self.assertAlmostEqual(-1e9, ret[0, 1, 0, 2].numpy())
        self.assertAlmostEqual(-1e9, ret[0, 2, 0, 0].numpy())
        self.assertAlmostEqual(-1e9, ret[0, 2, 0, 1].numpy())

        self.assertAlmostEqual(-1e9, ret[1, 0, 0, 0].numpy())
        self.assertAlmostEqual(-1e9, ret[1, 0, 0, 3].numpy())
        self.assertAlmostEqual(-1e9, ret[1, 1, 0, 1].numpy())
        self.assertAlmostEqual(-1e9, ret[1, 1, 0, 2].numpy())
        self.assertAlmostEqual(-1e9, ret[1, 2, 0, 2].numpy())
        self.assertAlmostEqual(-1e9, ret[1, 2, 0, 3].numpy())

        self.assertAlmostEqual(0, ret[0, 0, 0, 0].numpy())
        self.assertAlmostEqual(0, ret[0, 0, 0, 1].numpy())
        self.assertAlmostEqual(0, ret[0, 0, 0, 2].numpy())
        self.assertAlmostEqual(0, ret[0, 1, 0, 1].numpy())
        self.assertAlmostEqual(0, ret[0, 1, 0, 3].numpy())
        self.assertAlmostEqual(0, ret[0, 2, 0, 2].numpy())
        self.assertAlmostEqual(0, ret[0, 2, 0, 3].numpy())

        self.assertAlmostEqual(0, ret[1, 0, 0, 1].numpy())
        self.assertAlmostEqual(0, ret[1, 0, 0, 2].numpy())
        self.assertAlmostEqual(0, ret[1, 1, 0, 0].numpy())
        self.assertAlmostEqual(0, ret[1, 1, 0, 3].numpy())
        self.assertAlmostEqual(0, ret[1, 2, 0, 0].numpy())
        self.assertAlmostEqual(0, ret[1, 2, 0, 1].numpy())

    def test_make_masked_scores(self):
        mask = torch.tensor(np.array([
            [[1], [1], [1], [0]],
            [[0], [1], [0], [1]]
        ]), dtype=torch.float32)
        data = torch.randn(2, 4, 5)
        layer = attn.SelfAttentionLayer(dim_in=5, dim_hidden=3, dim_out=3)

        scores = layer._masked_attention(mask, data, data)
        self.assertEqual(scores.ndim, 3)
        self.assertEqual(scores.shape[0], 2)
        self.assertEqual(scores.shape[1], 4)
        self.assertEqual(scores.shape[2], 4)

        self.assertAlmostEqual(0, scores[0, 0, 3].numpy())
        self.assertAlmostEqual(0, scores[0, 1, 3].numpy())
        self.assertAlmostEqual(0, scores[0, 2, 3].numpy())
        self.assertAlmostEqual(0, scores[0, 3, 3].numpy())

        self.assertAlmostEqual(0, scores[1, 0, 0].numpy())
        self.assertAlmostEqual(0, scores[1, 1, 0].numpy())
        self.assertAlmostEqual(0, scores[1, 2, 0].numpy())
        self.assertAlmostEqual(0, scores[1, 3, 0].numpy())

        self.assertAlmostEqual(0, scores[1, 0, 2].numpy())
        self.assertAlmostEqual(0, scores[1, 1, 2].numpy())
        self.assertAlmostEqual(0, scores[1, 2, 2].numpy())
        self.assertAlmostEqual(0, scores[1, 3, 2].numpy())

    def test_make_masked_scores_4d(self):
        # mask has shape (2, 3, 4, 1)
        mask = torch.tensor(np.array([
            [
                [[1], [1], [1], [0]],
                [[0], [1], [0], [1]],
                [[0], [0], [1], [1]]
            ],
            [
                [[0], [1], [1], [0]],
                [[1], [0], [0], [1]],
                [[1], [1], [0], [0]]
            ]
        ]), dtype=torch.float32)
        data = torch.randn(2, 3, 4, 5)
        layer = attn.SelfAttentionLayer(dim_in=5, dim_hidden=3, dim_out=3)

        scores = layer._masked_attention(mask, data, data)
        self.assertEqual(scores.ndim, 4)
        self.assertEqual(scores.shape[0], 2)
        self.assertEqual(scores.shape[1], 3)
        self.assertEqual(scores.shape[2], 4)
        self.assertEqual(scores.shape[3], 4)

        self.assertAlmostEqual(0, scores[0, 0, 0, 3].numpy())
        self.assertAlmostEqual(0, scores[0, 0, 1, 3].numpy())
        self.assertAlmostEqual(0, scores[0, 0, 2, 3].numpy())
        self.assertAlmostEqual(0, scores[0, 0, 3, 3].numpy())

        self.assertAlmostEqual(0, scores[0, 1, 0, 0].numpy())
        self.assertAlmostEqual(0, scores[0, 1, 1, 0].numpy())
        self.assertAlmostEqual(0, scores[0, 1, 2, 0].numpy())
        self.assertAlmostEqual(0, scores[0, 1, 3, 0].numpy())

        self.assertAlmostEqual(0, scores[0, 1, 0, 2].numpy())
        self.assertAlmostEqual(0, scores[0, 1, 1, 2].numpy())
        self.assertAlmostEqual(0, scores[0, 1, 2, 2].numpy())
        self.assertAlmostEqual(0, scores[0, 1, 3, 2].numpy())

        self.assertAlmostEqual(0, scores[0, 2, 0, 0].numpy())
        self.assertAlmostEqual(0, scores[0, 2, 1, 0].numpy())
        self.assertAlmostEqual(0, scores[0, 2, 2, 0].numpy())
        self.assertAlmostEqual(0, scores[0, 2, 3, 0].numpy())

        self.assertAlmostEqual(0, scores[0, 2, 0, 1].numpy())
        self.assertAlmostEqual(0, scores[0, 2, 1, 1].numpy())
        self.assertAlmostEqual(0, scores[0, 2, 2, 1].numpy())
        self.assertAlmostEqual(0, scores[0, 2, 3, 1].numpy())

    def test_make_masked_scores_4d_1d(self):
        # mask has shape (2, 3, 4, 1)
        mask = torch.tensor(np.array([
            [
                [[1], [1], [1], [0]],
                [[0], [1], [0], [1]],
                [[0], [0], [1], [1]]
            ]
        ]), dtype=torch.float32)
        data = torch.randn(2, 3, 4, 5)
        layer = attn.SelfAttentionLayer(dim_in=5, dim_hidden=3, dim_out=3)

        scores = layer._masked_attention(mask, data, data)
        self.assertEqual(scores.ndim, 4)
        self.assertEqual(scores.shape[0], 2)
        self.assertEqual(scores.shape[1], 3)
        self.assertEqual(scores.shape[2], 4)
        self.assertEqual(scores.shape[3], 4)

        self.assertAlmostEqual(0, scores[0, 0, 0, 3].numpy())
        self.assertAlmostEqual(0, scores[0, 0, 1, 3].numpy())
        self.assertAlmostEqual(0, scores[0, 0, 2, 3].numpy())
        self.assertAlmostEqual(0, scores[0, 0, 3, 3].numpy())

        self.assertAlmostEqual(0, scores[0, 1, 0, 0].numpy())
        self.assertAlmostEqual(0, scores[0, 1, 1, 0].numpy())
        self.assertAlmostEqual(0, scores[0, 1, 2, 0].numpy())
        self.assertAlmostEqual(0, scores[0, 1, 3, 0].numpy())

        self.assertAlmostEqual(0, scores[0, 1, 0, 2].numpy())
        self.assertAlmostEqual(0, scores[0, 1, 1, 2].numpy())
        self.assertAlmostEqual(0, scores[0, 1, 2, 2].numpy())
        self.assertAlmostEqual(0, scores[0, 1, 3, 2].numpy())

        self.assertAlmostEqual(0, scores[0, 2, 0, 0].numpy())
        self.assertAlmostEqual(0, scores[0, 2, 1, 0].numpy())
        self.assertAlmostEqual(0, scores[0, 2, 2, 0].numpy())
        self.assertAlmostEqual(0, scores[0, 2, 3, 0].numpy())

        self.assertAlmostEqual(0, scores[0, 2, 0, 1].numpy())
        self.assertAlmostEqual(0, scores[0, 2, 1, 1].numpy())
        self.assertAlmostEqual(0, scores[0, 2, 2, 1].numpy())
        self.assertAlmostEqual(0, scores[0, 2, 3, 1].numpy())

        self.assertAlmostEqual(0, scores[1, 0, 0, 3].numpy())
        self.assertAlmostEqual(0, scores[1, 0, 1, 3].numpy())
        self.assertAlmostEqual(0, scores[1, 0, 2, 3].numpy())
        self.assertAlmostEqual(0, scores[1, 0, 3, 3].numpy())

        self.assertAlmostEqual(0, scores[1, 1, 0, 0].numpy())
        self.assertAlmostEqual(0, scores[1, 1, 1, 0].numpy())
        self.assertAlmostEqual(0, scores[1, 1, 2, 0].numpy())
        self.assertAlmostEqual(0, scores[1, 1, 3, 0].numpy())

        self.assertAlmostEqual(0, scores[1, 1, 0, 2].numpy())
        self.assertAlmostEqual(0, scores[1, 1, 1, 2].numpy())
        self.assertAlmostEqual(0, scores[1, 1, 2, 2].numpy())
        self.assertAlmostEqual(0, scores[1, 1, 3, 2].numpy())

        self.assertAlmostEqual(0, scores[1, 2, 0, 0].numpy())
        self.assertAlmostEqual(0, scores[1, 2, 1, 0].numpy())
        self.assertAlmostEqual(0, scores[1, 2, 2, 0].numpy())
        self.assertAlmostEqual(0, scores[1, 2, 3, 0].numpy())

        self.assertAlmostEqual(0, scores[1, 2, 0, 1].numpy())
        self.assertAlmostEqual(0, scores[1, 2, 1, 1].numpy())
        self.assertAlmostEqual(0, scores[1, 2, 2, 1].numpy())
        self.assertAlmostEqual(0, scores[1, 2, 3, 1].numpy())

    def test_forward(self):
        data = torch.randn(2, 3, 5)
        layer = attn.SelfAttentionLayer(dim_in=5, dim_hidden=11, dim_out=7)
        ret, scores = layer(keys=data, queries=data, values=data)
        self.assertEqual(ret.ndim, 3)
        self.assertEqual(ret.shape[0], 2)
        self.assertEqual(ret.shape[1], 3)
        self.assertEqual(ret.shape[2], 7)

        queries = torch.randn(2, 2, 7)
        layer = attn.SelfAttentionLayer(dim_in=5, dim_hidden=11, dim_out=7, dim_q=7)
        ret, scores = layer(keys=data, queries=queries, values=data)
        self.assertEqual(ret.ndim, 3, "Mismatch dims")
        self.assertEqual(ret.shape[0], 2, "Mismatch dim0")
        self.assertEqual(ret.shape[1], 2, "Mismatch dim1")
        self.assertEqual(ret.shape[2], 7, "Mismatch dim2")

        mask = torch.tensor(np.array([
            [[1.], [1.], [0.]],
            [[1.], [0.], [0.]],
        ]), dtype=torch.float32)
        ret, scores = layer(keys=data, queries=queries, values=data, attention_mask=mask)
        self.assertEqual(ret.ndim, 3, "Mismatch dims")
        self.assertEqual(ret.shape[0], 2, "Mismatch dim0")
        self.assertEqual(ret.shape[1], 2, "Mismatch dim1")
        self.assertEqual(ret.shape[2], 7, "Mismatch dim2")

        self.assertEqual(scores.ndim, 3)
        self.assertEqual(scores.shape[0], 2, "Mismatch scores dim0")
        self.assertEqual(scores.shape[1], 2, "Mismatch scores dim1")
        self.assertEqual(scores.shape[2], 3, "Mismatch scores dim2")
        self.assertAlmostEqual(0, scores[0, 0, 2].detach().numpy())
        self.assertAlmostEqual(0, scores[0, 1, 2].detach().numpy())

        self.assertAlmostEqual(0, scores[1, 0, 1].detach().numpy())
        self.assertAlmostEqual(0, scores[1, 0, 2].detach().numpy())
        self.assertAlmostEqual(0, scores[1, 1, 1].detach().numpy())
        self.assertAlmostEqual(0, scores[1, 1, 2].detach().numpy())
        self.assertAlmostEqual(1, scores[1, 0, 0].detach().numpy())
        self.assertAlmostEqual(1, scores[1, 1, 0].detach().numpy())


class TestMultiHeadAttention(unittest.TestCase):

    def test_params(self):
        layer = attn.MultiHeadAttentionLayer(3, "SelfAttentionLayer", 5, 5, 5)
        params = [p for p in layer.parameters()]
        self.assertEqual(3 * 3, len(params), "Number of trainable parameters does not match.")
