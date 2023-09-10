import unittest
import numpy as np
import torch
import models.utils as mutils


class TestMultiClass(unittest.TestCase):

    def test_function(self):
        x = np.array([[1., 1., 0., 0., 1.]], dtype=np.float32)
        logits = np.array([[10., 10., -10., -10., 10.]], dtype=np.float32)
        res = mutils.multi_class_loss(
            torch.tensor(logits),
            torch.tensor(x),
            torch.tensor([[1.]])
        )
        self.assertLessEqual(0, res, "Result is negative: {}".format(res))
        self.assertGreaterEqual(0.01, res, "Result is too large: {}".format(res))
        print(res)

        logits = np.array([[-10., -10., 10., 10., -10.]], dtype=np.float32)
        res = mutils.multi_class_loss(
            torch.tensor(logits),
            torch.tensor(x),
            torch.tensor([[1.]])
        )
        print(res)

        logits = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.,]], dtype=np.float32)
        res = mutils.multi_class_loss(
            torch.tensor(logits),
            torch.tensor(np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0.,]])),
            torch.tensor([[1.]])
        )
        print(res)

        logits = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.,]], dtype=np.float32)
        res = mutils.multi_class_loss(
            torch.tensor(logits),
            torch.tensor(np.array([[0., 1., 1., 1., 1., 0., 0., 0., 0.,]])),
            torch.tensor([[1.]])
        )
        print(res)

        logits = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.,]], dtype=np.float32)
        res = mutils.multi_class_loss(
            torch.tensor(logits),
            torch.tensor(np.array([[0., 1., 1., 1., 1., 1., 1., 1., 1.,]])),
            torch.tensor([[1.]])
        )
        print(res)

        logits = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.,]], dtype=np.float32)
        res = mutils.multi_class_loss(
            torch.tensor(logits),
            torch.tensor(np.array([[1., 1., 1., 0., 0., 0., 0., 0., 0.,]])),
            torch.tensor([[1.]])
        )
        print(res)
