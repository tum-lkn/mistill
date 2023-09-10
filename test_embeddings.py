import unittest
import torch
import numpy as np
import embeddings


class TestEmbeddings(unittest.TestCase):
    def setUp(self) -> None:
        a = np.array([
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                ], [
                    [0, 1, 1, 0],
                    [1, 0, 0, 1],
                    [0, 1, 0, 1],
                ]
            ],
            dtype=np.float32
        )
        self.x = torch.tensor(a)

    def test_permute(self):
        y = embeddings._permute_last_two_dims(self.x)
        self.assertEqual(y.dim(), 3)
        self.assertEqual(y.shape[0], 2)
        self.assertEqual(y.shape[1], 4)
        self.assertEqual(y.shape[2], 3)

    def test_hamming_distance(self):
        D = embeddings.hamming_distance(self.x)
        self.assertEqual(D.dim(), 3)
        self.assertEqual(D.shape[0], 2)
        self.assertEqual(D.shape[1], 3)
        self.assertEqual(D.shape[2], 3)
        self.assertEqual(D[0, 0, 0], 0)
        self.assertEqual(D[0, 0, 1], 1)
        self.assertEqual(D[0, 0, 2], 1)
        self.assertEqual(D[0, 1, 0], 1)
        self.assertEqual(D[0, 1, 1], 0)
        self.assertEqual(D[0, 1, 2], 2)
        self.assertEqual(D[0, 2, 0], 1)
        self.assertEqual(D[0, 2, 1], 2)
        self.assertEqual(D[0, 2, 2], 0)

        self.assertEqual(D[1, 0, 0], 0)
        self.assertEqual(D[1, 0, 1], 4)
        self.assertEqual(D[1, 0, 2], 2)
        self.assertEqual(D[1, 1, 0], 4)
        self.assertEqual(D[1, 1, 1], 0)
        self.assertEqual(D[1, 1, 2], 2)
        self.assertEqual(D[1, 2, 0], 2)
        self.assertEqual(D[1, 2, 1], 2)
        self.assertEqual(D[1, 2, 2], 0)


if __name__ == '__main__':
    unittest.main()
