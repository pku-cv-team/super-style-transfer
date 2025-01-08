import torch
import unittest
from style_transfer.utils.model_utils import compute_gama_matrix

class TestModelUtils(unittest.TestCase):
    def test_compute_gama_matrix_shape(self):
        features = torch.randn(2, 3, 4, 4)
        gama_matrix = compute_gama_matrix(features)
        self.assertEqual(gama_matrix.shape, (2, 3, 3))

    def test_compute_gama_matrix_values(self):
        features = torch.ones(1, 2, 2, 2)
        expected_gama_matrix = torch.tensor([[[4.0, 4.0], [4.0, 4.0]]])
        gama_matrix = compute_gama_matrix(features)
        print(gama_matrix)
        self.assertTrue(torch.allclose(gama_matrix, expected_gama_matrix))

    def test_compute_gama_matrix_zero_features(self):
        features = torch.zeros(1, 2, 2, 2)
        expected_gama_matrix = torch.zeros(1, 2, 2)
        gama_matrix = compute_gama_matrix(features)
        self.assertTrue(torch.equal(gama_matrix, expected_gama_matrix))

if __name__ == '__main__':
  unittest.main()