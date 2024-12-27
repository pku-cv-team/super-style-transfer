# pylint: disable=all

from style_transfer.data import read_img_to_tensor
from style_transfer.utils import visualization, metrices
import torch
import numpy as np


def test_show_img():
    image = read_img_to_tensor("tests/test_img/test.jpg")
    visualization.show_image(image)
    assert True


def test_compute_gama_shape():
    features = torch.rand(1, 3, 224, 224)
    gama_matrix = metrices.compute_gama_matrix(features)
    assert gama_matrix.shape == (1, 3, 3)


def test_compute_gama_value():
    features = torch.rand(1, 3, 224, 224)
    gama_matrix = metrices.compute_gama_matrix(features)
    features_np = features.numpy().reshape((1, 3, -1))
    gama_matrix_np = np.matmul(features_np, features_np.transpose(0, 2, 1))
    assert np.allclose(gama_matrix.numpy(), gama_matrix_np)
