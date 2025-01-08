"""模型支持测试"""

import torch
from style_transfer.utils.model_utils import compute_gama_matrix


def test_compute_gama_matrix_shape():
    """测试计算Gram矩阵的形状"""
    features = torch.randn(2, 3, 4, 4)
    gama_matrix = compute_gama_matrix(features)
    assert gama_matrix.shape == (2, 3, 3)


def test_compute_gama_matrix_values():
    """测试计算Gram矩阵的值"""
    features = torch.ones(1, 2, 2, 2)
    expected_gama_matrix = torch.tensor([[[4.0, 4.0], [4.0, 4.0]]])
    gama_matrix = compute_gama_matrix(features)
    print(gama_matrix)
    assert torch.allclose(gama_matrix, expected_gama_matrix)


def test_compute_gama_matrix_zero_features():
    """测试计算Gram矩阵的值"""
    features = torch.zeros(1, 2, 2, 2)
    expected_gama_matrix = torch.zeros(1, 2, 2)
    gama_matrix = compute_gama_matrix(features)
    assert torch.equal(gama_matrix, expected_gama_matrix)
