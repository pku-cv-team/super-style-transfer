"""模型工具函数"""

import torch


def compute_gama_matrix(features: torch.Tensor) -> torch.Tensor:
    """计算Gram矩阵

    Args:
        features: 特征，shape: (n, c, h, w)

    Returns:
        torch.Tensor: Gram矩阵，shape: (n, c, c)
    """
    n, c, h, w = features.shape
    features = features.view(n, c, h * w)
    gama_matrix = torch.bmm(features, features.transpose(1, 2))
    return gama_matrix


def save_model(model: torch.nn.Module, model_path: str):
    """保存模型"""
    torch.save(model.state_dict(), model_path)


def load_model(model: torch.nn.Module, model_path: str):
    """加载模型"""
    model.load_state_dict(torch.load(model_path, weights_only=True))
