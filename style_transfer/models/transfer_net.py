"""风格迁移网络"""

import torch
from torch import nn


class TransferNet(nn.Module):
    """风格迁移网络"""

    def __init__(self):
        # TODO(NOT_SPECIFIED): Initialize TransferNet
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # TODO(NOT_SPECIFIED): Implement forward


class DownSample(nn.Module):
    """下采样层"""

    def __init__(self):
        # TODO(NOT_SPECIFIED): Initialize DownSample
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # TODO(NOT_SPECIFIED): Implement forward


class UpSample(nn.Module):
    """上采样层"""

    def __init__(self):
        # TODO(NOT_SPECIFIED): Initialize UpSample
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # TODO(NOT_SPECIFIED): Implement forward


class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self):
        # TODO(NOT_SPECIFIED): Initialize ResidualBlock
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # TODO(NOT_SPECIFIED): Implement forward
