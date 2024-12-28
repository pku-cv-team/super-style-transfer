"""Gatys风格迁移模型装饰器，基于Gatys风格迁移模型，增加其他的损失函数等"""

from abc import ABC, abstractmethod
import torch
from torch import nn
from style_transfer.models.gatys import GatysStyleTransferModel


class GatysDecorator(ABC, nn.Module):
    """Gatys风格迁移模型装饰器"""

    model: GatysStyleTransferModel

    def __init__(self, model: GatysStyleTransferModel):
        self.model = model
        super().__init__()

    @abstractmethod
    def forward(self) -> torch.Tensor:
        """前向传播

        Returns:
            torch.Tensor: 损失
        """
