"""迁移模型装饰器，基于风格迁移模型，增加其他的损失函数等"""

import torch
from style_transfer.models.gatys import NeuralStyleTransferModel


class NeuralDecorator(NeuralStyleTransferModel):
    """风格迁移模型装饰器"""

    model: NeuralStyleTransferModel

    def __init__(self, model: NeuralStyleTransferModel):
        self.model = model
        super().__init__()

    def to(self, device: torch.device) -> "NeuralDecorator":
        """将风格迁移模型移动到指定设备

        Args:
            device: 设备

        Returns:
            NeuralDecorator: 移动后的风格迁移模型
        """
        self.model = self.model.to(device)
        return self
