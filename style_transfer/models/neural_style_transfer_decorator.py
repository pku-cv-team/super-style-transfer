"""迁移模型装饰器，基于风格迁移模型，增加其他的损失函数等"""

import torch
from style_transfer.models.gatys import NeuralStyleTransferModel


class NeuralDecorator(NeuralStyleTransferModel):
    """风格迁移模型装饰器"""

    _model: NeuralStyleTransferModel

    def __init__(self, model: NeuralStyleTransferModel):
        super().__init__()
        self._model = model

    def to(self, device: torch.device) -> "NeuralDecorator":
        """将风格迁移模型移动到指定设备

        Args:
            device: 设备

        Returns:
            NeuralDecorator: 移动后的风格迁移模型
        """
        self._model = self._model.to(device)
        return self

    @property
    def generated_image(self) -> torch.Tensor:
        """生成图像"""
        return self._model.generated_image

    @generated_image.setter
    def generated_image(self, value: torch.Tensor):
        """设置生成图像"""
        self._model.generated_image = value
