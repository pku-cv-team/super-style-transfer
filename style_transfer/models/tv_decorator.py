"""添加全变分损失的Gaty模型"""

from typing import override
import torch
from style_transfer.models.neural_style_transfer_decorator import (
    NeuralStyleTransferDecorator,
)


class TvDecorator(NeuralStyleTransferDecorator):
    """Laplacian风格迁移模型"""

    __tv_weight: float

    def __init__(self, model: NeuralStyleTransferDecorator, **kwargs):
        self.__tv_weight = kwargs.get("tv_weight", 1)
        super().__init__(model)

    @override
    def forward(self) -> torch.Tensor:
        """前向传播

        Returns:
            torch.Tensor: 损失
        """
        content_and_style_loss_with_weight = self._model.forward()
        tv_loss = self.__compute_tv_loss(self._model.generated_image)
        return content_and_style_loss_with_weight + self.__tv_weight * tv_loss

    @staticmethod
    def __compute_tv_loss(image: torch.Tensor) -> torch.Tensor:
        """计算全变分损失

        Args:
            image: 图像张量

        Returns:
            float: 全变分损失
        """
        tv_loss = 0.5 * (
                torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]).mean()
                + torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]).mean()
        )
        return tv_loss
