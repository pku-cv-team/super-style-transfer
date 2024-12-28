"""Laplacian风格迁移模型"""

from functools import cached_property
from typing import List, override
import torch
from style_transfer.models.neural_style_transfer_decorator import NeuralDecorator


class LapStyleTransferModel(NeuralDecorator):
    """Laplacian风格迁移模型"""

    __num_layers: int
    __lap_weight: float

    def __init__(self, model: NeuralDecorator, **kwargs):
        self.__num_layers = kwargs.get("num_layers", 5)
        self.__lap_weight = kwargs.get("lap_weight", 1e4)
        self.__lap_features = self.__compute_laplacian_pyramid(
            model._content_image, self.__num_layers
        )
        super().__init__(model)

    @override
    def forward(self) -> torch.Tensor:
        """前向传播

        Returns:
            torch.Tensor: 损失
        """
        content_and_style_loss_with_weight = self._model()
        lap_features = self.__compute_laplacian_pyramid(
            self._model.generated_image, self.__num_layers
        )
        lap_loss = self.__compute_laplacian_loss(self.__lap_features, lap_features)
        return content_and_style_loss_with_weight + self.__lap_weight * lap_loss

    @staticmethod
    def __compute_laplacian_loss(
        lap_features: List[torch.Tensor], generated_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """计算拉普拉斯损失

        Args:
            lap_features: 拉普拉斯特征
            generated_features: 生成特征

        Returns:
            torch.Tensor: 拉普拉斯损失
        """
        # TODO(NOT_SPECIFIC_ONE): 实现计算拉普拉斯损失的方法

    @staticmethod
    def __compute_laplacian_pyramid(
        image: torch.Tensor, num_layers: int
    ) -> List[torch.Tensor]:
        """计算拉普拉斯金字塔

        Args:
            image: 图像张量
            num_layers: 金字塔层数

        Returns:
            List[torch.Tensor]: 拉普拉斯金字塔
        """
        # TODO(NOT_SPECIFIC_ONE):

    @cached_property
    def cached_lap_features(self) -> List[torch.Tensor]:
        """缓存的拉普拉斯特征"""
        return self.__compute_laplacian_pyramid(
            self._model.content_image, self.__num_layers
        )
