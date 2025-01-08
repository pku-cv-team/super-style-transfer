"""神经风格迁移模型"""

from abc import ABC, abstractmethod
from typing import List
import torch
from torch import nn


class NeuralStyleTransferModel(ABC):
    """风格迁移模型"""

    @abstractmethod
    def forward(self) -> torch.Tensor:
        """前向传播

        Returns:
            torch.Tensor: 损失
        """

    def to(self, device: torch.device) -> "NeuralStyleTransferModel":
        """将风格迁移模型移动到指定设备

        Args:
            device: 设备

        Returns:
            NeuralStyleTransferModel: 风格迁移模型
        """
        self._content_image = self._content_image.to(device)
        self._style_image = self._style_image.to(device)
        self.generated_image.data = self.generated_image.data.to(device)
        if self.generated_image.grad is not None:
            self.generated_image.grad = self.generated_image.grad.to(device)
        return self

    @staticmethod
    def _compute_loss(
        features: List[torch.Tensor], generated_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """计算损失

        Args:
            features: 特征列表
            generated_features: 生成的特征列表

        Returns:
            torch.Tensor: 损失
        """
        loss = 0.0
        for feature, generated_feature in zip(
            features, generated_features
        ):
            loss += nn.MSELoss()(feature, generated_feature)
        return loss / len(features)

    @property
    def _content_image(self) -> torch.Tensor:
        """内容图像"""
        if not hasattr(self, "_content_image_storage"):
            raise AttributeError("Content image is not set")
        return self._content_image_storage

    @_content_image.setter
    def _content_image(self, value: torch.Tensor):
        """设置内容图像"""
        self._content_image_storage = value

    @property
    def _style_image(self) -> torch.Tensor:
        """风格图像"""
        if not hasattr(self, "_style_image_storage"):
            raise AttributeError("Style image is not set")
        return self._style_image_storage

    @_style_image.setter
    def _style_image(self, value: torch.Tensor):
        """设置风格图像"""
        self._style_image_storage = value

    @property
    def generated_image(self) -> torch.Tensor:
        """生成图像"""
        if not hasattr(self, "_generated_image_storage"):
            raise AttributeError("Generated image is not set")
        return self._generated_image_storage

    @generated_image.setter
    def generated_image(self, value: torch.Tensor):
        """设置生成图像"""
        self._generated_image_storage = value
