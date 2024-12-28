"""神经风格迁移模型"""

from abc import ABC, abstractmethod
import torch
from torch import nn


class NeuralStyleTransferModel(ABC, nn.Module):
    """风格迁移模型"""

    def __init__(self):
        """初始化Gatys风格迁移模型

        Args:
            content_weight: 内容损失权重
            style_weight: 风格损失权重
            feature_extractor: 特征提取器
            content_image: 内容图像
            style_image: 风格图像
        """
        super().__init__()

    @abstractmethod
    def forward(self) -> torch.Tensor:
        """前向传播

        Returns:
            torch.Tensor: 损失
        """

    # pylint: disable=arguments-differ
    def to(self, device: torch.device) -> "NeuralStyleTransferModel":
        """将风格迁移模型移动到指定设备

        Args:
            device: 设备

        Returns:
            NeuralStyleTransferModel: 风格迁移模型
        """
        self._content_image = self._content_image.to(device)
        self._style_image = self._style_image.to(device)
        super().to(device)
        return self

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
