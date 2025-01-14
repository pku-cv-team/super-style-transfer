"""神经风格迁移模型"""

from abc import ABC, abstractmethod
from typing import List
import torch
from torch import nn


class NeuralStyleTransferModel(ABC):
    """风格迁移模型"""

    def forward(self) -> torch.Tensor:
        """前向传播

        Returns:
            torch.Tensor: 损失
        """
        return self._compute_all_loss(self.generated_image)

    def to(self, device: torch.device) -> "NeuralStyleTransferModel":
        """将风格迁移模型移动到指定设备

        Args:
            device: 设备

        Returns:
            NeuralStyleTransferModel: 风格迁移模型
        """
        self._content_image = self._content_image.to(device)
        self._style_images = [img.to(device) for img in self._style_images]
        self.generated_image.data = self.generated_image.data.to(device)
        if self.generated_image.grad is not None:
            self.generated_image.grad = self.generated_image.grad.to(device)
        return self

    @abstractmethod
    def _compute_all_loss(self, stylized_image: torch.Tensor):
        """计算所有损失"""

    @staticmethod
    def _compute_loss(
        features: List[torch.Tensor], generated_features: List[torch.Tensor], **kargs
    ) -> torch.Tensor:
        """计算损失

        Args:
            features: 特征列表
            generated_features: 生成的特征列表
            weight_list: 权重列表

        Returns:
            torch.Tensor: 损失
        """
        weight_list = kargs.get("weight_list", [1.0] * len(features))
        if weight_list is None:
            weight_list = [1.0] * len(features)
        weight_sum = sum(weight_list)
        if weight_sum != 0:
            weight_list = [weight / weight_sum for weight in weight_list]
        loss = 0.0
        for feature, generated_feature, weight in zip(
            features, generated_features, weight_list
        ):
            loss += nn.MSELoss()(feature, generated_feature) * weight
        return loss

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
    def _style_images(self) -> List[torch.Tensor]:
        """风格图像"""
        if not hasattr(self, "_style_image_storage"):
            raise AttributeError("Style image is not set")
        return self._style_image_storage

    @_style_images.setter
    def _style_images(self, value: List[torch.Tensor]):
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
