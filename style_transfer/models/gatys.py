"""Gatys风格迁移模型"""

from typing import List, override
import torch
from torch import nn
from style_transfer.models.neural_style_transfer import (
    NeuralStyleTransferModel,
    FeatureExtractor,
)


# pylint: disable=too-many-instance-attributes
class GatysStyleTransferModel(NeuralStyleTransferModel):
    """Gatys风格迁移模型"""

    content_weight: float
    style_weight: float
    feature_extractor: FeatureExtractor
    content_features: List[torch.Tensor]
    style_features: List[torch.Tensor]
    generated_image: nn.Parameter

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        content_image: torch.Tensor,
        style_image: torch.Tensor,
        **kwargs,
    ):
        """初始化Gatys风格迁移模型

        Args:
            content_weight: 内容损失权重
            style_weight: 风格损失权重
            feature_extractor: 特征提取器
            content_image: 内容图像
            style_image: 风格图像
        """

        super().__init__()
        self._content_image = content_image
        self._style_image = style_image
        content_weight = kwargs.get("content_weight", 1e4)
        style_weight = kwargs.get("style_weight", 1e-2)
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.feature_extractor = feature_extractor
        self.content_features, _ = self.feature_extractor.extract_features(
            content_image
        )
        _, self.style_features = self.feature_extractor.extract_features(style_image)
        self.generated_image = nn.Parameter(content_image.clone().requires_grad_(True))
        self._content_image = content_image
        self._style_image = style_image

    @override
    def to(
        self,
        device: torch.device,
    ) -> NeuralStyleTransferModel:
        """将风格迁移模型移动到指定设备"""
        super().to(device)
        self.content_features = [
            feature.to(device) for feature in self.content_features
        ]
        self.style_features = [feature.to(device) for feature in self.style_features]
        self.feature_extractor = self.feature_extractor.to(device)
        return self

    @override
    def forward(self) -> torch.Tensor:
        """前向传播

        Args:
            generated_image: 生成图像，shape: (N, C, H, W)，其中 N 为 1

        Returns:
            loss: 损失，为内容损失和风格损失的加权和
        """
        content_features, style_features = self.feature_extractor.extract_features(
            self.generated_image
        )
        content_loss = self.__compute_content_loss(
            self.content_features, content_features
        )
        style_loss = self.__compute_style_loss(self.style_features, style_features)
        loss = self.content_weight * content_loss + self.style_weight * style_loss
        return loss

    @staticmethod
    def __compute_content_loss(
        content_features: List[torch.Tensor], generated_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """计算内容损失

        Args:
            content_features: 内容特征列表
            generated_features: 生成的特征列表

        Returns:
            torch.Tensor: 内容损失
        """
        loss = 0.0
        for content_feature, generated_feature in zip(
            content_features, generated_features
        ):
            loss += nn.MSELoss()(content_feature, generated_feature)
        return loss / len(content_features)

    @staticmethod
    def __compute_style_loss(
        style_features: List[torch.Tensor], generated_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """计算风格损失

        Args:
            style_features: 风格特征列表
            generated_features: 生成的特征列表

        Returns:
            torch.Tensor: 风格损失
        """
        loss = 0.0
        for style_feature, generated_feature in zip(style_features, generated_features):
            loss += nn.MSELoss()(style_feature, generated_feature)
        return loss / len(style_features)
