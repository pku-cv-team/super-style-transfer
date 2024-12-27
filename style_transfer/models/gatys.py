"""Gatys风格迁移模型"""

# pylint: disable=all

import torch
import torch.nn as nn
import torchvision.models
from abc import ABC, abstractmethod
from typing import List


class FeatureExtractor(ABC):
    """特征提取器的抽象基类"""

    @abstractmethod
    def extract_content_features(
        self, image_tensor: torch.Tensor
    ) -> List[torch.Tensor]:
        """提取内容特征

        Args:
            image_tensor: 图像张量

        Returns:
            List[torch.Tensor]: 内容特征列表

        Notes:
            图像张量的形状为(N, C, H, W)，其中 N 为 1
            内容特征是卷积层的输出，其形状为(N, C, H, W)，其中 N 为 1
        """
        pass

    @abstractmethod
    def extract_style_features(self, image_tensor: torch.Tensor) -> List[torch.Tensor]:
        """提取风格特征

        Args:
            image_tensor: 图像张量

        Returns:
            List[torch.Tensor]: 风格特征列表

        Notes:
            图像张量的形状为(N, C, H, W)，其中 N 为 1
            风格特征是Gram矩阵，形状为(N, C, C)，其中 N 为 1
        """
        pass


class VGGFeatureExtractor(FeatureExtractor):
    """使用VGG网络提取特征"""

    # TODO(NOT_SPECIFIC_ONE): 实现VGGFeatureExtractor类
    pass


class GatysStyleTransferModel(nn.Module):
    """Gatys风格迁移模型"""

    content_weight: float
    style_weight: float
    feature_extractor: FeatureExtractor
    content_features: List[torch.Tensor]
    style_features: List[torch.Tensor]

    def __init__(
        self,
        content_weight: float,
        style_weight: float,
        feature_extractor: FeatureExtractor,
        content_image: torch.Tensor,
        style_image: torch.Tensor,
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
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.feature_extractor = feature_extractor
        self.content_features = self.feature_extractor.extract_content_features(
            content_image
        )
        self.style_features = self.feature_extractor.extract_style_features(style_image)

    def forward(self, generated_image: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            generated_image: 生成图像

        Returns:
            loss: 损失
        """
        # TODO(NOT_SPECIFIC_ONE): 实现forward方法，计算损失
        pass

    @staticmethod
    def compute_content_loss(
        content_features: List[torch.Tensor], generated_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """计算内容损失

        Args:
            content_features: 内容特征列表
            generated_features: 生成的特征列表

        Returns:
            torch.Tensor: 内容损失
        """
        # TODO(NOT_SPECIFIC_ONE): 实现compute_content_loss方法
        pass

    @staticmethod
    def compute_style_loss(
        style_features: List[torch.Tensor], generated_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """计算风格损失

        Args:
            style_features: 风格特征列表
            generated_features: 生成的特征列表

        Returns:
            torch.Tensor: 风格损失
        """
        # TODO(NOT_SPECIFIC_ONE): 实现compute_style_loss方法
        pass
