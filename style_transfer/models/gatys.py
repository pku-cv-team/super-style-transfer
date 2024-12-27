"""Gatys风格迁移模型"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import torch
from torch import nn
import torchvision.models
from style_transfer.utils.metrices import compute_gama_matrix


# pylint: disable=too-few-public-methods
class FeatureExtractor(ABC):
    """特征提取器的抽象基类"""

    @abstractmethod
    def extract_features(self, image_tensor: torch.Tensor) -> List[torch.Tensor]:
        """提取风格特征

        Args:
            image_tensor: 图像张量

        Returns:
            List[torch.Tensor]: 风格特征列表

        Notes:
            图像张量的形状为(N, C, H, W)，其中 N 为 1
            风格特征是Gram矩阵，形状为(N, C, C)，其中 N 为 1
        """


# pylint: disable=too-few-public-methods
class VGGFeatureExtractor(FeatureExtractor):
    """使用VGG网络提取特征"""

    vgg19: nn.Module
    content_layers: List[int]
    style_layers: List[int]

    # TODO(NOT_SPECIFIC_ONE): 实现VGGFeatureExtractor类
    def __init__(
        self, content_layers: List[int] = None, style_layers: List[int] = None
    ):
        self.vgg19 = torchvision.models.vgg19(
            weights=torchvision.models.VGG19_Weights.DEFAULT
        ).features.eval()
        self.vgg19 = self.vgg19[:36]
        for param in self.vgg19.parameters():
            param.requires_grad = False
        if content_layers is None:
            content_layers = [31]
        if style_layers is None:
            style_layers = [1, 6, 11, 20, 29]
        self.content_layers = content_layers
        self.style_layers = style_layers

    def to(self, device: torch.device) -> "VGGFeatureExtractor":
        """将特征提取器移动到指定设备

        Args:
            device: 设备

        Returns:
            VGGFeatureExtractor: 特征提取器
        """
        self.vgg19.to(device)
        return self

    def extract_features(
        self, image_tensor: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """提取内容特征和风格特征

        Args:
            image_tensor: 图像张量

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]: 内容特征列表和风格特征列表
        """
        content_features, style_features = [], []
        x = image_tensor
        for i, layer in enumerate(self.vgg19):
            x = layer(x)
            if i in self.content_layers:
                content_features.append(x)
            if i in self.style_layers:
                # 论文中事实上这里需要除以一个常数，即四倍的特征图大小的平方和滤波器个数的平方
                # 但考虑到最后我们需要对风格损失加权，所以这里省略了这个常数
                style_features.append(compute_gama_matrix(x))
        return content_features, style_features


class GatysStyleTransferModel(nn.Module):
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

    # pylint: disable=arguments-differ
    def to(
        self,
        device: torch.device,
    ) -> "GatysStyleTransferModel":
        super().to(device)
        self.content_features = [
            feature.to(device) for feature in self.content_features
        ]
        self.style_features = [feature.to(device) for feature in self.style_features]
        self.feature_extractor = self.feature_extractor.to(device)
        return self

    def forward(self) -> torch.Tensor:
        """前向传播

        Args:
            generated_image: 生成图像，shape: (N, C, H, W)，其中 N 为 1

        Returns:
            loss: 损失，为内容损失和风格损失的加权和
        """
        # TODO(NOT_SPECIFIC_ONE): 实现forward方法，计算损失
        content_features, style_features = self.feature_extractor.extract_features(
            self.generated_image
        )
        # style_features = self.feature_extractor.extract_style_features(self.generated_image)
        content_loss = self.compute_content_loss(
            self.content_features, content_features
        )
        style_loss = self.compute_style_loss(self.style_features, style_features)
        loss = self.content_weight * content_loss + self.style_weight * style_loss
        return loss

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
        loss = 0.0
        for content_feature, generated_feature in zip(
            content_features, generated_features
        ):
            loss += nn.MSELoss()(content_feature, generated_feature)
        return loss / len(content_features)

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
        loss = 0.0
        for style_feature, generated_feature in zip(style_features, generated_features):
            loss += nn.MSELoss()(style_feature, generated_feature)
        return loss / len(style_features)
