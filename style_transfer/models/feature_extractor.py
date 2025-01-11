"""特征提取器模块"""

from abc import ABC, abstractmethod
from typing import List, Tuple, override
import torch
from torch import nn
import torchvision.models
from style_transfer.utils.model_utils import compute_gama_matrix


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

    @abstractmethod
    def to(self, device: torch.device) -> "FeatureExtractor":
        """将特征提取器移动到指定设备

        Args:
            device: 设备

        Returns:
            FeatureExtractor: 特征提取器
        """


class VGGFeatureExtractor(FeatureExtractor):
    """使用VGG网络提取特征"""

    __vgg19: nn.Module
    __content_layers: List[int]
    __style_layers: List[int]

    def __init__(
        self, content_layers: List[int] = None, style_layers: List[int] = None
    ):
        self.__vgg19 = torchvision.models.vgg19(
            weights=torchvision.models.VGG19_Weights.DEFAULT
        ).features.eval()
        self.__vgg19 = self.__vgg19[:36]
        for param in self.__vgg19.parameters():
            param.requires_grad = False
        if content_layers is None:
            content_layers = [31]
        if style_layers is None:
            style_layers = [1, 6, 11, 20, 29]
        self.__content_layers = content_layers
        self.__style_layers = style_layers

    @override
    def to(self, device: torch.device) -> "VGGFeatureExtractor":
        """将特征提取器移动到指定设备

        Args:
            device: 设备

        Returns:
            VGGFeatureExtractor: 特征提取器
        """
        self.__vgg19.to(device)
        return self

    @override
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
        # original_size = x.numel()
        for i, layer in enumerate(self.__vgg19):
            x = layer(x)
            if i in self.__content_layers:
                content_features.append(x)
            if i in self.__style_layers:
                # 论文中事实上还应该除以2,但考虑到我们还要对损失加权，所以这里略去
                # 乘以original_size是为了防止梯度消失，也便于调整权重
                style_features.append(
                    # compute_gama_matrix(x) * original_size / x.numel()
                    compute_gama_matrix(x)
                    / x.numel()
                )
        return content_features, style_features
