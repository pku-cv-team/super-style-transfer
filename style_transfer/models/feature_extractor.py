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
                style_features.append(compute_gama_matrix(x) / x.numel())
        return content_features, style_features


class ResNetFeatureExtractor(FeatureExtractor):
    """使用 ResNet 网络提取特征"""

    def __init__(
        self, content_layers: List[str] = None, style_layers: List[str] = None
    ):
        super().__init__()
        # 加载预训练的 ResNet50 并冻结权重
        self.__resnet50 = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.DEFAULT
        ).eval()
        for param in self.__resnet50.parameters():
            param.requires_grad = False

        # 获取 ResNet 的所有中间层
        self.layers = nn.ModuleDict(
            {
                "conv1": self.__resnet50.conv1,
                "bn1": self.__resnet50.bn1,
                "relu": self.__resnet50.relu,
                "maxpool": self.__resnet50.maxpool,
                "layer1": self.__resnet50.layer1,
                "layer2": self.__resnet50.layer2,
                "layer3": self.__resnet50.layer3,
                "layer4": self.__resnet50.layer4,
            }
        )

        # 设置默认的内容层和风格层
        if content_layers is None:
            content_layers = ["layer4"]  # 通常从更深的层提取内容特征
        if style_layers is None:
            style_layers = [
                "conv1",
                "layer1",
                "layer2",
                "layer3",
            ]  # 风格特征来自不同深度的层

        self.__content_layers = content_layers
        self.__style_layers = style_layers

    def to(self, device: torch.device) -> "ResNetFeatureExtractor":
        """将特征提取器移动到指定设备"""
        self.__resnet50.to(device)
        return self

    def extract_features(
        self, image_tensor: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """提取内容特征和风格特征"""
        content_features, style_features = [], []
        x = image_tensor

        for name, layer in self.layers.items():
            x = layer(x)
            if name in self.__content_layers:
                content_features.append(x)
            if name in self.__style_layers:
                style_features.append(compute_gama_matrix(x) / x.numel())

        return content_features, style_features


def feature_extractor_creater(feature_extractor_param: dict) -> FeatureExtractor:
    """特征提取器创建器"""
    if feature_extractor_param["type"] == "vgg19":
        return VGGFeatureExtractor(
            feature_extractor_param.get("content_layers"),
            feature_extractor_param.get("style_layers"),
        )
    if feature_extractor_param["type"] == "resnet50":
        return ResNetFeatureExtractor(
            feature_extractor_param.get("content_layers"),
            feature_extractor_param.get("style_layers"),
        )
    raise ValueError("Unknown feature extractor type.")
