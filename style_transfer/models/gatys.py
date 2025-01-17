"""Gatys风格迁移模型"""

from functools import cached_property
from typing import List
import torch
from style_transfer.models.neural_style_transfer import NeuralStyleTransferModel
from style_transfer.models.feature_extractor import FeatureExtractor


# pylint: disable=too-many-instance-attributes
class GatysStyleTransferModel(NeuralStyleTransferModel):
    """Gatys风格迁移模型"""

    __content_weight: float
    __style_weight: List[float]
    __feature_extractor: FeatureExtractor
    __content_layer_weights: List[float]
    __style_layer_weights: List[float]

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        content_image: torch.Tensor,
        style_images: List[torch.Tensor],
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
        self._style_image = style_images
        self.__content_weight = kwargs.get("content_weight", 1e4)
        self.__style_weight = kwargs.get("style_weight", [1e-2])
        self.__content_layer_weights = kwargs.get("content_layer_weights", None)
        self.__style_layer_weights = kwargs.get("style_layer_weights", None)
        # 初始化生成图像
        init_strategy = kwargs.get("init_strategy", "content")
        if init_strategy == "content":
            self.generated_image = content_image.clone()
        elif init_strategy == "noise":
            self.generated_image = torch.rand(*content_image.shape)
        else:
            raise ValueError("Unsupported initial strategy.")
        self.generated_image = self.generated_image.requires_grad_(True)
        self.__feature_extractor = feature_extractor
        self._content_image = content_image
        self._style_images = style_images

    def to(
        self,
        device: torch.device,
    ) -> NeuralStyleTransferModel:
        """将风格迁移模型移动到指定设备"""
        super().to(device)
        self.__feature_extractor = self.__feature_extractor.to(device)
        return self

    def _compute_all_loss(self, stylized_image: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            generated_image: 生成图像，shape: (N, C, H, W)，其中 N 为 1

        Returns:
            loss: 损失，为内容损失和风格损失的加权和
        """
        content_features, style_features = self.__feature_extractor.extract_features(
            stylized_image
        )
        loss = (
            self._compute_loss(
                self.cached_content_features,
                content_features,
                weight_list=self.__content_layer_weights,
            )
            * self.__content_weight
        )
        for cached_style_feature, weight in zip(
            self.cached_style_features, self.__style_weight
        ):
            loss += (
                self._compute_loss(
                    cached_style_feature,
                    style_features,
                    weight_list=self.__style_layer_weights,
                )
                * weight
            )
        return loss

    @cached_property
    def cached_content_features(self) -> List[torch.Tensor]:
        """内容特征列表"""
        return self.__feature_extractor.extract_features(self._content_image_storage)[0]

    @cached_property
    def cached_style_features(self) -> List[List[torch.Tensor]]:
        """风格特征列表"""
        return [
            self.__feature_extractor.extract_features(style_image)[1]
            for style_image in self._style_image_storage
        ]
