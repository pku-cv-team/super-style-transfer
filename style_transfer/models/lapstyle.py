"""Laplacian风格迁移模型"""

from functools import cached_property
from typing import List, override
import torch
from torch.nn import functional as F
from style_transfer.models.neural_style_transfer_decorator import (
    NeuralStyleTransferDecorator,
)


class LapstyleTransferModel(NeuralStyleTransferDecorator):
    """Laplacian风格迁移模型"""

    __pool_size: List[int]
    __pool_weight: List[float]
    __lap_weight: float

    def __init__(self, model: NeuralStyleTransferDecorator, **kwargs):
        self.__pool_size = kwargs.get("pool_size", [3, 5, 7])
        self.__pool_weight = kwargs.get("pool_size", [1.0] * len(self.__pool_size))
        self.__lap_weight = kwargs.get("lap_weight", 1e4)
        super().__init__(model)

    @override
    def forward(self) -> torch.Tensor:
        """前向传播

        Returns:
            torch.Tensor: 损失
        """
        content_and_style_loss_with_weight = self._model.forward()
        lap_features = self.__compute_laplacian_feature(
            self._model.generated_image, self.__pool_size
        )
        lap_loss = (
            self._compute_loss(
                self.cached_lap_features, lap_features, weight_list=self.__pool_weight
            )
            * self.__lap_weight
        )
        return content_and_style_loss_with_weight + lap_loss

    @staticmethod
    def __compute_laplacian_feature(
        image: torch.Tensor, pool_size: List[int]
    ) -> List[torch.Tensor]:
        """计算拉普拉斯特征

        Args:
            image: 图像张量
            pool_size: 平均池化卷积核大小

        Returns:
            torch.Tensor: 拉普拉斯特征
        """
        laplacian_kernel = (
            torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(image.shape[1], 1, 3, 3)
            .to(image.device)
        )
        # 这里的 F 是 torch.nn.functional 模块，访问 conv2d 和 avg_pool2d 应该是可以的，但是pylint似乎不能识别，所以这里禁用了这个检查
        # pylint: disable=not-callable
        return [
            F.conv2d(F.avg_pool2d(image, size), laplacian_kernel, groups=3).sum(dim=1)
            for size in pool_size
        ]

    @cached_property
    def cached_lap_features(self) -> List[torch.Tensor]:
        """缓存的拉普拉斯特征"""
        # 这里访问了成员变量 _model 的保护属性 _content_image，这是一个不好的代码实践，应该避免这样的写法，但是这里暂时这样写，因为我没有想到其他的办法解决这个问题
        # pylint: disable=protected-access
        return self.__compute_laplacian_feature(
            self._model._content_image, self.__pool_size
        )
