"""损失网络"""

from typing import Callable
import torch
from style_transfer.models.neural_style_transfer import NeuralStyleTransferModel


# pylint: disable=too-few-public-methods
class LossNet:
    """损失网络"""

    __neural_style_transfer_creator: Callable[[torch.Tensor], NeuralStyleTransferModel]

    def __init__(
        self,
        neural_style_transfer_creator: Callable[
            [torch.Tensor], NeuralStyleTransferModel
        ],
    ):
        self.__neural_style_transfer_creator = neural_style_transfer_creator
        super().__init__()

    def compute_loss(
        self, x: torch.Tensor, content_image: torch.Tensor
    ) -> torch.Tensor:
        """前向传播"""
        style_transfer: NeuralStyleTransferModel = self.__neural_style_transfer_creator(
            content_image
        ).to(x.device)
        # pylint: disable=protected-access
        return style_transfer._compute_all_loss(x)
