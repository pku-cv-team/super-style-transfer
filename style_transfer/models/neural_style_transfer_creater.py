"""风格迁移模型创建器"""

from typing import Callable
import torch
from style_transfer.models.neural_style_transfer import NeuralStyleTransferModel
from style_transfer.models.gatys import GatysStyleTransferModel
from style_transfer.models.lapstyle import LapstyleTransferModel
from style_transfer.models.tv_decorator import TvDecorator
from style_transfer.models.feature_extractor import feature_extractor_creater


def create_style_transfer_model(
    model_param: dict,
) -> Callable[[torch.Tensor, torch.Tensor], NeuralStyleTransferModel]:
    """创建风格迁移模型"""
    feature_extractor = feature_extractor_creater(model_param.pop("feature_extractor"))
    decorator_param = model_param.pop("decorator", [])
    if model_param["type"] == "gatys":

        def model_creator(
            content_image: torch.Tensor, style_image: torch.Tensor
        ) -> NeuralStyleTransferModel:
            model = GatysStyleTransferModel(
                feature_extractor=feature_extractor,
                content_image=content_image,
                style_image=style_image,
                **model_param,
            )
            for param in decorator_param:
                if param["type"] == "lap_loss":
                    print("lap_loss")
                    model = LapstyleTransferModel(model, **param)
                elif param["type"] == "tv_loss":
                    print("tv_loss")
                    model = TvDecorator(model, **param)
                else:
                    raise ValueError("Unsupported decorator type.")
            return model

        return model_creator
    raise ValueError("Unsupported style transfer model type.")
