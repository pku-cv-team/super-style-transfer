"""训练风格迁移模型"""

import argparse
from typing import Tuple
import torch
from torch import nn
from style_transfer.data import (
    read_img_to_tensor,
    save_img_from_tensor,
    normalize_img_tensor,
    unnormalize_img_tensor,
)
from style_transfer.utils.json_loader import JsonLoader
from style_transfer.models.gatys import GatysStyleTransferModel, VGGFeatureExtractor


def train(
    transfer_model: nn.Module,
    iterations: int,
    optimzer: torch.optim,
    generated_image: torch.Tensor,
) -> torch.Tensor:
    """训练风格迁移模型

    Args:
        transfer_model: 风格迁移模型

    Returns:
        torch.Tensor: 生成图像
    """
    for _ in range(iterations):
        optimzer.zero_grad()
        loss = transfer_model(generated_image)
        loss.backward()
        optimzer.step()
    return generated_image


# pylint: disable=too-many-locals
def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Train style transfer model.")
    parser.add_argument(
        "--config", type=str, help="Path to the config file.", required=True
    )
    args = parser.parse_args()
    config_path: str = args.config
    json_loader: JsonLoader = JsonLoader(config_path)
    model_type: str = json_loader.load("model")
    image_size: Tuple[int, int] = (
        json_loader.load("image_height"),
        json_loader.load("image_width"),
    )
    content_image = read_img_to_tensor(
        json_loader.load("content_image"), image_size=image_size
    )
    style_image = read_img_to_tensor(
        json_loader.load("style_image"), image_size=image_size
    )
    content_image, style_image = normalize_img_tensor(
        content_image
    ), normalize_img_tensor(style_image)
    content_weight: float = json_loader.load("content_weight")
    style_weight: float = json_loader.load("style_weight")
    transfer_model = None
    if model_type == "gatys":
        # pylint: disable=abstract-class-instantiated
        feature_extractor = VGGFeatureExtractor()
        transfer_model = GatysStyleTransferModel(
            content_weight, style_weight, feature_extractor, content_image, style_image
        )
    else:
        raise ValueError("Unsupported model type.")
    iterations: int = json_loader.load("iterations")
    learning_rate: float = json_loader.load("learning_rate")
    generated_image: torch.Tensor = content_image.clone()
    generated_image.requires_grad = True
    optimizer: torch.optim = torch.optim.Adam([generated_image], lr=learning_rate)
    generated_image = train(transfer_model, iterations, optimizer, generated_image)
    output_file_path = json_loader.load("output_image")
    save_img_from_tensor(unnormalize_img_tensor(generated_image), output_file_path)


if __name__ == "__main__":
    main()
