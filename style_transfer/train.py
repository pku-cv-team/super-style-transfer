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
    resize_img_tensor,
)
from style_transfer.utils.json_loader import JsonLoader
from style_transfer.models.gatys import GatysStyleTransferModel, VGGFeatureExtractor


def train(transfer_model: nn.Module, iterations: int, optimzer: torch.optim):
    """训练风格迁移模型

    Args:
        transfer_model: 风格迁移模型

    Returns:
        torch.Tensor: 生成图像
    """
    torch.autograd.set_detect_anomaly(True)
    for i in range(iterations):
        optimzer.zero_grad()
        loss = transfer_model()
        loss.backward()
        optimzer.step()
        print(f"iteration: {i}, loss: {loss.item()}")


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
    content_image = read_img_to_tensor(json_loader.load("content_image"))
    style_image = read_img_to_tensor(json_loader.load("style_image"))
    content_size: Tuple[int, int] = content_image.shape[-2:]
    content_image, style_image = resize_img_tensor(
        content_image, image_size
    ), resize_img_tensor(style_image, image_size)
    content_image, style_image = normalize_img_tensor(content_image).unsqueeze(
        0
    ), normalize_img_tensor(style_image).unsqueeze(0)
    content_weight: float = json_loader.load("content_weight")
    style_weight: float = json_loader.load("style_weight")
    content_image.requires_grad = False
    style_image.requires_grad = False
    transfer_model = None
    if model_type == "gatys":
        feature_extractor = VGGFeatureExtractor()
        transfer_model = GatysStyleTransferModel(
            feature_extractor,
            content_image,
            style_image,
            content_weight=content_weight,
            style_weight=style_weight,
        )
    else:
        raise ValueError("Unsupported model type.")
    iterations: int = json_loader.load("iterations")
    learning_rate: float = json_loader.load("learning_rate")
    device = torch.device(
        "cuda"
        if json_loader.load("device") == "cuda" and torch.cuda.is_available()
        else "cpu"
    )
    transfer_model = transfer_model.to(device)
    optimizer: torch.optim = torch.optim.Adam(
        transfer_model.parameters(), lr=learning_rate
    )
    train(transfer_model, iterations, optimizer)
    output_file_path = json_loader.load("output_image")
    result_img = resize_img_tensor(
        unnormalize_img_tensor(transfer_model.generated_image[0]), content_size
    )
    save_img_from_tensor(result_img, output_file_path)


if __name__ == "__main__":
    main()
