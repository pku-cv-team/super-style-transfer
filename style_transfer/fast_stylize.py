"""风格迁移模块，使用训练好的模型进行风格迁移。"""

import argparse
import torch
from style_transfer.models.transfer_net import TransferNet
from style_transfer.data import (
    read_img_to_tensor,
    save_img_from_tensor,
    unnormalize_img_tensor,
    normalize_img_tensor,
)
from style_transfer.utils.image_resizer import resize_img_tensor
from style_transfer.utils.model_utils import load_model

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH: str = "experiments/models/model.pth"


def stylize(content_image: torch.Tensor, model_path: str) -> torch.Tensor:
    """风格迁移函数，使用训练好的模型进行风格迁移。

    Args:
        content_image: 内容图片，shape: (1, c, h, w)
        model_path: 模型路径

    Returns:
        torch.Tensor: 迁移后的图片，shape: (1, c, h, w)
    """
    transfer_net = TransferNet()
    load_model(transfer_net, model_path)
    transfer_net.to(DEVICE)
    transfer_net.eval()
    return transfer_net(content_image)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Apply fast style transfer model.")
    parser.add_argument(
        "--input", type=str, help="Path to the input image.", required=True
    )
    parser.add_argument(
        "--output", type=str, help="Path to the output image.", required=True
    )
    args = parser.parse_args()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    content_image = read_img_to_tensor(args.input)
    original_size = content_image.shape[-2:]
    content_image = normalize_img_tensor(
        resize_img_tensor(content_image, (256, 256)), mean, std
    ).unsqueeze(0)
    content_image = content_image.to(DEVICE)
    output = (
        unnormalize_img_tensor(stylize(content_image, MODEL_PATH)[0], mean, std)
        .cpu()
        .detach()
        .clamp(0, 1)
    )
    save_img_from_tensor(resize_img_tensor(output, original_size), args.output)


if __name__ == "__main__":
    main()
