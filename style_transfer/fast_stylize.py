"""风格迁移模块，使用训练好的模型进行风格迁移。"""

import argparse
import torch
from torchvision import transforms
from style_transfer.models.transfer_net import TransferNet
from style_transfer.data import (
    read_img_to_tensor,
    save_img_from_tensor,
    unnormalize_img_tensor,
)

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH: str = (
    "experiments/models/model.pth"  # TODO(NOT_SPECIFIC_ONE) 模型保存路径应当根据实际修改
)


def stylize(content_image: torch.Tensor, model_path: str) -> torch.Tensor:
    """风格迁移函数，使用训练好的模型进行风格迁移。

    Args:
        content_image: 内容图片，shape: (1, c, h, w)
        model_path: 模型路径

    Returns:
        torch.Tensor: 迁移后的图片，shape: (1, c, h, w)
    """
    transfer_net = TransferNet()
    transfer_net.load_state_dict(torch.load(model_path, weights_only=True))
    transfer_net.to(DEVICE)
    transfer_net.eval()
    return transfer_net(content_image)


def main():
    """主函数"""
    # TODO 暂时使用命令行参数指定输入输出路径，后续可考虑设置专用文件夹
    parser = argparse.ArgumentParser(description="Apply fast style transfer model.")
    parser.add_argument(
        "--input", type=str, help="Path to the input image.", required=True
    )
    parser.add_argument(
        "--output", type=str, help="Path to the output image.", required=True
    )
    args = parser.parse_args()
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    input_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    content_image = read_img_to_tensor(args.input)
    content_image = input_transform(content_image).unsqueeze(0)
    content_image.to(DEVICE)
    # TODO 暂时以256*256的大小输出，若想引入image_resizer可再修改
    output = (
        unnormalize_img_tensor(stylize(content_image, MODEL_PATH)[0], mean, std)
        .cpu()
        .detach()
        .clamp(0, 1)
    )
    save_img_from_tensor(output, args.output)


if __name__ == "__main__":
    main()
