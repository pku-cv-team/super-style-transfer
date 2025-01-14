"""训练风格迁移模型"""

import argparse
import torch
from style_transfer.data import (
    read_img_to_tensor,
    save_img_from_tensor,
    normalize_img_tensor,
    unnormalize_img_tensor,
)
from style_transfer.utils.json_loader import JsonLoader
from style_transfer.utils.image_resizer import image_resizer_creater
from style_transfer.models.neural_style_transfer import NeuralStyleTransferModel
from style_transfer.models.neural_style_transfer_creater import (
    create_style_transfer_model,
)


def train(
    transfer_model: NeuralStyleTransferModel,
    iterations: int,
    optimzer: torch.optim,
    scheduler: torch.optim.lr_scheduler,
):
    """训练风格迁移模型

    Args:
        transfer_model: 风格迁移模型

    Returns:
        torch.Tensor: 生成图像
    """
    torch.autograd.set_detect_anomaly(True)
    i = 0
    while i < iterations:

        def closure():
            nonlocal i
            optimzer.zero_grad()
            loss = transfer_model.forward()
            loss.backward()
            i += 1
            scheduler.step()
            print(f"iteration: {i}, loss: {loss.item()}")
            return loss

        optimzer.step(closure)


# 这是主函数，需要从配置文件读取很多内容，因此局部变量较多，我不知道如何避免，暂时先这样
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

    # 加载图片
    content_image = read_img_to_tensor(json_loader.load("content_image"))
    style_images = [
        read_img_to_tensor(path) for path in json_loader.load_to_list("style_image")
    ]

    # 加载图像大小调整器
    resize_stragety: dict = json_loader.load_resize_stragety()
    content_image_resizer = image_resizer_creater(resize_stragety)
    style_image_resizer = image_resizer_creater(resize_stragety)

    # 调整图像大小
    content_image = content_image_resizer.resize_to(content_image)
    style_images = [style_image_resizer.resize_to(img) for img in style_images]

    # 图像处理
    content_image = normalize_img_tensor(content_image).unsqueeze(0)
    style_images = [normalize_img_tensor(img).unsqueeze(0) for img in style_images]
    content_image.requires_grad = False
    for img in style_images:
        img.requires_grad = False

    transfer_model = create_style_transfer_model(
        json_loader.load_style_transfer_param()
    )(content_image, style_images)
    iterations: int = json_loader.load("iterations")
    learning_rate: float = json_loader.load("learning_rate")
    device = torch.device(
        "cuda"
        if json_loader.load("device") == "cuda" and torch.cuda.is_available()
        else "cpu"
    )
    transfer_model = transfer_model.to(device)
    optimizer_type = json_loader.load("optimizer")
    optimizer = None
    if optimizer_type == "LBFGS":
        optimizer = torch.optim.LBFGS(
            [transfer_model.generated_image], lr=learning_rate
        )
    elif optimizer_type == "Adam":
        optimizer = torch.optim.Adam([transfer_model.generated_image], lr=learning_rate)
    else:
        raise ValueError("Unsupported optimizer type.")
    scheduler: torch.optim.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, iterations
    )
    train(transfer_model, iterations, optimizer, scheduler)
    output_file_path = json_loader.load("output_image")
    result_img = content_image_resizer.restore_from(
        unnormalize_img_tensor(transfer_model.generated_image[0])
        .cpu()
        .detach()
        .clamp(0, 1)
    ).clamp(0, 1)
    save_img_from_tensor(result_img, output_file_path)


if __name__ == "__main__":
    main()
