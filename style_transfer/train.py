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
from style_transfer.models.feature_extractor import VGGFeatureExtractor
from style_transfer.models.neural_style_transfer import NeuralStyleTransferModel
from style_transfer.models.gatys import GatysStyleTransferModel
from style_transfer.models.lapstyle import LapstyleTransferModel
from style_transfer.models.tv_decorator import TvDecorator


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
    while i <= iterations:

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
# pylint: disable=too-many-statements
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

    # 加载图片
    content_image = read_img_to_tensor(json_loader.load("content_image"))
    style_image = read_img_to_tensor(json_loader.load("style_image"))

    # 加载图像大小调整器
    resize_stragety: dict = json_loader.load_resize_stragety()
    content_image_resizer = image_resizer_creater(resize_stragety)
    style_image_resizer = image_resizer_creater(resize_stragety)

    # 调整图像大小
    content_image = content_image_resizer.resize_to(content_image)
    style_image = style_image_resizer.resize_to(style_image)

    # 图像处理
    content_image, style_image = normalize_img_tensor(content_image).unsqueeze(
        0
    ), normalize_img_tensor(style_image).unsqueeze(0)
    content_image.requires_grad = False
    style_image.requires_grad = False

    # 加载权重
    content_weight: float = json_loader.load("content_weight")
    style_weight: float = json_loader.load("style_weight")
    tv_weight: float = json_loader.load("tv_weight")

    # 创建初始化图像
    init_strategy: str = json_loader.load("init_strategy")
    init_image = None
    if init_strategy == "content":
        init_image = content_image.clone()
    elif init_strategy == "style":
        init_image = style_image.clone()
    elif init_strategy == "noise":
        init_image = torch.rand(*content_image.shape)
    else:
        raise ValueError("Unsupported initial strategy.")

    # 加载特征提取器及创建风格迁移模型
    content_layers = json_loader.load("content_layers")
    style_layers = json_loader.load("style_layers")
    content_layer_weights = json_loader.load("content_layer_weights")
    style_layer_weights = json_loader.load("style_layer_weights")
    transfer_model = None
    if model_type == "gatys":
        feature_extractor = VGGFeatureExtractor(
            content_layers=content_layers, style_layers=style_layers
        )
        transfer_model = GatysStyleTransferModel(
            feature_extractor,
            content_image,
            style_image,
            content_weight=content_weight,
            style_weight=style_weight,
            content_layer_weights=content_layer_weights,
            style_layer_weights=style_layer_weights,
            init_image=init_image,
        )
    elif model_type == "lapstyle":
        feature_extractor = VGGFeatureExtractor(
            content_layers=content_layers, style_layers=style_layers
        )
        gatsy_model = GatysStyleTransferModel(
            feature_extractor,
            content_image,
            style_image,
            content_weight=content_weight,
            style_weight=style_weight,
            content_layer_weights=content_layer_weights,
            style_layer_weights=style_layer_weights,
            init_image=init_image,
        )
        kernel_size: int = json_loader.load("pool_size")
        lap_weight: float = json_loader.load("lap_weight")
        transfer_model = LapstyleTransferModel(
            gatsy_model, kernel_size=kernel_size, lap_weight=lap_weight
        )
    else:
        raise ValueError("Unsupported model type.")
    if tv_weight > 1e-6:
        transfer_model = TvDecorator(transfer_model, tv_weight=tv_weight)
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
