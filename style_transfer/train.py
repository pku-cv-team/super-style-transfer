"""训练风格迁移模型"""

import argparse
from typing import Tuple
import torch
from style_transfer.data import (
    read_img_to_tensor,
    save_img_from_tensor,
    normalize_img_tensor,
    unnormalize_img_tensor,
    resize_img_tensor,
)
from style_transfer.utils.json_loader import JsonLoader
from style_transfer.models.feature_extractor import VGGFeatureExtractor
from style_transfer.models.neural_style_transfer import NeuralStyleTransferModel
from style_transfer.models.gatys import GatysStyleTransferModel
from style_transfer.models.lapstyle import LapstyleTransferModel
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from models.SRCNN import srcnn
from utils.pretransfer import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr


def train(
    transfer_model: NeuralStyleTransferModel, iterations: int, optimzer: torch.optim
):
    """训练风格迁移模型

    Args:
        transfer_model: 风格迁移模型

    Returns:
        torch.Tensor: 生成图像
    """
    torch.autograd.set_detect_anomaly(True)
    for i in range(iterations):
        optimzer.zero_grad()
        loss = transfer_model.forward()
        loss.backward()
        optimzer.step()
        print(f"iteration: {i}, loss: {loss.item()}")


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
    model_type: str = json_loader.load("model")
    content_image = read_img_to_tensor(json_loader.load("content_image"))
    style_image = read_img_to_tensor(json_loader.load("style_image"))
    image_size: Tuple[int, int] = (
        json_loader.load("image_height"),
        json_loader.load("image_width"),
    )
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
        )
        kernel_size: int = json_loader.load("pool_size")
        lap_weight: float = json_loader.load("lap_weight")
        transfer_model = LapstyleTransferModel(
            gatsy_model, kernel_size=kernel_size, lap_weight=lap_weight
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
    # TODO(NOT_SPECIFIC_ONE) 原论文中使用的是L-BFGS优化器，这里使用Adam优化器，在后续开发中应该考虑使用L-BFGS优化器
    # 这并不容易，你可能需要考虑创建新的子模块，直接在这里添加代码可能会使得这段代码变得复杂，难以维护
    optimizer: torch.optim = torch.optim.Adam(
        [transfer_model.generated_image], lr=learning_rate
    )
    train(transfer_model, iterations, optimizer)
    output_file_path = json_loader.load("output_image")
    #图像上采样
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    uppermodel=srcnn().to(device)
    scale=3
    pre_weight_path=r"C:\Users\YH\Desktop\CV大作业\super-style-transfer\style_transfer\srcnn_x3.pth"#scale为4
    state_dict=uppermodel.state_dict()
    for n,p in torch.load(pre_weight_path,map_location=lambda storage,loc:storage,weights_only=True).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)
    uppermodel.eval()
    result_img = resize_img_tensor(
        unnormalize_img_tensor(transfer_model.generated_image[0]).clamp(0, 1),
        content_size,
    )
    result_img_np = result_img.detach().cpu().numpy().transpose(1, 2, 0)
    result_img = pil_image.fromarray((result_img_np * 255).astype(np.uint8))


    result_image_width = (result_img.width // scale) * scale
    result_image_height = (result_img.height // scale) * scale
    result_img = result_img.resize((result_image_width, result_image_height), resample=pil_image.BICUBIC)
    result_img = result_img.resize((result_img.width // scale, result_img.height // scale), resample=pil_image.BICUBIC)
    result_img = result_img.resize((result_img.width * scale, result_img.height *scale), resample=pil_image.BICUBIC)

    result_img = np.array(result_img).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(result_img)

    y = ycbcr[..., 0]
    y /= 255.
    y = torch.from_numpy(y).to(device)
    y = y.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        preds = uppermodel(y).clamp(0.0, 1.0)

    # psnr = calc_psnr(y, preds)
    # print('PSNR: {:.2f}'.format(psnr))

    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    result_img = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    result_img = np.clip(convert_ycbcr_to_rgb(result_img), 0.0, 255.0).astype(np.uint8)
    result_img = pil_image.fromarray(result_img)
    result_img.save(output_file_path)


if __name__ == "__main__":
    main()
