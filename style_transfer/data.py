"""
数据处理模块，与数据处理相关的函数、类应该在这个文件
"""

from typing import List, Tuple
import torch
from torchvision import transforms
from PIL import Image
from style_transfer.utils.func_utils import my_check_return_type


@my_check_return_type(torch.Tensor)
def read_img_to_tensor(path: str) -> torch.Tensor:
    """读取图像文件并转换为模型输入的张量

    Args:
        path (str): 图像文件路径

    Returns:
        torch.Tensor: 模型输入张量
    """
    image: Image = Image.open(path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    return transform(image)


@my_check_return_type(torch.Tensor)
def normalize_img_tensor(
    image_tensor: torch.Tensor,
    mean: List[float] = None,
    std: List[float] = None,
) -> torch.Tensor:
    """将图像张量标准化

    Args:
        image_tensor (torch.Tensor): 图像张量
        mean (List[float]): 均值
        std (List[float]): 标准差

    Returns:
        torch.Tensor: 标准化后的图像张量
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    transform = transforms.Compose(
        [
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transform(image_tensor)


@my_check_return_type(torch.Tensor)
def unnormalize_img_tensor(
    image_tensor: torch.Tensor,
    mean: List[float] = None,
    std: List[float] = None,
) -> torch.Tensor:
    """将模型输出的张量转换为图像

    Args:
        image_tensor (torch.Tensor): 模型输出张量
        mean (List[float]): 均值
        std (List[float]): 标准差

    Returns:
        torch.Tensor: 图像张量
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    transform = transforms.Compose(
        [
            transforms.Normalize(
                mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]
            ),
        ]
    )
    return transform(image_tensor)


def img_tensor_to_pil(image_tensor: torch.Tensor) -> Image:
    """将图像张量转换为PIL图像

    Args:
        image_tensor (torch.Tensor): 图像张量

    Returns:
        Image: PIL图像
    """
    return transforms.ToPILImage()(image_tensor)


def save_img_from_tensor(image_tensor: torch.Tensor, output_file: str):
    """保存图像

    Args:
        image_tensor (torch.Tensor): 图像张量
        output_file (str): 输出文件路径
    """
    image = img_tensor_to_pil(image_tensor)
    image.save(output_file)


@my_check_return_type(torch.Tensor)
def resize_img_tensor(
    image_tensor: torch.Tensor, size: Tuple[int, int]
) -> torch.Tensor:
    """调整图像张量尺寸

    Args:
        image_tensor (torch.Tensor): 图像张量
        size (Tuple[int, int]): 尺寸

    Returns:
        torch.Tensor: 调整后的图像张量
    """
    transform = transforms.Compose(
        [
            transforms.Resize(size),
        ]
    )
    return transform(image_tensor)
