"""
数据处理模块，与数据处理相关的函数、类应该在这个文件
"""

from typing import List, Tuple
import torch
from torchvision import transforms
from PIL import Image


def read_img(
    path: str,
    image_size: Tuple[int, 2] = (224, 224),
    mean: List[float] = None,
    std: List[float] = None,
    unsqueeze: bool = True,
) -> torch.Tensor:
    """读取图像文件并转换为模型输入的张量

    Args:
        path (str): 图像文件路径
        image_size (Tuple[int, 2]): 图像尺寸
        mean (List[float]): 均值
        std (List[float]): 标准差
        unsqueeze (bool): 是否在第0维增加一个维度

    Returns:
        torch.Tensor: 模型输入张量
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    image: Image = Image.open(path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    image_tensor = transform(image)
    if unsqueeze:
        image_tensor = image_tensor.unsqueeze(0)
    return image_tensor
