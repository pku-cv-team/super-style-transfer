import torch
from PIL import Image
from torchvision import transforms
from typing import List


def read_img(
    path: str,
    image_height: int = 224,
    image_width: int = 224,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    unsqueeze: bool = True,
) -> torch.Tensor:
    """读取图像文件并转换为模型输入的张量

    Args:
        path (str): 图像文件路径
        image_height (int): 图像高度
        image_width (int): 图像宽度
        mean (List[float]): 均值
        std (List[float]): 标准差
        unsqueeze (bool): 是否在第0维增加一个维度

    Returns:
        torch.Tensor: 模型输入张量
    """
    image: Image = Image.open(path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((image_height, image_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    image_tensor = transform(image)
    if unsqueeze:
        image_tensor = image_tensor.unsqueeze(0)
    return image_tensor
