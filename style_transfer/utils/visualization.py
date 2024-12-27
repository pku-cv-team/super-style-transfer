"""
可视化模块，与图片展示、结果可视化等的函数和类应该在这个文件
"""

from typing import List
import torch
from torchvision import transforms
import matplotlib.pyplot as plt


def show_image(
    tensor: torch.Tensor,
    mean: List[float] = None,
    std: List[float] = None,
):
    """展示图片
    Args:
        tensor (torch.Tensor): 图片张量
        mean (List[float], optional): 图片均值. Defaults to [0.485, 0.456, 0.406].
        std (List[float], optional): 图片标准差. Defaults to [0.229, 0.224, 0.225].

    Returns:
        None
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    unnormalize = transforms.Compose(
        [
            transforms.Normalize(
                mean=[0, 0, 0], std=[1 / std[0], 1 / std[1], 1 / std[2]]
            ),
            transforms.Normalize(mean=[-mean[0], -mean[1], -mean[2]], std=[1, 1, 1]),
        ]
    )
    if len(tensor.shape) == 3:
        image = tensor
    else:
        image = tensor.squeeze(0)
    image = unnormalize(image)
    image = image.permute(1, 2, 0)
    plt.imshow(image.numpy())
    plt.axis("off")
    plt.show()
