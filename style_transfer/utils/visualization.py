"""
可视化模块，与图片展示、结果可视化等的函数和类应该在这个文件
"""

import torch
import matplotlib.pyplot as plt
from style_transfer.data import img_tensor_to_pil


def show_image(tensor: torch.Tensor):
    """展示图片
    Args:
        tensor (torch.Tensor): 图片张量
        mean (List[float], optional): 图片均值. Defaults to [0.485, 0.456, 0.406].
        std (List[float], optional): 图片标准差. Defaults to [0.229, 0.224, 0.225].

    Returns:
        None
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    plt.imshow(img_tensor_to_pil(tensor))
    plt.axis("off")
    plt.show()
