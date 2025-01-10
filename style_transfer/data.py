"""
数据处理模块，与数据处理相关的函数、类应该在这个文件
"""

from typing import List, Tuple
import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
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
        image_numpy (np.ndarray): 图像数组
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


def size_greater_than_or_equal_to_required(
    image_numpy: np.ndarray, size: Tuple[int, int]
) -> bool:
    """判断图像张量尺寸是否大于等于要求的尺寸"""
    return image_numpy.shape[-2] >= size[0] or image_numpy.shape[-1] >= size[1]


def scale_img_numpy(image_numpy: np.ndarray, size: Tuple[int, int]):
    """缩放图像张量

    Args:
        image_tensor (np.ndarray): 图像张量
        size (Tuple[int, int]): 尺寸

    Returns:
        np.ndarray: 缩放后的图像张量
        function: 恢复尺寸的函数
    """

    # pylint: disable=no-member
    # Pylint 似乎无法识别 cv2 的属性，因此这里禁用了这个检查
    if size[0] <= 0 or size[1] <= 0:
        raise cv2.error("Invalid size")
    gaussian_pyramid = [image_numpy]
    while size_greater_than_or_equal_to_required(image_numpy=image_numpy, size=size):
        image_numpy = cv2.pyrDown(image_numpy)
        gaussian_pyramid.append(image_numpy)

    if len(gaussian_pyramid) <= 1:
        return cv2.resize(image_numpy, size), lambda x: cv2.resize(
            x, image_numpy.shape[0:2]
        )
    gaussian_pyramid.pop()

    laplacian_pyramid = []

    for i in range(len(gaussian_pyramid) - 1, 0, -1):
        expand = cv2.pyrUp(gaussian_pyramid[i])
        if expand.shape[0:2] != gaussian_pyramid[i - 1].shape[0:2]:
            expand = cv2.resize(expand, gaussian_pyramid[i - 1].shape[0:2]).transpose(
                1, 0, 2
            )
        laplacian = cv2.subtract(gaussian_pyramid[i - 1], expand)
        laplacian_pyramid.append(laplacian)

    def restore_size(processed_image_numpy: np.ndarray):
        """恢复尺寸"""
        # 如果要取消通过金字塔恢复图像，可以取消下面语句的注释
        # return cv2.resize(
        #     processed_image_numpy, gaussian_pyramid[0].shape[0:2]
        # ).transpose(1, 0, 2)
        if gaussian_pyramid[-1].shape[0:2] != processed_image_numpy.shape[0:2]:
            processed_image_numpy = cv2.resize(
                processed_image_numpy, gaussian_pyramid[-1].shape[0:2]
            ).transpose(1, 0, 2)
        for laplacian in laplacian_pyramid:
            processed_image_numpy = cv2.pyrUp(processed_image_numpy)
            if processed_image_numpy.shape[0:2] != laplacian.shape[0:2]:
                processed_image_numpy = cv2.resize(
                    processed_image_numpy, laplacian.shape[0:2]
                ).transpose(1, 0, 2)
            processed_image_numpy = cv2.add(processed_image_numpy, laplacian)
        return processed_image_numpy

    if gaussian_pyramid[-1].shape[0:2] != size:
        image_numpy = cv2.resize(gaussian_pyramid[-1], size).transpose(1, 0, 2)
    else:
        image_numpy = gaussian_pyramid[-1]

    return image_numpy, restore_size


def scale_img_tensor(image_tensor: torch.Tensor, size: Tuple[int, int]):
    """缩放图像张量

    Args:
        image_tensor (torch.Tensor): 图像张量
        size (Tuple[int, int]): 尺寸

    Returns:
        torch.Tensor: 缩放后的图像张量
        function: 恢复尺寸的函数
    """
    image_numpy = image_tensor.numpy().transpose(1, 2, 0)
    image_numpy, restore_size = scale_img_numpy(image_numpy, size)
    image_tensor = torch.from_numpy(image_numpy.transpose(2, 0, 1))
    return image_tensor, lambda x: torch.from_numpy(
        restore_size(x.numpy().transpose(1, 2, 0))
    ).permute(2, 0, 1)
