"""图像调整器"""

from typing import Tuple, Callable
from abc import ABC, abstractmethod
import torch
from torch import nn
from torch.backends import cudnn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from style_transfer.utils.func_utils import my_check_return_type
from style_transfer.models.srcnn import Srcnn
from style_transfer.utils.pretransfer import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb


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


def scale_img_numpy(
    image_numpy: np.ndarray, size: Tuple[int, int]
) -> Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
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
            x, (image_numpy.shape[1], image_numpy.shape[0])
        )
    gaussian_pyramid.pop()

    laplacian_pyramid = []

    for i in range(len(gaussian_pyramid) - 1, 0, -1):
        expand = cv2.pyrUp(gaussian_pyramid[i])
        if expand.shape[0:2] != gaussian_pyramid[i - 1].shape[0:2]:
            expand = cv2.resize(
                expand,
                (gaussian_pyramid[i - 1].shape[1], gaussian_pyramid[i - 1].shape[0]),
            )
        laplacian = cv2.subtract(gaussian_pyramid[i - 1], expand)
        laplacian_pyramid.append(laplacian)

    def restore_size(processed_image_numpy: np.ndarray) -> np.ndarray:
        """恢复尺寸"""
        # 如果要取消通过金字塔恢复图像，可以取消下面语句的注释
        # return cv2.resize(
        #     processed_image_numpy, gaussian_pyramid[0].shape[0:2]
        # ).transpose(1, 0, 2)
        if gaussian_pyramid[-1].shape[0:2] != processed_image_numpy.shape[0:2]:
            processed_image_numpy = cv2.resize(
                processed_image_numpy,
                (gaussian_pyramid[-1].shape[1], gaussian_pyramid[-1].shape[0]),
            )
        for laplacian in laplacian_pyramid:
            processed_image_numpy = cv2.pyrUp(processed_image_numpy)
            if processed_image_numpy.shape[0:2] != laplacian.shape[0:2]:
                processed_image_numpy = cv2.resize(
                    processed_image_numpy, (laplacian.shape[1], laplacian.shape[0])
                )
            processed_image_numpy = cv2.add(processed_image_numpy, laplacian)
        return processed_image_numpy

    if gaussian_pyramid[-1].shape[0:2] != size:
        image_numpy = cv2.resize(gaussian_pyramid[-1], (size[1], size[0]))
    else:
        image_numpy = gaussian_pyramid[-1]

    return image_numpy, restore_size


def scale_img_tensor(
    image_tensor: torch.Tensor, size: Tuple[int, int]
) -> Tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
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
    return (
        image_tensor.contiguous(),
        lambda x: torch.from_numpy(restore_size(x.numpy().transpose(1, 2, 0)))
        .permute(2, 0, 1)
        .contiguous(),
    )


class ImageResizer(ABC):
    """图像调整器基类"""

    _size: Tuple[int, int]

    def __init__(self, size: Tuple[int, int]):
        if size[0] <= 0 or size[1] <= 0:
            raise ValueError("Invalid size")
        self._size = size

    @abstractmethod
    def resize_to(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """调整图像张量尺寸"""

    @abstractmethod
    def restore_from(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """恢复原来的尺寸"""


class TrivialResizer(ImageResizer):
    """简单的图像调整器"""

    __original_size: Tuple[int, int]

    def resize_to(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """调整图像张量尺寸"""
        self.__original_size = img_tensor.shape[-2:]
        return resize_img_tensor(img_tensor, self._size)

    def restore_from(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """恢复原来的尺寸"""
        if self.__original_size is None:
            raise ValueError("original_size is not set")
        return resize_img_tensor(img_tensor, self.__original_size)


class PyramidResizer(ImageResizer):
    """金字塔图像调整器"""

    __restore_func: Callable[[torch.Tensor], torch.Tensor]

    def resize_to(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """调整图像张量尺寸"""
        result, self.__restore_func = scale_img_tensor(img_tensor, self._size)
        return result

    def restore_from(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """恢复原来的尺寸"""
        return self.__restore_func(img_tensor)


class SrcnnResizer(ImageResizer):
    """Srcnn图像调整器"""

    __original_size: Tuple[int, int]
    __upper_model: nn.Module
    __scale: int

    def __init__(self, size: Tuple[int, int], model_path: str, scale: int):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.__upper_model = Srcnn().to(device)
        state_dict = self.__upper_model.state_dict()
        for n, p in torch.load(
            model_path, map_location=lambda storage, loc: storage, weights_only=True
        ).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)
        self.__scale = scale
        super().__init__(size)

    def resize_to(self, img_tensor: torch.Tensor) -> torch.Tensor:
        self.__original_size = img_tensor.shape[-2:]
        return resize_img_tensor(img_tensor, self._size)

    def restore_from(self, img_tensor):
        if self.__original_size is None:
            raise ValueError("original_size is not set")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cudnn.benchmark = True
        self.__upper_model.eval()
        result_img = resize_img_tensor(img_tensor, self.__original_size)
        result_img_np = result_img.detach().cpu().numpy().transpose(1, 2, 0)
        result_img = Image.fromarray((result_img_np * 255).astype(np.uint8))
        result_image_width = (result_img.width // self.__scale) * self.__scale
        result_image_height = (result_img.height // self.__scale) * self.__scale
        # pylint: disable=no-member
        # Pylint 无法识别 PIL.Image 的成员 BICUBIC ，所以这里禁用了这个检查
        result_img = result_img.resize(
            (result_image_width, result_image_height), resample=Image.BICUBIC
        )
        result_img = result_img.resize(
            (result_img.width // self.__scale, result_img.height // self.__scale),
            resample=Image.BICUBIC,
        )
        result_img = result_img.resize(
            (result_img.width * self.__scale, result_img.height * self.__scale),
            resample=Image.BICUBIC,
        )
        result_img = np.array(result_img).astype(np.float32)
        ycbcr = convert_rgb_to_ycbcr(result_img)

        y = ycbcr[..., 0]
        y /= 255.0
        y = torch.from_numpy(y).to(device)
        y = y.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            preds = self.__upper_model(y).clamp(0.0, 1.0)

        # psnr = calc_psnr(y, preds)
        # print('PSNR: {:.2f}'.format(psnr))

        preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

        result_img = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose(
            [1, 2, 0]
        )
        result_img = np.clip(convert_ycbcr_to_rgb(result_img), 0.0, 255.0).astype(
            np.uint8
        )
        return torch.from_numpy(result_img).permute(2, 0, 1).float() / 255.0


@my_check_return_type(ImageResizer)
def image_resizer_creater(resizer_param: dict) -> ImageResizer:
    """创建图像调整器"""
    if resizer_param["type"] == "trivial":
        return TrivialResizer(resizer_param["size"])
    if resizer_param["type"] == "pyramid":
        return PyramidResizer(resizer_param["size"])
    if resizer_param["type"] == "srcnn":
        return SrcnnResizer(
            resizer_param["size"], resizer_param["model_path"], resizer_param["scale"]
        )
    raise ValueError("Unsupported resizer type")
