"""预转换模块，将RGB转化为YCBCR格式"""

import torch
import numpy as np


# SR方法为什么只对于Y通道进行处理，因为人眼对于像素的亮度更加敏感
def convert_rgb_to_y(img):
    """将RGB图像转换为Y通道图像"""
    if img.isinstance(np.ndarray):
        return (
            16.0
            + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2])
            / 256.0
        )
    if img.isinstance(torch.Tensor):
        if len(img.shape) == 4:
            img = img.squeeze(0)
        return (
            16.0
            + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :])
            / 256.0
        )
    raise TypeError("Unknown Type", type(img))


# 得到RGB转化为YCBCR的格式，和论文对齐
def convert_rgb_to_ycbcr(img):
    """将RGB图像转换为YCBCR图像"""
    if img.isinstance(np.ndarray):
        y = (
            16.0
            + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2])
            / 256.0
        )
        cb = (
            128.0
            + (-37.945 * img[:, :, 0] - 74.494 * img[:, :, 1] + 112.439 * img[:, :, 2])
            / 256.0
        )
        cr = (
            128.0
            + (112.439 * img[:, :, 0] - 94.154 * img[:, :, 1] - 18.285 * img[:, :, 2])
            / 256.0
        )
        return np.array([y, cb, cr]).transpose([1, 2, 0])
    if img.isinstance(torch.Tensor):
        if len(img.shape) == 4:
            img = img.squeeze(0)
        y = (
            16.0
            + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :])
            / 256.0
        )
        cb = (
            128.0
            + (-37.945 * img[0, :, :] - 74.494 * img[1, :, :] + 112.439 * img[2, :, :])
            / 256.0
        )
        cr = (
            128.0
            + (112.439 * img[0, :, :] - 94.154 * img[1, :, :] - 18.285 * img[2, :, :])
            / 256.0
        )
        return torch.cat([y, cb, cr], 0).permute(1, 2, 0)
    raise TypeError("Unknown Type", type(img))


# 将YCBCR转化为RGB
def convert_ycbcr_to_rgb(img):
    """将YCBCR图像转换为RGB图像"""
    if img.isinstance(np.ndarray):
        r = 298.082 * img[:, :, 0] / 256.0 + 408.583 * img[:, :, 2] / 256.0 - 222.921
        g = (
            298.082 * img[:, :, 0] / 256.0
            - 100.291 * img[:, :, 1] / 256.0
            - 208.120 * img[:, :, 2] / 256.0
            + 135.576
        )
        b = 298.082 * img[:, :, 0] / 256.0 + 516.412 * img[:, :, 1] / 256.0 - 276.836
        return np.array([r, g, b]).transpose([1, 2, 0])
    if img.isinstance(torch.Tensor):
        if len(img.shape) == 4:
            img = img.squeeze(0)
        r = 298.082 * img[0, :, :] / 256.0 + 408.583 * img[2, :, :] / 256.0 - 222.921
        g = (
            298.082 * img[0, :, :] / 256.0
            - 100.291 * img[1, :, :] / 256.0
            - 208.120 * img[2, :, :] / 256.0
            + 135.576
        )
        b = 298.082 * img[0, :, :] / 256.0 + 516.412 * img[1, :, :] / 256.0 - 276.836
        return torch.cat([r, g, b], 0).permute(1, 2, 0)
    raise TypeError("Unknown Type", type(img))


# 计算PSNR,计算两幅图像之间的峰值信噪比,PSNR值越高表示图像的质量越好，失真越少
def calc_psnr(img1, img2):
    """计算两幅图像之间的PSNR"""
    return 10.0 * torch.log10(1.0 / torch.mean((img1 - img2) ** 2))
