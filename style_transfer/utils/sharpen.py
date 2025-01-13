"""锐化模块"""

import cv2
import numpy as np
import torch


# pylint: disable=no-member
def sharpen(result_img: torch.Tensor) -> torch.Tensor:
    """锐化处理"""
    result_img = result_img.permute(1, 2, 0).cpu().numpy()
    blurred_image = cv2.GaussianBlur(result_img, (3, 3), 1.0)
    result_img = cv2.addWeighted(result_img, 1.5, blurred_image, -0.5, 0)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(result_img, -1, kernel)
    sharpened = torch.from_numpy(sharpened).permute(2, 0, 1)
    return sharpened
