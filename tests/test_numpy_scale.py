"""测试 scale_img_numpy 函数，由 Kimi 生成"""

import numpy as np
import cv2
import pytest
from style_transfer.utils.image_resizer import scale_img_numpy


def test_scale_img_numpy_normal():
    """测试正常情况"""
    image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    size = (128, 128)
    scaled_image, restore_func = scale_img_numpy(image, size)
    assert scaled_image.shape == (128, 128, 3)
    restored_image = restore_func(scaled_image)
    assert restored_image.shape == image.shape


def test_scale_img_numpy_same_size():
    """测试缩放后尺寸相同的情况"""
    image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    size = (256, 256)
    scaled_image, restore_func = scale_img_numpy(image, size)
    assert scaled_image.shape == (256, 256, 3)
    restored_image = restore_func(scaled_image)
    assert np.allclose(restored_image, image)


def test_scale_img_numpy_smaller_size():
    """测试缩放后尺寸更小的情况"""
    image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    size = (64, 64)
    scaled_image, restore_func = scale_img_numpy(image, size)
    assert scaled_image.shape == (64, 64, 3)
    restored_image = restore_func(scaled_image)
    assert restored_image.shape == image.shape


def test_scale_img_numpy_larger_size():
    """测试缩放后尺寸更大的情况"""
    image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    size = (256, 256)
    scaled_image, restore_func = scale_img_numpy(image, size)
    assert scaled_image.shape == (256, 256, 3)
    restored_image = restore_func(scaled_image)
    assert restored_image.shape == image.shape


def test_scale_img_numpy_invalid_size():
    """测试无效的尺寸"""
    image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    size = (0, 0)
    # pylint: disable=no-member
    # Pylint 似乎无法识别 cv2 的属性，因此这里禁用了这个检查
    with pytest.raises(cv2.error):
        scale_img_numpy(image, size)


def test_scale_img_numpy_non_square_size():
    """测试非正方形尺寸"""
    image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    size = (128, 64)
    scaled_image, restore_func = scale_img_numpy(image, size)
    assert scaled_image.shape == (128, 64, 3)
    restored_image = restore_func(scaled_image)
    assert restored_image.shape == image.shape
