"""测试张量缩放功能，由 Kimi 生成"""

import torch
import pytest

from style_transfer.data import scale_img_tensor


# 测试用例
def test_scale_img_tensor_normal():
    """测试正常情况"""
    image_tensor = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
    size = (128, 128)
    scaled_tensor, restore_func = scale_img_tensor(image_tensor, size)
    assert scaled_tensor.shape == (3, 128, 128)
    restored_tensor = restore_func(scaled_tensor)
    assert restored_tensor.shape == image_tensor.shape


def test_scale_img_tensor_same_size():
    """测试缩放后尺寸相同的情况"""
    image_tensor = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
    size = (256, 256)
    scaled_tensor, restore_func = scale_img_tensor(image_tensor, size)
    assert scaled_tensor.shape == (3, 256, 256)
    restored_tensor = restore_func(scaled_tensor)
    assert torch.allclose(restored_tensor, image_tensor)


def test_scale_img_tensor_smaller_size():
    """测试缩放后尺寸更小的情况"""
    image_tensor = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
    size = (64, 64)
    scaled_tensor, restore_func = scale_img_tensor(image_tensor, size)
    assert scaled_tensor.shape == (3, 64, 64)
    restored_tensor = restore_func(scaled_tensor)
    assert restored_tensor.shape == image_tensor.shape


def test_scale_img_tensor_larger_size():
    """测试缩放后尺寸更大的情况"""
    image_tensor = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
    size = (256, 256)
    scaled_tensor, restore_func = scale_img_tensor(image_tensor, size)
    assert scaled_tensor.shape == (3, 256, 256)
    restored_tensor = restore_func(scaled_tensor)
    assert restored_tensor.shape == image_tensor.shape


def test_scale_img_tensor_invalid_size():
    """测试无效的尺寸"""
    image_tensor = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
    size = (0, 0)
    with pytest.raises(Exception):
        scale_img_tensor(image_tensor, size)


def test_scale_img_tensor_non_square_size():
    """测试非方形尺寸"""
    image_tensor = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
    size = (128, 64)
    scaled_tensor, restore_func = scale_img_tensor(image_tensor, size)
    assert scaled_tensor.shape == (3, 128, 64)
    restored_tensor = restore_func(scaled_tensor)
    assert restored_tensor.shape == image_tensor.shape
