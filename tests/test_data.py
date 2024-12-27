"""数据模块的测试，测试用例由 Moonshot AI 的 Kimi 提供"""

import pytest
from PIL import Image
import torch
from style_transfer.data import (
    read_img_to_tensor,
    normalize_img_tensor,
    unnormalize_img_tensor,
    img_tensor_to_pil,
    save_img_from_tensor,
    resize_img_tensor,
)

TEST_IMAGE_PATH = "tests/test_img/test.jpg"


def test_read_img_to_tensor():
    """测试 read_img_to_tensor 函数"""
    image_tensor = read_img_to_tensor(TEST_IMAGE_PATH)
    assert isinstance(image_tensor, torch.Tensor), "The output should be a torch.Tensor"


def test_normalize_img_tensor():
    """测试 normalize_img_tensor 函数"""
    image_tensor = read_img_to_tensor(TEST_IMAGE_PATH)
    normalized_tensor = normalize_img_tensor(image_tensor)
    assert isinstance(
        normalized_tensor, torch.Tensor
    ), "The output should be a torch.Tensor"


def test_unnormalize_img_tensor():
    """测试 unnormalize_img_tensor 函数"""
    image_tensor = read_img_to_tensor(TEST_IMAGE_PATH)
    unnormalized_tensor = unnormalize_img_tensor(image_tensor)
    assert isinstance(
        unnormalized_tensor, torch.Tensor
    ), "The output should be a torch.Tensor"


def test_img_tensor_to_pil():
    """测试 img_tensor_to_pil 函数"""
    image_tensor = read_img_to_tensor(TEST_IMAGE_PATH)
    pil_image = img_tensor_to_pil(image_tensor)
    assert isinstance(pil_image, Image.Image), "The output should be a PIL Image"


def test_save_img_from_tensor(tmp_path):
    """测试 save_img_from_tensor 函数"""
    image_tensor = read_img_to_tensor(TEST_IMAGE_PATH)
    output_file = tmp_path / "test_image.jpg"
    save_img_from_tensor(image_tensor, str(output_file))
    assert output_file.exists(), "The image should be saved to the specified path"


def test_resize_img_tensor():
    """测试 resize_img_tensor 函数"""
    image_tensor = read_img_to_tensor(TEST_IMAGE_PATH)
    resized_tensor = resize_img_tensor(image_tensor, (256, 256))
    assert (
        resized_tensor.size(1) == 256 and resized_tensor.size(2) == 256
    ), "The image should be resized to the specified size"


# 运行测试
if __name__ == "__main__":
    pytest.main()
