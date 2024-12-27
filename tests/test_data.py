import pytest
from style_transfer.data import read_img
import torch

def test_read_img_shape():
    image = read_img("tests/test_img/test.jpg")
    assert image.shape == (1, 3, 224, 224)

def test_read_img_valid_path():
    image = read_img("tests/test_img/test.jpg")
    assert isinstance(image, torch.Tensor)

def test_read_img_invalid_path():
    with pytest.raises(FileNotFoundError):
        read_img("tests/invalid.jpg")

def test_jpg_read_img():
    image = read_img("tests/test_img/test.jpg")
    assert image.shape == (1, 3, 224, 224)
    assert isinstance(image, torch.Tensor)

def test_png_read_img():
    image = read_img("tests/test_img/test.png")
    assert image.shape == (1, 3, 224, 224)
    assert isinstance(image, torch.Tensor)

def test_jpeg_read_img():
    image = read_img("tests/test_img/test.jpeg")
    assert image.shape == (1, 3, 224, 224)
    assert isinstance(image, torch.Tensor)
