from style_transfer.data import read_img
from style_transfer.utils.visualization import show_image
import pytest


def test_show_img():
    image = read_img("tests/test_img/test.jpg")
    show_image(image)
    assert True
