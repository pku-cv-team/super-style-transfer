"""测试可视化工具"""

from style_transfer.data import read_img_to_tensor
from style_transfer.utils import visualization


def test_show_img():
    """测试展示图片"""
    image = read_img_to_tensor("tests/test_img/test.jpg")
    visualization.show_image(image)
    assert True
