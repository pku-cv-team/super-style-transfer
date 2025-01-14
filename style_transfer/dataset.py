"""CoCo数据集加载及处理"""

import torch
from torchvision import transforms
from torchvision.datasets import CocoDetection


# modified from example code provided by ChatGPT
# pylint: disable=too-few-public-methods
class CocoDataset(CocoDetection):
    """Coco数据集加载及处理"""

    def __init__(self, root: str, ann_file: str, transform: transforms = None):
        super().__init__(root, ann_file)
        self.transform = transform

    def __getitem__(self, index: int) -> torch.Tensor:
        """获取图像"""
        img, _ = super().__getitem__(index)  # 忽略标注
        if self.transform:
            img = self.transform(img)
        return img
