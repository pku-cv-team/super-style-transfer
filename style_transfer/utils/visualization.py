import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
from typing import List

def show_image(tensor: torch.Tensor, mean: List[float] = [0.485, 0.456, 0.406], std: List[float] = [0.229, 0.224, 0.225]):
    unnormalize = transforms.Compose(
        [
            transforms.Normalize(mean=[0, 0, 0], std=[1 / std[0], 1 / std[1], 1 / std[2]]),
            transforms.Normalize(mean=[-mean[0], -mean[1], -mean[2]], std=[1, 1, 1]),
        ]
    )
    image = tensor.squeeze(0)
    image = unnormalize(image)
    image = image.permute(1, 2, 0)
    plt.imshow(image.numpy())
    plt.axis('off')
    plt.show()
