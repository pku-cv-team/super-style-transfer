"""评估风格迁移网络"""

import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from style_transfer.models.transfer_net import TransferNet
from style_transfer.models.loss_net import LossNet
from style_transfer.dataset import CocoDataset
from style_transfer.utils.json_loader import JsonLoader
from style_transfer.data import (
    read_img_to_tensor,
)
from style_transfer.models.neural_style_transfer_creater import (
    create_style_transfer_model,
)

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH: str = (
    "experiments/models/model.pth"  # TODO(NOT_SPECIFIC_ONE) 模型保存路径应当根据实际修改
)


def evaluate(
    transfer_net: TransferNet,
    loss_net: LossNet,
    dataloader: DataLoader,
):
    """评估风格迁移网络"""
    with torch.no_grad():
        sum_loss = 0.0
        cnt = 0
        for data in dataloader:
            data = data.to(DEVICE)
            generated_image = transfer_net(data)
            loss = loss_net.compute_loss(generated_image, data)
            sum_loss += loss.item()
            cnt += 1
        ave_loss = sum_loss / cnt
    print(f"average loss:{ave_loss:.2f}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Evaluate style transfer model.")
    parser.add_argument(
        "--config", type=str, help="Path to the config file.", required=True
    )
    args = parser.parse_args()
    config_path: str = args.config
    json_loader: JsonLoader = JsonLoader(config_path)

    # 加载数据集
    root_dir: str = json_loader.load("root_dir")
    ann_file: str = json_loader.load("ann_file")
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset: CocoDataset = CocoDataset(
        root=root_dir, annFile=ann_file, transform=transform
    )

    # 定义DataLoader
    batch_size: int = json_loader.load("batch_size")
    dataloader: DataLoader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    style_image: torch.Tensor = read_img_to_tensor(json_loader.load("style_image_path"))
    style_image = transform(style_image).unsqueeze(0).repeat(batch_size, 1, 1, 1)

    loss_net = LossNet(
        lambda content_image: create_style_transfer_model(
            json_loader.load_style_transfer_param()
        )(content_image, style_image)
    )

    transfer_net = TransferNet()
    transfer_net.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    transfer_net.to(DEVICE)
    style_image.to(DEVICE)

    evaluate(transfer_net, loss_net, dataloader)
