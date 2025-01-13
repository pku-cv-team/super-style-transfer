"""快速风格迁移训练模块"""

import argparse
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from style_transfer.models.transfer_net import TransferNet
from style_transfer.models.loss_net import LossNet
from style_transfer.dataset import CocoDataset
from style_transfer.utils.json_loader import JsonLoader
from style_transfer.data import read_img_to_pixel
from style_transfer.models.neural_style_transfer_creater import (
    create_style_transfer_model,
)

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH: str = "experiments/models/model.pth"
CHECKPOINT_DIR: str = "experiments/checkpoints"


# pylint: disable=too-many-arguments, too-many-positional-arguments
def train(
    transfer_net: TransferNet,
    loss_net: LossNet,
    dataloader: DataLoader,
    optimizer: torch.optim,
    epochs: int,
):
    """训练风格迁移网络"""
    torch.autograd.set_detect_anomaly(True)
    best_loss = float("inf")
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    # writer = SummaryWriter(config["log_dir"])
    for epoch in range(epochs):
        for i, data in enumerate(dataloader):
            data = data.to(DEVICE)
            optimizer.zero_grad()

            generated_image = transfer_net(data)
            loss = loss_net.compute_loss(generated_image, data)
            print(f"epoch: {epoch}, iteration: {i + 1}, loss: {loss.item()}")

            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(transfer_net.state_dict(), MODEL_PATH)
        scheduler.step()


# 这是主函数，需要从配置文件读取很多内容，因此局部变量较多，我不知道如何避免，暂时先这样
# pylint: disable=too-many-locals
def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Train style transfer model.")
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
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset: CocoDataset = CocoDataset(
        root=root_dir, annFile=ann_file, transform=transform
    )
    batch_size: int = json_loader.load("batch_size")
    dataloader: DataLoader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    style_image = (
        transform(read_img_to_pixel(json_loader.load("style_image_path")))
        .unsqueeze(0)
        .repeat(batch_size, 1, 1, 1)
    )

    # 网络构建
    loss_net = LossNet(
        lambda content_image: create_style_transfer_model(
            json_loader.load_style_transfer_param()
        )(content_image, style_image)
    )
    transfer_net = TransferNet()
    # 优化器构建(?使用AdamW?还是其它优化器呢)
    # pylint: disable=no-member
    # pylint 似乎不能正确识别优化器，这里禁用 no-member 检查
    optimizer = torch.optim.AdamW(
        transfer_net.parameters(), lr=json_loader.load("learning_rate")
    )

    transfer_net.to(DEVICE)
    style_image.to(DEVICE)

    train(transfer_net, loss_net, dataloader, optimizer, json_loader.load("epochs"))


if __name__ == "__main__":
    main()
