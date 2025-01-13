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
from style_transfer.data import read_img_to_tensor, img_tensor_to_pil
from style_transfer.models.neural_style_transfer_creater import (
    create_style_transfer_model,
)

# pylint: disable=unused-import
# 似乎train函数中使用了但是似乎没办法正确检查到，pylint检查时会标记未被引用
from style_transfer.utils.model_utils import save_checkpoint

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH: str = "experiments/models/model.pth"
CHECKPOINT_DIR: str = "experiments/checkpoints"


# pylint: disable=too-many-arguments, too-many-positional-arguments
def train(
    transfer_net: TransferNet,
    loss_net: LossNet,
    dataloader: DataLoader,
    optimizer: torch.optim,
    config: dict,
    scheduler: torch.optim.lr_scheduler,
):
    """训练风格迁移网络"""
    torch.autograd.set_detect_anomaly(True)
    best_loss = float("inf")
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    early_stop_counter = 0
    # writer = SummaryWriter(config["log_dir"])
    for epoch in range(config["epochs"]):
        for i, data in enumerate(dataloader):
            data = data.to(DEVICE)
            optimizer.zero_grad()

            generated_image = transfer_net(data)
            loss = loss_net.compute_loss(generated_image, data)
            print(f"epoch: {epoch}, iteration: {i + 1}, loss: {loss.item()}")

            loss.backward()
            optimizer.step()
            scheduler.step()

            if loss.item() < best_loss:
                early_stop_counter = 0
                best_loss = loss.item()
                torch.save(transfer_net.state_dict(), MODEL_PATH)
            else:
                early_stop_counter += 1
            # pylint: disable=condition-evals-to-constant
            # 暂时禁用早停
            if early_stop_counter >= config["patience"] and False:
                print(f"Early stopping at epoch {epoch}, iteration {i + 1}")
                return  # 早停，如果训练的loss一直无法降低到达patience的阈值，那么就停止训练

            if i % config["checkpoint_fre"] == 1:
                save_checkpoint(
                    transfer_net,
                    optimizer,
                    scheduler,
                    epoch,
                    loss.item(),
                    f"{CHECKPOINT_DIR}/checkpoint_{epoch}_{i + 1}.pth",
                )
            # writer.add_scalar(
            #     "Loss/train", loss.item(), epoch * len(dataloader) + i
            # )
        scheduler.step()


def load_dataset(json_loader):
    """加载数据集"""
    root_dir: str = json_loader.load("root_dir")
    ann_file: str = json_loader.load("ann_file")
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop((256, 256)),
            transforms.ColorJitter(
                brightness=0.7, contrast=0.5, saturation=0.5, hue=0.5
            ),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    dataset: CocoDataset = CocoDataset(
        root=root_dir, annFile=ann_file, transform=transform
    )
    return dataset


def create_dataloader(dataset, json_loader):
    """创建数据加载器"""
    batch_size: int = json_loader.load("batch_size")
    dataloader: DataLoader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    return dataloader


def load_style_image(json_loader, transfrom, batch_size):
    """加载风格图像"""
    style_image: torch.Tensor = read_img_to_tensor(json_loader.load("style_image_path"))
    style_image = (
        transfrom(img_tensor_to_pil(style_image))
        .unsqueeze(0)
        .repeat(batch_size, 1, 1, 1)
    )
    return style_image


def create_loss_net(json_loader, style_image):
    """创建损失网络"""
    loss_net = LossNet(
        lambda content_image: create_style_transfer_model(
            json_loader.load_style_transfer_param()
        )(content_image, style_image)
    )
    return loss_net


def create_optimizer(transfer_net: TransferNet, json_loader: JsonLoader):
    """创建优化器"""
    optimizer = torch.optim.AdamW(
        transfer_net.parameters(), lr=json_loader.load("learning_rate")
    )
    return optimizer


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
    dataset = load_dataset(json_loader)
    dataloader = create_dataloader(dataset, json_loader)
    # 处理图像
    style_image = load_style_image(
        json_loader, dataset.transform, json_loader.load("batch_size")
    )
    # 网络构建
    loss_net = create_loss_net(json_loader, style_image)
    transfer_net = TransferNet()
    # 优化器构建(?使用AdamW?还是其它优化器呢)
    # pylint: disable=no-member
    # pylint 似乎不能正确识别优化器，这里禁用 no-member 检查
    optimizer = create_optimizer(transfer_net, json_loader)

    transfer_net.to(DEVICE)
    style_image.to(DEVICE)

    config_state = {
        "epochs": json_loader.load("epochs"),
        "checkpoint_fre": 1000,
        "patience": 10,
        "log_dir": "logs",
    }

    scheduler: torch.optim.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, json_loader.load("epochs")
    )

    train(transfer_net, loss_net, dataloader, optimizer, config_state, scheduler)


if __name__ == "__main__":
    main()
