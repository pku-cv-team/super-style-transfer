"""模型工具函数"""

import torch


def compute_gama_matrix(features: torch.Tensor) -> torch.Tensor:
    """计算Gram矩阵

    Args:
        features: 特征，shape: (n, c, h, w)

    Returns:
        torch.Tensor: Gram矩阵，shape: (n, c, c)
    """
    n, c, h, w = features.shape
    features = features.view(n, c, h * w)
    gama_matrix = torch.bmm(features, features.transpose(1, 2))
    return gama_matrix


# pylint: disable=too-many-arguments, too-many-positional-arguments
def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    epoch: int,
    loss: float,
    checkpoint_path: str,
):
    """保存检查点"""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": loss,
        },
        checkpoint_path,
    )


def save_model(model: torch.nn.Module, model_path: str):
    """保存模型"""
    torch.save(model.state_dict(), model_path)


# pylint: disable=too-many-arguments, too-many-positional-arguments
def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    checkpoint_path: str,
):
    """加载检查点"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    return epoch, loss


def load_model(model: torch.nn.Module, model_path: str):
    """加载模型"""
    model.load_state_dict(torch.load(model_path))
    return model
