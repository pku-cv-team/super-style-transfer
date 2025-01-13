"""风格迁移网络"""

import torch
from torch import nn


# pylint: disable=too-many-instance-attributes, too-few-public-methods
class TransferNet(nn.Module):
    """风格迁移网络"""

    def __init__(self):
        super().__init__()
        self.downsample1 = DownSample(3, 32, 9, 1)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.downsample2 = DownSample(32, 64, 3, 2)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.downsample3 = DownSample(64, 128, 3, 2)
        self.in3 = nn.InstanceNorm2d(128, affine=True)
        self.residual1 = ResidualBlock(128)
        self.residual2 = ResidualBlock(128)
        self.residual3 = ResidualBlock(128)
        self.residual4 = ResidualBlock(128)
        self.residual5 = ResidualBlock(128)
        self.upsample1 = UpSample(128, 64, 3, 1, 2)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        self.upsample2 = UpSample(64, 32, 3, 1, 2)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        self.upsample3 = DownSample(32, 3, 9, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        out = self.relu(self.in1(self.downsample1(x)))
        out = self.relu(self.in2(self.downsample2(out)))
        out = self.relu(self.in3(self.downsample3(out)))
        out = self.residual1(out)
        out = self.residual2(out)
        out = self.residual3(out)
        out = self.residual4(out)
        out = self.residual5(out)
        out = self.relu(self.in4(self.upsample1(out)))
        out = self.relu(self.in5(self.upsample2(out)))
        out = self.upsample3(out)
        return out


# pylint: disable=too-few-public-methods
class DownSample(nn.Module):
    """下采样层"""

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.padding = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.conv(self.padding(x))


# pylint: disable=too-few-public-methods
class UpSample(nn.Module):
    """上采样层"""

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super().__init__()
        self.upsample = upsample
        self.padding = nn.ReflectionPad2d(kernel_size // 2)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(
                x_in, mode="nearest", scale_factor=self.upsample
            )
        return self.conv2d(self.padding(x_in))


# pylint: disable=too-few-public-methods
class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, channels):
        super().__init__()
        self.padding1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(channels, channels, 3, 1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()
        self.padding2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        residual = x
        out = self.relu(self.in1(self.conv1(self.padding1(x))))
        out = self.in2(self.conv2(self.padding2(out)))
        out = out + residual
        return out
