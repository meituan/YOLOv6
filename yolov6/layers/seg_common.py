import warnings
import torch
import torch.nn as nn


class SegReLU(nn.Module):
    """ Activation of SegReLU """
    @staticmethod
    def forward(x):
        return torch.where(x > 0, x, torch.nn.Softsign()(x))


class SegConv(nn.Module):
    """Normal Conv with Segrelu activation"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = SegReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class SegConvWrapper(nn.Module):
    """Wrapper for normal Conv with SegReLU activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, bias=True):
        super().__init__()
        self.block = SegConv(in_channels, out_channels, kernel_size, stride, groups, bias)

    def forward(self, x):
        return self.block(x)


class SegSPPF(nn.Module):
    """Simplified SPPF with SegReLU activation"""
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        c_ = in_channels // 2  # hidden channels
        self.cv1 = SegConv(in_channels, c_, 1, 1)
        self.cv2 = SegConv(c_ * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


