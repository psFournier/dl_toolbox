from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, first_stride=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, stride=first_stride, bias=False),
            nn.InstanceNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, max_pool=False):
        super().__init__()
        if not max_pool:
            self.down = DoubleConv(in_channels, out_channels, first_stride=2)
        else:
            self.down = nn.Sequential(
                nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
            )

    def forward(self, x):
        return self.down(x)



class Vgg(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        max_pool=False, 
        *args, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_pool = max_pool
        factor=1  #see Unet
        self.inc = DoubleConv(in_channels, 32)
        self.down0 = Down(32, 64, max_pool)
        self.down1 = Down(64, 128, max_pool)
        self.down2 = Down(128, 256, max_pool)
        self.down3 = Down(256, 512 // factor, max_pool)
        # travail en cours
        self.inc = DoubleConv(in_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 256)
        self.down5 = Down(256, 16)
        self.fc1 = nn.Linear(16 * 8 * 8, 512)
        self.pred = nn.Linear(512, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        fc1 = self.relu(self.fc1(x6.view(-1, 16 * 8 * 8)))
        return self.pred(fc1)
