from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class EncodeThenUpsample(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(JustEfficientnet, self).__init__()
        self.in_channels = in_channels
        self.f = torchvision.models.efficientnet_v2_s().features
        # todo changer la couche d'entrée pour prendre 5chs
        self.classif = torch.nn.Conv2d(1280, out_channels, kernel_size=1)

    def forward(self, x):
        _, _, h, w = x.shape
        x = ((x / 255) - 0.5) / 0.25
        x = self.f(x)
        x = self.classif(x)
        x = torch.nn.functional.interpolate(x, size=(h, w), mode="bilinear")
        return x
