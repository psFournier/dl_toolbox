import torch
import torchvision.transforms.functional as F
from .utils import OneOf, NoOp


class Gamma(torch.nn.Module):
    def __init__(self, bounds=(0.9, 0.9), p=0.5):
        super().__init__()
        self.bounds = bounds
        self.p = p

    def apply(self, img, label=None, factor=1.0):
        return F.adjust_gamma(img, factor), label

    def forward(self, img, label=None):
        if torch.rand(1).item() < self.p:
            factor = float(torch.empty(1).uniform_(self.bounds[0], self.bounds[1]))
            return self.apply(img, label, factor)

        return img, label


class Saturation(torch.nn.Module):
    def __init__(self, bounds=(0.9, 0.9), p=0.5):
        super().__init__()
        self.bounds = bounds
        self.p = p

    def forward(self, img, label=None):
        factor = float(torch.empty(1).uniform_(self.bounds[0], self.bounds[1]))
        if torch.rand(1).item() < self.p:
            return F.adjust_saturation(img, factor), label
        return img, label


class Brightness(torch.nn.Module):
    def __init__(self, bounds=(0.9, 0.9), p=0.5):
        super().__init__()
        self.bounds = bounds
        self.p = p

    def forward(self, img, label=None):
        factor = float(torch.empty(1).uniform_(self.bounds[0], self.bounds[1]))
        if torch.rand(1).item() < self.p:
            return F.adjust_brightness(img, factor), label
        return img, label


class Contrast(torch.nn.Module):
    def __init__(self, bounds=(0.9, 0.9), p=0.5):
        super().__init__()
        self.bounds = bounds
        self.p = p

    def forward(self, img, label=None):
        factor = float(torch.empty(1).uniform_(self.bounds[0], self.bounds[1]))
        if torch.rand(1).item() < self.p:
            return F.adjust_contrast(img, factor), label
        return img, label


class Color:
    def __init__(self, bounds):
        self.color = OneOf(
            [
                NoOp(),
                Saturation(p=1, bounds=bounds),
                Contrast(p=1, bounds=bounds),
                Gamma(p=1, bounds=bounds),
                Brightness(p=1, bounds=bounds),
            ],
            transforms_ps=[1, 1, 1, 1, 1],
        )

    def __call__(self, img, label=None):
        return self.color(img, label)
