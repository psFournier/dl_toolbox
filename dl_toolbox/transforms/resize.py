import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
import random

class Resize:
    def __init__(self, sizes):
        self.sizes = sizes

    def __call__(self, img, label=None):
        size = random.choice(self.sizes)
        img = F.resize(img, size=size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        if label is not None:
            label = F.resize(label, size=size, interpolation=InterpolationMode.NEAREST, antialias=False)

        return img, label
