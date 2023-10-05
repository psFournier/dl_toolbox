import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
import random

class Resize:
    def __init__(self, factors):
        self.factors = factors

    def __call__(self, img, label=None):
        factor = random.choice(self.factors)
        size = (int(factor*img.shape[-2]), int(factor*img.shape[-1]))
        img = F.resize(img, size=size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        if label is not None:
            label = F.resize(label, size=size, interpolation=InterpolationMode.NEAREST, antialias=False)

        return img, label
