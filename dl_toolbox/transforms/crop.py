import torchvision.transforms.functional as F
from torchvision.transforms import RandomCrop, RandomResizedCrop, InterpolationMode


class RandomCrop2(RandomCrop):
    def __init__(self, size):
        super(RandomCrop2, self).__init__(size, padding=None, pad_if_needed=False)

    def __call__(self, img, label=None):
        i, j, h, w = self.get_params(img, self.size)
        img = F.crop(img, i, j, h, w)
        if label is not None and label.dim() > 2:
            label = F.crop(label, i, j, h, w)
        return img, label
    
class RandomResizedCrop(RandomResizedCrop):
    def __init__(self, size, scale, ratio):
        super().__init__(size, scale, ratio)

    def __call__(self, img, label=None):
        params = self.get_params(img, scale=self.scale, ratio=self.ratio)
        img = F.resized_crop(img, *params, size=self.size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        if label is not None and label.dim() > 2:
            label = F.resized_crop(label, *params, size=self.size, interpolation=InterpolationMode.NEAREST, antialias=True)
        return img, label
    
