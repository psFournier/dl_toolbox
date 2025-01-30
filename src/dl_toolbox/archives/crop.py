import torchvision.transforms.functional as F
from torchvision.transforms import RandomCrop, RandomResizedCrop, InterpolationMode, Pad

class PadSymmetric(Pad):
    def __init__(self, ltrb):
        l,t,r,b = list(ltrb)
        super().__init__(padding=(l,t,r,b), fill=0, padding_mode='symmetric') #left top right bottom

    def __call__(self, img, label=None):
        img = self.forward(img)
        if label is not None and label.dim() > 2:
            label = self.forward(label)
        return img, label
    
class RemovePad:
    def __init__(self, ltrb):
        self.ltrb = ltrb #left top right bottom

    def __call__(self, img, label=None):
        imgh,imgw = img.shape[-2:]
        l,t,r,b = list(self.ltrb)
        img = F.crop(img, t, l, imgh-t-b, imgw-l-r)
        if label is not None and label.dim() > 2:
            label = F.crop(label, t, l, imgh-t-b, imgw-l-r)
        return img, label

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
    
