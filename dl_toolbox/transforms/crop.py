import torchvision.transforms.functional as F
from torchvision.transforms import RandomCrop


class RandomCrop2(RandomCrop):
    def __init__(self, size):
        super(RandomCrop2, self).__init__(size, padding=None, pad_if_needed=False)

    def __call__(self, img, label=None):
        i, j, h, w = self.get_params(img, self.size)
        img = F.crop(img, i, j, h, w)
        if label is not None:
            label = F.crop(label, i, j, h, w)
        return img, label
