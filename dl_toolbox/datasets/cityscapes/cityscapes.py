import enum
import numpy as np
import torch
from torchvision.io import read_image
from torchvision import tv_tensors
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2 
from torchvision import tv_tensors
from torchvision.datasets import Cityscapes as tv_cityscapes

from dl_toolbox.utils import label, merge_labels

void_cls = {c.id for c in tv_cityscapes.classes if c.train_id in {255, -1}}
all19 = [label("void", (0, 0, 0), void_cls)] + [
    label(c.name, c.color, {c.id}) for c in tv_cityscapes.classes if c.train_id not in {255, -1}
]

class Cityscapes(Dataset):
    # https://pytorch.org/vision/0.15/_modules/torchvision/datasets/cityscapes.html#Cityscapes
    
    all_class_lists = enum.Enum(
        "CityscapesClasses",
        {
            "all19": all19,
        },
    )

    def __init__(self, imgs, msks, merge, transforms):
        self.imgs = imgs
        self.msks = msks
        self.class_list = self.all_class_lists[merge].value
        self.merges = [list(l.values) for l in self.class_list]
        self.transforms = v2.ToDtype(dtype={
            tv_tensors.Image: torch.float32,
            tv_tensors.Mask: torch.int64,
            "others":None
        }, scale=True)
        if transforms is not None:
            self.transforms = v2.Compose([self.transforms, transforms])
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image_path = self.imgs[idx]
        image = tv_tensors.Image(read_image(image_path))
        target = None
        if self.msks:
            mask = read_image(self.msks[idx])
            mask = merge_labels(mask, self.merges)
            target = tv_tensors.Mask(mask)
        image, target = self.transforms(image, target)
        if self.msks:
            target = target.squeeze()
        return {'image':image, 'target':target, 'image_path':image_path}