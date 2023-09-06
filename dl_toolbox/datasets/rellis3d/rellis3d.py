import enum
import numpy as np
from PIL import Image
import torch

from torch.utils.data import Dataset

import dl_toolbox.transforms as transforms
from dl_toolbox.utils import merge_labels, label

onto0 = {0: 'void', 1: 'dirt', 3: 'grass', 4: 'tree', 5: 'pole', 6: 'water', 7: 'sky', 8: 'vehicle', 9: 'object', 10: 'asphalt', 12: 'building', 15: 'log', 17: 'person', 18: 'fence', 19: 'bush', 23: 'concrete', 27: 'barrier', 31: 'puddle', 33: 'mud', 34: 'rubble'}

onto1 = {0: [0, 0, 0], 1: [108, 64, 20], 3: [0, 102, 0], 4: [0, 255, 0], 5: [0, 153, 153], 6: [0, 128, 255], 7: [0, 0, 255], 8: [255, 255, 0], 9: [255, 0, 127], 10: [64, 64, 64], 12: [255, 0, 0], 15: [102, 0, 0], 17: [204, 153, 255], 18: [102, 0, 204], 19: [255, 153, 204], 23: [170, 170, 170], 27: [41, 121, 255], 31: [134, 255, 239], 33: [99, 66, 34], 34: [110, 22, 138]}

all19 = [label(v, tuple(onto1[k]), {k}) for k, v in onto0.items()]

classes = enum.Enum(
    "Rellis3dClasses",
    {
        "all19": all19,
    },
)

class Rellis3d(Dataset):
    
    classes = classes

    def __init__(self, imgs, msks, merge, transforms):
        self.imgs = imgs
        self.msks = msks
        self.class_list = self.classes[merge].value
        self.merges = [list(l.values) for l in self.class_list]
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        with open(self.imgs[idx], "rb") as f:
            image = np.array(Image.open(f))
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        label = None
        if self.msks:
            with open(self.msks[idx], "rb") as f:
                label = np.array(Image.open(f))
            label = merge_labels(label, self.merges)
            label = torch.from_numpy(label).long()
        image, label = self.transforms(img=image, label=label)
        return {
            "image": image,
            "label": None if label is None else label.squeeze()
        }