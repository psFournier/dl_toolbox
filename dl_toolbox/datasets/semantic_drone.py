import enum
import numpy as np
from PIL import Image
import torch

from torch.utils.data import Dataset

import dl_toolbox.transforms as transforms
from dl_toolbox.utils import merge_labels, label

class_list = [
    ['unlabeled', (0, 0, 0), {0}],
    ['paved-area', (128, 64, 128), {1}],
    ['dirt', (130, 76, 0), {2}],
    ['grass', (0, 102, 0), {3}],
    ['gravel', (112, 103, 87), {4}],
    ['water', (28, 42, 168), {5}],
    ['rocks', (48, 41, 30), {6}],
    ['pool', (0, 50, 89), {7}],
    ['vegetation', (107, 142, 35), {8}],
    ['roof', (70, 70, 70), {9}],
    ['wall', (102, 102, 156), {10}],
    ['window', (254, 228, 12), {11}],
    ['door', (254, 148, 12), {12}],
    ['fence', (190, 153, 153), {13}],
    ['fence-pole', (153, 153, 153), {14}],
    ['person', (255, 22, 96), {15}],
    ['dog', (102, 51, 0), {16}],
    ['car', (9, 143, 150), {17}],
    ['bicycle', (119, 11, 32), {18}],
    ['tree', (51, 51, 0), {19}],
    ['bald-tree', (190, 250, 190), {20}],
    ['ar-marker', (112, 150, 146), {21}],
    ['obstacle', (2, 135, 115), {22}],
    ['conflicting', (255, 0, 0), {23}],
]

all23 = [label(*l) for l in class_list]

classes = enum.Enum(
    "SemanticDroneClasses",
    {
        "all23": all23,
    },
)

class SemanticDroneDataset(Dataset):
    
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
            image = Image.open(f)
            image = np.array(image.crop((200,200,700,700)))
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.
        label = None
        if self.msks:
            with open(self.msks[idx], "rb") as f:
                label = Image.open(f)
                label = np.array(label.crop((200,200,700,700)))
            label = merge_labels(label, self.merges)
            label = torch.from_numpy(label).long()
        image, label = self.transforms(img=image, label=label)
        return {
            "image": image,
            "label": None if label is None else label.squeeze()
        }