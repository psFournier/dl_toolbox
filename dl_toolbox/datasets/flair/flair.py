import enum
import os
import random

import matplotlib.colors as colors
import numpy as np
import rasterio

import torch
from torch.utils.data import Dataset

import dl_toolbox.transforms as transforms
from dl_toolbox.utils import merge_labels, label
from torchvision import tv_tensors
import torchvision.transforms.v2 as v2 


lut_colors = {
    1: "#db0e9a",
    2: "#938e7b",
    3: "#f80c00",
    4: "#a97101",
    5: "#1553ae",
    6: "#194a26",
    7: "#46e483",
    8: "#f3a60d",
    9: "#660082",
    10: "#55ff00",
    11: "#fff30d",
    12: "#e4df7c",
    13: "#3de6eb",
    14: "#ffffff",
    15: "#8ab3a0",
    16: "#6b714f",
    17: "#c5dc42",
    18: "#9999ff",
    19: "#000000",
}

lut_classes = {
    1: "building",
    2: "pervious",
    3: "impervious",
    4: "bare soil",
    5: "water",
    6: "coniferous",
    7: "deciduous",
    8: "brushwood",
    9: "vineyard",
    10: "herbaceous",
    11: "agricultural",
    12: "plowed land",
    13: "swimmingpool",
    14: "snow",
    15: "clear cut",
    16: "mixed",
    17: "ligneous",
    18: "greenhouse",
    19: "other",
}


def hex2color(hex):
    return tuple([int(z * 255) for z in colors.hex2color(hex)])


all19 = [label(lut_classes[i], hex2color(lut_colors[i]), {i}) for i in range(1, 20)]

main13 = [label("other", (0, 0, 0), {13, 14, 15, 16, 17, 18, 19})] + [
    label(lut_classes[i], hex2color(lut_colors[i]), {i}) for i in range(1, 13)
]

hierarchical6 = [
    label("other", (0,0,0), {19}),
    label("anthropized", hex2color(lut_colors[2]), {1,2,3,13,18}),
    label("natural", hex2color(lut_colors[4]), {4,5,14}),
    label("woody vegetation", hex2color(lut_colors[7]), {6,7,8,15,16,17}),
    label("agricultural", hex2color(lut_colors[11]), {9,11,12}),
    label("herbaceous", hex2color(lut_colors[10]), {10}),
]

classes = enum.Enum(
    "FlairClasses",
    {
        "all19": all19,
        "main13": main13,
        "hierarchical6": hierarchical6
    },
)

class Flair(Dataset):
    
    classes = classes

    def __init__(self, imgs, msks, bands, merge, transforms):
        self.imgs = imgs
        self.msks = msks
        self.bands = bands
        self.class_list = self.classes[merge].value
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
        with rasterio.open(self.imgs[idx], "r") as file:
            image = file.read(out_dtype=np.uint8, indexes=self.bands)
        image = tv_tensors.Image(torch.from_numpy(image))
        target = None
        if self.msks:
            target = {}
            with rasterio.open(self.msks[idx], "r") as file:
                mask = file.read(out_dtype=np.uint8)
            mask = merge_labels(torch.from_numpy(mask), self.merges) 
            target['masks'] = tv_tensors.Mask(mask)
        image, target = self.transforms(image, target)
        if self.msks:
            target['masks'] = target['masks'].squeeze()
        return image, target