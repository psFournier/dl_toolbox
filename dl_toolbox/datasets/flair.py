import enum
import os
import random

import matplotlib.colors as colors
import numpy as np
import rasterio

import torch
from rasterio.windows import Window
from torch.utils.data import Dataset

import dl_toolbox.transforms as transforms
from dl_toolbox.datasets import TiledRasterDataset
from dl_toolbox.utils import merge_labels, label


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

class DatasetFlairTiled(TiledRasterDataset):
    classes = classes

class DatasetFlair2(Dataset):
    classes = classes

    def __init__(self, imgs, msks, bands, merge, transforms):
        self.imgs = imgs
        self.msks = msks
        self.bands = bands
        self.class_list = self.classes[merge].value
        self.merges = [list(l.values) for l in self.class_list]
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        with rasterio.open(self.imgs[idx], "r") as file:
            image = file.read(out_dtype=np.float32, indexes=self.bands)
        image = torch.from_numpy(image) / 255.
        label = None
        if self.msks:
            with rasterio.open(self.msks[idx], "r") as file:
                label = file.read(out_dtype=np.uint8)
            label = merge_labels(label, self.merges) 
            label = torch.from_numpy(label).long()
        image, label = self.transforms(img=image, label=label)
        return {
            "image": image,
            "label": None if label is None else label.squeeze()
        }


#    def __len__(self):
#        return len(self.list_imgs)
#
#    def __getitem__(self, index):
#
#        image_file = self.list_imgs[index]
#
#        cx = np.random.randint(0, 512 - self.crop_size + 1)
#        cy = np.random.randint(0, 512 - self.crop_size + 1)
#        window = Window(cx, cy, self.crop_size, self.crop_size)
#        image = self.read_image(image_file, window)
#        image = torch.from_numpy(image).float().contiguous()
#
#        label = None
#        if self.list_msks.size: # a changer
#            mask_file = self.list_msks[index]
#            label = self.read_label(mask_file, window)
#            label = torch.from_numpy(label).long().contiguous()
#
#        if self.img_aug is not None:
#            end_image, end_mask = self.img_aug(img=image, label=label)
#        else:
#            end_image, end_mask = image, label
#
#        # todo : gerer metadata
#
#        return {
#            #'orig_image':image,
#            #'orig_mask':label,
#            'image':end_image,
#            'label':end_mask,
#            'path': '/'.join(image_file.split('/')[-4:])
#        }

#        if self.use_metadata == True:
#            mtd = self.list_metadata[index]
#            return {"img": torch.as_tensor(img, dtype=torch.float),
#                    "mtd": torch.as_tensor(mtd, dtype=torch.float),
#                    "msk": torch.as_tensor(msk, dtype=torch.float)}

# class Predict_Dataset(Dataset):
#
#    def __init__(self,
#                 dict_files,
#                 num_classes=13, use_metadata=True
#                 ):
#        self.list_imgs = np.array(dict_files["IMG"])
#        self.num_classes = num_classes
#        self.use_metadata = use_metadata
#        if use_metadata == True:
#            self.list_metadata = np.array(dict_files["MTD"])
#
#    def read_img(self, raster_file: str) -> np.ndarray:
#        with rasterio.open(raster_file) as src_img:
#            array = src_img.read()
#            return array
#
#    def __len__(self):
#        return len(self.list_imgs)
#
#    def __getitem__(self, index):
#        image_file = self.list_imgs[index]
#        img = self.read_img(raster_file=image_file)
#        img = img_as_float(img)
#
#        if self.use_metadata == True:
#            mtd = self.list_metadata[index]
#            return {"img": torch.as_tensor(img, dtype=torch.float),
#                    "mtd": torch.as_tensor(mtd, dtype=torch.float),
#                    "id": '/'.join(image_file.split('/')[-4:])}
#        else:
#
#            return {"img": torch.as_tensor(img, dtype=torch.float),
#                    "id": '/'.join(image_file.split('/')[-4:])}
