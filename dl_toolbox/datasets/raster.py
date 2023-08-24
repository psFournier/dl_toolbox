import os
import random
import matplotlib.colors as colors
import numpy as np
import rasterio

import torch
from rasterio.windows import Window
from torch.utils.data import Dataset

import dl_toolbox.transforms as transforms
from dl_toolbox.utils import merge_labels
import imagesize


class RasterDataset(Dataset):
    
    classes = None

    def __init__(
        self,
        img,
        msk,
        bands,
        merge,
        transforms,
        crop_size
    ):
        self.img = img
        self.msk = msk
        self.bands = bands
        self.class_list = self.classes[merge].value
        self.crop_size = crop_size
        self.transforms = transforms
        self.w, self.h = imagesize.get(img)
        self.merges = [list(l.values) for l in self.class_list]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        cx = random.randint(0, self.w - self.crop_size)
        cy = random.randint(0, self.h - self.crop_size)
        window = Window(cx, cy, self.crop_size, self.crop_size)
        with rasterio.open(self.img, "r") as file:
            image = file.read(window=window, out_dtype=np.float32, indexes=self.bands)
        image = torch.from_numpy(image)
        label = None
        if self.msk:
            with rasterio.open(self.msk, "r") as file:
                label = file.read(window=window, out_dtype=np.uint8)
            label = merge_labels(label, self.merges)
            label = torch.from_numpy(label).long()
        image, label = self.transforms(img=image, label=label)
        return {
            "image": image,
            "label": None if label is None else label.squeeze()
        }
