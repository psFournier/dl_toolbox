import os
import numpy as np
import rasterio
import torch

from rasterio.windows import Window
from torch.utils.data import Dataset

import dl_toolbox.transforms as transforms
from dl_toolbox.utils import merge_labels, get_tiles


class TiledRasterDataset(Dataset):
    
    classes = None

    def __init__(
        self,
        img,
        msk,
        bands,
        merge,
        transforms,
        crop_size,
        window,
        crop_step
    ):
        self.img = img
        self.msk = msk
        self.bands = bands
        self.class_list = self.classes[merge].value
        self.merges = [list(l.values) for l in self.class_list]
        self.transforms = transforms
        col_off, row_off, width, height = window
        self.crops = list(get_tiles(
                nols=width,
                nrows=height,
                width=crop_size,
                step_w=crop_step,
                row_offset=row_off,
                col_offset=col_off,
                cover_all=True
        ))

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        crop = self.crops[idx]
        window = Window(*crop)
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
            "label": label,
            "crop": crop
        }
