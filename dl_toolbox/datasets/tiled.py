import os

import imagesize

import matplotlib.colors as colors
import numpy as np
import rasterio
import torch

from rasterio.windows import Window
from torch.utils.data import Dataset

import dl_toolbox.transforms as transforms
from dl_toolbox.utils import get_tiles
from .utils import read_image, read_label


class TiledTif(Dataset):

    def __init__(
        self,
        img_path,
        label_path,
        window,
        bands,
        merge,
        crop_size,
        crop_step,
        transforms
    ):
        self.img_path = img_path
        self.label_path = label_path
        self.bands = bands
        self.class_list = self.classes[merge].value
        self.crop_size = crop_size
        self.crop_step = crop_step
        self.transforms = transforms
        col_off, row_off, width, height = window
        self.crops = list(get_tiles(
                nols=width,
                nrows=height,
                width=crop_size,
                step_w=crop_step,
                row_offset=row_off,
                col_offset=col_off
        ))

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        window = Window(*self.crops[idx])
        image = read_image(self.img_path, window, self.bands)            
        label = None
        if self.label_path:
            label = read_label(self.label_path, window, self.class_list)
        image, label = self.transforms(img=image, label=label)
        return {
            "image": image,
            "label": label,
            "crop": crop
        }
