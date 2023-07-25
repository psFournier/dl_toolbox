import os

import imagesize

import matplotlib.colors as colors
import numpy as np
import rasterio
import torch

from rasterio.windows import Window
from torch.utils.data import Dataset

import dl_toolbox.transforms as transforms
from dl_toolbox.utils import merge_labels
from dl_toolbox.datasets.utils import *
from dl_toolbox.utils import get_tiles


def read_image(path, window=None, bands=None):
    with rasterio.open(path, "r") as file:
        image = file.read(window=window, out_dtype=np.float32, indexes=bands)
    return torch.from_numpy(image)

def read_label(path, window=None, classes=None):
    with rasterio.open(path, "r") as file:
        label = file.read(window=window, out_dtype=np.uint8)
    if classes is not None:
        label = merge_labels(
            label.squeeze(), [list(l.values) for l in classes]
        )
    return torch.from_numpy(label)

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
