import enum
import random
from collections import namedtuple

import rasterio
import numpy as np
import torch
import rasterio.windows as windows

from dl_toolbox.utils import get_tiles, merge_labels


label = namedtuple("label", ["name", "color", "values"])


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
    return torch.from_numpy(label).long()

class FixedCropFromWindow:
    def __init__(self, window, crop_size, crop_step=None):
        self.window = window
        self.crop_size = crop_size
        self.crop_step = crop_step

        self.crops = [
            crop
            for crop in get_tiles(
                nols=window.width,
                nrows=window.height,
                size=crop_size,
                step=crop_step if crop_step else crop_size,
                row_offset=window.row_off,
                col_offset=window.col_off,
            )
        ]

    def __call__(self, idx):
        return self.crops[idx]


class RandomCropFromWindow:
    def __init__(self, window, crop_size):
        self.window = window
        self.crop_size = crop_size

    def __call__(self, idx):
        col_off, row_off, width, height = self.window.flatten()
        cx = col_off + random.randint(0, width - self.crop_size)
        cy = row_off + random.randint(0, height - self.crop_size)
        crop = windows.Window(cx, cy, self.crop_size, self.crop_size)

        return crop
