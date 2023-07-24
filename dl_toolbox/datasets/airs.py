import enum
from collections import namedtuple
from dataclasses import dataclass

import numpy as np
import rasterio

import torch


label = namedtuple("label", ["name", "color", "values"])

AirsNomenclatures = enum.Enum(
    "AirsNomenclatures",
    {
        "building": [
            label("other", (0, 0, 0), {0}),
            label("building", (255, 255, 255), {1}),
        ],
    },
)


@dataclass
class Airs:
    image_path: ... = None
    zone: ... = None
    mins: ... = None
    maxs: ... = None
    label_path: ... = None

    def __post_init__(self):
        with rasterio.open(self.image_path) as src:
            self.meta = src.meta

    def read_image(self, window=None, bands=None):
        with rasterio.open(self.image_path, "r") as file:
            image = file.read(window=window, out_dtype=np.float32, indexes=bands)

        return image

    def read_label(self, window=None):
        with rasterio.open(self.label_path) as file:
            label = file.read(window=window, out_dtype=np.float32)
            label /= 255.0

        return label
