import numpy as np
import rasterio
import torch
import enum
import random
from torch.utils.data import Dataset

from dl_toolbox.utils import label, get_tiles, merge_labels
import rasterio.windows as W
from fiona.transform import transform as f_transform
import rasterio.warp
from shapely.geometry import box
from rasterio.enums import Resampling

all9 = [
    label("nodata", (250, 250, 250), {0}),
    label("bare_ground", (100, 50, 0), {1}),
    label("low_vegetation", (0, 250, 50), {2}),
    label("water", (0, 50, 250), {3}),
    label("building", (250, 50, 50), {4}),
    label("high_vegetation", (0, 100, 50), {5}),
    label("parking", (200, 200, 200), {6}),
    label("road", (100, 100, 100), {7}),
    label("railways", (200, 100, 200), {8}),
    label("swimmingpool", (50, 150, 250), {9}),
]

main6 = [
    label("other", (0, 0, 0), {0, 1, 6, 9}),
    label("low vegetation", (0, 250, 50), {2}),
    label("high vegetation", (0, 100, 50), {5}),
    label("water", (0, 50, 250), {3}),
    label("building", (250, 50, 50), {4}),
    label("road", (100, 100, 100), {7}),
    label("railways", (200, 100, 200), {8}),
]


classes = enum.Enum(
    "Digitanie9Classes",
    {
        "all9": all9,
        "main6": main6
    }
)

class Digitanie(Dataset):
    
    classes = classes

    def __init__(
        self,
        imgs,
        msks,
        bands,
        merge,
        transforms,
    ):
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
            image = file.read(
                out_dtype=np.float32,
                indexes=self.bands
            )
        image = torch.from_numpy(image)
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