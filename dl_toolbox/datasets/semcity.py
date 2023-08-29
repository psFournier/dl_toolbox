import numpy as np
import rasterio
import torch
import enum

from torch.utils.data import Dataset
from rasterio.windows import Window
from dl_toolbox.utils import label, get_tiles, merge_labels


all7 = [
    label("void", (255, 255, 255), {0}),
    label("impervious surface", (38, 38, 38), {1}),
    label("building", (238, 118, 33), {2}),
    label("pervious surface", (34, 139, 34), {3}),
    label("high vegetation", (0, 222, 137), {4}),
    label("car", (255, 0, 0), {5}),
    label("water", (0, 0, 238), {6}),
    label("sport venue", (160, 30, 230), {7}),
]

classes = enum.Enum(
    "SemcityClasses",
    {
        "all7": all7,
    },
)


class SemcityDataset(Dataset):
    
    classes = classes
    
    def __init__(
        self,
        imgs,
        msks,
        bands,
        merge,
        transforms,
        crop_size,
        window,
        crop_step
    ):
        self.imgs = imgs
        self.msks = msks
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
        return len(self.crops) * len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx // len(self.crops)]
        crop = self.crops[idx % len(self.crops)]
        window = Window(*crop)
        with rasterio.open(img, "r") as file:
            image = file.read(window=window, out_dtype=np.float32, indexes=self.bands)
        image = torch.from_numpy(image)
        label = None
        if self.msks:
            msk = self.msks[idx // len(self.crops)]
            with rasterio.open(msk, "r") as file:
                label = file.read(window=window, out_dtype=np.uint8)
            label = merge_labels(label, self.merges)
            label = torch.from_numpy(label).long()
        image, label = self.transforms(img=image, label=label)
        return {
            "image": image,
            "label": None if label is None else label.squeeze()
        }