import numpy as np
import rasterio
import torch
import enum
import random
from torch.utils.data import Dataset
from rasterio.windows import Window

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

_all43 = [
    label("nodata", (250, 250, 250), None),
    label("construction_site", (140, 90, 100), None),
    label("bare_ground", (71, 58, 17), None),
    label("bare_parking", (160, 130, 105), None),
    label("bare_road", (160, 119, 34), None),
    label("bare_pedestrian", (230, 167, 31), None),
    label("sand", (234, 220, 90), None),
    label("snow", (240, 239, 220), None),
    label("field", (235, 255, 6), None),
    label("sport_vegetation", (190, 215, 165), None),
    label("grassland", (140, 240, 118), None),
    label("aquaculture", (11, 222, 189), None),
    label("hedge", (119, 211, 0), None),
    label("shrub", (113, 184, 48), None),
    label("vegetation", (0, 210, 50), None),
    label("arboriculture", (120, 155, 100), None),
    label("tree", (0, 120, 15), None),
    label("spiney", (30, 143, 100), None),
    label("forest", (0, 70, 0), None),
    label("winter_high_vegetation", (90, 180, 170), None),
    label("ice", (59, 173, 240), None),
    label("river", (30, 145, 246), None),
    label("pond", (0, 75, 190), None),
    label("sea", (0, 55, 105), None),
    label("swimmingpool", (100, 130, 255), None),
    label("bridge", (160, 160, 246), None),
    label("boat", (130, 41, 244), None),
    label("railways", (75, 20, 132), None),
    label("road", (141, 91, 210), None),
    label("private_road", (205, 140, 242), None),
    label("central_reservation", (163, 127, 180), None),
    label("parking", (170, 60, 160), None),
    label("pedestrian", (190, 38, 194), None),
    label("yard", (255, 175, 200), None),
    label("sport", (255, 115, 180), None),
    label("cemetery", (125, 15, 70), None),
    label("impervious", (219, 20, 123), None),
    label("terrace", (200, 20, 79), None),
    label("container", (255, 65, 75), None),
    label("storage_tank", (195, 85, 0), None),
    label("greenhouse", (255, 150, 85), None),
    label("building", (240, 0, 0), None),
    label("high_building", (127, 1, 0), None),
    label("pipeline", (35, 85, 85), None)
]
all43 = [label(l.name, l.color, {i}) for i, l in enumerate(_all43)]
          

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
        windows,
        bands,
        merge,
        transforms,
    ):
        self.imgs = imgs
        self.msks = msks
        self.windows = windows
        self.bands = bands
        self.class_list = self.classes[merge].value
        self.merges = [list(l.values) for l in self.class_list]
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        #img = self.imgs[idx // len(self.crops)]
        #crop = self.crops[idx % len(self.crops)]
        img = self.imgs[idx]
        win = self.windows[idx]
        window = Window(*win)
        with rasterio.open(img, "r") as file:
            image = file.read(window=window, out_dtype=np.float32, indexes=self.bands)
        image = torch.from_numpy(image)
        label = None
        if self.msks:
            #msk = self.msks[idx // len(self.crops)]
            msk = self.msks[idx]
            with rasterio.open(msk, "r") as file:
                label = file.read(window=window, out_dtype=np.uint8)
            label = merge_labels(label, self.merges)
            label = torch.from_numpy(label).long()
        image, label = self.transforms(img=image, label=label)
        return {
            "image": image,
            "label": None if label is None else label.squeeze(),
            "image_path": img,
            "window": win,
            "label_path": None if label is None else msk
        }