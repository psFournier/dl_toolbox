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

_all43 = [
    label("nodata", (250, 250, 250), 0),
    label("construction_site", (140, 90, 100), 1),
    label("bare_ground", (71, 58, 17), 2),
    label("bare_parking", (160, 130, 105), 3),
    label("bare_road", (160, 119, 34), 4),
    label("bare_pedestrian", (230, 167, 31), 5),
    label("sand", (234, 220, 90), 6),
    label("snow", (240, 239, 220), 7),
    label("field", (235, 255, 6), 8),
    label("sport_vegetation", (190, 215, 165), 9),
    label("grassland", (140, 240, 118), 10),
    label("aquaculture", (11, 222, 189), 11),
    label("hedge", (119, 211, 0), 12),
    label("shrub", (113, 184, 48), 13),
    label("vegetation", (0, 210, 50), 14),
    label("arboriculture", (120, 155, 100), 15),
    label("tree", (0, 120, 15), 16),
    label("spiney", (30, 143, 100), 17),
    label("forest", (0, 70, 0), 18),
    label("winter_high_vegetation", (90, 180, 170), 19),
    label("ice", (59, 173, 240), 20),
    label("river", (30, 145, 246), 21),
    label("pond", (0, 75, 190), 22),
    label("sea", (0, 55, 105), 23),
    label("swimmingpool", (100, 130, 255), 24),
    label("bridge", (160, 160, 246), 25),
    label("boat", (130, 41, 244), 26),
    label("railways", (75, 20, 132), 27),
    label("road", (141, 91, 210), 28),
    label("private_road", (205, 140, 242), 29),
    label("central_reservation", (163, 127, 180), 30),
    label("parking", (170, 60, 160), 31),
    label("pedestrian", (190, 38, 194), 32),
    label("yard", (255, 175, 200), 33),
    label("sport", (255, 115, 180), 34),
    label("cemetery", (125, 15, 70), 35),
    label("impervious", (219, 20, 123), 36),
    label("terrace", (200, 20, 79), 37),
    label("container", (255, 65, 75), 38),
    label("storage_tank", (195, 85, 0), 39),
    label("greenhouse", (255, 150, 85), 40),
    label("building", (240, 0, 0), 41),
    label("high_building", (127, 1, 0), 42),
    label("pipeline", (35, 85, 85), 43)
]
all43 = [label(l.name, l.color, {i}) for i, l in enumerate(_all43)]
          
nomenc = [
    label("Nodata", (10,10,10), {0,37,33,35,43,27}),
    label("Building", (240,0,0), {42,41,40,39}),
    label("Swimming pool", (100,130,255), {24}),
    label("Impervious", (130,130,130), {38,36,34,32,31,30,29,28,25}),
    label("Bare soil", (71,58,17), {1,2,3,4,5,6}),
    label("Aquaculture", (11,222,189), {11}),
    label("Natural water", (0,50,250), {21,22,23,26}),
    label("Snow&Ice", (0,250,250), {7,20}),
    label("High vegetation", (0,120,15), {12,13,14,15,16,17,18,19}),
    label("Herbaceous vegetation", (140,240,118), {8,9,10})
]

main5 = [
    label("void", (255, 255, 255), {0,43}),
    label("impervious surface", (38, 38, 38), {38,36,34,32,31,30,29,28,25}),
    label("building", (238, 118, 33), {42,41,40,39,38,37}),
    label("pervious surface", (34, 139, 34), {1,2,3,4,5,6,7,8,9,10,33,27,35}),
    label("high vegetation", (0, 222, 137), {12,13,14,15,16,17,18,19}),
    label("water", (0, 0, 238), {11,20,21,22,23,24,26})
]

classes = enum.Enum(
    "DigitanieClasses",
    {
        "all43": all43,
        "nomenc": nomenc,
        "main5": main5
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
        image, label = self.transforms(image, label)
        return {
            "image": image,
            "label": None if label is None else label.squeeze(),
            "image_path": img,
            "window": win,
            "label_path": None if label is None else msk
        }