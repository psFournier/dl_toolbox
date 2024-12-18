import numpy as np
import rasterio
import torch
import enum

from torch.utils.data import Dataset
from rasterio.windows import Window
from dl_toolbox.utils import label, merge_labels
import torchvision.transforms.v2 as v2
from torchvision import tv_tensors


all7 = [
    label("void", (255, 255, 255), {0}),
    label("impervious", (38, 38, 38), {1}),
    label("building", (238, 118, 33), {2}),
    label("pervious", (34, 139, 34), {3}),
    label("high vege", (0, 222, 137), {4}),
    label("car", (255, 0, 0), {5}),
    label("water", (0, 0, 238), {6}),
    label("sport", (160, 30, 230), {7}),
]

main5 = [
    label("void", (255, 255, 255), {0,7}),
    label("impervious", (38, 38, 38), {1,5}),
    label("building", (238, 118, 33), {2}),
    label("pervious", (34, 139, 34), {3}),
    label("high vege", (0, 222, 137), {4}),
    label("water", (0, 0, 238), {6})
]

building = [
    label("void", (255, 255, 255), {0,1,3,4,5,6,7}),
    label("building", (238, 118, 33), {2})
]


class Semcity(Dataset):
    
    all_class_lists = enum.Enum(
        "SemcityClasses",
        {
            "all7": all7,
            "main5": main5,
            "building": building
        },
    )
    # RGB = bands 4,3,2
    max_val_per_bands = np.array([652.,732.,1078.,1131.,756.,1089.,1125.,1046.])
    
    def __init__(self, imgs, msks, windows, bands, merge, transforms):
        self.imgs = imgs
        self.msks = msks
        self.transforms = v2.ToDtype(dtype={
            tv_tensors.Image: torch.float32,
            tv_tensors.Mask: torch.int64,
            "others":None
        }, scale=True)
        if transforms is not None:
            self.transforms = v2.Compose([self.transforms, transforms])
        self.merges = None
        if merge:
            self.class_list = self.all_class_lists[merge].value
            self.merges = [list(l.values) for l in self.class_list]
        self.windows = windows
        self.bands = bands

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        
        image_path = self.imgs[idx]
        window = self.windows[idx]
        with rasterio.open(image_path, "r") as file:
            image = file.read(out_dtype=np.float32, window=Window(*window), indexes=self.bands)
        # images are uint16, but dividing by 2**16 makes them too dark
        max_vals_per_band = self.max_val_per_bands[[b-1 for b in self.bands]]
        image /= max_vals_per_band.reshape(-1,1,1)
        image = tv_tensors.Image(torch.from_numpy(image))
        
        target = None
        if self.msks:
            mask_path = self.msks[idx]
            with rasterio.open(mask_path, "r") as file:
                mask = file.read(window=Window(*window), out_dtype=np.uint8)
            mask = torch.from_numpy(mask)
            if self.merges:
                mask = merge_labels(mask, self.merges)
            target = tv_tensors.Mask(mask)
            
        image, target = self.transforms(image, target)
        if self.msks:
            target = target.squeeze()
            
        return {'image':image, 'target':target, 'image_path':image_path, 'window':window}