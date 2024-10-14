import numpy as np
import rasterio
import torch
import enum

from torch.utils.data import Dataset
from rasterio.windows import Window
from dl_toolbox.utils import label, merge_labels
import torchvision.transforms.v2 as v2
from torchvision import tv_tensors


all_class_lists = enum.Enum(
    "AirsClasses",
    {
        "building": [
            label("other", (0, 0, 0), {0}),
            label("building", (255, 255, 255), {1}),
        ],
    },
)

class Airs(Dataset):
    
    all_class_lists = all_class_lists

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
        self.class_list = self.all_class_lists[merge].value
        self.merges = [list(l.values) for l in self.class_list]
        self.transforms = v2.ToDtype(dtype={
            tv_tensors.Image: torch.float32,
            tv_tensors.Mask: torch.int64,
            "others":None
        }, scale=True)
        if transforms:
            self.transforms = v2.Compose([self.transforms, transforms])
            
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        
        image_path = self.imgs[idx]
        window = self.windows[idx]
        with rasterio.open(image_path, "r") as file:
            image = file.read(window=Window(*window), indexes=self.bands)
        image = tv_tensors.Image(torch.from_numpy(image))
        
        target = None
        if self.msks:
            mask_path = self.msks[idx]
            with rasterio.open(mask_path, "r") as file:
                mask = file.read(window=Window(*window), out_dtype=np.uint8)
            mask = torch.from_numpy(mask)
            mask = merge_labels(mask, self.merges)
            target = tv_tensors.Mask(mask)
            
        image, target = self.transforms(image, target)
        if self.msks:
            target = target.squeeze()
            
        return {'image':image, 'target':target, 'image_path':image_path, 'window':window}