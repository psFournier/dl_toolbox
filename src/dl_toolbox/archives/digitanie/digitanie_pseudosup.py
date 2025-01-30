import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
from rasterio.windows import Window

import rasterio.warp

class DigitaniePseudosup(Dataset):
    
    def __init__(
        self,
        imgs,
        msks,
        windows,
        bands,
        transforms,
    ):
        self.imgs = imgs
        self.msks = msks
        self.windows = windows
        self.bands = bands
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
            msk = self.msks[idx]
            with rasterio.open(msk, "r") as file:
                label = file.read(out_dtype=np.uint8)
            label = torch.from_numpy(label).long()
        image, label = self.transforms(image, label)
        return {
            "image": image,
            "label": None if label is None else label.squeeze(),
            "image_path": img,
            "window": win,
            "label_path": None if label is None else msk
        }