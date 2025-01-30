import numpy as np
import rasterio
import torch

import rasterio.warp

from .digitanie import Digitanie
    
class DigitanieUnlabeledToa(Digitanie):
    
    def __init__(
        self,
        toa,
        bands,
        windows,
        transforms
    ):
        self.toa = toa
        self.bands = bands
        self.transforms = transforms
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        with rasterio.open(self.toa, "r") as file:
            image = file.read(window=window, out_dtype=np.float32, indexes=self.bands)
        image = torch.from_numpy(image).float()
        image, label = self.transforms(img=image, label=None)
        return {
            "image": image,
            "label": None,
            "image_path": self.toa,
            "window": window.flatten(),
            "label_path": None
        }