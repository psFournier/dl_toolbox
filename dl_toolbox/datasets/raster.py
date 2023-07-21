import os
import numpy as np
import rasterio
from rasterio.windows import Window

import torch
from torch.utils.data import Dataset
import dl_toolbox.transforms as transforms
from dl_toolbox.utils import merge_labels


class Raster(Dataset):
    
    @classmethod
    def read_image(cls, path, window=None, bands=None):
        
        with rasterio.open(path, 'r') as file:
            image = file.read(
                window=window,
                out_dtype=np.float32,
                indexes=bands
            )
            
        return torch.from_numpy(image)
    
    @classmethod
    def read_label(cls, path, window=None, merge=None):
        
        with rasterio.open(path, 'r') as file:
            label = file.read(
                window=window,
                out_dtype=np.uint8
            )
            
        if merge is not None:
            label = merge_labels(
                label.squeeze(),
                [list(l.values) for l in cls.classes[merge].value]
            )
        
        return torch.from_numpy(label)

    def __init__(
        self,
        img_path,
        label_path,
        bands,
        merge,
        crop_size,
        shuffle,
        transforms,
        crop_step=None
    ):
        self.img_path = img_path
        self.label_path = label_path
        self.bands=bands
        self.merge = merge
        self.crop_size = crop_size
        self.transforms = transforms
        
        w, h = imagesize.get(img_path)
        self.nb_crops = h*w/(crop_size**2)
        
        if shuffle:
            self.get_crop = RandomCropFromWindow(
                rasterio.windows.Window(0, 0, w, h),
                crop_size
            )
        else:
            self.get_crop = FixedCropFromWindow(
                rasterio.windows.Window(0, 0, w, h),
                crop_size,
                crop_step
            )
            
    def __len__(self):
        
        if self.shuffle:
            return int(self.nb_crops)
        else:
            return len(self.get_crop.crops)
        
    def __getitem__(self, idx):
        
        crop = self.get_crop(idx)
        
        image = self.read_img(
            self.img_path,
            crop,
            self.bands
        )
        
        label=None
        if self.label_path:
            label = self.read_label(
                self.label_path,
                crop,
                self.merge)
            )
        
        image, label = self.transforms(img=raw_image, label=raw_label) 
        
        return {
            'image':image,
            'label':label,
        }