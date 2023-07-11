import torch
import numpy as np

import rasterio

from .utils import FixedCropFromWindow, RandomCropFromWindow


class Raster(torch.utils.data.Dataset):
    
    def __init__(
        self,
        data_src,
        crop_size,
        shuffle,
        transforms,
        crop_step=None
    ):
        self.data_src = data_src
        self.crop_size = crop_size
        self.shuffle = shuffle
        self.transforms = transforms
        
        h = data_src.zone.height
        w = data_src.zone.width
        self.nb_crops = h*w/(crop_size**2)
        
        if shuffle:
            self.get_crop = RandomCropFromWindow(
                data_src.zone,
                crop_size
            )
        else:
            self.get_crop = FixedCropFromWindow(
                data_src.zone,
                crop_size,
                crop_step
            )
        
    def read_crop(self, crop):
        
        image = self.data_src.read_image(crop)
        image = torch.from_numpy(image)
        
        label = None
        if self.data_src.label_path:
            label = self.data_src.read_label(crop)
            label = torch.from_numpy(label)
            
        return image, label
            
    def __len__(self):
        
        if self.shuffle:
            return int(self.nb_crops)
        else:
            return len(self.get_crop.crops)
        
    def __getitem__(self, idx):
        
        crop = self.get_crop(idx)
        raw_image, raw_label = self.read_crop(crop)
        image, label = self.transforms(img=raw_image, label=raw_label) 
        
        # Here transform means the affine transform from matrix coords to long/lat
        crop_transform = rasterio.windows.transform(
            crop,
            transform=self.data_src.meta['transform']
        )
        
        return {
            'image':image,
            'label':label,
            'crop':crop,
            'crop_tf':crop_transform,
            'path': self.data_src.image_path,
            'crs': self.data_src.meta['crs']
        }