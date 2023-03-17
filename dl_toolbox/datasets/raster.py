import torch
import rasterio
import numpy as np

from rasterio.windows import Window
from argparse import ArgumentParser 

from dl_toolbox.utils import get_tiles
from dl_toolbox.utils import MergeLabels, OneHot, LabelsToRGB, RGBToLabels
import dl_toolbox.augmentations as augmentations


def crop_from_window(window, crop_size):
    
    cx = window.col_off + np.random.randint(0, window.width - crop_size + 1)
    cy = window.row_off + np.random.randint(0, window.height - crop_size + 1)
    crop = Window(cx, cy, crop_size, crop_size)
    
    return crop

def intersect(window, holes):
    
    for hole in holes:
        try:
            inter = window.intersection(hole)
            print(f'{window} intersects with {hole}') 
            return True
        except rasterio.errors.WindowError:
            pass
        
    return False

def crop_avoiding_holes(window, crop_size, holes):
    
    intersect = True
    while intersect:
        cx = window.col_off + np.random.randint(0, window.width - crop_size + 1)
        cy = window.row_off + np.random.randint(0, window.height - crop_size + 1)
        crop = Window(cx, cy, crop_size, crop_size)
        intersect = intersect(crop, holes)
    
    return crop
    
class Raster(torch.utils.data.Dataset):
    
    def __init__(
        self,
        raster,
        crop_size,
        aug,
        bands,
        labels,
        holes=[]
    ):
        
        self.raster = raster
        assert raster.image_path is not None
        self.crop_size = crop_size
        self.aug = augmentations.get_transforms(aug)
        self.bands = bands
        self.labels = raster.nomenclatures[labels].value
        self.holes = holes
        
        self.labels_merger = MergeLabels(self.labels.merge)
        with rasterio.open(raster.image_path) as raster_img:
            self.raster_tf = raster_img.transform
            self.crs = raster_img.crs
            
    def __len__(self):
        
        return 1
    
    def sample_crop(self, idx):
                
        return crop_avoiding_holes(self.raster.window, self.crop_size, self.holes)
        
    def __getitem__(self, idx):
        
        crop = self.sample_crop(idx)
        crop_tf = rasterio.windows.transform(crop, transform=self.raster_tf)
            
        image = self.raster.read_image(crop, self.bands)
        image = torch.from_numpy(image).float().contiguous()

        label = None
        if self.raster.label_path:
            label = self.raster.read_label(crop)
            label = self.labels_merger(np.squeeze(label))
            label = torch.from_numpy(label).long().contiguous()

        if self.aug is not None:
            image, label = self.aug(img=image, label=label)

        return {
            'image':image,
            'label':label,
            'crop':crop,
            'crop_tf':crop_tf,
            'path': self.raster.image_path,
            'crs': self.crs
        }
        
class PretiledRaster(Raster):
    
    def __init__(self, crop_step, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        self.crop_step = crop_step
        self.tiles = []
        for tile in get_tiles(
            nols=self.raster.window.width, 
            nrows=self.raster.window.height, 
            size=self.crop_size, 
            step=crop_step if crop_step else self.crop_size,
            row_offset=self.raster.window.row_off, 
            col_offset=self.raster.window.col_off
        ):
            if not intersect(tile, self.holes):
                self.tiles.append(tile)
        
    def __len__(self):
        
        return len(self.tiles)
    
    def sample_crop(self, idx):
        
        return self.tiles[idx]