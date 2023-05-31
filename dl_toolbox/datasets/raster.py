import random

import torch
import rasterio
import numpy as np
import shapely

import rasterio.windows as windows
from argparse import ArgumentParser
import torchvision.transforms.functional as F

from dl_toolbox.utils import get_tiles
from dl_toolbox.utils import MergeLabels, OneHot, LabelsToRGB, RGBToLabels
import dl_toolbox.augmentations as augmentations
from dl_toolbox.utils import minmax


def polygon_from_bbox(bbox):
    """
    Generates a list of coordinates: [[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x1,y1]]
    """
    return [[bbox[0],bbox[1]],
             [bbox[2],bbox[1]],
             [bbox[2],bbox[3]],
             [bbox[0],bbox[3]],
             [bbox[0],bbox[1]]]

class CropFromWindow:
    
    def __init__(self, window):
        
        self.window = window
    
    def __call__(self, crop_size):
    
        col_off, row_off, width, height = self.window.flatten()
        cx = col_off + random.randint(0, width - crop_size)
        cy = row_off + random.randint(0, height - crop_size)
        crop = windows.Window(cx, cy, crop_size, crop_size)

        return crop

class CropFromPoly:
    
    def __init__(self, poly, tf):
        
        self.poly = poly
        self.minx, self.miny, self.maxx, self.maxy = poly.bounds
        self.tf = tf
        self.transformer = rasterio.transform.AffineTransformer(tf)
        
    def __call__(self, crop_size):
        
        while True:
            pnt = shapely.Point(
                np.random.uniform(self.minx, self.maxx),
                np.random.uniform(self.miny, self.maxy)
            )
            if self.poly.contains(pnt):
                cx, cy = self.transformer.rowcol(pnt.x, pnt.y)
                crop = windows.Window(cx, cy, crop_size, crop_size)
                bbox = windows.bounds(crop, transform=self.tf)
                crop_poly = shapely.Polygon(polygon_from_bbox(bbox))
                #print(self.poly, crop_poly)
                if self.poly.contains(crop_poly): 
                    return crop

class Raster(torch.utils.data.Dataset):
    
    def __init__(
        self,
        data_src,
        crop_size,
        aug,
        bands,
        nomenclature
    ):
        
        self.data_src = data_src
        self.crop_size = crop_size
        self.aug = augmentations.get_transforms(aug)
        self.bands = bands
        self.nomenclature = nomenclature
        self.labels_merger = MergeLabels([list(l.values) for l in nomenclature])
        
        if isinstance(data_src.zone, windows.Window):
            self.get_crop = CropFromWindow(
                data_src.zone
            )
        elif isinstance(data_src.zone, shapely.Polygon):
            self.get_crop = CropFromPoly(
                data_src.zone,
                data_src.meta['transform']
            )
            
    def read_crop(self, crop):
        
        image = self.data_src.read_image(crop, self.bands)
        image = torch.from_numpy(image).float().contiguous()
        label = None
        if self.data_src.label_path:
            label = self.data_src.read_label(crop)
            label = self.labels_merger(np.squeeze(label))
            label = torch.from_numpy(label).float().contiguous()
            
        return image, label
    
    def rnd_rotate_and_crop(self, image, crop_size, label=None):

        angle = np.random.randint(0, 180)
        image = F.rotate(image, angle=angle)
        image = F.center_crop(image, output_size=crop_size)
        if label is not None:
            if label.dim() == 2:
                label = label.unsqueeze(0)
            label = F.rotate(label, angle=angle)
            label = F.center_crop(label, output_size=crop_size)

        return image, label

    def normalize(self, image):
        
        bands_idxs = np.array(self.bands).astype(int) - 1
        mins = torch.Tensor(self.data_src.mins[bands_idxs])
        maxs = torch.Tensor(self.data_src.maxs[bands_idxs])
        normalized = torch.clip((image - mins) / (maxs - mins), 0, 1)
        
        return normalized
            
    def __len__(self):
        
        return 1
        
    def __getitem__(self, idx):
        
        pre_crop_size = int(np.ceil(np.sqrt(2) * self.crop_size))
        crop = self.get_crop(pre_crop_size) 
        pre_image, pre_label = self.read_crop(crop)
        image, label = self.rnd_rotate_and_crop(pre_image, self.crop_size, pre_label)       
        image = self.normalize(image)
        
        if self.aug is not None:
            image, label = self.aug(img=image, label=label)
        if label is not None:
            label = label.long().squeeze()
            
        crop_transform = windows.transform(
            crop,
            transform=self.data_src.meta['transform']
        )
        
        return {
            'pre_image': pre_image,
            'image':image,
            'label':label,
            'crop':crop,
            'crop_tf':crop_transform,
            'path': self.data_src.image_path,
            'crs': self.data_src.meta['crs']
        }
        
class PretiledRaster(Raster):
    
    def __init__(self, crop_step, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        self.crop_step = crop_step
        zone = self.data_src.zone
        assert isinstance(zone, windows.Window)
        self.tiles = [
            tile for tile in get_tiles(
                nols=zone.width, 
                nrows=zone.height, 
                size=self.crop_size, 
                step=crop_step if crop_step else self.crop_size,
                row_offset=zone.row_off, 
                col_offset=zone.col_off
            )
        ]
        
    def __len__(self):
        
        return len(self.tiles)
    
    def __getitem__(self, idx):
        
        crop = self.tiles[idx]
        
        image, label = self.read_crop(crop)
        image = self.normalize(image)
        
        if self.aug is not None:
            image, label = self.aug(img=image, label=label)
        if label is not None:
            label = label.long().squeeze()
            
        crop_transform = windows.transform(
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