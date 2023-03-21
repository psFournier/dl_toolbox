import torch
import rasterio
import numpy as np

from rasterio.windows import Window
from argparse import ArgumentParser
import torchvision.transforms.functional as F

from dl_toolbox.utils import get_tiles
from dl_toolbox.utils import MergeLabels, OneHot, LabelsToRGB, RGBToLabels
import dl_toolbox.augmentations as augmentations
from dl_toolbox.utils import minmax


def crop_from_window(window, crop_size):
    
    cx = window.col_off + np.random.randint(0, window.width - crop_size + 1)
    cy = window.row_off + np.random.randint(0, window.height - crop_size + 1)
    crop = Window(cx, cy, crop_size, crop_size)
    
    return crop

def intersect(window, holes):
    
    for hole in holes:
        try:
            inter = window.intersection(hole)
            #print(f'{window} intersects with {hole}') 
            return True
        except rasterio.errors.WindowError:
            pass
        
    return False

def crop_avoiding_holes(window, crop_size, holes):
    
    while True:
        cx = window.col_off + np.random.randint(0, window.width - crop_size + 1)
        cy = window.row_off + np.random.randint(0, window.height - crop_size + 1)
        crop = Window(cx, cy, crop_size, crop_size)
        if not intersect(crop, holes) : return crop
    
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
        self.pre_crop_size = int(np.ceil(np.sqrt(2) * crop_size))
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
                
        return crop_avoiding_holes(self.raster.window, self.pre_crop_size, self.holes)
    
    def check_valid(self, image, no_data_vals):
        
        all_valid = True
        for no_data_val in no_data_vals:
            no_data = np.isclose(image, no_data_val).all(axis=0)
            all_valid = all_valid and not no_data.any()
        
        return all_valid
    
    def rnd_rotate_and_true_crop(self, image, label=None):
        
        angle = np.random.randint(0, 180)
        image = F.rotate(image, angle=angle)
        image = F.center_crop(image, output_size=self.crop_size)
        if label is not None:
            if label.dim() == 2:
                label = label.unsqueeze(0)
            label = F.rotate(label, angle=angle)
            label = F.center_crop(label, output_size=self.crop_size)
            
        return image, label
        
    def __getitem__(self, idx):
        
        bands_idxs = np.array(self.bands).astype(int) - 1
        safety = 0
        while True:
            
            assert safety <= 100
            crop = self.sample_crop(idx)
            
            image = self.raster.read_image(crop, self.bands)
            image = torch.from_numpy(image).float().contiguous()
            label = None
            if self.raster.label_path:
                label = self.raster.read_label(crop)
                label = self.labels_merger(np.squeeze(label))
                label = torch.from_numpy(label).float().contiguous()
            
            image, label = self.rnd_rotate_and_true_crop(image, label)

            no_data_vals = [v[bands_idxs] for v in self.raster.no_data_vals]
            if self.check_valid(image, no_data_vals): break
            safety += 1
        
        mins = torch.Tensor(self.raster.mins[bands_idxs])
        maxs = torch.Tensor(self.raster.maxs[bands_idxs])
        image = torch.clip((image - mins) / (maxs - mins), 0, 1)
        
        if self.aug is not None:
            image, label = self.aug(img=image, label=label)
        if label is not None:
            label = label.long().squeeze()
        

        return {
            'image':image,
            'label':label,
            'crop':crop,
            'crop_tf':rasterio.windows.transform(crop, transform=self.raster_tf),
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
                image = self.raster.read_image(tile, self.bands)
                bands_idxs = np.array(self.bands).astype(int) - 1
                no_data_vals = [v[bands_idxs] for v in self.raster.no_data_vals]
                if self.check_valid(image, no_data_vals):
                    self.tiles.append(tile)
                #    print(f'Tile {tile} accepted')
                #else:
                #    print(f'Tile {tile} rejected')
        
    def __len__(self):
        
        return len(self.tiles)
    
    def __getitem__(self, idx):
        
        crop = self.tiles[idx]
        
        image = self.raster.read_image(crop, self.bands)
        image = torch.from_numpy(image).float().contiguous()
        label = None
        if self.raster.label_path:
            label = self.raster.read_label(crop)
            label = self.labels_merger(np.squeeze(label))
            label = torch.from_numpy(label).float().contiguous()
        
        bands_idxs = np.array(self.bands).astype(int) - 1
        mins = torch.Tensor(self.raster.mins[bands_idxs])
        maxs = torch.Tensor(self.raster.maxs[bands_idxs])
        image = torch.clip((image - mins) / (maxs - mins), 0, 1)
        
        if self.aug is not None:
            image, label = self.aug(img=image, label=label)
            
        if label is not None:
            label = label.long().squeeze()
        
        return {
            'image':image,
            'label':label,
            'crop':crop,
            'crop_tf':rasterio.windows.transform(crop, transform=self.raster_tf),
            'path': self.raster.image_path,
            'crs': self.crs
        }