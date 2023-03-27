import torch
import rasterio
import numpy as np

from rasterio.windows import Window, from_bounds
from argparse import ArgumentParser
import torchvision.transforms.functional as F
from shapely import Polygon

from dl_toolbox.utils import get_tiles
from dl_toolbox.utils import MergeLabels, OneHot, LabelsToRGB, RGBToLabels
import dl_toolbox.augmentations as augmentations
from dl_toolbox.utils import minmax


def crop_from_polygon(window, crop_size, polygon, tf):
    
    while True:
        
        cx = window.col_off + np.random.randint(0, window.width - crop_size + 1)
        cy = window.row_off + np.random.randint(0, window.height - crop_size + 1)
        crop = Window(cx, cy, crop_size, crop_size)
        
        left, bottom, right, top = rasterio.windows.bounds(
            crop,
            transform=tf
        )

        crop_poly = Polygon(generate_polygon((left, bottom, right, top)))
        
        if polygon.contains(crop_poly): return crop
    
class Raster(torch.utils.data.Dataset):
    
    def __init__(
        self,
        src,
        crop_size,
        aug,
        bands,
        labels,
        holes=[]
    ):
        
        self.src = src
        self.crop_size = crop_size
        self.aug = augmentations.get_transforms(aug)
        self.bands = bands
        self.holes = holes
        
        self.pre_crop_size = int(np.ceil(np.sqrt(2) * crop_size))
        self.labels = src.nomenclatures[labels].value
        self.labels_merger = MergeLabels(self.labels.merge)
        with rasterio.open(src.image_path) as raster_img:
            self.raster_tf = raster_img.transform
            self.crs = raster_img.crs
        
        self.polygon_window = from_bounds(
            *shapely.Polygon(src.polygon).bounds, 
            transform=self.raster_tf
        ).round_offsets().round_lengths()
            
    def __len__(self):
        
        return 1
        
    def __getitem__(self, idx):
        
        bands_idxs = np.array(self.bands).astype(int) - 1
        crop = self.sample_crop(idx)
        
        crop = self.crop_from_window(self.window, self.pre_crop_size):
        left, bottom, right, top = rasterio.windows.bounds(
            crop,
            transform=self.raster_img_tf
        )
        crop_poly = shapely.Polygon(generate_polygon((left, bottom, right, top)))

        # checking that the crop intersects the train_zone enough
        intersection = shapely.intersection(crop_poly, self.polygon)
        area = shapely.area(intersection) / shapely.area(crop_poly)
        
        
        image = self.raster.read_image(crop, self.bands)
        image = torch.from_numpy(image).float().contiguous()
        label = None
        if self.raster.label_path:
            label = self.raster.read_label(crop)
            label = self.labels_merger(np.squeeze(label))
            label = torch.from_numpy(label).float().contiguous()
        image, label = self.rnd_rotate_and_true_crop(image, label)  
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