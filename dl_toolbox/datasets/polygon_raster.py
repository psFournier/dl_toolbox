import torch
import rasterio
import numpy as np

from rasterio.windows import Window
from argparse import ArgumentParser 

from dl_toolbox.utils import get_tiles
from dl_toolbox.utils import MergeLabels, OneHot, LabelsToRGB, RGBToLabels
import dl_toolbox.augmentations as augmentations



class PolyRaster(torch.utils.data.Dataset):
    
    def __init__(
        self,
        raster_img_path,
        raster_label_path,
        polygon,
        crop_size,
        labels='base',
        raster_mask_poly=None
    ):
        self.raster_img_path = raster_img_path
        self.raster_label_path = raster_label_path
        self.polygon = polygon
        self.crop_size = crop_size
        with rasterio.open(raster_img_path) as raster_img: 
            self.raster_img_tf = raster_img.transform
        self.labels = labels_dict[labels]
        self.label_merger = MergeLabels(mergers[labels])
        
        #assert crs?
        self.tile = rasterio.windows.from_bounds(
            *polygon.bounds, 
            transform=self.raster_img_tf
        ).round_offsets().round_lengths()
        
        if raster_mask_poly:
            self.raster_mask_poly = raster_mask_poly
        else:
            with rasterio.open(raster_img_path) as raster_img:
                mask = raster_img.dataset_mask()
            mask_polys = []
            for shape, value in rasterio.features.shapes(
                mask, mask=mask, connectivity=8, transform=raster_img.transform
            ):
                mask_polys.append(shapely.geometry.shape(shape))
            self.raster_mask_poly = shapely.geometry.MultiPolygon(mask_polys)
            
    def __len__(self):

        return 1
        
    def __getitem__(self, idx):
        
        area = 0
        full = False
        while area <= 0.8 or not full:
            cx = self.tile.col_off + np.random.randint(0, self.tile.width - self.crop_size + 1)
            cy = self.tile.row_off + np.random.randint(0, self.tile.height - self.crop_size + 1)
            # raster pixel coords
            crop = rasterio.windows.Window(cx, cy, self.crop_size, self.crop_size)
                            
            left, bottom, right, top = rasterio.windows.bounds(
                crop,
                transform=self.raster_img_tf
            )
                
            crop_poly = shapely.Polygon(generate_polygon((left, bottom, right, top)))

            # checking that the crop intersects the train_zone enough
            intersection = shapely.intersection(crop_poly, self.polygon)
            area = shapely.area(intersection) / shapely.area(crop_poly)

            # checking that the crop does not contain pixels out of the initial raster mask
            intersection_mask = shapely.intersection(crop_poly, self.raster_mask_poly)
            full = shapely.area(intersection_mask) / shapely.area(crop_poly) == 1
        
        with rasterio.open(self.raster_img_path) as raster_img:
            crop_img = raster_img.read(window=crop, out_dtype=np.uint8)[[0]]
            crop_img = torch.from_numpy(crop_img).float().contiguous()
            #crop_img = -1 + 2 * (crop_img - self.mins) / (self.maxs - self.mins)
            tf_img = rasterio.windows.transform(crop, transform=raster_img.transform)
        
        with rasterio.open(self.raster_label_path) as raster_label:
            crop_label = raster_label.read(window=crop, out_dtype=np.uint8)[0]
            crop_label = np.squeeze(crop_label)
            crop_label = self.label_merger(crop_label)
            crop_label = torch.from_numpy(crop_label).long().contiguous()
            tf_label = rasterio.windows.transform(crop, transform=raster_label.transform)
        
        #return crop_img, tf_img, crop_label, tf_label
        return {
            'image':crop_img,
            'tf_image':tf_img,
            'crop':crop,
            'label':crop_label,
            'tf_label':tf_label
        }