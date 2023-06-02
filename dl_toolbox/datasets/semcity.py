from dataclasses import dataclass
from collections import namedtuple
import enum

import torch
import rasterio
import numpy as np
from shapely import Polygon

label = namedtuple('label', ['name', 'color', 'values'])

initial_nomenclature = [
    label('void', (255, 255, 255), {0}),
    label('impervious surface', (38, 38, 38), {1}),
    label('building', (238, 118, 33), {2}),
    label('pervious surface', (34, 139, 34), {3}),
    label('high vegetation', (0, 222, 137), {4}),
    label('car', (255, 0, 0), {5}),
    label('water', (0, 0, 238), {6}),
    label('sport venue', (160, 30, 230), {7})
]

main_nomenclature = [
    label('void', (250,250,250), {0, 7}),
    label('low vegetation', (0,250, 50), {3}),
    label('high vegetation', (0,100,50), {4}),
    label('water', (0, 50, 250), {6}),
    label('building', (250, 50, 50), {2}),
    label('road', (100, 100, 100), {1,5})
]

SemcityNomenclatures = enum.Enum(
    'SemcityNomenclatures',
    {
        'all':initial_nomenclature,
        'main':main_nomenclature,
    }
)

@dataclass
class Semcity:
    
    image_path: str = None
    zone: object = None
    no_data_vals: object = ()
    mins: ... = None
    maxs: ... = None
    label_path: ... = None
    nomenclatures: object = SemcityNomenclatures
    
    def __post_init__(self):
        
        with rasterio.open(self.image_path) as src:
            self.meta = src.meta

    def read_image(self, window=None, bands=None):
        
        with rasterio.open(self.image_path, 'r') as file:
            image = file.read(window=window, out_dtype=np.float32, indexes=bands)

        return image
    
    def read_label(self, window=None):
        
        with rasterio.open(self.label_path) as file:
            rgb = file.read(window=window, out_dtype=np.float32)
            
        rgb = rgb.transpose((1,2,0))
        labels = np.zeros(shape=rgb.shape[:-1], dtype=np.uint8)
        for i, label in enumerate(initial_nomenclature):
            c = label.color
            d = rgb[..., 0] == c[0]
            d = np.logical_and(d, (rgb[..., 1] == c[1]))
            d = np.logical_and(d, (rgb[..., 2] == c[2]))
            labels[d] = i

        return labels
        
    def get_transform(self):
        
        with rasterio.open(self.image_path, 'r') as ds:
            tf = ds.transform
            if self.window is not None:
                return rasterio.windows.transform(self.window, transform=tf)
            else:
                return tf