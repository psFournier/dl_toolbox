from dataclasses import dataclass
from collections import namedtuple
from enum import Enum

import torch
import rasterio
import numpy as np
from shapely import Polygon


DATA_POLYGON = Polygon(
    [[359326,4833160],
     [376735,4842547],
     [385238,4826271],
     [367914,4816946],
     [359326,4833160]]
)

label = namedtuple('label', ['name', 'color', 'values'])

initial_nomenclature = [
    label('nodata', (0, 0, 0), {0}),
    label('bare_ground', (100, 50, 0), {1}),
    label('low_vegetation', (0, 250, 50), {2}),
    label('water', (0, 50, 250), {3}),
    label('building', (250, 50, 50), {4}),
    label('high_vegetation', (0, 100, 50), {5}),
    label('parking', (200, 200, 200), {6}),
    label('road', (100, 100, 100), {7}),
    label('railways', (200, 100, 200), {8}),
    label('swimmingpool', (50, 150, 250), {9})
]

main_nomenclature = [
    label('nodata', (0, 0, 0), {0}),
    label('low vegetation', (0,250, 50), {2}),
    label('high vegetation', (0,100,50), {5}),
    label('water', (0, 50, 250), {3}),
    label('building', (250, 50, 50), {4}),
    label('road', (100, 100, 100), {7}),
    label('other', (250,250,250), {1, 6, 8, 9})
]

def get_subnomenc(nomenc, idx):
    return [
        label('nodata', (0, 0, 0), {0}),
        label('other', (250, 250, 250), set(range(1, len(nomenc))) - {idx}),
        nomenc[idx]
    ]

class DigitanieNomenclatures(Enum):
    initial = initial_nomenclature
    main = main_nomenclature
    building = get_subnomenc(initial_nomenclature, 4)
    low_vege = get_subnomenc(initial_nomenclature, 2)
    high_vege = get_subnomenc(initial_nomenclature, 5)
    water = get_subnomenc(initial_nomenclature, 3)
    road = get_subnomenc(initial_nomenclature, 7)
            
@dataclass
class Digitanie:
    
    image_path: str = None
    zone: object = DATA_POLYGON
    no_data_vals: object = (
        np.array([0., 0., 0., 0.]).reshape(-1, 1, 1),
        np.array([0.0195023, 0.0336404, 0.0569544, 0.00735826]).reshape(-1, 1, 1)
    )
    mins: ... = np.array([0., 0., 0., 0.]).reshape(-1, 1, 1)
    maxs: ... = np.array([1.101, 0.979, 0.948, 1.514]).reshape(-1, 1, 1)
    label_path: ... = None
    nomenclatures = DigitanieNomenclatures
    
    def __post_init__(self):
        
        with rasterio.open(self.image_path) as src:
            self.meta = src.meta

    def read_image(self, window=None, bands=None):
        
        with rasterio.open(self.image_path, 'r') as file:
            image = file.read(window=window, out_dtype=np.float32, indexes=bands)

        return image
    
    def read_label(self, window=None):
        
        with rasterio.open(self.label_path) as file:
            label = file.read(window=window, out_dtype=np.float32)
        
        return label
    

                
    #def get_transform(self):
    #    
    #    with rasterio.open(self.image_path, 'r') as ds:
    #        tf = ds.transform
    #        if self.window is not None:
    #            return rasterio.windows.transform(self.window, transform=tf)
    #        else:
    #            return tf