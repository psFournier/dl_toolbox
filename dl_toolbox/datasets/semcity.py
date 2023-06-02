import torch
import rasterio
import numpy as np
from dataclasses import dataclass
from enum import Enum

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

def get_subnomenc(nomenc, idx):
    return [
        label('nodata', (0, 0, 0), {0}),
        label('other', (250, 250, 250), set(range(1, len(nomenc))) - {idx}),
        nomenc[idx]
    ]

class SemcityNomenclatures(Enum):
    base = mainFuseVege_nom
    building = get_subnomenc(initial_nomenclature, 2)

@dataclass
class Semcity:
    
    image_path: str = None
    window: object = None
    no_data_vals: object = ()
    mins: ... = # default to be calculated
    maxs: ... = # default to be calculated
    label_path: ... = None
    nomenclatures: object = SemcityNomenclatures

    def read_image(self, window=None, bands=None):
        
        with rasterio.open(self.image_path, 'r') as file:
            image = file.read(window=window, out_dtype=np.float32, indexes=bands)

        return image
    
    def read_label(self, window=None):
        
        with rasterio.open(self.label_path) as file:
            rgb = file.read(window=window, out_dtype=np.float32)
            
        rgb = rgb.transpose((1,2,0))
        labels = np.zeros(shape=rgb.shape[:-1], dtype=np.uint8)
        for label, key in enumerate(semcity_labels['base']):
            c = semcity_labels['base'][key]['color']
            d = rgb[..., 0] == c[0]
            d = np.logical_and(d, (rgb[..., 1] == c[1]))
            d = np.logical_and(d, (rgb[..., 2] == c[2]))
            labels[d] = label

        return labels
        
    def get_transform(self):
        
        with rasterio.open(self.image_path, 'r') as ds:
            tf = ds.transform
            if self.window is not None:
                return rasterio.windows.transform(self.window, transform=tf)
            else:
                return tf