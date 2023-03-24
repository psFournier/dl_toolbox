import torch
import rasterio
import numpy as np
from dataclasses import dataclass
from enum import Enum

label = namedtuple('label', ['idx', 'name', 'color'])
nomenclature = namedtuple('nomenclature', ['labels', 'merge'])


base_nom = nomenclature(
    labels=[
        label(idx=0, name='nodata', color=(0, 0, 0)),
        label(idx=1, name='vegetation', color=(0, 250, 50)),
        label(idx=2, name='water', color=(0, 50, 250)),
        label(idx=3, name='building', color=(250, 50, 50)),
        label(idx=4, name='road', color=(100, 100, 100)),
        label(idx=5, name='other', color=(250,250,250))
    ],
    merge=[[0], [2, 5], [3], [4], [7], [1, 6, 8, 9]]
)

building_nom = nomenclature(
    labels=[
        label(idx=0, name='building', color=(250, 50, 50)),
        label(idx=1, name='other', color=(250,250,250))
    ],
    merge=[[3], [0,1, 2, 4, 5, 6,7,8,9]]
)

semcity_labels = {

    'base' : {
        'void': {'color': (255, 255, 255), 'count': 3080233},
        'impervious surface': {'color': (38, 38, 38), 'count': 45886967},
        'building': {'color': (238, 118, 33), 'count':43472320},
        'pervious surface': {'color': (34, 139, 34), 'count':58879144 },
        'high vegetation': {'color': (0, 222, 137), 'count':31261675 },
        'car': {'color': (255, 0, 0), 'count': 3753288},
        'water': {'color': (0, 0, 238), 'count': 7199301},
        'sport venue': {'color': (160, 30, 230), 'count': 0}
    },
    'semcity' : {
        'other': {'color': (255, 255, 255)},
        'pervious surface': {'color': (34, 139, 34)},
        'water': {'color': (0, 0, 238)},
        'building': {'color': (238, 118, 33)},
        'high vegetation': {'color': (0, 222, 137)},
        'impervious surface': {'color': (38, 38, 38)}
    },
    'building': {
        'background': {'color': (0,0,0)},
        'building': {'color': (255, 255, 255)}
    }
}

mergers = {
    'base' : [[0], [1], [2], [3], [4], [5], [6], [7]],
    'semcity' : [[0,7], [3], [6], [2], [4], [1, 5]],
    'building' : [[0,1,4,3,5,6,7],[2]]
}


class SemcityNom(Enum):
    base = mainFuseVege_nom
    bulding = building_nom

@dataclass
class Semcity:
    
    image_path: str = None
    window: object = None
    no_data_vals: object = ()
    mins: ... = # default to be calculated
    maxs: ... = # default to be calculated
    label_path: ... = None
    nomenclatures: object = SemcityNom

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