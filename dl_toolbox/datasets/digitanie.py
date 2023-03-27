import torch
import rasterio
import numpy as np
from dataclasses import dataclass
from collections import namedtuple
from enum import Enum

label = namedtuple('label', ['idx', 'name', 'color'])
nomenclature = namedtuple('nomenclature', ['labels', 'merge'])

mainFuseVege_nom = nomenclature(
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

road_nom = nomenclature(
    labels=[
        label(idx=0, name='road', color=(100, 100, 100)),
        label(idx=1, name='other', color=(250,250,250))
    ],
    merge=[[7], [0,1, 2, 4, 5, 6,3,8,9]]
)

class DigitanieNom(Enum):
    mainFuseVege = mainFuseVege_nom
    building = building_nom
    road = road_nom
    

#labels_desc = (
#    
#    #'base':{
#    #    'nodata': {'color': (0, 0, 0)},
#    #    'bare_ground': {'color':(100, 50, 0)},
#    #    'low_vegetation': {'color':(0, 250, 50)},
#    #    'water': {'color':(0, 50, 250)},
#    #    'building': {'color':(250, 50, 50)},
#    #    'high_vegetation': {'color':(0, 100, 50)},
#    #    'parking': {'color':(200, 200, 200)},
#    #    'road': {'color':(100, 100, 100)},
#    #    'railways': {'color':(200, 100, 200)},
#    #    'swimmingpool': {'color':(50, 150, 250)}
#    #},
#    #'6': {
#    #    'other': {'color': (0, 0, 0)},
#    #    'low_vegetation': {'color':(0, 250, 50)},
#    #    'water': {'color':(0, 50, 250)},
#    #    'building': {'color':(250, 50, 50)},
#    #    'high_vegetation': {'color':(0, 100, 50)},
#    #    'road': {'color':(100, 100, 100)}
#    #},
#    #'7main': {
#    #    'nodata': {'color': (0, 0, 0)},
#    #    'low_vegetation': {'color':(0, 250, 50)},
#    #    'water': {'color':(0, 50, 250)},
#    #    'building': {'color':(250, 50, 50)},
#    #    'high_vegetation': {'color':(0, 100, 50)},
#    #    'road': {'color':(100, 100, 100)},
#    #    'other': {'color': (250,250,250)}
#    #},
#    '6mainFuseVege': {
#        'desc': {
#            'nodata': (0, 0, 0),
#            'vegetation': (0, 250, 50),
#            'water': (0, 50, 250),
#            'building': (250, 50, 50),
#            'road': (100, 100, 100),
#            'other': (250,250,250)
#        },
#        'merge': [[0], [2, 5], [3], [4], [7], [1, 6, 8, 9]]
#    }
#}

@dataclass
class Digitanie:
    
    image_path: str = None
    image_zone: object = None
    no_data_vals: object = (
        np.array([0., 0., 0., 0.]).reshape(-1, 1, 1),
        np.array([0.0195023, 0.0336404, 0.0569544, 0.00735826]).reshape(-1, 1, 1)
    )
    mins: ... = np.array([0., 0., 0., 0.]).reshape(-1, 1, 1)
    maxs: ... = np.array([1.101, 0.979, 0.948, 1.514]).reshape(-1, 1, 1)
    label_path: ... = None
    label_zone: object = None
    nomenclatures: object = DigitanieNom

    def read_image(self, window=None, bands=None):
        
        with rasterio.open(self.image_path, 'r') as file:
            image = file.read(window=window, out_dtype=np.float32, indexes=bands)

        return image
    
    def read_label(self, window=None):
        
        with rasterio.open(self.label_path) as file:
            label = file.read(window=window, out_dtype=np.float32)
        
        return label
        
    def get_transform(self):
        
        with rasterio.open(self.image_path, 'r') as ds:
            tf = ds.transform
            if self.window is not None:
                return rasterio.windows.transform(self.window, transform=tf)
            else:
                return tf