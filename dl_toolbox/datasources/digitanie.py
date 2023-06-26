from dataclasses import dataclass
from collections import namedtuple
import enum

import torch
import rasterio
import numpy as np
from shapely import Polygon
from dl_toolbox.utils import MergeLabels

DATA_POLYGON = Polygon(
    [[359326,4833160],
     [376735,4842547],
     [385238,4826271],
     [367914,4816946],
     [359326,4833160]]
)

label = namedtuple(
    'label',
    [
        'name',
        'color',
        'values'
    ]
)

initial_nomenclature = [
    label('nodata', (250,250,250), {0}),
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
    label('other', (0, 0, 0), {0, 1, 6, 9}),
    label('low vegetation', (0,250, 50), {2}),
    label('high vegetation', (0,100,50), {5}),
    label('water', (0, 50, 250), {3}),
    label('building', (250, 50, 50), {4}),
    label('road', (100, 100, 100), {7}),
    label('railways', (200, 100, 200), {8}),
]

nomenc24 = [
    label('nodata', (250, 250, 250), {0}),
    label('bare_ground', (100, 50, 0), {1}),
    label('low_vegetation',(0, 250, 50), {2}),
    label('water',(0, 50, 250), {3}),
    label('building', (250, 50, 50), {4}),
    label('high_vegetation', (0, 100, 50), {5}),
    label('parking',(200, 200, 200), {6}),
    label('road', (100, 100, 100), {7}),
    label('railways', (200, 100, 200), {8}),
    label('swimmingpool', (50, 150, 250), {9}),
    label('arboriculture', (50, 150, 100), {10}),
    label('snow', (250, 250, 250), {11}),
    label('sportsground', (250, 200, 50), {12}),
    label('storage_tank', (180, 180, 180), {13}),
    label('pond', (150, 200, 200), {14}),
    label('pedestrian', (150, 50, 50), {15}),
    label('roundabout', (50, 50, 50), {16}),
    label('container', (250, 250, 0), {17}),
    label('aquaculture', (50, 150, 200), {18}),
    label('port', (80, 80, 80), {19}),
    label('boat', (0, 250, 250), {20}),
    label('high building', (200, 0, 0), {21}),
    label('winter_vegetation', (200, 250, 200), {22}),
    label('industry', (200, 0, 250), {23}),
    label('beach', (250, 250, 100), {24})
]

def one_class_w_void(nomenc, name):
    idx = [l.name for l in nomenc].index(name)
    return [
        label('nodata', (250,250,250), {0}),
        label('other', (0, 0, 0), set(range(1, len(nomenc))) - {idx}),
        nomenc[idx]
    ]

def one_class(nomenc, name):
    idx = [l.name for l in nomenc].index(name)
    return [
        label('other', (0, 0, 0), set(range(0, len(nomenc))) - {idx}),
        nomenc[idx]
    ]

DigitanieNomenclatures = enum.Enum(
    'DigitanieNomenclatures',
    {
        'all':initial_nomenclature,
        'main':main_nomenclature,
        'building_void': one_class_w_void(initial_nomenclature, 'building'),
        'building': one_class(initial_nomenclature, 'building'),
        '24': nomenc24
    }
)
            
@dataclass
class Digitanie:

    bands: ... = None
    image_path: str = None
    zone: object = None #DATA_POLYGON
    minval: ... = None 
    maxval: ... = None 
    meanval: ... = None
    label_path: ... = None
    nomenclature_name: ... = None
    all_cls_counts: ... = None

    def __post_init__(self):
        
        with rasterio.open(self.image_path) as src:
            self.meta = src.meta
            
        self.nomenclature = DigitanieNomenclatures[self.nomenclature_name].value
        merges = [list(l.values) for l in self.nomenclature]
        self.labels_merger = MergeLabels(merges)
        self.cls_counts = np.array([np.sum(self.all_cls_counts[np.array(merge)]) for merge in merges])
        #self.weights_multiclass = [np.round(max(self.cls_counts)/c,1) for c in self.cls_counts]
        #self.weights_binary = [np.round((sum(self.cls_counts) - c)/c,1) for c in self.cls_counts]
        
    def read_image(self, window=None):
        
        with rasterio.open(self.image_path, 'r') as file:
            image = file.read(
                window=window,
                out_dtype=np.float32,
                indexes=self.bands
            )

        return image
    
    def read_label(self, window=None):
        
        with rasterio.open(self.label_path) as file:
            label = file.read(
                window=window,
                out_dtype=np.float32
            )
        label = self.labels_merger(np.squeeze(label))
        
        return label