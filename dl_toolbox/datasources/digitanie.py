from dataclasses import dataclass
from collections import namedtuple
import enum
from .tif_datasource import TifDatasource

import torch
import rasterio
import numpy as np


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
            
@dataclass
class Digitanie9(TifDatasource):

    classes = enum.Enum(
        'Digitanie9classes',
        {
            'all':initial_nomenclature,
            'main':main_nomenclature,
        }
    )
    
@dataclass
class Digitanie24(TifDatasource):

    classes = enum.Enum(
        'Digitanie24classes',
        {
            'all':nomenc24,
        }
    )
