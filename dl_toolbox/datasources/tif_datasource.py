from dataclasses import dataclass
from collections import namedtuple
import enum

import torch
import rasterio
import numpy as np
from shapely import Polygon
from dl_toolbox.utils import merge_labels

            
@dataclass
class TifDatasource:

    image_path: ... = None
    label_path: ... = None
    zone: ... = None 
    minval: ... = None
    maxval: ... = None 
    meanval: ... = None
    
    def __post_init__(self):
        
        with rasterio.open(self.image_path) as src:
            self.meta = src.meta
            
        #self.nomenclature = DigitanieNomenclatures[self.nomenclature_name].value
        #merges = [list(l.values) for l in self.nomenclature]
        #self.labels_merger = MergeLabels(merges)
        
    def read_image(self, window=None, bands=None):
        
        with rasterio.open(self.image_path, 'r') as file:
            image = file.read(
                window=window,
                out_dtype=np.float32,
                indexes=bands
            )
            
        return image
    
    def read_label(self, window=None, merge=None):
        
        with rasterio.open(self.label_path, 'r') as file:
            label = file.read(
                window=window,
                out_dtype=np.uint8
            )
            
        if merge is not None:
            label = merge_labels(
                label.squeeze(),
                [list(l.values) for l in self.classes[merge].value]
            )
        
        return label