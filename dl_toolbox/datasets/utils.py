import random

import rasterio.windows as windows

from dl_toolbox.utils import get_tiles
from collections import namedtuple
import enum

label = namedtuple(
    'label',
    [
        'name',
        'color',
        'values'
    ]
)

class FixedCropFromWindow:
    
    def __init__(self, window, crop_size, crop_step=None):
        
        self.window = window
        self.crop_size = crop_size
        self.crop_step = crop_step
        
        self.crops = [
            crop for crop in get_tiles(
                nols=window.width, 
                nrows=window.height, 
                size=crop_size, 
                step=crop_step if crop_step else crop_size,
                row_offset=window.row_off, 
                col_offset=window.col_off
            )
        ]         
    
    def __call__(self, idx):
        
        return self.crops[idx]  
    
class RandomCropFromWindow:
    
    def __init__(self, window, crop_size):
        
        self.window = window
        self.crop_size = crop_size        
    
    def __call__(self, idx):
        
        col_off, row_off, width, height = self.window.flatten()
        cx = col_off + random.randint(0, width - self.crop_size)
        cy = row_off + random.randint(0, height - self.crop_size)
        crop = windows.Window(cx, cy, self.crop_size, self.crop_size)
        
        return crop