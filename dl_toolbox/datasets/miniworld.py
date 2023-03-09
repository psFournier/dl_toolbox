import torch
import rasterio
import numpy as np

from dl_toolbox.utils import minmax
from dl_toolbox.torch_datasets import RasterDs


labels_dict = {
    
    'base':{
        'other': {'color': (0, 0, 0)},
        'building': {'color': (255, 255, 255)},
    }
    
}

class Miniworld(RasterDs):

    def __init__(self, labels, *args, **kwargs):
 
        self.labels = labels_dict[labels]
        super().__init__(*args, **kwargs)

    def read_image(self, image_path, window):
        
        with rasterio.open(image_path) as image_file:
            image = image_file.read(window=window, out_dtype=np.float32)
            
        mins = np.array([stat.min for stat in self.info['stats']])
        maxs = np.array([stat.max for stat in self.info['stats']])
        image = minmax(image[:3], mins[:3], maxs[:3])

        return image

    def read_label(self, label_path, window):
    
        with rasterio.open(label_path) as label_file:
            label = label_file.read(window=window, out_dtype=np.float32)
            
        label = np.squeeze(label) / 255
        
        return label