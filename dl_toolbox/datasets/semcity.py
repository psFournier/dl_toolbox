import torch
import rasterio
import numpy as np

from dl_toolbox.utils import minmax, MergeLabels
from dl_toolbox.torch_datasets import RasterDs


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

class Semcity(RasterDs):

    def __init__(self, labels, bands, *args, **kwargs):
        
        self.labels = semcity_labels[labels]
        self.bands = bands
        super().__init__(*args, **kwargs)
        self.label_merger = MergeLabels(mergers[labels])

    def read_image(self, image_path, window):

        with rasterio.open(image_path) as image_file:
            image = image_file.read(window=window, out_dtype=np.float32, indexes=self.bands)
            
        bands_idxs = np.array(self.bands) - 1
        image = minmax(image, self.mins[bands_idxs], self.maxs[bands_idxs])
            
        return image

    def read_label(self, label_path, window):
 
        with rasterio.open(label_path) as label_file:
            rgb = label_file.read(window=window, out_dtype=np.float32)
            
        rgb = rgb.transpose((1,2,0))
        labels = np.zeros(shape=rgb.shape[:-1], dtype=np.uint8)
        for label, key in enumerate(semcity_labels['base']):
            c = semcity_labels['base'][key]['color']
            d = rgb[..., 0] == c[0]
            d = np.logical_and(d, (rgb[..., 1] == c[1]))
            d = np.logical_and(d, (rgb[..., 2] == c[2]))
            labels[d] = label
        label = self.label_merger(labels)

        return label