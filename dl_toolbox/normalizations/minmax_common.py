import torchvision.transforms.functional as F
from .utils import stretch_to_minmax
import torch
import numpy as np


class StretchToMinmaxCommon:
    
    def __init__(self, minval, maxval, meanval):
                
        self.minval = np.array(minval, dtype=np.float32).reshape((-1, 1, 1))
        self.maxval = np.array(maxval, dtype=np.float32).reshape((-1, 1, 1))
        
    def __call__(self, img, source):
        
        bands = np.array(source.bands)-1
        mins = torch.from_numpy(self.minval[bands])
        maxs = torch.from_numpy(self.maxval[bands])
                
        img = stretch_to_minmax(img, mins, maxs)
        img = torch.clip(img, 0, 1)
                
        return img