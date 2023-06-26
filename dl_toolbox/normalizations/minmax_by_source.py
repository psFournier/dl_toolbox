import torchvision.transforms.functional as F
from .utils import stretch_to_minmax
import torch
import numpy as np
        
class StretchToMinmaxBySource:
        
    def __call__(self, img, source):   
        
        bands = np.array(source.bands)-1
        mins = torch.from_numpy(source.minval[bands])
        maxs = torch.from_numpy(source.maxval[bands])
        means = torch.from_numpy(source.meanval[bands])
        
        img = stretch_to_minmax(img, mins, maxs)
        img = torch.clip(img, 0, 1)
        img -= means
        
        return img 
        