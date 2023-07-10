import torchvision.transforms.functional as F
from .utils import stretch_to_minmax
import torch
import numpy as np
        
class StretchToMinmaxCommon:
    
    def __init__(self, minval, maxval):
        
        self.mins = torch.Tensor(minval).reshape((-1, 1, 1))
        self.maxs = torch.Tensor(maxval).reshape((-1, 1, 1))
        
    def __call__(self, img, label=None):   
        
        img = stretch_to_minmax(img, self.mins, self.maxs)
        
        return img, label
        
class StretchToMinmaxBySource:
    
    def __init__(self, source):
        
        bands = np.array(source.bands)-1
        self.mins = np.array(source.minval, dtype=np.float32)[bands].reshape((-1, 1, 1))        
        self.maxs = np.array(source.maxval, dtype=np.float32)[bands].reshape((-1, 1, 1))
        
    def __call__(self, img):   
        
        img = stretch_to_minmax(img, self.mins, self.maxs)
        
        return img 
        
        