import torchvision.transforms.functional as F
from .utils import stretch_to_minmax
import torch
import numpy as np
        
class StretchToMinmaxCommon:
    
    def __init__(self, minval, maxval, source):
        
        bands = np.array(source.bands)-1
        self.mins = np.array(minval, dtype=np.float32)[bands].reshape((-1, 1, 1))
        self.maxs = np.array(maxval, dtype=np.float32)[bands].reshape((-1, 1, 1))
        
    def __call__(self, img):   
        
        img = stretch_to_minmax(img, self.mins, self.maxs)
        img = np.clip(img, 0, 1)
        
        return img 
        
class StretchToMinmaxBySource:
    
    def __init__(self, source):
        
        bands = np.array(source.bands)-1
        self.mins = np.array(source.minval, dtype=np.float32)[bands].reshape((-1, 1, 1))        
        self.maxs = np.array(source.maxval, dtype=np.float32)[bands].reshape((-1, 1, 1))
        
    def __call__(self, img):   
        
        img = stretch_to_minmax(img, self.mins, self.maxs)
        img = np.clip(img, 0, 1)
        
        return img 
        
        