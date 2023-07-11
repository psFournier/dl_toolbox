from .utils import stretch_to_minmax
from torch import Tensor
import numpy as np
        
class StretchToMinmaxCommon:
    
    def __init__(self, minval, maxval):
        
        self.mins = Tensor(minval).reshape((-1, 1, 1))
        self.maxs = Tensor(maxval).reshape((-1, 1, 1))
        
    def __call__(self, img, label=None):   
        
        img = stretch_to_minmax(img, self.mins, self.maxs)
        
        return img, label
        
class StretchToMinmaxBySource:
    
    def __init__(self, source):
        
        minval = [source.minval[i-1] for i in source.bands]
        self.mins = Tensor(minval).reshape((-1,1,1))
        maxval = [source.maxval[i-1] for i in source.bands]
        self.maxs = Tensor(maxval).reshape((-1,1,1))
        
    def __call__(self, img, label=None):   
        
        img = stretch_to_minmax(img, self.mins, self.maxs)
        
        return img, label
        
        