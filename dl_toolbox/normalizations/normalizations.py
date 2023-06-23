import torchvision.transforms.functional as F
from .utils import stretch_to_minmax
import torch
import numpy as np

class ImagenetNormalize:

    def __call__(self, img, label=None):

        img = F.normalize(
                img,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                )
        return img, label

class StretchToMinmaxCommon:
    
    def __init__(self, minval, maxval, meanval):
                
        self.minval = np.array(minval).reshape((-1, 1, 1))
        self.maxval = np.array(maxval).reshape((-1, 1, 1))
        self.meanval = np.array(meanval).reshape((-1, 1, 1))
        
    def __call__(self, img, source):
        
        bands = np.array(source.bands)-1
        mins = torch.from_numpy(self.minval[bands])
        maxs = torch.from_numpy(self.maxval[bands])
        means = torch.from_numpy(self.meanval[bands])
        
        img = stretch_to_minmax(img, mins, maxs)
        img = torch.clip(img, 0, 1)
        img -= means
        
        return img
        
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
        