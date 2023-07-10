import torchvision.transforms.functional as F
from .utils import stretch_to_minmax
import torch
import numpy as np


class ZeroAverageCommon:
    
    def __init__(self, meanval):
        
        self.means = torch.Tensor(meanval).reshape((-1, 1, 1))
        
    def __call__(self, img, label=None):   
                
        return img - self.means, label
    
class ZeroAverageBySource:
    
    def __init__(self, source):
        
        bands = np.array(source.bands)-1
        means = np.array(source.meanval, dtype=np.float32).reshape((-1, 1, 1))
        self.means = torch.from_numpy(means[bands])
                
    def __call__(self, img):   
                
        return img - self.means