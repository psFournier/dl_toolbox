import torchvision.transforms.functional as F
from .utils import stretch_to_minmax

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
        
        self.minval = minval
        self.maxval = maxval
        self.meanval = meanval
        
    def __call__(self, img, source=None):
        
        img = stretch_to_minmax(img, self.minval, self.maxval)
        img = torch.clip(img, 0, 1)
        img -= self.meanval
        
        return img
        
class StretchToMinmaxBySource:
        
    def __call__(self, img, source):        
        
        img = stretch_to_minmax(img, source.minval, source.maxval)
        img = torch.clip(img, 0, 1)
        img -= source.meanval
        
        return img 
        