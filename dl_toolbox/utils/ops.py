import torch
import torch.nn.functional as F

def one_hot_mask(mask, num_classes):
    assert mask.ndim==2 or mask.ndim==3
    onehotdim = 0 if mask.ndim==2 else 1 
    return torch.movedim(F.one_hot(mask, num_classes),-1,onehotdim)