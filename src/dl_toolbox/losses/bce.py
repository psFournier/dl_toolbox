import torch
from scipy import ndimage
import numpy as np

def w_from_dist_to_edge(binary, factor=0.1):
    #binary = ndimage.binary_erosion(binary).astype(int)
    edges = binary - ndimage.binary_erosion(binary, structure=np.ones((1,3,3)))
    dist = ndimage.distance_transform_edt(1-edges, sampling=[1000, 1, 1])+1
    w = 1/dist**factor
    return w

class BCE(torch.nn.BCEWithLogitsLoss):

    def __init__(
        self,
        #predict_zero,
        pixel_weights,
        factor,
        *args,
        **kwargs
    ):
        if pixel_weights is not None:
            kwargs.update(reduction='none')
        super().__init__(*args, **kwargs)
        #self.predict_zero = predict_zero
        self.pixel_weights = pixel_weights
        self.factor = factor
        self.__name__ = 'binary cross entropy'


    def forward(self, logits, targets):
        #if not self.predict_zero:
        #    logits=logits[:,1:,...]
        #    targets=targets[:,1:,...]
        loss = super().forward(logits, targets.float()) # bce requires float
        if self.pixel_weights is not None:
            t = targets.cpu()
            w = torch.Tensor(w_from_dist_to_edge(t,self.factor)).to(loss.device)
            return (loss * w).sum() / w.sum()
        return loss
        
    def prob(self, logits):
        #if not self.predict_zero:
        #    p = torch.sigmoid(logits[:,1:,...])
        #    confs, preds = torch.max(p, axis=1)
        #    p_zero = torch.unsqueeze(1 - confs, 1)
        #    return torch.cat([p_zero, p], axis=1)
        return torch.sigmoid(logits)
    
    def pred(self, probs):
        return probs.ge(0.5).long()