import numpy as np
import torch
from dl_toolbox.utils import get_tiles


class NoOp:
    def __init__(self, p=1):
        pass

    def __call__(self, img, label=None):
        return img, label
    
class TTA:
    def __init__(self, transforms, reverse):
        self.transforms = transforms
        self.reverse = reverse

    def __call__(self, img):
        imgs = []
        for t in self.transforms:
            imgs.append(t(img, None)[0])
        return imgs
    
    def revert(self, imgs):
        res = []
        for r, img in zip(self.reverse, imgs):
            res.append(r(img, None)[0])
        return res
    
class Sliding:
    def __init__(
        self,
        nols,
        nrows,
        width,
        height,
        step_w,
        step_h
    ):
        self.nols = nols
        self.nrows = nrows
        self.tiles = list(get_tiles(nols, nrows, width, height, step_w, step_h))
        
    def __call__(self, img):
        imgs = []
        for co, ro, w, h in self.tiles:
            imgs.append(img[...,ro:ro+h,co:co+w])
        return imgs
    
    def merge(self, preds):
        bs, nc, h, w = preds[0].shape
        d = preds[0].device
        #mask = torch.ones((h, w)).float().to(d)
        merged = torch.zeros((bs, nc, self.nrows, self.nols)).to(d)
        weight = torch.zeros((bs, nc, self.nrows, self.nols)).to(d)
        for (co, ro, w, h), pred in zip(self.tiles, preds):
            merged[:, :, ro:ro+h, co:co+w] += pred #mask * pred
            weight[:, :, ro:ro+h, co:co+w] += 1 #mask
        return torch.div(merged, weight)
        
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label=None):
        for t in self.transforms:
            img, label = t(img, label)
        return img, label


class OneOf:
    def __init__(self, transforms, transforms_ps):
        self.transforms = transforms
        s = sum(transforms_ps)
        self.transforms_ps = [p / s for p in transforms_ps]

    def __call__(self, img, label=None):
        t = np.random.choice(self.transforms, p=self.transforms_ps)
        img, label = t(img, label)
        return img, label


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def stretch_to_minmax(img, mins, maxs):
    res = (img - mins) / (maxs - mins)
    return torch.clip(res, 0, 1)
