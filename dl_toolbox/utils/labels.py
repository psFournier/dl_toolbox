import numpy as np
import torch
from collections import namedtuple


label = namedtuple("label", ["name", "color", "values"])

def merge_labels(labels, merge):
    ret = torch.zeros(labels.shape, dtype=labels.dtype)
    for i, val in enumerate(merge):
        for j in val:
            ret[labels == j] = i
    return ret

def labels_to_rgb(labels, colors):
    rgb = torch.zeros((*labels.shape, 3), dtype=torch.uint8)
    for label, color in colors:
        rgb[labels == label] = torch.Tensor(color)
    return rgb

#def rgb_to_labels(rgb, colors):
#    labels = np.zeros(shape=rgb.shape[:-1], dtype=np.uint8)
#    for label, color in colors:
#        d = rgb[..., 0] == color[0]
#        d = np.logical_and(d, (rgb[..., 1] == color[1]))
#        d = np.logical_and(d, (rgb[..., 2] == color[2]))
#        labels[d] = label
#    return labels