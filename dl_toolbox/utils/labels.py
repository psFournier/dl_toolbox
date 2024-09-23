import numpy as np
import torch
from collections import namedtuple
from functools import reduce


label = namedtuple("label", ["name", "color", "values"])

def merge_labels(labels, merge):
    ret = torch.zeros(labels.shape, dtype=labels.dtype)
    for i, val in enumerate(merge):
        for j in val:
            ret[labels == j] = i
    return ret

def merge_labels_boxes(labels, boxes, merge):
    """
    Args: 
        labels: tensor shape L
        boxes: tensor shape Lx4
    Returns:

    """
    merged_labels = []
    merged_boxes = []
    for i, l in enumerate(merge, 1): 
        # indices of tgts whose label belongs to the i-th merge
        idx = reduce(torch.logical_or, [labels == v for v in l.values])
        # i+1 because in detection, class label 0 should be left for no-obj in algos
        merged_labels.append(i * torch.ones_like(labels[idx]))
        merged_boxes.append(boxes[idx])
    merged_labels = torch.cat(merged_labels, dim=0)
    merged_boxes = torch.cat(merged_boxes, dim=0)
    return merged_labels, merged_boxes

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