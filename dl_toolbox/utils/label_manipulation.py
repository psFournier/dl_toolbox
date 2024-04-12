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
    rgb = np.zeros(shape=(*labels.shape, 3), dtype=np.uint8)
    for label, color in colors:
        mask = np.array(labels == label)
        rgb[mask] = np.array(color)
    return rgb

def rgb_to_labels(rgb, colors):
    labels = np.zeros(shape=rgb.shape[:-1], dtype=np.uint8)
    for label, color in colors:
        d = rgb[..., 0] == color[0]
        d = np.logical_and(d, (rgb[..., 1] == color[1]))
        d = np.logical_and(d, (rgb[..., 2] == color[2]))
        labels[d] = label
    return labels


class LabelsToRGB:
    # Inputs shape : B,H,W or H,W
    # Outputs shape : B,H,W,3 or H,W,3

    def __init__(self, labels):
        self.labels = labels

    def __call__(self, labels):
        rgb = np.zeros(shape=(*labels.shape, 3), dtype=np.uint8)
        for label, key in enumerate(self.labels):
            mask = np.array(labels == label)
            rgb[mask] = np.array(self.labels[key]["color"])

        return rgb


class NomencToRgb:
    def __init__(self, nomenc):
        self.nomenc = nomenc

    def __call__(self, labels):
        rgb = np.zeros(shape=(*labels.shape, 3), dtype=np.uint8)

        for label, key in enumerate(self.nomenc):
            mask = np.array(labels == label)
            rgb[mask] = np.array(key.color)

        return rgb


class RGBToLabels:
    # Inputs shape : B,H,W,3 or H,W,3
    # Outputs shape : B,H,W or H,W
    def __init__(self, labels):
        self.labels = labels

    def __call__(self, rgb):
        labels = np.zeros(shape=rgb.shape[:-1], dtype=np.uint8)
        for label, key in enumerate(self.labels):
            c = self.labels[key]["color"]
            d = rgb[..., 0] == c[0]
            d = np.logical_and(d, (rgb[..., 1] == c[1]))
            d = np.logical_and(d, (rgb[..., 2] == c[2]))
            labels[d] = label

        return labels
