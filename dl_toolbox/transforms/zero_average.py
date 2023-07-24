from .utils import stretch_to_minmax
from torch import Tensor
import numpy as np


class ZeroAverageCommon:
    def __init__(self, meanval):
        self.means = Tensor(meanval).reshape((-1, 1, 1))

    def __call__(self, img, label=None):
        return img - self.means, label


class ZeroAverageBySource:
    def __init__(self, source, bands):
        meanval = [source.meanval[i - 1] for i in bands]
        self.means = Tensor(meanval).reshape((-1, 1, 1))

    def __call__(self, img, label=None):
        return img - self.means, label
