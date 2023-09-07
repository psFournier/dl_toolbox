import numpy as np
from torch import Tensor
import csv
import pandas as pd

from .utils import stretch_to_minmax


class StretchToMinmax:
    def __init__(self, mins, maxs):
        self.mins = Tensor(mins).reshape((-1, 1, 1))
        self.maxs = Tensor(maxs).reshape((-1, 1, 1))

    def __call__(self, img, label=None):
        img = stretch_to_minmax(img, self.mins, self.maxs)
        return img, label
    
class StretchToMinmaxFromCsv:
    def __init__(self, csv, min_p, max_p, bands):
        stats = pd.read_csv(csv, index_col=0)
        mins = [stats[min_p].loc[f'band_{i}'] for i in bands]
        maxs = [stats[max_p].loc[f'band_{i}'] for i in bands]
        self.mins = Tensor(mins).reshape((-1, 1, 1))
        self.maxs = Tensor(maxs).reshape((-1, 1, 1))

    def __call__(self, img, label=None):
        img = stretch_to_minmax(img, self.mins, self.maxs)
        return img, label


#class StretchToMinmaxBySource:
#    def __init__(self, source, bands):
#        minval = [source.stats["p0"][i - 1] for i in bands]
#        self.mins = Tensor(minval).reshape((-1, 1, 1))
#        maxval = [source.stats["p100"][i - 1] for i in bands]
#        self.maxs = Tensor(maxval).reshape((-1, 1, 1))
#
#    def __call__(self, img, label=None):
#        img = stretch_to_minmax(img, self.mins, self.maxs)
#
#        return img, label
