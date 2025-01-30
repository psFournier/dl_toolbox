import numpy as np
import torch 
import pandas as pd

from .utils import stretch_to_minmax


class To_0_1:
    def __init__(self, mins, maxs):
        self.mins = torch.Tensor(mins).reshape((-1, 1, 1))
        self.maxs = torch.Tensor(maxs).reshape((-1, 1, 1))

    def __call__(self, img, label=None):
        img = stretch_to_minmax(img, self.mins, self.maxs)
        return img, label
    
class To_0_1_fromCsv:
    def __init__(self, csv, min_p, max_p, bands):
        stats = pd.read_csv(csv, index_col=0)
        mins = [stats[min_p].loc[f'band_{i}'] for i in bands]
        maxs = [stats[max_p].loc[f'band_{i}'] for i in bands]
        self.mins = torch.Tensor(mins).reshape((-1, 1, 1))
        self.maxs = torch.Tensor(maxs).reshape((-1, 1, 1))

    def __call__(self, img, label=None):
        img = stretch_to_minmax(img, self.mins, self.maxs)
        return img, label

class To_0_1_fromNpy:
    def __init__(self, npy, bands, city):
        stats = np.load(npy, allow_pickle=True).item()
        city_stats = stats[city.title()]
        mins = [0. for _ in bands]
        maxs = [city_stats[f'perc_995'][i-1] for i in bands]
        self.mins = torch.Tensor(mins).reshape((-1, 1, 1))
        self.maxs = torch.Tensor(maxs).reshape((-1, 1, 1))

    def __call__(self, img, label=None):
        img = stretch_to_minmax(img, self.mins, self.maxs)
        return img, label
    
class NormalizeFromNpy:
    def __init__(self, npy, min_p, max_p, bands, city):
        stats = np.load(npy, allow_pickle=True).item()
        city_stats = stats[city.title()]
        #mins = [city_stats[f'perc_{min_p}'][i-1] for i in bands]
        mins = [0. for _ in bands]
        maxs = [city_stats[f'perc_{max_p}'][i-1] for i in bands]
        means = [city_stats[f'mean_clip{max_p}'][i-1] for i in bands]
        stds = [city_stats[f'std_clip{max_p}'][i-1] for i in bands]
        self.mins, self.maxs, self.means, self.stds = map(
            lambda x: torch.Tensor(x).reshape((-1,1,1)),
            [mins, maxs, means, stds]
        )

    def __call__(self, img, label=None):
        img = torch.clamp(img, min=self.mins, max=self.maxs)
        img -= self.means
        img /= self.stds
        return img, label
    


#class StretchToMinmaxBySource:
#    def __init__(self, source, bands):
#        minval = [source.stats["p0"][i - 1] for i in bands]
#        self.mins = torch.Tensor(minval).reshape((-1, 1, 1))
#        maxval = [source.stats["p100"][i - 1] for i in bands]
#        self.maxs = torch.Tensor(maxval).reshape((-1, 1, 1))
#
#    def __call__(self, img, label=None):
#        img = stretch_to_minmax(img, self.mins, self.maxs)
#
#        return img, label
