from typing import Callable

import pandas as pd
import torch
import torch.utils.data
import torchvision


class BalancedConcatSampler(torch.utils.data.sampler.Sampler):
    """
    """

    def __init__(
        self,
        lengths
    ):

        weights = [1.0 / length for length in lengths]
        self.weights = torch.DoubleTensor(weights)

   def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
