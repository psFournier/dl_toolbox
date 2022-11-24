from typing import Callable

import pandas as pd
import torch
import torch.utils.data
import torchvision


class BalancedConcat(torch.utils.data.sampler.Sampler):
    """
    """

    def __init__(
        self,
        lengths,
        num_samples
    ):

        weights = [1.0 / length for length in lengths for _ in range(length)]
        self.weights = torch.Tensor(weights).double()
        self.num_samples = num_samples

    def __iter__(self):
        return (i for i in torch.multinomial(self.weights, self.num_samples,
                                             replacement=True))

    def __len__(self):
        return self.num_samples
