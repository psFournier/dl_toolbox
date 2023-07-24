import numpy as np
import torch


def worker_init_function(id):
    uint64_seed = torch.initial_seed()
    np.random.seed([uint64_seed >> 32, uint64_seed & 0xFFFF_FFFF])
