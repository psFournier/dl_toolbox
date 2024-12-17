import os
from functools import partial
from pathlib import Path

import numpy as np
import rasterio
import rasterio.windows as windows
import torch
from pytorch_lightning import Callback


def dist_to_edge_mask(crop_size):
    def dist_to_edge(i, j, h, w):
        mi = np.minimum(i + 1, h - i)
        mj = np.minimum(j + 1, w - j)
        return np.minimum(mi, mj)

    crop_mask = np.fromfunction(
        function=partial(dist_to_edge, h=crop_size, w=crop_size),
        shape=(crop_size, crop_size),
        dtype=int,
    )
    crop_mask = torch.from_numpy(crop_mask).float()

    return crop_mask

class MergePreds(Callback):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def on_predict_epoch_start(self, trainer, pl_module):
        w, h = self.size
        self.merged = torch.zeros((pl_module.num_classes, h, w))
        self.weights = torch.zeros((h, w))

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        logits = outputs.cpu()
        #probas = pl_module.logits2probas(logits)
        crops = batch["window"]
        for logit, crop in zip(logits, crops):
            co, ro, w, h = crop
            mask = torch.ones((h, w)).float()
            #mask = dist_to_edge_mask(crop_size)
            self.merged[:, ro:ro+h, co:co+w] += (logit * mask)
            self.weights[ro:ro+h, co:co+w] += mask

    def on_predict_epoch_end(self, trainer, pl_module):
        self.merged = torch.div(self.merged, self.weights)