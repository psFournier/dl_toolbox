import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from pytorch_lightning.callbacks import BasePredictionWriter
import rasterio.windows as W
import torchmetrics.functional.classification as metrics

class TiffPredsWriter(BasePredictionWriter):
    def __init__(self, out_path, base):
        super().__init__(write_interval="batch")
        self.out_path = Path(out_path)
        self.out_path.mkdir(parents=False, exist_ok=False)
        self.base = Path(base)
        self.stats = {'img_path': [], 'pred_path': [], 'conf': [], 'acc': []}

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        probs = outputs.cpu()
        confs, preds = pl_module.loss.pred(probs)
        _, y, paths = batch
        for m, p, c, path in zip(y['masks'], preds, confs, paths):
            acc = metrics.accuracy(p, m.cpu(), **pl_module.metric_args)
            pred = p.numpy()[np.newaxis,...]
            r = Path(path).relative_to(self.base)    
            pred_path = self.out_path/r
            pred_path.parent.mkdir(exist_ok=True, parents=True)
            with rasterio.open(path) as img:
                meta = img.meta
            meta["count"] = 1 #pl_module.num_classes
            meta["dtype"] = np.uint8 #np.float32
            meta["nodata"] = None
            meta["width"], meta["height"] = tuple(p.shape)
            #meta["transform"] = W.transform(W.Window(*win), meta["transform"])
            with rasterio.open(pred_path, "w+", **meta) as dst:
                dst.write(pred)
            self.stats['pred_path'].append(pred_path)
            self.stats['img_path'].append(path)
            self.stats['conf'].append(float(c.mean()))
            self.stats['acc'].append(acc)
                
    def on_predict_epoch_end(self, trainer, pl_module):
        stats = pd.DataFrame(self.stats, columns=['img_path', 'pred_path', 'conf', 'acc'])
        stats.to_csv(self.out_path / 'stats.csv')