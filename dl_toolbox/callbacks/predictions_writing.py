import os
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import rasterio
import torch
import torchvision
from pytorch_lightning.callbacks import BasePredictionWriter
import rasterio.windows as W
import torchmetrics.functional.classification as metrics

class PredictionsWriting(BasePredictionWriter):

    def __init__(self, every_n_batch, mode, out_path, base_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = every_n_batch
        assert mode=='segmentation'
        self.mode = mode
        self.out_path = Path(out_path)
        self.out_path.mkdir(parents=False, exist_ok=False)
        self.base_path = Path(base_path)
        self.stats = {'img_path': [], 'pred_path': [], 'conf': [], 'acc': []}
        
    def write_batch(self, trainer, module, outputs, batch):
        paths = batch['image_path']
        probs = outputs.cpu()
        confs, preds = torch.max(probs, dim=1)
        y = batch['target'].cpu() # deal later with no target case
        for msk, pred, conf, path in zip(y, preds, confs, paths):
            acc = metrics.accuracy(pred, msk, **module.metric_args)
            rel_path = Path(path).relative_to(self.base_path)    
            pred_path = self.out_path/rel_path
            pred_path.parent.mkdir(exist_ok=True, parents=True)
            im = Image.fromarray(pred.numpy().astype(np.uint8))
            im.save(fp=str(pred_path))
            self.stats['pred_path'].append(pred_path)
            self.stats['img_path'].append(path)
            self.stats['conf'].append(float(conf.mean()))
            self.stats['acc'].append(acc)
            
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.freq>0 and batch_idx % self.freq == 0:
            self.write_batch(trainer, pl_module, outputs, batch)
            
    def on_predict_epoch_end(self, trainer, pl_module):
        stats = pd.DataFrame(self.stats, columns=['img_path', 'pred_path', 'conf', 'acc'])
        stats.to_csv(self.out_path / 'stats.csv')