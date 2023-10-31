import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from pytorch_lightning.callbacks import BasePredictionWriter
import rasterio.windows as W


class TiffPredsWriter(BasePredictionWriter):
    def __init__(self, out_path, base):
        super().__init__(write_interval="batch")
        self.out_path = Path(out_path)
        self.out_path.mkdir(parents=False, exist_ok=False)
        self.base = Path(base)
        self.stats = {'img':[], 'msk': [], 'win': [], 'avg_cert': []}

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        probs = outputs.cpu()
        confs, preds = pl_module.probas2confpreds(probs)
        for p, c, path, win in zip(preds, confs, batch["image_path"], batch["window"]):
            r = Path(path).relative_to(self.base) 
            co, ro, _, _ = win
            new_rel_path = r.parent/(str(r.stem)+f'_{co}_{ro}'+r.suffix)
            out_msk = self.out_path/new_rel_path   
            self.stats['msk'].append(new_rel_path)
            self.stats['img'].append(r)
            self.stats['win'].append("_".join([str(i) for i in win]))
            self.stats['avg_cert'].append(float(c.mean()))
            out_msk.parent.mkdir(exist_ok=True, parents=True)
            with rasterio.open(path) as img:
                meta = img.meta
            meta["count"] = 1 #pl_module.num_classes
            meta["dtype"] = np.uint8 #np.float32
            meta["nodata"] = None
            meta["width"], meta["height"] = tuple(p.shape)
            meta["transform"] = W.transform(W.Window(*win), meta["transform"])
            with rasterio.open(out_msk, "w+", **meta) as dst:
                dst.write(p.numpy()[np.newaxis,...])
                
    def on_predict_epoch_end(self, trainer, pl_module):
        stats = pd.DataFrame(self.stats, columns=['msk', 'img', 'win', 'avg_cert'])
        stats.to_csv(self.out_path / 'stats.csv')
        #df = pd.read_csv(self.out_path / 'stats.csv', index_col=0)
        #print(df['avg_cert'].loc['/data/outputs/flair2_3_97/supervised_dummy/2023-09-05_102306/checkpoints/last_preds/FLAIR_1/train/D007_2020/Z1_AA/img/IMG_003576.tif'])