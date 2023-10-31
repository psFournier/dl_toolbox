import os
import numpy as np
import pandas as pd
from pathlib import Path

from pytorch_lightning.callbacks import BasePredictionWriter


class ClassifPredsWriter(BasePredictionWriter):
    def __init__(self, out_path, base):
        super().__init__(write_interval="batch")
        self.out_path = Path(out_path)
        self.out_path.mkdir(exist_ok=True, parents=True)
        self.base = Path(base)
        self.stats = {'img':[], 'preds': [], 'confs': []}
        
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        probs = outputs.cpu()
        confs, preds = pl_module.probas2confpreds(probs)
        for p, c, path in zip(preds, confs, batch["path"]):
            relative = Path(path).relative_to(self.base) 
            self.stats['img'].append(relative)
            self.stats['preds'].append(int(p))
            self.stats['confs'].append(float(c))
                        
    def on_predict_epoch_end(self, trainer, pl_module):
        stats_df = pd.DataFrame(
            zip(self.stats['preds'], self.stats['confs']),
            index=self.stats['img'],
            columns = ['prediction', 'confidence']
        )
        stats_df.to_csv(self.out_path / 'stats.csv')


