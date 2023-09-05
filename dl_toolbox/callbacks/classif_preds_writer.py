import os
import numpy as np
import pandas as pd
from pathlib import Path

from pytorch_lightning.callbacks import BasePredictionWriter


class ClassifPredsWriter(BasePredictionWriter):
    def __init__(self, out_path, base, cls_names):
        super().__init__(write_interval="batch")
        self.out_path = Path(out_path)
        self.out_path.mkdir(exist_ok=True, parents=True)
        self.base = Path(base)
        self.stats = {'img':[], 'probas': []}
        self.cls_names = cls_names
        
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        probas = pl_module.logits2probas(outputs.cpu())
        for p, path in zip(probas, batch["path"]):
            relative = Path(path).relative_to(self.base) 
            self.stats['img'].append(relative)
            self.stats['probas'].append(p.numpy())
                
    def on_predict_epoch_end(self, trainer, pl_module):
        stats_df = pd.DataFrame(
            np.stack(self.stats['probas']),
            index=self.stats['img'],
            columns = self.cls_names
        )
        stats_df.to_csv(self.out_path / 'stats.csv')


