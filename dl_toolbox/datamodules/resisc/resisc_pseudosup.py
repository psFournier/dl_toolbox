import os
import random
import pandas as pd
from pathlib import Path
from functools import partial

import numpy as np
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import CombinedLoader
from torch.utils.data import DataLoader, Subset

from dl_toolbox.datasets import DatasetResisc
from dl_toolbox.utils import CustomCollate

from .resisc import DatamoduleResisc1


class ResiscPseudosup(DatamoduleResisc1):

    def __init__(
        self,
        pl_dir,
        thresh,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.pl_dir = Path(pl_dir)
        self.stats = pd.read_csv(self.pl_dir/'stats.csv', index_col=0)
        self.stats.sort_values('confidence', ascending=False, inplace=True)
        self.thresh = thresh

    def prepare_data(self):
        super().prepare_data()  
        top_pl_img = list(self.stats.index[:self.thresh])
        top_pl_preds = list(self.stats['prediction'][:self.thresh])
        for img, pred in zip(top_pl_img, top_pl_preds):
            name = self.class_names[pred]
            dst = self.pl_dir/img
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(src=self.data_path/img, dst=dst)        

    def setup(self, stage):
        super().setup(stage)
        data_path = self.pl_dir/'NWPU-RESISC45'
        if stage in ("fit"):
            self.pl_set = DatasetResisc(data_path, self.train_tf, self.merge),
        
    def train_dataloader(self):
        train_dataloaders = {}
        train_dataloaders["sup"] = self.get_loader(self.train_set)(
            shuffle=True,
            drop_last=True,
        )
        train_dataloaders["pseudosup"] = self.get_loader(self.pl_set)(
            shuffle=True,
            drop_last=True
        )
        return CombinedLoader(train_dataloaders, mode="max_size_cycle")