import os
import random
import pandas as pd
from pathlib import Path
from functools import partial

import numpy as np
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import CombinedLoader
from torch.utils.data import DataLoader, Subset

import dl_toolbox.datasets as datasets
from dl_toolbox.utils import CustomCollate

from .resisc import Resisc

import os, errno

def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e

class ResiscPseudosup(Resisc):

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
        counts = [0] * self.num_classes
        for img, pred in zip(top_pl_img, top_pl_preds):
            class_name = self.class_names[pred]
            pl_name = class_name+f'_{counts[pred]:03d}.jpg'
            dst = self.pl_dir/'NWPU-RESISC45'/class_name/pl_name
            dst.parent.mkdir(parents=True, exist_ok=True)
            #os.symlink(src=self.data_path/img, dst=dst)   
            symlink_force(target=self.data_path/img, link_name=dst)
            counts[pred] += 1

    def setup(self, stage):
        super().setup(stage)
        data_path = self.pl_dir/'NWPU-RESISC45'
        if stage in ("fit"):
            self.pl_set = datasets.Resisc(data_path, self.dataset_tf, self.merge)
        
    def train_dataloader(self):
        train_dataloaders = {}
        train_dataloaders["sup"] = DataLoader(
            dataset=self.train_set,
            collate_fn=CustomCollate(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True
        )
        train_dataloaders["pseudosup"] = DataLoader(
            dataset=self.pl_set,
            collate_fn=CustomCollate(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True
        )
        return CombinedLoader(train_dataloaders, mode="max_size_cycle")