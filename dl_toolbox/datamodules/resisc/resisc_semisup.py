from pathlib import Path
from functools import partial

import numpy as np
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import CombinedLoader
from torch.utils.data import DataLoader, Subset

from dl_toolbox.datasets import DatasetResisc
from dl_toolbox.utils import CustomCollate

from .resisc import DatamoduleResisc1

class DatamoduleResisc1Semisup(DatamoduleResisc1):
    
    def __init__(
        self,
        unlabeled_prop,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.unlabeled_prop = unlabeled_prop
        
    def prepare_data(self):
        num_item = sum([len(label.values)*700 for label in self.classes])
        self.train_idx, self.val_idx, self.test_idx = [], [], []
        self.unlabeled_idx = []
        for i in range(num_item):
            if self.unlabeled_prop[0] <= i%100 <= self.unlabeled_prop[1]:
                self.unlabeled_idx.append(i)
            #if self.prop <= i%100 <= self.prop + self.unlabeled_prop: self.unlabeled_idx.append(i)
            if i%100 < self.prop: self.train_idx.append(i)
            elif i%100 >= 90: self.val_idx.append(i)
            else: self.test_idx.append(i) 
            
    def setup(self, stage):
        super().setup(stage)
        if stage in ("fit"):
            self.unlabeled_set = Subset(
                DatasetResisc(self.data_path, self.train_tf, self.merge),
                indices=self.unlabeled_idx,
            )

    def train_dataloader(self):
        train_dataloaders = {}
        train_dataloaders["sup"] = self.get_loader(self.train_set)(
            shuffle=True,
            drop_last=True,
        )
        train_dataloaders["unsup"] = self.get_loader(self.unlabeled_set)(
            shuffle=True,
            drop_last=True
        )
        return CombinedLoader(train_dataloaders, mode="max_size_cycle")