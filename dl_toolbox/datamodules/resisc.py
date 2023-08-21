import os
import random
import re
from os.path import join
from pathlib import Path
from functools import partial

import numpy as np
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import CombinedLoader
from torch.utils.data import DataLoader, Subset

from dl_toolbox.datasets import DatasetResisc
from dl_toolbox.utils import CustomCollate


class DatamoduleResisc1(LightningDataModule):
    
    def __init__(
        self,
        data_path,
        merge,
        prop,
        train_tf,
        val_tf,
        test_tf,
        batch_size,
        num_workers,
        pin_memory,
        class_weights=None,
        *args,
        **kwargs
    ):
        super().__init__()
        self.data_path = Path(data_path) / 'NWPU-RESISC45'
        self.merge = merge
        self.prop = prop
        assert 0 < prop < 90
        self.train_tf = train_tf
        self.val_tf = val_tf
        self.test_tf = test_tf
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        self.in_channels = 3
        self.classes = DatasetResisc.classes[merge].value
        self.num_classes = len(self.classes)
        self.class_names = [l.name for l in self.classes]
        self.class_colors = [(i, l.color) for i, l in enumerate(self.classes)]
        self.class_weights = (
            [1.0] * self.num_classes if class_weights is None else class_weights
        )
    
    def prepare_data(self):
        num_item = sum([len(label.values)*700 for label in self.classes])
        self.train_idx, self.val_idx, self.test_idx = [], [], []
        for i in range(num_item):
            if i%100 < self.prop: self.train_idx.append(i)
            elif i%100 >= 90: self.val_idx.append(i)
            else: self.test_idx.append(i)

    def setup(self, stage):
        if stage in ("fit", "validate"):
            self.train_set = Subset(
                DatasetResisc(self.data_path, self.train_tf, self.merge),
                indices=self.train_idx,
            )
            self.val_set = Subset(
                DatasetResisc(self.data_path, self.val_tf, self.merge),
                indices=self.val_idx,
            )
        if stage in ("test", "predict"):
            dataset = Subset(
                DatasetResisc(self.data_path, self.test_tf, self.merge),
                indices=self.test_idx,
            )
            if stage == "test":
                self.test_set = dataset
            else:
                self.pred_set = dataset
                
    def get_loader(self, dataset):
        return partial(
            DataLoader,
            dataset=dataset,
            collate_fn=CustomCollate(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
                       
    def train_dataloader(self):
        train_dataloaders = {}
        train_dataloaders["sup"] = self.get_loader(self.train_set)(
            shuffle=True,
            drop_last=True,
        )
        return CombinedLoader(train_dataloaders, mode="min_size")
    
    def val_dataloader(self):
        return self.get_loader(self.val_set)(
            shuffle=False,
            drop_last=False,
        )

    def predict_dataloader(self):
        return self.get_loader(self.pred_set)(
            shuffle=False,
            drop_last=False,
        )
    
    def test_dataloader(self):
        return self.get_loader(self.test_set)(
            shuffle=False,
            drop_last=False,
        )

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
        return CombinedLoader(train_dataloaders, mode="max_size")