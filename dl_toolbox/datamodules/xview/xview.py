from torch.utils.data import DataLoader, Subset
from pathlib import Path
import pandas as pd
import numpy as np
from pytorch_lightning import LightningDataModule
from functools import partial
import dl_toolbox.datasets as datasets
import torch
import random
from pytorch_lightning.utilities import CombinedLoader


class xView(LightningDataModule):
    
    def __init__(
        self,
        data_path,
        train_tf,
        test_tf,
        batch_tf,
        batch_size,
        num_workers,
        pin_memory,
        *args,
        **kwargs
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.train_tf = train_tf
        self.test_tf = test_tf
        self.batch_tf = batch_tf
        self.batch_size = batch_size
        self.in_channels = 3
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.classes = datasets.xView1.classes
        self.num_classes = len(self.classes)
        self.class_names = [l.name for l in self.classes]
        self.class_colors = [(i, l.color) for i, l in enumerate(self.classes)]
        self.dataloader = partial(
            DataLoader,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    def setup(self, stage=None):
        path = self.data_path/'XVIEW1'
        xview = datasets.xView1(path/'train_images', path/'xView_train.json', self.train_tf)
        l,L = int(0.8*len(xview)), len(xview)
        idxs=random.sample(range(L), L)
        self.train_set = Subset(xview, idxs[:l])
        self.val_set = Subset(xview, idxs[l:])
        self.val_set.transforms = self.test_tf
    
    def collate(self, batch, train):
        images_b, targets_b, paths_b = tuple(zip(*batch))
        # ignore batch_tf for detection 
        # don't stack bb because each batch elem may not have the same nb of bb
        return torch.stack(images_b), targets_b, paths_b 
    
    def train_dataloader(self):
        train_dataloaders = {}
        train_dataloaders["sup"] = self.dataloader(
            dataset=self.train_set,
            shuffle=True,
            drop_last=True,
            collate_fn=partial(self.collate, train=True)
        )
        return CombinedLoader(train_dataloaders, mode="max_size_cycle")
    
    def val_dataloader(self):
        return self.dataloader(
            dataset=self.val_set,
            shuffle=False,
            drop_last=False,
            collate_fn=partial(self.collate, train=False)
        )

    def predict_dataloader(self):
        return self.dataloader(
            dataset=self.predict_set,
            shuffle=False,
            drop_last=False,
            collate_fn=partial(self.collate, train=False)
        )