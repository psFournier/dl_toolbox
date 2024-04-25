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


class xView1(LightningDataModule):
    
    def __init__(
        self,
        data_path,
        train_tf,
        test_tf,
        merge,
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
        self.batch_size = batch_size
        self.in_channels = 3
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.classes = datasets.xView1.classes
        self.num_classes = len(self.classes)
        self.class_names = [l.name for l in self.classes]
        self.class_colors = [(i, l.color) for i, l in enumerate(self.classes)]
    
    def setup(self, stage=None):
        path = self.data_path/'XVIEW1'
        xview1 = datasets.xView1(path/'train_images', path/'xView_train.json', self.train_tf)
        l,L = int(0.8*len(xview1)), len(xview1)
        idxs=random.sample(range(L), L)
        self.train_set = Subset(xview1, idxs[:l])
        self.val_set = Subset(xview1, idxs[l:])
        self.val_set.transforms = self.test_tf
    
    @staticmethod
    def _collate(batch):
        images_b, targets_b = tuple(zip(*batch))
        # don't stack bb because each batch elem may not have the same nb of bb
        return torch.stack(images_b), targets_b 
    
    def dataloader(self, dataset):
        return partial(
            DataLoader,
            dataset=dataset,
            collate_fn=self._collate,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
                       
    def train_dataloader(self):
        train_dataloaders = {}
        train_dataloaders["sup"] = self.dataloader(self.train_set)(
            shuffle=True,
            drop_last=True,
        )
        #if self.unsup > 0:
        #    train_dataloaders["unsup"] = self.dataloader(self.unlabeled_set)(
        #        shuffle=True,
        #        drop_last=True,
        #    )
        return CombinedLoader(train_dataloaders, mode="max_size_cycle")
    
    def val_dataloader(self):
        return self.dataloader(self.val_set)(
            shuffle=False,
            drop_last=False,
        )