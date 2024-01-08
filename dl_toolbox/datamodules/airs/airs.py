import json

import os
import random
import re
import csv
from os.path import join
from pathlib import Path
from functools import partial
from itertools import product

import numpy as np
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import CombinedLoader
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset

import dl_toolbox.datasets as datasets
from dl_toolbox.utils import CustomCollate, get_tiles
from dl_toolbox.transforms import Compose, NoOp, RandomCrop2


class Airs(LightningDataModule):
   
    def __init__(
        self,
        data_path,
        filter_path,
        merge,
        sup,
        unsup,
        bands,
        to_0_1,
        train_tf,
        test_tf,
        batch_size_s,
        batch_size_u,
        steps_per_epoch,
        num_workers,
        pin_memory,
        class_weights=None,
        *args,
        **kwargs
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.filter_path = filter_path
        self.merge = merge
        self.sup = sup
        self.unsup = unsup
        self.bands = bands
        self.to_0_1 = to_0_1
        self.train_tf = train_tf
        self.test_tf = test_tf
        self.batch_size_s = batch_size_s
        self.batch_size_u = batch_size_u
        self.steps_per_epoch = steps_per_epoch
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.in_channels = len(self.bands)
        self.classes = datasets.Airs.classes[merge].value
        self.num_classes = len(self.classes)
        self.class_names = [l.name for l in self.classes]
        self.class_colors = [(i, l.color) for i, l in enumerate(self.classes)]
    
    def prepare_data(self):
        imgs_msks = []
        with open(self.filter_path, newline='') as f:
            r = csv.reader(f, delimiter=',')
            for img,msk in r:
                imgs_msks.append((self.data_path/'AIRS'/img,self.data_path/'AIRS'/msk))
        self.train_s = [(*i_m, w) for i_m in imgs_msks[:500] for w in get_tiles(10000, 10000, 768, 768, cover_all=False)]
        self.val = [(*i_m, w) for i_m in imgs_msks[500:600] for w in get_tiles(10000, 10000, 768, 768, cover_all=False)]
        self.test = [(*i_m, w) for i_m in imgs_msks[600:700] for w in get_tiles(10000, 10000, 768, 768, cover_all=False)]
        self.train_u = [(*i_m, w) for i_m in imgs_msks[700:] for w in get_tiles(10000, 10000, 768, 768, cover_all=False)]
                
    def setup(self, stage):
        self.train_s_set = datasets.Airs(
            *[list(t) for t in zip(*self.train_s)],
            self.bands,
            self.merge,
            transforms=Compose([self.to_0_1, self.train_tf])
        )
        if self.unsup != -1:
            self.train_u_set = datasets.Airs(
                *[list(t) for t in zip(*self.train_u)],
                self.bands,
                self.merge,
                transforms=Compose([self.to_0_1, RandomCrop2(256)])
            )
        self.val_set = datasets.Airs(
            *[list(t) for t in zip(*self.val)],
            self.bands,
            self.merge,
            transforms=Compose([self.to_0_1, self.test_tf])
        )
        self.test_set = datasets.Airs(
            *[list(t) for t in zip(*self.test)],
            self.bands,
            self.merge,
            transforms=Compose([self.to_0_1, self.test_tf])
        )
                
    def dataloader(self, dataset):
        return partial(
            DataLoader,
            dataset=dataset,
            collate_fn=CustomCollate(),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
                       
    def train_dataloader(self):
        train_dataloaders = {}
        train_dataloaders["sup"] = self.dataloader(self.train_s_set)(
            sampler=RandomSampler(
                self.train_s_set,
                replacement=True,
                num_samples=self.steps_per_epoch*self.batch_size_s
            ),
            drop_last=True,
            batch_size=self.batch_size_s
        )
        if self.unsup != -1:
            train_dataloaders["unsup"] = self.dataloader(self.train_u_set)(
                sampler=RandomSampler(
                    self.train_u_set,
                    replacement=True,
                    num_samples=self.steps_per_epoch*self.batch_size_u
                ),
                drop_last=True,
                batch_size=self.batch_size_u
            )
        return CombinedLoader(train_dataloaders, mode="max_size_cycle")
    
    def val_dataloader(self):
        return self.dataloader(self.val_set)(
            shuffle=False,
            drop_last=False,
            batch_size=self.batch_size_s
        )
    
    def test_dataloader(self):
        return self.dataloader(self.test_set)(
            shuffle=False,
            drop_last=False,
            batch_size=self.batch_size_s
        )