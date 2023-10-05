import json

import os
import random
import re
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


class Semcity(LightningDataModule):
   
    def __init__(
        self,
        data_path,
        merge,
        sup,
        unsup,
        bands,
        dataset_tf,
        batch_size_s,
        batch_size_u,
        steps_per_epoch,
        num_workers,
        pin_memory,
        *args,
        **kwargs
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.merge = merge
        self.sup = sup
        self.unsup = unsup
        self.bands = bands
        self.dataset_tf = dataset_tf
        self.batch_size_s = batch_size_s
        self.batch_size_u = batch_size_u
        self.steps_per_epoch = steps_per_epoch
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.in_channels = len(self.bands)
        self.classes = datasets.Semcity.classes[merge].value
        self.num_classes = len(self.classes)
        self.class_names = [l.name for l in self.classes]
        self.class_colors = [(i, l.color) for i, l in enumerate(self.classes)]
    
    def prepare_data(self):
        def paths(num):
            img_dir = self.data_path/'SemCity-Toulouse-bench/img_multispec_05/TLS_BDSD_RGB'
            msk_dir = self.data_path/'SemCity-Toulouse-bench/semantic_05/TLS_indMap'
            return img_dir/f'TLS_BDSD_RGB_{num:02}.tif', msk_dir/f'TLS_indMap_{num:02}_1.tif'
        self.test = [(*paths(num), t) for num in [6,11] for t in get_tiles(3504, 3452, 512, step_w=256)]
        self.val = [(*paths(7), t) for t in get_tiles(3504, 3452, 512, step_w=256)]
        train = [(*paths(num), t) for num in set(range(1, 17))-{6,7,11} for t in get_tiles(3504, 3452, 876, 863)]
        self.train_s = train[::self.sup]
        if self.unsup != -1: self.train_u = train[::self.unsup]

    def setup(self, stage):
        self.train_s_set = datasets.Semcity(
            *[list(t) for t in zip(*self.train_s)],
            self.bands,
            self.merge,
            transforms=self.dataset_tf
        )
        if self.unsup != -1:
            self.train_u_set = datasets.Semcity(
                *[list(t) for t in zip(*self.train_u)],
                self.bands,
                self.merge,
                transforms=self.dataset_tf
            )
        self.val_set = datasets.Semcity(
            *[list(t) for t in zip(*self.val)],
            self.bands,
            self.merge,
            transforms=self.dataset_tf
        )
        self.test_set = datasets.Semcity(
            *[list(t) for t in zip(*self.test)],
            self.bands,
            self.merge,
            transforms=self.dataset_tf
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