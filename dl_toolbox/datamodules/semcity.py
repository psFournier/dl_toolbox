import json

import os
import random
import re
from os.path import join
from pathlib import Path
from functools import partial

import numpy as np
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import CombinedLoader
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset

from dl_toolbox.datasets import SemcityDataset
from dl_toolbox.utils import CustomCollate, get_tiles


class SemcityDatamodule(LightningDataModule):
   
    def __init__(
        self,
        data_path,
        merge,
        prop,
        bands,
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
        self.data_path = Path(data_path)
        self.merge = merge
        self.prop = prop
        self.bands = bands
        self.train_tf = train_tf
        self.val_tf = val_tf
        self.test_tf = test_tf
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.in_channels = len(self.bands)
        self.classes = SemcityDataset.classes[merge].value
        self.num_classes = len(self.classes)
        self.class_names = [l.name for l in self.classes]
        self.class_colors = [(i, l.color) for i, l in enumerate(self.classes)]
        self.class_weights = (
            [1.0] * self.num_classes if class_weights is None else class_weights
        )
    
    def prepare_data(self):
        num_img = [f'TLS_BDSD_M_{i:02}.tif' for i in range(1, 16)]
        num_msk = [f'TLS_indMap_{i:02}_1.tif' for i in range(1, 16)]
        img_dir = self.data_path/'SemCity-Toulouse-bench/img_multispec_05/TLS_BDSD_M'
        msk_dir = self.data_path/'SemCity-Toulouse-bench/semantic_05/TLS_indMap'
        imgs = [img_dir/f'{num}' for num in num_img]
        msks = [msk_dir/f'{num}' for num in num_msk]
        train_windows = [(0, 0, 1752, 1726), (1752, 0, 1752, 1726), (0, 1726, 1752, 1726)]
        val_windows = [(1752, 1726, 1752, 1726)]
        self.dict_train = {'IMG': imgs, 'MSK': msks, 'WIN': train_windows}
        self.dict_val = {'IMG': imgs, 'MSK': msks, 'WIN': val_windows}

    def setup(self, stage):
        if stage in ("fit", "validate"):
            self.train_set = ConcatDataset([
                SemcityDataset(
                    self.dict_train["IMG"],
                    self.dict_train["MSK"],
                    self.bands,
                    self.merge,
                    transforms=self.train_tf,
                    crop_size=512,
                    window=window,
                    crop_step=256
                ) for window in self.dict_train["WIN"]
            ])

            self.val_set = ConcatDataset([
                SemcityDataset(
                    self.dict_val["IMG"],
                    self.dict_val["MSK"],
                    self.bands,
                    self.merge,
                    transforms=self.val_tf,
                    crop_size=512,
                    window=window,
                    crop_step=256
                ) for window in self.dict_val["WIN"]
            ])
                
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
