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
        self.sup = sup
        self.unsup = unsup
        self.bands = bands
        self.dataset_tf = dataset_tf
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.in_channels = len(self.bands)
        self.classes = datasets.Semcity.classes[merge].value
        self.num_classes = len(self.classes)
        self.class_names = [l.name for l in self.classes]
        self.class_colors = [(i, l.color) for i, l in enumerate(self.classes)]
        self.class_weights = (
            [1.0] * self.num_classes if class_weights is None else class_weights
        )
    
    def prepare_data(self):
        num_img = [f'TLS_BDSD_RGB_{i:02}.tif' for i in range(1, 16)]
        num_msk = [f'TLS_indMap_{i:02}_1.tif' for i in range(1, 16)]
        img_dir = self.data_path/'SemCity-Toulouse-bench/img_multispec_05/TLS_BDSD_RGB'
        msk_dir = self.data_path/'SemCity-Toulouse-bench/semantic_05/TLS_indMap'
        imgs = [img_dir/f'{num}' for num in num_img]
        msks = [msk_dir/f'{num}' for num in num_msk]
        windows = get_tiles(3504, 3452, 876, 863)
        self.dict_train = {"IMG": [], "MSK": [], "WIN": []}
        self.dict_train_unlabeled = {"IMG": [], "MSK": [], "WIN": []}
        self.dict_val = {"IMG": [], "MSK": [], "WIN": []}
        for i, prod in enumerate(product(windows, zip(imgs, msks))):
            win, (img, msk) = prod
            if self.sup <= i%100 < self.sup + self.unsup:
                self.dict_train_unlabeled["IMG"].append(img)
                self.dict_train_unlabeled["WIN"].append(win)
            if i%100 < self.sup:
                self.dict_train["IMG"].append(img)
                self.dict_train["MSK"].append(msk)
                self.dict_train["WIN"].append(win)
            elif 90 <= i%100:
                col_off, row_off, width, height = win
                val_tiles = get_tiles(
                    width, height, 256, step_w=224, 
                    col_offset=col_off, row_offset=row_off
                )
                for subwin in val_tiles:
                    self.dict_val["IMG"].append(img)
                    self.dict_val["MSK"].append(msk)
                    self.dict_val["WIN"].append(subwin)

    def setup(self, stage):
        if stage in ("fit", "validate"):
            self.train_set = datasets.Semcity(
                self.dict_train["IMG"],
                self.dict_train["MSK"],
                self.dict_train["WIN"],
                self.bands,
                self.merge,
                transforms=self.dataset_tf
            )
            self.val_set = datasets.Semcity(
                self.dict_val["IMG"],
                self.dict_val["MSK"],
                self.dict_val["WIN"],
                self.bands,
                self.merge,
                transforms=self.dataset_tf
            )
            if self.unsup > 0:
                self.unlabeled_set = datasets.Semcity(
                    self.dict_train_unlabeled["IMG"],
                    [],
                    self.dict_train_unlabeled["WIN"],
                    self.bands,
                    self.merge,
                    transforms=self.dataset_tf
                )
                
    def dataloader(self, dataset):
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
        train_dataloaders["sup"] = self.dataloader(self.train_set)(
            shuffle=True,
            drop_last=True,
        )
        if self.unsup > 0:
            train_dataloaders["unsup"] = self.dataloader(self.unlabeled_set)(
                shuffle=True,
                drop_last=True,
            )
        return CombinedLoader(train_dataloaders, mode="max_size_cycle")
    
    def val_dataloader(self):
        return self.dataloader(self.val_set)(
            shuffle=False,
            drop_last=False,
        )