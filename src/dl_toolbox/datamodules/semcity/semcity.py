import json

import os
import random
import re
from os.path import join
from pathlib import Path
from functools import partial
from itertools import product

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import CombinedLoader
from torch.utils.data import DataLoader, Subset

import dl_toolbox.datasets as datasets
from dl_toolbox.utils import get_tiles, list_of_dicts_to_dict_of_lists



class Semcity(LightningDataModule):
   
    def __init__(
        self,
        data_path,
        merge,
        bands,
        train_tf,
        test_tf,
        batch_size,
        num_workers,
        pin_memory,
        *args,
        **kwargs
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.merge = merge
        self.bands = bands
        self.train_tf = train_tf
        self.test_tf = test_tf
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.in_channels = len(self.bands)
        self.class_list = datasets.Semcity.all_class_lists[merge].value
        self.dataloader = partial(
            DataLoader,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    def setup(self, stage):
        
        def paths(num):
            img_dir = self.data_path/'SemCity-Toulouse-bench/img_multispec_05/TLS_BDSD_RGB'
            msk_dir = self.data_path/'SemCity-Toulouse-bench/semantic_05/TLS_indMap'
            return img_dir/f'TLS_BDSD_RGB_{num:02}.tif', msk_dir/f'TLS_indMap_{num:02}_1.tif'
        
        windows = list(get_tiles(3504, 3452, 876, 863))
        imgs_msks = [paths(i) for i in range(1,17)]
        all_img_msk_win = [(img,msk,w) for img,msk in imgs_msks for w in windows]
        imgs, msks, windows = tuple(zip(*all_img_msk_win))
        
        semcity = partial(datasets.Semcity, imgs, msks, windows, self.bands, self.merge)
        l, L = int(0.8*len(imgs)), len(imgs)
        idxs=random.sample(range(L), L)
        self.train_set = Subset(semcity(self.train_tf), idxs[:l])
        self.val_set = Subset(semcity(self.test_tf), idxs[l:])
                
    def collate(self, batch):
        batch = list_of_dicts_to_dict_of_lists(batch)
        batch['image'] = torch.stack(batch['image'])
        batch['target'] = torch.stack(batch['target'])
        return batch
                       
    def train_dataloader(self):
        train_dataloaders = {}
        train_dataloaders["sup"] = self.dataloader(
            dataset=self.train_set,
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate
        )
        return CombinedLoader(train_dataloaders, mode="max_size_cycle")
    
    def val_dataloader(self):
        return self.dataloader(
            dataset=self.val_set,
            shuffle=False,
            drop_last=False,
            collate_fn=self.collate
        )

    def predict_dataloader(self):
        return self.dataloader(
            dataset=self.predict_set,
            shuffle=False,
            drop_last=False,
            collate_fn=self.collate
        )