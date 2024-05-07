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
from torch.utils.data import DataLoader, Subset

import dl_toolbox.datasets as datasets
from dl_toolbox.utils import CustomCollate
from dl_toolbox.transforms import Compose, NoOp, RandomCrop2


class Rellis3d(LightningDataModule):
    
    sequences = [
        "00000",
        "00001",
        "00002",
        "00003",
        "00004",
    ]

    def __init__(
        self,
        data_path,
        merge,
        sup,
        unsup,
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
        self.sup = sup
        self.unsup = unsup
        self.train_tf = train_tf
        self.test_tf = test_tf
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.in_channels = 3
        self.classes = datasets.Rellis3d.classes[merge].value
        self.num_classes = len(self.classes)
        self.class_names = [l.name for l in self.classes]
        self.class_colors = [(i, l.color) for i, l in enumerate(self.classes)]

    def setup(self, stage):
        
        imgs = []
        msks = []
        for s in self.sequences:
            img_dir = self.data_path/'Rellis-3D'/f'{s}'/'pylon_camera_node'
            msk_dir = self.data_path/'Rellis-3D'/f'{s}'/'pylon_camera_node_label_id'
            for msk_name in os.listdir(msk_dir):
                img_name = "{}.{}".format(msk_name.split('.')[0], "jpg")
                imgs.append(img_dir/img_name)
                msks.append(msk_dir/msk_name)
                    
        rellis = partial(datasets.Rellis3d, imgs, msks, self.merge)
        l, L = int(0.8*len(imgs)), len(imgs)
        idxs=random.sample(range(L), L)
        self.train_set = Subset(rellis(self.train_tf), idxs[:l])
        self.val_set = Subset(rellis(self.test_tf), idxs[l:])
                
    def dataloader(self, dataset):
        return partial(
            DataLoader,
            dataset=dataset,
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
        return CombinedLoader(train_dataloaders, mode="max_size_cycle")
    
    def val_dataloader(self):
        return self.dataloader(self.val_set)(
            shuffle=False,
            drop_last=False,
        )