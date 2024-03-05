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
from torch.utils.data import DataLoader, RandomSampler

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
        to_0_1,
        train_tf,
        test_tf,
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
        self.to_0_1 = to_0_1
        self.train_tf = train_tf
        self.test_tf = test_tf
        self.batch_size_s = batch_size_s
        self.batch_size_u = batch_size_u
        self.steps_per_epoch = steps_per_epoch
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.in_channels = 3
        self.classes = datasets.Rellis3d.classes[merge].value
        self.num_classes = len(self.classes)
        self.class_names = [l.name for l in self.classes]
        self.class_colors = [(i, l.color) for i, l in enumerate(self.classes)]
        #self.class_weights = (
        #    [1.0] * self.num_classes if class_weights is None else class_weights
        #)
        
    
    def prepare_data(self):
        def get_seq_imgs_msks(seq):
            imgs = []
            msks = []
            img_dir = self.data_path/'Rellis-3D'/f'{seq}'/'pylon_camera_node'
            msk_dir = self.data_path/'Rellis-3D'/f'{seq}'/'pylon_camera_node_label_id'
            for msk_name in os.listdir(msk_dir):
                img_name = "{}.{}".format(msk_name.split('.')[0], "jpg")
                imgs.append(img_dir/img_name)
                msks.append(msk_dir/msk_name)
            return zip(imgs, msks)
        imgs_msks_from_seqs = []
        for s in self.sequences:
            imgs_msks_from_seqs += get_seq_imgs_msks(s)
        random.shuffle(imgs_msks_from_seqs)
        L = len(imgs_msks_from_seqs)
        self.train_s = imgs_msks_from_seqs[:int(L*0.6):self.sup]
        self.val = imgs_msks_from_seqs[int(L*0.6):int(L*0.8)]
        self.test = imgs_msks_from_seqs[int(L*0.8):]
        self.train_u = imgs_msks_from_seqs
        self.predict = self.val[:]

    def setup(self, stage):
        self.train_s_set = datasets.Rellis3d(
            *[list(t) for t in zip(*self.train_s)],
            self.merge,
            transforms=Compose([self.to_0_1, self.train_tf])
        )
        if self.unsup != -1:
            self.train_u_set = datasets.Rellis3d(
                *[list(t) for t in zip(*self.train_u)],
                self.merge,
                transforms=Compose([self.to_0_1, RandomCrop2(256)])
            )
        self.val_set = datasets.Rellis3d(
            *[list(t) for t in zip(*self.val)],
            self.merge,
            transforms=Compose([self.to_0_1, self.test_tf])
        )
        self.test_set = datasets.Rellis3d(
            *[list(t) for t in zip(*self.test)],
            self.merge,
            transforms=Compose([self.to_0_1, self.test_tf])
        )
        self.predict_set = datasets.Rellis3d(
            *[list(t) for t in zip(*self.predict)],
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
    
    def predict_dataloader(self):
        return self.dataloader(self.predict_set)(
            shuffle=False,
            drop_last=False,
            batch_size=self.batch_size_s
        )