import os
import random
from pathlib import Path
from functools import partial

import numpy as np
import torch

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import CombinedLoader
from torch.utils.data import DataLoader, Subset, RandomSampler
from torch.utils.data._utils.collate import default_collate

import dl_toolbox.datasets as datasets
from dl_toolbox.utils import list_of_dicts_to_dict_of_lists

class Rellis(LightningDataModule):
    
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
        train_tf,
        val_tf,
        test_tf,
        batch_size,
        epoch_steps,
        num_workers,
        pin_memory,
        *args,
        **kwargs
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.merge = merge
        self.train_tf = train_tf
        self.val_tf = val_tf
        self.test_tf = test_tf
        self.batch_size = batch_size
        self.in_channels = 3
        self.class_list = datasets.Rellis3d.all_class_lists[merge].value
        self.dataloader = partial(
            DataLoader,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        self.epoch_steps = epoch_steps

    def setup(self, stage):
        def get_imgs_msks(seqs, start, end):
            imgs = []
            msks = []
            for s in seqs:
                img_dir = self.data_path/'Rellis-3D'/s/'pylon_camera_node'
                msk_dir = self.data_path/'Rellis-3D'/s/'pylon_camera_node_label_id'
                for msk_name in sorted(os.listdir(msk_dir))[start:end]:
                    img_name = "{}.{}".format(msk_name.split('.')[0], "jpg")
                    imgs.append(img_dir/img_name)
                    msks.append(msk_dir/msk_name)
            return imgs, msks
        train_imgs, train_msks = get_imgs_msks(self.sequences, 0, 500)
        train_set = datasets.Rellis3d(train_imgs, train_msks, self.merge, self.train_tf)
        self.train_set = Subset(train_set, indices=list(range(0, len(train_set), 1)))   
        val_imgs, val_msks = get_imgs_msks(self.sequences, 700, -1)
        val_set = datasets.Rellis3d(val_imgs, val_msks, self.merge, self.val_tf)
        self.val_set = Subset(val_set, indices=list(range(0, len(val_set), 1)))
        
        pred_set = datasets.Rellis3d(val_imgs, val_msks, self.merge, self.test_tf)
        self.pred_set = Subset(pred_set, indices=list(range(0, len(pred_set), 10)))
        
    def collate(self, batch):
        batch = list_of_dicts_to_dict_of_lists(batch)
        batch['image'] = torch.stack(batch['image'])
        batch['target'] = torch.stack(batch['target'])
        return batch
                       
    def train_dataloader(self):
        train_dataloaders = {}
        train_dataloaders["sup"] = self.dataloader(
            dataset=self.train_set,
            sampler=RandomSampler(
                self.train_set,
                replacement=True,
                num_samples=self.epoch_steps*self.batch_size
            ),
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
            dataset=self.pred_set,
            shuffle=False,
            drop_last=False,
            collate_fn=self.collate
        )