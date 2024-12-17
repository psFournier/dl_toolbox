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
from dl_toolbox.utils import list_of_dicts_to_dict_of_lists

class Coco(LightningDataModule):
        
    def __init__(
        self,
        data_path,
        merge,
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
        self.train_tf = train_tf
        self.test_tf = test_tf
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.class_list = datasets.Coco.classes[merge].value
        self.dataloader = partial(
            DataLoader,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    def setup(self, stage=None):
        path = self.data_path/'coco'
        self.train_set = datasets.Coco(
            path/"train2017",
            path/"annotations/instances_train2017.json",
            self.train_tf,
            merge=self.merge
        )
        self.val_set = datasets.Coco(
            path/"val2017",
            path/"annotations/instances_val2017.json",
            self.test_tf,
            merge=self.merge
        )
        
    def collate(self, batch, train):
        batch = list_of_dicts_to_dict_of_lists(batch)
        batch['image'] = torch.stack(batch['image'])
        # don't stack targets because each batch elem may not have the same nb of bb
        return batch
    
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