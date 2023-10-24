from pathlib import Path
from functools import partial

import numpy as np
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import CombinedLoader
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset, Subset

import dl_toolbox.datasets as datasets
from dl_toolbox.utils import CustomCollate
from dl_toolbox.transforms import Compose, NoOp


class Resisc(LightningDataModule):
    
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
        class_weights=None,
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
        self.classes = datasets.Resisc.classes[merge].value
        self.num_classes = len(self.classes)
        self.class_names = [l.name for l in self.classes]
        self.class_colors = [(i, l.color) for i, l in enumerate(self.classes)]
        self.class_weights = (
            [1.0] * self.num_classes if class_weights is None else class_weights
        )
    
    def prepare_data(self):
        n_idxs = sum([len(label.values)*700 for label in self.classes])
        self.test_idx = [i for i in range(n_idxs) if i%700<70]
        self.val_idx = [i for i in range(n_idxs) if 70<=i%700<100]
        train_idx = [i for i in range(n_idxs) if 100<=i%700]
        self.train_s_idx = train_idx[::self.sup]
        if self.unsup != -1: self.train_u_idx = train_idx[::self.unsup]   
#        self.test_idx = [700*n+i for l in self.classes for n, v in enumerate(l.values) for i in range(100)]
#        self.test_idx = [700*nc+i for nc in range(self.num_classes) for i in range(100)]
#        self.val_idx = [700*nc+i for nc in range(self.num_classes) for i in range(100, 150)]
#        train_idx = [700*nc+i for nc in range(self.num_classes) for i in range(100, 150)]
#        self.train_idx, self.val_idx, self.test_idx = [], [], []
#        self.unsup_idx = []
#        for i in range(num_item):
#            m = i%100
#            if self.sup <= m < self.sup + self.unsup: self.unsup_idx.append(i)
#            if 0 <= m < self.sup: self.train_idx.append(i)
#            elif 80 <= m < 90: self.val_idx.append(i)
#            elif 90 <= m < 100: self.test_idx.append(i)
#            else: pass

    def setup(self, stage):
        data_path = self.data_path/'NWPU-RESISC45'
        self.train_s_set = Subset(
            datasets.Resisc(
                data_path,Compose([self.to_0_1, self.train_tf]), self.merge
            ),
            indices=self.train_s_idx,
        )
        self.val_set = Subset(
            datasets.Resisc(
                data_path, Compose([self.to_0_1, self.test_tf]), self.merge
            ),
            indices=self.val_idx,
        )
        if self.unsup > 0:
            self.train_u_set = Subset(
                datasets.Resisc(
                    data_path, Compose([self.to_0_1, NoOp()]), self.merge
                ),
                indices=self.train_u_idx,
            )
        self.test_set = Subset(
            datasets.Resisc(
                data_path, Compose([self.to_0_1, self.test_tf]), self.merge
            ),
            indices=self.test_idx,
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