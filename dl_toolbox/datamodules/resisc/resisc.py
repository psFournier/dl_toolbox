from pathlib import Path
from functools import partial

import numpy as np
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import CombinedLoader
from torch.utils.data import DataLoader, Subset

import dl_toolbox.datasets as datasets
from dl_toolbox.utils import CustomCollate


class Resisc(LightningDataModule):
    
    def __init__(
        self,
        data_path,
        merge,
        sup,
        unsup,
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
        self.dataset_tf = dataset_tf
        self.batch_size = batch_size
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
        num_item = sum([len(label.values)*700 for label in self.classes])
        self.train_idx, self.val_idx, self.test_idx = [], [], []
        self.unsup_idx = []
        for i in range(num_item):
            m = i%100
            if self.sup <= m < self.sup + self.unsup: self.unsup_idx.append(i)
            if 0 <= m < self.sup: self.train_idx.append(i)
            elif 90 <= m < 100: self.val_idx.append(i)
            else: pass

    def setup(self, stage):
        data_path = self.data_path/'NWPU-RESISC45'
        if stage in ("fit", "validate"):
            self.train_set = Subset(
                datasets.Resisc(data_path, self.dataset_tf, self.merge),
                indices=self.train_idx,
            )
            self.val_set = Subset(
                datasets.Resisc(data_path, self.dataset_tf, self.merge),
                indices=self.val_idx,
            )
            if self.unsup > 0:
                self.unsup_set = Subset(
                    datasets.Resisc(data_path, self.dataset_tf, self.merge),
                    indices=self.unsup_idx,
                )
                
    def train_dataloader(self):
        train_dataloaders = {}
        train_dataloaders["sup"] = DataLoader(
            dataset=self.train_set,
            collate_fn=CustomCollate(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True
        )
        if self.unsup > 0:
            train_dataloaders["unsup"] = DataLoader(
                dataset=self.unsup_set,
                collate_fn=CustomCollate(),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=True,
                drop_last=True
            )
        return CombinedLoader(train_dataloaders, mode="max_size_cycle")
                
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_set,
            collate_fn=CustomCollate(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )

#    def predict_dataloader(self):
#        return self.get_loader(self.pred_set)(
#            shuffle=False,
#            drop_last=False,
#        )