from pathlib import Path
from functools import partial
import random
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import CombinedLoader
from torch.utils.data import DataLoader, Subset

import dl_toolbox.datasets as datasets
from dl_toolbox.utils import list_of_dicts_to_dict_of_lists

class Resisc(LightningDataModule):
    
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
        self.in_channels = 3
        self.class_list = datasets.Resisc.all_class_lists[merge].value
        self.dataloader = partial(
            DataLoader,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    def setup(self, stage):        
        resisc = partial(
            datasets.Resisc,
            data_path=self.data_path/'NWPU-RESISC45',
            merge=self.merge
        )
        nb_all_imgs = sum([len(label.values)*700 for label in self.class_list])
        split = int(0.8*nb_all_imgs)
        idxs=random.sample(range(nb_all_imgs), nb_all_imgs)
        self.train_set = Subset(resisc(transforms=self.train_tf), idxs[:split])
        self.val_set = Subset(resisc(transforms=self.test_tf), idxs[split:])
        self.predict_set = Subset(resisc(transforms=self.test_tf), idxs[split:])
        
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