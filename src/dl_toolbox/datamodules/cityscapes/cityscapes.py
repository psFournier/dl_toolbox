import os
from pathlib import Path
from functools import partial

import torch

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import CombinedLoader
from torch.utils.data import DataLoader

import dl_toolbox.datasets as datasets
from dl_toolbox.utils import list_of_dicts_to_dict_of_lists


class Cityscapes(LightningDataModule):
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
        self.in_channels = 3
        self.class_list = datasets.Cityscapes.all_class_lists[merge].value
        self.dataloader = partial(
            DataLoader,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
    def setup(self, stage):
        
        def get_split_data(split):
            imgs = []
            msks = []
            img_dir = self.data_path/'Cityscapes'/'leftImg8bit'/split
            msk_dir = self.data_path/'Cityscapes'/'gtFine'/split
            for city in os.listdir(img_dir):
                for file_name in os.listdir(img_dir/city):
                    target_name = "{}_{}".format(
                        file_name.split("_leftImg8bit")[0], "gtFine_labelIds.png"
                    )
                    imgs.append(img_dir/city/file_name)
                    msks.append(msk_dir/city/target_name)
            return imgs, msks
        
        train_imgs, train_msks = get_split_data('train')
        self.train_set = datasets.Cityscapes(train_imgs, train_msks, self.merge, self.train_tf)
        val_imgs, val_msks = get_split_data('val')
        self.val_set = datasets.Cityscapes(val_imgs, val_msks, self.merge, self.test_tf)      

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
