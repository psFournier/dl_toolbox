
from pathlib import Path
from functools import partial

import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import CombinedLoader
from torch.utils.data import DataLoader, RandomSampler

import dl_toolbox.datasets as datasets
from dl_toolbox.utils import get_tiles, list_of_dicts_to_dict_of_lists



class Semcity(LightningDataModule):
   
    def __init__(
        self,
        data_path,
        merge,
        train_tf,
        val_tf,
        test_tf,
        batch_size,
        num_windows,
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
        self.in_channels = 3 #len(bands)
        self.class_list = datasets.Semcity.all_class_lists[merge].value
        self.dataloader = partial(
            DataLoader,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        self.epoch_steps = epoch_steps
        self.num_windows = num_windows
    
    def setup(self, stage):
        
        def paths(num):
            img_dir = self.data_path/'SemCity-Toulouse-bench/img_multispec_05/TLS_BDSD_RGB'
            msk_dir = self.data_path/'SemCity-Toulouse-bench/semantic_05/TLS_indMap'
            return img_dir/f'TLS_BDSD_RGB_{num:02}.tif', msk_dir/f'TLS_indMap_{num:02}_1.tif'
        
        windows = list(get_tiles(3504, 3452, 876, 863))
        imgs_msks = [paths(i) for i in range(1,17)]
        
        train_windows = windows[:self.num_windows]
        train_img_msk_win = [(img,msk,w) for img,msk in imgs_msks for w in train_windows]
        train_img, train_msk, train_win = tuple(zip(*train_img_msk_win))
        self.train_set = datasets.Semcity(train_img, train_msk, train_win, [1,2,3], self.merge, self.train_tf)
        
        val_windows = windows[-4:]
        val_img_msk_win = [(img,msk,w) for img,msk in imgs_msks for w in val_windows]
        val_img, val_msk, val_win = tuple(zip(*val_img_msk_win))
        self.val_set = datasets.Semcity(val_img, val_msk, val_win, [1,2,3], self.merge, self.val_tf)
        
        self.pred_set = datasets.Semcity(val_img, val_msk, val_win, [1,2,3], self.merge, self.test_tf)
                
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