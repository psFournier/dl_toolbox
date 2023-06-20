from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, Dataset, DataLoader, RandomSampler
from torchvision.transforms import transforms
from pytorch_lightning.utilities import CombinedLoader

import dl_toolbox.datasets as datasets
import dl_toolbox.datasources as datasources
from dl_toolbox.utils import CustomCollate
from pathlib import Path

import rasterio
import csv
import numpy as np
import ast 
import rasterio.windows as windows

from functools import partial


def data_src_from_csv(datapath, csvpath, folds):
    
    with open(csvpath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            _, _, image_path, label_path, x0, y0, w, h, fold, mins, maxs = row[:11]
            if int(fold) in folds:
                co, ro, w, h = [int(e) for e in [x0, y0, w, h]]
                yield {
                    'image_path': datapath/image_path,
                    'label_path': datapath/label_path if label_path != 'none' else None,
                    'mins': np.array(ast.literal_eval(mins)).reshape(-1, 1, 1),
                    'maxs': np.array(ast.literal_eval(maxs)).reshape(-1, 1, 1),
                    'zone': windows.Window(co, ro, w, h)
                }

class SplitFromCsv(LightningDataModule):

    def __init__(
        self,
        datasource,
        train_set,
        val_set,
        data_path,
        csv_path,
        train_aug,
        epoch_len,
        batch_size,
        num_workers,
        pin_memory,
        train_idx,
        val_idx
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        #self.save_hyperparameters()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_set = train_set
        self.val_set = val_set
        self.train_aug = train_aug
        self.num_samples = epoch_len * batch_size
        
        src_gen = partial(
            data_src_from_csv,
            datapath=Path(data_path),
            csvpath=Path(csv_path)
        )        
        self.train_srcs = [datasource(**d) for d in src_gen(folds=train_idx)]
        self.val_srcs = [datasource(**d) for d in src_gen(folds=val_idx)]   
        
    def setup(self, stage):
        
        train_sets = [self.train_set(src, aug=self.train_aug) for src in self.train_srcs]
        val_sets = [self.val_set(src, aug=None) for src in self.val_srcs]
        self.train_set = ConcatDataset(train_sets)
        self.val_set = ConcatDataset(val_sets)
                    
    @property
    def num_classes(self):
        return len(self.train_srcs[0].nomenclature)
    
    @property
    def class_names(self):
        return [l.name for l in self.train_srcs[0].nomenclature]    
                    
    @property
    def input_dim(self):
        return len(self.val_srcs[0].bands)
                    
    @property
    def class_weights(self):
        return [1.]*self.num_classes

    def train_dataloader(self):
        
        train_dataloaders = {}
        train_dataloaders['sup'] = DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            collate_fn=CustomCollate(),
            sampler=RandomSampler(
                data_source=self.train_set,
                replacement=True,
                num_samples=self.num_samples
            ),
            num_workers=self.num_workers,
            drop_last=True
        )
        return CombinedLoader(
            train_dataloaders,
            mode='min_size'
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_set,
            sampler=RandomSampler(
                data_source=self.val_set,
                replacement=True,
                num_samples=self.num_samples//10
            ),
            collate_fn=CustomCollate(),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )    
    
class DigitanieFromCsv(SplitFromCsv):
    
    def __init__(self, *args, **kwargs):
        
        super().__init__(
            train_idx=[i+j for i in [0, 66, 88, 99, 110, 154, 165] for j in range(1,8)],
            val_idx=[i+j for i in [0, 66, 88, 99, 110, 154, 165] for j in range(8,9)],
            *args,
            **kwargs
        )
        
    #def init_srcs(self):
    #    
    #    train_idx=[i+j for i in [0, 66, 88, 99, 110, 154, 165] for j in range(1,8)]
    #    self.train_srcs = [datasource(**d) for d in self.src_gen(folds=train_idx)]
    #    val_idx=[i+j for i in [0, 66, 88, 99, 110, 154, 165] for j in range(8,9)]
    #    self.val_srcs = [datasource(**d) for d in self.src_gen(folds=val_idx)] 