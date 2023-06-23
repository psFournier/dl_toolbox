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
import pandas as pd

import rasterio
import csv
import numpy as np
import ast 
import rasterio.windows as windows

from functools import partial


#def splits_from_csv(datasrc_cls, datapath, csvpath, folds):
#    
#    splits = {
#        'train':[],
#        'val':[],
#        'test':[]
#    }
#    
#    with open(csvpath, newline='') as csvfile:
#        reader = csv.reader(csvfile)
#        next(reader)
#        for row in reader:
#            idx, image_path, x0, y0, w, h, fold, mins, maxs = row[:11]
#            co, ro, w, h = [int(e) for e in [x0, y0, w, h]]
#            splits[fold].append(datasrc_cls(
#                'image_path'=datapath/image_path,
#                'label_path'=datapath/label_path if label_path != 'none' else None,
#                'zone'=windows.Window(co, ro, w, h)
#            ))
#    
#    return splits['train'], splits['val'], splits['test']

def splits_from_csv(datasrc, datapath, csvpath):
    
    splits = [[],[],[]]

    df_split = pd.read_csv(csvpath/'split.csv', index_col=0)
    df_stats = pd.read_csv(csvpath/'stats.csv', index_col=0)
    df_cls = pd.read_csv(csvpath/'cls.csv', index_col=0)

    for index, row in df_split.iterrows():

        minval = [df_stats.loc[index][f'min_{i}'] for i in range(1,5)]
        maxval = [df_stats.loc[index][f'max_{i}'] for i in range(1,5)]
        meanval = [df_stats.loc[index][f'mean_{i}'] for i in range(1,5)]
        cls_counts = list(df_cls.loc[index][1:])

        splits[row['split']].append(
            datasrc(
                image_path=datapath/row['img'],
                zone=windows.Window(
                    row['col_off'],
                    row['row_off'],
                    row['width'],
                    row['height']
                ),
                label_path=datapath/df_cls.loc[index]['mask'],
                minval=np.array(minval).reshape((-1, 1, 1)),
                maxval=np.array(maxval).reshape((-1, 1, 1)),
                meanval=np.array(meanval).reshape((-1, 1, 1)),
                all_cls_counts=np.array(cls_counts)
            )
        )
        
    return splits

class SplitFromCsv(LightningDataModule):

    def __init__(
        self,
        datasource,
        train_set,
        val_set,
        data_path,
        csv_path,
        normalization,
        train_aug,
        epoch_len,
        batch_size,
        num_workers,
        pin_memory
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
        self.normalization = normalization
        
        self.train_srcs, self.val_srcs, self.test_srcs = splits_from_csv(
            datasource,
            Path(data_path),
            Path(csv_path)
        )
        
    def setup(self, stage):
        
        train_sets = [self.train_set(src, aug=self.train_aug, normalization=self.normalization) for src in self.train_srcs]
        val_sets = [self.val_set(src, aug=None, normalization=self.normalization) for src in self.val_srcs]
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
    
#class DigitanieFromCsv(SplitFromCsv):
#    
#    def __init__(self, *args, **kwargs):
#        
#        super().__init__(
#            train_idx=[i+j for i in list(range(0,200, 11)) for j in range(1,8)],
#            val_idx=[i+j for i in [0, 66, 88, 99, 110, 154, 165] for j in range(8,9)],
#            *args,
#            **kwargs
#        )